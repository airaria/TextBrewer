
from textbrewer.distiller_utils import *
from textbrewer.distiller_basic import BasicDistiller
from pyemd import emd_with_flow
from scipy.special import softmax

class EMDDistiller(BasicDistiller):
    """
    BERT-EMD

    Args:
        train_config (:class:`TrainingConfig`): training configuration.
        distill_config (:class:`DistillationConfig`): distillation configuration.
        model_T (:class:`torch.nn.Module`): teacher model.
        model_S (:class:`torch.nn.Module`): student model.
        adaptor_T (Callable): teacher model's adaptor.
        adaptor_S (Callable): student model's adaptor.
        emd (dict): configuration for EMD

    The roles of `adaptor_T` and `adaptor_S` are explained in :py:func:`adaptor`.

    """
    def __init__(self, train_config,
                 distill_config,
                 model_T,
                 model_S,
                 adaptor_T,
                 adaptor_S,
                 emd: Optional[Dict] = None):
        # emd:{'layer_num_S' : 4+1,   
        #      'layer_num_T' : 12+1,
        #      'feature': 'hidden', 
        #      'loss': 'mse', 
        #      'weight' : 1.0, 
        # 'proj':['linear',312,768]}

        super(EMDDistiller, self).__init__(train_config, distill_config, model_T, model_S, adaptor_T, adaptor_S)

        self.projs = []
        self.projs_group = []
        # for im in self.d_config.intermediate_matches:
        #     if im.proj is not None:
        #         projection = im.proj[0]
        #         dim_in = im.proj[1]
        #         dim_out = im.proj[2]
        #         self.projs_group.append(im.proj[3])
        #         self.projs.append(PROJ_MAP[projection](dim_in,dim_out))
        #         self.projs[-1].to(self.t_config.device)
        #     else:
        #         self.projs.append(None)
        #         self.projs_group.append(None)
        
        self.d_config.is_caching_logits = False

        self.layer_num_S = emd['layer_num_S']
        self.layer_num_T = emd['layer_num_T']
        self.emd_feature = emd['feature']
        self.emd_loss_type = emd['loss']
        self.emd_loss_weight = emd['weight']
        if self.emd_feature != 'hidden' or self.emd_loss_type != 'hidden_mse':
            raise NotImplementedError

        self.feature_weight_S = np.ones(self.layer_num_S - 1) / (self.layer_num_S - 1) # excluding the embedding layer
        self.feature_weight_T = np.ones(self.layer_num_T - 1) / (self.layer_num_T - 1)

        if isinstance(emd['proj'],list) and len(emd['proj']) > 0:
            projection,dim_in,dim_out = emd['proj']
            for im in range(self.layer_num_S):
                self.projs.append(PROJ_MAP[projection](dim_in,dim_out))
                self.projs[-1].to(self.t_config.device)
                self.projs_group.append(dict())



    def train_on_batch(self, batch, args):

        (teacher_batch, results_T), (student_batch, results_S) = get_outputs_from_batch(batch, self.t_config.device, self.model_T, self.model_S, args)

        results_T = post_adaptor(self.adaptor_T(teacher_batch,results_T))
        results_S = post_adaptor(self.adaptor_S(student_batch,results_S))

        total_loss, losses_dict = self.compute_loss(results_S, results_T)

        return total_loss, losses_dict


    def compute_loss(self,results_S,results_T):

        losses_dict = dict()

        total_loss  = 0
        if 'logits' in results_T and 'logits' in results_S:
            logits_list_T = results_T['logits']  # list of tensor
            logits_list_S = results_S['logits']  # list of tensor
            total_kd_loss = 0
            if 'logits_mask' in results_S:
                masks_list_S = results_S['logits_mask']
                logits_list_S = select_logits_with_mask(logits_list_S,masks_list_S)  #(mask_sum, num_of_class)
            if 'logits_mask' in results_T:
                masks_list_T = results_T['logits_mask']
                logits_list_T = select_logits_with_mask(logits_list_T,masks_list_T)  #(mask_sum, num_of_class)

            for l_T,l_S in zip(logits_list_T,logits_list_S):
                if self.d_config.temperature_scheduler is not None:
                    temperature = self.d_config.temperature_scheduler(l_S, l_T, self.d_config.temperature)
                else:
                    temperature = self.d_config.temperature
                total_kd_loss += self.kd_loss(l_S, l_T, temperature)

            total_loss += total_kd_loss * self.d_config.kd_loss_weight
            losses_dict['unweighted_kd_loss'] = total_kd_loss

        inters_T = {feature: results_T.get(feature,[]) for feature in FEATURES}
        inters_S = {feature: results_S.get(feature,[]) for feature in FEATURES}
        inputs_mask_T = results_T.get('inputs_mask',None)
        inputs_mask_S = results_S.get('inputs_mask',None)

        #hidden states and embedding
        feature = self.emd_feature
        emd_loss_weight = self.emd_loss_weight
        loss_type = self.emd_loss_type
        match_loss = MATCH_LOSS_MAP[loss_type]

        feature_maps_S = inters_S[feature][1:] # list of features
        feature_maps_T = inters_T[feature][1:] # list of features

        embeddings_S = inters_S[feature][0]
        embeddings_T = inters_T[feature][0]

        assert isinstance(feature_maps_S, (tuple,list))
        assert isinstance(feature_maps_T, (tuple,list))
        assert isinstance(feature_maps_S[0], torch.Tensor)
        assert isinstance(feature_maps_T[0], torch.Tensor)
        assert len(feature_maps_S) == self.layer_num_S - 1
        assert len(feature_maps_T) == self.layer_num_T - 1


        if len(self.projs) > 0:
            assert len(self.projs) == self.layer_num_S
            embeddings_S = self.projs[0](embeddings_S)
            feature_maps_S = [proj(s) for proj,s in zip(self.projs[1:],feature_maps_S)]


        feature_num_S = len(feature_maps_S)
        feature_num_T = len(feature_maps_T)
        feature_num_A = feature_num_S + feature_num_T

        distance_matrix = torch.zeros([feature_num_A, feature_num_A]).to(feature_maps_S[0])
        for s in range(feature_num_S):
            f_S = feature_maps_S[s]
            for t in range(feature_num_T):
                f_T = feature_maps_T[t]
                distance_matrix[s][t+feature_num_S] = distance_matrix[t+feature_num_S][s] = match_loss(f_S, f_T, mask=inputs_mask_S) 

        feature_weight_S = np.concatenate([self.feature_weight_S, np.zeros(feature_num_T)])
        feature_weight_T = np.concatenate([np.zeros(feature_num_S), self.feature_weight_T])


        _, trans_matrix = emd_with_flow(feature_weight_S, feature_weight_T, distance_matrix.detach().cpu().numpy().astype('float64'))
        trans_matrix = torch.tensor(trans_matrix).to(distance_matrix)


        emd_loss = torch.sum(trans_matrix * distance_matrix)

        total_loss += emd_loss * emd_loss_weight

        losses_dict[f'unweighted_{feature}_{loss_type}_emd'] = emd_loss

        if (self.feature_weight_S<=0).any() or (self.feature_weight_T<=0).any():
            import sys
            logger.info(f"{self.feature_weight_S}")
            logger.info(f"{self.feature_weight_T}")

        if np.isnan(self.feature_weight_S).any() or np.isnan(self.feature_weight_T).any():
            import sys
            logger.info(f"{self.feature_weight_S}")
            logger.info(f"{self.feature_weight_T}")
            sys.exit()

        #feature_weight_S = np.copy(self.feature_weight_S)
        #feature_weight_T = np.copy(self.feature_weight_T)
        #self.feature_weight_S, self.feature_weight_T = get_new_feature_weight(
        #    trans_matrix, distance_matrix.detach(), feature_weight_S, feature_weight_T, self.d_config.temperature)

        #embedding matching
        embedding_loss = match_loss(embeddings_S, embeddings_T, mask=inputs_mask_S)
        total_loss += embedding_loss * emd_loss_weight #sharing the same weight
        losses_dict[f'unweighted_embedding_{loss_type}'] = embedding_loss


        if 'losses' in results_S:
            total_hl_loss = 0
            for loss in results_S['losses']:
                # in case of multi-GPU
                total_hl_loss += loss.mean() 
            total_loss += total_hl_loss * self.d_config.hard_label_weight
            losses_dict['unweighted_hard_label_loss'] = total_hl_loss
        return total_loss, losses_dict


def get_new_feature_weight(trans_matrix, distance_matrix, feature_weight_S, feature_weight_T, temperature):

    #distance_matrix = distance_matrix.detach().cpu().numpy().astype('float64')
    trans_weight_S = torch.sum(trans_matrix     * distance_matrix, dim=-1).cpu().numpy()
    trans_weight_T = torch.sum(trans_matrix.t() * distance_matrix, dim=-1).cpu().numpy()
    # new_student_weight = torch.zeros(stu_layer_num).cuda()

    feature_num_S = len(feature_weight_S)
    feature_num_T = len(feature_weight_T)

    feature_weight_S = trans_weight_S[:feature_num_S] / feature_weight_S
    weight_sum = np.sum(feature_weight_S)
    for i in range(feature_num_S):
        if feature_weight_S[i] != 0:
            feature_weight_S[i] = weight_sum / feature_weight_S[i]
   #feature_weight_S = weight_sum / (feature_weight_S + 1e-8)

    feature_weight_T = trans_weight_T[feature_num_S:] / feature_weight_T
    weight_sum = np.sum(feature_weight_T)
    for j in range(feature_num_T):
        if feature_weight_T[j] != 0:
            feature_weight_T[j] = weight_sum / feature_weight_T[j]
    #feature_weight_T = weight_sum / (feature_weight_T + 1e-8)

    feature_weight_S = softmax(feature_weight_S / temperature)
    feature_weight_T = softmax(feature_weight_T / temperature)

    return feature_weight_S, feature_weight_T
