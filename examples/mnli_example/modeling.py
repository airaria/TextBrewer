import logging
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn.functional as F
import config as Conf
from pytorch_pretrained_bert.my_modeling import BertModel, BertLayerNorm
logger = logging.getLogger(__name__)


def initializer_builder(std):
    _std = std
    def init_bert_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=_std)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    return init_bert_weights


class BertForGLUE(nn.Module):
    def __init__(self, config, num_labels):
        super(BertForGLUE, self).__init__()
        self.num_labels = num_labels
        output_sum = None if Conf.args.output_sum < 0 else Conf.args.output_sum
        self.bert = BertModel(config,Conf.args.output_score, output_sum)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        initializer = initializer_builder(config.initializer_range)
        self.apply(initializer)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, logits_T=None,
                attention_probs_sum_layer=None, attention_probs_sum_T=None, hidden_match_layer=None, hidden_match_T=None):
        if hidden_match_layer is not None:
            sequence_output, pooled_output, attention_probs_sum = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True,  output_attention_layer=attention_probs_sum_layer)
            hidden_states = [sequence_output[i] for i in hidden_match_layer]
            sequence_output = sequence_output[-1]
        else:
            sequence_output, pooled_output, attention_probs_sum = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False,  output_attention_layer=attention_probs_sum_layer)
            hidden_states = []


        output_for_cls = self.dropout(pooled_output)
        logits  = self.classifier(output_for_cls)  # output size: batch_size,num_labels

        if logits_T is not None or labels is not None or attention_probs_sum_T is not None:
            total_loss = 0
            hidden_losses = None
            att_losses = None
            if logits_T is not None and self.num_labels!=1:
                temp=Conf.args.temperature
                logits_T /= temp
                logits /= temp
                prob_T = F.softmax(logits_T,dim=-1)
                ce_loss = -(prob_T * F.log_softmax(logits, dim=-1)).sum(dim=-1)
                ce_loss = ce_loss.mean() #* temp_2
                total_loss += ce_loss
            if attention_probs_sum_T:
                if Conf.args.mask_inter==1:
                    mask = attention_mask.to(attention_probs_sum[0])
                    valid_count = torch.pow(mask.sum(dim=1),2).sum()
                    att_losses = [(F.mse_loss(attention_probs_sum[i], attention_probs_sum_T[i], reduction='none') * mask.unsqueeze(-1) * mask.unsqueeze(1)).sum() / valid_count for i in range(len(attention_probs_sum_T))]
                else:
                    att_losses = [F.mse_loss(attention_probs_sum[i], attention_probs_sum_T[i]) for i in range(len(attention_probs_sum_T))]
                att_loss = sum(att_losses) * Conf.args.att_loss_weight
                total_loss += att_loss
                #mle_loss = (F.mse_loss(start_logits,start_logits_T) + F.mse_loss(end_logits,end_logits_T))/2
                #total_loss += mle_loss
            if hidden_match_T:
                if Conf.args.mask_inter==1:
                    mask = attention_mask.to(hidden_states[0])
                    valid_count = mask.sum() * hidden_states[0].size(-1)
                    hidden_losses = [(F.mse_loss(hidden_states[i],hidden_match_T[i], reduction='none')*mask.unsqueeze(-1)).sum() / valid_count for i in range(len(hidden_match_layer))]
                else:
                    hidden_losses = [F.mse_loss(hidden_states[i],hidden_match_T[i]) for i in range(len(hidden_match_layer))]
                hidden_loss = sum(hidden_losses) * Conf.args.hidden_loss_weight
                total_loss += hidden_loss

            if labels is not None:
                if self.num_labels == 1:
                    loss = F.mse_loss(logits.view(-1), labels.view(-1))
                else:
                    loss = F.cross_entropy(logits,labels)
                total_loss += loss
            return total_loss, att_losses, hidden_losses
        else:
            if attention_probs_sum_layer is not None or hidden_match_layer is not None:
                return logits, attention_probs_sum, hidden_states
            else:
                return logits, None

class BertForGLUESimple(nn.Module):
    def __init__(self, config, num_labels, args):
        super(BertForGLUESimple, self).__init__()
        self.num_labels = num_labels
        self.output_encoded_layers   = (args.output_encoded_layers=='true')
        self.output_attention_layers = (args.output_attention_layers=='true')
        self.bert = BertModel(config, output_score=(args.output_att_score=='true'), output_sum=(args.output_att_sum=='true'))
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        initializer = initializer_builder(config.initializer_range)
        self.apply(initializer)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        sequence_output, pooled_output, attention_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                                    output_all_encoded_layers=(self.output_encoded_layers),
                                                                    output_all_attention_layers=(self.output_attention_layers))
        output_for_cls = self.dropout(pooled_output)
        logits  = self.classifier(output_for_cls)  # output size: batch_size,num_labels
        #assert len(sequence_output)==self.bert.config.num_hidden_layers + 1  # embeddings + 12 hiddens
        #assert len(attention_output)==self.bert.config.num_hidden_layers + 1 # None + 12 attentions
        if labels is not None:
            if self.num_labels == 1:
                loss = F.mse_loss(logits.view(-1), labels.view(-1))
            else:
                loss = F.cross_entropy(logits,labels)
            return logits, sequence_output, attention_output, loss
        else:
            return logits



def BertForGLUESimpleAdaptor(batch, model_outputs, no_logits, no_mask):
    dict_obj = {'hidden': model_outputs[1], 'attention': model_outputs[2]}
    if no_mask is False:
        dict_obj['inputs_mask'] = batch[1]
    if no_logits is False:
        dict_obj['logits'] = (model_outputs[0],)
    return dict_obj

def BertForGLUESimpleAdaptorTraining(batch, model_outputs):
    return {'losses':(model_outputs[3],)}
