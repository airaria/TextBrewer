import torch
from collections import OrderedDict
from tqdm import tqdm
from torch import nn
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter
import os, random, json
import numpy as np
import logging
from typing import Optional, Dict, Union
from .presets import *
from .configurations import TrainingConfig, DistillationConfig

logger = logging.getLogger("Distillation")
logger.setLevel(logging.INFO)

handler_stream = logging.StreamHandler()
handler_stream.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -  %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
handler_stream.setFormatter(formatter)
logger.addHandler(handler_stream)

class CustomMatch:
    def __init__(self, module_T, module_S, weight, loss,
                 proj_func =None, proj_group = None):
        self.module_T = module_T
        self.module_S = module_S
        self.loss     = loss,
        self.weight   = weight,
        self.proj_func     = proj_func
        if proj_group is None:
            self.proj_group = dict()
        else:
            self.proj_group = proj_group
    def to_dict(self):
        return {'module_T':self.module_T,
                'module_S':self.module_S,
                'weight':self.weight,
                'loss':self.loss,
                'proj_func':self.proj_func,
                'proj_group':self.proj_group}
    @classmethod
    def from_dict(cls,dict_object):
        return cls(**dict_object)


class DistillationContext:
    def __init__(self):
        self.model_S = None
        self.model_T = None
    def __enter__(self):
        if isinstance(self.model_T,(list,tuple)):
            self.model_T_is_training = [model_t.training for model_t in self.model_T]
            for model_t in self.model_T:
                model_t.eval()
        elif isinstance(self.model_T,dict):
            self.model_T_is_training = {name:model.training for name,model in self.model_T.items()}
        else:
            self.model_T_is_training = self.model_T.training
            self.model_T.eval()

        self.model_S_is_training = self.model_S.training
        self.model_S.train()

    def __exit__(self, exc_type, exc_val, exc_tb):
        #Restore model status
        if isinstance(self.model_T,(list,tuple)):
            for i in range(len(self.model_T_is_training)):
                self.model_T[i].train(self.model_T_is_training[i])
        elif isinstance(self.model_T,dict):
            for name,is_training  in self.model_T_is_training.items():
                self.model_T[name].train(is_training)
        else:
            self.model_T.train(self.model_T_is_training)

        self.model_S.train(self.model_S_is_training)


class AbstractDistiller(DistillationContext):
    def __init__(self, train_config: TrainingConfig,
                       distill_config: DistillationConfig,
                       model_T, model_S, adaptor_T, adaptor_S):
        super(AbstractDistiller, self).__init__()
        self.t_config = train_config
        self.d_config = distill_config

        self.model_T = model_T
        self.model_S = model_S
        self.adaptor_S = adaptor_S
        self.adaptor_T = adaptor_T

        self.kd_loss = KD_LOSS_MAP[self.d_config.kd_loss_type]
        if self.t_config.log_dir is not None:
            self.tb_writer = SummaryWriter(log_dir = self.t_config.log_dir)
        else:
            self.tb_writer = no_op
        
        self.print_freq = 20

class BasicDistiller(AbstractDistiller):
    def __init__(self, train_config: TrainingConfig,
                       distill_config: DistillationConfig,
                 model_T: Union[List,Dict,torch.nn.Module],
                 model_S: torch.nn.Module,
                 adaptor_T,
                 adaptor_S):
        super(BasicDistiller, self).__init__(train_config, distill_config, model_T, model_S, adaptor_T, adaptor_S)

    def save_and_callback(self,global_step, step, epoch, callback):
        logger.info(f"Saving at global step {global_step}, epoch step {step + 1} epoch {epoch+1}")
        coreModel = self.model_S.module if \
            'DataParallel' in self.model_S.__class__.__name__ else self.model_S
        state_dict = coreModel.state_dict()
        torch.save(state_dict, os.path.join(self.t_config.output_dir, f"gs{global_step}.pkl"))
        if callback is not None:
            logger.info("Running callback function...")
            callback(model=self.model_S, step=global_step)
            self.model_S.train()

    def write_loss(self, total_loss, writer_step):

        cpu_total_loss = total_loss.cpu().item() * self.t_config.gradient_accumulation_steps
        self.tb_writer.add_scalar('scalar/total_loss', cpu_total_loss, writer_step)

        #for name, loss in losses_dict.items():
        #    cpu_loss = loss.cpu().item() * self.t_config.gradient_accumulation_steps
        #    self.tb_writer.add_scalar(f"scalar/{name}", cpu_loss, writer_step)


    def train(self, optimizer, scheduler, dataloader, num_epochs, num_steps=None, callback=None, batch_postprocessor=None, **args):
        if num_steps is not None:
            total_global_steps = num_steps
            ckpt_steps =self.t_config.ckpt_steps
            print_every = ckpt_steps // self.print_freq
            if print_every == 0:
                print_every = ckpt_steps
            checkpoints = [ i * ckpt_steps for i in range(1,num_steps//ckpt_steps+1)] + [total_global_steps]
            logger.info(f"Total training steps: {total_global_steps}")
            logger.info(f"Checkpoints(step): {checkpoints}")

            global_step = 0
            writer_step = 0
            for step, batch in tqdm(enumerate(cycle(dataloader)),disable=None):
                if batch_postprocessor is not None:
                    batch = batch_postprocessor(batch)
                total_loss = self.train_on_batch(batch,args)
                total_loss /= self.t_config.gradient_accumulation_steps
                total_loss.backward()

                self.write_loss(total_loss, writer_step)
                writer_step += 1

                if (step+1)%self.t_config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    self.model_S.zero_grad()
                    global_step += 1
                    if self.d_config.kd_loss_weight_scheduler is not None:
                        self.d_config.kd_loss_weight = \
                            self.d_config.kd_loss_weight_scheduler(global_step/total_global_steps)
                    if self.d_config.hard_label_weight_scheduler is not None:
                        self.d_config.hard_label_weight = \
                            self.d_config.hard_label_weight_scheduler(global_step/total_global_steps)

                    if (global_step) % print_every == 0:
                        logger.info(f"Global step: {global_step}, epoch step:{step+1}")
                    if (global_step%ckpt_steps==0) or global_step==total_global_steps:
                        self.save_and_callback(global_step, step, 0, callback)
            logger.info("Training finished")
            return


        train_steps_per_epoch = len(dataloader)//self.t_config.gradient_accumulation_steps
        total_global_steps = train_steps_per_epoch * num_epochs
        print_every = train_steps_per_epoch // self.print_freq
        if print_every == 0:
            print_every = train_steps_per_epoch
        checkpoints = [int(train_steps_per_epoch*ci/self.t_config.ckpt_frequency) for ci in range(self.t_config.ckpt_frequency)]
        logger.info(f"Training steps per epoch: {train_steps_per_epoch}")
        logger.info(f"Checkpoints(step): {checkpoints}")

        global_step = 0
        writer_step = 0
        for current_epoch in tqdm(range(int(num_epochs)),disable=None):
            logger.info(f"Epoch {current_epoch+1}")
            self.model_S.zero_grad()
            logger.info(f"Length of current epoch in forward batch: {len(dataloader)}")
            for step, batch in tqdm(enumerate(dataloader),disable=None):
                if batch_postprocessor is not None:
                    batch = batch_postprocessor(batch)
                total_loss = self.train_on_batch(batch,args)
                total_loss /= self.t_config.gradient_accumulation_steps
                total_loss.backward()

                self.write_loss(total_loss, writer_step)
                writer_step += 1

                if (step+1)%self.t_config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    self.model_S.zero_grad()
                    global_step += 1
                    if self.d_config.kd_loss_weight_scheduler is not None:
                        self.d_config.kd_loss_weight = \
                            self.d_config.kd_loss_weight_scheduler(global_step/total_global_steps)
                    if self.d_config.hard_label_weight_scheduler is not None:
                        self.d_config.hard_label_weight = \
                            self.d_config.hard_label_weight_scheduler(global_step/total_global_steps)

                    if (global_step) % print_every == 0:
                        logger.info(f"Global step: {global_step}, epoch step:{step+1}")
                    if (global_step%train_steps_per_epoch in checkpoints) \
                            and ((current_epoch+1)%self.t_config.ckpt_epoch_frequency==0 or current_epoch+1==num_epochs):
                        self.save_and_callback(global_step, step, current_epoch, callback)

            logger.info(f"Epoch {current_epoch+1} finished")

    def train_on_batch(self, batch, args):
        #TODO implement caching
        if type(batch) is dict:
            for k,v in batch.items():
                if type(v) is torch.Tensor:
                    batch[k] = v.to(self.t_config.device)
            with torch.no_grad():
                results_T = self.model_T(**batch, **args)
            results_S = self.model_S(**batch, **args)
        else:
            moved_batch = tuple(item.to(self.t_config.device) if type(item) is torch.Tensor else item for item in batch)
            batch = moved_batch
            with torch.no_grad():
                results_T = self.model_T(*batch, **args)
            results_S = self.model_S(*batch, **args)

        results_T = post_adaptor(self.adaptor_T(batch,results_T))
        results_S = post_adaptor(self.adaptor_S(batch,results_S))
        logits_list_T = results_T['logits']  # list of tensor
        logits_list_S = results_S['logits']  # list of tensor
        total_loss  = 0

        if 'logits_mask' in results_S:
            masks_list_S = results_S['logits_mask']
            logits_list_S = _select_logits_with_mask(logits_list_S,masks_list_S)  #(mask_sum, num_of_class)
        if 'logits_mask' in results_T:
            masks_list_T = results_T['logits_mask']
            logits_list_T = _select_logits_with_mask(logits_list_T,masks_list_T)  #(mask_sum, num_of_class)

        if self.d_config.probability_shift is True:
            labels_list = results_S['labels']
            for l_T, l_S, labels in zip(logits_list_T, logits_list_S, labels_list):
                l_T = probability_shift_(l_T, labels)
                if self.d_config.temperature_scheduler is not None:
                    temperature = self.d_config.temperature_scheduler(l_S, l_T, self.d_config.temperature)
                else:
                    temperature = self.d_config.temperature
                kd_loss = self.kd_loss(l_S, l_T, temperature) * self.d_config.kd_loss_weight
                total_loss += kd_loss
        else:
            for l_T,l_S in zip(logits_list_T,logits_list_S):
                if self.d_config.temperature_scheduler is not None:
                    temperature = self.d_config.temperature_scheduler(l_S, l_T, self.d_config.temperature)
                else:
                    temperature = self.d_config.temperature
                kd_loss = self.kd_loss(l_S, l_T, temperature) * self.d_config.kd_loss_weight
                total_loss += kd_loss

        if 'losses' in results_S:
            for loss in results_S['losses']:
                # in case of multi-GPU
                total_loss += loss.mean() * self.d_config.hard_label_weight

        return total_loss


class MultiTeacherDistiller(BasicDistiller):
    def __init__(self, train_config: TrainingConfig,
                 distill_config: DistillationConfig,
                 model_T: List[nn.Module],
                 model_S: nn.Module,
                 adaptor_T,
                 adaptor_S):
        super(MultiTeacherDistiller, self).__init__(
            train_config, distill_config,
            model_T, model_S,
            adaptor_T, adaptor_S)
        if hasattr(self.adaptor_T,'__iter__'):
            assert len(self.adaptor_T)==len(self.model_T)
        self.avg = True

    def train_on_batch(self, batch, args):
        # Basic uses no cache

        selected = None
        num_T = len(self.model_T)

        if type(batch) is dict:
            for k,v in batch.items():
                if type(v) is torch.Tensor:
                    batch[k] = v.to(self.t_config.device)
            with torch.no_grad():
                if self.avg:
                    results_T = [model_t(**batch, **args) for model_t in self.model_T]
                else:
                    selected = random.choice(range(num_T))
                    results_T = [self.model_T[selected](**batch,**args)]
            results_S = self.model_S(**batch, **args)
        else:
            moved_batch = tuple(item.to(self.t_config.device) if type(item) is torch.Tensor else item for item in batch)
            batch = moved_batch
            with torch.no_grad():
                if self.avg:
                    results_T = [model_T(*batch, **args) for model_T in self.model_T]
                else:
                    selected = random.choice(range(num_T))
                    results_T = [self.model_T[selected](*batch,**args)]
            results_S = self.model_S(*batch, **args)

        if hasattr(self.adaptor_T,'__iter__'):
            if self.avg:
                results_T = [post_adaptor(adpt_t(batch,results_t)) for results_t,adpt_t in zip(results_T,self.adaptor_T)]
            else:
                results_T = [post_adaptor(self.adaptor_T[selected](batch,results_T[0]))]
        else:
            results_T = [post_adaptor(self.adaptor_T(batch,results_t)) for results_t in results_T]
        results_S = post_adaptor(self.adaptor_S(batch,results_S))

        logits_list_T = [results_t['logits'] for results_t in results_T]  # list of tensor
        logits_list_S = results_S['logits']  # list of tensor
        total_loss  = 0

        if 'logits_mask' in results_S:
            masks_list_S = results_S['logits_mask']
            logits_list_S = _select_logits_with_mask(logits_list_S,masks_list_S)  #(mask_sum, num_of_class)
        if 'logits_mask' in results_T[0]:
            masks_list_T = results_T[0]['logits_mask']
            logits_list_T = [_select_logits_with_mask(logits_list_t,masks_list_T)
                             for logits_list_t in logits_list_T] #(mask_sum, num_of_class)

        if self.d_config.probability_shift is True:
            labels_list = results_S['labels']
            for l_T, l_S, labels in zip(zip(*logits_list_T),logits_list_S,labels_list):
                mean_l_T = sum(l_T)/len(l_T)
                mean_l_T = probability_shift_(mean_l_T, labels)
                if self.d_config.temperature_scheduler is not None:
                    temperature = self.d_config.temperature_scheduler(l_S, mean_l_T, self.d_config.temperature)
                else:
                    temperature = self.d_config.temperature
                total_loss += self.kd_loss(l_S, mean_l_T, temperature) * self.d_config.kd_loss_weight
        else:
            for l_T, l_S in zip(zip(*logits_list_T),logits_list_S):
                mean_l_T = sum(l_T)/len(l_T)
                if self.d_config.temperature_scheduler is not None:
                    temperature = self.d_config.temperature_scheduler(l_S, mean_l_T, self.d_config.temperature)
                else:
                    temperature = self.d_config.temperature
                total_loss += self.kd_loss(l_S, mean_l_T, temperature) * self.d_config.kd_loss_weight

        if 'losses' in results_S:
            for loss in results_S['losses']:
                # in case of multi-GPU
                total_loss += loss.mean() * self.d_config.hard_label_weight
        return total_loss


class GeneralDistiller(BasicDistiller):
    def __init__(self, train_config: TrainingConfig,
                 distill_config: DistillationConfig,
                 model_T: torch.nn.Module,
                 model_S: torch.nn.Module,
                 adaptor_T,
                 adaptor_S,
                 custom_matches: Optional[List[CustomMatch]] = None):
        # custom_matches=[{'module_T': module_T, 'module_S':module_S,
        #                 'loss': loss, 'weight': weight},...]
        super(GeneralDistiller, self).__init__(train_config, distill_config, model_T, model_S, adaptor_T, adaptor_S)

        self.projs = []
        self.projs_group = []
        for im in self.d_config.intermediate_matches:
            if im.proj is not None:
                projection = im.proj[0]
                dim_in = im.proj[1]
                dim_out = im.proj[2]
                self.projs_group.append(im.proj[3])
                self.projs.append(PROJ_MAP[projection](dim_in,dim_out))
                self.projs[-1].to(self.t_config.device)
            else:
                self.projs.append(None)
                self.projs_group.append(None)

        self.has_custom_matches = False
        if custom_matches:
            self.handles_T = []
            self.handles_S = []
            self.custom_matches_cache = {'hook_outputs_T': [], 'hook_outputs_S': [], 'match_proj_funcs': [],
                                         'match_weights':  [], 'match_losses':   [], 'match_proj_groups': []}
            for match in custom_matches:
                self.add_match(match)
            self.has_custom_matches = True

    def save_and_callback(self,global_step, step, epoch, callback):
        if self.has_custom_matches:
            handles_T = self.model_T._forward_hooks
            handles_S = self.model_S._forward_hooks
            self.model_S._forward_hooks = OrderedDict()  # clear hooks
            self.model_T._forward_hooks = OrderedDict()

        super(GeneralDistiller, self).save_and_callback(global_step, step, epoch, callback)

        if self.has_custom_matches:
            self.model_S._forward_hooks = handles_S  # restore hooks
            self.model_T._forward_hooks = handles_T

    def train(self, optimizer, scheduler, dataloader, num_epochs, num_steps=None, callback=None, batch_postprocessor=None, **args):
        # update optimizer for projection layer
        for proj,proj_group in zip(self.projs, self.projs_group):
            if proj is not None:
                assert isinstance(proj,nn.Module)
                optimizer.add_param_group({**{'params':proj.parameters()},**proj_group})

        if self.has_custom_matches:
            for proj_func,proj_group in zip(self.custom_matches_cache['match_proj_funcs'],
                                                   self.custom_matches_cache['match_proj_groups']):
                if isinstance(proj_func,nn.Module):
                    optimizer.add_param_group({**{'params':proj_func.parameters()},**proj_group})

        logger.debug("Optimizer param group: ")
        for group in optimizer.param_groups:
            for k,v in group.items():
                logger.debug(f"{k}:{v}")

        super(GeneralDistiller, self).train(optimizer, scheduler, dataloader, num_epochs, num_steps, callback, batch_postprocessor, **args)

    def train_on_batch(self, batch, args):
        if type(batch) is dict:
            for k,v in batch.items():
                if type(v) is torch.Tensor:
                    batch[k] = v.to(self.t_config.device)
            with torch.no_grad():
                results_T = self.model_T(**batch, **args)
            results_S = self.model_S(**batch, **args)
        else:
            moved_batch = tuple(item.to(self.t_config.device) if type(item) is torch.Tensor else item for item in batch)
            batch = moved_batch
            with torch.no_grad():
                results_T = self.model_T(*batch, **args)
            results_S = self.model_S(*batch, **args)

        results_T = post_adaptor(self.adaptor_T(batch,results_T))
        results_S = post_adaptor(self.adaptor_S(batch,results_S))

        total_loss  = 0
        if 'logits' in results_T and 'logits' in results_S:
            logits_list_T = results_T['logits']  # list of tensor
            logits_list_S = results_S['logits']  # list of tensor

            if 'logits_mask' in results_S:
                masks_list_S = results_S['logits_mask']
                logits_list_S = _select_logits_with_mask(logits_list_S,masks_list_S)  #(mask_sum, num_of_class)
            if 'logits_mask' in results_T:
                masks_list_T = results_T['logits_mask']
                logits_list_T = _select_logits_with_mask(logits_list_T,masks_list_T)  #(mask_sum, num_of_class)

            if self.d_config.probability_shift is True:
                labels_list = results_S['labels']
                for l_T, l_S, labels in zip(logits_list_T, logits_list_S, labels_list):
                    l_T = probability_shift_(l_T, labels)
                    if self.d_config.temperature_scheduler is not None:
                        temperature = self.d_config.temperature_scheduler(l_S, l_T, self.d_config.temperature)
                    else:
                        temperature = self.d_config.temperature
                    kd_loss = self.kd_loss(l_S, l_T, temperature) * self.d_config.kd_loss_weight
                    total_loss += kd_loss
            else:
                for l_T,l_S in zip(logits_list_T,logits_list_S):
                    if self.d_config.temperature_scheduler is not None:
                        temperature = self.d_config.temperature_scheduler(l_S, l_T, self.d_config.temperature)
                    else:
                        temperature = self.d_config.temperature
                    kd_loss = self.kd_loss(l_S, l_T, temperature) * self.d_config.kd_loss_weight
                    total_loss += kd_loss

        inters_T = {feature: results_T.get(feature,[]) for feature in FEATURES}
        inters_S = {feature: results_S.get(feature,[]) for feature in FEATURES}
        inputs_mask_T = results_T.get('inputs_mask',None)
        inputs_mask_S = results_S.get('inputs_mask',None)
        for ith,inter_match in enumerate(self.d_config.intermediate_matches):
            layer_T = inter_match.layer_T
            layer_S = inter_match.layer_S
            feature = inter_match.feature
            loss_type = inter_match.loss
            match_weight = inter_match.weight
            match_loss = MATCH_LOSS_MAP[loss_type]

            if type(layer_S) is list and type(layer_T) is list:
                inter_S = [inters_S[feature][s] for s in layer_S]
                inter_T = [inters_T[feature][t] for t in layer_T]
                if self.projs[ith]:
                    #inter_T = [self.projs[ith](t) for t in inter_T]
                    inter_S = [self.projs[ith](s) for s in inter_S]
            else:
                inter_S = inters_S[feature][layer_S]
                inter_T = inters_T[feature][layer_T]
                if self.projs[ith]:
                    #inter_T = self.projs[ith](inter_T)
                    inter_S = self.projs[ith](inter_S)
            total_loss += match_loss(inter_S, inter_T, mask=inputs_mask_S) * match_weight


        if self.has_custom_matches:
            for hook_T, hook_S, match_weight, match_loss, proj_func  in \
                    zip(self.custom_matches_cache['hook_outputs_T'], self.custom_matches_cache['hook_outputs_S'],
                        self.custom_matches_cache['match_weghts'], self.custom_matches_cache['match_losses'],
                        self.custom_matches_cache['match_proj_funcs']):
                if proj_func is not None:
                    hook_S = proj_func(hook_S)
                total_loss += match_weight * match_loss(hook_S,hook_T,inputs_mask_S,inputs_mask_T)
            self.custom_matches_cache['hook_outputs_T'] = []
            self.custom_matches_cache['hook_outputs_S'] = []

        if 'losses' in results_S:
            for loss in results_S['losses']:
                # in case of multi-GPU
                total_loss += loss.mean() * self.d_config.hard_label_weight

        return total_loss

    def add_match(self,match: CustomMatch):
        if type(match.module_T) is str or type(match.module_S) is str:
            raise NotImplementedError
        else:
            module_T   = match.module_T
            module_S   = match.module_S
            weight     = match.weight
            loss       = match.loss
            proj_func = match.proj_func
            proj_group = match.proj_group
        self.add_match_by_module(module_T,module_S,proj_func,proj_group,weight,loss)


    def add_match_by_module(self,module_T : torch.nn.Module,
                                 module_S : torch.nn.Module,
                                 proj_func, proj_group,
                                 match_weight, match_loss):

        self.handles_T = module_T.register_forward_hook(self._hook_T)
        self.handles_S = module_S.register_forward_hook(self._hook_S)
        self.custom_matches_cache['match_weights'].append(match_weight)
        self.custom_matches_cache['match_losses'].append(match_loss)
        self.custom_matches_cache['match_proj_funcs'].append(proj_func)
        if isinstance(proj_func,nn.Module):
            self.custom_matches_cache['match_proj_funcs'][-1].to(self.t_config.device)
        self.custom_matches_cache['match_proj_groups'].append(proj_group)

    def _hook_T(self,module,input, output):
        self.custom_matches_cache['hook_outputs_T'].append(output)

    def _hook_S(self, module, input, output):
        self.custom_matches_cache['hook_outputs_S'].append(output)



class MultiTaskDistiller(BasicDistiller):
    def __init__(self, train_config: TrainingConfig,
                 distill_config: DistillationConfig,
                 model_T,
                 model_S,
                 adaptor_T,
                 adaptor_S):
        '''
        :param config:
        :param model_T: Dict of teacher model: {task1:model_1, task2:model_2, .... }. Keys are tasknames
        :param model_S:
        :param output_adaptor_T: Dict of teacher adaptors: {task1:adpt_1, task2:adpt_2, .... }. Keys are tasknames
        :param output_adaptor_S: Dict of student adaptors: {task1:adpt_1, task2:adpt_2, .... }. Keys are tasknames
        '''
        super(MultiTaskDistiller, self).__init__(
            train_config, distill_config,
            model_T, model_S,
            adaptor_T, adaptor_S)
        if hasattr(self.adaptor_T,'__iter__'):
            assert len(self.adaptor_T)==len(self.model_T)==len(self.adaptor_S)
        assert (self.d_config.kd_loss_weight_scheduler is None) and (self.d_config.hard_label_weight_scheduler is None),\
                "BasicMultiTaskDistiller does not support WEIGHT_SCHEDULER in the current version."


    def train(self, optimizer, scheduler, dataloaders, num_steps, tau=1, callback=None, batch_postprocessors=None, **args):
        '''
        :param dataloaders:  {taskname1: dataloader1, taskname2: dataloader2, ... }
        :param batch_postprocessors: {taskname1: proc1, taskname2: proc2, ...}
        '''
        total_global_steps = num_steps
        ckpt_steps =self.t_config.ckpt_steps
        print_every = ckpt_steps // self.print_freq
        if print_every == 0:
            print_every = ckpt_steps
        checkpoints = [ i * ckpt_steps for i in range(1,num_steps//ckpt_steps+1)] + [total_global_steps]
        logger.info(f"Total training steps: {total_global_steps}")
        logger.info(f"Checkpoints(step): {checkpoints}")

        dataiters = {k:cycle(v) for k,v in dataloaders}
        if all(hasattr(v,'__len__') for v in dataloaders.values()):
            dataloader_sizes = {k:len(v) for k,v in dataloaders.items()}
            total_size = sum(v for k,v in dataloader_sizes.items())//self.t_config.gradient_accumulation_steps
            logger.info(f"Total size of all datasets (in number of batch_size):{total_size}")
            Z = sum(pow(v,tau) for v in dataloader_sizes.values())
            tasknames, sampling_weights = zip(*((k,pow(v,tau)/Z) for k,v in dataloader_sizes.items()))
        else:
            logger.info("The size of some datasets are unknown, so tau=1")
            tasknames = tuple(dataloaders.keys())
            sampling_weights = None

            
        global_step = 0
        writer_step = 0
        self.model_S.zero_grad()
        while global_step < num_steps:
            global_step += 1
            for _ in range(self.t_config.gradient_accumulation_steps):
                #sampling taskname
                taskname = np.random.choice(tasknames,p=sampling_weights)
                dataiter = dataiters[taskname]
                batch = next(dataiter)
                if batch_postprocessors is not None:
                    batch = batch_postprocessors[taskname](batch)
                batch_taskname = (batch, taskname)
                total_loss = self.train_on_batch(batch_taskname, args)
                total_loss /= self.t_config.gradient_accumulation_steps
                total_loss.backward()
                scalar_total_loss = total_loss.cpu().item() * self.t_config.gradient_accumulation_steps
                self.tb_writer.add_scalar('scalar/total_loss', scalar_total_loss, writer_step)
                writer_step += 1
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            self.model_S.zero_grad()

            if self.d_config.kd_loss_weight_scheduler is not None:
                self.d_config.kd_loss_weight = \
                    self.d_config.kd_loss_weight_scheduler(global_step/total_global_steps)
            if self.d_config.hard_label_weight_scheduler is not None:
                self.d_config.hard_label_weight = \
                    self.d_config.hard_label_weight_scheduler(global_step/total_global_steps)

            if (global_step) % print_every == 0:
                logger.info(f"Global step: {global_step}/{num_steps}")
            if (global_step % ckpt_steps == 0) or global_step==total_global_steps:
                self.save_and_callback(global_step, global_step-1, 0, callback)
        logger.info("Training finished")

    def train_on_batch(self, batch_taskname, args) -> torch.Tensor:
        # Basic uses no cache
        batch, taskname = batch_taskname
        model_T = self.model_T[taskname]
        adaptor_T = self.adaptor_T[taskname]
        adaptor_S = self.adaptor_S[taskname]

        if type(batch) is dict:
            for k,v in batch.items():
                if type(v) is torch.Tensor:
                    batch[k] = v.to(self.t_config.device)
            with torch.no_grad():
                results_T = model_T(**batch, **args)
            results_S = self.model_S(**batch, **args)
        else:
            moved_batch = tuple(item.to(self.t_config.device) if type(item) is torch.Tensor else item for item in batch)
            batch = moved_batch
            with torch.no_grad():
                results_T = model_T(*batch, **args)
            results_S = self.model_S(*batch, **args)

        results_T = post_adaptor(adaptor_T(batch,results_T))
        results_S = post_adaptor(adaptor_S(batch,results_S))

        logits_list_T = results_T['logits']  # list of tensor
        logits_list_S = results_S[taskname]['logits']  # list of tensor
        total_loss  = 0

        if 'logits_mask' in results_S[taskname]:
            masks_list_S = results_S[taskname]['logits_mask']
            logits_list_S = _select_logits_with_mask(logits_list_S,masks_list_S)  #(mask_sum, num_of_class)
        if 'logits_mask' in results_T: #TODO
            masks_list_T = results_T['logits_mask']
            logits_list_T = _select_logits_with_mask(logits_list_T,masks_list_T)

        if self.d_config.probability_shift is True:
            labels_list = results_S['labels']
            for l_T, l_S, labels in zip(logits_list_T, logits_list_S, labels_list):
                l_T = probability_shift_(l_T, labels)
                if self.d_config.temperature_scheduler is not None:
                    temperature = self.d_config.temperature_scheduler(l_S, l_T, self.d_config.temperature)
                else:
                    temperature = self.d_config.temperature
                kd_loss = self.kd_loss(l_S, l_T, temperature) * self.d_config.kd_loss_weight
                total_loss += kd_loss
        else:
            for l_T,l_S in zip(logits_list_T,logits_list_S):
                if self.d_config.temperature_scheduler is not None:
                    temperature = self.d_config.temperature_scheduler(l_S, l_T, self.d_config.temperature)
                else:
                    temperature = self.d_config.temperature
                total_loss += self.kd_loss(l_S, l_T, temperature) * self.d_config.kd_loss_weight

        if 'losses' in results_S:
            for loss in results_S[taskname]['losses']:
                # in case of multi-GPU
                total_loss += loss.mean() * self.d_config.hard_label_weight

        return total_loss


def _select_logits_with_mask(logits_list, masks_list):
    output_logits = []
    if len(masks_list)==len(logits_list):
        for logits,mask in zip(logits_list,masks_list):
            if len(logits.shape)==3:
                mask = mask.unsqueeze(-1).expand_as(logits).to(torch.uint8)
                logits_select = torch.masked_select(logits,mask).view(-1,logits.size(-1))
            else:
                logits_select = logits #Logits_mask has no effect on logits of shape (batch_size, logits_to_be_softmaxed)
            output_logits.append(logits_select)
    elif len(masks_list)==1:
        mask = masks_list[0]
        for logits in logits_list:
            if len(logits.shape)==3:
                mask = mask.unsqueeze(-1).expand_as(logits).to(torch.uint8)
                logits_select = torch.masked_select(logits,mask).view(-1,logits.size(-1))
            else:
                logits_select = logits #Logits_mask has no effect on logits of shape (batch_size, logits_to_be_softmaxed)
            output_logits.append(logits_select)
    else:
        raise AssertionError("lengths of logits list and masks list mismatch")
    return output_logits


class BasicAdaptor:
    def __init__(self):
        self.batch = None
        self.model_outputs = None
    def __call__(self,batch,model_outputs):
        self.batch = batch
        self.model_outputs = model_outputs
    def __getattr__(self, item):
        raise NotImplementedError


def post_adaptor(dict_object):
    if 'logits' in dict_object:
        logits = dict_object['logits']
        if not isinstance(logits,(list,tuple)):
            dict_object['logits'] = [ logits ]
    if 'logits_mask' in dict_object:
        logits_mask = dict_object['logits_mask']
        if not isinstance(logits_mask,(list,tuple)):
            dict_object['logits_mask'] = [ logits_mask ]
    if 'losses' in dict_object:
        losses = dict_object['losses']
        if not isinstance(losses,(list,tuple)):
            dict_object['losses'] = [ losses ]
    if 'labels' in dict_object:
        labels = dict_object['labels']
        if not isinstance(labels,(list,tuple)):
            dict_object['labels'] = [ labels ]
    return dict_object


class BasicTrainer:
    def __enter__(self):
        self.model_is_training = self.model.training
        self.model.train()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore model status
        self.model.train(self.model_is_training)

    def __init__(self, train_config: TrainingConfig,
                 model: torch.nn.Module, adaptor):
        super(BasicTrainer, self).__init__()
        self.t_config = train_config
        self.model = model
        self.adaptor = adaptor
        if self.t_config.log_dir is not None:
            self.tb_writer = SummaryWriter(log_dir = self.t_config.log_dir)
        else:
            self.tb_writer = no_op
        self.print_freq = 20

    def train(self, optimizer, scheduler, dataloader, num_epochs, num_steps=None, callback=None, batch_postprocessor=None, **args):
        if num_steps is not None:
            total_global_steps = num_steps
            ckpt_steps =self.t_config.ckpt_steps
            print_every = ckpt_steps // self.print_freq
            if print_every == 0:
                print_every = ckpt_steps
            checkpoints = [ i * ckpt_steps for i in range(1,num_steps//ckpt_steps+1)] + [total_global_steps]
            logger.info(f"Total training steps: {total_global_steps}")
            logger.info(f"Checkpoints: {checkpoints}")

            global_step = 0
            writer_step = 0
            for step, batch in tqdm(enumerate(cycle(dataloader)),disable=None):
                if batch_postprocessor is not None:
                    batch = batch_postprocessor(batch)
                total_loss = self.train_on_batch(batch,args)
                total_loss /= self.t_config.gradient_accumulation_steps
                total_loss.backward()

                scalar_total_loss = total_loss.cpu().item() * self.t_config.gradient_accumulation_steps
                self.tb_writer.add_scalar('scalar/total_loss', scalar_total_loss, writer_step)
                writer_step += 1

                if (step+1)%self.t_config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    self.model.zero_grad()
                    global_step += 1
                    if (global_step) % print_every == 0:
                        logger.info(f"Global step: {global_step}, epoch step:{step+1}")
                    if (global_step%ckpt_steps==0) or global_step==total_global_steps:
                        logger.info(f"Saving at global step {global_step}")
                        coreModel = self.model.module if \
                            'DataParallel' in self.model.__class__.__name__ else self.model
                        state_dict = coreModel.state_dict()
                        torch.save(state_dict, os.path.join(self.t_config.output_dir,f"gs{global_step}.pkl"))
                        if callback is not None:
                            logger.info("Running callback function...")
                            callback(model=self.model, step=global_step)
                            self.model.train()
            logger.info("Training finished")
            return

        train_steps_per_epoch = len(dataloader)//self.t_config.gradient_accumulation_steps
        print_every = train_steps_per_epoch // self.print_freq
        if print_every == 0:
            print_every = train_steps_per_epoch
        checkpoints = [int(train_steps_per_epoch*ci/self.t_config.ckpt_frequency) for ci in range(self.t_config.ckpt_frequency)]
        logger.info(f"Training steps per epoch: {train_steps_per_epoch}")
        logger.info(f"Checkpoints(step): {checkpoints}")

        global_step = 0
        writer_step = 0
        for current_epoch in tqdm(range(int(num_epochs)),disable=None):
            logger.info(f"Epoch {current_epoch+1}")
            self.model.zero_grad()
            logger.info(f"Length of current epoch in forward batch: {len(dataloader)}")
            for step, batch in tqdm(enumerate(dataloader),disable=None):
                if batch_postprocessor is not None:
                    batch = batch_postprocessor(batch)
                total_loss = self.train_on_batch(batch,args)
                total_loss /= self.t_config.gradient_accumulation_steps
                total_loss.backward()

                scalar_total_loss = total_loss.cpu().item() * self.t_config.gradient_accumulation_steps
                self.tb_writer.add_scalar('scalar/total_loss', scalar_total_loss, writer_step)
                writer_step += 1

                if (step+1)%self.t_config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    self.model.zero_grad()
                    global_step += 1
                    if (global_step) % print_every == 0:
                        logger.info(f"Global step: {global_step}, epoch step:{step+1}")
                    if (global_step%train_steps_per_epoch in checkpoints) \
                            and ((current_epoch+1)%self.t_config.ckpt_epoch_frequency==0 or current_epoch+1==num_epochs):
                        logger.info(f"Saving at global step {global_step}, epoch step {step+1} epoch {current_epoch+1}")
                        coreModel = self.model.module if \
                            'DataParallel' in self.model.__class__.__name__ else self.model
                        state_dict = coreModel.state_dict()
                        torch.save(state_dict, os.path.join(self.t_config.output_dir,f"gs{global_step}.pkl"))
                        if callback is not None:
                            logger.info("Running callback function...")
                            callback(model=self.model, step=global_step)
                            self.model.train()

            logger.info(f"Epoch {current_epoch+1} finished")

    def train_on_batch(self, batch, args) -> torch.Tensor:
        if type(batch) is dict:
            for k,v in batch.items():
                if type(v) is torch.Tensor:
                    batch[k] = v.to(self.t_config.device)
            results = self.model(**batch, **args)
        else:
            moved_batch = tuple(item.to(self.t_config.device) if type(item) is torch.Tensor else item for item in batch)
            batch = moved_batch
            results = self.model(*batch, **args)

        results = post_adaptor(self.adaptor(batch,results))
        total_loss  = 0

        if 'losses' not in results:
            raise KeyError("'losses' not in the output of adaptor. Nothing to optimize!")
        else:
            for loss in results['losses']:
                # in case of multi-GPU
                total_loss += loss.mean()

        return total_loss


def probability_shift_(tensor, labels):  # In-place operation. shape (batch_size, num_classes), (batch_size,)
    if len(tensor.shape)==2:
        max_position = tensor.argmax(dim=-1) # shape (batch_size,)
        index = torch.arange(tensor.size(0)).to(tensor.device)
        max_clone = tensor[index,max_position].clone()
        truth_clone = tensor[index,labels].clone()

        tensor[index,max_position] = truth_clone
        tensor[index,labels] = max_clone
        return tensor

    elif len(tensor.shape)==3:   # shape (batch_size, length, num_classes)
        original_shape = tensor.size()

        tensor = tensor.view(-1,tensor.size(-1))   # (batch_size * length, num_classes)

        max_position = tensor.argmax(dim=-1) # shape (batch_size * length, )
        labels = labels.view(-1) # (batch_size * length, )
        nonneg_labels = torch.where(labels<0, max_position, labels)

        index = torch.arange(tensor.size(0)).to(tensor.device)   # (batch_size * length)

        max_clone = tensor[index,max_position].clone()
        truth_clone = tensor[index,nonneg_labels].clone()

        tensor[index,max_position] = truth_clone
        tensor[index,nonneg_labels] = max_clone
        tensor = tensor.view(original_shape)
        return tensor
    else:
        raise TypeError("Rank of tensor must be 2 or 3")

class no_op:
    @staticmethod
    def add_scalar(*args, **kwargs):
        pass
