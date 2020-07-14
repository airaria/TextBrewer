from .distiller_utils import *
from .distiller_basic import BasicDistiller

class MultiTaskDistiller(BasicDistiller):
    """
    distills multiple teacher models (of different tasks) into a single student. **It doesn't support intermediate feature matching**.

    Args:
        train_config (:class:`TrainingConfig`): training configuration.
        distill_config (:class:`DistillationConfig`): distillation configuration.
        model_T (dict): dict of teacher models: {task1:model1, task2:model2, .... }. Keys are tasknames.
        model_S (torch.nn.Module): student model.
        adaptor_T (dict): dict of teacher adaptors: {task1:adpt1, task2:adpt2, .... }. Keys are tasknames.
        adaptor_S (dict): dict of student adaptors: {task1:adpt1, task2:adpt2, .... }. Keys are tasknames.

    """
    
    def __init__(self, train_config,
                 distill_config,
                 model_T,
                 model_S,
                 adaptor_T,
                 adaptor_S):

        super(MultiTaskDistiller, self).__init__(
            train_config, distill_config,
            model_T, model_S,
            adaptor_T, adaptor_S)
        if hasattr(self.adaptor_T,'__iter__'):
            assert len(self.adaptor_T)==len(self.model_T)==len(self.adaptor_S)
        assert (self.d_config.kd_loss_weight_scheduler is None) and (self.d_config.hard_label_weight_scheduler is None),\
                "BasicMultiTaskDistiller does not support WEIGHT_SCHEDULER in the current version."

        self.d_config.is_caching_logits = False

    def train(self, optimizer, dataloaders, num_steps, scheduler_class=None, scheduler_args=None, scheduler=None, max_grad_norm = -1.0, tau=1, callback=None, batch_postprocessors=None, **args):
        """
        trains the student model.

        Args:
            optimizer: optimizer.
            dataloaders (dict): dict of dataset iterator. Keys are tasknames, values are corresponding dataloaders.
            num_steps (int): number of training steps.
            scheduler_class (class): the class of the scheduler to be constructed.
            scheduler_args (dict): arguments (excluding `optimizer`) passed to the `scheduler_class` to construct the scheduler object.
            scheduler (deprecated): used to adjust learning rate, optional, can be None, is deprecated in favor of `scheduler_class` and `scheduler_args`.
            max_grad_norm (float): Maximum norm for the gradients (-1 means no clipping). Default: -1.0
            tau (float): the probability of sampling an example from task `d` is proportional to \|d\|^{tau}, where \|d\| is the size of `d`'s training set. If the size of any dataset is unknown, ignores tau and samples examples unifromly from each dataset.
            callback (Callable): function called after each epoch, can be None. It is called as ``callback(model=self.model_S, step = global_step)``. It can be used to do evaluation of the model at each checkpoint.
            batch_postprocessors (dict): a dict of batch_postprocessors. Keys are tasknames, values are corresponding batch_postprocessors. Each batch_postprocessor should take a batch and return a batch.
            **args: additional arguments fed to the model.
        """

        # update scheduler
        if scheduler_class is not None:
            # overwrite scheduler
            scheduler = scheduler_class(**{'optimizer':optimizer},**scheduler_args)

        if self.t_config.fp16:
            if not has_apex:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            tasknames, model_Ts = zip(*self.model_T.items())
            models = [self.model_S] + list(model_Ts)
            models, optimizer = amp.initialize(models, optimizer, opt_level=self.t_config.fp16_opt_level)
            self.model_S = models[0]
            self.model_T = dict(zip(tasknames,models[1:]))
        if self.t_config.data_parallel:
            self.model_S = torch.nn.DataParallel(self.model_S)
            self.model_T = {k:torch.nn.DataParallel(v) for k,v in self.model_T.items()}

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
        optimizer.zero_grad()
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
                if self.t_config.fp16:
                    with amp.scale_loss(total_loss,optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    total_loss.backward()
                scalar_total_loss = total_loss.cpu().item() * self.t_config.gradient_accumulation_steps
                self.tb_writer.add_scalar('scalar/total_loss', scalar_total_loss, writer_step)
                writer_step += 1
            if max_grad_norm > 0:
                if self.t_config.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model_S.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

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
            logits_list_S = select_logits_with_mask(logits_list_S,masks_list_S)  #(mask_sum, num_of_class)
        if 'logits_mask' in results_T: #TODO
            masks_list_T = results_T['logits_mask']
            logits_list_T = select_logits_with_mask(logits_list_T,masks_list_T)

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