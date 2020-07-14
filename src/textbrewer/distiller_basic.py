from .distiller_utils import *

class BasicDistiller(AbstractDistiller):
    """
    Performs **single-teacher single-task** distillation, provides basic distillation strategies.

    Args:
        train_config (:class:`TrainingConfig`): training configuration.
        distill_config (:class:`DistillationConfig`): distillation configuration.
        model_T (:class:`torch.nn.Module`): teacher model.
        model_S (:class:`torch.nn.Module`): student model.
        adaptor_T (Callable): teacher model's adaptor.
        adaptor_S (Callable): student model's adaptor.

    The roles of `adaptor_T` and `adaptor_S` are explained in :py:func:`adaptor`.

    """
    def __init__(self, train_config,
                       distill_config,
                 model_T,
                 model_S,
                 adaptor_T,
                 adaptor_S):
        super(BasicDistiller, self).__init__(train_config, distill_config, model_T, model_S, adaptor_T, adaptor_S)

    def save_and_callback(self,global_step, step, epoch, callback):
        logger.info(f"Saving at global step {global_step}, epoch step {step + 1} epoch {epoch+1}")
        coreModel = self.model_S.module if hasattr(self.model_S, "module") else self.model_S
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


    def train(self, optimizer, dataloader, num_epochs, scheduler_class=None, scheduler_args=None, scheduler=None, max_grad_norm = -1.0, num_steps=None, callback=None, batch_postprocessor=None, **args):
        """
        trains the student model.

        Args:
            optimizer: optimizer.
            dataloader: dataset iterator.
            num_epochs (int): number of training epochs.
            num_steps (int): number of training steps. If it is not None, distiller will ignore `num_epochs` and trains for `num_steps`, and dataloader can have an unkonwn size, i.e., has no `__len__` attribute. Dataloader will be cycled automatically after iterating over the whole dataset.
            callback (Callable): function called after each epoch, can be None. It is called as ``callback(model=self.model_S, step = global_step)``. It can be used to evaluate the model at each checkpoint.
            batch_postprocessor (Callable): a function for post-processing batches. It should take a batch and return a batch. Its output is fed to the models and adaptors.
            scheduler_class (class): the class of the scheduler to be constructed.
            scheduler_args (dict): arguments (excluding `optimizer`) passed to the `scheduler_class` to construct the scheduler object. See the example below.
            scheduler (deprecated): used to adjust learning rate, optional, can be None, is deprecated in favor of `scheduler_class` and `scheduler_args`.
            max_grad_norm (float): Maximum norm for the gradients (-1 means no clipping). Default: -1.0
            **args: additional arguments fed to the model.
        Note:
            * If the batch is a list or tuple, model is called as: ``model(*batch, **args)``. Make sure the order of elements in the batch matches their order in ``model.forward``.
            * If the batch is a dict, model is called as: ``model(**batch,**args)``. Make sure the keys of the batch match the arguments of the ``model.forward``.
        Note:
            If you want to provide a lr scheduler, DON'T USE `scheduler` , use `scheduler_class` and `scheduler_args` instead. Example:

            .. code-block::

                from transformers import get_linear_schedule_with_warmup
                distiller.train(optimizer, scheduler_class = get_linear_schedule_with_warmup, scheduler_args= {'num_warmup_steps': 100, 'num_training_steps': 1000})
        """

        # update scheduler
        if scheduler_class is not None:
            # overwrite scheduler
            scheduler = scheduler_class(**{'optimizer':optimizer},**scheduler_args)

        if self.t_config.fp16:
            if not has_apex:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            if isinstance(self.model_T,(list,tuple)):
                models = [self.model_S] + list(self.model_T)
                models, optimizer = amp.initialize(models, optimizer, opt_level=self.t_config.fp16_opt_level)
                self.model_S = models[0]
                self.model_T =models[1:]
            else:
                (self.model_S, self.model_T), optimizer = amp.initialize([self.model_S, self.model_T], optimizer, opt_level=self.t_config.fp16_opt_level)
        if self.t_config.data_parallel:
            self.model_S = torch.nn.DataParallel(self.model_S)
            if isinstance(self.model_T,(list,tuple)):
                self.model_T = [torch.nn.DataParallel(model_t) for model_t in self.model_T]
            else:
                self.model_T = torch.nn.DataParallel(self.model_T)

        if num_steps is not None:
            if self.d_config.is_caching_logits is True:
                logger.warning("is_caching_logits is True, but num_steps is not None!")
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
                if self.t_config.fp16:
                    with amp.scale_loss(total_loss,optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    total_loss.backward()

                self.write_loss(total_loss, writer_step)
                writer_step += 1

                if (step+1)%self.t_config.gradient_accumulation_steps == 0:
                    if max_grad_norm > 0:
                        if self.t_config.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model_S.parameters(), max_grad_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
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

        if self.d_config.is_caching_logits is True:
            logger.info(f"Caching batches and teacher's logits...")
            for step, batch in tqdm(enumerate(dataloader),disable=None):
                self.cache_logits(batch, args, batch_postprocessor)

        for current_epoch in tqdm(range(int(num_epochs)),disable=None):
            logger.info(f"Epoch {current_epoch+1}")
            optimizer.zero_grad()
            if self.d_config.is_caching_logits:
                random.shuffle(self.logits_cache)
                dataloader = self.logits_cache
            logger.info(f"Length of current epoch in forward batch: {len(dataloader)}")
            for step, batch in tqdm(enumerate(dataloader),disable=None):
                if self.d_config.is_caching_logits is False and batch_postprocessor is not None:
                        batch = batch_postprocessor(batch)
                total_loss = self.train_on_batch(batch,args)
                total_loss /= self.t_config.gradient_accumulation_steps
                if self.t_config.fp16:
                    with amp.scale_loss(total_loss,optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    total_loss.backward()

                self.write_loss(total_loss, writer_step)
                writer_step += 1

                if (step+1)%self.t_config.gradient_accumulation_steps == 0:
                    if max_grad_norm > 0:
                        if self.t_config.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model_S.parameters(), max_grad_norm) 
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if self.d_config.kd_loss_weight_scheduler is not None:
                        self.d_config.kd_loss_weight = \
                            self.d_config.kd_loss_weight_scheduler(global_step/total_global_steps)
                    if self.d_config.hard_label_weight_scheduler is not None:
                        self.d_config.hard_label_weight = \
                            self.d_config.hard_label_weight_scheduler(global_step/total_global_steps)

                    if (global_step) % print_every == 0:
                        logger.info(f"Global step: {global_step}, epoch step:{step+1}")
                        #logger.info(f"lrs:{[g['lr'] for g in optimizer.param_groups]}")
                    if (global_step%train_steps_per_epoch in checkpoints) \
                            and ((current_epoch+1)%self.t_config.ckpt_epoch_frequency==0 or current_epoch+1==num_epochs):
                        self.save_and_callback(global_step, step, current_epoch, callback)

            logger.info(f"Epoch {current_epoch+1} finished")

    def train_on_batch(self, batch, args):
        if self.d_config.is_caching_logits is False:
            if type(batch) is dict:
                for k,v in batch.items():
                    if type(v) is torch.Tensor:
                        batch[k] = v.to(self.t_config.device)
                with torch.no_grad():
                    results_T = self.model_T(**batch, **args)
                results_S = self.model_S(**batch, **args)
            else:
                batch = tuple(item.to(self.t_config.device) if type(item) is torch.Tensor else item for item in batch)
                with torch.no_grad():
                    results_T = self.model_T(*batch, **args)
                results_S = self.model_S(*batch, **args)
            results_T = post_adaptor(self.adaptor_T(batch,results_T))
            results_S = post_adaptor(self.adaptor_S(batch,results_S))

        else:
            batch, cached_logits = batch
            if type(batch) is dict:
                new_batch = {}
                for k,v in batch.items():
                    if type(v) is torch.Tensor:
                        new_batch[k] = v.to(self.t_config.device)
                    else:
                        new_batch[k] = v
                batch = new_batch
                results_S = self.model_S(**batch, **args)
            else:
                batch = tuple(item.to(self.t_config.device) if type(item) is torch.Tensor else item for item in batch)
                results_S = self.model_S(*batch, **args)
            results_S = post_adaptor(self.adaptor_S(batch,results_S))

            results_T = {'logits':[logits.to(self.t_config.device) for logits in cached_logits]}
            if 'logits_mask' in results_S:
                results_T['logits_mask'] = results_S['logits_mask']
    
        logits_list_T = results_T['logits']  # list of tensor
        logits_list_S = results_S['logits']  # list of tensor
        total_loss  = 0

        if 'logits_mask' in results_S:
            masks_list_S = results_S['logits_mask']
            logits_list_S = select_logits_with_mask(logits_list_S,masks_list_S)  #(mask_sum, num_of_class)
        if 'logits_mask' in results_T:
            masks_list_T = results_T['logits_mask']
            logits_list_T = select_logits_with_mask(logits_list_T,masks_list_T)  #(mask_sum, num_of_class)

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

    def cache_logits(self, batch, args, batch_postprocessor):
            if batch_postprocessor is not None:
                batch = batch_postprocessor(batch)

            if type(batch) is dict:
                new_batch = {}
                for k,v in batch.items():
                    if type(v) is torch.Tensor:
                        new_batch[k] = v.to(self.t_config.device)
                    else:
                        new_batch[k] = v
                with torch.no_grad():
                    results_T = self.model_T(**new_batch, **args)
            else:
                new_batch = tuple(item.to(self.t_config.device) if type(item) is torch.Tensor else item for item in batch)
                with torch.no_grad():
                    results_T = self.model_T(*new_batch, **args)
            results_T = post_adaptor(self.adaptor_T(batch,results_T))

            self.logits_cache.append([batch, [logits.to('cpu') for logits in results_T['logits']]])