from .distiller_utils import *

class BasicTrainer:
    """
    It performs supervised training, not distillation. It can be used for training the teacher model.

    Args:
        train_config (:class:`TrainingConfig`): training configuration.
        model (:class:`torch.nn.Module`): model to be trained.
        adaptor (Callable)：adaptor of the model.
    
    The role of `adaptor` is explained in :py:func:`adaptor`.
    """

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

    def train(self, optimizer, dataloader, num_epochs, scheduler_class=None, scheduler_args=None, scheduler=None, max_grad_norm = -1.0, num_steps=None, callback=None, batch_postprocessor=None, **args):
        """
        trains the model. See :meth:`BasicDistiller.train`.
        """
        # update scheduler
        if scheduler_class is not None:
            # overwrite scheduler
            scheduler = scheduler_class(**{'optimizer':optimizer},**scheduler_args)

        if self.t_config.fp16:
            if not has_apex:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.t_config.fp16_opt_level)

        #dataparallel multi-gpu training
        if self.t_config.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

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
                if self.t_config.fp16:
                    with amp.scale_loss(total_loss,optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    total_loss.backward()

                scalar_total_loss = total_loss.cpu().item() * self.t_config.gradient_accumulation_steps
                self.tb_writer.add_scalar('scalar/total_loss', scalar_total_loss, writer_step)
                writer_step += 1

                if (step+1)%self.t_config.gradient_accumulation_steps == 0:
                    if max_grad_norm > 0:
                        if self.t_config.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if (global_step) % print_every == 0:
                        logger.info(f"Global step: {global_step}, epoch step:{step+1}")
                    if (global_step%ckpt_steps==0) or global_step==total_global_steps:
                        logger.info(f"Saving at global step {global_step}")
                        coreModel = self.model.module if hasattr(self.model, "module") else self.model
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
            optimizer.zero_grad()
            logger.info(f"Length of current epoch in forward batch: {len(dataloader)}")
            for step, batch in tqdm(enumerate(dataloader),disable=None):
                if batch_postprocessor is not None:
                    batch = batch_postprocessor(batch)
                total_loss = self.train_on_batch(batch,args)
                total_loss /= self.t_config.gradient_accumulation_steps
                if self.t_config.fp16:
                    with amp.scale_loss(total_loss,optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    total_loss.backward()

                scalar_total_loss = total_loss.cpu().item() * self.t_config.gradient_accumulation_steps
                self.tb_writer.add_scalar('scalar/total_loss', scalar_total_loss, writer_step)
                writer_step += 1

                if (step+1)%self.t_config.gradient_accumulation_steps == 0:
                    if max_grad_norm > 0:
                        if self.t_config.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm) 
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if (global_step) % print_every == 0:
                        logger.info(f"Global step: {global_step}, epoch step:{step+1}")
                    if (global_step%train_steps_per_epoch in checkpoints) \
                            and ((current_epoch+1)%self.t_config.ckpt_epoch_frequency==0 or current_epoch+1==num_epochs):
                        logger.info(f"Saving at global step {global_step}, epoch step {step+1} epoch {current_epoch+1}")
                        coreModel = self.model.module if hasattr(self.model, "module") else self.model
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