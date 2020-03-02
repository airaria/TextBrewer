import logging
logger = logging.getLogger("Main")
logger.setLevel(logging.INFO)
handler_stream = logging.StreamHandler()
handler_stream.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -  %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
handler_stream.setFormatter(formatter)
logger.addHandler(handler_stream)

import os,random
import numpy as np
import torch
from utils_glue import output_modes, processors
from pytorch_pretrained_bert.my_modeling import BertConfig
from pytorch_pretrained_bert import BertTokenizer
from optimization import BERTAdam
import config
from utils import divide_parameters, load_and_cache_examples
from modeling import BertForGLUESimple,BertForGLUESimpleAdaptor

from textbrewer import DistillationConfig, TrainingConfig, MultiTeacherDistiller
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from tqdm import tqdm
from utils_glue import compute_metrics
from functools import partial


def args_check(args):
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        logger.warning("Output directory () already exists and is not empty.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))
    args.n_gpu = n_gpu
    args.device = device
    return device, n_gpu

def predict(model,eval_datasets,step,args):
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_output_dir = args.output_dir
    results = {}
    for eval_task,eval_dataset in zip(eval_task_names, eval_datasets):
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        logger.info("Predicting...")
        logger.info("***** Running predictions *****")
        logger.info(" task name = %s", eval_task)
        logger.info("  Num  examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.predict_batch_size)
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.predict_batch_size)
        model.eval()

        pred_logits = []
        label_ids = []
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=None):
            input_ids, input_mask, segment_ids, labels = batch
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            segment_ids = segment_ids.to(args.device)
            with torch.no_grad():
                logits = model(input_ids, input_mask, segment_ids)
            cpu_logits = logits.detach().cpu()
            for i in range(len(cpu_logits)):
                pred_logits.append(cpu_logits[i].numpy())
                label_ids.append(labels[i])

        pred_logits = np.array(pred_logits)
        label_ids   = np.array(label_ids)

        if args.output_mode == "classification":
            preds = np.argmax(pred_logits, axis=1)
        else: # args.output_mode == "regression":
            preds = np.squeeze(pred_logits)
        result = compute_metrics(eval_task, preds, label_ids)
        logger.info(f"task:,{eval_task}")
        logger.info(f"result: {result}")
        results.update(result)

    output_eval_file = os.path.join(eval_output_dir, "eval_results-%s.txt" % eval_task)
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {} task {} *****".format(step, eval_task))
        writer.write("step: %d ****\n " % step)
        for key in sorted(results.keys()):
            logger.info("%s = %s", key, str(results[key]))
            writer.write("%s = %s\n" % (key, str(results[key])))
    model.train()
    return results

def main():
    #parse arguments
    config.parse()
    args = config.args
    for k,v in vars(args).items():
        logger.info(f"{k}:{v}")
    #set seeds
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    #arguments check
    device, n_gpu = args_check(args)
    os.makedirs(args.output_dir, exist_ok=True)
    forward_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    args.forward_batch_size = forward_batch_size

    #load bert config
    bert_config_T = BertConfig.from_json_file(args.bert_config_file_T)
    bert_config_S = BertConfig.from_json_file(args.bert_config_file_S)
    assert args.max_seq_length <= bert_config_T.max_position_embeddings
    assert args.max_seq_length <= bert_config_S.max_position_embeddings

    #Prepare GLUE task
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    #read data
    train_dataset = None
    eval_datasets  = None
    num_train_steps = None
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        if args.aux_task_name:
            aux_train_dataset = load_and_cache_examples(args, args.aux_task_name, tokenizer, evaluate=False, is_aux=True)
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, aux_train_dataset])
        num_train_steps = int(len(train_dataset)/args.train_batch_size) * args.num_train_epochs
    if args.do_predict:
        eval_datasets = []
        eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
        for eval_task in eval_task_names:
            eval_datasets.append(load_and_cache_examples(args, eval_task, tokenizer, evaluate=True))
    logger.info("Data loaded")


    #Build Model and load checkpoint

    model_S = BertForGLUESimple(bert_config_S, num_labels=num_labels,args=args)
    #Load teacher
    if args.tuned_checkpoint_Ts:
        model_Ts = [BertForGLUESimple(bert_config_T, num_labels=num_labels,args=args) for i in range(len(args.tuned_checkpoint_Ts))]
        for model_T, ckpt_T in zip(model_Ts,args.tuned_checkpoint_Ts):
            logger.info("Load state dict %s" % ckpt_T)
            state_dict_T = torch.load(ckpt_T, map_location='cpu')
            model_T.load_state_dict(state_dict_T)
            model_T.eval()
    else:
        assert args.do_predict is True
    #Load student
    if args.load_model_type=='bert':
        assert args.init_checkpoint_S is not None
        state_dict_S = torch.load(args.init_checkpoint_S, map_location='cpu')
        state_weight = {k[5:]:v for k,v in state_dict_S.items() if k.startswith('bert.')}
        missing_keys,_ = model_S.bert.load_state_dict(state_weight,strict=False)
        assert len(missing_keys)==0
    elif args.load_model_type=='all':
        assert args.tuned_checkpoint_S is not None
        state_dict_S = torch.load(args.tuned_checkpoint_S,map_location='cpu')
        model_S.load_state_dict(state_dict_S)
    else:
        logger.info("Model is randomly initialized.")
    if args.do_train:
        for model_T in model_Ts:
            model_T.to(device)
    model_S.to(device)

    if args.local_rank != -1 or n_gpu > 1:
        if args.local_rank != -1:
            raise NotImplementedError
        elif n_gpu > 1:
            if args.do_train:
                model_Ts = [torch.nn.DataParallel(model_T) for model_T in model_Ts]
            model_S = torch.nn.DataParallel(model_S) #,output_device=n_gpu-1)

    if args.do_train:
        #parameters
        params = list(model_S.named_parameters())
        all_trainable_params = divide_parameters(params, lr=args.learning_rate)
        logger.info("Length of all_trainable_params: %d", len(all_trainable_params))

        optimizer = BERTAdam(all_trainable_params,lr=args.learning_rate,
                             warmup=args.warmup_proportion,t_total=num_train_steps,schedule=args.schedule,
                             s_opt1=args.s_opt1, s_opt2=args.s_opt2, s_opt3=args.s_opt3)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Forward batch size = %d", forward_batch_size)
        logger.info("  Num backward steps = %d", num_train_steps)

        ########### DISTILLATION ###########
        train_config = TrainingConfig(
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            ckpt_frequency = args.ckpt_frequency,
            log_dir = args.output_dir,
            output_dir = args.output_dir,
            device = args.device)

        distill_config = DistillationConfig(
            temperature = args.temperature,
            kd_loss_type = 'ce')

        logger.info(f"{train_config}")
        logger.info(f"{distill_config}")
        adaptor = partial(BertForGLUESimpleAdaptor, no_logits=False, no_mask = False)


        distiller = MultiTeacherDistiller(train_config = train_config,
                            distill_config = distill_config,
                            model_T = model_Ts, model_S = model_S,
                            adaptor_T=adaptor,
                            adaptor_S=adaptor)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            raise NotImplementedError
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.forward_batch_size,drop_last=True)
        callback_func = partial(predict, eval_datasets=eval_datasets, args=args)
        with distiller:
            distiller.train(optimizer, scheduler=None, dataloader=train_dataloader,
                              num_epochs=args.num_train_epochs, callback=callback_func)

    if not args.do_train and args.do_predict:
        res = predict(model_S,eval_datasets,step=0,args=args)
        print (res)

if __name__ == "__main__":
    main()
