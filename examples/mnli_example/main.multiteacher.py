import logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    )
logger = logging.getLogger("Main")

import os,random
import numpy as np
import torch
from utils_glue import output_modes, processors
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup, BertTokenizer
import config
from utils import divide_parameters, load_and_cache_examples
from modeling import BertForGLUESimple, BertForGLUESimpleAdaptorTrain, BertForGLUESimpleAdaptor
from textbrewer import DistillationConfig, TrainingConfig, MultiTeacherDistiller
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from tqdm import tqdm
from utils_glue import compute_metrics
from functools import partial
import re
from predict_function import predict
from parse import parse_model_config, MODEL_CLASSES

def args_check(args):
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        logger.warning("Output directory () already exists and is not empty.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.local_rank == -1 or args.no_cuda:
        if not args.no_cuda and not torch.cuda.is_available():
            raise ValueError("No CUDA available!")
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

    #load config
    teachers_and_student = parse_model_config(args.model_config_json)

    #Prepare GLUE task
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    #read data
    train_dataset = None
    eval_datasets  = None
    num_train_steps = None

    tokenizer_S = teachers_and_student['student']['tokenizer']
    prefix_S = teachers_and_student['student']['prefix']

    if args.do_train:
        train_dataset = load_and_cache_examples(
        args, args.task_name,tokenizer_S, prefix=prefix_S,evaluate=False)
    if args.do_predict:
        eval_datasets = []
        eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
        for eval_task in eval_task_names:
            eval_datasets.append(load_and_cache_examples(args, eval_task, tokenizer_S, prefix=prefix_S, evaluate=True))
    logger.info("Data loaded")


    #Build Model and load checkpoint
    if args.do_train:
        model_Ts = []
        for teacher in teachers_and_student['teachers']:
            model_type_T = teacher['model_type']
            model_config_T = teacher['config']
            checkpoint_T = teacher['checkpoint']

            _,_,model_class_T = MODEL_CLASSES[model_type_T]
            model_T = model_class_T(model_config_T, num_labels=num_labels)
            state_dict_T = torch.load(checkpoint_T,map_location='cpu')
            missing_keys, un_keys = model_T.load_state_dict(state_dict_T,strict=True)
            logger.info(f"Teacher Model {model_type_T} loaded")
            model_T.to(device)
            model_T.eval()
            model_Ts.append(model_T)

    student = teachers_and_student['student']
    model_type_S = student['model_type']
    model_config_S = student['config']
    checkpoint_S = student['checkpoint']
    _,_,model_class_S = MODEL_CLASSES[model_type_S]
    model_S = model_class_S(model_config_S, num_labels=num_labels)
    if checkpoint_S is not None:
        state_dict_S = torch.load(checkpoint_S, map_location='cpu')
        missing_keys, un_keys = model_S.load_state_dict(state_dict_S,strict=False)
        logger.info(f"missing keys:{missing_keys}")
        logger.info(f"unexpected keys:{un_keys}")
    else:
        logger.warning("Initializing student randomly")
    logger.info("Student Model loaded")
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

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            raise NotImplementedError
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.forward_batch_size,drop_last=True)
        num_train_steps = int(len(train_dataloader)//args.gradient_accumulation_steps * args.num_train_epochs)

        ########## DISTILLATION ###########
        train_config = TrainingConfig(
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            ckpt_frequency = args.ckpt_frequency,
            log_dir = args.output_dir,
            output_dir = args.output_dir,
            fp16 = args.fp16,
            device = args.device)

        distill_config = DistillationConfig(
            temperature=args.temperature,
            kd_loss_type = 'ce')

        logger.info(f"{train_config}")
        logger.info(f"{distill_config}")
        adaptor_T = BertForGLUESimpleAdaptor
        adaptor_S = BertForGLUESimpleAdaptor

        distiller = MultiTeacherDistiller(train_config = train_config,
                                   distill_config = distill_config,
                                   model_T = model_Ts, model_S = model_S,
                                   adaptor_T = adaptor_T,
                                   adaptor_S = adaptor_S)

        optimizer = AdamW(all_trainable_params,lr=args.learning_rate)
        scheduler_class = get_linear_schedule_with_warmup
        scheduler_args = {'num_warmup_steps': int(args.warmup_proportion*num_train_steps), 
                          'num_training_steps': num_train_steps}

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Forward batch size = %d", forward_batch_size)
        logger.info("  Num backward steps = %d", num_train_steps)


        callback_func = partial(predict, eval_datasets=eval_datasets, args=args)
        with distiller:
            distiller.train(optimizer, scheduler_class=scheduler_class, scheduler_args=scheduler_args, dataloader = train_dataloader,
                              num_epochs = args.num_train_epochs, callback=callback_func,max_grad_norm=1)

    if not args.do_train and args.do_predict:
        res = predict(model_S,eval_datasets,step=0,args=args)
        print (res)

if __name__ == "__main__":
    main()
