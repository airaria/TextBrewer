import logging


import os,random
import numpy as np
import torch
from utils_ner import read_features, label2id_dict
from utils import divide_parameters
from transformers import ElectraConfig, AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, BertTokenizer
import config
from modeling import ElectraForTokenClassification, ElectraForTokenClassificationAdaptor
from textbrewer import DistillationConfig, TrainingConfig, GeneralDistiller
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from functools import partial

from train_eval import predict, ddp_predict

def args_check(logger, args):
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
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
        logger.info("rank %d device %s n_gpu %d distributed training %r", torch.distributed.get_rank(), device, n_gpu, bool(args.local_rank != -1))
    args.n_gpu = n_gpu
    args.device = device
    return device, n_gpu

def main():
    #parse arguments
    config.parse()
    args = config.args

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN
        )
    logger = logging.getLogger("Main")
    #arguments check
    device, n_gpu = args_check(logger, args)
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.local_rank != -1:
        logger.warning(f"Process rank: {torch.distributed.get_rank()}, device : {args.device}, n_gpu : {args.n_gpu}, distributed training : {bool(args.local_rank!=-1)}")

    for k,v in vars(args).items():
        logger.info(f"{k}:{v}")
    #set seeds
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)


    forward_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    args.forward_batch_size = forward_batch_size

    #load bert config
    bert_config_T = ElectraConfig.from_json_file(args.bert_config_file_T)
    bert_config_S = ElectraConfig.from_json_file(args.bert_config_file_S)
    bert_config_S.output_hidden_states = (args.output_encoded_layers=='true')
    bert_config_T.output_hidden_states = (args.output_encoded_layers=='true')
    bert_config_S.num_labels = len(label2id_dict)
    bert_config_T.num_labels = len(label2id_dict)

    assert args.max_seq_length <= bert_config_T.max_position_embeddings
    assert args.max_seq_length <= bert_config_S.max_position_embeddings

    #read data
    train_examples = None
    train_dataset = None
    eval_examples = None
    eval_dataset = None
    num_train_steps = None

    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    if args.do_train:
        train_examples,train_dataset = read_features(args.train_file,max_seq_length=args.max_seq_length)
    if args.do_predict:
        eval_examples,eval_dataset = read_features(args.predict_file,max_seq_length=args.max_seq_length)

    if args.local_rank == 0:
        torch.distributed.barrier()
    #Build Model and load checkpoint
    model_T = ElectraForTokenClassification(bert_config_T)
    model_S = ElectraForTokenClassification(bert_config_S)
    #Load teacher
    if args.tuned_checkpoint_T is not None:
        state_dict_T = torch.load(args.tuned_checkpoint_T, map_location='cpu')
        model_T.load_state_dict(state_dict_T)
        model_T.eval()
    else:
        assert args.do_predict is True
    #Load student
    if args.load_model_type=='bert':
        assert args.init_checkpoint_S is not None
        state_dict_S = torch.load(args.init_checkpoint_S, map_location='cpu')
        missing_keys, unexpected_keys = model_S.load_state_dict(state_dict_S,strict=False)
        logger.info(f"missing keys:{missing_keys}")
        logger.info(f"unexpected keys:{unexpected_keys}")
    elif args.load_model_type=='all':
        assert args.tuned_checkpoint_S is not None
        state_dict_S = torch.load(args.tuned_checkpoint_S,map_location='cpu')
        model_S.load_state_dict(state_dict_S)
    else:
        logger.info("Model is randomly initialized.")
    model_T.to(device)
    model_S.to(device)


    if args.do_train:
        #parameters
        if args.lr_decay is not None:
            outputs_params = list(model_S.classifier.named_parameters())
            outputs_params = divide_parameters(outputs_params, lr = args.learning_rate)

            electra_params = []
            n_layers = len(model_S.electra.encoder.layer)
            assert n_layers==12
            for i,n in enumerate(reversed(range(n_layers))):
                encoder_params = list(model_S.electra.encoder.layer[n].named_parameters())
                lr = args.learning_rate * args.lr_decay**(i+1)
                electra_params += divide_parameters(encoder_params, lr = lr)
                logger.info(f"{i},{n},{lr}")
            embed_params = [(name,value) for name,value in model_S.electra.named_parameters() if 'embedding' in name]
            logger.info(f"{[name for name,value in embed_params]}")
            lr = args.learning_rate * args.lr_decay**(n_layers+1)
            electra_params += divide_parameters( embed_params, lr = lr)
            logger.info(f"embed lr:{lr}")
            all_trainable_params = outputs_params + electra_params
            assert sum(map(lambda x:len(x['params']), all_trainable_params))==len(list(model_S.parameters())),\
                (sum(map(lambda x:len(x['params']), all_trainable_params)), len(list(model_S.parameters())))
        else:
            params = list(model_S.named_parameters())
            all_trainable_params = divide_parameters(params, lr=args.learning_rate)
        logger.info("Length of all_trainable_params: %d", len(all_trainable_params))


        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.forward_batch_size,drop_last=True)

        num_train_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        optimizer =   AdamW(all_trainable_params, lr=args.learning_rate, correct_bias = False)
        if args.official_schedule == 'const':
            scheduler_class = get_constant_schedule_with_warmup
            scheduler_args = {'num_warmup_steps':int(args.warmup_proportion*num_train_steps)}
            #scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion*num_train_steps))
        elif args.official_schedule == 'linear':
            scheduler_class = get_linear_schedule_with_warmup
            scheduler_args = {'num_warmup_steps':int(args.warmup_proportion*num_train_steps), 'num_training_steps': num_train_steps}
            #scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(args.warmup_proportion*num_train_steps), num_training_steps = num_train_steps)
        else:
            raise NotImplementedError

        logger.warning("***** Running training *****")
        logger.warning("local_rank %d Num orig examples = %d",  args.local_rank, len(train_examples))
        logger.warning("local_rank %d Num split examples = %d", args.local_rank, len(train_dataset))
        logger.warning("local_rank %d Forward batch size = %d", args.local_rank, forward_batch_size)
        logger.warning("local_rank %d Num backward steps = %d", args.local_rank, num_train_steps)

        ########### DISTILLATION ###########
        train_config = TrainingConfig(
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            ckpt_frequency = args.ckpt_frequency,
            log_dir = args.output_dir,
            output_dir = args.output_dir,
            device = args.device,
            fp16=args.fp16,
            local_rank = args.local_rank)
        logger.info(f"{train_config}")

        from matches import matches
        intermediate_matches = None
        if isinstance(args.matches,(list,tuple)):
            intermediate_matches = []
            for match in args.matches:
                intermediate_matches += matches[match]
        logger.info(f"{intermediate_matches}")
        distill_config = DistillationConfig(
            temperature = args.temperature,
            intermediate_matches=intermediate_matches)

        adaptor_T = ElectraForTokenClassificationAdaptor
        adaptor_S = ElectraForTokenClassificationAdaptor

        distiller = GeneralDistiller(train_config = train_config,
                                   distill_config = distill_config,
                                   model_T = model_T, model_S = model_S,
                                   adaptor_T = adaptor_T,
                                   adaptor_S = adaptor_S)

        # evluate the model in a single process in ddp_predict
        callback_func = partial(ddp_predict, 
                eval_examples=eval_examples,
                eval_dataset=eval_dataset,
                args=args)
        with distiller:
            distiller.train(optimizer, scheduler_class=scheduler_class, 
                              scheduler_args=scheduler_args,
                              max_grad_norm = 1.0,
                              dataloader=train_dataloader,
                              num_epochs=args.num_train_epochs, callback=callback_func)

    if not args.do_train and args.do_predict:
        res = ddp_predict(model_S,eval_examples,eval_dataset,step=0,args=args)
        print (res)




if __name__ == "__main__":
    main()
