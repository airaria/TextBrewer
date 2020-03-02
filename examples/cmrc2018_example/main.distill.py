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
from processing import convert_examples_to_features, read_squad_examples
from processing import ChineseFullTokenizer
from pytorch_pretrained_bert.my_modeling import BertConfig
from optimization import BERTAdam
import config
from utils import read_and_convert, divide_parameters
from modeling import BertForQASimple, BertForQASimpleAdaptor
from textbrewer import DistillationConfig, TrainingConfig, GeneralDistiller
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from functools import partial

from train_eval import predict

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

    #read data
    train_examples = None
    train_features = None
    eval_examples = None
    eval_features = None
    num_train_steps = None

    tokenizer = ChineseFullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    convert_fn = partial(convert_examples_to_features,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length,
                         doc_stride=args.doc_stride,
                         max_query_length=args.max_query_length)
    if args.do_train:
        train_examples,train_features = read_and_convert(args.train_file,is_training=True, do_lower_case=args.do_lower_case,
                                                         read_fn=read_squad_examples,convert_fn=convert_fn)
        if args.fake_file_1:
            fake_examples1,fake_features1 = read_and_convert(args.fake_file_1,is_training=True, do_lower_case=args.do_lower_case,
                                                             read_fn=read_squad_examples,convert_fn=convert_fn)
            train_examples += fake_examples1
            train_features += fake_features1
        if args.fake_file_2:
            fake_examples2, fake_features2 = read_and_convert(args.fake_file_2,is_training=True, do_lower_case=args.do_lower_case,
                                                              read_fn=read_squad_examples,convert_fn=convert_fn)
            train_examples += fake_examples2
            train_features += fake_features2

        num_train_steps = int(len(train_features)/args.train_batch_size) * args.num_train_epochs

    if args.do_predict:
        eval_examples,eval_features = read_and_convert(args.predict_file,is_training=False, do_lower_case=args.do_lower_case,
                                                         read_fn=read_squad_examples,convert_fn=convert_fn)

    #Build Model and load checkpoint
    model_T = BertForQASimple(bert_config_T,args)
    model_S = BertForQASimple(bert_config_S,args)
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
        state_weight = {k[5:]:v for k,v in state_dict_S.items() if k.startswith('bert.')}
        missing_keys,_ = model_S.bert.load_state_dict(state_weight,strict=False)
        assert len(missing_keys)==0
    elif args.load_model_type=='all':
        assert args.tuned_checkpoint_S is not None
        state_dict_S = torch.load(args.tuned_checkpoint_S,map_location='cpu')
        model_S.load_state_dict(state_dict_S)
    else:
        logger.info("Model is randomly initialized.")
    model_T.to(device)
    model_S.to(device)

    if args.local_rank != -1 or n_gpu > 1:
        if args.local_rank != -1:
            raise NotImplementedError
        elif n_gpu > 1:
            model_T = torch.nn.DataParallel(model_T) #,output_device=n_gpu-1)
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
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Forward batch size = %d", forward_batch_size)
        logger.info("  Num backward steps = %d", num_train_steps)

        ########### DISTILLATION ###########
        train_config = TrainingConfig(
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            ckpt_frequency = args.ckpt_frequency,
            log_dir = args.output_dir,
            output_dir = args.output_dir,
            device = args.device)

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

        adaptor_T = partial(BertForQASimpleAdaptor, no_logits=args.no_logits, no_mask = args.no_inputs_mask)
        adaptor_S = partial(BertForQASimpleAdaptor, no_logits=args.no_logits, no_mask = args.no_inputs_mask)

        distiller = GeneralDistiller(train_config = train_config,
                                   distill_config = distill_config,
                                   model_T = model_T, model_S = model_S,
                                   adaptor_T = adaptor_T,
                                   adaptor_S = adaptor_S)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_doc_mask = torch.tensor([f.doc_mask for f in train_features], dtype=torch.float)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)

        train_dataset = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_doc_mask,
                                   all_start_positions, all_end_positions)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            raise NotImplementedError
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.forward_batch_size,drop_last=True)
        callback_func = partial(predict, 
                eval_examples=eval_examples,
                eval_features=eval_features,
                args=args)
        with distiller:
            distiller.train(optimizer, scheduler=None, dataloader=train_dataloader,
                              num_epochs=args.num_train_epochs, callback=callback_func)

    if not args.do_train and args.do_predict:
        res = predict(model_S,eval_examples,eval_features,step=0,args=args)
        print (res)




if __name__ == "__main__":
    main()
