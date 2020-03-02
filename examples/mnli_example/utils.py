import pickle
import os
import config
import logging
logger = logging.getLogger(__name__)
import torch
from torch.utils.data import TensorDataset
from utils_glue import processors, output_modes, convert_examples_to_features

def read_and_convert(fn,is_training,read_fn,convert_fn):
    data_dirname, data_basename = os.path.split(fn)
    cased = '' if config.args.do_lower_case else 'cased'
    if config.args.max_seq_length != 416:
        data_pklname = data_basename + '%s%d_l%d_cHA.pkl' % (cased,config.args.doc_stride,config.args.max_seq_length)
    else:
        data_pklname = data_basename + '%s%d_cHA.pkl' % (cased,config.args.doc_stride)
    full_pklname = os.path.join(data_dirname,data_pklname)
    if os.path.exists(full_pklname):
        print("Loading dataset %s " % data_pklname)
        with open(full_pklname,'rb') as f:
            examples,features = pickle.load(f)
    else:
        print ("Building dataset %s " % data_pklname)
        examples = read_fn(input_file=fn,is_training=is_training)
        print ("Size: ",len(examples))
        features = convert_fn(examples=examples,is_training=is_training)
        try:
            with open(full_pklname,'wb') as f:
                pickle.dump((examples,features),f)
        except:
            print ("Can't save train data file.")
    return examples,features

def load_and_cache_examples(args, task, tokenizer, evaluate=False, is_aux=False):
    if is_aux:
        data_dir = args.aux_data_dir
    else:
        data_dir = args.data_dir
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(data_dir, 'cached_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(data_dir) if evaluate else processor.get_train_examples(data_dir)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                                cls_token_segment_id=0,pad_token_segment_id=0)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def divide_parameters(named_parameters,lr=None):
    no_decay = ['bias', 'LayerNorm.bias','LayerNorm.weight']
    decay_parameters_names = list(zip(*[(p,n) for n,p in named_parameters if not any((di in n) for di in no_decay)]))
    no_decay_parameters_names = list(zip(*[(p,n) for n,p in named_parameters if any((di in n) for di in no_decay)]))
    param_group = []
    if len(decay_parameters_names)>0:
        decay_parameters, decay_names = decay_parameters_names
        #print ("decay:",decay_names)
        if lr is not None:
            decay_group = {'params':decay_parameters,   'weight_decay_rate': config.args.weight_decay_rate, 'lr':lr}
        else:
            decay_group = {'params': decay_parameters, 'weight_decay_rate': config.args.weight_decay_rate}
        param_group.append(decay_group)

    if len(no_decay_parameters_names)>0:
        no_decay_parameters, no_decay_names = no_decay_parameters_names
        #print ("no decay:", no_decay_names)
        if lr is not None:
            no_decay_group = {'params': no_decay_parameters, 'weight_decay_rate': 0.0, 'lr': lr}
        else:
            no_decay_group = {'params': no_decay_parameters, 'weight_decay_rate': 0.0}
        param_group.append(no_decay_group)

    assert len(param_group)>0
    return param_group
