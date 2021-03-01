import pickle
import os
import config
import logging
logger = logging.getLogger(__name__)
import torch
from torch.utils.data import TensorDataset
from utils_glue import processors, output_modes, convert_examples_to_features, trans3_convert_examples_to_features

def load_and_cache_examples(args, task, tokenizer, prefix='bert', evaluate=False, return_dataset=True):
    data_dir = args.data_dir
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    #seg_ids = 1 if 'bert' in prefix else 0
    if False: #args.do_test:
        pass #stage = 'test'
    else:
        if evaluate:
            stage = 'dev' if args.do_test is False else 'test'
        else:
            stage = 'train'
    cached_features_file = os.path.join(data_dir, '{}_{}_{}_{}'.format(
        prefix,
        stage, #'dev' if evaluate else 'train',
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        if evaluate:
            if args.do_test:
                examples = processor.get_test_examples(data_dir)
            else:
                examples = processor.get_dev_examples(data_dir)
        else:
            examples = processor.get_train_examples(data_dir)
        features = trans3_convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    # Convert to Tensors and build dataset
    if True:
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label for f in features], dtype=torch.float)
    if return_dataset:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)
        return dataset
    else:
        return all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids


def divide_parameters(named_parameters,lr=None):
    no_decay = ['bias', 'LayerNorm.bias','LayerNorm.weight']
    decay_parameters_names = list(zip(*[(p,n) for n,p in named_parameters if not any((di in n) for di in no_decay)]))
    no_decay_parameters_names = list(zip(*[(p,n) for n,p in named_parameters if any((di in n) for di in no_decay)]))
    param_group = []
    if len(decay_parameters_names)>0:
        decay_parameters, decay_names = decay_parameters_names
        #print ("decay:",decay_names)
        if lr is not None:
            decay_group = {'params':decay_parameters,   'weight_decay': config.args.weight_decay_rate, 'lr':lr}
        else:
            decay_group = {'params': decay_parameters, 'weight_decay': config.args.weight_decay_rate}
        param_group.append(decay_group)

    if len(no_decay_parameters_names)>0:
        no_decay_parameters, no_decay_names = no_decay_parameters_names
        #print ("no decay:", no_decay_names)
        if lr is not None:
            no_decay_group = {'params': no_decay_parameters, 'weight_decay': 0.0, 'lr': lr}
        else:
            no_decay_group = {'params': no_decay_parameters, 'weight_decay': 0.0}
        param_group.append(no_decay_group)

    assert len(param_group)>0
    return param_group
