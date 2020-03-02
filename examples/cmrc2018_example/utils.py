import pickle
import os
import config
import logging
logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)

def read_and_convert(fn,is_training,read_fn,convert_fn,do_lower_case):
    data_dirname, data_basename = os.path.split(fn)
    cased = '' if do_lower_case else 'cased'
    if config.args.max_seq_length != 416:
        data_pklname = data_basename + '%s%d_l%d_cHA.t%s.pkl' % (cased,config.args.doc_stride,config.args.max_seq_length,config.args.tag)
    else:
        data_pklname = data_basename + '%s%d_cHA.t%s.pkl' % (cased,config.args.doc_stride,config.args.tag)
    full_pklname = os.path.join(data_dirname,data_pklname)
    if os.path.exists(full_pklname):
        logger.info("Loading dataset %s " % data_pklname)
        with open(full_pklname,'rb') as f:
            examples,features = pickle.load(f)
    else:
        logger.info("Building dataset %s " % data_pklname)
        examples = read_fn(input_file=fn,is_training=is_training,do_lower_case=do_lower_case)
        logger.info(f"Size: {len(examples)}")
        features = convert_fn(examples=examples,is_training=is_training)
        try:
            with open(full_pklname,'wb') as f:
                pickle.dump((examples,features),f)
        except:
            logger.info("Can't save train data file.")
    return examples,features


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
