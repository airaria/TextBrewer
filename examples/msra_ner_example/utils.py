import pickle
import os
import config
import logging
logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)

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
