import collections

from .losses import *
from .schedulers import *
from .utils import cycle
from .projections import linear_projection, projection_with_activation

class DynamicKeyDict:
    def __init__(self, kv_dict):
        self.store = kv_dict
    def __getitem__(self, key):
        if not isinstance(key,(list,tuple)):
            return self.store[key]
        else:
            name = key[0]
            args = key[1:]
            return self.store[name](*args)
    def __setitem__(self, key, value):
        self.store[key] = value
    def __contains__(self, key):
        if isinstance(key, (list,tuple)):
            return key[0] in self.store
        else:
            return key in self.store

TEMPERATURE_SCHEDULER=DynamicKeyDict(
    {'constant': constant_temperature_scheduler,
     'flsw': flsw_temperature_scheduler_builder,
     'cwsm':cwsm_temperature_scheduler_builder})



FEATURES = ['hidden','attention']

ADAPTOR_KEYS = ['logits','logits_mask','losses','inputs_mask','labels'] + FEATURES

KD_LOSS_MAP = {'mse': kd_mse_loss,
                'ce': kd_ce_loss}

MATCH_LOSS_MAP = {'attention_mse_sum': att_mse_sum_loss,
                  'attention_mse': att_mse_loss,
                  'attention_ce_mean': att_ce_mean_loss,
                  'attention_ce': att_ce_loss,
                  'hidden_mse'    : hid_mse_loss,
                  'cos'  : cos_loss,
                  'pkd'  : pkd_loss,
                  'gram' : fsp_loss,
                  'fsp'  : fsp_loss,
                  'mmd'  : mmd_loss,
                  'nst'  : mmd_loss}

PROJ_MAP = {'linear': linear_projection,
            'relu'  : projection_with_activation('ReLU'),
            'tanh'  : projection_with_activation('Tanh')
            }

WEIGHT_SCHEDULER = {'linear_decay': linear_decay_weight_scheduler,
                    'linear_growth' : linear_growth_weight_scheduler}

#TEMPERATURE_SCHEDULER = {'constant': constant_temperature_scheduler,
#                         'flsw_scheduler': flsw_temperature_scheduler_builder(1,1)}


MAPS = {'kd_loss': KD_LOSS_MAP,
        'match_Loss': MATCH_LOSS_MAP,
        'projection': PROJ_MAP,
        'weight_scheduler': WEIGHT_SCHEDULER,
        'temperature_scheduler': TEMPERATURE_SCHEDULER}


def register_new(map_name, name, func):
    assert map_name in MAPS
    assert callable(func), "Functions to be registered is not callable"
    MAPS[map_name][name] = func


'''
Add new loss:
def my_L1_loss(feature_S, feature_T, mask=None):
    return (feature_S-feature_T).abs().mean()

MATCH_LOSS_MAP['my_L1_loss'] = my_L1_loss
'''
