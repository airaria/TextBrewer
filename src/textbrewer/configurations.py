import json
import os
from typing import Union, List, Optional, Dict
from .presets import *

class Config:
    def __init__(self,**kwargs):
        pass

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file,'r') as f:
            json_data = json.load(f)
        return cls.from_dict(json_data)

    @classmethod
    def from_dict(cls, dict_object):
        config = cls(**dict_object)
        return config

    def __str__(self):
        str = ""
        for k,v in self.__dict__.items():
            str += f"{k} : {v}\n"
        return str

    def __repr__(self):
        classname = self.__class__.__name__
        return classname +":\n"+self.__str__()


class TrainingConfig(Config):
    def __init__(self,gradient_accumulation_steps = 1,
                 ckpt_frequency = 1,
                 ckpt_epoch_frequency = 1,
                 log_dir = './logs',
                 output_dir = './saved_models',
                 device = 'cuda'
                 ):
        super(TrainingConfig, self).__init__()

        self.gradient_accumulation_steps =gradient_accumulation_steps
        self.ckpt_frequency = ckpt_frequency
        self.ckpt_epoch_frequency = ckpt_epoch_frequency
        self.log_dir = log_dir
        self.output_dir = output_dir
        self.device = device

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


class IntermediateMatch:
    def __init__(self,layer_T: Union[int,List[int]], layer_S: Union[int,List[int]],
                 weight: float, loss: str, feature: str, proj: Optional[List] = None):
        self.layer_T = layer_T
        self.layer_S = layer_S
        self.feature = feature
        self.weight = weight
        self.loss = loss
        self.proj = proj
        assert feature in FEATURES
        if proj:
            assert proj[0] in PROJ_MAP.keys()
            assert type(proj[1]) is int and type(proj[2]) is int
            if len(proj)==3:
                self.proj.append(dict())   # ['linear', dim_T, dim_S, {...}]
            else:
                assert type(proj[3]) is dict

    def __str__(self):
        str = ""
        for k,v in self.__dict__.items():
            str += f"{k} : {v}, "
        return str[:-2]

    def __repr__(self):
        classname = self.__class__.__name__
        return '\n'+classname +": "+self.__str__()

    @classmethod
    def from_dict(cls,dict_object):
        if dict_object is None:
            return None
        else:
            return cls(**dict_object)


class DistillationConfig(Config):
    def __init__(self,temperature=4,
                      temperature_scheduler = 'none',
                      hard_label_weight=0,
                      hard_label_weight_scheduler = 'none',
                      kd_loss_type='ce',
                      kd_loss_weight=1,
                      kd_loss_weight_scheduler = 'none',
                      probability_shift = False,
                      intermediate_matches:Optional[List[Dict]]=None):
        super(DistillationConfig, self).__init__()

        self.temperature = temperature
        self.temperature_scheduler = None
        if temperature_scheduler is not 'none':
            assert temperature_scheduler in TEMPERATURE_SCHEDULER, \
                    "Invalid temperature_scheduler"
            self.temperature_scheduler = TEMPERATURE_SCHEDULER[temperature_scheduler]

        self.hard_label_weight = hard_label_weight
        self.hard_label_weight_scheduler = None
        if hard_label_weight_scheduler is not 'none':
            assert hard_label_weight_scheduler in WEIGHT_SCHEDULER, \
                    "Invalid hard_label_weight_scheduler"
            self.hard_label_weight_scheduler = WEIGHT_SCHEDULER[hard_label_weight_scheduler]

        self.kd_loss_type = kd_loss_type
        self.kd_loss_weight = kd_loss_weight
        self.kd_loss_weight_scheduler = None
        if kd_loss_weight_scheduler is not 'none':
            assert kd_loss_weight_scheduler in WEIGHT_SCHEDULER, \
                    "Invalid kd_loss_weight_scheduler"
            self.kd_loss_weight_scheduler = WEIGHT_SCHEDULER[kd_loss_weight_scheduler]

        self.probability_shift = probability_shift

        self.intermediate_matches:[List[IntermediateMatch]] = []
        if intermediate_matches:
            self.intermediate_matches = [IntermediateMatch.from_dict(im) for im in intermediate_matches]
