from transformers import (
    BertConfig, BertTokenizer, BertForSequenceClassification,
    #CamembertConfig, CamembertTokenizer, CamembertForSequenceClassification,
    #XLMRobertaConfig,XLMRobertaTokenizer, XLMRobertaForSequenceClassification,
    #RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification,
)
import json
from typing import Dict,List
from modeling import BertForGLUESimple

MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizer, BertForGLUESimple),
}

def parse_model_config(config) -> Dict :

    results = {"teachers":[]}

    if isinstance(config,str):
        with open(config,'r') as f:
            config = json.load(f)
    else:
        assert isinstance(config,dict)
    teachers = config['teachers']
    for teacher in teachers:
        if teacher['disable'] is False:
            model_config, model_tokenizer, _ = MODEL_CLASSES[teacher['model_type']]
            if teacher['vocab_file'] is not None:
                kwargs = teacher.get('tokenizer_kwargs',{})
                teacher['tokenizer'] = model_tokenizer(vocab_file=teacher['vocab_file'],**kwargs)
            if teacher['config_file'] is not None:
                teacher['config'] = model_config.from_json_file(teacher['config_file'])
            results['teachers'].append(teacher)

    student = config['student']
    if student['disable'] is False:
        model_config, model_tokenizer, _ = MODEL_CLASSES[student['model_type']]
        if student['vocab_file'] is not None:
            kwargs = student.get('tokenizer_kwargs',{})
            student['tokenizer'] = model_tokenizer(vocab_file=student['vocab_file'],**kwargs)
        if student['config_file'] is not None:
            student['config'] = model_config.from_json_file(student['config_file'])
        if 'num_hidden_layers' in student:
            student['config'].num_hidden_layers = student['num_hidden_layers']
        results['student'] = student

    return results
