import os, pickle
import torch
from torch.utils.data import TensorDataset

label2id_dict = {
    'O': 0,
    'B-LOC': 1,
    'I-LOC': 2,
    'B-ORG': 3,
    'I-ORG': 4,
    'B-PER': 5,
    'I-PER': 6
}

id2label_dict = {
    0: 'O',
    1: 'B-LOC',
    2: 'I-LOC',
    3: 'B-ORG',
    4: 'I-ORG',
    5: 'B-PER',
    6: 'I-PER'
}

class Examples:
    def __init__(self, tokens, label_ids):
        self.tokens = tokens
        self.label_ids = label_ids

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += f"tokens: {''.join(self.tokens)}\n"
        s += f"labels: {' '.join(str(i) for i in self.label_ids)}\n"
        return s

class Featues:
    def __init__(self, token_ids, input_mask, label_ids):
        self.token_ids = token_ids
        self.input_mask = input_mask
        self.label_ids = label_ids


    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += f"token_ids: {' '.join(str(i) for i in  self.token_ids)}\n"
        s += f"label_ids: {' '.join(str(i) for i in self.label_ids)}\n"
        s += f"input_mask:{' '.join(str(i) for i in self.input_mask)}\n"
        return s

def read_examples(input_file):
    examples = []
    tokens = []
    label_ids = []
    errors = 0
    with open(input_file) as f:
        for idx,line in enumerate(f):
            if len(line.strip())==0:
                if len(tokens)>0:
                    examples.append(Examples(tokens,label_ids))
                    tokens = []
                    label_ids = []
                continue
            try:
                token, label = line.strip().split('\t')
            except ValueError:
                errors +=1
                continue
            tokens.append(token)
            label_ids.append(label2id_dict[label])
        if len(tokens) > 0:
            examples.append(Examples(tokens, label_ids))
    print ("Num errors: ", errors)
    return examples

def convert_example_to_features(input_file, tokenizer, max_seq_length,
                                cls_token='[CLS]', sep_token='[SEP]', pad_token_id=0):
    features = []

    examples = read_examples(input_file)

    #convert token to ids
    pad_label = [label2id_dict['O']]
    for example in examples:
        tokens = [cls_token] + example.tokens[:max_seq_length-2] + [sep_token]
        label_ids = pad_label + example.label_ids[:max_seq_length-2] + pad_label


        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(token_ids)

        padding_length = max_seq_length - len(token_ids)
        token_ids = token_ids + [pad_token_id] * padding_length
        input_mask = input_mask + [0] * padding_length
        label_ids = label_ids + pad_label * padding_length

        assert len(token_ids) == len(input_mask) == len(label_ids)

        features.append(Featues(token_ids=token_ids,input_mask=input_mask,label_ids=label_ids))

    return examples, features

def read_features(input_file,  max_seq_length=160, tokenizer=None, cls_token='[CLS]', sep_token='[SEP]', pad_token_id=0):
    cached_features_file = input_file +f'.cached_feat_{max_seq_length}'
    if os.path.exists(cached_features_file):
        with open(cached_features_file,'rb') as f:
            examples, features = pickle.load(f)
    else:
        examples, features = convert_example_to_features(input_file,tokenizer,max_seq_length,cls_token,sep_token,pad_token_id)
        with open(cached_features_file, 'wb') as f:
            pickle.dump([examples, features],f)

    all_token_ids = torch.tensor([f.token_ids for f in features],dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features],dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_token_ids,all_input_mask,all_label_ids)

    return examples, dataset

if __name__ == '__main__':
    #from transformers import BertTokenizer
    #tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    input_file = 'msra_train_bio.txt'
    max_seq_length = 128

    #dataset = read_features(input_file, 128,tokenizer)
    #print (f"length of dataset: {len(dataset)}")
    #print (dataset[0])
    #print (dataset[-1])

    examples = read_examples(input_file)
    length = [len(example.tokens) for example in examples]
    import numpy as np
    print (np.max(length),np.mean(length),np.percentile(length,99))
    print (sum(i>160 for i in length)/len(length))