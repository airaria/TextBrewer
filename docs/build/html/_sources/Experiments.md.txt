
# Experiments

We have performed distillation experiments on several typical English and Chinese NLP datasets. The setups and configurations are listed below.

## Models

* For English tasks, the teacher model is [**BERT-base-cased**](https://github.com/google-research/bert).
* For Chinese tasks, the teacher model is [**RoBERTa-wwm-ext**](https://github.com/ymcui/Chinese-BERT-wwm) released by the Joint Laboratory of HIT and iFLYTEK Research.

We have tested different student models. To compare with public results, the student models are built with standard transformer blocks except BiGRU which is a single-layer bidirectional GRU. The architectures are listed below. Note that the number of parameters includes the embedding layer but does not include the output layer of the each specific task. 

| Model                 | \#Layers | Hidden_size | Feed-forward size | \#Params | Relative size |
| :--------------------- | --------- | ----------- | ----------------- | -------- | ------------- |
| BERT-base-cased (teacher)  | 12        | 768         | 3072              | 108M     | 100%          |
| RoBERTa-wwm-ext (teacher) | 12        | 768         | 3072              | 108M     | 100%          |
| T6 (student)              | 6         | 768         | 3072              | 65M      | 60%           |
| T3 (student)              | 3         | 768         | 3072              | 44M      | 41%           |
| T3-small (student)        | 3         | 384         | 1536              | 17M      | 16%           |
| T4-Tiny (student)         | 4         | 312         | 1200              | 14M      | 13%           |
| BiGRU (student)           | -         | 768         | -                 | 31M      | 29%           |

* T6 archtecture is the same as [DistilBERT<sup>[1]</sup>](https://arxiv.org/abs/1910.01108), [BERT<sub>6</sub>-PKD<sup>[2]</sup>](https://arxiv.org/abs/1908.09355), and  [BERT-of-Theseus<sup>[3]</sup>](https://arxiv.org/abs/2002.02925).
* T4-tiny archtecture is the same as [TinyBERT<sup>[4]</sup>](https://arxiv.org/abs/1909.10351).
* T3 architecure is the same as [BERT<sub>3</sub>-PKD<sup>[2]</sup>](https://arxiv.org/abs/1908.09355).

## Configurations

### Distillation Configurations

```python
distill_config = DistillationConfig(temperature = 8, intermediate_matches = matches)
# Others arguments take the default values
```

`matches` are differnt for different models:

| Model    | matches                                                      |
| :-------- | ------------------------------------------------------------ |
| BiGRU    | None                                                         |
| T6       | L6_hidden_mse + L6_hidden_smmd                               |
| T3       | L3_hidden_mse + L3_hidden_smmd                               |
| T3-small | L3n_hidden_mse + L3_hidden_smmd                              |
| T4-Tiny  | L4t_hidden_mse + L4_hidden_smmd                              |

The definitions of `matches` are at [exmaple/matches/matches.py](https://github.com/airaria/TextBrewer/blob/master/examples/matches/matches.py). 

We use `GeneralDistiller` in all the distillation experiments.

### Training Configurations

* Learning rate is 1e-4 (unless otherwise specified).  
* We train all the models for 30~60 epochs.

## Results on English Datasets

We experiment on the following typical Enlgish datasets:

| Dataset    | Task type | Metrics | \#Train | \#Dev | Note |
| :---------- | -------- | ------- | ------- | ---- | ---- | 
| [**MNLI**](https://www.nyu.edu/projects/bowman/multinli/)       | text classification | m/mm Acc | 393K    | 20K  | sentence-pair 3-class classification |
| [**SQuAD 1.1**](https://rajpurkar.github.io/SQuAD-explorer/)   | reading comprehension | EM/F1   | 88K     | 11K  | span-extraction machine reading comprehension | 
| [**CoNLL-2003**](https://www.clips.uantwerpen.be/conll2003/ner) | sequence labeling | F1      | 23K     | 6K   | named entity recognition |

We list the public results from [DistilBERT](https://arxiv.org/abs/1910.01108), [BERT-PKD](https://arxiv.org/abs/1908.09355), [BERT-of-Theseus](https://arxiv.org/abs/2002.02925), [TinyBERT](https://arxiv.org/abs/1909.10351) and our results below for comparison.

Public results:

  | Model (public) | MNLI  | SQuAD  | CoNLL-2003 |
  | :-------------  | --------------- | ------------- | --------------- |
  | DistilBERT (T6)    | 81.6 / 81.1 | 78.1 / 86.2   | -               |
  | BERT<sub>6</sub>-PKD (T6)     | 81.5 / 81.0     | 77.1 / 85.3   | -|
  | BERT-of-Theseus (T6) | 82.4/  82.1   | -        | -                |
  | BERT<sub>3</sub>-PKD (T3)     | 76.7 / 76.3     | -             | -|
  | TinyBERT (T4-tiny) | 82.8 / 82.9                | 72.7 / 82.1   | -|

Our results:

| Model (ours) | MNLI  | SQuAD  | CoNLL-2003 |
| :-------------  | --------------- | ------------- | --------------- |
| **BERT-base-cased**  | 83.7 / 84.0     | 81.5 / 88.6   | 91.1  |
| BiGRU          | -               | -             | 85.3            |
| T6             | 83.5 / 84.0     | 80.8 / 88.1   | 90.7            |
| T3             | 81.8 / 82.7     | 76.4 / 84.9   | 87.5            |
| T3-small       | 81.3 / 81.7     | 72.3 / 81.4   | 57.4            |
| T4-tiny        | 82.0 / 82.6     | 75.2 / 84.0   | 79.6            |

**Note**:

1. The equivlent model architectures of public models are shown in the brackets after their names. 
2. When distilling to T4-tiny, NewsQA is used for data augmentation on SQuAD and HotpotQA is used for data augmentation on CoNLL-2003.



## Results on Chinese Datasets

We experiment on the following typical Chinese datasets:


| Dataset | Task type | Metrics | \#Train | \#Dev | Note |
| :------- | ---- | ------- | ------- | ---- | ---- |
| [**XNLI**](https://github.com/google-research/bert/blob/master/multilingual.md) | text classification | Acc | 393K | 2.5K | Chinese translation version of MNLI |
| [**LCQMC**](http://icrc.hitsz.edu.cn/info/1037/1146.htm) | text classification | Acc | 239K | 8.8K | sentence-pair matching, binary classification |
| [**CMRC 2018**](https://github.com/ymcui/cmrc2018) | reading comprehension | EM/F1 | 10K | 3.4K | span-extraction machine reading comprehension |
| [**DRCD**](https://github.com/DRCKnowledgeTeam/DRCD) | reading comprehension | EM/F1 | 27K | 3.5K | span-extraction machine reading comprehension (Traditional Chinese) |

The results are listed below.

| Model           | XNLI | LCQMC | CMRC 2018 | DRCD |
| :--------------- | ---------- | ----------- | ---------------- | ------------ |
| **RoBERTa-wwm-ext** | 79.9       | 89.4        | 68.8 / 86.4      | 86.5 / 92.5  |
| T3          | 78.4       | 89.0        | 66.4 / 84.2      | 78.2 / 86.4  |
| T3-small    | 76.0       | 88.1        | 58.0 / 79.3      | 65.5 / 78.6  |
| T4-tiny     | 76.2       | 88.4        | 61.8 / 81.8      | 73.3 / 83.5  |


**Note**:

1. On CMRC2018 and DRCD, learning rates are 1.5e-4 and 7e-5 respectively and there is no learning rate decay.
2. CMRC2018 and DRCD take each other as the augmentation dataset In the experiments. 