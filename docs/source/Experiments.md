
# Experiments

We have performed distillation experiments on several typical English and Chinese NLP datasets. The setups and configurations are listed below.

## Models

* For English tasks, the teacher model is [**BERT-base-cased**](https://github.com/google-research/bert).
* For Chinese tasks, the teacher models are [**RoBERTa-wwm-ext**](https://github.com/ymcui/Chinese-BERT-wwm) and [**Electra-base**](https://github.com/ymcui/Chinese-ELECTRA) released by the Joint Laboratory of HIT and iFLYTEK Research.

We have tested different student models. To compare with public results, the student models are built with standard transformer blocks except for BiGRU which is a single-layer bidirectional GRU. The architectures are listed below. Note that the number of parameters includes the embedding layer but does not include the output layer of each specific task. 

#### English models

| Model                 | \#Layers | Hidden size | Feed-forward size | \#Params | Relative size |
| :--------------------- | --------- | ----------- | ----------------- | -------- | ------------- |
| BERT-base-cased (teacher) | 12        | 768         | 3072              | 108M     | 100%          |
| T6 (student)              | 6         | 768         | 3072              | 65M      | 60%           |
| T3 (student)              | 3         | 768         | 3072              | 44M      | 41%           |
| T3-small (student)        | 3         | 384         | 1536              | 17M      | 16%           |
| T4-Tiny (student)         | 4         | 312         | 1200              | 14M      | 13%           |
| T12-nano (student)        | 12        | 256         | 1024              | 17M      | 16%           |
| BiGRU (student)           | -         | 768         | -                 | 31M      | 29%           |

#### Chinese models

| Model                 | \#Layers | Hidden size | Feed-forward size | \#Params | Relative size   |
| :--------------------- | --------- | ----------- | ----------------- | -------- | ------------- |
| RoBERTa-wwm-ext (teacher) | 12        | 768         | 3072              | 102M      | 100%          |
| Electra-base (teacher)    | 12        | 768         | 3072              | 102M      | 100%          |
| T3 (student)              | 3         | 768         | 3072              | 38M       | 37%           |
| T3-small (student)        | 3         | 384         | 1536              | 14M       | 14%           |
| T4-Tiny (student)         | 4         | 312         | 1200              | 11M       | 11%           |
| Electra-small (student)   | 12        | 256         | 1024              | 12M       | 12%           |

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

| Model        | matches                                             |
| :--------    | --------------------------------------------------- |
| BiGRU        | None                                                |
| T6           | L6_hidden_mse + L6_hidden_smmd                      |
| T3           | L3_hidden_mse + L3_hidden_smmd                      |
| T3-small     | L3n_hidden_mse + L3_hidden_smmd                     |
| T4-Tiny      | L4t_hidden_mse + L4_hidden_smmd                     |
| T12-nano     | small_hidden_mse + small_hidden_smmd                |
| Electra-small| small_hidden_mse + small_hidden_smmd                |

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

Our results (see [Experimental Results](ExperimentResults.md) for details):

| Model (ours) | MNLI  | SQuAD  | CoNLL-2003 |
| :-------------  | --------------- | ------------- | --------------- |
| **BERT-base-cased** (teacher) | 83.7 / 84.0     | 81.5 / 88.6   | 91.1  |
| BiGRU          | -               | -             | 85.3            |
| T6             | 83.5 / 84.0     | 80.8 / 88.1   | 90.7            |
| T3             | 81.8 / 82.7     | 76.4 / 84.9   | 87.5            |
| T3-small       | 81.3 / 81.7     | 72.3 / 81.4   | 78.6            |
| T4-tiny        | 82.0 / 82.6     | 75.2 / 84.0   | 89.1            |
| T12-nano       | 83.2 / 83.9     | 79.0 / 86.6   | 89.6            |

**Note**:

1. The equivalent model structures of public models are shown in the brackets after their names. 
2. When distilling to T4-tiny, NewsQA is used for data augmentation on SQuAD and HotpotQA is used for data augmentation on CoNLL-2003.
3. When distilling to T12-nano, HotpotQA is used for data augmentation on CoNLL-2003.



## Results on Chinese Datasets

We experiment on the following typical Chinese datasets:


| Dataset | Task type | Metrics | \#Train | \#Dev | Note |
| :------- | ---- | ------- | ------- | ---- | ---- |
| [**XNLI**](https://github.com/google-research/bert/blob/master/multilingual.md) | text classification | Acc | 393K | 2.5K | Chinese translation version of MNLI |
| [**LCQMC**](http://icrc.hitsz.edu.cn/info/1037/1146.htm) | text classification | Acc | 239K | 8.8K | sentence-pair matching, binary classification |
| [**CMRC 2018**](https://github.com/ymcui/cmrc2018) | reading comprehension | EM/F1 | 10K | 3.4K | span-extraction machine reading comprehension |
| [**DRCD**](https://github.com/DRCKnowledgeTeam/DRCD) | reading comprehension | EM/F1 | 27K | 3.5K | span-extraction machine reading comprehension (Traditional Chinese) |
| [**MSRA NER**](https://faculty.washington.edu/levow/papers/sighan06.pdf) | sequence labeling | F1 | 45K | 3.4K (test) | Chinese named entity recognition |

The results are listed below (see [Experimental Results](ExperimentResults.md) for details).

| Model           | XNLI | LCQMC | CMRC 2018 | DRCD |
| :--------------- | ---------- | ----------- | ---------------- | ------------ |
| **RoBERTa-wwm-ext** (teacher) | 79.9       | 89.4        | 68.8 / 86.4      | 86.5 / 92.5  |
| T3          | 78.4       | 89.0        | 66.4 / 84.2      | 78.2 / 86.4  |
| T3-small    | 76.0       | 88.1        | 58.0 / 79.3      | 75.8 / 84.8  |
| T4-tiny     | 76.2       | 88.4        | 61.8 / 81.8      | 77.3 / 86.1  |

| Model                      | XNLI       | LCQMC       | CMRC 2018        | DRCD        | MSRA NER |
| :---------------           | ---------- | ----------- | ---------------- | ------------|----------|
| **Electra-base** (teacher) | 77.8       | 89.8        | 65.6 / 84.7     | 86.9 / 92.3  | 95.14    |
| Electra-small              | 77.7       | 89.3        | 66.5 / 84.9     | 85.5 / 91.3  | 93.48    |


**Note**:

1. Learning rate decay is not used in distillation on CMRC 2018 and DRCD.
2. CMRC 2018 and DRCD take each other as the augmentation dataset in the distillation.
3. The settings of training Electra-base teacher model can be found at [**Chinese-ELECTRA**](https://github.com/ymcui/Chinese-ELECTRA).
4. Electra-small student model is intialized with the [pretrained weights](https://github.com/ymcui/Chinese-ELECTRA).