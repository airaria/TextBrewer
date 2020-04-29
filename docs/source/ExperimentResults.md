# Experimental Results


## English Datasets

### MNLI

* Training without Distillation:

| Model（ours)  | MNLI |
| ------------- | ------------- |
| **BERT-base-cased** | 83.7 / 84.0   |
| T3                  | 76.1 / 76.5   |

* Single-teacher distillation with `GeneralDistiller`:

| Model (ours)         | MNLI           |
| :-------------       | -------------- |
| **BERT-base-cased** (teacher)  | 83.7 / 84.0    |
| T6 (student)                   | 83.5 / 84.0    |
| T3  (student)                  | 81.8 / 82.7    |
| T3-small (student)             | 81.3 / 81.7    |
| T4-tiny (student)              | 82.0 / 82.6    |
| T12-nano (student)             | 83.2 / 83.9    |

* Multi-teacher distillation with `MultiTeacherDistiller`:

| Model (ours)         | MNLI           |
| :-------------       | -------------- |
| **BERT-base-cased** (teacher #1)  | 83.7 / 84.0    |
| **BERT-base-cased** (teacher #2)  | 83.6 / 84.2    |
| **BERT-base-cased** (teacher #3)  | 83.7 / 83.8    |
| ensemble (average of #1, #2 and #3)  | 84.3 / 84.7    |
| BERT-base-cased (student)         | **84.8 / 85.3**|

### SQuAD

* Training without Distillation:

| Model（ours)  | SQuAD | 
| ------------- | ------------- |
| **BERT-base-cased** | 81.5 / 88.6   |
| T6            | 75.0 / 83.3   |
| T3            | 63.0 / 74.3   |

* Single-teacher distillation with `GeneralDistiller`:

| Model（ours)            | SQuAD | 
| -------------           | ------------- |
| **BERT-base-cased** (teacher) | 81.5 / 88.6   |
| T6 (student)            | 80.8 / 88.1   |
| T3 (student)            | 76.4 / 84.9   |
| T3-small (student)      | 72.3 / 81.4   |
| T4-tiny (student)       | 73.7 / 82.5   |
| &nbsp;&nbsp;+ DA        | 75.2 / 84.0   |
| T12-nano (student)      | 79.0 / 86.6   |

**Note**: When distilling to T4-tiny, NewsQA is used for data augmentation on SQuAD.

* Multi-teacher distillation with `MultiTeacherDistiller`:

| Model (ours)         | SQuAD          |
| :-------------       | -------------- |
| **BERT-base-cased** (teacher #1)  | 81.1 / 88.6    |
| **BERT-base-cased** (teacher #2)  | 81.2 / 88.5    |
| **BERT-base-cased** (teacher #3)  | 81.2 / 88.7    |
| ensemble (average of #1, #2 and #3)  | 82.3 / 89.4 |
| BERT-base-cased (student)         | **83.5 / 90.0**|

### CoNLL-2003 English NER

* Training without Distillation:

| Model（ours)  |  CoNLL-2003 |
| ------------- | ----------- |
| **BERT-base-cased** | 91.1  |
| BiGRU               | 81.1  |
| T3                  | 85.3  |

* Single-teacher distillation with `GeneralDistiller`:

| Model（ours)            | CoNLL-2003 | 
| -------------           | ------------- |
| **BERT-base-cased** (teacher) | 91.1    |
| BiGRU                   | 85.3 |
| T6 (student)            | 90.7 |
| T3 (student)            | 87.5 |
| &nbsp;&nbsp;+ DA        | 90.0 |
| T3-small (student)      | 78.6 |
| &nbsp;&nbsp;+ DA        | -    |
| T4-tiny (student)       | 77.5 |
| &nbsp;&nbsp;+ DA        | 89.1 |
| T12-nano (student)      | 78.8 |
| &nbsp;&nbsp;+ DA        | 89.6 |

**Note**: HotpotQA is used for data augmentation on CoNLL-2003.

## Chinese Datasets (RoBERTa-wwm-ext as the teacher)

### XNLI

| Model           | XNLI |
| :--------------- | ----------------- |
| **RoBERTa-wwm-ext** (teacher) | 79.9 |
| T3 (student)         | 78.4       |
| T3-small (student)   | 76.0       |
| T4-tiny (student)    | 76.2       |

### LCQMC

| Model            | LCQMC |
| :--------------- | ----------- |
| **RoBERTa-wwm-ext** (teacher) | 89.4        | 
| T3 (student)                  | 89.0        |
| T3-small (student)            | 88.1        |
| T4-tiny (student)             | 88.4        |

### CMRC 2018 and DRCD

| Model           | CMRC 2018 | DRCD |
| --------------- | ---------------- | ------------ |
| **RoBERTa-wwm-ext** (teacher) | 68.8 / 86.4      | 86.5 / 92.5  |
| T3 (student)                  | 63.4 / 82.4      | 76.7 / 85.2  |
| &nbsp;&nbsp;+ DA              | 66.4 / 84.2      | 78.2 / 86.4  |
| T3-small (student)            | 46.1 / 71.0      | 71.4 / 82.2  |
| &nbsp;&nbsp;+  DA             | 58.0 / 79.3      | 75.8 / 84.8  |
| T4-tiny (student)             | 54.3 / 76.8      | 75.5 / 84.9  |
| &nbsp;&nbsp;+  DA             | 61.8 / 81.8      | 77.3 / 86.1  |

**Note**: CMRC 2018 and DRCD take each other as the augmentation dataset on the experiments. 

## Chinese Datasets (Electra-base as the teacher)

* Training without Distillation:

| Model                      | XNLI       | LCQMC  | CMRC 2018     | DRCD         | MSRA NER|
|:---------------------------|------------|--------| --------------| -------------|---------|
| **Electra-base** (teacher) | 77.8       | 89.8   | 65.6 / 84.7   | 86.9 / 92.3  | 95.14   |
| Electra-small (pretrained) | 72.5       | 86.3   | 62.9 / 80.2   | 79.4 / 86.4  |         |

* Single-teacher distillation with `GeneralDistiller`:

| Model                       | XNLI       | LCQMC       | CMRC 2018       | DRCD         | MSRA NER |
| :---------------------------|------------|-------------|-----------------| -------------|----------|
| **Electra-base** (teacher)  | 77.8       | 89.8        | 65.6 / 84.7     | 86.9 / 92.3  | 95.14    |
| Electra-small (random)      | 77.2       | 89.0        | 66.5 / 84.9     | 84.8 / 91.0  |          |
| Electra-small (pretrained)  | 77.7       | 89.3        | 66.5 / 84.9     | 85.5 / 91.3  |93.48     |

**Note**: 

1. Random: randomly initialized
2. Pretrained: initialized with pretrained weights

A good initialization of the student (Electra-small) improves the performance.