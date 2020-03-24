# Experimental Results


## Results on English Datasets

### MNLI

* Single-teacher distillation with `GeneralDistiller`:

| Model (ours)         | MNLI           |
| :-------------       | -------------- |
| **BERT-base-cased** (teacher)  | 83.7 / 84.0    |
| T6 (student)                   | 83.5 / 84.0    |
| T3  (student)                  | 81.8 / 82.7    |
| T3-small (student)             | 81.3 / 81.7    |
| T4-tiny (student)              | 82.0 / 82.6    |

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
| &nbsp;&nbsp;+ DA                 | 75.2 / 84.0   |

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

| Model（ours)  | SQuAD | 
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
| T3-small (student)      | 57.4 |
| &nbsp;&nbsp;+ DA        | 76.5 |
| T4-tiny (student)       | 54.7 |
| &nbsp;&nbsp;+ DA        | 79.6 |

**Note**: HotpotQA is used for data augmentation on CoNLL-2003.

## Results on Chinese Datasets

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

### CMRC2018 and DRCD

| Model           | CMRC2018 | DRCD |
| --------------- | ---------------- | ------------ |
| **RoBERTa-wwm-ext** (teacher) | 68.8 / 86.4      | 86.5 / 92.5  |
| T3 (student)                  | 63.4 / 82.4      | 76.7 / 85.2  |
| &nbsp;&nbsp;+ DA              | 66.4 / 84.2      | 78.2 / 86.4  |
| T3-small (student)            | 24.4 / 48.1      | 42.2 / 63.2  |
| &nbsp;&nbsp;+  DA             | 58.0 / 79.3      | 65.5 / 78.6  |
| T4-tiny (student)             | -                | -            |
| &nbsp;&nbsp;+  DA             | 61.8 / 81.8      | 73.3 / 83.5  |

**Note**: CMRC2018 and DRCD take each other as the augmentation dataset In the experiments. 