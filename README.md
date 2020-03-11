 [**English**](README.md) | [**中文说明**](README_ZH.md)

<p align="center">
    <br>
    <img src="./pics/banner.png" width="500"/>
    <br>
<p>
<p align="center">
    <a href="https://github.com/airaria/TextBrewer/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/airaria/TextBrewer.svg?color=blue&style=flat-square">
    </a>
    <a href="https://pypi.org/project/textbrewer">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/textbrewer">
    </a>    
    <a href="https://github.com/airaria/TextBrewer/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/v/release/airaria/TextBrewer?include_prereleases">
    </a>
</p>

**TextBrewer** is a PyTorch-based toolkit for **distillation of NLP models**. It includes various distilltion techniques from both NLP and CV, and provides an easy-to-use distillation framkework, which allows users to quickly experiment with state-of-the-art distillation methods to compress the model with a relatively small sacrifice in performance, increase the inference speed and reduce the memory usage.

Paper: [https://arxiv.org/abs/2002.12620](https://arxiv.org/abs/2002.12620)

[API Documentation](API.md)

## Update

**Mar 11, 2020**

* Updated to 0.1.8 (Improvements on TrainingConfig and train method). See details in [releases](https://github.com/airaria/TextBrewer/releases/tag/v0.1.8).

**Mar 2, 2020**

* Initial public version 0.1.7 has been released. See details in [releases](https://github.com/airaria/TextBrewer/releases/tag/v0.1.7).


## Table of Contents

<!-- TOC -->

| Section | Contents |
|-|-|
| [Introduction](#introduction) | Introduction to TextBrewer |
| [Installation](#installation) | How to install |
| [Workflow](#workflow) | Two stages of TextBrewer workflow |
| [Quickstart](#quickstart) | Example: distilling BERT-base to a 3-layer BERT |
| [Experiments](#experiments) | Distillation experiments on typical English and Chinese datasets |
| [Core Concepts](#core-concepts) | Brief explanations of the core concepts in TextBrewer |
| [FAQ](#faq) | Frequently asked questions |
| [Known Issues](#known-issues) | Known issues |
| [Citation](#citation) | Citation to TextBrewer |
| [Follow Us](#follow-us) | - |

<!-- /TOC -->

## Introduction
**Textbrewer** is designed for the knowledge distillation of NLP models. It provides various distillation methods and offers a distillation framework for quickly setting up experiments. 

The main features of **TextBrewer** are:

* Wide-support: it supports various model architectures (especially **transformer**-based models)
* Flexibility: design your own distillation scheme by combining different techniques; it also supports user-defined loss functions, modules, etc.
* Easy-to-use: users don't need to modify the model architectures
* Built for NLP: it is suitable for a wide variety of NLP tasks: text classification, machine reading comprehension, sequence labeling, ...

**TextBrewer** currently is shipped with the following distillation techniques: 

* Mixed soft-label and hard-label training
* Dynamic loss weight adjustment and temperature adjustment
* Various distillation loss functions: hidden states MSE, attention-matrix-based loss, neuron selectivity transfer, ...
* Freely adding intermediate features matching losses
* Multi-teacher distillation
* ...

**TextBrewer** includes:

1. **Distillers**: the cores of distillation. Different distillers perform different distillation modes. There are GeneralDistiller, MultiTeacherDistiller, BasicTrainer, etc. 
2. **Configurations and presets**: Configuration classes for training and distillation, and predefined distillation loss functions and strategies. 
3. **Utilities**: auxiliary tools such as model parameters analysis. 


To start distillation, users need to provide

1. the models (the trained **teacher** model and the un-trained **student** model)
2. datasets and experiment configurations 

**TextBrewer** has achieved impressive results on several typical NLP tasks. See [Experiments](#experiments).

See [API documentation](API.md) for detailed usages.

## Installation

* Requirements
  * Python >= 3.6
  * PyTorch >= 1.1.0
  * TensorboardX or Tensorboard
  * NumPy
  * tqdm
  * Transformers >= 2.0 (optional, used by some examples)

* Install from PyPI

  ```shell
  pip install textbrewer
  ```

* Install from the Github source

  ```shell
  git clone https://github.com/airaria/TextBrewer.git
  pip install ./textbrewer
  ```

## Workflow

![](pics/distillation_workflow_en.png)

* **Stage 1**: Preparation:
  1. Train the teacher model
  2. Define and intialize the student model
  3. Construct a dataloader, an optimizer and a learning rate scheduler

* **Stage 2**: Distillation with TextBrewer:
  1. Construct a **TraningConfig** and a **DistillationConfig**, initialize a **distiller**
  2. Define an **adaptor** and a **callback**. The **adaptor** is used for adaptation of model inputs and outputs. The **callback** is called by the distiller during training
  3. Call the **train** method of the **distiller**


## Quickstart

Here we show the usage of TextBrewer by distilling BERT-base to a 3-layer BERT.

Before distillation, we assume users have provided:

* A trained teacher model `teacher_model` (BERT-base) and a to-be-trained student model `student_model` (3-layer BERT).
* a `dataloader` of the dataset, an `optimizer` and a learning rate `scheduler`.

Distill with TextBrewer:

```python 
import textbrewer
from textbrewer import GeneralDistiller
from textbrewer import TrainingConfig, DistillationConfig

# Show the statistics of model parameters
print("\nteacher_model's parametrers:")
_ = textbrewer.utils.display_parameters(teacher_model,max_level=3)

print("student_model's parametrers:")
_ = textbrewer.utils.display_parameters(student_model,max_level=3)

# Define an adaptor for translating the model inputs and outputs
def simple_adaptor(batch, model_outputs):
  	# The second and third elements of model outputs are the logits and hidden states
    return {'logits': model_outputs[1],
            'hidden': model_outputs[2]}

# Training configuration 
train_config = TrainingConfig()
# Distillation configuration
# Matching different layers of the student and the teacher
distill_config = DistillationConfig(
    intermediate_matches=[    
     {'layer_T':0, 'layer_S':0, 'feature':'hidden', 'loss': 'hidden_mse','weight' : 1},
     {'layer_T':8, 'layer_S':2, 'feature':'hidden', 'loss': 'hidden_mse','weight' : 1}])

# Build distiller
distiller = GeneralDistiller(
    train_config=train_config, distill_config = distill_config,
    model_T = teacher_model, model_S = student_model, 
    adaptor_T = simple_adaptor, adaptor_S = simple_adaptor)

# Start!
with distiller:
    distiller.train(optimizer, scheduler, dataloader, num_epochs=1, callback=None)
```

**Examples can be found in the `examples` directory :**

* [examples/random_token_example](examples/random_token_example) : a simple runable toy example which demonstrates the usage of TextBrewer. This example performs distillation on the text classification task with random tokens as inputs.
* [examples/cmrc2018\_example](examples/cmrc2018_example) (Chinese): distillation on CMRC2018, a Chinese MRC task, using DRCD as data augmentation.
* [examples/mnli\_example](examples/mnli_example) (English): distillation on MNLI, an English sentence-pair classification task. This example also shows how to perform multi-teacher distillation.



## Experiments

We have performed distillation experiments on several typical English and Chinese NLP datasets. The setups and configurations are listed below.

### Models

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

The definitions of matches are at [exmaple/matches/matches.py](exmaple/matches/matches.py). 

We use GeneralDistiller in all the distillation experiments.

### Training Configurations

* Learning rate is 1e-4 (unless otherwise specified).  
* We train all the models for 30~60 epochs.

### Results on English Datasets

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

1. The equivlent model architectures of public models are shown in the brackets. 
2. When distilling to T4-tiny, NewsQA is used for data augmentation on SQuAD and HotpotQA is used for data augmentation on CoNLL-2003.



### Results on Chinese Datasets

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

## Core Concepts

### Configurations

* `TrainingConfig`: configuration related to general deep learning model training
* `DistillationConfig`: configuration related to distillation methods

### Distillers

Distillers are in charge of conducting the actual experiments. The following distillers are available:

* `BasicDistiller`: **single-teacher single-task** distillation, provides basic distillation strategies.
* `GeneralDistiller` (Recommended): **single-teacher single-task** distillation, supports intermediate features matching. **Recommended most of the time**.
* `MultiTeacherDistiller`: **multi-teacher** distillation, which distills multiple teacher models (of the same task) into a single student model. **This class doesn't support Intermediate features matching.**
* `MultiTaskDistiller`: **multi-task** distillation, which distills multiple teacher models (of different tasks) into a single student. **This class doesn't support Intermediate features matching.**
* `BasicTrainer`: Supervised training a single model on a labeled dataset, not for distillation. **It can be used to train a teacher model**.


### User-Defined Functions

In TextBrewer, there are two functions that should be implemented by users: **callback** and **adaptor**.

####  **Callback** 

At each checkpoint, after saving the student model, the callback function will be called by the distiller. Callback can be used to evaluate the performance of the student model at each checkpoint.

#### Adaptor
It converts the model inputs and outputs to the specified format so that they could be recognized by the distiller, and distillation losses can be computed. At each training step, batch and model outputs will be passed to the adaptor; adaptor re-organize the data and returns a dictionary.

Fore more details, see the explanations in [API documentation](API.md)

## FAQ

**Q**: How to initialize the student model?

**A**: The student model could be randomly initialized (i.e., with no prior knwledge) or be initialized by pre-trained weights.
For example, when distilling a BERT-base model to a 3-layer BERT, you could initialize the student model with [RBT3](#https://github.com/ymcui/Chinese-BERT-wwm) (for Chinese tasks) or the first three layers of BERT (for English tasks) to avoid cold start problem. 
We recommend that users use pre-trained student models whenever possible to fully take the advantage of large-scale pre-training.

**Q**: How to set training hyperparamters for the distillation experiments？

**A**: Knowledge distillation usually requires more training epochs and larger learning rate than training on labeled dataset. For example, training SQuAD on BERT-base usually takes 3 epochs with lr=3e-5; however, distillation takes 30~50 epochs with lr=1e-4. **The conclusions are based on our experiments, and you are advised to try on your own data**.

## Known Issues

* Compatibility with FP16 training has not been tested.
* Multi-GPU training support is only available through `DataParallel` currently.

## Citation

If you find TextBrewer is helpful, please cite [our paper](https://arxiv.org/abs/2002.12620):
```
@article{textbrewer,
  title={TextBrewer: An Open-Source Knowledge Distillation Toolkit for Natural Language Processing},
  author={Yang, Ziqing and Cui, Yiming and Chen, Zhipeng and Che, Wanxiang and Liu, Ting and Wang, Shijin and Hu, Guoping},
  journal={arXiv preprint arXiv:2002.12620},
  year={2020}
 }
```

## Follow Us
Follow our official WeChat account to keep updated with our latest technologies!

![](pics/hfl_qrcode.jpg)
