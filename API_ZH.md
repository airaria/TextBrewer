# API

这是 TextBrewer的完整API使用文档。我们之后会将文档迁移到readthedoc。

<!-- TOC -->

- [API](#api)
    - [核心概念](#核心概念)
        - [变量与规范](#变量与规范)
        - [Config和Distiller](#config和distiller)
        - [用户定义函数](#用户定义函数)
    - [Classes and functions](#classes-and-functions)
        - [Configurations](#configurations)
        - [Distillers](#distillers)
        - [utils](#utils)
        - [data_utils](#data_utils)
    - [预定义列表与字典 (presets)](#预定义列表与字典-presets)
        - [自定义](#自定义)
        - [中间特征损失函数](#中间特征损失函数)

<!-- /TOC -->

## 核心概念

### 变量与规范 

我们采用以下名称约定：

* **Model_T** (教师模型) : `torch.nn.Module`的实例。教师模型，一般来说参数量等于大于学生模型。

* **Model_S** (学生模型)：`torch.nn.Module`的实例。学生模型，蒸馏的目标模型。

* **optimizer**: 优化器，`torch.optim.Optimizer`的实例。

* **scheduler**: 动态调整学习率。`torch.optim.lr_scheduler`下的类的实例，提供单独的学习率调整策略。

* **dataloader**: 迭代器，用于获取 batch，一般用`torch.utils.data.Dataloader`构造。batch的类型可以是`tuple`或`dict`:

  ```python
  for batch in dataloader:
    # if batch_postprocessor is not None:
    batch = batch_postprocessor(batch)
    # check batch datatype
    # passes batch to the model and adaptors
  ```
  
  **注意：**:
  1. 训练循环中会判断batch是否是dict。如果是dict，那么以model(\*\*batch, \*\*args) 的形式调用model，否则以 model(\*batch, \*\*args)的形式调用model。所以当batch不是dict时，**注意batch中每个元素的顺序和model.forward的参数的顺序相一致**。`args`则用于传递额外的参数。
  2. 如果用户需要对从dataloader得到的batch进行后处理，可以定义batch_postprocessor函数，接受一个batch并返回处理后的batch。详见[Distillers](#Distillers)中关于`train`方法的说明。

### Config和Distiller

#### Configurations

* **TrainingConfig**：训练相关的通用配置
* **DistillationConfig**：蒸馏相关的配置


#### Distillers

Distiller负责执行实际的蒸馏过程。目前实现了以下的distillers:

* `BasicDistiller`: 提供**单模型单任务**蒸馏方式。可用作测试或简单实验。
* `GeneralDistiller` (**常用**): 提供**单模型单任务**蒸馏方式，并且支持**中间层特征匹配**，一般情况下**推荐使用**。
* `MultiTeacherDistiller`: 多教师蒸馏。将多个（同任务）教师模型蒸馏到一个学生模型上。**暂不支持中间层特征匹配**。
* `MultiTaskDistiller`：多任务蒸馏。将多个（不同任务）单任务教师模型蒸馏到一个多任务学生模型上。**暂不支持中间层特征匹配**。
* `BasicTrainer`：用于单个模型的有监督训练，而非蒸馏。**可用于训练教师模型**。

### 用户定义函数

蒸馏实验中，有两个组件需要由用户提供，分别是**callback** 和 **adaptor**。他们的作用与约定如下：

####  **Callback** 
回调函数，可选，可以为None。在每个checkpoint，distiller在保存模型后会调用callback，并传入参数 `model=model_S, step=global_step`。可以借由回调函数在每个checkpoint评测模型效果。**如果在callback中评测模型，别忘了在callback中调用 model.eval()**。callback的签名为：

```python
  callback(model: torch.nn.Module, step: int) -> Any
```

#### Adaptor
将模型的输入和输出转换为指定的格式，向distiller解释模型的输入和输出，以便distiller根据不同的策略进行不同的计算。具体地说，在每个训练步，batch和模型的输出model_outputs会作为参数传递给adaptor，adaptor负责重新组织这些数据，返回一个字典：

```python
adatpor(batch: Union[Dict,Tuple], model_outputs: Tuple) -> Dict
```

它的作用示意图如下

![](pics/adaptor.png)



 返回的dict可以包含如下有效键值：

  * '**logits**' : `List[torch.Tensor]` or `torch.Tensor` : 

      需要计算蒸馏损失的logits，通常为模型最后softmax之前的输出。每个tensor的形状为 (batch_size, num_labels) 或 (batch_size, length, num_labels)

  * '**logits_mask**' : `List[torch.Tensor]` or `torch.Tensor`:  
  
      0/1矩阵，对logits的某些位置做mask。如果不想对logits某些位置计算损失，用mask遮掩掉（对应位置设为0）。每个tensor的形状为 (batch_size, length)

  * **'labels'**: `List[torch.Tensor]` or `torch.Tensor`: 
    ground-truth标签，列表中每一个tensor形状为 (batch_size,) 或 (batch_size, length)

    **注意**: 
    
    * **logits_mask** 仅对形状为 (batch_size, length, num_labels) 的 logits 有效，用于在length维度做mask，一般用于序列标注类型的任务

    * **logits** 和 **logits_mask** 和 **labels** **要么同为 list/tuple of tensor, 要么都是tensor**

  * ’**losses**' : `List[torch.Tensor]` : 

    如果模型中已经计算了一些损失并想利用这些损失训练，例如预测的logits和ground-truth的交叉熵，可放在这里。训练时 'losses'下的所有损失将求和并乘以**hard_label_weight**,和蒸馏的损失加在一起做backward。列表中每一个tensor应为scalar，即shape为[]

  * '**attention**': `List[torch.Tensor]` :

    attention矩阵的列表，用于计算中间层特征匹配。每个tensor的形状为 (batch_size, num_heads, length, length) 或 (batch_size, length, length) ，取决于应用于attention的损失的选取。各种损失函数详见[中间特征损失函数](#中间特征损失函数)

* '**hidden**': `List[torch.Tensor]` :

  hidden states的列表，用于计算中间层特征匹配。每个tensor的形状为(batch_size, length,hidden_dim)

* '**inputs_mask**' : `torch.Tensor` : 

  0/1矩阵，对'attention'和“hidden'中张量做mask。形状为 (batch_size, length)


这些key**都是可选**的：

* 如果没有 **'inputs_mask'** 或 **'logits_mask'**，则视为不做masking，或者说相应的mask全为1
* 如果不做中间层特征匹配，可不提供 **'attention'** 或 **'hidden'**
* 如果不想利用有标签数据上的损失，可令hard_label_weight=0，并忽略 **'losses'**
* 如果不提供 **'logits'**，会略去最后输出层的蒸馏损失的计算
* **'labels'** 仅在 probability_shift==True 时需要
* 当然也不能什么都没有，那就不会进行任何训练

**一般情况下，除非做multi-stage的训练，否则'logits' 是必须要有的。**


## Classes and functions

### Configurations

class **textbrewer.TrainingConfig** (**gradient_accumulation_steps** = 1, **ckpt_frequency** = 1, **ckpt_epoch_frequency**=1, **ckpt_steps** = None, **log_dir** = None, **output_dir** = './saved_models', **device** = 'cuda')

* **gradient_accumulation_steps** (`int`) : 梯度累加以节约显存。每计算 *gradient_accumulation_steps* 个batch的梯度，调用一次optimizer.step()。大于1时用于在大batch_size情况下节约显存。
* **ckpt_frequency** (`int`) : 存储模型权重的频率。每训练一个epoch储存模型权重的次数。
* **ckpt_epoch_frequency** (`int`)：每多少个epoch储存模型。
  * **ckpt_frequency**=1, **ckpt_epoch_frequency**=1 : 每个epoch结束时存一次 （默认行为）。
  * **ckpt_frequency**=2, **ckpt_epoch_frequency**=1 : 在每个epoch的一半和结束时，各存一次。
  * **ckpt_frequency**=1, **ckpt_epoch_frequency**=2 : 每两个epoch结束时，存一次。
  * **ckpt_frequency**=2, **ckpt_epoch_frequency**=2 : 每2个epoch，仅在第2个epoch的一半和结束时各存一次（一般不会这样设置）。
* **ckpt_steps** (`int`) : 每ckpt_steps步存一次模型，仅当调用distiller.train时指定了训练步数`num_steps`时有效，不会与ckpt_frequency以及ckpt_epoch_frequency同时起效。
* **log_dir** (`str`) : 存放tensorboard日志的位置。如果为None，不启用tensorboard。
* **output_dir** (`str`) : 储存模型权重的位置。
* **device** (`str`, `torch.device`) : 在CPU或GPU上训练。

示例：

```python
from textbrewer import TrainingConfig
#一般情况下，除了log_dir和output_dir自己设置，其他用默认值即可
train_config = TrainingConfig(log_dir=my_log_dir, output_dir=my_output_dir)
```

* (classmethod) **TrainingConfig.from_json_file**(json_file : `str`)
  * 从json文件读取配置

* (classmethod) **TrainingConfig.from_dict**(dict_object : `Dict`)
  * 从字典读取配置



class **textbrewer.DistillationConfig** (**temperature** = 4, **temperature_scheduler**='none', **hard_label_weight** = 0, **hard_label_weight_scheduler**='none', **kd_loss_type** = 'ce', **kd_loss_weight**=1, **kd_loss_weight_scheduler**='none', **probability_shift**=False, **intermediate_matches** = None)

* **temperature** (`float`) : 蒸馏的温度。计算logits上的损失时教师和学生的logits将除以temperature。
* **temperature_scheduler** (`str`): 动态温度调节。有效取值见[**presets**](#预定义列表与字典-(presets))下的**TEMPERATURE_SCHEDULER**。
* **kd_loss_weight** (`float`): 'logits'项上的kd_loss的权重。
* **hard_label_weight** (`float`) : 'losses'项的权重。一般来说'losses'项是ground-truth上的损失。

  若**hard_label_weight**>0，且在adaptor中提供了 'losses'，那么如下的损失将被包含于最终的total loss:

  ```
  kd_loss_weight * kd_loss + hard_label_weight * sum(losses)
  ```

* **kd_loss_weight_scheduler** (`str`) 和 **hard_label_weight_scheduler**(str): 动态损失权重调节。有效取值见[**presets**](#预定义列表与字典-(presets))下的 **WEIGHT_SCHEDULER**

* **kd_loss_type** (`str`) : 模型最后输出的logits上的蒸馏损失函数。有效取值见[**presets**](#预定义列表与字典-(presets))下的 **KD_LOSS_MAP**。可用的有：
  
  * 'ce': 计算学生和教师的logits的交叉熵损失
  * 'mse':计算学生和教师的logits的mse损失
  
* **probability_shift** (`bool`): 是否启用probabliity shift 策略：交换教师模型预测的概率最高标签的logit和ground-truth标签的logit，使得ground-truth标签的logit总是最高。需要adaptor提供'labels'。

* **intermediate_matches** (`List[Dict]` or `None`) : 可选，模型中间层匹配损失的配置，list的每一个元素为一个字典，表示一对匹配配置。字典的key和value如下：
  
  * 'layer_T': layer_T (`int`):  选取教师的第layer_T层
  
  * 'layer_S': layer_S (`int`):  选取学生的第layer_S层
  
    **注意**：
    
    1. layer_T 和 layer_S选取的层数是adaptor返回的字典中的'attention'或'hidden'下的列表中元素的指标，不直接代表网络中实际的层数。
    2. layer_T 和 layer_S一般来说都为int。但计算FSP/NST loss时，需要分别选取teacher的两层和student的两层。因此当loss为[FSP](#fsp) 或[NST](#nst-(mmd))时，layer_T和layer_S是一个包含两个整数的列表，分别表示选取的teacher的两层和student的两层。可参见下文示例中的蒸馏配置
  
  * 'feature' : feature(`str`) : 中间层的特征名，有效取值见[**presets**](#预定义列表与字典 (presets))下的 **FEATURES** 。可为：
  
    * 'attention' : 表示attention矩阵，大小应为 (batch_size, num_heads, length,length) 或 (batch_size, length, length)
    * 'hidden'：表示隐层输出，大小应为 (batch_size, length, hidden_dim)
  
  * 'loss' : loss(`str`) :损失函数的具体形式，有效取值见[**presets**](#预定义列表与字典 (presets))下的**MATCH_LOSS_MAP**。常用的有：
  
    * 'attention_mse'
    * 'attention_ce'
    * 'hidden_mse'
    * 'nst'
    * ......
    
  * 'weight': weight (`float`) : 损失的权重
  
  * 'proj' : proj(`List`, optional) : 教师和学生的feature维度一样时，可选；不一样时，必选。为了匹配教师和学生中间层feature，所需要的投影函数，将学生侧的输入的维数转换为与教师侧相同。是一个列表，元素为：
  
    * proj[0] (`str`) :具体的转换函数，可取值有'linear'，'relu'，'tanh'。 见[**presets**](#预定义列表与字典 (presets))下的 **PROJ_MAP**
    * proj[1] (`int`):  转换函数的输入维度（学生侧的维度）
    * proj[2] (`int`):  转换函数的输出维度（教师侧的维度）
    * proj[3] (`Dict`): 可选，转换函数的学习率等优化相关配置字典。如果不提供，projection的学习率等优化器相关参数将采用optimzer的defaults配置，否则采用这里提供的参数。

  **示例：**

  ```python
  from textbrewer import DistillationConfig

  #最简单的配置：仅做最基本的蒸馏，用默认值即可，或尝试不同的temperature
  distill_config = DistillationConfig(temperature=8)

  #加入中间单层匹配的配置。
  #此配置下，adaptor_T/S返回的字典results_T/S要包含'hidden'键。
  #教师的 results_T['hidden'][10]和学生的results_S['hidden'][3]将计算hidden_mse loss
  distill_config = DistillationConfig(
    temperature = 8,
    intermediate_matches = [{'layer_T':10, 'layer_S':3, 'feature':'hidden','loss': 'hidden_mse', 'weight' : 1}]
  )

  #多层匹配。假设教师和学生的hidden dim分别为768和384，在学生和教师间增加投影（转换）函数
  distill_config = DistillationConfig(
    temperature = 8, 
    intermediate_matches = [ \
      {'layer_T':0, 'layer_S':0, 'feature':'hidden','loss': 'hidden_mse', 'weight' : 1,'proj':['linear',384,768]},
      {'layer_T':4, 'layer_S':1, 'feature':'hidden','loss': 'hidden_mse', 'weight' : 1,'proj':['linear',384,768]},
      {'layer_T':8, 'layer_S':2, 'feature':'hidden','loss': 'hidden_mse', 'weight' : 1,'proj':['linear',384,768]},
      {'layer_T':12, 'layer_S':3, 'feature':'hidden','loss': 'hidden_mse', 'weight' : 1,'proj':['linear',384,768]},
    {'layer_T':[0,0], 'layer_S':[0,0], 'feature':'hidden','loss': 'nst', 'weight' : 1},
    {'layer_T':[4,4], 'layer_S':[1,1], 'feature':'hidden','loss': 'nst', 'weight' : 1},
    {'layer_T':[8,8], 'layer_S':[2,2], 'feature':'hidden','loss': 'nst', 'weight' : 1},
    {'layer_T':[12,12],'layer_S':[3,3],'feature':'hidden','loss': 'nst', 'weight' : 1}]
  )
  ```

* (classmethod) **DistillationConfig.from_json_file**(json_file : `str`)
  * 从json文件读取配置

* (classmethod) **DistillationConfig.from_dict**(dict_object : `Dict`)
  * 从字典读取配置

### Distillers

初始化某个distiller后，调用其`train`方法开始训练/蒸馏。各个distiller的`train`方法的参数相同。

#### GeneralDistiller 

单模型单任务蒸馏推荐使用。

* class **textbrewer.GeneralDistiller** (**train_config**, **distill_config**, **model_T**, **model_S**, **adaptor_T**, **adaptor_S**, **custom_matches** = None)

  * train_config (`TrainingConfig`): 训练配置
  * distill_config (`DistillationConfig`)：蒸馏配置
  * model_T (`torch.nn.Module`)：教师模型
  * model_S (`torch.nn.Module`)：学生模型
  * adaptor_T (`Callable`, function)：教师模型的adaptor
  * adaptor_S (`Callable`, function)：学生模型的adaptor

    * adaptor (batch, model_outputs) -> Dict

    为适配不同模型的输入与输出，adaptor需要由用户提供。Adaptor接受两个输入，分别为batch(dataloader的输出)和model_outputs(模型的输出)，返回一个字典。
  * custom_matches (`List`) : 支持更灵活的特征匹配 (测试功能)

* **textbrewer.GeneralDistiller.train** (**optimizer**, **schduler**, **dataloader**, **num_epochs**, **num_steps**=None, **callback**=None, **batch_postprocessor**=None, **\*\*args**)
  * optimizer: 优化器
  * schduler: 调整学习率，可以为None
  * dataloader: 数据集迭代器
  * num_epochs (`int`): 训练的轮数
  * num_steps  (`int`): 指定训练的步数。当num_steps不为None时，distiller将忽略num_epochs而按num_steps设定的步数训练。此时不要求dataloader具有__len__属性，适用于数据集大小未知的情形。每当完成一次遍历，dataloader将被自动循环。
  * callback (`Callable`): 回调函数，可选。在每个checkpoint会被distiller调用，调用方式为callback(model=self.model_S, step = global_step)。可用于在每个checkpoint做evaluation。
  * batch_postprocessor (`Callable`): 函数，用于对batch做后处理，接受batch作为参数，返回处理后的batch。在distiller内部以如下方式调用：
    ```
    for batch in dataloader:
        batch = batch_postprocessor(batch)
        # check batch datatype
        # passes batch to the model and adaptors
    ```
  * \*\*args：额外的需要提供给模型的参数

调用模型过程说明：

* 如果batch是list或tuple，那么调用模型的形式为model(\*batch, \*\*args)。所以**请注意batch中各个字段的顺序和模型forward方法接受的顺序相匹配。**
* 如果batch是dict，那么调用模型的形式为model(\*\*batch,\*\*args)。所以**请注意batch中的key和模型接受参数名相匹配。**

#### BasicTrainer

进行有监督训练，而非蒸馏。可用于训练教师模型。

* class **textbrewer.BasicTrainer** (**train_config**, **model**, **adaptor**)
  * train_config (`TrainingConfig`): 训练配置
  * model (`torch.nn.Module`)：待训练的模型
  * adaptor (`Callable`, function)：待训练的模型的adaptor
* BasicTrainer.train 同 GeneralDistiller.train

#### BasicDistiller

用于单模型单任务蒸馏，**不支持中间层特征匹配**。可作为调试或测试使用。

* class **textbrewer.BasicDIstiller** (**train_config**, **distill_config**, **model_T**, **model_S**, **adaptor_T**, **adaptor_S**)
  * train_config (`TrainingConfig`): 训练配置
  * distill_config (`DistillationConfig`)：蒸馏配置
  * model_T (`torch.nn.Module`)：教师模型
  * model_S (`torch.nn.Module`)：学生模型
  * adaptor_T (`Callable`, function)：教师模型的adaptor
  * adaptor_S (`Callable`, function)：学生模型的adaptor
* BasicDistiller.train 同 GeneralDistiller.train

#### MultiTeacherDistiller

多教师蒸馏。将多个（同任务）教师模型蒸馏到一个学生模型上。**不支持中间层特征匹配**。

* class **textbrewer.MultiTeacherDistiller** (**train_config**, **distill_config**, **model_T**, **model_S**, **adaptor_T**, **adaptor_S**)

  * train_config (`TrainingConfig`): 训练配置
  * distill_config (`DistillationConfig`)：蒸馏配置
  * model_T (`List[torch.nn.Module]`)：教师模型的列表
  * model_S (`torch.nn.Module`)：学生模型
  * adaptor_T (`Callable`, function)：教师模型的adaptor
  * adaptor_S (`Callable`, function)：学生模型的adaptor

* MultiTeacherDistiller.train 同 GeneralDistiller.train

#### MultiTaskDistiller

多任务蒸馏，将多个（不同任务）的教师模型蒸馏到一个学生模型上。**它的参数形式不同于其他distiller**:

* class **textbrewer.MultiTaskDistiller** (**train_config**, **distill_config**, **model_T**, **model_S**, **adaptor_T**, **adaptor_S**)

  * train_config (`TrainingConfig`): 训练配置。因为MultiTaskDistiller按训练步数num_steps而不是训练轮数num_epochs执行训练过程，所以ckpt_steps必须被指定。
  * distill_config (`DistillationConfig`)：蒸馏配置
  * model_T (`Dict[str,torch.nn.Module]`)：教师模型的字典，key为任务名，value为模型
  * model_S (`torch.nn.Module`)：学生模型
  * adaptor_T (`Dict[str,Callable]`)：教师模型的adaptor字典，key为任务名，value为对应的adaptor
  * adaptor_S (`Dict[str,Callable]`)：学生模型的adaptor字典，key为任务名，value为对应的adaptor

* **textbrewer.MultiTaskDistiller.train** (**optimizer**, **schduler**, **dataloaders**, **num_steps**, **tau**=1, **callback**=None, **batch_postprocessors**=None, **\*\*args**)
  * optimizer: 优化器
  * schduler: 调整学习率，可以为None
  * dataloaders : 数据集迭代器字典，key为任务名，value为对应数据的dataloader
  * num_steps (`int`): 训练步数。每当完成一次遍历，dataloader将被自动循环。
  * tau (`float`): 训练样本来自任务d的的概率正比如|d|的tau次方，|d|是任务d的训练集大小。如果某一个dataloader的长度未知，那么tau无效，即以等概率从各个任务采样。
  * callback (`Callable`): 回调函数，可以为None。在每个checkpoint会被distiller调用，调用方式为callback(model=self.model_S, step = global_step)。可用于在每个checkpoint做evaluation
  * batch_postprocessors (`Dict[Callable]`):  batch_postprocessors的字典，key为任务名，value为对应任务数据的batch_postprocessor。用于对batch做后处理，每个batch_postprocessor接受batch作为参数，返回处理后的batch。在distiller内部以如下方式调用：
      ```python
      batch = next(dataloaders[taskname])
      # if batch_postprocessors is not None:
      batch = batch_postprocessors[taskname](batch)
      # check batch datatype
      # passes batch to the model and adaptors
    ```
  * \*\*args：额外的需要提供给模型的参数

### utils

* function **textbrewer.utils.display_parameters(model, max_level=None)**

  显示模型各个子模块的参数与内存占用量。

  * model (`torch.nn.Module`) : 待分析的模型
  * max_level(`int or None`): 显示到第max_level层次。如果max_level==None, 则显示所有层次的参数。

### data_utils

该模块提供了以下几种数据增强方法。

* function **textbrewer.data_utils.masking(tokens: List, p = 0.1, mask = '[MASK]')** -> List 

  返回以概率p用mask替换tokens中的元素后的列表。

* function **textbrewer.data_utils.deleting(tokens: List, p = 0.1)** -> List
  
  返回以概率p删除tokens中的元素后的列表。

* function **textbrewer.data_utils.n_gram_sampling(tokens: List, p_ng = [0.2,0.2,0.2,0.2,0.2], l_ng = [1,2,3,4,5])** -> List

  按概率p_ng从l_ng中采样长度l，随机截取tokens中长度为l的片段返回。

* function **textbrewer.data_utils.short_disorder(tokens: List, p = [0.9,0.1,0,0,0])** -> List
  
  返回以概率p对tokens的每个可能位置进行打乱后的列表。假设abc是tokens中连续的三个元素，那么五个概率值对应五种打乱方式:

  * abc -> abc
  * abc -> bac
  * abc -> cba
  * abc -> cab
  * abc -> bca

* function **textbrewer.data_utils.long_disorder(tokens: List, p = 0.1, length = 20)** -> List

  对tokens进行更长距离的打乱。如果length>1，则对tokens中每个长度为length的span，交换它们的前后一半；如果length<=1，则视length为相对于tokens的相对长度。例如：
  ```python
  long_disorder([0,1,2,3,4,5,6,7,8,9,10], p=1, length=0.4)
  # [2, 3, 0, 1, 6, 7, 4, 5, 8, 9]
  ```


## 预定义列表与字典 (presets)

Presets中包含了一些预先实现的损失函数和蒸馏策略等模块

* **textbrewer.presets.ADAPTOR_KEYS** (`List`)

  adaptor返回的字典用到的keys ：

  *  'logits', 'logits_mask', 'losses', 'inputs_mask', 'labels', 'hidden', 'attention'

* **textbrewer.presets.KD_LOSS_MAP** (`Dict`)

  可用的kd_loss种类

  * 'mse' : logits上的mse损失
  * 'ce': logits上的交叉熵损失

* **PROJ_MAP** (`Dict`)

  用于匹配不同的中间层特征维度的转换层

  * 'linear' : 线性变换，无激活函数
  * 'relu' : 激活函数为ReLU
  * 'tanh': 激活函数为Tanh

* **MATCH_LOSS_MAP** (`Dict`)

  中间层特征损失函数

  * 包含 'attention_mse_sum', 'attention_mse', ‘attention_ce_mean', 'attention_ce', 'hidden_mse', 'cos', 'pkd', 'fsp', 'nst'，细节参见 [中间特征损失函数](#中间特征损失函数)

* **WEIGHT_SCHEDULER** (`Dict`)

  用于动态调整kd_loss权重和hard_label_loss权重的scheduler

  * ‘linear_decay' : 从1衰减到0
  * 'linear_growth' : 从0增加到1

* **TEMPERATURE_SCHEDULER** (`DynamicDict`)

  用于动态调整蒸馏温度

  * 'constant' : 温度不变

  * 'flsw' :  具体参见 [Preparing Lessons: Improve Knowledge Distillation with Better Supervision](https://arxiv.org/abs/1911.07471)，使用此策略时需要提供两个参数beta和gamma
    
  * 'cwsm': 具体参见[Preparing Lessons: Improve Knowledge Distillation with Better Supervision](https://arxiv.org/abs/1911.07471)，使用此策略时需要提供一个参数beta

  与其他选项配置不同，使用’flsw'和'cwsm'时，应附带上需要的额外的参数，例如：

    ```python
  #flsw
  distill_config = DistillationConfig(
    temperature_scheduler = ['flsw', 1， 1]
    ...)
    
  #cwsm
  distill_config = DistillationConfig(
    temperature_scheduler = ['cwsm', 1]
    ...)
    ```

### 自定义

如果预设模块不能满足需求，可向上述字典中添加自定义的函数/模块，例如

```python
MATCH_LOSS_MAP['my_L1_loss'] = my_L1_loss
WEIGHT_SCHEDULER['my_weight_scheduler'] = my_weight_scheduler
```

在DistiilationConfig中使用：

```python
distill_config = DistillationConfig(
  kd_loss_weight_scheduler = 'my_weight_scheduler'
  intermediate_matches = [{'layer_T':0, 'layer_S':0, 'feature':'hidden','loss': 'my_L1_loss', 'weight' : 1}]
  ...)
  
```

函数的参数规范参见源代码（之后将在更详细的文档中给出）。



### 中间特征损失函数

#### attention_mse

* 接收两个形状为 (**batch_size, num_heads, len, len**)的矩阵，计算两个矩阵间的mse损失。
* 如果adaptor提供了inputs_mask，计算中会按inputs_mask遮掩对应位置。

#### attention_mse_sum

* 接收两个矩阵，如果形状为(**batch_size, len, len**) ，直接计算两个矩阵间的mse损失；如果形状为 (**batch_size, num_heads, len, len**)，将num_heads维度求和，再计算两个矩阵间的mse损失。
* 如果adaptor提供了inputs_mask，计算中会按inputs_mask遮掩对应位置。

#### attention_ce

* 接收两个形状为 (**batch_size, num_heads, len, len**)的矩阵，取dim=-1为softmax的维度，计算两个矩阵间的交叉熵损失。
* 如果adaptor提供了inputs_mask，计算中会按inputs_mask遮掩对应位置。

#### attention_ce_mean

* 接收两个矩阵，如果形状为(**batch_size, len, len**) ，直接计算两个矩阵间的交叉熵损失；如果形状为 (**batch_size, num_heads, len, len**)，将num_heads维度求平均，再计算两个矩阵间的交叉熵损失。计算方式同**attention_ce** 。
* 如果adaptor提供了inputs_mask，计算中会按inputs_mask遮掩对应位置。

#### hidden_mse

* 接收两个形状为 (**batch_size, len, hidden_size**)的矩阵，计算两个矩阵间的mse损失。
* 如果adaptor提供了inputs_mask，计算中会按inputs_mask遮掩对应位置。
* 用于蒸馏时，如果student和teacher的hidden size不同，需要在intermediate_matches中添加'proj'参数用以转换维度。

#### cos

* 接收两个形状为 (**batch_size, len, hidden_size**)的矩阵，计算它们间的余弦相似度损失。
* 参考：[DistilBERT](https://arxiv.org/abs/1910.01108)
* 如果adaptor提供了inputs_mask，计算中会按inputs_mask遮掩对应位置。
* 用于蒸馏时，如果student和teacher的hidden size不同，需要在intermediate_matches中添加'proj'参数用以转换维度。

#### pkd

* 接收两个形状为 (**batch_size, len, hidden_size**)的矩阵，计算len维度上0位置上的归一化向量mse损失。
* 参考： [Patient Knowledge Distillation for BERT Model Compression](https://arxiv.org/abs/1908.09355)
* 用于蒸馏时，如果student和teacher的hidden size不同，需要在intermediate_matches中添加'proj'参数用以转换维度。

#### nst (mmd)

* 接收两个矩阵列表A和B，每个列表中包含两个形状为(**batch_size, len, hidden_size**)的矩阵。A中的矩阵的hidden_size和B中矩阵的hidden_size不必相同。计算A中的两个矩阵的相似度矩阵 ( (**batch_size, len, len**) ) 和B中的两个矩阵的相似度矩阵  ( (**batch_size, len, len**) ) 的mse损失。
* 参考：[Like What You Like: Knowledge Distill via Neuron Selectivity Transfer](https://arxiv.org/abs/1707.01219)
* 如果adaptor提供了inputs_mask，计算中会按inputs_mask遮掩对应位置。

#### fsp

* 接收两个矩阵列表A和B，每个列表中包含两个形状为(**batch_size, len, hidden_size**)的矩阵。计算A中的两个矩阵的相似度矩阵 ( (**batch_size, hidden_size, hidden_size**) ) 和B中的两个矩阵的相似度矩阵  ( (**batch_size, hidden_size, hidden_size**) ) 的mse损失。

* 参考：[A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf)
* 如果adaptor提供了inputs_mask，计算中会按inputs_mask遮掩对应位置。
* 用于蒸馏时，如果student和teacher的hidden size不同，需要添加'proj'参数用以转换维度：

  ```python
    intermediate_matches = [
    {'layer_T':[0,0], 'layer_S':[0,0], 'feature':'hidden','loss': 'fsp', 'weight' : 1, 'proj':['linear',384,768]},
    ...]
  ```