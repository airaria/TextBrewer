[**中文说明**](README_ZH.md) | [**English**](README.md)

这个例子展示CoNLL-2003英文NER任务上的蒸馏。

* run_conll2003_train.sh : 在CoNLL-2003英文NER数据集上训练教师模型(BERT-base-cased)
* run_conll2003_distill_T3.sh : 在CoNLL-2003英文NER数据集上蒸馏教师模型到T3(三层BERT)

运行要求

* Transformers
* seqeval

运行脚本前，请根据自己的环境修改相应变量：

* BERT_MODEL : 存放BERT-base模型的目录，包含vocab.txt, pytorch_model.bin, config.json
* OUTPUT_DIR : 存放训练好的模型权重文件
* BERT_MODEL_TEACHER : 存放训练好的teacher模型的目录, 包含vocab.txt, pytorch_model.bin, config.json
* data : 包含CoNLL-2003数据集(包含train.txt,dev.txt,test.txt 三个文件)

示例包含两个部分：

* 训练教师模型：./run_conll2003_train.sh
* 蒸馏三层BERT学生模型：./run_conll2003_distill_T3.sh
