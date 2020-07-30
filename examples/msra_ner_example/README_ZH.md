[**中文说明**](README_ZH.md) | [**English**](README.md)

这个例子展示MSRA NER(中文命名实体识别)任务上，在**分布式数据并行训练**(Distributed Data-Parallel, DDP)模式(single node, muliti-GPU)下的[Chinese-ELECTRA-base](https://github.com/ymcui/Chinese-ELECTRA)模型蒸馏。


* ner_ElectraTrain_dist.sh : 训练教师模型(ELECTRA-base)。
* ner_ElectraDistill_dist.sh : 将教师模型蒸馏到学生模型(ELECTRA-small)。


运行脚本前，请根据自己的环境设置相应变量：

* ELECTRA_DIR_BASE :  存放Chinese-ELECTRA-base模型的目录，包含vocab.txt，pytorch_model.bin和config.json。

* OUTPUT_DIR : 存放训练好的模型权重文件和日志。
* DATA_DIR : MSRA NER数据集目录，包含
  * msra_train_bio.txt
  * msra_test_bio.txt

对于蒸馏，需要设置:

* ELECTRA_DIR_SMALL :  Chinese-ELECTRA-small预训练权重所在目录。应包含pytorch_model.bin。 也可不提供预训练权重，则学生模型将随机初始化。
* student_config_file : 学生模型配置文件，一般文件名为config.json，也位于 $\{ELECTRA_DIR_SMALL\}。
* trained_teacher_model_file : 在MSRA NER任务上训练好的ELECTRA-base教师模型。

该脚本在 **PyTorch==1.2, Transformers==2.8** 下测试通过。