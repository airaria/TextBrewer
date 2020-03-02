[**中文说明**](README_ZH.md) | [**English**](README.md)

这个例子展示CMRC2018阅读理解任务上的蒸馏，并使用DRCD数据集作为数据增强。

* run_cmrc2018_train.sh : 在cmrc2018数据集上训练教师模型(roberta-wwm-base)
* run_cmrc2018_distill_T3.sh : 在cmrc2018和drcd数据集上蒸馏教师模型到T3
* run_cmrc2018_distill_T4tiny.sh : 在cmrc2018和drcd数据集上蒸馏教师模型到T4-tiny

运行脚本前，请根据自己的环境修改相应变量：

* BERT_DIR : 存放RoBERTa-wwm-base模型的目录，包含vocab.txt, pytorch_model.bin, bert_config.json
* OUTPUT_ROOT_DIR : 存放训练好的模型权重文件和日志
* DATA_ROOT_DIR : 包含cmrc2018数据集和drcd数据集:
  * \$\{DATA_ROOT_DIR\}/cmrc2018/squad-style-data/cmrc2018_train.json
  * \$\{DATA_ROOT_DIR\}/cmrc2018/squad-style-data/cmrc2018_dev.json
  * \$\{DATA_ROOT_DIR\}/drcd/DRCD_training.json
* 如果是运行 run_cmrc2018_distill_T3.sh 和 run_cmrc2018_distill_T4tiny.sh, 还需要指定训练好的教师模型权重文件 trained_teacher_model
