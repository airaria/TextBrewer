[**中文说明**](README_ZH.md) | [**English**](README.md)

这个例子展示MNLI句对分类任务上的蒸馏。GLUE中的其他任务的蒸馏也类似。

* run_mnli_train.sh : 在MNLI数据上训练教师模型(bert-base-cased)
* run_mnli_distill_T4tiny.sh : 在MNLI上蒸馏教师模型到T4Tiny
* run_mnli_distill_multiteacher.sh : 执行多教师蒸馏，将多个教师模型压缩到一个学生模型

运行脚本前，请根据自己的环境修改相应变量：

* BERT_DIR : 存放BERT-base-cased模型的目录，包含vocab.txt, pytorch_model.bin, bert_config.json
* OUTPUT_ROOT_DIR : 存放训练好的模型和日志
* DATA_ROOT_DIR : 包含MNLI数据集:
  * \$\{DATA_ROOT_DIR\}/MNLI/train.tsv
  * \$\{DATA_ROOT_DIR\}/MNLI/dev_matched.tsv
  * \$\{DATA_ROOT_DIR\}/MNLI/dev_mismatched.tsv
* 如果是运行 run_mnli_distill_T4tiny.sh, 还需要指定训练好的教师模型权重文件 trained_teacher_model
* 如果是运行 run_mnli_distill_multiteacher.sh, 需要指定多个训练好的教师模型权重文件 trained_teacher_model_1, trained_teacher_model_2, trained_teacher_model_3
