[**中文说明**](README_ZH.md) | [**English**](README.md)

This example demonstrates distilltion on CMRC2018 task, and using DRCD dataset as data augmentation.


* run_cmrc2018_train.sh : trains a treacher model (roberta-wwm-base) on CMRC2018.
* run_cmrc2018_distill_T3.sh : distills the teacher to T3 with CMRC2018 and DRCD datasets.
* run_cmrc2018_distill_T4tiny.sh :  distills the teacher to T4tiny with CMRC2018 and DRCD datasets.

Modify the following variables in the shell scripts before running:

* BERT_DIR :  where RoBERTa-wwm-base stores，including vocab.txt, pytorch_model.bin, bert_config.json
* OUTPUT_ROOT_DIR : this directory stores logs and trained model weights
* DATA_ROOT_DIR : it includes CMRC2018 and DRCD datasets:
  * \$\{DATA_ROOT_DIR\}/cmrc2018/squad-style-data/cmrc2018_train.json
  * \$\{DATA_ROOT_DIR\}/cmrc2018/squad-style-data/cmrc2018_dev.json
  * \$\{DATA_ROOT_DIR\}/drcd/DRCD_training.json
* The trained teacher weights file *trained_teacher_model* has to be specified if running run_cmrc2018_distill_T3.sh or run_cmrc2018_distill_T4tiny.sh.
