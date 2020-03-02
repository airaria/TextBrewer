[**中文说明**](README_ZH.md) | [**English**](README.md)

This example demonstrates distilltion on MNLI task.

* run_mnli_train.sh : trains a teacher model (bert-base-cased) on MNLI.
* run_mnli_distill_T4tiny.sh : distills the teacher to T4tiny.
* run_mnli_distill_multiteacher.sh : runs multi-teacher distillation，distilling several teacher models into a student model.

Modify the following variables in the shell scripts before running:

* BERT_DIR : where BERT-base-cased stores，including vocab.txt, pytorch_model.bin, bert_config.json
* OUTPUT_ROOT_DIR : this directory stores logs and trained model weights
* DATA_ROOT_DIR : it includes MNLI dataset:
  * \$\{DATA_ROOT_DIR\}/MNLI/train.tsv
  * \$\{DATA_ROOT_DIR\}/MNLI/dev_matched.tsv
  * \$\{DATA_ROOT_DIR\}/MNLI/dev_mismatched.tsv
* The trained teacher weights file *trained_teacher_model* has to be specified if running run_mnli_distill_T4tiny.sh
* Multiple teacher weights file *trained_teacher_model_1, trained_teacher_model_2, trained_teacher_model_3* has to be specified if running run_mnli_distill_multiteacher.sh
