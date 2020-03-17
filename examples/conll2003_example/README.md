[**中文说明**](README_ZH.md) | [**English**](README.md)

This example demonstrates distillation on CoNLL-2003 English NER task.

* run_conll2003_train.sh : trains a treacher model (BERT-base-cased) on CoNLL-2003.
* run_conll2003_distill_T3.sh : distills the teacher to T3.

Requirements

* Transformers
* seqeval

Set the following variables in the shell scripts before running:

* BERT_MODEL : this is where BERT-base-cased stores, including vocab.txt, pytorch_model.bin, config.json
* OUTPUT_DIR : this directory stores model weights
* BERT_MODEL_TEACHER : this directory stores the trained teacher model weights (for distillation).
* data : this directory includes CoNLL-2003 dataset (contains train.txt, dev.txt and test.txt)

This example contains：

* Teacher Model training : ./run_conll2003_train.sh
* Distillation to student model T3 : ./run_conll2003_distill_T3.sh
