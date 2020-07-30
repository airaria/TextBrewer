[**中文说明**](README_ZH.md) | [**English**](README.md)

This example demonstrates distilling a [Chinese-ELECTRA-base](https://github.com/ymcui/Chinese-ELECTRA) model on the MSRA NER task with **distributed data-parallel training**(single node, muliti-GPU).


* ner_ElectraTrain_dist.sh : trains a treacher model (Chinese-ELECTRA-base) on MSRA NER.
* ner_ElectraDistill_dist.sh : distills the teacher to a ELECTRA-small model.


Set the following variables in the shell scripts before running:

* ELECTRA_DIR_BASE :  where Chinese-ELECTRA-base locates, should includ vocab.txt, pytorch_model.bin and config.json.

* OUTPUT_DIR : this directory stores the logs and the trained model weights.
* DATA_DIR : it includes MSRA NER dataset:
  * msra_train_bio.txt
  * msra_test_bio.txt

For distillation:

* ELECTRA_DIR_SMALL :  where the pretrained Chinese-ELECTRA-small weight locates, should include pytorch_model.bin. This is optional. If you don't provide the ELECTRA-small weight, the student model will be initialized randomly.
* student_config_file : the model config file (i.e., config.json) for the student. Usually it should be in $\{ELECTRA_DIR_SMALL\}.
* trained_teacher_model_file : the ELECTRA-base teacher model that has been fine-tuned.

The scripts have been tested under **PyTorch==1.2, Transformers==2.8**.