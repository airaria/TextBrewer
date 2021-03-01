[**中文说明**](README_ZH.md) | [**English**](README.md)

This example demonstrates distilltion on MNLI task and **how to write a new distiller**.

* run_mnli_train.sh : trains a teacher model (bert-base) on MNLI.
* run_mnli_distill_T4tiny.sh : distills the teacher to T4tiny.
* run_mnli_distill_T4tiny_emd.sh : distills the teacher to T4tiny with many-to-many intermediate matches using EMD, so there is no need to specifying the mathcing scheme. This example also demonstrates how to write a custom distiller (see below for details).
* run_mnli_distill_multiteacher.sh : runs multi-teacher distillation, distilling several teacher models into a student model.

Examples have been tested on **PyTorch==1.2.0, transformers==3.0.2**. 

## Run

1. Set the following variables in the bash scripts before running:

* OUTPUT_ROOT_DIR : this directory stores logs and trained model weights
* DATA_ROOT_DIR : it includes MNLI dataset:
  * \$\{DATA_ROOT_DIR\}/MNLI/train.tsv
  * \$\{DATA_ROOT_DIR\}/MNLI/dev_matched.tsv
  * \$\{DATA_ROOT_DIR\}/MNLI/dev_mismatched.tsv

2. Set the path to BERT:
   * If you are running run_mnli_train.sh: open jsons/TrainBertTeacher.json and set "vocab_file","config_file"和"checkpoint" which are under the key "student".
   * If you are running run_mnli_distill_T4tiny.sh or run_mnli_distill_T4tiny_emd.sh: open jsons/DistillBertToTiny.json and set "vocab_file", "config_file" and"checkpoint" which are under the key "teachers".
   * If you are running run_mnli_distill_multiteacher.sh: open jsons/DistillMultiBert.json and set all the "vocab_file","config_file" and "checkpoint" under the key "teachers". You can also add more teachers to the json.

3. Run the bash script and have fun.

## BERT-EMD and custom distiller
[BERT-EMD](https://www.aclweb.org/anthology/2020.emnlp-main.242/) allows each intermediate student layer to learn from any intermediate teacher layers adaptively, bassed on optimizing Earth Mover’s Distance. So there is no need to specify the mathcing scheme. 

Based on the [original implementation](https://github.com/lxk00/BERT-EMD), we have written a new distiller (EMDDistiller) to implement a simplified viersion of BERT-EMD (which ignores mappings between attentions). The code of the algorithm is in distiller_emd.py. The EMDDistiller is much like the other distillers:
```python
from distiller_emd import EMDDistiller
distiller = EMDDistiller(...)
with distiller:
    distiller.train(...)
```
see main.emd.py for detailed usages.

EMDDistiller requires pyemd package：
```bash
pip install pyemd
```
