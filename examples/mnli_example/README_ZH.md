[**中文说明**](README_ZH.md) | [**English**](README.md)

这个例子展示MNLI句对分类任务上的蒸馏，同时提供了一个**自定义distiller**的例子。

* run_mnli_train.sh : 在MNLI数据上训练教师模型(bert-base)。
* run_mnli_distill_T4tiny.sh : 在MNLI上蒸馏教师模型到T4Tiny。
* run_mnli_distill_T4tiny_emd.sh：使用EMD方法自动计算隐层与隐层的匹配，而无需人工指定。该例子同时展示了如何自定义distiller（见下文详解）。
* run_mnli_distill_multiteacher.sh : 多教师蒸馏，将多个教师模型压缩到一个学生模型。

**PyTorch==1.2.0，transformers==3.0.2**  上测试通过。

## 运行

1. 运行以上任一个脚本前，请根据自己的环境设置sh文件中相应变量：


* OUTPUT_ROOT_DIR : 存放训练好的模型和日志
* DATA_ROOT_DIR : 包含MNLI数据集:
  * \$\{DATA_ROOT_DIR\}/MNLI/train.tsv
  * \$\{DATA_ROOT_DIR\}/MNLI/dev_matched.tsv
  * \$\{DATA_ROOT_DIR\}/MNLI/dev_mismatched.tsv

2. 设置BERT模型路径：
   * 如果运行run_mnli_train.sh，修改jsons/TrainBertTeacher.json中"student"键下的"vocab_file","config_file"和"checkpoint"路径
   * 如果运行 run_mnli_distill_T4tiny.sh 或 run_mnli_distill_T4tiny_emd.sh，修改jsons/DistillBertToTiny.json中"teachers"键下的"vocab_file","config_file"和"checkpoint"路径
   * 如果运行 run_mnli_distill_multiteacher.sh, 修改jsons/DistillMultiBert.json中"teachers"键下的所有"vocab_file","config_file"和"checkpoint"路径。可以自行添加更多teacher。

3. 设置完成，执行sh文件开始训练。

## BERT-EMD与自定义distiller
[BERT-EMD](https://www.aclweb.org/anthology/2020.emnlp-main.242/) 通过优化中间层之间的Earth Mvoer's Distance以自适应地调整教师与学生之间中间层匹配。

我们参照了其[原始实现](https://github.com/lxk00/BERT-EMD)，并以distiller的形式实现了其一个简化版本EMDDistiller（忽略了attention间的mapping）。
BERT-EMD相关代码位于distiller_emd.py。EMDDistiller使用方法与其他distiller无太大差异：
```python
from distiller_emd import EMDDistiller
distiller = EMDDistiller(...)
with distiller:
    distiller.train(...)
```
使用方式详见 main.emd.py。

EMDDistiller要求pyemd包：
```bash
pip install pyemd
```
