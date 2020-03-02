# Simple Example

This runable example demonstrates the usage of TextBrewer.

Teacher is BERT-base. Student is a 3-layer BERT.

Task is text classification. We generate some random token ids and labels as inputs.

So this simple example is for pedagogical purpose only.

We also list the summarization of model parameters using the utility provided by the toolkit.


## Requirements

PyTorch >= 1.0

transformers >= 2.0

tensorboard

## Run
```python
python distill.py
```

## Screenshots

![screenshot1](screenshots/screenshot1.png)

![screenshot2](screenshots/screenshot2.png)