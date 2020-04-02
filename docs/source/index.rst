.. TextBrewer documentation master file, created by
   sphinx-quickstart on Tue Mar 17 17:23:13 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. image:: ../../pics/banner.png
  :width: 500
  :align: center

|

**TextBrewer** is a PyTorch-based model distillation toolkit for natural language processing.

It includes various distillation techniques from both NLP and CV field and provides an easy-to-use distillation framework, which allows users to quickly experiment with the state-of-the-art distillation methods to compress the model with a relatively small sacrifice in the performance, increasing the inference speed and reducing the memory usage.

Main features
-------------

* **Wide-support** : it supports various model architectures (especially **transformer**-based models).
* **Flexibility** : design your own distillation scheme by combining different techniques.
* **Easy-to-use** : users don't need to modify the model architectures.
* **Built for NLP** : it is suitable for a wide variety of NLP tasks: text classification, machine reading comprehension, sequence labeling, ...


Paper: `TextBrewer: An Open-Source Knowledge Distillation Toolkit for Natural Language Processing <https://arxiv.org/abs/2002.12620>`_


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   Tutorial
   Concepts

.. toctree::
   :maxdepth: 2
   :caption: Experiments

   Experiments

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   Configurations
   Distillers
   Presets
   Losses
   Utils

.. toctree::
   :maxdepth: 2
   :caption: Appendices

   ExperimentResults

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`