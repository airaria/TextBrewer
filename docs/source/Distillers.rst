.. _distillers:

Distillers
===========

Distillers perform the actual experiments.

Initialize a distiller object, call its `train` method to start training/distillation. 

BasicDistiller
---------------

.. autoclass:: textbrewer.BasicDistiller
    :members: train

GeneralDistiller
-----------------

.. autoclass:: textbrewer.GeneralDistiller
    :members: train

MultiTeacherDistiller
---------------------

.. autoclass:: textbrewer.MultiTeacherDistiller
    
    .. method:: train(self, optimizer, scheduler, dataloader, num_epochs, num_steps=None, callback=None, batch_postprocessor=None, **args)

        trains the student model. See :meth:`BasicDistiller.train`.

MultiTaskDistiller
------------------
.. autoclass:: textbrewer.MultiTaskDistiller
    :members: train

BasicTrainer
------------
.. autoclass:: textbrewer.BasicTrainer
    :members: train