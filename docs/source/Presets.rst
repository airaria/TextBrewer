Presets
=======


Presets include module variables that define pre-defined loss functions and strategies.

Module variables
----------------

ADAPTOR_KEYS
^^^^^^^^^^^^
.. autodata:: textbrewer.presets.ADAPTOR_KEYS
    :annotation:

KD_LOSS_MAP
^^^^^^^^^^^^
.. autodata:: textbrewer.presets.KD_LOSS_MAP
    :annotation:

PROJ_MAP
^^^^^^^^
.. autodata:: textbrewer.presets.PROJ_MAP
    :annotation:

MATCH_LOSS_MAP
^^^^^^^^^^^^^^
.. autodata:: textbrewer.presets.MATCH_LOSS_MAP
    :annotation:

WEIGHT_SCHEDULER
^^^^^^^^^^^^^^^^
.. autodata:: textbrewer.presets.WEIGHT_SCHEDULER
    :annotation:

TEMPERATURE_SCHEDULER
^^^^^^^^^^^^^^^^^^^^^
.. autodata:: textbrewer.presets.TEMPERATURE_SCHEDULER
    :annotation:

Customization
-------------

If the pre-defined modules do not satisfy your requirements, you can add your own defined modules to the above dict. 

For example::

    MATCH_LOSS_MAP['my_L1_loss'] = my_L1_loss
    WEIGHT_SCHEDULER['my_weight_scheduler'] = my_weight_scheduler

then used in :class:`~textbrewer.DistillationConfig`::

    distill_config = DistillationConfig(
    kd_loss_weight_scheduler = 'my_weight_scheduler'
    intermediate_matches = [{'layer_T':0, 'layer_S':0, 'feature':'hidden','loss': 'my_L1_loss', 'weight' : 1}]
    ...)

Refer to the source code for more details on inputs and outputs conventions (will be explained in detail in a later version of the documentation).