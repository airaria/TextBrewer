.. _intermediate_losses:

Intermediate Losses
===================
Here we list the definitions of pre-defined intermediate losses. 
Usually, users don't need to refer to these functions directly, but refer to them by the names in :obj:`MATCH_LOSS_MAP`.

attention_mse
-------------
.. autofunction:: textbrewer.losses.att_mse_loss

attention_mse_sum 
-----------------
.. autofunction:: textbrewer.losses.att_mse_sum_loss

attention_ce 
-----------------
.. autofunction:: textbrewer.losses.att_ce_loss

attention_ce_mean
-----------------
.. autofunction:: textbrewer.losses.att_ce_mean_loss

hidden_mse
----------
.. autofunction:: textbrewer.losses.hid_mse_loss

cos
---
.. autofunction:: textbrewer.losses.cos_loss

pkd
---
.. autofunction:: textbrewer.losses.pkd_loss

nst (mmd)
---------
.. autofunction:: textbrewer.losses.mmd_loss


fsp (gram)
----------
.. autofunction:: textbrewer.losses.fsp_loss