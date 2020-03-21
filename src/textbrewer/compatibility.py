import torch

if torch.__version__ < '1.2':
    mask_dtype = torch.uint8
else:
    mask_dtype = torch.bool