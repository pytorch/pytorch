
import torch

import copy
import lazy_tensor_core
import lazy_tensor_core.debug.metrics as metrics
import torch.optim as optim
import lazy_tensor_core.core.lazy_model as ltm

lazy_tensor_core._LAZYC._ltc_init_ts_backend()


dev = 'lazy'
x = torch.rand(2, 2, device=dev)
x_relu = torch.ops.lazy_cuda.lazy_custom_relu(x)
print (x_relu.cpu())