import torch
import lazy_tensor_core
lazy_tensor_core._LAZYC._ltc_init_ts_backend()
import lazy_tensor_core.core.lazy_model as ltm
from lazy_tensor_core.core import lazy_tensor

t = torch.rand(2, 2, device="lazy")
print(t.size())
print(type(t.sum()).__name__) # b)
print(type(t).__name__)       # c)
