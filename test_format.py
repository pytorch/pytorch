import torch
import sys
from torch.jit.mobile import _load_for_lite_interpreter
import time
m = _load_for_lite_interpreter('qihan_model_false.pt')
m2 = _load_for_lite_interpreter('qihan_model_true.pt')

inp = torch.randn(1, 28*28)
print('========================')
print(m(inp))
print('========================')
print(m2(inp))

