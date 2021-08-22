import torch
import sys
from torch.jit.mobile import _load_for_lite_interpreter
import time
m = _load_for_lite_interpreter('qihan_model_false.pt')
m2 = _load_for_lite_interpreter('qihan_model_true.pt')

inp = torch.randn(1, 28*28)

print(m(inp), m2(inp))
print(m(inp) - m2(inp))
