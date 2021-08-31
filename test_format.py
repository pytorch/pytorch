import torch
import sys
from torch.jit.mobile import _load_for_lite_interpreter
import time

inp = torch.randn(1, 28*28)

start = time.time()
m2 = _load_for_lite_interpreter(sys.argv[1])
print(m2(inp))
end = time.time()
print('time :', end - start)
