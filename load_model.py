import torch
import sys
from torch.jit.mobile import _load_for_lite_interpreter
import time
start = time.time()
m = _load_for_lite_interpreter(sys.argv[1])
end = time.time()
print('Total time measured in Python: ', end - start, 'seconds')
