

import torch
from pprint import pprint
import gc
from torch.cuda.profiler import save_memory_flamegraph, enable_memory_history

enable_memory_history()

from torchvision.models import resnet18

rn = resnet18().cuda()

rn.eval()

input = torch.rand(1, 3, 224, 224).cuda()
with torch.no_grad():
    for i in range(10):
        r = rn(input)

save_memory_flamegraph('memory.txt')
