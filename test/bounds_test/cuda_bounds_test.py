#!/usr/bin/env python
import torch
import sys
idx = 0 if len(sys.argv) == 1 else int(sys.argv[1])
x = torch.rand(10).cuda()
print(f'x.shape={x.shape}')
try:
  y = x[torch.tensor([idx])]
  print(f'x[torch.tensor([{idx})]={y}')
except RuntimeError as err:
  print("Error:", err)
  sys.exit(-1)
