import torch
import os
print(torch.cuda.device_count())
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(torch.cuda.device_count())
print(torch.empty(10, device='cuda'))
