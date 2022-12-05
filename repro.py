#!/bin/bash python3

import torch
torch.use_deterministic_algorithms(True)
x = torch.zeros((5,4), device=torch.device('cuda:0'))
x[[False,True,False,True,True]] = torch.tensor([1.0, 1.0, 1.0, 1.0], device=torch.device('cuda:0'), dtype=torch.float32)