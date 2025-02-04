r"""
It's used to check basic rnn features with cuda.
For example, it would throw exception if missing some components are missing
"""

import torch
import torch.nn as nn


device = torch.device("cuda:0")
rnn = nn.RNN(10, 20, 2).to(device)
inputs = torch.randn(5, 3, 10).to(device)
h0 = torch.randn(2, 3, 20).to(device)
output, hn = rnn(inputs, h0)
