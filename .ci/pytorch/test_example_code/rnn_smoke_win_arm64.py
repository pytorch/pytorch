r"""
It's used to check basic rnn features with cpu-only.
For example, it would throw exception if missing some components are missing
"""

import torch
import torch.nn as nn


rnn = nn.RNN(10, 20, 2)
inputs = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(inputs, h0)
