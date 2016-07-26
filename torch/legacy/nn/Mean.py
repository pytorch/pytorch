import torch
from torch.legacy import nn

"""

This file is still here because of backward compatibility.

Please use instead "nn.Sum(dimension, nInputDims, sizeAverage)"

"""

class Mean(nn.Sum):

    def __init__(self, dimension, nInputDims):
        super(Mean, self).__init__(self, dimension, nInputDims, True)

