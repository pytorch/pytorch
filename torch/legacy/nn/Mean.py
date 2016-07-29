import torch
from torch.legacy import nn

"""

This file is still here because of backward compatibility.

Please use instead "nn.Sum(dimension, nInputDims, sizeAverage)"

"""

class Mean(nn.Sum):

    def __init__(self, dimension):
        super(Mean, self).__init__(dimension, True)

