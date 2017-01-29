import torch
from .Sum import Sum

"""

This file is still here because of backward compatibility.

Please use instead "nn.Sum(dimension, nInputDims, sizeAverage)"

"""


class Mean(Sum):

    def __init__(self, dimension):
        super(Mean, self).__init__(dimension, True)
