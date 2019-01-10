import torch
from .Module import Module
from .Sequential import Sequential
from .Square import Square
from .Power import Power
from .SpatialAveragePooling import SpatialAveragePooling
from .MulConstant import MulConstant
from .Sqrt import Sqrt


class SpatialLPPooling(Sequential):

    def __init__(self, nInputPlane, pnorm, kW, kH, dW=None, dH=None):
        super(SpatialLPPooling, self).__init__()

        dW = dW or kW
        dH = dH or kH

        self.kW = kW
        self.kH = kH
        self.dW = dW
        self.dH = dH

        if pnorm == 2:
            self.add(Square())
        else:
            self.add(Power(pnorm))

        self.add(SpatialAveragePooling(kW, kH, dW, dH))
        self.add(MulConstant(kW * kH))
        if pnorm == 2:
            self.add(Sqrt())
        else:
            self.add(Power(1. / pnorm))

    # the module is a Sequential: by default, it'll try to learn the parameters
    # of the sub sampler: we avoid that by redefining its methods.
    def reset(self, stdev=None):
        pass

    def accGradParameters(self, input, gradOutput):
        pass

    def accUpdateGradParameters(self, input, gradOutput, lr):
        pass

    def zeroGradParameters(self):
        pass

    def updateParameters(self, learningRate):
        pass
