import torch
from torch.legacy import nn
from functools import wraps
import sys

class Container(nn.Module):

    def __init__(self, *args):
         super(Container, self).__init__(*args)
         self.modules = []

    def add(self, module):
        self.modules.append(module)
        return self

    def get(self, index):
        return self.modules[index]

    def size(self):
         return len(self.modules)

    def applyToModules(self, func):
         for module in self.modules:
             func(module)

    def zeroGradParameters(self):
        self.applyToModules(lambda m: m.zeroGradParameters())

    def updateParameters(self, learningRate):
        self.applyToModules(lambda m: m.updateParameters(learningRate))

    def training(self):
        self.applyToModules(lambda m: m.training())
        super(Container, self).training()

    def evaluate(self, ):
        self.applyToModules(lambda m: m.evaluate())
        super(Container, self).evaluate()

    def share(self, mlp, *args):
        for module, other_module in zip(self.modules, mlp.modules):
            module.share(other_module, *args)

    def reset(self, stdv=None):
        self.applyToModules(lambda m: m.reset(stdv))

    def parameters(self):
         w = []
         gw = []
         for module in self.modules:
             mparam = module.parameters()
             if mparam is not None:
                 w.extend(mparam[0])
                 gw.extend(mparam[1])
         if not w:
             return
         return w, gw

    def clearState(self):
        nn.utils.clear('output')
        nn.utils.clear('gradInput')
        for module in self.modules:
            module.clearState()
        return self

