from torch.autograd import Variable
from .module import Module
from collections import OrderedDict


class Container(Module):

    def __init__(self, **kwargs):
        super(Container, self).__init__()
        self.modules = []
        for key, value in kwargs.items():
            self._assign_module(key, value)

    def _assign_module(self, name, module):
        # TODO: error message
        assert not hasattr(self, name)
        setattr(self, name, module)
        self.modules.append(module)

    def parameters(self):
        for module in self.modules:
            for p in module.parameters():
                yield p


class Sequential(Container):

    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self._assign_module(key, module)
        else:
            idx = 0
            for module in args:
                self._assign_module(str(idx), module)
                idx += 1

    def __getitem__(self, idx):
        return self.modules[idx]

    def _forward(self, input):
        for module in self.modules:
            input = module(input)
        return (input,)
