from collections import OrderedDict

import torch
from torch.autograd import Variable
from .module import Module


class Container(Module):
    """This is the base container class for all neural networks you would define.
    You will subclass your container from this class.
    In the constructor you define the modules that you would want to use,
    and in the __call__ function you use the constructed modules in
    your operations.

    To make it easier to understand, given is a small example.
    ```
    # Example of using Container
     class Net(nn.Container):
        def __init__(self):
            super(Net, self).__init__(
                conv1 = nn.Conv2d(1, 20, 5),
                relu  = nn.ReLU()
             )
        def __call__(self, input):
            output = self.relu(self.conv1(x))
            return output
     model = Net()
     ```

    One can also add new modules to a container after construction.
    You can do this with the add_module function.

    ```
    # one can add modules to the container after construction
    model.add_module('pool1', nn.MaxPool2d(2, 2))
    ```

    The container has one additional method `parameters()` which
    returns the list of learnable parameters in the container instance.
    """
    def __init__(self, **kwargs):
        super(Container, self).__init__()
        self.modules = OrderedDict()
        for key, value in kwargs.items():
            self.add_module(key, value)

    def add_module(self, name, module):
        if hasattr(self, name):
            raise KeyError("attribute already exists '{}'".format(name))
        if not isinstance(module, Module) and module is not None:
            raise ValueError("{} is not a Module subclass".format(
                torch.typename(module)))
        self.modules[name] = module

    def __getattr__(self, name):
        if 'modules' in self.__dict__:
            modules = self.__dict__['modules']
            if name in modules:
                return modules[name]
        return Module.__getattr__(self, name)

    def parameters(self, memo=None):
        if memo is None:
            memo = set()
        super(Container, self).parameters(memo)
        for module in self.modules.values():
            for p in module.parameters(memo):
                yield p

    def type(self, type, *forwarded_args):
        for module in self.modules.values():
            module.type(type, *forwarded_args)
        return super(Container, self).type(type, *forwarded_args)


class Sequential(Container):

    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            idx = 0
            for module in args:
                self.add_module(str(idx), module)
                idx += 1

    def __getitem__(self, idx):
        if idx >= len(self.modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = self.modules.values()
        for i in range(idx-1):
            it.next()
        return it.next()

    def forward(self, input):
        for module in self.modules.values():
            input = module(input)
        return input

