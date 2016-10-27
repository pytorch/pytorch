import os
import warnings
import difflib
import inspect
from collections import OrderedDict

import torch
from .module import Module

class SourceChangeWarning(Warning):
    pass

class Container(Module):
    """This is the base container class for all neural networks you would define.
    You will subclass your container from this class.
    In the constructor you define the modules that you would want to use,
    and in the "forward" function you use the constructed modules in
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
        def forward(self, input):
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

    dump_patches = False

    def __init__(self, **kwargs):
        super(Container, self).__init__()
        self._modules = OrderedDict()
        for key, value in kwargs.items():
            self.add_module(key, value)

    def add_module(self, name, module):
        if hasattr(self, name):
            raise KeyError("attribute already exists '{}'".format(name))
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(
                torch.typename(module)))
        self._modules[name] = module

    def __getattr__(self, name):
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        return Module.__getattr__(self, name)

    def __setattr__(self, name, value):
        _modules = self.__dict__.get('_modules')
        if isinstance(value, Module):
            if _modules is None:
                raise AttributeError(
                    "cannot assign module before Container.__init__() call")
            _modules[name] = value
        elif _modules is not None and name in _modules:
            if value is not None:
                raise TypeError("cannot assign '{}' as child module '{}' "
                                "(torch.nn.Module or None expected)"
                                 .format(torch.typename(value), name))
            _modules[name] = value
        else:
            Module.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._modules:
            del self._modules[name]
        else:
            Module.__delattr__(self, name)

    def parameter_dict(self, destination=None, prefix=''):
        result = super(Container, self).parameter_dict(destination, prefix)
        for name, module in self._modules.items():
            if module is not None:
                module.parameter_dict(result, prefix + name + '.')
        return result

    def load_parameter_dict(self, param_dict):
        super(Container, self).load_parameter_dict(param_dict)
        for name, module in self._modules.items():
            if module is not None:
                filtered_params = {param_name[len(name)+1:]: param
                        for param_name, param in param_dict.items()
                        if param_name.startswith(name)}
                module.load_parameter_dict(filtered_params)

    def parameters(self, memo=None):
        if memo is None:
            memo = set()
        for p in super(Container, self).parameters(memo):
            yield p
        for module in self.children():
            for p in module.parameters(memo):
                yield p

    def children(self):
        memo = set()
        for module in self._modules.values():
            if module is not None and module not in memo:
                memo.add(module)
                yield module

    def modules(self, memo=None):
        if memo is None:
            memo = set()
        if self not in memo:
            for m in super(Container, self).modules(memo):
                yield m
            for module in self.children():
                for m in module.modules(memo):
                    yield m

    def train(self):
        super(Container, self).train()
        for module in self.children():
            module.train()
        return self

    def eval(self):
        super(Container, self).eval()
        for module in self.children():
            module.eval()
        return self

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)
        return super(Container, self)._apply(fn)

    def __getstate__(self):
        dump_source = type(self) != Container
        if dump_source:
            self.source_file = inspect.getsourcefile(type(self))
            self.source = inspect.getsource(type(self))
        return self.__dict__

    def __setstate__(self, state):
        if 'source' in state:
            original_source = state['source']
            current_source = inspect.getsource(type(self))
            if original_source != current_source:
                if self.dump_patches:
                    file_name = type(self).__name__ + '.patch'
                    diff = difflib.unified_diff(
                            current_source.split('\n'),
                            original_source.split('\n'),
                            state['source_file'],
                            state['source_file'], lineterm="")
                    lines = '\n'.join(diff)
                    try:
                        with open(file_name, 'a+') as f:
                            file_size = f.seek(0, 2)
                            f.seek(0)
                            if file_size == 0:
                                f.write(lines)
                            elif file_size != len(lines) or f.read() != lines:
                                raise IOError
                        msg = ("Saved a reverse patch to " + file_name + ". "
                            "Run `patch -p0 <" + file_name + "` to revert your "
                            "changes.")
                    except IOError as e:
                        msg = ("Tried to save a patch, but couldn't create a "
                            "writable file " + file_name + ". Make sure it "
                            "doesn't exist and your working directory is "
                            "writable.")
                else:
                    msg = ("you can retrieve the original source code by "
                        "accessing the object's source attribute or set "
                        "torch.nn.Container.dump_patches to True and use the "
                        "patch tool to revert the changes.")
                warnings.warn("source code of class " + torch.typename(self) +
                        " has changed. " + msg, SourceChangeWarning)
        self.__dict__.update(state)


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
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input
