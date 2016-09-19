from collections import OrderedDict
import string
import torch
from .module import Module

def _addindent(s_, numSpaces):
    s = string.split(s_, '\n')
    # dont do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = string.join(s, '\n')
    s = first + '\n' + s
    return s

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
    You can do this with the add_module function 
    or by assigning them as Container attributes.

    ```python
    # one can add modules to the container after construction
    model.add_module('pool1', nn.MaxPool2d(2, 2))

    # one can also set modules as attributes of the container
    model.conv1 = nn.Conv2d(12, 24, 3)
    ```

    The container has some important additional methods: 
    
    **`[generator] parameters()`**
    
    returns a generator over all learnable parameters in the container instance. 
    This can typically be passed to the optimizer API

    ```python
    # .parameters()
    >>> for param in model.parameters():
    >>>     print(type(param.data), param.size())
    <class 'torch.FloatTensor'> (20L,)
    <class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)
    ```
    
    **`[dict] parameter_dict()`**
    
    returns a dictionary of learnable parameters of the Container.
    For example: ['conv1.weight' : Parameter(torch.FloatTensor(20x1x5x5)),
                  'conv1.bias'   : Parameter(torch.FloatTensor(20)),
                 ]

    ```python
    # .parameter_dict()
    >>> pdict = model.parameter_dict()
    >>> print(pdict.keys())
    ['conv1.bias', 'conv1.weight']
    ```


    **`load_parameter_dict(dict)`**

    Given a parameter dict, sets the parameters of self to be the given dict.
    It loads loads the parameters recursively.
    Excessive or non-matching parameter names are ignored.
    For example, the input dict has an entry 'conv44.weight', but 
    if the container does not have a module named 'conv44', then this entry is ignored.

    **`children()`**

    Returns a generator over all the children modules of self

    **`train()`**

    Sets the Container (and all it's child modules) to training mode (for modules such as batchnorm, dropout etc.)

    **`eval()`**

    Sets the Container (and all it's child modules) to evaluate mode (for modules such as batchnorm, dropout etc.)

    **`apply(closure)`**

    Applies the given closure to each parameter of the container. 


    **__Note: Apart from these, the container will define the base functions that it has derived from nn.Module __**
    

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

    def apply(self, fn):
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        for key, module in self._modules.items():
            modstr = module.__repr__()
            modstr = _addindent(modstr, 2)
            tmpstr = tmpstr + '  ('  + key + '): ' + modstr + '\n'
        tmpstr = tmpstr + ')'
        return tmpstr


class Sequential(Container):
    """A sequential Container. It is derived from the base nn.Container class
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, given is a small example.
    ```
    # Example of using Sequential
    model = nn.Sequential(
              nn.Conv2d(1,20,5),
              nn.ReLU(),
              nn.Conv2d(20,64,5),
              nn.ReLU()
            )

    # Example of using Sequential with OrderedDict
    model = nn.Sequential(OrderedDict([
              ('conv1', nn.Conv2d(1,20,5)),
              ('relu1', nn.ReLU()),
              ('conv2', nn.Conv2d(20,64,5)),
              ('relu2', nn.ReLU())
            ]))
     ```
    """

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

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        for key, module in self._modules.items():
            modstr = module.__repr__()
            modstr = _addindent(modstr, 2)
            tmpstr = tmpstr + '  ('  + key + '): ' + modstr + '\n'
        tmpstr = tmpstr + ')'
        return tmpstr
