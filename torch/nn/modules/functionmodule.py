from types import FunctionType
from functools import partial
from inspect import signature
from .module import Module

class FunctionModule(Module):
    def __init__(self, fun, *args, **kwargs):
        super(FunctionModule, self).__init__()
        if not isinstance(fun, FunctionType):
            raise TypeError("FunctionModule argument type should be a function ;"
                            " is " + type(lambda_fun).__name__")
        self.fun = partial(fun, *args, **kwargs)
        
    def __repr__(self):
        name = self.fun.__name__
        sig_str = signature(self.fun).__str__()
        return 'Callable {0}, signature {1}'.format(name, sig_str)
    
    def forward(self, *args, **kwargs):
        return self.fun(*args, **kwargs)
