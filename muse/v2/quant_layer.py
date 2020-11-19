import functools

import torch
import torch.nn as nn

# https://realpython.com/primer-on-python-decorators/#creating-singletons


def _singleton(cls):
    """Make a class a Singleton class (only one instance)"""
    @functools.wraps(cls)
    def wrapper_singleton(*args, **kwargs):
        if not wrapper_singleton.instance:
            wrapper_singleton.instance = cls(*args, **kwargs)
        return wrapper_singleton.instance
    wrapper_singleton.instance = None
    return wrapper_singleton


@_singleton
class _SingletonParameter:
    """ Use singleton instead of class attribute,
    so that some layers can have individual alpha
    """

    def __init__(self, t):
        self._param = nn.Parameter(torch.tensor(t))

    @property
    def param(self):
        return self._param


class QuantLayer(nn.Module):
    def __init__(self, alpha=10.0, share=True):
        super(QuantLayer, self).__init__()
        if share:
            self.alpha = _SingletonParameter(alpha).param
            assert self.alpha.item() == alpha, (
                "When sharing alpha, init value should be the same.\n"
                f"Previous={self.alpha.item()} vs. Now={alpha}"
            )
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, input):
        return 0.5 * (input.add(0.5*self.alpha).abs() - input.sub(0.5*self.alpha).abs())

    def extra_repr(self) -> str:
        return f'alpha={self.alpha}'

    @property
    def exp(self):
        ''' q = bits - exp '''
        return torch.log2(torch.as_tensor(0.5*self.alpha)).ceil()
