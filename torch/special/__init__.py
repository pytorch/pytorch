import sys

import torch
from torch._C import _add_docstr, _special  # type: ignore

Tensor = torch.Tensor

gammaln = _add_docstr(_special.special_gammaln,
                     r"""
gammaln(input) -> Tensor

Computes the logarithm of the gamma function on :attr:`input`.

.. math::
    \text{out}_{i} = \log \Gamma(\text{input}_{i})
""" + """
Args:
    input (Tensor): the input tensor.

Example::

    >>> a = torch.arange(0.5, 2, 0.5)
    >>> torch.special.gammaln(a)
    tensor([ 0.5724,  0.0000, -0.1208])

""")
