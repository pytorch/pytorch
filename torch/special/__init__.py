import sys

import torch
from torch._C import _add_docstr, _special  # type: ignore

Tensor = torch.Tensor

lgamma = _add_docstr(_special.special_lgamma, r"""
lgamma(input, *, out=None) -> Tensor

Computes the logarithm of the gamma function on :attr:`input`.

.. math::
    \text{out}_{i} = \log \Gamma(\text{input}_{i})
""" + """
Args:
    {input}
    {out}

Example::

    >>> a = torch.arange(0.5, 2, 0.5)
    >>> torch.lgamma(a)
    tensor([ 0.5724,  0.0000, -0.1208])

""")