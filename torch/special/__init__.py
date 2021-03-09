import sys

import torch
from torch._C import _add_docstr, _special  # type: ignore

Tensor = torch.Tensor

gammaln = _add_docstr(_special.special_gammaln,
                      r"""
gammaln(input, *, out=None) -> Tensor

Computes the natural logarithm of the absolute value of the gamma function on :attr:`input`.

.. math::
    \text{out}_{i} = \ln \Gamma(|\text{input}_{i}|)
""" + """
Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.arange(0.5, 2, 0.5)
    >>> torch.special.gammaln(a)
    tensor([ 0.5724,  0.0000, -0.1208])

""")

entr = _add_docstr(_special.special_entr,
                   r"""
entr(input, *, out=None) -> Tensor

Computes the entropy on :attr:`input`, elementwise.

.. math::
    \text{entr(x)} = \begin{cases}
        -x * \log(x)  & x > 0 \\
        0 &  x = 0.0 \\
        \infty & x < 0
    \end{cases}
""" + """
Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.arange(-0.5, 1, 0.5)
    >>> a
    tensor([-0.5000,  0.0000,  0.5000])
    >>> torch.special.entr(a)
    tensor([  -inf, 0.0000, 0.3466])
""")
