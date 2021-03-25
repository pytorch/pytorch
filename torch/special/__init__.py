import sys

import torch
from torch._C import _add_docstr, _special  # type: ignore
from torch._torch_docs import common_args  # type: ignore

Tensor = torch.Tensor

entr = _add_docstr(_special.special_entr,
                   r"""
entr(input, *, out=None) -> Tensor
Computes the entropy on :attr:`input` (as defined below), elementwise.

.. math::
    \text{entr(x)} = \begin{cases}
        -x * \ln(x)  & x > 0 \\
        0 &  x = 0.0 \\
        -\infty & x < 0
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

gammaln = _add_docstr(_special.special_gammaln,
                      r"""
gammaln(input, *, out=None) -> Tensor

Computes the natural logarithm of the absolute value of the gamma function on :attr:`input`.

.. math::
    \text{out}_{i} = \ln \Gamma(|\text{input}_{i}|)
""" + """
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> a = torch.arange(0.5, 2, 0.5)
    >>> torch.special.gammaln(a)
    tensor([ 0.5724,  0.0000, -0.1208])

""".format(**common_args))

erf = _add_docstr(_special.special_erf,
                  r"""
erf(input, *, out=None) -> Tensor

Computes the error function of :attr:`input`. The error function is defined as follows:

.. math::
    \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt
""" + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> torch.erf(torch.tensor([0, -1., 10.]))
    tensor([ 0.0000, -0.8427,  1.0000])
""".format(**common_args))

erfc = _add_docstr(_special.special_erfc,
                   r"""
erfc(input, *, out=None) -> Tensor

Computes the complementary error function of :attr:`input`.
The complementary error function is defined as follows:

.. math::
    \mathrm{erfc}(x) = 1 - \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt
""" + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> torch.erfc(torch.tensor([0, -1., 10.]))
    tensor([ 1.0000, 1.8427,  0.0000])
""".format(**common_args))

erfinv = _add_docstr(_special.special_erfinv,
                     r"""
erfinv(input, *, out=None) -> Tensor

Computes the inverse error function of :attr:`input`.
The inverse error function is defined in the range :math:`(-1, 1)` as:

.. math::
    \mathrm{erfinv}(\mathrm{erf}(x)) = x
""" + r"""

Args:
    {input}

Keyword args:
    {out}

Example::

    >>> torch.erfinv(torch.tensor([0, 0.5, -1.]))
    tensor([ 0.0000,  0.4769,    -inf])
""".format(**common_args))

exp2 = _add_docstr(_special.special_exp2,
                   r"""
exp2(input, *, out=None) -> Tensor

Computes the base two exponential function of :attr:`input`.

.. math::
    y_{i} = 2^{x_{i}}

""" + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> torch.special.exp2(torch.tensor([0, math.log2(2.), 3, 4]))
    tensor([ 1.,  2.,  8., 16.])
""".format(**common_args))

expm1 = _add_docstr(_special.special_expm1,
                    r"""
expm1(input, *, out=None) -> Tensor

Computes the exponential of the elements minus 1
of :attr:`input`.

..

.. math::
    y_{i} = e^{x_{i}} - 1

.. note:: This function provides greater precision than exp(x) - 1 for small values of x.

""" + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> torch.expm1(torch.tensor([0, math.log(2.)]))
    tensor([ 0.,  1.])
""".format(**common_args))
