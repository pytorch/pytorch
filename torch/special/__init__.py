import sys

import torch
from torch._C import _add_docstr, _special  # type: ignore[attr-defined]
from torch._torch_docs import common_args, multi_dim_common

Tensor = torch.Tensor

entr = _add_docstr(_special.special_entr,
                   r"""
entr(input, *, out=None) -> Tensor
Computes the entropy on :attr:`input` (as defined below), elementwise.

.. math::
    \begin{align}
    \text{entr(x)} = \begin{cases}
        -x * \ln(x)  & x > 0 \\
        0 &  x = 0.0 \\
        -\infty & x < 0
    \end{cases}
    \end{align}
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

psi = _add_docstr(_special.special_psi,
                  r"""
psi(input, *, out=None) -> Tensor

Alias for :func:`torch.special.digamma`.
""")

digamma = _add_docstr(_special.special_digamma,
                      r"""
digamma(input, *, out=None) -> Tensor

Computes the logarithmic derivative of the gamma function on `input`.

.. math::
    \digamma(x) = \frac{d}{dx} \ln\left(\Gamma\left(x\right)\right) = \frac{\Gamma'(x)}{\Gamma(x)}
""" + r"""
Args:
    input (Tensor): the tensor to compute the digamma function on

Keyword args:
    {out}

.. note::  This function is similar to SciPy's `scipy.special.digamma`.

.. note::  From PyTorch 1.8 onwards, the digamma function returns `-Inf` for `0`.
           Previously it returned `NaN` for `0`.

Example::

    >>> a = torch.tensor([1, 0.5])
    >>> torch.special.digamma(a)
    tensor([-0.5772, -1.9635])

""".format(**common_args))

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

    >>> torch.special.erf(torch.tensor([0, -1., 10.]))
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

    >>> torch.special.erfc(torch.tensor([0, -1., 10.]))
    tensor([ 1.0000, 1.8427,  0.0000])
""".format(**common_args))

erfcx = _add_docstr(_special.special_erfcx,
                    r"""
erfcx(input, *, out=None) -> Tensor

Computes the scaled complementary error function for each element of :attr:`input`.
The scaled complementary error function is defined as follows:

.. math::
    \mathrm{erfcx}(x) = e^{x^2} \mathrm{erfc}(x)
""" + r"""

""" + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> torch.special.erfcx(torch.tensor([0, -1., 10.]))
    tensor([ 1.0000, 5.0090, 0.0561])
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

    >>> torch.special.erfinv(torch.tensor([0, 0.5, -1.]))
    tensor([ 0.0000,  0.4769,    -inf])
""".format(**common_args))

logit = _add_docstr(_special.special_logit,
                    r"""
logit(input, eps=None, *, out=None) -> Tensor

Returns a new tensor with the logit of the elements of :attr:`input`.
:attr:`input` is clamped to [eps, 1 - eps] when eps is not None.
When eps is None and :attr:`input` < 0 or :attr:`input` > 1, the function will yields NaN.

.. math::
    \begin{align}
    y_{i} &= \ln(\frac{z_{i}}{1 - z_{i}}) \\
    z_{i} &= \begin{cases}
        x_{i} & \text{if eps is None} \\
        \text{eps} & \text{if } x_{i} < \text{eps} \\
        x_{i} & \text{if } \text{eps} \leq x_{i} \leq 1 - \text{eps} \\
        1 - \text{eps} & \text{if } x_{i} > 1 - \text{eps}
    \end{cases}
    \end{align}
""" + r"""
Args:
    {input}
    eps (float, optional): the epsilon for input clamp bound. Default: ``None``

Keyword args:
    {out}

Example::

    >>> a = torch.rand(5)
    >>> a
    tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
    >>> torch.special.logit(a, eps=1e-6)
    tensor([-0.9466,  2.6352,  0.6131, -1.7169,  0.6261])
""".format(**common_args))

logsumexp = _add_docstr(_special.special_logsumexp,
                        r"""
logsumexp(input, dim, keepdim=False, *, out=None)

Alias for :func:`torch.logsumexp`.
""".format(**multi_dim_common))

expit = _add_docstr(_special.special_expit,
                    r"""
expit(input, *, out=None) -> Tensor

Computes the expit (also known as the logistic sigmoid function) of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \frac{1}{1 + e^{-\text{input}_{i}}}
""" + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> t = torch.randn(4)
    >>> t
    tensor([ 0.9213,  1.0887, -0.8858, -1.7683])
    >>> torch.special.expit(t)
    tensor([ 0.7153,  0.7481,  0.2920,  0.1458])
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

.. math::
    y_{i} = e^{x_{i}} - 1

.. note:: This function provides greater precision than exp(x) - 1 for small values of x.

""" + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> torch.special.expm1(torch.tensor([0, math.log(2.)]))
    tensor([ 0.,  1.])
""".format(**common_args))

xlog1py = _add_docstr(_special.special_xlog1py,
                      r"""
xlog1py(input, other, *, out=None) -> Tensor

Computes ``input * log1p(other)`` with the following cases.

.. math::
    \text{out}_{i} = \begin{cases}
        \text{NaN} & \text{if } \text{other}_{i} = \text{NaN} \\
        0 & \text{if } \text{input}_{i} = 0.0 \text{ and } \text{other}_{i} != \text{NaN} \\
        \text{input}_{i} * \text{log1p}(\text{other}_{i})& \text{otherwise}
    \end{cases}

Similar to SciPy's `scipy.special.xlog1py`.

""" + r"""

Args:
    input (Number or Tensor) : Multiplier
    other (Number or Tensor) : Argument

.. note:: At least one of :attr:`input` or :attr:`other` must be a tensor.

Keyword args:
    {out}

Example::

    >>> x = torch.zeros(5,)
    >>> y = torch.tensor([-1, 0, 1, float('inf'), float('nan')])
    >>> torch.special.xlog1py(x, y)
    tensor([0., 0., 0., 0., nan])
    >>> x = torch.tensor([1, 2, 3])
    >>> y = torch.tensor([3, 2, 1])
    >>> torch.special.xlog1py(x, y)
    tensor([1.3863, 2.1972, 2.0794])
    >>> torch.special.xlog1py(x, 4)
    tensor([1.6094, 3.2189, 4.8283])
    >>> torch.special.xlog1py(2, y)
    tensor([2.7726, 2.1972, 1.3863])
""".format(**common_args))

i0 = _add_docstr(_special.special_i0,
                 r"""
i0(input, *, out=None) -> Tensor

Computes the zeroth order modified Bessel function of the first kind for each element of :attr:`input`.

.. math::
    \text{out}_{i} = I_0(\text{input}_{i}) = \sum_{k=0}^{\infty} \frac{(\text{input}_{i}^2/4)^k}{(k!)^2}

""" + r"""
Args:
    input (Tensor): the input tensor

Keyword args:
    {out}

Example::

    >>> torch.i0(torch.arange(5, dtype=torch.float32))
    tensor([ 1.0000,  1.2661,  2.2796,  4.8808, 11.3019])

""".format(**common_args))

i0e = _add_docstr(_special.special_i0e,
                  r"""
i0e(input, *, out=None) -> Tensor
Computes the exponentially scaled zeroth order modified Bessel function of the first kind (as defined below)
for each element of :attr:`input`.

.. math::
    \text{out}_{i} = \exp(-|x|) * i0(x) = \exp(-|x|) * \sum_{k=0}^{\infty} \frac{(\text{input}_{i}^2/4)^k}{(k!)^2}

""" + r"""
Args:
    {input}

Keyword args:
    {out}

Example::
    >>> torch.special.i0e(torch.arange(5, dtype=torch.float32))
    tensor([1.0000, 0.4658, 0.3085, 0.2430, 0.2070])
""".format(**common_args))

i1 = _add_docstr(_special.special_i1,
                 r"""
i1(input, *, out=None) -> Tensor
Computes the first order modified Bessel function of the first kind (as defined below)
for each element of :attr:`input`.

.. math::
    \text{out}_{i} = \frac{(\text{input}_{i})}{2} * \sum_{k=0}^{\infty} \frac{(\text{input}_{i}^2/4)^k}{(k!) * (k+1)!}

""" + r"""
Args:
    {input}

Keyword args:
    {out}

Example::
    >>> torch.special.i1(torch.arange(5, dtype=torch.float32))
    tensor([0.0000, 0.5652, 1.5906, 3.9534, 9.7595])
""".format(**common_args))

i1e = _add_docstr(_special.special_i1e,
                  r"""
i1e(input, *, out=None) -> Tensor
Computes the exponentially scaled first order modified Bessel function of the first kind (as defined below)
for each element of :attr:`input`.

.. math::
    \text{out}_{i} = \exp(-|x|) * i1(x) =
        \exp(-|x|) * \frac{(\text{input}_{i})}{2} * \sum_{k=0}^{\infty} \frac{(\text{input}_{i}^2/4)^k}{(k!) * (k+1)!}

""" + r"""
Args:
    {input}

Keyword args:
    {out}

Example::
    >>> torch.special.i1e(torch.arange(5, dtype=torch.float32))
    tensor([0.0000, 0.2079, 0.2153, 0.1968, 0.1788])
""".format(**common_args))

ndtr = _add_docstr(_special.special_ndtr,
                   r"""
ndtr(input, *, out=None) -> Tensor
Computes the area under the standard Gaussian probability density function,
integrated from minus infinity to :attr:`input`, elementwise.

.. math::
    \text{ndtr}(x) = \frac{1}{\sqrt{2 \pi}}\int_{-\infty}^{x} e^{-\frac{1}{2}t^2} dt

""" + r"""
Args:
    {input}

Keyword args:
    {out}

Example::
    >>> torch.special.ndtr(torch.tensor([-3., -2, -1, 0, 1, 2, 3]))
    tensor([0.0013, 0.0228, 0.1587, 0.5000, 0.8413, 0.9772, 0.9987])
""".format(**common_args))

ndtri = _add_docstr(_special.special_ndtri,
                    r"""
ndtri(input, *, out=None) -> Tensor
Computes the argument, x, for which the area under the Gaussian probability density function
(integrated from minus infinity to x) is equal to :attr:`input`, elementwise.

.. math::
    \text{ndtri}(p) = \sqrt{2}\text{erf}^{-1}(2p - 1)

.. note::
    Also known as quantile function for Normal Distribution.

""" + r"""
Args:
    {input}

Keyword args:
    {out}

Example::
    >>> torch.special.ndtri(torch.tensor([0, 0.25, 0.5, 0.75, 1]))
    tensor([   -inf, -0.6745,  0.0000,  0.6745,     inf])
""".format(**common_args))

log1p = _add_docstr(_special.special_log1p,
                    r"""
log1p(input, *, out=None) -> Tensor

Alias for :func:`torch.log1p`.
""")

sinc = _add_docstr(_special.special_sinc,
                   r"""
sinc(input, *, out=None) -> Tensor

Computes the normalized sinc of :attr:`input.`

.. math::
    \text{out}_{i} =
    \begin{cases}
      1, & \text{if}\ \text{input}_{i}=0 \\
      \sin(\pi \text{input}_{i}) / (\pi \text{input}_{i}), & \text{otherwise}
    \end{cases}
""" + r"""

Args:
    {input}

Keyword args:
    {out}

Example::
    >>> t = torch.randn(4)
    >>> t
    tensor([ 0.2252, -0.2948,  1.0267, -1.1566])
    >>> torch.special.sinc(t)
    tensor([ 0.9186,  0.8631, -0.0259, -0.1300])
""".format(**common_args))

round = _add_docstr(_special.special_round,
                    r"""
round(input, *, out=None) -> Tensor

Alias for :func:`torch.round`.
""")

log_softmax = _add_docstr(_special.special_log_softmax,
                          r"""
log_softmax(input, dim, *, dtype=None) -> Tensor
Computes softmax followed by a logarithm.

While mathematically equivalent to log(softmax(x)), doing these two
operations separately is slower and numerically unstable. This function
is computed as:

.. math::
    \text{log\_softmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)
""" + r"""

Args:
    input (Tensor): input
    dim (int): A dimension along which log_softmax will be computed.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is cast to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.

Example::
    >>> t = torch.ones(2, 2)
    >>> torch.special.log_softmax(t, 0)
    tensor([[-0.6931, -0.6931],
            [-0.6931, -0.6931]])
""")

zeta = _add_docstr(_special.special_zeta,
                   r"""
zeta(input, other, *, out=None) -> Tensor

Computes the Hurwitz zeta function, elementwise.

.. math::
    \zeta(x, q) = \sum_{k=0}^{\infty} \frac{1}{(k + q)^x}

""" + r"""
Args:
    input (Tensor): the input tensor corresponding to `x`.
    other (Tensor): the input tensor corresponding to `q`.

.. note::
    The Riemann zeta function corresponds to the case when `q = 1`

Keyword args:
    {out}

Example::
    >>> x = torch.tensor([2., 4.])
    >>> torch.special.zeta(x, 1)
    tensor([1.6449, 1.0823])
    >>> torch.special.zeta(x, torch.tensor([1., 2.]))
    tensor([1.6449, 0.0823])
    >>> torch.special.zeta(2, torch.tensor([1., 2.]))
    tensor([1.6449, 0.6449])
""".format(**common_args))
