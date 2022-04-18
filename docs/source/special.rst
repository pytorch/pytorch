.. role:: hidden
    :class: hidden-section

torch.special
=============

The torch.special module, modeled after SciPy's `special <https://docs.scipy.org/doc/scipy/reference/special.html>`_ module.

.. automodule:: torch.special
.. currentmodule:: torch.special

Bessel
------

Bessel functions are solutions to Bessel’s ordinary differential equation:

.. math:
    x^{2}{\\frac{d^{2}y}{dx^{2}}} + x{\\frac{dy}{dx}} + (x^{2} - \\alpha^{2})y = 0

.. autofunction:: bessel_j0
.. autofunction:: bessel_j1
.. autofunction:: bessel_y0
.. autofunction:: bessel_y1
.. autofunction:: i0
.. autofunction:: i0e
.. autofunction:: i1
.. autofunction:: i1e
.. autofunction:: bessel_k0
.. autofunction:: bessel_k0e
.. autofunction:: bessel_k1
.. autofunction:: bessel_k1e

Elementary
----------

.. autofunction:: exp2
.. autofunction:: expm1
.. autofunction:: log1p
.. autofunction:: log_softmax
.. autofunction:: logsumexp
.. autofunction:: round
.. autofunction:: sinc
.. autofunction:: softmax
.. autofunction:: xlog1py
.. autofunction:: xlogy
.. autofunction:: zeta

Error
-----

.. autofunction:: erf
.. autofunction:: erfc
.. autofunction:: erfcx
.. autofunction:: erfinv

Gamma
-----

.. autofunction:: digamma
.. autofunction:: gammainc
.. autofunction:: gammaincc
.. autofunction:: gammaln
.. autofunction:: multigammaln
.. autofunction:: polygamma
.. autofunction:: psi

Statistical
-----------

.. autofunction:: log_ndtr
.. autofunction:: logit
.. autofunction:: ndtr
.. autofunction:: ndtri
.. autofunction:: expit

Information Theoretic
---------------------

.. autofunction:: entr
