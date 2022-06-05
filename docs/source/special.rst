.. role:: hidden
    :class: hidden-section

torch.special
=============

A special function is a mathematical function that frequently arises in the solutions of classical problems in fields like geometry, mathematical analysis, or probability theory. PyTorch provides practical closed-form solutions for a broad and deep set of special functions.

.. currentmodule:: torch.special

Functions
-----------------------

.. autofunction:: entr
.. autofunction:: erf
.. autofunction:: erfc
.. autofunction:: erfcx
.. autofunction:: erfinv
.. autofunction:: expit
.. autofunction:: expm1
.. autofunction:: exp2
.. autofunction:: gammaln
.. autofunction:: gammainc
.. autofunction:: gammaincc
.. autofunction:: polygamma
.. autofunction:: digamma
.. autofunction:: psi
.. autofunction:: i0
.. autofunction:: i0e
.. autofunction:: i1
.. autofunction:: i1e
.. autofunction:: logit
.. autofunction:: logsumexp
.. autofunction:: log1p
.. autofunction:: log_softmax
.. autofunction:: multigammaln
.. autofunction:: ndtr
.. autofunction:: ndtri
.. autofunction:: log_ndtr
.. autofunction:: round
.. autofunction:: sinc
.. autofunction:: softmax
.. autofunction:: xlog1py
.. autofunction:: xlogy
.. autofunction:: zeta

Orthogonal Polynomials
----------------------

Orthogonal polynomials are families of polynomials such that any two different polynomials in a sequence are orthogonal to each under some inner product. PyTorch provides the most common orthogonal polynomials: Hermite (both physicist and probabilist standardizations), Laguerre, and Jacobi (including Chebyshev and Legendre).

Hermite Polynomials
^^^^^^^^^^^^^^^^^^^

.. autofunction:: hermite_polynomial_h
.. autofunction:: hermite_polynomial_he

Laguerre Polynomials
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: laguerre_polynomial_l

Chebyshev Polynomials
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: chebyshev_polynomial_t
.. autofunction:: chebyshev_polynomial_u
.. autofunction:: chebyshev_polynomial_v
.. autofunction:: chebyshev_polynomial_w

Shifted Chebyshev Polynomials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: shifted_chebyshev_polynomial_t
.. autofunction:: shifted_chebyshev_polynomial_u
.. autofunction:: shifted_chebyshev_polynomial_v
.. autofunction:: shifted_chebyshev_polynomial_w

Legendre Polynomials
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: legendre_polynomial_p

Bessel and Related Functions
----------------------------

Bessel Functions
^^^^^^^^^^^^^^^^

.. autofunction:: bessel_j0
.. autofunction:: bessel_j1
.. autofunction:: bessel_y0
.. autofunction:: bessel_y1

Modified Bessel Functions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: modified_bessel_i0
.. autofunction:: modified_bessel_i1
.. autofunction:: modified_bessel_k0
.. autofunction:: modified_bessel_k1

Scaled Modified Bessel Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: scaled_modified_bessel_i0
.. autofunction:: scaled_modified_bessel_i1
.. autofunction:: scaled_modified_bessel_k0
.. autofunction:: scaled_modified_bessel_k1

Airy Functions
^^^^^^^^^^^^^^

.. autofunction:: airy_ai
.. autofunction:: airy_bi

Spherical Bessel Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: spherical_bessel_j0
.. autofunction:: spherical_bessel_j1
.. autofunction:: spherical_bessel_y0
.. autofunction:: spherical_bessel_y1
