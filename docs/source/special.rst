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

.. autosummary::
    :toctree: generated
    :nosignatures:

    hermite_polynomial_h
    hermite_polynomial_he

Laguerre Polynomials
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated
    :nosignatures:

    laguerre_polynomial_l

Chebyshev Polynomials
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated
    :nosignatures:

    chebyshev_polynomial_t
    chebyshev_polynomial_u
    chebyshev_polynomial_v
    chebyshev_polynomial_w

Shifted Chebyshev Polynomials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated
    :nosignatures:

    shifted_chebyshev_polynomial_t
    shifted_chebyshev_polynomial_u
    shifted_chebyshev_polynomial_v
    shifted_chebyshev_polynomial_w

Legendre Polynomials
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated
    :nosignatures:

    legendre_polynomial_p

Bessel and Related Functions
----------------------------

Bessel Functions
^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated
    :nosignatures:

    bessel_j0
    bessel_j1
    bessel_y0
    bessel_y1

Modified Bessel Functions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated
    :nosignatures:

    modified_bessel_i0
    modified_bessel_i1
    modified_bessel_k0
    modified_bessel_k1

Scaled Modified Bessel Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated
    :nosignatures:

    scaled_modified_bessel_i0
    scaled_modified_bessel_i1
    scaled_modified_bessel_k0
    scaled_modified_bessel_k1

Airy Functions
^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated
    :nosignatures:

    airy_ai
    airy_bi

Spherical Bessel Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated
    :nosignatures:

    spherical_bessel_j0
    spherical_bessel_j1
    spherical_bessel_y0
    spherical_bessel_y1
