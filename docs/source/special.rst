.. role:: hidden
    :class: hidden-section

torch.special
=============

A special function is a mathematical function that frequently arises in the
solutions of classical problems in fields like geometry, mathematical analysis,
or probability theory. PyTorch aims to provide practical closed-form solutions
for a broad and deep set of special functions.

.. automodule:: torch.special
.. currentmodule:: torch.special

Elementary Functions
--------------------

Piecewise Functions
^^^^^^^^^^^^^^^^^^^

.. note::
    The standard piecewise functions, ``min``, ``min``, and ``max``, are found
    in the ``torch`` module.

Exponential Functions
^^^^^^^^^^^^^^^^^^^^^

.. note::
    The exponential function, ``exp``, is found in the ``torch`` module.

.. autofunction:: exp2
.. autofunction:: expit
.. autofunction:: expm1

Logarithmic Functions
^^^^^^^^^^^^^^^^^^^^^

.. note::
    The standard logarithmic functions, ``log``, ``log2``, and ``log10``, are
    found in the ``torch`` module.

.. autofunction:: log1p
.. autofunction:: logit
.. autofunction:: logsumexp
.. autofunction:: xlog1py
.. autofunction:: xlogy

Trigonometric Functions
^^^^^^^^^^^^^^^^^^^^^^^

.. note::
    The majority of trigonometric, inverse trigonmetric, hyperbolic, and inverse
    hyperbolic functions are found in the ``torch`` module.

.. autofunction:: sinc

Logistic Functions
^^^^^^^^^^^^^^^^^^

.. autofunction:: log_softmax
.. autofunction:: softmax

Number Theoretic Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: round

Information Theoretic Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: entr

Error Functions and Exponential Integrals
-----------------------------------------

Error Functions
^^^^^^^^^^^^^^^

.. autofunction:: erf
.. autofunction:: erfc
.. autofunction:: erfcx
.. autofunction:: erfinv

Gamma and Related Functions
---------------------------

Gamma Functions
^^^^^^^^^^^^^^^

.. autofunction:: gammainc
.. autofunction:: gammaincc
.. autofunction:: gammaln
.. autofunction:: multigammaln

Polygamma Functions
^^^^^^^^^^^^^^^^^^^

.. autofunction:: digamma
.. autofunction:: polygamma
.. autofunction:: psi

Bessel and Related Functions
----------------------------

Modified Bessel Functions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: i0
.. autofunction:: i1

Scaled Modified Bessel Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: i0e
.. autofunction:: i1e

Zeta and Related Functions
--------------------------

Zeta Functions
^^^^^^^^^^^^^^

.. autofunction:: zeta

Probabilistic Functions
-----------------------

Cumulative Distribution Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: log_ndtr
.. autofunction:: ndtr
.. autofunction:: ndtri
