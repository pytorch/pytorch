.. role:: hidden
    :class: hidden-section

torch.special
=============

A special function is a mathematical function that frequently arises in the solutions of classical problems in fields like geometry, mathematical analysis, or probability theory. PyTorch provides practical closed-form solutions for a broad and deep set of special functions.

.. automodule:: torch.special
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

Bessel functions are the canonical solutions to Bessel’s differential equation:

.. math::
    x^{2}{\frac{d^{2}y}{dx^{2}}}+x{\frac{dy}{dx}}+\left(x^{2}-\alpha^{2}\right)y=0

Bessel equations arise everywhere. But, perhaps most relevantly to PyTorch users, they occur in probability theory (e.g., the probability density function of the product of two normally distributed random variables), signal processing (e.g., analog filter design and FM synthesis), and physics (e.g., electromagnetic waves in a cylindrical waveguide, diffusion problems on lattices, and the dynamics of floating bodies).

Bessel functions of the first kind, denoted :math:`J_{\alpha}(x)`, are solutions to Bessel’s equation. For integer or positive :math:`\alpha`, Bessel functions of the first kind are finite at the origin, :math:`x = 0`; while for negative non-integer :math:`\alpha`, Bessel functions of the first kind diverge as :math:`x` approaches zero. It is possible to define the function by its series expansion around :math:`x = 0`, which can be found by applying the Frobenius method to Bessel’s equation:

.. math::
    J_{\alpha}(x)=\sum_{m=0}^{\infty}{\frac{(-1)^{m}}{m!\Gamma(m+\alpha+1)}}{\left({\frac{x}{2}}\right)}^{2m+\alpha}

where :math:`\Gamma(x)` is the Euler gamma function (``torch.special.gamma``).

.. autofunction:: bessel_j0
.. autofunction:: bessel_j1

The Bessel functions of the second kind, denoted :math:`Y_{\alpha}(x)`, are solutions to Bessel differential equation that have a singularity at the origin and are multivalued. For non-integer :math:`\alpha`, :math:`Y_{\alpha}(x)` is related to :math:`J_{\alpha}(x)` by:

.. math::
    Y_{\alpha}(x)={\frac{J_{\alpha }(x)\cos(\alpha\pi)-J_{-\alpha}(x)}{\sin(\alpha\pi)}}

In the case of integer order :math:`n`, the function is defined by taking the limit as a non-integer :math:`\alpha` tends to :math:`n`:

.. math::
    Y_{n}(x)=\lim_{\alpha\to n}Y_{\alpha }(x)

In the case of non-negative integer order :math:`n`, the function is defined by the series:

.. math::
    Y_{n}(x)=-{\frac{\left({\frac{x}{2}}\right)^{-n}}{\pi}}\sum_{k=0}^{n-1}{\frac{(n-k-1)!}{k!}}\left({\frac{x^{2}}{4}}\right)^{k}+{\frac{2}{\pi}}J_{n}(x)\ln{\frac{x}{2}}-{\frac{\left({\frac{x}{2}}\right)^{n}}{\pi}}\sum_{k=0}^{\infty}(\psi(k+1)+\psi(n+k+1)){\frac{\left(-{\frac{x^{2}}{4}}\right)^{k}}{k!(n+k)!}}

where :math:`\psi(x)` is the digamma function, the logarithmic derivative of the gamma function (``torch.special.digamma``).

.. autofunction:: bessel_y0
.. autofunction:: bessel_y1

Modified Bessel Functions
^^^^^^^^^^^^^^^^^^^^^^^^^

Bessel functions are valid even for complex arguments :math:`x`, and an important special case is that of an imaginary argument. In this case, the solutions to the Bessel equation are denoted as the modified Bessel functions.

The modfified Bessel functions of the first kind, denoted :math:`I_{\alpha}(x)`, are defined as:

.. math::
    I_{\alpha}(x)=i^{-\alpha}J_{\alpha}(ix)=\sum _{m=0}^{\infty }{\frac {1}{m!\,\Gamma (m+\alpha +1)}}\left({\frac {x}{2}}\right)^{2m+\alpha}

.. autofunction:: modified_bessel_i0
.. autofunction:: modified_bessel_i1

The modified Bessel functions of the second kind, denoted :math:`K_{\alpha}(x)`, are defined as:

.. math::
    K_{\alpha}(x)={\frac{\pi}{2}}{\frac{I_{-\alpha}(x)-I_{\alpha}(x)}{\sin\alpha\pi}}

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

The Airy functions, :math:`\operatorname{Ai}(x)` and :math:`\operatorname{Ai}(x)`, are linearly independent solutions to the Stokes differential equation:

.. math::
    {\frac{d^{2}y}{dx^{2}}}-xy=0

The Stokes equation is noteworthy, from an analytic perspective, because it is the simplest second-order linear differential equation with a turning point (a point where the character of the solutions changes from oscillatory to exponential). Airy functions are especially useful in computing other special functions. Still, they arise elsewhere, most notably in fluid dynamics, optics, and wave propagation.

For real values of :math:`x`, the Airy function of the first kind, denoted :math:`\operatorname{Ai}(x)`, is typically defined by the improper Riemann integral:

.. math::
    \operatorname{Ai}(x)={\dfrac{1}{\pi}}\int_{0}^{\infty}\cos\left({\dfrac{t^{3}}{3}}+xt\right)\,dt\equiv{\dfrac{1}{\pi}}\lim_{b\to\infty}\int_{0}^{b}\cos\left({\dfrac{t^{3}}{3}}+xt\right)\,dt

.. autofunction:: airy_ai

:math:`y = \operatorname{Ai}(x)` satisfies the Airy equation:

.. math::
    y''-xy=0.

This equation has two linearly independent solutions. Up to scalar multiplication, :math:`\operatorname{Ai}(x)` is the solution subject to the condition :math:`y \rightarrow 0` as :math:`x \rightarrow \infty`. The standard choice for the other solution is the Airy function of the second kind, denoted :math:`\operatorname{Bi}(x)`. It is defined as the solution with the same amplitude of oscillation as :math:`\operatorname{Ai}(x)` as :math:`x \rightarrow \infty` which differs in phase by :math:`\frac{\pi}{2}`:

.. math::
    \operatorname{Bi}(x)={\frac{1}{\pi}}\int_{0}^{\infty}\left[\exp \left(-{\tfrac{t^{3}}{3}}+xt\right)+\sin\left({\tfrac{t^{3}}{3}}+xt\right)\,\right]dt.

.. autofunction:: airy_bi

Spherical Bessel Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: spherical_bessel_j0
.. autofunction:: spherical_bessel_j1
.. autofunction:: spherical_bessel_y0
.. autofunction:: spherical_bessel_y1
