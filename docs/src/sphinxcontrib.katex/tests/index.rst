Math Rendering Examples
=======================

The examples start always with a code box showing the commands, which is
followed by the resulting Sphinx output.

Inline math
-----------

.. code-block:: rst

    Some inline math :math:`x_1 + x_2 + ... + x_n, n \in \mathbb{Z}`,
    followed by text.

Some inline math :math:`x_1 + x_2 + ... + x_n, n \in \mathbb{Z}`,
followed by text.


Macros
------

You can define macros directly in your math directive.

.. code-block:: rst

    .. math::

        \def \x {\mathbf{x}}
        \def \w {\omega}
        \def \d {\operatorname{d}\!}

        P(\x,\w) = \oint_{\partial V} D(\x_0,\w) G(\x-\x_0,\w) \d A(\x_0)

.. math::

    \def \x {\mathbf{x}}
    \def \w {\omega}
    \def \d {\operatorname{d}\!}

    P(\x,\w) = \oint_{\partial V} D(\x_0,\w) G(\x-\x_0,\w) \d A(\x_0)

If you want to use them in the whole document, the best is to define them in
:file:`conf.py` as part of the ``katex_options``.
Afterwards, you can use them in every math directive.

Aligned environment
-------------------

.. code-block:: rst

    .. math::

        \begin{aligned}
            \dot{x} & = \sigma(y-x) \\
            \dot{y} & = \rho x - y - xz \\
            \dot{z} & = -\beta z + xy
        \end{aligned}

.. math::

    \begin{aligned}
        \dot{x} & = \sigma(y-x) \\
        \dot{y} & = \rho x - y - xz \\
        \dot{z} & = -\beta z + xy
    \end{aligned}


Array environment
-----------------

.. code-block:: rst

    .. math::

        \begin{array}{c:c:c:c:c:c}
            \Gamma & \Delta & \Theta & \Lambda & \Xi & \Pi \\ \hdashline
            \gamma & \delta & \theta & \lambda & \xi & \pi
        \end{array}

.. math::

    \begin{array}{c:c:c:c:c:c}
        \Gamma & \Delta & \Theta & \Lambda & \Xi & \Pi \\ \hdashline
        \gamma & \delta & \theta & \lambda & \xi & \pi
    \end{array}


Case definitions
----------------

.. code-block:: rst

    .. math::

        f(n) = \begin{cases}
            \frac{n}{2}, & \text{if } n\text{ is even} \\
            3n+1,        & \text{if } n\text{ is odd}
        \end{cases}

.. math::

     f(n) = \begin{cases}
        \frac{n}{2}, & \text{if } n\text{ is even} \\
        2n+1,        & \text{if } n\text{ is odd}
    \end{cases}


Matrices
--------

A simple matrix defined with the ``pmatrix`` environment:

.. code-block:: rst

    .. math::

        \begin{pmatrix}
            a_{11} & a_{12} & a_{13}\\
            a_{21} & a_{22} & a_{23}\\
            a_{31} & a_{32} & a_{33}
        \end{pmatrix}

.. math::

    \begin{pmatrix}
        a_{11} & a_{12} & a_{13}\\
        a_{21} & a_{22} & a_{23}\\
        a_{31} & a_{32} & a_{33}
    \end{pmatrix}


The ``pmatrix*`` environment is not available, but you can use the ``array``
environment for more complex matrices:

.. code-block:: rst

    .. math::

        \def \msum {-\textstyle\sum}
        \def \psum {\phantom{-}\textstyle\sum}
        I_{ik} = \left(
        \begin{array}{lll}
            \psum m (y^2+z^2) & \msum m x y       & \msum m x z         \\
            \msum m y x       & \psum m (x^2+z^2) & \msum m y z         \\
            \msum m z x       & \msum m z y       & \psum m (x^2 + y^2)
        \end{array}
        \right)

.. math::

    \def \msum {-\textstyle\sum}
    \def \psum {\phantom{-}\textstyle\sum}
    I_{ik} = \left(
    \begin{array}{lll}
        \psum m (y^2+z^2) & \msum m x y       & \msum m x z         \\
        \msum m y x       & \psum m (x^2+z^2) & \msum m y z         \\
        \msum m z x       & \msum m z y       & \psum m (x^2 + y^2)
    \end{array}
    \right)
