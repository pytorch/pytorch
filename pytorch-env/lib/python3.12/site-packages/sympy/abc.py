"""
This module exports all latin and greek letters as Symbols, so you can
conveniently do

    >>> from sympy.abc import x, y

instead of the slightly more clunky-looking

    >>> from sympy import symbols
    >>> x, y = symbols('x y')

Caveats
=======

1. As of the time of writing this, the names ``O``, ``S``, ``I``, ``N``,
``E``, and ``Q`` are colliding with names defined in SymPy. If you import them
from both ``sympy.abc`` and ``sympy``, the second import will "win".
This is an issue only for * imports, which should only be used for short-lived
code such as interactive sessions and throwaway scripts that do not survive
until the next SymPy upgrade, where ``sympy`` may contain a different set of
names.

2. This module does not define symbol names on demand, i.e.
``from sympy.abc import foo`` will be reported as an error because
``sympy.abc`` does not contain the name ``foo``. To get a symbol named ``foo``,
you still need to use ``Symbol('foo')`` or ``symbols('foo')``.
You can freely mix usage of ``sympy.abc`` and ``Symbol``/``symbols``, though
sticking with one and only one way to get the symbols does tend to make the code
more readable.

The module also defines some special names to help detect which names clash
with the default SymPy namespace.

``_clash1`` defines all the single letter variables that clash with
SymPy objects; ``_clash2`` defines the multi-letter clashing symbols;
and ``_clash`` is the union of both. These can be passed for ``locals``
during sympification if one desires Symbols rather than the non-Symbol
objects for those names.

Examples
========

>>> from sympy import S
>>> from sympy.abc import _clash1, _clash2, _clash
>>> S("Q & C", locals=_clash1)
C & Q
>>> S('pi(x)', locals=_clash2)
pi(x)
>>> S('pi(C, Q)', locals=_clash)
pi(C, Q)

"""

from typing import Any, Dict as tDict

import string

from .core import Symbol, symbols
from .core.alphabets import greeks
from sympy.parsing.sympy_parser import null

##### Symbol definitions #####

# Implementation note: The easiest way to avoid typos in the symbols()
# parameter is to copy it from the left-hand side of the assignment.

a, b, c, d, e, f, g, h, i, j = symbols('a, b, c, d, e, f, g, h, i, j')
k, l, m, n, o, p, q, r, s, t = symbols('k, l, m, n, o, p, q, r, s, t')
u, v, w, x, y, z = symbols('u, v, w, x, y, z')

A, B, C, D, E, F, G, H, I, J = symbols('A, B, C, D, E, F, G, H, I, J')
K, L, M, N, O, P, Q, R, S, T = symbols('K, L, M, N, O, P, Q, R, S, T')
U, V, W, X, Y, Z = symbols('U, V, W, X, Y, Z')

alpha, beta, gamma, delta = symbols('alpha, beta, gamma, delta')
epsilon, zeta, eta, theta = symbols('epsilon, zeta, eta, theta')
iota, kappa, lamda, mu = symbols('iota, kappa, lamda, mu')
nu, xi, omicron, pi = symbols('nu, xi, omicron, pi')
rho, sigma, tau, upsilon = symbols('rho, sigma, tau, upsilon')
phi, chi, psi, omega = symbols('phi, chi, psi, omega')


##### Clashing-symbols diagnostics #####

# We want to know which names in SymPy collide with those in here.
# This is mostly for diagnosing SymPy's namespace during SymPy development.

_latin = list(string.ascii_letters)
# QOSINE should not be imported as they clash; gamma, pi and zeta clash, too
_greek = list(greeks) # make a copy, so we can mutate it
# Note: We import lamda since lambda is a reserved keyword in Python
_greek.remove("lambda")
_greek.append("lamda")

ns: tDict[str, Any] = {}
exec('from sympy import *', ns)
_clash1: tDict[str, Any] = {}
_clash2: tDict[str, Any] = {}
while ns:
    _k, _ = ns.popitem()
    if _k in _greek:
        _clash2[_k] = null
        _greek.remove(_k)
    elif _k in _latin:
        _clash1[_k] = null
        _latin.remove(_k)
_clash = {}
_clash.update(_clash1)
_clash.update(_clash2)

del _latin, _greek, Symbol, _k, null
