"""
This module contains pdsolve() and different helper functions that it
uses. It is heavily inspired by the ode module and hence the basic
infrastructure remains the same.

**Functions in this module**

    These are the user functions in this module:

    - pdsolve()     - Solves PDE's
    - classify_pde() - Classifies PDEs into possible hints for dsolve().
    - pde_separate() - Separate variables in partial differential equation either by
                       additive or multiplicative separation approach.

    These are the helper functions in this module:

    - pde_separate_add() - Helper function for searching additive separable solutions.
    - pde_separate_mul() - Helper function for searching multiplicative
                           separable solutions.

**Currently implemented solver methods**

The following methods are implemented for solving partial differential
equations.  See the docstrings of the various pde_hint() functions for
more information on each (run help(pde)):

  - 1st order linear homogeneous partial differential equations
    with constant coefficients.
  - 1st order linear general partial differential equations
    with constant coefficients.
  - 1st order linear partial differential equations with
    variable coefficients.

"""
from functools import reduce

from itertools import combinations_with_replacement
from sympy.simplify import simplify  # type: ignore
from sympy.core import Add, S
from sympy.core.function import Function, expand, AppliedUndef, Subs
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Symbol, Wild, symbols
from sympy.functions import exp
from sympy.integrals.integrals import Integral, integrate
from sympy.utilities.iterables import has_dups, is_sequence
from sympy.utilities.misc import filldedent

from sympy.solvers.deutils import _preprocess, ode_order, _desolve
from sympy.solvers.solvers import solve
from sympy.simplify.radsimp import collect

import operator


allhints = (
    "1st_linear_constant_coeff_homogeneous",
    "1st_linear_constant_coeff",
    "1st_linear_constant_coeff_Integral",
    "1st_linear_variable_coeff"
    )


def pdsolve(eq, func=None, hint='default', dict=False, solvefun=None, **kwargs):
    """
    Solves any (supported) kind of partial differential equation.

    **Usage**

        pdsolve(eq, f(x,y), hint) -> Solve partial differential equation
        eq for function f(x,y), using method hint.

    **Details**

        ``eq`` can be any supported partial differential equation (see
            the pde docstring for supported methods).  This can either
            be an Equality, or an expression, which is assumed to be
            equal to 0.

        ``f(x,y)`` is a function of two variables whose derivatives in that
            variable make up the partial differential equation. In many
            cases it is not necessary to provide this; it will be autodetected
            (and an error raised if it could not be detected).

        ``hint`` is the solving method that you want pdsolve to use.  Use
            classify_pde(eq, f(x,y)) to get all of the possible hints for
            a PDE.  The default hint, 'default', will use whatever hint
            is returned first by classify_pde().  See Hints below for
            more options that you can use for hint.

        ``solvefun`` is the convention used for arbitrary functions returned
            by the PDE solver. If not set by the user, it is set by default
            to be F.

    **Hints**

        Aside from the various solving methods, there are also some
        meta-hints that you can pass to pdsolve():

        "default":
                This uses whatever hint is returned first by
                classify_pde(). This is the default argument to
                pdsolve().

        "all":
                To make pdsolve apply all relevant classification hints,
                use pdsolve(PDE, func, hint="all").  This will return a
                dictionary of hint:solution terms.  If a hint causes
                pdsolve to raise the NotImplementedError, value of that
                hint's key will be the exception object raised.  The
                dictionary will also include some special keys:

                - order: The order of the PDE.  See also ode_order() in
                  deutils.py
                - default: The solution that would be returned by
                  default.  This is the one produced by the hint that
                  appears first in the tuple returned by classify_pde().

        "all_Integral":
                This is the same as "all", except if a hint also has a
                corresponding "_Integral" hint, it only returns the
                "_Integral" hint.  This is useful if "all" causes
                pdsolve() to hang because of a difficult or impossible
                integral.  This meta-hint will also be much faster than
                "all", because integrate() is an expensive routine.

        See also the classify_pde() docstring for more info on hints,
        and the pde docstring for a list of all supported hints.

    **Tips**
        - You can declare the derivative of an unknown function this way:

            >>> from sympy import Function, Derivative
            >>> from sympy.abc import x, y # x and y are the independent variables
            >>> f = Function("f")(x, y) # f is a function of x and y
            >>> # fx will be the partial derivative of f with respect to x
            >>> fx = Derivative(f, x)
            >>> # fy will be the partial derivative of f with respect to y
            >>> fy = Derivative(f, y)

        - See test_pde.py for many tests, which serves also as a set of
          examples for how to use pdsolve().
        - pdsolve always returns an Equality class (except for the case
          when the hint is "all" or "all_Integral"). Note that it is not possible
          to get an explicit solution for f(x, y) as in the case of ODE's
        - Do help(pde.pde_hintname) to get help more information on a
          specific hint


    Examples
    ========

    >>> from sympy.solvers.pde import pdsolve
    >>> from sympy import Function, Eq
    >>> from sympy.abc import x, y
    >>> f = Function('f')
    >>> u = f(x, y)
    >>> ux = u.diff(x)
    >>> uy = u.diff(y)
    >>> eq = Eq(1 + (2*(ux/u)) + (3*(uy/u)), 0)
    >>> pdsolve(eq)
    Eq(f(x, y), F(3*x - 2*y)*exp(-2*x/13 - 3*y/13))

    """

    if not solvefun:
        solvefun = Function('F')

    # See the docstring of _desolve for more details.
    hints = _desolve(eq, func=func, hint=hint, simplify=True,
                     type='pde', **kwargs)
    eq = hints.pop('eq', False)
    all_ = hints.pop('all', False)

    if all_:
        # TODO : 'best' hint should be implemented when adequate
        # number of hints are added.
        pdedict = {}
        failed_hints = {}
        gethints = classify_pde(eq, dict=True)
        pdedict.update({'order': gethints['order'],
                        'default': gethints['default']})
        for hint in hints:
            try:
                rv = _helper_simplify(eq, hint, hints[hint]['func'],
                    hints[hint]['order'], hints[hint][hint], solvefun)
            except NotImplementedError as detail:
                failed_hints[hint] = detail
            else:
                pdedict[hint] = rv
        pdedict.update(failed_hints)
        return pdedict

    else:
        return _helper_simplify(eq, hints['hint'], hints['func'],
                                hints['order'], hints[hints['hint']], solvefun)


def _helper_simplify(eq, hint, func, order, match, solvefun):
    """Helper function of pdsolve that calls the respective
    pde functions to solve for the partial differential
    equations. This minimizes the computation in
    calling _desolve multiple times.
    """
    solvefunc = globals()["pde_" + hint.removesuffix("_Integral")]
    return _handle_Integral(solvefunc(eq, func, order,
        match, solvefun), func, order, hint)


def _handle_Integral(expr, func, order, hint):
    r"""
    Converts a solution with integrals in it into an actual solution.

    Simplifies the integral mainly using doit()
    """
    if hint.endswith("_Integral"):
        return expr

    elif hint == "1st_linear_constant_coeff":
        return simplify(expr.doit())

    else:
        return expr


def classify_pde(eq, func=None, dict=False, *, prep=True, **kwargs):
    """
    Returns a tuple of possible pdsolve() classifications for a PDE.

    The tuple is ordered so that first item is the classification that
    pdsolve() uses to solve the PDE by default.  In general,
    classifications near the beginning of the list will produce
    better solutions faster than those near the end, though there are
    always exceptions.  To make pdsolve use a different classification,
    use pdsolve(PDE, func, hint=<classification>).  See also the pdsolve()
    docstring for different meta-hints you can use.

    If ``dict`` is true, classify_pde() will return a dictionary of
    hint:match expression terms. This is intended for internal use by
    pdsolve().  Note that because dictionaries are ordered arbitrarily,
    this will most likely not be in the same order as the tuple.

    You can get help on different hints by doing help(pde.pde_hintname),
    where hintname is the name of the hint without "_Integral".

    See sympy.pde.allhints or the sympy.pde docstring for a list of all
    supported hints that can be returned from classify_pde.


    Examples
    ========

    >>> from sympy.solvers.pde import classify_pde
    >>> from sympy import Function, Eq
    >>> from sympy.abc import x, y
    >>> f = Function('f')
    >>> u = f(x, y)
    >>> ux = u.diff(x)
    >>> uy = u.diff(y)
    >>> eq = Eq(1 + (2*(ux/u)) + (3*(uy/u)), 0)
    >>> classify_pde(eq)
    ('1st_linear_constant_coeff_homogeneous',)
    """

    if func and len(func.args) != 2:
        raise NotImplementedError("Right now only partial "
            "differential equations of two variables are supported")

    if prep or func is None:
        prep, func_ = _preprocess(eq, func)
        if func is None:
            func = func_

    if isinstance(eq, Equality):
        if eq.rhs != 0:
            return classify_pde(eq.lhs - eq.rhs, func)
        eq = eq.lhs

    f = func.func
    x = func.args[0]
    y = func.args[1]
    fx = f(x,y).diff(x)
    fy = f(x,y).diff(y)

    # TODO : For now pde.py uses support offered by the ode_order function
    # to find the order with respect to a multi-variable function. An
    # improvement could be to classify the order of the PDE on the basis of
    # individual variables.
    order = ode_order(eq, f(x,y))

    # hint:matchdict or hint:(tuple of matchdicts)
    # Also will contain "default":<default hint> and "order":order items.
    matching_hints = {'order': order}

    if not order:
        if dict:
            matching_hints["default"] = None
            return matching_hints
        return ()

    eq = expand(eq)

    a = Wild('a', exclude = [f(x,y)])
    b = Wild('b', exclude = [f(x,y), fx, fy, x, y])
    c = Wild('c', exclude = [f(x,y), fx, fy, x, y])
    d = Wild('d', exclude = [f(x,y), fx, fy, x, y])
    e = Wild('e', exclude = [f(x,y), fx, fy])
    n = Wild('n', exclude = [x, y])
    # Try removing the smallest power of f(x,y)
    # from the highest partial derivatives of f(x,y)
    reduced_eq = eq
    if eq.is_Add:
        power = None
        for i in set(combinations_with_replacement((x,y), order)):
            coeff = eq.coeff(f(x,y).diff(*i))
            if coeff == 1:
                continue
            match = coeff.match(a*f(x,y)**n)
            if match and match[a]:
                if power is None or match[n] < power:
                    power = match[n]
        if power:
            den = f(x,y)**power
            reduced_eq = Add(*[arg/den for arg in eq.args])

    if order == 1:
        reduced_eq = collect(reduced_eq, f(x, y))
        r = reduced_eq.match(b*fx + c*fy + d*f(x,y) + e)
        if r:
            if not r[e]:
                ## Linear first-order homogeneous partial-differential
                ## equation with constant coefficients
                r.update({'b': b, 'c': c, 'd': d})
                matching_hints["1st_linear_constant_coeff_homogeneous"] = r
            elif r[b]**2 + r[c]**2 != 0:
                ## Linear first-order general partial-differential
                ## equation with constant coefficients
                r.update({'b': b, 'c': c, 'd': d, 'e': e})
                matching_hints["1st_linear_constant_coeff"] = r
                matching_hints["1st_linear_constant_coeff_Integral"] = r

        else:
            b = Wild('b', exclude=[f(x, y), fx, fy])
            c = Wild('c', exclude=[f(x, y), fx, fy])
            d = Wild('d', exclude=[f(x, y), fx, fy])
            r = reduced_eq.match(b*fx + c*fy + d*f(x,y) + e)
            if r:
                r.update({'b': b, 'c': c, 'd': d, 'e': e})
                matching_hints["1st_linear_variable_coeff"] = r

    # Order keys based on allhints.
    rettuple = tuple(i for i in allhints if i in matching_hints)

    if dict:
        # Dictionaries are ordered arbitrarily, so make note of which
        # hint would come first for pdsolve().  Use an ordered dict in Py 3.
        matching_hints["default"] = None
        matching_hints["ordered_hints"] = rettuple
        for i in allhints:
            if i in matching_hints:
                matching_hints["default"] = i
                break
        return matching_hints
    return rettuple


def checkpdesol(pde, sol, func=None, solve_for_func=True):
    """
    Checks if the given solution satisfies the partial differential
    equation.

    pde is the partial differential equation which can be given in the
    form of an equation or an expression. sol is the solution for which
    the pde is to be checked. This can also be given in an equation or
    an expression form. If the function is not provided, the helper
    function _preprocess from deutils is used to identify the function.

    If a sequence of solutions is passed, the same sort of container will be
    used to return the result for each solution.

    The following methods are currently being implemented to check if the
    solution satisfies the PDE:

        1. Directly substitute the solution in the PDE and check. If the
           solution has not been solved for f, then it will solve for f
           provided solve_for_func has not been set to False.

    If the solution satisfies the PDE, then a tuple (True, 0) is returned.
    Otherwise a tuple (False, expr) where expr is the value obtained
    after substituting the solution in the PDE. However if a known solution
    returns False, it may be due to the inability of doit() to simplify it to zero.

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy.solvers.pde import checkpdesol, pdsolve
    >>> x, y = symbols('x y')
    >>> f = Function('f')
    >>> eq = 2*f(x,y) + 3*f(x,y).diff(x) + 4*f(x,y).diff(y)
    >>> sol = pdsolve(eq)
    >>> assert checkpdesol(eq, sol)[0]
    >>> eq = x*f(x,y) + f(x,y).diff(x)
    >>> checkpdesol(eq, sol)
    (False, (x*F(4*x - 3*y) - 6*F(4*x - 3*y)/25 + 4*Subs(Derivative(F(_xi_1), _xi_1), _xi_1, 4*x - 3*y))*exp(-6*x/25 - 8*y/25))
    """

    # Converting the pde into an equation
    if not isinstance(pde, Equality):
        pde = Eq(pde, 0)

    # If no function is given, try finding the function present.
    if func is None:
        try:
            _, func = _preprocess(pde.lhs)
        except ValueError:
            funcs = [s.atoms(AppliedUndef) for s in (
                sol if is_sequence(sol, set) else [sol])]
            funcs = set().union(funcs)
            if len(funcs) != 1:
                raise ValueError(
                    'must pass func arg to checkpdesol for this case.')
            func = funcs.pop()

    # If the given solution is in the form of a list or a set
    # then return a list or set of tuples.
    if is_sequence(sol, set):
        return type(sol)([checkpdesol(
            pde, i, func=func,
            solve_for_func=solve_for_func) for i in sol])

    # Convert solution into an equation
    if not isinstance(sol, Equality):
        sol = Eq(func, sol)
    elif sol.rhs == func:
        sol = sol.reversed

    # Try solving for the function
    solved = sol.lhs == func and not sol.rhs.has(func)
    if solve_for_func and not solved:
        solved = solve(sol, func)
        if solved:
            if len(solved) == 1:
                return checkpdesol(pde, Eq(func, solved[0]),
                    func=func, solve_for_func=False)
            else:
                return checkpdesol(pde, [Eq(func, t) for t in solved],
                    func=func, solve_for_func=False)

    # try direct substitution of the solution into the PDE and simplify
    if sol.lhs == func:
        pde = pde.lhs - pde.rhs
        s = simplify(pde.subs(func, sol.rhs).doit())
        return s is S.Zero, s

    raise NotImplementedError(filldedent('''
        Unable to test if %s is a solution to %s.''' % (sol, pde)))



def pde_1st_linear_constant_coeff_homogeneous(eq, func, order, match, solvefun):
    r"""
    Solves a first order linear homogeneous
    partial differential equation with constant coefficients.

    The general form of this partial differential equation is

    .. math:: a \frac{\partial f(x,y)}{\partial x}
              + b \frac{\partial f(x,y)}{\partial y} + c f(x,y) = 0

    where `a`, `b` and `c` are constants.

    The general solution is of the form:

    .. math::
        f(x, y) = F(- a y + b x ) e^{- \frac{c (a x + b y)}{a^2 + b^2}}

    and can be found in SymPy with ``pdsolve``::

        >>> from sympy.solvers import pdsolve
        >>> from sympy.abc import x, y, a, b, c
        >>> from sympy import Function, pprint
        >>> f = Function('f')
        >>> u = f(x,y)
        >>> ux = u.diff(x)
        >>> uy = u.diff(y)
        >>> genform = a*ux + b*uy + c*u
        >>> pprint(genform)
          d               d
        a*--(f(x, y)) + b*--(f(x, y)) + c*f(x, y)
          dx              dy

        >>> pprint(pdsolve(genform))
                                 -c*(a*x + b*y)
                                 ---------------
                                      2    2
                                     a  + b
        f(x, y) = F(-a*y + b*x)*e

    Examples
    ========

    >>> from sympy import pdsolve
    >>> from sympy import Function, pprint
    >>> from sympy.abc import x,y
    >>> f = Function('f')
    >>> pdsolve(f(x,y) + f(x,y).diff(x) + f(x,y).diff(y))
    Eq(f(x, y), F(x - y)*exp(-x/2 - y/2))
    >>> pprint(pdsolve(f(x,y) + f(x,y).diff(x) + f(x,y).diff(y)))
                          x   y
                        - - - -
                          2   2
    f(x, y) = F(x - y)*e

    References
    ==========

    - Viktor Grigoryan, "Partial Differential Equations"
      Math 124A - Fall 2010, pp.7

    """
    # TODO : For now homogeneous first order linear PDE's having
    # two variables are implemented. Once there is support for
    # solving systems of ODE's, this can be extended to n variables.

    f = func.func
    x = func.args[0]
    y = func.args[1]
    b = match[match['b']]
    c = match[match['c']]
    d = match[match['d']]
    return Eq(f(x,y), exp(-S(d)/(b**2 + c**2)*(b*x + c*y))*solvefun(c*x - b*y))


def pde_1st_linear_constant_coeff(eq, func, order, match, solvefun):
    r"""
    Solves a first order linear partial differential equation
    with constant coefficients.

    The general form of this partial differential equation is

    .. math:: a \frac{\partial f(x,y)}{\partial x}
              + b \frac{\partial f(x,y)}{\partial y}
              + c f(x,y) = G(x,y)

    where `a`, `b` and `c` are constants and `G(x, y)` can be an arbitrary
    function in `x` and `y`.

    The general solution of the PDE is:

    .. math::
        f(x, y) = \left. \left[F(\eta) + \frac{1}{a^2 + b^2}
        \int\limits^{a x + b y} G\left(\frac{a \xi + b \eta}{a^2 + b^2},
        \frac{- a \eta + b \xi}{a^2 + b^2} \right)
        e^{\frac{c \xi}{a^2 + b^2}}\, d\xi\right]
        e^{- \frac{c \xi}{a^2 + b^2}}
        \right|_{\substack{\eta=- a y + b x\\ \xi=a x + b y }}\, ,

    where `F(\eta)` is an arbitrary single-valued function. The solution
    can be found in SymPy with ``pdsolve``::

        >>> from sympy.solvers import pdsolve
        >>> from sympy.abc import x, y, a, b, c
        >>> from sympy import Function, pprint
        >>> f = Function('f')
        >>> G = Function('G')
        >>> u = f(x, y)
        >>> ux = u.diff(x)
        >>> uy = u.diff(y)
        >>> genform = a*ux + b*uy + c*u - G(x,y)
        >>> pprint(genform)
          d               d
        a*--(f(x, y)) + b*--(f(x, y)) + c*f(x, y) - G(x, y)
          dx              dy
        >>> pprint(pdsolve(genform, hint='1st_linear_constant_coeff_Integral'))
                  //          a*x + b*y                                             \         \|
                  ||              /                                                 |         ||
                  ||             |                                                  |         ||
                  ||             |                                      c*xi        |         ||
                  ||             |                                     -------      |         ||
                  ||             |                                      2    2      |         ||
                  ||             |      /a*xi + b*eta  -a*eta + b*xi\  a  + b       |         ||
                  ||             |     G|------------, -------------|*e        d(xi)|         ||
                  ||             |      |   2    2         2    2   |               |         ||
                  ||             |      \  a  + b         a  + b    /               |  -c*xi  ||
                  ||             |                                                  |  -------||
                  ||            /                                                   |   2    2||
                  ||                                                                |  a  + b ||
        f(x, y) = ||F(eta) + -------------------------------------------------------|*e       ||
                  ||                                  2    2                        |         ||
                  \\                                 a  + b                         /         /|eta=-a*y + b*x, xi=a*x + b*y

    Examples
    ========

    >>> from sympy.solvers.pde import pdsolve
    >>> from sympy import Function, pprint, exp
    >>> from sympy.abc import x,y
    >>> f = Function('f')
    >>> eq = -2*f(x,y).diff(x) + 4*f(x,y).diff(y) + 5*f(x,y) - exp(x + 3*y)
    >>> pdsolve(eq)
    Eq(f(x, y), (F(4*x + 2*y)*exp(x/2) + exp(x + 4*y)/15)*exp(-y))

    References
    ==========

    - Viktor Grigoryan, "Partial Differential Equations"
      Math 124A - Fall 2010, pp.7

    """

    # TODO : For now homogeneous first order linear PDE's having
    # two variables are implemented. Once there is support for
    # solving systems of ODE's, this can be extended to n variables.
    xi, eta = symbols("xi eta")
    f = func.func
    x = func.args[0]
    y = func.args[1]
    b = match[match['b']]
    c = match[match['c']]
    d = match[match['d']]
    e = -match[match['e']]
    expterm = exp(-S(d)/(b**2 + c**2)*xi)
    functerm = solvefun(eta)
    solvedict = solve((b*x + c*y - xi, c*x - b*y - eta), x, y)
    # Integral should remain as it is in terms of xi,
    # doit() should be done in _handle_Integral.
    genterm = (1/S(b**2 + c**2))*Integral(
        (1/expterm*e).subs(solvedict), (xi, b*x + c*y))
    return Eq(f(x,y), Subs(expterm*(functerm + genterm),
        (eta, xi), (c*x - b*y, b*x + c*y)))


def pde_1st_linear_variable_coeff(eq, func, order, match, solvefun):
    r"""
    Solves a first order linear partial differential equation
    with variable coefficients. The general form of this partial
    differential equation is

    .. math:: a(x, y) \frac{\partial f(x, y)}{\partial x}
                + b(x, y) \frac{\partial f(x, y)}{\partial y}
                + c(x, y) f(x, y) = G(x, y)

    where `a(x, y)`, `b(x, y)`, `c(x, y)` and `G(x, y)` are arbitrary
    functions in `x` and `y`. This PDE is converted into an ODE by
    making the following transformation:

    1. `\xi` as `x`

    2. `\eta` as the constant in the solution to the differential
       equation `\frac{dy}{dx} = -\frac{b}{a}`

    Making the previous substitutions reduces it to the linear ODE

    .. math:: a(\xi, \eta)\frac{du}{d\xi} + c(\xi, \eta)u - G(\xi, \eta) = 0

    which can be solved using ``dsolve``.

    >>> from sympy.abc import x, y
    >>> from sympy import Function, pprint
    >>> a, b, c, G, f= [Function(i) for i in ['a', 'b', 'c', 'G', 'f']]
    >>> u = f(x,y)
    >>> ux = u.diff(x)
    >>> uy = u.diff(y)
    >>> genform = a(x, y)*u + b(x, y)*ux + c(x, y)*uy - G(x,y)
    >>> pprint(genform)
                                         d                     d
    -G(x, y) + a(x, y)*f(x, y) + b(x, y)*--(f(x, y)) + c(x, y)*--(f(x, y))
                                         dx                    dy


    Examples
    ========

    >>> from sympy.solvers.pde import pdsolve
    >>> from sympy import Function, pprint
    >>> from sympy.abc import x,y
    >>> f = Function('f')
    >>> eq =  x*(u.diff(x)) - y*(u.diff(y)) + y**2*u - y**2
    >>> pdsolve(eq)
    Eq(f(x, y), F(x*y)*exp(y**2/2) + 1)

    References
    ==========

    - Viktor Grigoryan, "Partial Differential Equations"
      Math 124A - Fall 2010, pp.7

    """
    from sympy.solvers.ode import dsolve

    eta = symbols("eta")
    f = func.func
    x = func.args[0]
    y = func.args[1]
    b = match[match['b']]
    c = match[match['c']]
    d = match[match['d']]
    e = -match[match['e']]


    if not d:
         # To deal with cases like b*ux = e or c*uy = e
        if not (b and c):
            if c:
                try:
                    tsol = integrate(e/c, y)
                except NotImplementedError:
                    raise NotImplementedError("Unable to find a solution"
                        " due to inability of integrate")
                else:
                    return Eq(f(x,y), solvefun(x) + tsol)
            if b:
                try:
                    tsol = integrate(e/b, x)
                except NotImplementedError:
                    raise NotImplementedError("Unable to find a solution"
                        " due to inability of integrate")
                else:
                    return Eq(f(x,y), solvefun(y) + tsol)

    if not c:
        # To deal with cases when c is 0, a simpler method is used.
        # The PDE reduces to b*(u.diff(x)) + d*u = e, which is a linear ODE in x
        plode = f(x).diff(x)*b + d*f(x) - e
        sol = dsolve(plode, f(x))
        syms = sol.free_symbols - plode.free_symbols - {x, y}
        rhs = _simplify_variable_coeff(sol.rhs, syms, solvefun, y)
        return Eq(f(x, y), rhs)

    if not b:
        # To deal with cases when b is 0, a simpler method is used.
        # The PDE reduces to c*(u.diff(y)) + d*u = e, which is a linear ODE in y
        plode = f(y).diff(y)*c + d*f(y) - e
        sol = dsolve(plode, f(y))
        syms = sol.free_symbols - plode.free_symbols - {x, y}
        rhs = _simplify_variable_coeff(sol.rhs, syms, solvefun, x)
        return Eq(f(x, y), rhs)

    dummy = Function('d')
    h = (c/b).subs(y, dummy(x))
    sol = dsolve(dummy(x).diff(x) - h, dummy(x))
    if isinstance(sol, list):
        sol = sol[0]
    solsym = sol.free_symbols - h.free_symbols - {x, y}
    if len(solsym) == 1:
        solsym = solsym.pop()
        etat = (solve(sol, solsym)[0]).subs(dummy(x), y)
        ysub = solve(eta - etat, y)[0]
        deq = (b*(f(x).diff(x)) + d*f(x) - e).subs(y, ysub)
        final = (dsolve(deq, f(x), hint='1st_linear')).rhs
        if isinstance(final, list):
            final = final[0]
        finsyms = final.free_symbols - deq.free_symbols - {x, y}
        rhs = _simplify_variable_coeff(final, finsyms, solvefun, etat)
        return Eq(f(x, y), rhs)

    else:
        raise NotImplementedError("Cannot solve the partial differential equation due"
            " to inability of constantsimp")


def _simplify_variable_coeff(sol, syms, func, funcarg):
    r"""
    Helper function to replace constants by functions in 1st_linear_variable_coeff
    """
    eta = Symbol("eta")
    if len(syms) == 1:
        sym = syms.pop()
        final = sol.subs(sym, func(funcarg))

    else:
        for sym in syms:
            final = sol.subs(sym, func(funcarg))

    return simplify(final.subs(eta, funcarg))


def pde_separate(eq, fun, sep, strategy='mul'):
    """Separate variables in partial differential equation either by additive
    or multiplicative separation approach. It tries to rewrite an equation so
    that one of the specified variables occurs on a different side of the
    equation than the others.

    :param eq: Partial differential equation

    :param fun: Original function F(x, y, z)

    :param sep: List of separated functions [X(x), u(y, z)]

    :param strategy: Separation strategy. You can choose between additive
        separation ('add') and multiplicative separation ('mul') which is
        default.

    Examples
    ========

    >>> from sympy import E, Eq, Function, pde_separate, Derivative as D
    >>> from sympy.abc import x, t
    >>> u, X, T = map(Function, 'uXT')

    >>> eq = Eq(D(u(x, t), x), E**(u(x, t))*D(u(x, t), t))
    >>> pde_separate(eq, u(x, t), [X(x), T(t)], strategy='add')
    [exp(-X(x))*Derivative(X(x), x), exp(T(t))*Derivative(T(t), t)]

    >>> eq = Eq(D(u(x, t), x, 2), D(u(x, t), t, 2))
    >>> pde_separate(eq, u(x, t), [X(x), T(t)], strategy='mul')
    [Derivative(X(x), (x, 2))/X(x), Derivative(T(t), (t, 2))/T(t)]

    See Also
    ========
    pde_separate_add, pde_separate_mul
    """

    do_add = False
    if strategy == 'add':
        do_add = True
    elif strategy == 'mul':
        do_add = False
    else:
        raise ValueError('Unknown strategy: %s' % strategy)

    if isinstance(eq, Equality):
        if eq.rhs != 0:
            return pde_separate(Eq(eq.lhs - eq.rhs, 0), fun, sep, strategy)
    else:
        return pde_separate(Eq(eq, 0), fun, sep, strategy)

    if eq.rhs != 0:
        raise ValueError("Value should be 0")

    # Handle arguments
    orig_args = list(fun.args)
    subs_args = [arg for s in sep for arg in s.args]

    if do_add:
        functions = reduce(operator.add, sep)
    else:
        functions = reduce(operator.mul, sep)

    # Check whether variables match
    if len(subs_args) != len(orig_args):
        raise ValueError("Variable counts do not match")
    # Check for duplicate arguments like  [X(x), u(x, y)]
    if has_dups(subs_args):
        raise ValueError("Duplicate substitution arguments detected")
    # Check whether the variables match
    if set(orig_args) != set(subs_args):
        raise ValueError("Arguments do not match")

    # Substitute original function with separated...
    result = eq.lhs.subs(fun, functions).doit()

    # Divide by terms when doing multiplicative separation
    if not do_add:
        eq = 0
        for i in result.args:
            eq += i/functions
        result = eq

    svar = subs_args[0]
    dvar = subs_args[1:]
    return _separate(result, svar, dvar)


def pde_separate_add(eq, fun, sep):
    """
    Helper function for searching additive separable solutions.

    Consider an equation of two independent variables x, y and a dependent
    variable w, we look for the product of two functions depending on different
    arguments:

    `w(x, y, z) = X(x) + y(y, z)`

    Examples
    ========

    >>> from sympy import E, Eq, Function, pde_separate_add, Derivative as D
    >>> from sympy.abc import x, t
    >>> u, X, T = map(Function, 'uXT')

    >>> eq = Eq(D(u(x, t), x), E**(u(x, t))*D(u(x, t), t))
    >>> pde_separate_add(eq, u(x, t), [X(x), T(t)])
    [exp(-X(x))*Derivative(X(x), x), exp(T(t))*Derivative(T(t), t)]

    """
    return pde_separate(eq, fun, sep, strategy='add')


def pde_separate_mul(eq, fun, sep):
    """
    Helper function for searching multiplicative separable solutions.

    Consider an equation of two independent variables x, y and a dependent
    variable w, we look for the product of two functions depending on different
    arguments:

    `w(x, y, z) = X(x)*u(y, z)`

    Examples
    ========

    >>> from sympy import Function, Eq, pde_separate_mul, Derivative as D
    >>> from sympy.abc import x, y
    >>> u, X, Y = map(Function, 'uXY')

    >>> eq = Eq(D(u(x, y), x, 2), D(u(x, y), y, 2))
    >>> pde_separate_mul(eq, u(x, y), [X(x), Y(y)])
    [Derivative(X(x), (x, 2))/X(x), Derivative(Y(y), (y, 2))/Y(y)]

    """
    return pde_separate(eq, fun, sep, strategy='mul')


def _separate(eq, dep, others):
    """Separate expression into two parts based on dependencies of variables."""

    # FIRST PASS
    # Extract derivatives depending our separable variable...
    terms = set()
    for term in eq.args:
        if term.is_Mul:
            for i in term.args:
                if i.is_Derivative and not i.has(*others):
                    terms.add(term)
                    continue
        elif term.is_Derivative and not term.has(*others):
            terms.add(term)
    # Find the factor that we need to divide by
    div = set()
    for term in terms:
        ext, sep = term.expand().as_independent(dep)
        # Failed?
        if sep.has(*others):
            return None
        div.add(ext)
    # FIXME: Find lcm() of all the divisors and divide with it, instead of
    # current hack :(
    # https://github.com/sympy/sympy/issues/4597
    if len(div) > 0:
        # double sum required or some tests will fail
        eq = Add(*[simplify(Add(*[term/i for i in div])) for term in eq.args])
    # SECOND PASS - separate the derivatives
    div = set()
    lhs = rhs = 0
    for term in eq.args:
        # Check, whether we have already term with independent variable...
        if not term.has(*others):
            lhs += term
            continue
        # ...otherwise, try to separate
        temp, sep = term.expand().as_independent(dep)
        # Failed?
        if sep.has(*others):
            return None
        # Extract the divisors
        div.add(sep)
        rhs -= term.expand()
    # Do the division
    fulldiv = reduce(operator.add, div)
    lhs = simplify(lhs/fulldiv).expand()
    rhs = simplify(rhs/fulldiv).expand()
    # ...and check whether we were successful :)
    if lhs.has(*others) or rhs.has(dep):
        return None
    return [lhs, rhs]
