"""Utility functions for classifying and solving
ordinary and partial differential equations.

Contains
========
_preprocess
ode_order
_desolve

"""
from sympy.core import Pow
from sympy.core.function import Derivative, AppliedUndef
from sympy.core.relational import Equality
from sympy.core.symbol import Wild

def _preprocess(expr, func=None, hint='_Integral'):
    """Prepare expr for solving by making sure that differentiation
    is done so that only func remains in unevaluated derivatives and
    (if hint does not end with _Integral) that doit is applied to all
    other derivatives. If hint is None, do not do any differentiation.
    (Currently this may cause some simple differential equations to
    fail.)

    In case func is None, an attempt will be made to autodetect the
    function to be solved for.

    >>> from sympy.solvers.deutils import _preprocess
    >>> from sympy import Derivative, Function
    >>> from sympy.abc import x, y, z
    >>> f, g = map(Function, 'fg')

    If f(x)**p == 0 and p>0 then we can solve for f(x)=0
    >>> _preprocess((f(x).diff(x)-4)**5, f(x))
    (Derivative(f(x), x) - 4, f(x))

    Apply doit to derivatives that contain more than the function
    of interest:

    >>> _preprocess(Derivative(f(x) + x, x))
    (Derivative(f(x), x) + 1, f(x))

    Do others if the differentiation variable(s) intersect with those
    of the function of interest or contain the function of interest:

    >>> _preprocess(Derivative(g(x), y, z), f(y))
    (0, f(y))
    >>> _preprocess(Derivative(f(y), z), f(y))
    (0, f(y))

    Do others if the hint does not end in '_Integral' (the default
    assumes that it does):

    >>> _preprocess(Derivative(g(x), y), f(x))
    (Derivative(g(x), y), f(x))
    >>> _preprocess(Derivative(f(x), y), f(x), hint='')
    (0, f(x))

    Do not do any derivatives if hint is None:

    >>> eq = Derivative(f(x) + 1, x) + Derivative(f(x), y)
    >>> _preprocess(eq, f(x), hint=None)
    (Derivative(f(x) + 1, x) + Derivative(f(x), y), f(x))

    If it's not clear what the function of interest is, it must be given:

    >>> eq = Derivative(f(x) + g(x), x)
    >>> _preprocess(eq, g(x))
    (Derivative(f(x), x) + Derivative(g(x), x), g(x))
    >>> try: _preprocess(eq)
    ... except ValueError: print("A ValueError was raised.")
    A ValueError was raised.

    """
    if isinstance(expr, Pow):
        # if f(x)**p=0 then f(x)=0 (p>0)
        if (expr.exp).is_positive:
            expr = expr.base
    derivs = expr.atoms(Derivative)
    if not func:
        funcs = set().union(*[d.atoms(AppliedUndef) for d in derivs])
        if len(funcs) != 1:
            raise ValueError('The function cannot be '
                'automatically detected for %s.' % expr)
        func = funcs.pop()
    fvars = set(func.args)
    if hint is None:
        return expr, func
    reps = [(d, d.doit()) for d in derivs if not hint.endswith('_Integral') or
            d.has(func) or set(d.variables) & fvars]
    eq = expr.subs(reps)
    return eq, func


def ode_order(expr, func):
    """
    Returns the order of a given differential
    equation with respect to func.

    This function is implemented recursively.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.solvers.deutils import ode_order
    >>> from sympy.abc import x
    >>> f, g = map(Function, ['f', 'g'])
    >>> ode_order(f(x).diff(x, 2) + f(x).diff(x)**2 +
    ... f(x).diff(x), f(x))
    2
    >>> ode_order(f(x).diff(x, 2) + g(x).diff(x, 3), f(x))
    2
    >>> ode_order(f(x).diff(x, 2) + g(x).diff(x, 3), g(x))
    3

    """
    a = Wild('a', exclude=[func])
    if expr.match(a):
        return 0

    if isinstance(expr, Derivative):
        if expr.args[0] == func:
            return len(expr.variables)
        else:
            args = expr.args[0].args
            rv = len(expr.variables)
            if args:
                rv += max(ode_order(_, func) for _ in args)
            return rv
    else:
        return max(ode_order(_, func) for _ in expr.args) if expr.args else 0


def _desolve(eq, func=None, hint="default", ics=None, simplify=True, *, prep=True, **kwargs):
    """This is a helper function to dsolve and pdsolve in the ode
    and pde modules.

    If the hint provided to the function is "default", then a dict with
    the following keys are returned

    'func'    - It provides the function for which the differential equation
                has to be solved. This is useful when the expression has
                more than one function in it.

    'default' - The default key as returned by classifier functions in ode
                and pde.py

    'hint'    - The hint given by the user for which the differential equation
                is to be solved. If the hint given by the user is 'default',
                then the value of 'hint' and 'default' is the same.

    'order'   - The order of the function as returned by ode_order

    'match'   - It returns the match as given by the classifier functions, for
                the default hint.

    If the hint provided to the function is not "default" and is not in
    ('all', 'all_Integral', 'best'), then a dict with the above mentioned keys
    is returned along with the keys which are returned when dict in
    classify_ode or classify_pde is set True

    If the hint given is in ('all', 'all_Integral', 'best'), then this function
    returns a nested dict, with the keys, being the set of classified hints
    returned by classifier functions, and the values being the dict of form
    as mentioned above.

    Key 'eq' is a common key to all the above mentioned hints which returns an
    expression if eq given by user is an Equality.

    See Also
    ========
    classify_ode(ode.py)
    classify_pde(pde.py)
    """
    if isinstance(eq, Equality):
        eq = eq.lhs - eq.rhs

    # preprocess the equation and find func if not given
    if prep or func is None:
        eq, func = _preprocess(eq, func)
        prep = False

    # type is an argument passed by the solve functions in ode and pde.py
    # that identifies whether the function caller is an ordinary
    # or partial differential equation. Accordingly corresponding
    # changes are made in the function.
    type = kwargs.get('type', None)
    xi = kwargs.get('xi')
    eta = kwargs.get('eta')
    x0 = kwargs.get('x0', 0)
    terms = kwargs.get('n')

    if type == 'ode':
        from sympy.solvers.ode import classify_ode, allhints
        classifier = classify_ode
        string = 'ODE '
        dummy = ''

    elif type == 'pde':
        from sympy.solvers.pde import classify_pde, allhints
        classifier = classify_pde
        string = 'PDE '
        dummy = 'p'

    # Magic that should only be used internally.  Prevents classify_ode from
    # being called more than it needs to be by passing its results through
    # recursive calls.
    if kwargs.get('classify', True):
        hints = classifier(eq, func, dict=True, ics=ics, xi=xi, eta=eta,
        n=terms, x0=x0, hint=hint, prep=prep)

    else:
        # Here is what all this means:
        #
        # hint:    The hint method given to _desolve() by the user.
        # hints:   The dictionary of hints that match the DE, along with other
        #          information (including the internal pass-through magic).
        # default: The default hint to return, the first hint from allhints
        #          that matches the hint; obtained from classify_ode().
        # match:   Dictionary containing the match dictionary for each hint
        #          (the parts of the DE for solving).  When going through the
        #          hints in "all", this holds the match string for the current
        #          hint.
        # order:   The order of the DE, as determined by ode_order().
        hints = kwargs.get('hint',
                           {'default': hint,
                            hint: kwargs['match'],
                            'order': kwargs['order']})
    if not hints['default']:
        # classify_ode will set hints['default'] to None if no hints match.
        if hint not in allhints and hint != 'default':
            raise ValueError("Hint not recognized: " + hint)
        elif hint not in hints['ordered_hints'] and hint != 'default':
            raise ValueError(string + str(eq) + " does not match hint " + hint)
        # If dsolve can't solve the purely algebraic equation then dsolve will raise
        # ValueError
        elif hints['order'] == 0:
            raise ValueError(
                str(eq) + " is not a solvable differential equation in " + str(func))
        else:
            raise NotImplementedError(dummy + "solve" + ": Cannot solve " + str(eq))
    if hint == 'default':
        return _desolve(eq, func, ics=ics, hint=hints['default'], simplify=simplify,
                      prep=prep, x0=x0, classify=False, order=hints['order'],
                      match=hints[hints['default']], xi=xi, eta=eta, n=terms, type=type)
    elif hint in ('all', 'all_Integral', 'best'):
        retdict = {}
        gethints = set(hints) - {'order', 'default', 'ordered_hints'}
        if hint == 'all_Integral':
            for i in hints:
                if i.endswith('_Integral'):
                    gethints.remove(i[:-len('_Integral')])
            # special cases
            for k in ["1st_homogeneous_coeff_best", "1st_power_series",
                "lie_group", "2nd_power_series_ordinary", "2nd_power_series_regular"]:
                if k in gethints:
                    gethints.remove(k)
        for i in gethints:
            sol = _desolve(eq, func, ics=ics, hint=i, x0=x0, simplify=simplify, prep=prep,
                classify=False, n=terms, order=hints['order'], match=hints[i], type=type)
            retdict[i] = sol
        retdict['all'] = True
        retdict['eq'] = eq
        return retdict
    elif hint not in allhints:  # and hint not in ('default', 'ordered_hints'):
        raise ValueError("Hint not recognized: " + hint)
    elif hint not in hints:
        raise ValueError(string + str(eq) + " does not match hint " + hint)
    else:
        # Key added to identify the hint needed to solve the equation
        hints['hint'] = hint
    hints.update({'func': func, 'eq': eq})
    return hints
