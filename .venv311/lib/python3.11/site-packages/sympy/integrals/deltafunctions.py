from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.functions import DiracDelta, Heaviside
from .integrals import Integral, integrate


def change_mul(node, x):
    """change_mul(node, x)

       Rearranges the operands of a product, bringing to front any simple
       DiracDelta expression.

       Explanation
       ===========

       If no simple DiracDelta expression was found, then all the DiracDelta
       expressions are simplified (using DiracDelta.expand(diracdelta=True, wrt=x)).

       Return: (dirac, new node)
       Where:
         o dirac is either a simple DiracDelta expression or None (if no simple
           expression was found);
         o new node is either a simplified DiracDelta expressions or None (if it
           could not be simplified).

       Examples
       ========

       >>> from sympy import DiracDelta, cos
       >>> from sympy.integrals.deltafunctions import change_mul
       >>> from sympy.abc import x, y
       >>> change_mul(x*y*DiracDelta(x)*cos(x), x)
       (DiracDelta(x), x*y*cos(x))
       >>> change_mul(x*y*DiracDelta(x**2 - 1)*cos(x), x)
       (None, x*y*cos(x)*DiracDelta(x - 1)/2 + x*y*cos(x)*DiracDelta(x + 1)/2)
       >>> change_mul(x*y*DiracDelta(cos(x))*cos(x), x)
       (None, None)

       See Also
       ========

       sympy.functions.special.delta_functions.DiracDelta
       deltaintegrate
    """

    new_args = []
    dirac = None

    #Sorting is needed so that we consistently collapse the same delta;
    #However, we must preserve the ordering of non-commutative terms
    c, nc = node.args_cnc()
    sorted_args = sorted(c, key=default_sort_key)
    sorted_args.extend(nc)

    for arg in sorted_args:
        if arg.is_Pow and isinstance(arg.base, DiracDelta):
            new_args.append(arg.func(arg.base, arg.exp - 1))
            arg = arg.base
        if dirac is None and (isinstance(arg, DiracDelta) and arg.is_simple(x)):
            dirac = arg
        else:
            new_args.append(arg)
    if not dirac:  # there was no simple dirac
        new_args = []
        for arg in sorted_args:
            if isinstance(arg, DiracDelta):
                new_args.append(arg.expand(diracdelta=True, wrt=x))
            elif arg.is_Pow and isinstance(arg.base, DiracDelta):
                new_args.append(arg.func(arg.base.expand(diracdelta=True, wrt=x), arg.exp))
            else:
                new_args.append(arg)
        if new_args != sorted_args:
            nnode = Mul(*new_args).expand()
        else:  # if the node didn't change there is nothing to do
            nnode = None
        return (None, nnode)
    return (dirac, Mul(*new_args))


def deltaintegrate(f, x):
    """
    deltaintegrate(f, x)

    Explanation
    ===========

    The idea for integration is the following:

    - If we are dealing with a DiracDelta expression, i.e. DiracDelta(g(x)),
      we try to simplify it.

      If we could simplify it, then we integrate the resulting expression.
      We already know we can integrate a simplified expression, because only
      simple DiracDelta expressions are involved.

      If we couldn't simplify it, there are two cases:

      1) The expression is a simple expression: we return the integral,
         taking care if we are dealing with a Derivative or with a proper
         DiracDelta.

      2) The expression is not simple (i.e. DiracDelta(cos(x))): we can do
         nothing at all.

    - If the node is a multiplication node having a DiracDelta term:

      First we expand it.

      If the expansion did work, then we try to integrate the expansion.

      If not, we try to extract a simple DiracDelta term, then we have two
      cases:

      1) We have a simple DiracDelta term, so we return the integral.

      2) We didn't have a simple term, but we do have an expression with
         simplified DiracDelta terms, so we integrate this expression.

    Examples
    ========

        >>> from sympy.abc import x, y, z
        >>> from sympy.integrals.deltafunctions import deltaintegrate
        >>> from sympy import sin, cos, DiracDelta
        >>> deltaintegrate(x*sin(x)*cos(x)*DiracDelta(x - 1), x)
        sin(1)*cos(1)*Heaviside(x - 1)
        >>> deltaintegrate(y**2*DiracDelta(x - z)*DiracDelta(y - z), y)
        z**2*DiracDelta(x - z)*Heaviside(y - z)

    See Also
    ========

    sympy.functions.special.delta_functions.DiracDelta
    sympy.integrals.integrals.Integral
    """
    if not f.has(DiracDelta):
        return None

    # g(x) = DiracDelta(h(x))
    if f.func == DiracDelta:
        h = f.expand(diracdelta=True, wrt=x)
        if h == f:  # can't simplify the expression
            #FIXME: the second term tells whether is DeltaDirac or Derivative
            #For integrating derivatives of DiracDelta we need the chain rule
            if f.is_simple(x):
                if (len(f.args) <= 1 or f.args[1] == 0):
                    return Heaviside(f.args[0])
                else:
                    return (DiracDelta(f.args[0], f.args[1] - 1) /
                        f.args[0].as_poly().LC())
        else:  # let's try to integrate the simplified expression
            fh = integrate(h, x)
            return fh
    elif f.is_Mul or f.is_Pow:  # g(x) = a*b*c*f(DiracDelta(h(x)))*d*e
        g = f.expand()
        if f != g:  # the expansion worked
            fh = integrate(g, x)
            if fh is not None and not isinstance(fh, Integral):
                return fh
        else:
            # no expansion performed, try to extract a simple DiracDelta term
            deltaterm, rest_mult = change_mul(f, x)

            if not deltaterm:
                if rest_mult:
                    fh = integrate(rest_mult, x)
                    return fh
            else:
                from sympy.solvers import solve
                deltaterm = deltaterm.expand(diracdelta=True, wrt=x)
                if deltaterm.is_Mul:  # Take out any extracted factors
                    deltaterm, rest_mult_2 = change_mul(deltaterm, x)
                    rest_mult = rest_mult*rest_mult_2
                point = solve(deltaterm.args[0], x)[0]

                # Return the largest hyperreal term left after
                # repeated integration by parts.  For example,
                #
                #   integrate(y*DiracDelta(x, 1),x) == y*DiracDelta(x,0),  not 0
                #
                # This is so Integral(y*DiracDelta(x).diff(x),x).doit()
                # will return y*DiracDelta(x) instead of 0 or DiracDelta(x),
                # both of which are correct everywhere the value is defined
                # but give wrong answers for nested integration.
                n = (0 if len(deltaterm.args)==1 else deltaterm.args[1])
                m = 0
                while n >= 0:
                    r = S.NegativeOne**n*rest_mult.diff(x, n).subs(x, point)
                    if r.is_zero:
                        n -= 1
                        m += 1
                    else:
                        if m == 0:
                            return r*Heaviside(x - point)
                        else:
                            return r*DiracDelta(x,m-1)
                # In some very weak sense, x=0 is still a singularity,
                # but we hope will not be of any practical consequence.
                return S.Zero
    return None
