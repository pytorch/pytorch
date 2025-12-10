r"""
This module contains :py:meth:`~sympy.solvers.ode.dsolve` and different helper
functions that it uses.

:py:meth:`~sympy.solvers.ode.dsolve` solves ordinary differential equations.
See the docstring on the various functions for their uses.  Note that partial
differential equations support is in ``pde.py``.  Note that hint functions
have docstrings describing their various methods, but they are intended for
internal use.  Use ``dsolve(ode, func, hint=hint)`` to solve an ODE using a
specific hint.  See also the docstring on
:py:meth:`~sympy.solvers.ode.dsolve`.

**Functions in this module**

    These are the user functions in this module:

    - :py:meth:`~sympy.solvers.ode.dsolve` - Solves ODEs.
    - :py:meth:`~sympy.solvers.ode.classify_ode` - Classifies ODEs into
      possible hints for :py:meth:`~sympy.solvers.ode.dsolve`.
    - :py:meth:`~sympy.solvers.ode.checkodesol` - Checks if an equation is the
      solution to an ODE.
    - :py:meth:`~sympy.solvers.ode.homogeneous_order` - Returns the
      homogeneous order of an expression.
    - :py:meth:`~sympy.solvers.ode.infinitesimals` - Returns the infinitesimals
      of the Lie group of point transformations of an ODE, such that it is
      invariant.
    - :py:meth:`~sympy.solvers.ode.checkinfsol` - Checks if the given infinitesimals
      are the actual infinitesimals of a first order ODE.

    These are the non-solver helper functions that are for internal use.  The
    user should use the various options to
    :py:meth:`~sympy.solvers.ode.dsolve` to obtain the functionality provided
    by these functions:

    - :py:meth:`~sympy.solvers.ode.ode.odesimp` - Does all forms of ODE
      simplification.
    - :py:meth:`~sympy.solvers.ode.ode.ode_sol_simplicity` - A key function for
      comparing solutions by simplicity.
    - :py:meth:`~sympy.solvers.ode.constantsimp` - Simplifies arbitrary
      constants.
    - :py:meth:`~sympy.solvers.ode.ode.constant_renumber` - Renumber arbitrary
      constants.
    - :py:meth:`~sympy.solvers.ode.ode._handle_Integral` - Evaluate unevaluated
      Integrals.

    See also the docstrings of these functions.

**Currently implemented solver methods**

The following methods are implemented for solving ordinary differential
equations.  See the docstrings of the various hint functions for more
information on each (run ``help(ode)``):

  - 1st order separable differential equations.
  - 1st order differential equations whose coefficients or `dx` and `dy` are
    functions homogeneous of the same order.
  - 1st order exact differential equations.
  - 1st order linear differential equations.
  - 1st order Bernoulli differential equations.
  - Power series solutions for first order differential equations.
  - Lie Group method of solving first order differential equations.
  - 2nd order Liouville differential equations.
  - Power series solutions for second order differential equations
    at ordinary and regular singular points.
  - `n`\th order differential equation that can be solved with algebraic
    rearrangement and integration.
  - `n`\th order linear homogeneous differential equation with constant
    coefficients.
  - `n`\th order linear inhomogeneous differential equation with constant
    coefficients using the method of undetermined coefficients.
  - `n`\th order linear inhomogeneous differential equation with constant
    coefficients using the method of variation of parameters.

**Philosophy behind this module**

This module is designed to make it easy to add new ODE solving methods without
having to mess with the solving code for other methods.  The idea is that
there is a :py:meth:`~sympy.solvers.ode.classify_ode` function, which takes in
an ODE and tells you what hints, if any, will solve the ODE.  It does this
without attempting to solve the ODE, so it is fast.  Each solving method is a
hint, and it has its own function, named ``ode_<hint>``.  That function takes
in the ODE and any match expression gathered by
:py:meth:`~sympy.solvers.ode.classify_ode` and returns a solved result.  If
this result has any integrals in it, the hint function will return an
unevaluated :py:class:`~sympy.integrals.integrals.Integral` class.
:py:meth:`~sympy.solvers.ode.dsolve`, which is the user wrapper function
around all of this, will then call :py:meth:`~sympy.solvers.ode.ode.odesimp` on
the result, which, among other things, will attempt to solve the equation for
the dependent variable (the function we are solving for), simplify the
arbitrary constants in the expression, and evaluate any integrals, if the hint
allows it.

**How to add new solution methods**

If you have an ODE that you want :py:meth:`~sympy.solvers.ode.dsolve` to be
able to solve, try to avoid adding special case code here.  Instead, try
finding a general method that will solve your ODE, as well as others.  This
way, the :py:mod:`~sympy.solvers.ode` module will become more robust, and
unhindered by special case hacks.  WolphramAlpha and Maple's
DETools[odeadvisor] function are two resources you can use to classify a
specific ODE.  It is also better for a method to work with an `n`\th order ODE
instead of only with specific orders, if possible.

To add a new method, there are a few things that you need to do.  First, you
need a hint name for your method.  Try to name your hint so that it is
unambiguous with all other methods, including ones that may not be implemented
yet.  If your method uses integrals, also include a ``hint_Integral`` hint.
If there is more than one way to solve ODEs with your method, include a hint
for each one, as well as a ``<hint>_best`` hint.  Your ``ode_<hint>_best()``
function should choose the best using min with ``ode_sol_simplicity`` as the
key argument.  See
:obj:`~sympy.solvers.ode.single.HomogeneousCoeffBest`, for example.
The function that uses your method will be called ``ode_<hint>()``, so the
hint must only use characters that are allowed in a Python function name
(alphanumeric characters and the underscore '``_``' character).  Include a
function for every hint, except for ``_Integral`` hints
(:py:meth:`~sympy.solvers.ode.dsolve` takes care of those automatically).
Hint names should be all lowercase, unless a word is commonly capitalized
(such as Integral or Bernoulli).  If you have a hint that you do not want to
run with ``all_Integral`` that does not have an ``_Integral`` counterpart (such
as a best hint that would defeat the purpose of ``all_Integral``), you will
need to remove it manually in the :py:meth:`~sympy.solvers.ode.dsolve` code.
See also the :py:meth:`~sympy.solvers.ode.classify_ode` docstring for
guidelines on writing a hint name.

Determine *in general* how the solutions returned by your method compare with
other methods that can potentially solve the same ODEs.  Then, put your hints
in the :py:data:`~sympy.solvers.ode.allhints` tuple in the order that they
should be called.  The ordering of this tuple determines which hints are
default.  Note that exceptions are ok, because it is easy for the user to
choose individual hints with :py:meth:`~sympy.solvers.ode.dsolve`.  In
general, ``_Integral`` variants should go at the end of the list, and
``_best`` variants should go before the various hints they apply to.  For
example, the ``undetermined_coefficients`` hint comes before the
``variation_of_parameters`` hint because, even though variation of parameters
is more general than undetermined coefficients, undetermined coefficients
generally returns cleaner results for the ODEs that it can solve than
variation of parameters does, and it does not require integration, so it is
much faster.

Next, you need to have a match expression or a function that matches the type
of the ODE, which you should put in :py:meth:`~sympy.solvers.ode.classify_ode`
(if the match function is more than just a few lines.  It should match the
ODE without solving for it as much as possible, so that
:py:meth:`~sympy.solvers.ode.classify_ode` remains fast and is not hindered by
bugs in solving code.  Be sure to consider corner cases.  For example, if your
solution method involves dividing by something, make sure you exclude the case
where that division will be 0.

In most cases, the matching of the ODE will also give you the various parts
that you need to solve it.  You should put that in a dictionary (``.match()``
will do this for you), and add that as ``matching_hints['hint'] = matchdict``
in the relevant part of :py:meth:`~sympy.solvers.ode.classify_ode`.
:py:meth:`~sympy.solvers.ode.classify_ode` will then send this to
:py:meth:`~sympy.solvers.ode.dsolve`, which will send it to your function as
the ``match`` argument.  Your function should be named ``ode_<hint>(eq, func,
order, match)`.  If you need to send more information, put it in the ``match``
dictionary.  For example, if you had to substitute in a dummy variable in
:py:meth:`~sympy.solvers.ode.classify_ode` to match the ODE, you will need to
pass it to your function using the `match` dict to access it.  You can access
the independent variable using ``func.args[0]``, and the dependent variable
(the function you are trying to solve for) as ``func.func``.  If, while trying
to solve the ODE, you find that you cannot, raise ``NotImplementedError``.
:py:meth:`~sympy.solvers.ode.dsolve` will catch this error with the ``all``
meta-hint, rather than causing the whole routine to fail.

Add a docstring to your function that describes the method employed.  Like
with anything else in SymPy, you will need to add a doctest to the docstring,
in addition to real tests in ``test_ode.py``.  Try to maintain consistency
with the other hint functions' docstrings.  Add your method to the list at the
top of this docstring.  Also, add your method to ``ode.rst`` in the
``docs/src`` directory, so that the Sphinx docs will pull its docstring into
the main SymPy documentation.  Be sure to make the Sphinx documentation by
running ``make html`` from within the doc directory to verify that the
docstring formats correctly.

If your solution method involves integrating, use :py:obj:`~.Integral` instead of
:py:meth:`~sympy.core.expr.Expr.integrate`.  This allows the user to bypass
hard/slow integration by using the ``_Integral`` variant of your hint.  In
most cases, calling :py:meth:`sympy.core.basic.Basic.doit` will integrate your
solution.  If this is not the case, you will need to write special code in
:py:meth:`~sympy.solvers.ode.ode._handle_Integral`.  Arbitrary constants should be
symbols named ``C1``, ``C2``, and so on.  All solution methods should return
an equality instance.  If you need an arbitrary number of arbitrary constants,
you can use ``constants = numbered_symbols(prefix='C', cls=Symbol, start=1)``.
If it is possible to solve for the dependent function in a general way, do so.
Otherwise, do as best as you can, but do not call solve in your
``ode_<hint>()`` function.  :py:meth:`~sympy.solvers.ode.ode.odesimp` will attempt
to solve the solution for you, so you do not need to do that.  Lastly, if your
ODE has a common simplification that can be applied to your solutions, you can
add a special case in :py:meth:`~sympy.solvers.ode.ode.odesimp` for it.  For
example, solutions returned from the ``1st_homogeneous_coeff`` hints often
have many :obj:`~sympy.functions.elementary.exponential.log` terms, so
:py:meth:`~sympy.solvers.ode.ode.odesimp` calls
:py:meth:`~sympy.simplify.simplify.logcombine` on them (it also helps to write
the arbitrary constant as ``log(C1)`` instead of ``C1`` in this case).  Also
consider common ways that you can rearrange your solution to have
:py:meth:`~sympy.solvers.ode.constantsimp` take better advantage of it.  It is
better to put simplification in :py:meth:`~sympy.solvers.ode.ode.odesimp` than in
your method, because it can then be turned off with the simplify flag in
:py:meth:`~sympy.solvers.ode.dsolve`.  If you have any extraneous
simplification in your function, be sure to only run it using ``if
match.get('simplify', True):``, especially if it can be slow or if it can
reduce the domain of the solution.

Finally, as with every contribution to SymPy, your method will need to be
tested.  Add a test for each method in ``test_ode.py``.  Follow the
conventions there, i.e., test the solver using ``dsolve(eq, f(x),
hint=your_hint)``, and also test the solution using
:py:meth:`~sympy.solvers.ode.checkodesol` (you can put these in a separate
tests and skip/XFAIL if it runs too slow/does not work).  Be sure to call your
hint specifically in :py:meth:`~sympy.solvers.ode.dsolve`, that way the test
will not be broken simply by the introduction of another matching hint.  If your
method works for higher order (>1) ODEs, you will need to run ``sol =
constant_renumber(sol, 'C', 1, order)`` for each solution, where ``order`` is
the order of the ODE.  This is because ``constant_renumber`` renumbers the
arbitrary constants by printing order, which is platform dependent.  Try to
test every corner case of your solver, including a range of orders if it is a
`n`\th order solver, but if your solver is slow, such as if it involves hard
integration, try to keep the test run time down.

Feel free to refactor existing hints to avoid duplicating code or creating
inconsistencies.  If you can show that your method exactly duplicates an
existing method, including in the simplicity and speed of obtaining the
solutions, then you can remove the old, less general method.  The existing
code is tested extensively in ``test_ode.py``, so if anything is broken, one
of those tests will surely fail.

"""

from sympy.core import Add, S, Mul, Pow, oo
from sympy.core.containers import Tuple
from sympy.core.expr import AtomicExpr, Expr
from sympy.core.function import (Function, Derivative, AppliedUndef, diff,
    expand, expand_mul, Subs)
from sympy.core.multidimensional import vectorize
from sympy.core.numbers import nan, zoo, Number
from sympy.core.relational import Equality, Eq
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, Wild, Dummy, symbols
from sympy.core.sympify import sympify
from sympy.core.traversal import preorder_traversal

from sympy.logic.boolalg import (BooleanAtom, BooleanTrue,
                                BooleanFalse)
from sympy.functions import exp, log, sqrt
from sympy.functions.combinatorial.factorials import factorial
from sympy.integrals.integrals import Integral
from sympy.polys import (Poly, terms_gcd, PolynomialError, lcm)
from sympy.polys.polytools import cancel
from sympy.series import Order
from sympy.series.series import series
from sympy.simplify import (collect, logcombine, powsimp,  # type: ignore
    separatevars, simplify, cse)
from sympy.simplify.radsimp import collect_const
from sympy.solvers import checksol, solve

from sympy.utilities import numbered_symbols
from sympy.utilities.iterables import uniq, sift, iterable
from sympy.solvers.deutils import _preprocess, ode_order, _desolve


#: This is a list of hints in the order that they should be preferred by
#: :py:meth:`~sympy.solvers.ode.classify_ode`. In general, hints earlier in the
#: list should produce simpler solutions than those later in the list (for
#: ODEs that fit both).  For now, the order of this list is based on empirical
#: observations by the developers of SymPy.
#:
#: The hint used by :py:meth:`~sympy.solvers.ode.dsolve` for a specific ODE
#: can be overridden (see the docstring).
#:
#: In general, ``_Integral`` hints are grouped at the end of the list, unless
#: there is a method that returns an unevaluable integral most of the time
#: (which go near the end of the list anyway).  ``default``, ``all``,
#: ``best``, and ``all_Integral`` meta-hints should not be included in this
#: list, but ``_best`` and ``_Integral`` hints should be included.
allhints = (
    "factorable",
    "nth_algebraic",
    "separable",
    "1st_exact",
    "1st_linear",
    "Bernoulli",
    "1st_rational_riccati",
    "Riccati_special_minus2",
    "1st_homogeneous_coeff_best",
    "1st_homogeneous_coeff_subs_indep_div_dep",
    "1st_homogeneous_coeff_subs_dep_div_indep",
    "almost_linear",
    "linear_coefficients",
    "separable_reduced",
    "1st_power_series",
    "lie_group",
    "nth_linear_constant_coeff_homogeneous",
    "nth_linear_euler_eq_homogeneous",
    "nth_linear_constant_coeff_undetermined_coefficients",
    "nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients",
    "nth_linear_constant_coeff_variation_of_parameters",
    "nth_linear_euler_eq_nonhomogeneous_variation_of_parameters",
    "Liouville",
    "2nd_linear_airy",
    "2nd_linear_bessel",
    "2nd_hypergeometric",
    "2nd_hypergeometric_Integral",
    "nth_order_reducible",
    "2nd_power_series_ordinary",
    "2nd_power_series_regular",
    "nth_algebraic_Integral",
    "separable_Integral",
    "1st_exact_Integral",
    "1st_linear_Integral",
    "Bernoulli_Integral",
    "1st_homogeneous_coeff_subs_indep_div_dep_Integral",
    "1st_homogeneous_coeff_subs_dep_div_indep_Integral",
    "almost_linear_Integral",
    "linear_coefficients_Integral",
    "separable_reduced_Integral",
    "nth_linear_constant_coeff_variation_of_parameters_Integral",
    "nth_linear_euler_eq_nonhomogeneous_variation_of_parameters_Integral",
    "Liouville_Integral",
    "2nd_nonlinear_autonomous_conserved",
    "2nd_nonlinear_autonomous_conserved_Integral",
    )



def get_numbered_constants(eq, num=1, start=1, prefix='C'):
    """
    Returns a list of constants that do not occur
    in eq already.
    """

    ncs = iter_numbered_constants(eq, start, prefix)
    Cs = [next(ncs) for i in range(num)]
    return (Cs[0] if num == 1 else tuple(Cs))


def iter_numbered_constants(eq, start=1, prefix='C'):
    """
    Returns an iterator of constants that do not occur
    in eq already.
    """

    if isinstance(eq, (Expr, Eq)):
        eq = [eq]
    elif not iterable(eq):
        raise ValueError("Expected Expr or iterable but got %s" % eq)

    atom_set = set().union(*[i.free_symbols for i in eq])
    func_set = set().union(*[i.atoms(Function) for i in eq])
    if func_set:
        atom_set |= {Symbol(str(f.func)) for f in func_set}
    return numbered_symbols(start=start, prefix=prefix, exclude=atom_set)


def dsolve(eq, func=None, hint="default", simplify=True,
    ics= None, xi=None, eta=None, x0=0, n=6, **kwargs):
    r"""
    Solves any (supported) kind of ordinary differential equation and
    system of ordinary differential equations.

    For single ordinary differential equation
    =========================================

    It is classified under this when number of equation in ``eq`` is one.
    **Usage**

        ``dsolve(eq, f(x), hint)`` -> Solve ordinary differential equation
        ``eq`` for function ``f(x)``, using method ``hint``.

    **Details**

        ``eq`` can be any supported ordinary differential equation (see the
            :py:mod:`~sympy.solvers.ode` docstring for supported methods).
            This can either be an :py:class:`~sympy.core.relational.Equality`,
            or an expression, which is assumed to be equal to ``0``.

        ``f(x)`` is a function of one variable whose derivatives in that
            variable make up the ordinary differential equation ``eq``.  In
            many cases it is not necessary to provide this; it will be
            autodetected (and an error raised if it could not be detected).

        ``hint`` is the solving method that you want dsolve to use.  Use
            ``classify_ode(eq, f(x))`` to get all of the possible hints for an
            ODE.  The default hint, ``default``, will use whatever hint is
            returned first by :py:meth:`~sympy.solvers.ode.classify_ode`.  See
            Hints below for more options that you can use for hint.

        ``simplify`` enables simplification by
            :py:meth:`~sympy.solvers.ode.ode.odesimp`.  See its docstring for more
            information.  Turn this off, for example, to disable solving of
            solutions for ``func`` or simplification of arbitrary constants.
            It will still integrate with this hint. Note that the solution may
            contain more arbitrary constants than the order of the ODE with
            this option enabled.

        ``xi`` and ``eta`` are the infinitesimal functions of an ordinary
            differential equation. They are the infinitesimals of the Lie group
            of point transformations for which the differential equation is
            invariant. The user can specify values for the infinitesimals. If
            nothing is specified, ``xi`` and ``eta`` are calculated using
            :py:meth:`~sympy.solvers.ode.infinitesimals` with the help of various
            heuristics.

        ``ics`` is the set of initial/boundary conditions for the differential equation.
          It should be given in the form of ``{f(x0): x1, f(x).diff(x).subs(x, x2):
          x3}`` and so on.  For power series solutions, if no initial
          conditions are specified ``f(0)`` is assumed to be ``C0`` and the power
          series solution is calculated about 0.

        ``x0`` is the point about which the power series solution of a differential
          equation is to be evaluated.

        ``n`` gives the exponent of the dependent variable up to which the power series
          solution of a differential equation is to be evaluated.

    **Hints**

        Aside from the various solving methods, there are also some meta-hints
        that you can pass to :py:meth:`~sympy.solvers.ode.dsolve`:

        ``default``:
                This uses whatever hint is returned first by
                :py:meth:`~sympy.solvers.ode.classify_ode`. This is the
                default argument to :py:meth:`~sympy.solvers.ode.dsolve`.

        ``all``:
                To make :py:meth:`~sympy.solvers.ode.dsolve` apply all
                relevant classification hints, use ``dsolve(ODE, func,
                hint="all")``.  This will return a dictionary of
                ``hint:solution`` terms.  If a hint causes dsolve to raise the
                ``NotImplementedError``, value of that hint's key will be the
                exception object raised.  The dictionary will also include
                some special keys:

                - ``order``: The order of the ODE.  See also
                  :py:meth:`~sympy.solvers.deutils.ode_order` in
                  ``deutils.py``.
                - ``best``: The simplest hint; what would be returned by
                  ``best`` below.
                - ``best_hint``: The hint that would produce the solution
                  given by ``best``.  If more than one hint produces the best
                  solution, the first one in the tuple returned by
                  :py:meth:`~sympy.solvers.ode.classify_ode` is chosen.
                - ``default``: The solution that would be returned by default.
                  This is the one produced by the hint that appears first in
                  the tuple returned by
                  :py:meth:`~sympy.solvers.ode.classify_ode`.

        ``all_Integral``:
                This is the same as ``all``, except if a hint also has a
                corresponding ``_Integral`` hint, it only returns the
                ``_Integral`` hint.  This is useful if ``all`` causes
                :py:meth:`~sympy.solvers.ode.dsolve` to hang because of a
                difficult or impossible integral.  This meta-hint will also be
                much faster than ``all``, because
                :py:meth:`~sympy.core.expr.Expr.integrate` is an expensive
                routine.

        ``best``:
                To have :py:meth:`~sympy.solvers.ode.dsolve` try all methods
                and return the simplest one.  This takes into account whether
                the solution is solvable in the function, whether it contains
                any Integral classes (i.e.  unevaluatable integrals), and
                which one is the shortest in size.

        See also the :py:meth:`~sympy.solvers.ode.classify_ode` docstring for
        more info on hints, and the :py:mod:`~sympy.solvers.ode` docstring for
        a list of all supported hints.

    **Tips**

        - You can declare the derivative of an unknown function this way:

            >>> from sympy import Function, Derivative
            >>> from sympy.abc import x # x is the independent variable
            >>> f = Function("f")(x) # f is a function of x
            >>> # f_ will be the derivative of f with respect to x
            >>> f_ = Derivative(f, x)

        - See ``test_ode.py`` for many tests, which serves also as a set of
          examples for how to use :py:meth:`~sympy.solvers.ode.dsolve`.
        - :py:meth:`~sympy.solvers.ode.dsolve` always returns an
          :py:class:`~sympy.core.relational.Equality` class (except for the
          case when the hint is ``all`` or ``all_Integral``).  If possible, it
          solves the solution explicitly for the function being solved for.
          Otherwise, it returns an implicit solution.
        - Arbitrary constants are symbols named ``C1``, ``C2``, and so on.
        - Because all solutions should be mathematically equivalent, some
          hints may return the exact same result for an ODE. Often, though,
          two different hints will return the same solution formatted
          differently.  The two should be equivalent. Also note that sometimes
          the values of the arbitrary constants in two different solutions may
          not be the same, because one constant may have "absorbed" other
          constants into it.
        - Do ``help(ode.ode_<hintname>)`` to get help more information on a
          specific hint, where ``<hintname>`` is the name of a hint without
          ``_Integral``.

    For system of ordinary differential equations
    =============================================

    **Usage**
        ``dsolve(eq, func)`` -> Solve a system of ordinary differential
        equations ``eq`` for ``func`` being list of functions including
        `x(t)`, `y(t)`, `z(t)` where number of functions in the list depends
        upon the number of equations provided in ``eq``.

    **Details**

        ``eq`` can be any supported system of ordinary differential equations
        This can either be an :py:class:`~sympy.core.relational.Equality`,
        or an expression, which is assumed to be equal to ``0``.

        ``func`` holds ``x(t)`` and ``y(t)`` being functions of one variable which
        together with some of their derivatives make up the system of ordinary
        differential equation ``eq``. It is not necessary to provide this; it
        will be autodetected (and an error raised if it could not be detected).

    **Hints**

        The hints are formed by parameters returned by classify_sysode, combining
        them give hints name used later for forming method name.

    Examples
    ========

    >>> from sympy import Function, dsolve, Eq, Derivative, sin, cos, symbols
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> dsolve(Derivative(f(x), x, x) + 9*f(x), f(x))
    Eq(f(x), C1*sin(3*x) + C2*cos(3*x))

    >>> eq = sin(x)*cos(f(x)) + cos(x)*sin(f(x))*f(x).diff(x)
    >>> dsolve(eq, hint='1st_exact')
    [Eq(f(x), -acos(C1/cos(x)) + 2*pi), Eq(f(x), acos(C1/cos(x)))]
    >>> dsolve(eq, hint='almost_linear')
    [Eq(f(x), -acos(C1/cos(x)) + 2*pi), Eq(f(x), acos(C1/cos(x)))]
    >>> t = symbols('t')
    >>> x, y = symbols('x, y', cls=Function)
    >>> eq = (Eq(Derivative(x(t),t), 12*t*x(t) + 8*y(t)), Eq(Derivative(y(t),t), 21*x(t) + 7*t*y(t)))
    >>> dsolve(eq)
    [Eq(x(t), C1*x0(t) + C2*x0(t)*Integral(8*exp(Integral(7*t, t))*exp(Integral(12*t, t))/x0(t)**2, t)),
    Eq(y(t), C1*y0(t) + C2*(y0(t)*Integral(8*exp(Integral(7*t, t))*exp(Integral(12*t, t))/x0(t)**2, t) +
    exp(Integral(7*t, t))*exp(Integral(12*t, t))/x0(t)))]
    >>> eq = (Eq(Derivative(x(t),t),x(t)*y(t)*sin(t)), Eq(Derivative(y(t),t),y(t)**2*sin(t)))
    >>> dsolve(eq)
    {Eq(x(t), -exp(C1)/(C2*exp(C1) - cos(t))), Eq(y(t), -1/(C1 - cos(t)))}
    """
    if iterable(eq):
        from sympy.solvers.ode.systems import dsolve_system

        # This may have to be changed in future
        # when we have weakly and strongly
        # connected components. This have to
        # changed to show the systems that haven't
        # been solved.
        try:
            sol = dsolve_system(eq, funcs=func, ics=ics, doit=True)
            return sol[0] if len(sol) == 1 else sol
        except NotImplementedError:
            pass

        match = classify_sysode(eq, func)

        eq = match['eq']
        order = match['order']
        func = match['func']
        t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]

        # keep highest order term coefficient positive
        for i in range(len(eq)):
            for func_ in func:
                if isinstance(func_, list):
                    pass
                else:
                    if eq[i].coeff(diff(func[i],t,ode_order(eq[i], func[i]))).is_negative:
                        eq[i] = -eq[i]
        match['eq'] = eq
        if len(set(order.values()))!=1:
            raise ValueError("It solves only those systems of equations whose orders are equal")
        match['order'] = list(order.values())[0]
        def recur_len(l):
            return sum(recur_len(item) if isinstance(item,list) else 1 for item in l)
        if recur_len(func) != len(eq):
            raise ValueError("dsolve() and classify_sysode() work with "
            "number of functions being equal to number of equations")
        if match['type_of_equation'] is None:
            raise NotImplementedError
        else:
            if match['is_linear'] == True:
                solvefunc = globals()['sysode_linear_%(no_of_equation)seq_order%(order)s' % match]
            else:
                solvefunc = globals()['sysode_nonlinear_%(no_of_equation)seq_order%(order)s' % match]
            sols = solvefunc(match)
            if ics:
                constants = Tuple(*sols).free_symbols - Tuple(*eq).free_symbols
                solved_constants = solve_ics(sols, func, constants, ics)
                return [sol.subs(solved_constants) for sol in sols]
            return sols
    else:
        given_hint = hint  # hint given by the user

        # See the docstring of _desolve for more details.
        hints = _desolve(eq, func=func,
            hint=hint, simplify=True, xi=xi, eta=eta, type='ode', ics=ics,
            x0=x0, n=n, **kwargs)
        eq = hints.pop('eq', eq)
        all_ = hints.pop('all', False)
        if all_:
            retdict = {}
            failed_hints = {}
            gethints = classify_ode(eq, dict=True, hint='all')
            orderedhints = gethints['ordered_hints']
            for hint in hints:
                try:
                    rv = _helper_simplify(eq, hint, hints[hint], simplify)
                except NotImplementedError as detail:
                    failed_hints[hint] = detail
                else:
                    retdict[hint] = rv
            func = hints[hint]['func']

            retdict['best'] = min(list(retdict.values()), key=lambda x:
                ode_sol_simplicity(x, func, trysolving=not simplify))
            if given_hint == 'best':
                return retdict['best']
            for i in orderedhints:
                if retdict['best'] == retdict.get(i, None):
                    retdict['best_hint'] = i
                    break
            retdict['default'] = gethints['default']
            retdict['order'] = gethints['order']
            retdict.update(failed_hints)
            return retdict

        else:
            # The key 'hint' stores the hint needed to be solved for.
            hint = hints['hint']
            return _helper_simplify(eq, hint, hints, simplify, ics=ics)


def _helper_simplify(eq, hint, match, simplify=True, ics=None, **kwargs):
    r"""
    Helper function of dsolve that calls the respective
    :py:mod:`~sympy.solvers.ode` functions to solve for the ordinary
    differential equations. This minimizes the computation in calling
    :py:meth:`~sympy.solvers.deutils._desolve` multiple times.
    """
    r = match
    func = r['func']
    order = r['order']
    match = r[hint]

    if isinstance(match, SingleODESolver):
        solvefunc = match
    else:
        solvefunc = globals()['ode_' + hint.removesuffix('_Integral')]

    free = eq.free_symbols
    cons = lambda s: s.free_symbols.difference(free)

    if simplify:
        # odesimp() will attempt to integrate, if necessary, apply constantsimp(),
        # attempt to solve for func, and apply any other hint specific
        # simplifications
        if isinstance(solvefunc, SingleODESolver):
            sols = solvefunc.get_general_solution()
        else:
            sols = solvefunc(eq, func, order, match)
        if iterable(sols):
            rv = []
            for s in sols:
                simp = odesimp(eq, s, func, hint)
                if iterable(simp):
                    rv.extend(simp)
                else:
                    rv.append(simp)
        else:
            rv = odesimp(eq, sols, func, hint)
    else:
        # We still want to integrate (you can disable it separately with the hint)
        if isinstance(solvefunc, SingleODESolver):
            exprs = solvefunc.get_general_solution(simplify=False)
        else:
            match['simplify'] = False  # Some hints can take advantage of this option
            exprs = solvefunc(eq, func, order, match)
        if isinstance(exprs, list):
            rv = [_handle_Integral(expr, func, hint) for expr in exprs]
        else:
            rv = _handle_Integral(exprs, func, hint)

    if isinstance(rv, list):
        assert all(isinstance(i, Eq) for i in rv), rv  # if not => internal error
        if simplify:
            rv = _remove_redundant_solutions(eq, rv, order, func.args[0])
        if len(rv) == 1:
            rv = rv[0]
    if ics and 'power_series' not in hint:
        if isinstance(rv, (Expr, Eq)):
            solved_constants = solve_ics([rv], [r['func']], cons(rv), ics)
            rv = rv.subs(solved_constants)
        else:
            rv1 = []
            for s in rv:
                try:
                    solved_constants = solve_ics([s], [r['func']], cons(s), ics)
                except ValueError:
                    continue
                rv1.append(s.subs(solved_constants))
            if len(rv1) == 1:
                return rv1[0]
            rv = rv1
    return rv


def solve_ics(sols, funcs, constants, ics):
    """
    Solve for the constants given initial conditions

    ``sols`` is a list of solutions.

    ``funcs`` is a list of functions.

    ``constants`` is a list of constants.

    ``ics`` is the set of initial/boundary conditions for the differential
    equation. It should be given in the form of ``{f(x0): x1,
    f(x).diff(x).subs(x, x2):  x3}`` and so on.

    Returns a dictionary mapping constants to values.
    ``solution.subs(constants)`` will replace the constants in ``solution``.

    Example
    =======
    >>> # From dsolve(f(x).diff(x) - f(x), f(x))
    >>> from sympy import symbols, Eq, exp, Function
    >>> from sympy.solvers.ode.ode import solve_ics
    >>> f = Function('f')
    >>> x, C1 = symbols('x C1')
    >>> sols = [Eq(f(x), C1*exp(x))]
    >>> funcs = [f(x)]
    >>> constants = [C1]
    >>> ics = {f(0): 2}
    >>> solved_constants = solve_ics(sols, funcs, constants, ics)
    >>> solved_constants
    {C1: 2}
    >>> sols[0].subs(solved_constants)
    Eq(f(x), 2*exp(x))

    """
    # Assume ics are of the form f(x0): value or Subs(diff(f(x), x, n), (x,
    # x0)): value (currently checked by classify_ode). To solve, replace x
    # with x0, f(x0) with value, then solve for constants. For f^(n)(x0),
    # differentiate the solution n times, so that f^(n)(x) appears.
    x = funcs[0].args[0]
    diff_sols = []
    subs_sols = []
    diff_variables = set()
    for funcarg, value in ics.items():
        if isinstance(funcarg, AppliedUndef):
            x0 = funcarg.args[0]
            matching_func = [f for f in funcs if f.func == funcarg.func][0]
            S = sols
        elif isinstance(funcarg, (Subs, Derivative)):
            if isinstance(funcarg, Subs):
                # Make sure it stays a subs. Otherwise subs below will produce
                # a different looking term.
                funcarg = funcarg.doit()
            if isinstance(funcarg, Subs):
                deriv = funcarg.expr
                x0 = funcarg.point[0]
                variables = funcarg.expr.variables
                matching_func = deriv
            elif isinstance(funcarg, Derivative):
                deriv = funcarg
                x0 = funcarg.variables[0]
                variables = (x,)*len(funcarg.variables)
                matching_func = deriv.subs(x0, x)
            for sol in sols:
                if sol.has(deriv.expr.func):
                    diff_sols.append(Eq(sol.lhs.diff(*variables), sol.rhs.diff(*variables)))
            diff_variables.add(variables)
            S = diff_sols
        else:
            raise NotImplementedError("Unrecognized initial condition")

        for sol in S:
            if sol.has(matching_func):
                sol2 = sol
                sol2 = sol2.subs(x, x0)
                sol2 = sol2.subs(funcarg, value)
                # This check is necessary because of issue #15724
                if not isinstance(sol2, BooleanAtom) or not subs_sols:
                    subs_sols = [s for s in subs_sols if not isinstance(s, BooleanAtom)]
                    subs_sols.append(sol2)

    # TODO: Use solveset here
    try:
        solved_constants = solve(subs_sols, constants, dict=True)
    except NotImplementedError:
        solved_constants = []

    # XXX: We can't differentiate between the solution not existing because of
    # invalid initial conditions, and not existing because solve is not smart
    # enough. If we could use solveset, this might be improvable, but for now,
    # we use NotImplementedError in this case.
    if not solved_constants:
        raise ValueError("Couldn't solve for initial conditions")

    if solved_constants == True:
        raise ValueError("Initial conditions did not produce any solutions for constants. Perhaps they are degenerate.")

    if len(solved_constants) > 1:
        raise NotImplementedError("Initial conditions produced too many solutions for constants")

    return solved_constants[0]

def classify_ode(eq, func=None, dict=False, ics=None, *, prep=True, xi=None, eta=None, n=None, **kwargs):
    r"""
    Returns a tuple of possible :py:meth:`~sympy.solvers.ode.dsolve`
    classifications for an ODE.

    The tuple is ordered so that first item is the classification that
    :py:meth:`~sympy.solvers.ode.dsolve` uses to solve the ODE by default.  In
    general, classifications at the near the beginning of the list will
    produce better solutions faster than those near the end, thought there are
    always exceptions.  To make :py:meth:`~sympy.solvers.ode.dsolve` use a
    different classification, use ``dsolve(ODE, func,
    hint=<classification>)``.  See also the
    :py:meth:`~sympy.solvers.ode.dsolve` docstring for different meta-hints
    you can use.

    If ``dict`` is true, :py:meth:`~sympy.solvers.ode.classify_ode` will
    return a dictionary of ``hint:match`` expression terms. This is intended
    for internal use by :py:meth:`~sympy.solvers.ode.dsolve`.  Note that
    because dictionaries are ordered arbitrarily, this will most likely not be
    in the same order as the tuple.

    You can get help on different hints by executing
    ``help(ode.ode_hintname)``, where ``hintname`` is the name of the hint
    without ``_Integral``.

    See :py:data:`~sympy.solvers.ode.allhints` or the
    :py:mod:`~sympy.solvers.ode` docstring for a list of all supported hints
    that can be returned from :py:meth:`~sympy.solvers.ode.classify_ode`.

    Notes
    =====

    These are remarks on hint names.

    ``_Integral``

        If a classification has ``_Integral`` at the end, it will return the
        expression with an unevaluated :py:class:`~.Integral`
        class in it.  Note that a hint may do this anyway if
        :py:meth:`~sympy.core.expr.Expr.integrate` cannot do the integral,
        though just using an ``_Integral`` will do so much faster.  Indeed, an
        ``_Integral`` hint will always be faster than its corresponding hint
        without ``_Integral`` because
        :py:meth:`~sympy.core.expr.Expr.integrate` is an expensive routine.
        If :py:meth:`~sympy.solvers.ode.dsolve` hangs, it is probably because
        :py:meth:`~sympy.core.expr.Expr.integrate` is hanging on a tough or
        impossible integral.  Try using an ``_Integral`` hint or
        ``all_Integral`` to get it return something.

        Note that some hints do not have ``_Integral`` counterparts. This is
        because :py:func:`~sympy.integrals.integrals.integrate` is not used in
        solving the ODE for those method. For example, `n`\th order linear
        homogeneous ODEs with constant coefficients do not require integration
        to solve, so there is no
        ``nth_linear_homogeneous_constant_coeff_Integrate`` hint. You can
        easily evaluate any unevaluated
        :py:class:`~sympy.integrals.integrals.Integral`\s in an expression by
        doing ``expr.doit()``.

    Ordinals

        Some hints contain an ordinal such as ``1st_linear``.  This is to help
        differentiate them from other hints, as well as from other methods
        that may not be implemented yet. If a hint has ``nth`` in it, such as
        the ``nth_linear`` hints, this means that the method used to applies
        to ODEs of any order.

    ``indep`` and ``dep``

        Some hints contain the words ``indep`` or ``dep``.  These reference
        the independent variable and the dependent function, respectively. For
        example, if an ODE is in terms of `f(x)`, then ``indep`` will refer to
        `x` and ``dep`` will refer to `f`.

    ``subs``

        If a hints has the word ``subs`` in it, it means that the ODE is solved
        by substituting the expression given after the word ``subs`` for a
        single dummy variable.  This is usually in terms of ``indep`` and
        ``dep`` as above.  The substituted expression will be written only in
        characters allowed for names of Python objects, meaning operators will
        be spelled out.  For example, ``indep``/``dep`` will be written as
        ``indep_div_dep``.

    ``coeff``

        The word ``coeff`` in a hint refers to the coefficients of something
        in the ODE, usually of the derivative terms.  See the docstring for
        the individual methods for more info (``help(ode)``).  This is
        contrast to ``coefficients``, as in ``undetermined_coefficients``,
        which refers to the common name of a method.

    ``_best``

        Methods that have more than one fundamental way to solve will have a
        hint for each sub-method and a ``_best`` meta-classification. This
        will evaluate all hints and return the best, using the same
        considerations as the normal ``best`` meta-hint.


    Examples
    ========

    >>> from sympy import Function, classify_ode, Eq
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> classify_ode(Eq(f(x).diff(x), 0), f(x))
    ('nth_algebraic',
    'separable',
    '1st_exact',
    '1st_linear',
    'Bernoulli',
    '1st_homogeneous_coeff_best',
    '1st_homogeneous_coeff_subs_indep_div_dep',
    '1st_homogeneous_coeff_subs_dep_div_indep',
    '1st_power_series', 'lie_group', 'nth_linear_constant_coeff_homogeneous',
    'nth_linear_euler_eq_homogeneous',
    'nth_algebraic_Integral', 'separable_Integral', '1st_exact_Integral',
    '1st_linear_Integral', 'Bernoulli_Integral',
    '1st_homogeneous_coeff_subs_indep_div_dep_Integral',
    '1st_homogeneous_coeff_subs_dep_div_indep_Integral')
    >>> classify_ode(f(x).diff(x, 2) + 3*f(x).diff(x) + 2*f(x) - 4)
    ('factorable', 'nth_linear_constant_coeff_undetermined_coefficients',
    'nth_linear_constant_coeff_variation_of_parameters',
    'nth_linear_constant_coeff_variation_of_parameters_Integral')

    """
    ics = sympify(ics)

    if func and len(func.args) != 1:
        raise ValueError("dsolve() and classify_ode() only "
        "work with functions of one variable, not %s" % func)

    if isinstance(eq, Equality):
        eq = eq.lhs - eq.rhs

    # Some methods want the unprocessed equation
    eq_orig = eq

    if prep or func is None:
        eq, func_ = _preprocess(eq, func)
        if func is None:
            func = func_
    x = func.args[0]
    f = func.func
    y = Dummy('y')
    terms = 5 if n is None else n

    order = ode_order(eq, f(x))
    # hint:matchdict or hint:(tuple of matchdicts)
    # Also will contain "default":<default hint> and "order":order items.
    matching_hints = {"order": order}

    df = f(x).diff(x)
    a = Wild('a', exclude=[f(x)])
    d = Wild('d', exclude=[df, f(x).diff(x, 2)])
    e = Wild('e', exclude=[df])
    n = Wild('n', exclude=[x, f(x), df])
    c1 = Wild('c1', exclude=[x])
    a3 = Wild('a3', exclude=[f(x), df, f(x).diff(x, 2)])
    b3 = Wild('b3', exclude=[f(x), df, f(x).diff(x, 2)])
    c3 = Wild('c3', exclude=[f(x), df, f(x).diff(x, 2)])
    boundary = {}  # Used to extract initial conditions
    C1 = Symbol("C1")

    # Preprocessing to get the initial conditions out
    if ics is not None:
        for funcarg in ics:
            # Separating derivatives
            if isinstance(funcarg, (Subs, Derivative)):
                # f(x).diff(x).subs(x, 0) is a Subs, but f(x).diff(x).subs(x,
                # y) is a Derivative
                if isinstance(funcarg, Subs):
                    deriv = funcarg.expr
                    old = funcarg.variables[0]
                    new = funcarg.point[0]
                elif isinstance(funcarg, Derivative):
                    deriv = funcarg
                    # No information on this. Just assume it was x
                    old = x
                    new = funcarg.variables[0]

                if (isinstance(deriv, Derivative) and isinstance(deriv.args[0],
                    AppliedUndef) and deriv.args[0].func == f and
                    len(deriv.args[0].args) == 1 and old == x and not
                    new.has(x) and all(i == deriv.variables[0] for i in
                    deriv.variables) and x not in ics[funcarg].free_symbols):

                    dorder = ode_order(deriv, x)
                    temp = 'f' + str(dorder)
                    boundary.update({temp: new, temp + 'val': ics[funcarg]})
                else:
                    raise ValueError("Invalid boundary conditions for Derivatives")


            # Separating functions
            elif isinstance(funcarg, AppliedUndef):
                if (funcarg.func == f and len(funcarg.args) == 1 and
                    not funcarg.args[0].has(x) and x not in ics[funcarg].free_symbols):
                    boundary.update({'f0': funcarg.args[0], 'f0val': ics[funcarg]})
                else:
                    raise ValueError("Invalid boundary conditions for Function")

            else:
                raise ValueError("Enter boundary conditions of the form ics={f(point): value, f(x).diff(x, order).subs(x, point): value}")

    ode = SingleODEProblem(eq_orig, func, x, prep=prep, xi=xi, eta=eta)
    user_hint = kwargs.get('hint', 'default')
    # Used when dsolve is called without an explicit hint.
    # We exit early to return the first valid match
    early_exit = (user_hint=='default')
    user_hint = user_hint.removesuffix('_Integral')
    user_map = solver_map
    # An explicit hint has been given to dsolve
    # Skip matching code for other hints
    if user_hint not in ['default', 'all', 'all_Integral', 'best'] and user_hint in solver_map:
        user_map = {user_hint: solver_map[user_hint]}

    for hint in user_map:
        solver = user_map[hint](ode)
        if solver.matches():
            matching_hints[hint] = solver
            if user_map[hint].has_integral:
                matching_hints[hint + "_Integral"] = solver
            if dict and early_exit:
                matching_hints["default"] = hint
                return matching_hints

    eq = expand(eq)
    # Precondition to try remove f(x) from highest order derivative
    reduced_eq = None
    if eq.is_Add:
        deriv_coef = eq.coeff(f(x).diff(x, order))
        if deriv_coef not in (1, 0):
            r = deriv_coef.match(a*f(x)**c1)
            if r and r[c1]:
                den = f(x)**r[c1]
                reduced_eq = Add(*[arg/den for arg in eq.args])
    if not reduced_eq:
        reduced_eq = eq

    if order == 1:

        # NON-REDUCED FORM OF EQUATION matches
        r = collect(eq, df, exact=True).match(d + e * df)
        if r:
            r['d'] = d
            r['e'] = e
            r['y'] = y
            r[d] = r[d].subs(f(x), y)
            r[e] = r[e].subs(f(x), y)

            # FIRST ORDER POWER SERIES WHICH NEEDS INITIAL CONDITIONS
            # TODO: Hint first order series should match only if d/e is analytic.
            # For now, only d/e and (d/e).diff(arg) is checked for existence at
            # at a given point.
            # This is currently done internally in ode_1st_power_series.
            point = boundary.get('f0', 0)
            value = boundary.get('f0val', C1)
            check = cancel(r[d]/r[e])
            check1 = check.subs({x: point, y: value})
            if not check1.has(oo) and not check1.has(zoo) and \
                not check1.has(nan) and not check1.has(-oo):
                check2 = (check1.diff(x)).subs({x: point, y: value})
                if not check2.has(oo) and not check2.has(zoo) and \
                    not check2.has(nan) and not check2.has(-oo):
                    rseries = r.copy()
                    rseries.update({'terms': terms, 'f0': point, 'f0val': value})
                    matching_hints["1st_power_series"] = rseries

    elif order == 2:
        # Homogeneous second order differential equation of the form
        # a3*f(x).diff(x, 2) + b3*f(x).diff(x) + c3
        # It has a definite power series solution at point x0 if, b3/a3 and c3/a3
        # are analytic at x0.
        deq = a3*(f(x).diff(x, 2)) + b3*df + c3*f(x)
        r = collect(reduced_eq,
            [f(x).diff(x, 2), f(x).diff(x), f(x)]).match(deq)
        ordinary = False
        if r:
            if not all(r[key].is_polynomial() for key in r):
                n, d = reduced_eq.as_numer_denom()
                reduced_eq = expand(n)
                r = collect(reduced_eq,
                    [f(x).diff(x, 2), f(x).diff(x), f(x)]).match(deq)
        if r and r[a3] != 0:
            p = cancel(r[b3]/r[a3])  # Used below
            q = cancel(r[c3]/r[a3])  # Used below
            point = kwargs.get('x0', 0)
            check = p.subs(x, point)
            if not check.has(oo, nan, zoo, -oo):
                check = q.subs(x, point)
                if not check.has(oo, nan, zoo, -oo):
                    ordinary = True
                    r.update({'a3': a3, 'b3': b3, 'c3': c3, 'x0': point, 'terms': terms})
                    matching_hints["2nd_power_series_ordinary"] = r

            # Checking if the differential equation has a regular singular point
            # at x0. It has a regular singular point at x0, if (b3/a3)*(x - x0)
            # and (c3/a3)*((x - x0)**2) are analytic at x0.
            if not ordinary:
                p = cancel((x - point)*p)
                check = p.subs(x, point)
                if not check.has(oo, nan, zoo, -oo):
                    q = cancel(((x - point)**2)*q)
                    check = q.subs(x, point)
                    if not check.has(oo, nan, zoo, -oo):
                        coeff_dict = {'p': p, 'q': q, 'x0': point, 'terms': terms}
                        matching_hints["2nd_power_series_regular"] = coeff_dict


    # Order keys based on allhints.
    retlist = [i for i in allhints if i in matching_hints]
    if dict:
        # Dictionaries are ordered arbitrarily, so make note of which
        # hint would come first for dsolve().  Use an ordered dict in Py 3.
        matching_hints["default"] = retlist[0] if retlist else None
        matching_hints["ordered_hints"] = tuple(retlist)
        return matching_hints
    else:
        return tuple(retlist)


def classify_sysode(eq, funcs=None, **kwargs):
    r"""
    Returns a dictionary of parameter names and values that define the system
    of ordinary differential equations in ``eq``.
    The parameters are further used in
    :py:meth:`~sympy.solvers.ode.dsolve` for solving that system.

    Some parameter names and values are:

    'is_linear' (boolean), which tells whether the given system is linear.
    Note that "linear" here refers to the operator: terms such as ``x*diff(x,t)`` are
    nonlinear, whereas terms like ``sin(t)*diff(x,t)`` are still linear operators.

    'func' (list) contains the :py:class:`~sympy.core.function.Function`s that
    appear with a derivative in the ODE, i.e. those that we are trying to solve
    the ODE for.

    'order' (dict) with the maximum derivative for each element of the 'func'
    parameter.

    'func_coeff' (dict or Matrix) with the coefficient for each triple ``(equation number,
    function, order)```. The coefficients are those subexpressions that do not
    appear in 'func', and hence can be considered constant for purposes of ODE
    solving. The value of this parameter can also be a  Matrix if the system of ODEs are
    linear first order of the form X' = AX where X is the vector of dependent variables.
    Here, this function returns the coefficient matrix A.

    'eq' (list) with the equations from ``eq``, sympified and transformed into
    expressions (we are solving for these expressions to be zero).

    'no_of_equations' (int) is the number of equations (same as ``len(eq)``).

    'type_of_equation' (string) is an internal classification of the type of
    ODE.

    'is_constant' (boolean), which tells if the system of ODEs is constant coefficient
    or not. This key is temporary addition for now and is in the match dict only when
    the system of ODEs is linear first order constant coefficient homogeneous. So, this
    key's value is True for now if it is available else it does not exist.

    'is_homogeneous' (boolean), which tells if the system of ODEs is homogeneous. Like the
    key 'is_constant', this key is a temporary addition and it is True since this key value
    is available only when the system is linear first order constant coefficient homogeneous.

    References
    ==========
    -https://eqworld.ipmnet.ru/en/solutions/sysode/sode-toc1.htm
    -A. D. Polyanin and A. V. Manzhirov, Handbook of Mathematics for Engineers and Scientists

    Examples
    ========

    >>> from sympy import Function, Eq, symbols, diff
    >>> from sympy.solvers.ode.ode import classify_sysode
    >>> from sympy.abc import t
    >>> f, x, y = symbols('f, x, y', cls=Function)
    >>> k, l, m, n = symbols('k, l, m, n', Integer=True)
    >>> x1 = diff(x(t), t) ; y1 = diff(y(t), t)
    >>> x2 = diff(x(t), t, t) ; y2 = diff(y(t), t, t)
    >>> eq = (Eq(x1, 12*x(t) - 6*y(t)), Eq(y1, 11*x(t) + 3*y(t)))
    >>> classify_sysode(eq)
    {'eq': [-12*x(t) + 6*y(t) + Derivative(x(t), t), -11*x(t) - 3*y(t) + Derivative(y(t), t)], 'func': [x(t), y(t)],
     'func_coeff': {(0, x(t), 0): -12, (0, x(t), 1): 1, (0, y(t), 0): 6, (0, y(t), 1): 0, (1, x(t), 0): -11, (1, x(t), 1): 0, (1, y(t), 0): -3, (1, y(t), 1): 1}, 'is_linear': True, 'no_of_equation': 2, 'order': {x(t): 1, y(t): 1}, 'type_of_equation': None}
    >>> eq = (Eq(diff(x(t),t), 5*t*x(t) + t**2*y(t) + 2), Eq(diff(y(t),t), -t**2*x(t) + 5*t*y(t)))
    >>> classify_sysode(eq)
    {'eq': [-t**2*y(t) - 5*t*x(t) + Derivative(x(t), t) - 2, t**2*x(t) - 5*t*y(t) + Derivative(y(t), t)],
     'func': [x(t), y(t)], 'func_coeff': {(0, x(t), 0): -5*t, (0, x(t), 1): 1, (0, y(t), 0): -t**2, (0, y(t), 1): 0,
     (1, x(t), 0): t**2, (1, x(t), 1): 0, (1, y(t), 0): -5*t, (1, y(t), 1): 1}, 'is_linear': True, 'no_of_equation': 2,
      'order': {x(t): 1, y(t): 1}, 'type_of_equation': None}

    """

    # Sympify equations and convert iterables of equations into
    # a list of equations
    def _sympify(eq):
        return list(map(sympify, eq if iterable(eq) else [eq]))

    eq, funcs = (_sympify(w) for w in [eq, funcs])
    for i, fi in enumerate(eq):
        if isinstance(fi, Equality):
            eq[i] = fi.lhs - fi.rhs

    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    matching_hints = {"no_of_equation":i+1}
    matching_hints['eq'] = eq
    if i==0:
        raise ValueError("classify_sysode() works for systems of ODEs. "
        "For scalar ODEs, classify_ode should be used")

    # find all the functions if not given
    order = {}
    if funcs==[None]:
        funcs = _extract_funcs(eq)

    funcs = list(set(funcs))
    if len(funcs) != len(eq):
        raise ValueError("Number of functions given is not equal to the number of equations %s" % funcs)

    # This logic of list of lists in funcs to
    # be replaced later.
    func_dict = {}
    for func in funcs:
        if not order.get(func, False):
            max_order = 0
            for i, eqs_ in enumerate(eq):
                order_ = ode_order(eqs_,func)
                if max_order < order_:
                    max_order = order_
                    eq_no = i
            if eq_no in func_dict:
                func_dict[eq_no] = [func_dict[eq_no], func]
            else:
                func_dict[eq_no] = func
            order[func] = max_order

    funcs = [func_dict[i] for i in range(len(func_dict))]
    matching_hints['func'] = funcs
    for func in funcs:
        if isinstance(func, list):
            for func_elem in func:
                if len(func_elem.args) != 1:
                    raise ValueError("dsolve() and classify_sysode() work with "
                    "functions of one variable only, not %s" % func)
        else:
            if func and len(func.args) != 1:
                raise ValueError("dsolve() and classify_sysode() work with "
                "functions of one variable only, not %s" % func)

    # find the order of all equation in system of odes
    matching_hints["order"] = order

    # find coefficients of terms f(t), diff(f(t),t) and higher derivatives
    # and similarly for other functions g(t), diff(g(t),t) in all equations.
    # Here j denotes the equation number, funcs[l] denotes the function about
    # which we are talking about and k denotes the order of function funcs[l]
    # whose coefficient we are calculating.
    def linearity_check(eqs, j, func, is_linear_):
        for k in range(order[func] + 1):
            func_coef[j, func, k] = collect(eqs.expand(), [diff(func, t, k)]).coeff(diff(func, t, k))
            if is_linear_ == True:
                if func_coef[j, func, k] == 0:
                    if k == 0:
                        coef = eqs.as_independent(func, as_Add=True)[1]
                        for xr in range(1, ode_order(eqs,func) + 1):
                            coef -= eqs.as_independent(diff(func, t, xr), as_Add=True)[1]
                        if coef != 0:
                            is_linear_ = False
                    else:
                        if eqs.as_independent(diff(func, t, k), as_Add=True)[1]:
                            is_linear_ = False
                else:
                    for func_ in funcs:
                        if isinstance(func_, list):
                            for elem_func_ in func_:
                                dep = func_coef[j, func, k].as_independent(elem_func_, as_Add=True)[1]
                                if dep != 0:
                                    is_linear_ = False
                        else:
                            dep = func_coef[j, func, k].as_independent(func_, as_Add=True)[1]
                            if dep != 0:
                                is_linear_ = False
        return is_linear_

    func_coef = {}
    is_linear = True
    for j, eqs in enumerate(eq):
        for func in funcs:
            if isinstance(func, list):
                for func_elem in func:
                    is_linear = linearity_check(eqs, j, func_elem, is_linear)
            else:
                is_linear = linearity_check(eqs, j, func, is_linear)
    matching_hints['func_coeff'] = func_coef
    matching_hints['is_linear'] = is_linear


    if len(set(order.values())) == 1:
        order_eq = list(matching_hints['order'].values())[0]
        if matching_hints['is_linear'] == True:
            if matching_hints['no_of_equation'] == 2:
                if order_eq == 1:
                    type_of_equation = check_linear_2eq_order1(eq, funcs, func_coef)
                else:
                    type_of_equation = None
            # If the equation does not match up with any of the
            # general case solvers in systems.py and the number
            # of equations is greater than 2, then NotImplementedError
            # should be raised.
            else:
                type_of_equation = None

        else:
            if matching_hints['no_of_equation'] == 2:
                if order_eq == 1:
                    type_of_equation = check_nonlinear_2eq_order1(eq, funcs, func_coef)
                else:
                    type_of_equation = None
            elif matching_hints['no_of_equation'] == 3:
                if order_eq == 1:
                    type_of_equation = check_nonlinear_3eq_order1(eq, funcs, func_coef)
                else:
                    type_of_equation = None
            else:
                type_of_equation = None
    else:
        type_of_equation = None

    matching_hints['type_of_equation'] = type_of_equation

    return matching_hints


def check_linear_2eq_order1(eq, func, func_coef):
    x = func[0].func
    y = func[1].func
    fc = func_coef
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    r = {}
    # for equations Eq(a1*diff(x(t),t), b1*x(t) + c1*y(t) + d1)
    # and Eq(a2*diff(y(t),t), b2*x(t) + c2*y(t) + d2)
    r['a1'] = fc[0,x(t),1] ; r['a2'] = fc[1,y(t),1]
    r['b1'] = -fc[0,x(t),0]/fc[0,x(t),1] ; r['b2'] = -fc[1,x(t),0]/fc[1,y(t),1]
    r['c1'] = -fc[0,y(t),0]/fc[0,x(t),1] ; r['c2'] = -fc[1,y(t),0]/fc[1,y(t),1]
    forcing = [S.Zero,S.Zero]
    for i in range(2):
        for j in Add.make_args(eq[i]):
            if not j.has(x(t), y(t)):
                forcing[i] += j
    if not (forcing[0].has(t) or forcing[1].has(t)):
        # We can handle homogeneous case and simple constant forcings
        r['d1'] = forcing[0]
        r['d2'] = forcing[1]
    else:
        # Issue #9244: nonhomogeneous linear systems are not supported
        return None

    # Conditions to check for type 6 whose equations are Eq(diff(x(t),t), f(t)*x(t) + g(t)*y(t)) and
    # Eq(diff(y(t),t), a*[f(t) + a*h(t)]x(t) + a*[g(t) - h(t)]*y(t))
    p = 0
    q = 0
    p1 = cancel(r['b2']/(cancel(r['b2']/r['c2']).as_numer_denom()[0]))
    p2 = cancel(r['b1']/(cancel(r['b1']/r['c1']).as_numer_denom()[0]))
    for n, i in enumerate([p1, p2]):
        for j in Mul.make_args(collect_const(i)):
            if not j.has(t):
                q = j
            if q and n==0:
                if ((r['b2']/j - r['b1'])/(r['c1'] - r['c2']/j)) == j:
                    p = 1
            elif q and n==1:
                if ((r['b1']/j - r['b2'])/(r['c2'] - r['c1']/j)) == j:
                    p = 2
    # End of condition for type 6

    if r['d1']!=0 or r['d2']!=0:
        return None
    else:
        if not any(r[k].has(t) for k in 'a1 a2 b1 b2 c1 c2'.split()):
            return None
        else:
            r['b1'] = r['b1']/r['a1'] ; r['b2'] = r['b2']/r['a2']
            r['c1'] = r['c1']/r['a1'] ; r['c2'] = r['c2']/r['a2']
            if p:
                return "type6"
            else:
                # Equations for type 7 are Eq(diff(x(t),t), f(t)*x(t) + g(t)*y(t)) and Eq(diff(y(t),t), h(t)*x(t) + p(t)*y(t))
                return "type7"
def check_nonlinear_2eq_order1(eq, func, func_coef):
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    f = Wild('f')
    g = Wild('g')
    u, v = symbols('u, v', cls=Dummy)
    def check_type(x, y):
        r1 = eq[0].match(t*diff(x(t),t) - x(t) + f)
        r2 = eq[1].match(t*diff(y(t),t) - y(t) + g)
        if not (r1 and r2):
            r1 = eq[0].match(diff(x(t),t) - x(t)/t + f/t)
            r2 = eq[1].match(diff(y(t),t) - y(t)/t + g/t)
        if not (r1 and r2):
            r1 = (-eq[0]).match(t*diff(x(t),t) - x(t) + f)
            r2 = (-eq[1]).match(t*diff(y(t),t) - y(t) + g)
        if not (r1 and r2):
            r1 = (-eq[0]).match(diff(x(t),t) - x(t)/t + f/t)
            r2 = (-eq[1]).match(diff(y(t),t) - y(t)/t + g/t)
        if r1 and r2 and not (r1[f].subs(diff(x(t),t),u).subs(diff(y(t),t),v).has(t) \
        or r2[g].subs(diff(x(t),t),u).subs(diff(y(t),t),v).has(t)):
            return 'type5'
        else:
            return None
    for func_ in func:
        if isinstance(func_, list):
            x = func[0][0].func
            y = func[0][1].func
            eq_type = check_type(x, y)
            if not eq_type:
                eq_type = check_type(y, x)
            return eq_type
    x = func[0].func
    y = func[1].func
    fc = func_coef
    n = Wild('n', exclude=[x(t),y(t)])
    f1 = Wild('f1', exclude=[v,t])
    f2 = Wild('f2', exclude=[v,t])
    g1 = Wild('g1', exclude=[u,t])
    g2 = Wild('g2', exclude=[u,t])
    for i in range(2):
        eqs = 0
        for terms in Add.make_args(eq[i]):
            eqs += terms/fc[i,func[i],1]
        eq[i] = eqs
    r = eq[0].match(diff(x(t),t) - x(t)**n*f)
    if r:
        g = (diff(y(t),t) - eq[1])/r[f]
    if r and not (g.has(x(t)) or g.subs(y(t),v).has(t) or r[f].subs(x(t),u).subs(y(t),v).has(t)):
        return 'type1'
    r = eq[0].match(diff(x(t),t) - exp(n*x(t))*f)
    if r:
        g = (diff(y(t),t) - eq[1])/r[f]
    if r and not (g.has(x(t)) or g.subs(y(t),v).has(t) or r[f].subs(x(t),u).subs(y(t),v).has(t)):
        return 'type2'
    g = Wild('g')
    r1 = eq[0].match(diff(x(t),t) - f)
    r2 = eq[1].match(diff(y(t),t) - g)
    if r1 and r2 and not (r1[f].subs(x(t),u).subs(y(t),v).has(t) or \
    r2[g].subs(x(t),u).subs(y(t),v).has(t)):
        return 'type3'
    r1 = eq[0].match(diff(x(t),t) - f)
    r2 = eq[1].match(diff(y(t),t) - g)
    num, den = (
        (r1[f].subs(x(t),u).subs(y(t),v))/
        (r2[g].subs(x(t),u).subs(y(t),v))).as_numer_denom()
    R1 = num.match(f1*g1)
    R2 = den.match(f2*g2)
    # phi = (r1[f].subs(x(t),u).subs(y(t),v))/num
    if R1 and R2:
        return 'type4'
    return None


def check_nonlinear_2eq_order2(eq, func, func_coef):
    return None

def check_nonlinear_3eq_order1(eq, func, func_coef):
    x = func[0].func
    y = func[1].func
    z = func[2].func
    fc = func_coef
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    u, v, w = symbols('u, v, w', cls=Dummy)
    a = Wild('a', exclude=[x(t), y(t), z(t), t])
    b = Wild('b', exclude=[x(t), y(t), z(t), t])
    c = Wild('c', exclude=[x(t), y(t), z(t), t])
    f = Wild('f')
    F1 = Wild('F1')
    F2 = Wild('F2')
    F3 = Wild('F3')
    for i in range(3):
        eqs = 0
        for terms in Add.make_args(eq[i]):
            eqs += terms/fc[i,func[i],1]
        eq[i] = eqs
    r1 = eq[0].match(diff(x(t),t) - a*y(t)*z(t))
    r2 = eq[1].match(diff(y(t),t) - b*z(t)*x(t))
    r3 = eq[2].match(diff(z(t),t) - c*x(t)*y(t))
    if r1 and r2 and r3:
        num1, den1 = r1[a].as_numer_denom()
        num2, den2 = r2[b].as_numer_denom()
        num3, den3 = r3[c].as_numer_denom()
        if solve([num1*u-den1*(v-w), num2*v-den2*(w-u), num3*w-den3*(u-v)],[u, v]):
            return 'type1'
    r = eq[0].match(diff(x(t),t) - y(t)*z(t)*f)
    if r:
        r1 = collect_const(r[f]).match(a*f)
        r2 = ((diff(y(t),t) - eq[1])/r1[f]).match(b*z(t)*x(t))
        r3 = ((diff(z(t),t) - eq[2])/r1[f]).match(c*x(t)*y(t))
    if r1 and r2 and r3:
        num1, den1 = r1[a].as_numer_denom()
        num2, den2 = r2[b].as_numer_denom()
        num3, den3 = r3[c].as_numer_denom()
        if solve([num1*u-den1*(v-w), num2*v-den2*(w-u), num3*w-den3*(u-v)],[u, v]):
            return 'type2'
    r = eq[0].match(diff(x(t),t) - (F2-F3))
    if r:
        r1 = collect_const(r[F2]).match(c*F2)
        r1.update(collect_const(r[F3]).match(b*F3))
        if r1:
            if eq[1].has(r1[F2]) and not eq[1].has(r1[F3]):
                r1[F2], r1[F3] = r1[F3], r1[F2]
                r1[c], r1[b] = -r1[b], -r1[c]
            r2 = eq[1].match(diff(y(t),t) - a*r1[F3] + r1[c]*F1)
        if r2:
            r3 = (eq[2] == diff(z(t),t) - r1[b]*r2[F1] + r2[a]*r1[F2])
        if r1 and r2 and r3:
            return 'type3'
    r = eq[0].match(diff(x(t),t) - z(t)*F2 + y(t)*F3)
    if r:
        r1 = collect_const(r[F2]).match(c*F2)
        r1.update(collect_const(r[F3]).match(b*F3))
        if r1:
            if eq[1].has(r1[F2]) and not eq[1].has(r1[F3]):
                r1[F2], r1[F3] = r1[F3], r1[F2]
                r1[c], r1[b] = -r1[b], -r1[c]
            r2 = (diff(y(t),t) - eq[1]).match(a*x(t)*r1[F3] - r1[c]*z(t)*F1)
        if r2:
            r3 = (diff(z(t),t) - eq[2] == r1[b]*y(t)*r2[F1] - r2[a]*x(t)*r1[F2])
        if r1 and r2 and r3:
            return 'type4'
    r = (diff(x(t),t) - eq[0]).match(x(t)*(F2 - F3))
    if r:
        r1 = collect_const(r[F2]).match(c*F2)
        r1.update(collect_const(r[F3]).match(b*F3))
        if r1:
            if eq[1].has(r1[F2]) and not eq[1].has(r1[F3]):
                r1[F2], r1[F3] = r1[F3], r1[F2]
                r1[c], r1[b] = -r1[b], -r1[c]
            r2 = (diff(y(t),t) - eq[1]).match(y(t)*(a*r1[F3] - r1[c]*F1))
        if r2:
            r3 = (diff(z(t),t) - eq[2] == z(t)*(r1[b]*r2[F1] - r2[a]*r1[F2]))
        if r1 and r2 and r3:
            return 'type5'
    return None


def check_nonlinear_3eq_order2(eq, func, func_coef):
    return None


@vectorize(0)
def odesimp(ode, eq, func, hint):
    r"""
    Simplifies solutions of ODEs, including trying to solve for ``func`` and
    running :py:meth:`~sympy.solvers.ode.constantsimp`.

    It may use knowledge of the type of solution that the hint returns to
    apply additional simplifications.

    It also attempts to integrate any :py:class:`~sympy.integrals.integrals.Integral`\s
    in the expression, if the hint is not an ``_Integral`` hint.

    This function should have no effect on expressions returned by
    :py:meth:`~sympy.solvers.ode.dsolve`, as
    :py:meth:`~sympy.solvers.ode.dsolve` already calls
    :py:meth:`~sympy.solvers.ode.ode.odesimp`, but the individual hint functions
    do not call :py:meth:`~sympy.solvers.ode.ode.odesimp` (because the
    :py:meth:`~sympy.solvers.ode.dsolve` wrapper does).  Therefore, this
    function is designed for mainly internal use.

    Examples
    ========

    >>> from sympy import sin, symbols, dsolve, pprint, Function
    >>> from sympy.solvers.ode.ode import odesimp
    >>> x, u2, C1= symbols('x,u2,C1')
    >>> f = Function('f')

    >>> eq = dsolve(x*f(x).diff(x) - f(x) - x*sin(f(x)/x), f(x),
    ... hint='1st_homogeneous_coeff_subs_indep_div_dep_Integral',
    ... simplify=False)
    >>> pprint(eq, wrap_line=False)
                            x
                           ----
                           f(x)
                             /
                            |
                            |   /        1   \
                            |  -|u1 + -------|
                            |   |        /1 \|
                            |   |     sin|--||
                            |   \        \u1//
    log(f(x)) = log(C1) +   |  ---------------- d(u1)
                            |          2
                            |        u1
                            |
                           /

    >>> pprint(odesimp(eq, f(x), 1, {C1},
    ... hint='1st_homogeneous_coeff_subs_indep_div_dep'
    ... )) #doctest: +SKIP
        x
    --------- = C1
       /f(x)\
    tan|----|
       \2*x /

    """
    x = func.args[0]
    f = func.func
    C1 = get_numbered_constants(eq, num=1)
    constants = eq.free_symbols - ode.free_symbols

    # First, integrate if the hint allows it.
    eq = _handle_Integral(eq, func, hint)
    if hint.startswith("nth_linear_euler_eq_nonhomogeneous"):
        eq = simplify(eq)
    if not isinstance(eq, Equality):
        raise TypeError("eq should be an instance of Equality")

    # allow simplifications under assumption that symbols are nonzero
    eq = eq.xreplace((_:={i: Dummy(nonzero=True) for i in constants})).xreplace({_[i]: i for i in _})

    # Second, clean up the arbitrary constants.
    # Right now, nth linear hints can put as many as 2*order constants in an
    # expression.  If that number grows with another hint, the third argument
    # here should be raised accordingly, or constantsimp() rewritten to handle
    # an arbitrary number of constants.
    eq = constantsimp(eq, constants)

    # Lastly, now that we have cleaned up the expression, try solving for func.
    # When CRootOf is implemented in solve(), we will want to return a CRootOf
    # every time instead of an Equality.

    # Get the f(x) on the left if possible.
    if eq.rhs == func and not eq.lhs.has(func):
        eq = [Eq(eq.rhs, eq.lhs)]

    # make sure we are working with lists of solutions in simplified form.
    if eq.lhs == func and not eq.rhs.has(func):
        # The solution is already solved
        eq = [eq]

    else:
        # The solution is not solved, so try to solve it
        try:
            floats = any(i.is_Float for i in eq.atoms(Number))
            eqsol = solve(eq, func, force=True, rational=False if floats else None)
            if not eqsol:
                raise NotImplementedError
        except (NotImplementedError, PolynomialError):
            eq = [eq]
        else:
            def _expand(expr):
                numer, denom = expr.as_numer_denom()

                if denom.is_Add:
                    return expr
                else:
                    return powsimp(expr.expand(), combine='exp', deep=True)

            # XXX: the rest of odesimp() expects each ``t`` to be in a
            # specific normal form: rational expression with numerator
            # expanded, but with combined exponential functions (at
            # least in this setup all tests pass).
            eq = [Eq(f(x), _expand(t)) for t in eqsol]

        # special simplification of the lhs.
        if hint.startswith("1st_homogeneous_coeff"):
            for j, eqi in enumerate(eq):
                newi = logcombine(eqi, force=True)
                if isinstance(newi.lhs, log) and newi.rhs == 0:
                    newi = Eq(newi.lhs.args[0]/C1, C1)
                eq[j] = newi

    # We cleaned up the constants before solving to help the solve engine with
    # a simpler expression, but the solved expression could have introduced
    # things like -C1, so rerun constantsimp() one last time before returning.
    for i, eqi in enumerate(eq):
        eq[i] = constantsimp(eqi, constants)
        eq[i] = constant_renumber(eq[i], ode.free_symbols)

    # If there is only 1 solution, return it;
    # otherwise return the list of solutions.
    if len(eq) == 1:
        eq = eq[0]
    return eq


def ode_sol_simplicity(sol, func, trysolving=True):
    r"""
    Returns an extended integer representing how simple a solution to an ODE
    is.

    The following things are considered, in order from most simple to least:

    - ``sol`` is solved for ``func``.
    - ``sol`` is not solved for ``func``, but can be if passed to solve (e.g.,
      a solution returned by ``dsolve(ode, func, simplify=False``).
    - If ``sol`` is not solved for ``func``, then base the result on the
      length of ``sol``, as computed by ``len(str(sol))``.
    - If ``sol`` has any unevaluated :py:class:`~sympy.integrals.integrals.Integral`\s,
      this will automatically be considered less simple than any of the above.

    This function returns an integer such that if solution A is simpler than
    solution B by above metric, then ``ode_sol_simplicity(sola, func) <
    ode_sol_simplicity(solb, func)``.

    Currently, the following are the numbers returned, but if the heuristic is
    ever improved, this may change.  Only the ordering is guaranteed.

    +----------------------------------------------+-------------------+
    | Simplicity                                   | Return            |
    +==============================================+===================+
    | ``sol`` solved for ``func``                  | ``-2``            |
    +----------------------------------------------+-------------------+
    | ``sol`` not solved for ``func`` but can be   | ``-1``            |
    +----------------------------------------------+-------------------+
    | ``sol`` is not solved nor solvable for       | ``len(str(sol))`` |
    | ``func``                                     |                   |
    +----------------------------------------------+-------------------+
    | ``sol`` contains an                          | ``oo``            |
    | :obj:`~sympy.integrals.integrals.Integral`   |                   |
    +----------------------------------------------+-------------------+

    ``oo`` here means the SymPy infinity, which should compare greater than
    any integer.

    If you already know :py:meth:`~sympy.solvers.solvers.solve` cannot solve
    ``sol``, you can use ``trysolving=False`` to skip that step, which is the
    only potentially slow step.  For example,
    :py:meth:`~sympy.solvers.ode.dsolve` with the ``simplify=False`` flag
    should do this.

    If ``sol`` is a list of solutions, if the worst solution in the list
    returns ``oo`` it returns that, otherwise it returns ``len(str(sol))``,
    that is, the length of the string representation of the whole list.

    Examples
    ========

    This function is designed to be passed to ``min`` as the key argument,
    such as ``min(listofsolutions, key=lambda i: ode_sol_simplicity(i,
    f(x)))``.

    >>> from sympy import symbols, Function, Eq, tan, Integral
    >>> from sympy.solvers.ode.ode import ode_sol_simplicity
    >>> x, C1, C2 = symbols('x, C1, C2')
    >>> f = Function('f')

    >>> ode_sol_simplicity(Eq(f(x), C1*x**2), f(x))
    -2
    >>> ode_sol_simplicity(Eq(x**2 + f(x), C1), f(x))
    -1
    >>> ode_sol_simplicity(Eq(f(x), C1*Integral(2*x, x)), f(x))
    oo
    >>> eq1 = Eq(f(x)/tan(f(x)/(2*x)), C1)
    >>> eq2 = Eq(f(x)/tan(f(x)/(2*x) + f(x)), C2)
    >>> [ode_sol_simplicity(eq, f(x)) for eq in [eq1, eq2]]
    [28, 35]
    >>> min([eq1, eq2], key=lambda i: ode_sol_simplicity(i, f(x)))
    Eq(f(x)/tan(f(x)/(2*x)), C1)

    """
    # TODO: if two solutions are solved for f(x), we still want to be
    # able to get the simpler of the two

    # See the docstring for the coercion rules.  We check easier (faster)
    # things here first, to save time.

    if iterable(sol):
        # See if there are Integrals
        for i in sol:
            if ode_sol_simplicity(i, func, trysolving=trysolving) == oo:
                return oo

        return len(str(sol))

    if sol.has(Integral):
        return oo

    # Next, try to solve for func.  This code will change slightly when CRootOf
    # is implemented in solve().  Probably a CRootOf solution should fall
    # somewhere between a normal solution and an unsolvable expression.

    # First, see if they are already solved
    if sol.lhs == func and not sol.rhs.has(func) or \
            sol.rhs == func and not sol.lhs.has(func):
        return -2
    # We are not so lucky, try solving manually
    if trysolving:
        try:
            sols = solve(sol, func)
            if not sols:
                raise NotImplementedError
        except NotImplementedError:
            pass
        else:
            return -1

    # Finally, a naive computation based on the length of the string version
    # of the expression.  This may favor combined fractions because they
    # will not have duplicate denominators, and may slightly favor expressions
    # with fewer additions and subtractions, as those are separated by spaces
    # by the printer.

    # Additional ideas for simplicity heuristics are welcome, like maybe
    # checking if a equation has a larger domain, or if constantsimp has
    # introduced arbitrary constants numbered higher than the order of a
    # given ODE that sol is a solution of.
    return len(str(sol))


def _extract_funcs(eqs):
    funcs = []
    for eq in eqs:
        derivs = [node for node in preorder_traversal(eq) if isinstance(node, Derivative)]
        func = []
        for d in derivs:
            func += list(d.atoms(AppliedUndef))
        for func_ in func:
            funcs.append(func_)
    funcs = list(uniq(funcs))

    return funcs


def _get_constant_subexpressions(expr, Cs):
    Cs = set(Cs)
    Ces = []
    def _recursive_walk(expr):
        expr_syms = expr.free_symbols
        if expr_syms and expr_syms.issubset(Cs):
            Ces.append(expr)
        else:
            if expr.func == exp:
                expr = expr.expand(mul=True)
            if expr.func in (Add, Mul):
                d = sift(expr.args, lambda i : i.free_symbols.issubset(Cs))
                if len(d[True]) > 1:
                    x = expr.func(*d[True])
                    if not x.is_number:
                        Ces.append(x)
            elif isinstance(expr, Integral):
                if expr.free_symbols.issubset(Cs) and \
                            all(len(x) == 3 for x in expr.limits):
                    Ces.append(expr)
            for i in expr.args:
                _recursive_walk(i)
        return
    _recursive_walk(expr)
    return Ces

def __remove_linear_redundancies(expr, Cs):
    cnts = {i: expr.count(i) for i in Cs}
    Cs = [i for i in Cs if cnts[i] > 0]

    def _linear(expr):
        if isinstance(expr, Add):
            xs = [i for i in Cs if expr.count(i)==cnts[i] \
                and 0 == expr.diff(i, 2)]
            d = {}
            for x in xs:
                y = expr.diff(x)
                if y not in d:
                    d[y]=[]
                d[y].append(x)
            for y in d:
                if len(d[y]) > 1:
                    d[y].sort(key=str)
                    for x in d[y][1:]:
                        expr = expr.subs(x, 0)
        return expr

    def _recursive_walk(expr):
        if len(expr.args) != 0:
            expr = expr.func(*[_recursive_walk(i) for i in expr.args])
        expr = _linear(expr)
        return expr

    if isinstance(expr, Equality):
        lhs, rhs = [_recursive_walk(i) for i in expr.args]
        f = lambda i: isinstance(i, Number) or i in Cs
        if isinstance(lhs, Symbol) and lhs in Cs:
            rhs, lhs = lhs, rhs
        if lhs.func in (Add, Symbol) and rhs.func in (Add, Symbol):
            dlhs = sift([lhs] if isinstance(lhs, AtomicExpr) else lhs.args, f)
            drhs = sift([rhs] if isinstance(rhs, AtomicExpr) else rhs.args, f)
            for i in [True, False]:
                for hs in [dlhs, drhs]:
                    if i not in hs:
                        hs[i] = [0]
            # this calculation can be simplified
            lhs = Add(*dlhs[False]) - Add(*drhs[False])
            rhs = Add(*drhs[True]) - Add(*dlhs[True])
        elif lhs.func in (Mul, Symbol) and rhs.func in (Mul, Symbol):
            dlhs = sift([lhs] if isinstance(lhs, AtomicExpr) else lhs.args, f)
            if True in dlhs:
                if False not in dlhs:
                    dlhs[False] = [1]
                lhs = Mul(*dlhs[False])
                rhs = rhs/Mul(*dlhs[True])
        return Eq(lhs, rhs)
    else:
        return _recursive_walk(expr)

@vectorize(0)
def constantsimp(expr, constants):
    r"""
    Simplifies an expression with arbitrary constants in it.

    This function is written specifically to work with
    :py:meth:`~sympy.solvers.ode.dsolve`, and is not intended for general use.

    Simplification is done by "absorbing" the arbitrary constants into other
    arbitrary constants, numbers, and symbols that they are not independent
    of.

    The symbols must all have the same name with numbers after it, for
    example, ``C1``, ``C2``, ``C3``.  The ``symbolname`` here would be
    '``C``', the ``startnumber`` would be 1, and the ``endnumber`` would be 3.
    If the arbitrary constants are independent of the variable ``x``, then the
    independent symbol would be ``x``.  There is no need to specify the
    dependent function, such as ``f(x)``, because it already has the
    independent symbol, ``x``, in it.

    Because terms are "absorbed" into arbitrary constants and because
    constants are renumbered after simplifying, the arbitrary constants in
    expr are not necessarily equal to the ones of the same name in the
    returned result.

    If two or more arbitrary constants are added, multiplied, or raised to the
    power of each other, they are first absorbed together into a single
    arbitrary constant.  Then the new constant is combined into other terms if
    necessary.

    Absorption of constants is done with limited assistance:

    1. terms of :py:class:`~sympy.core.add.Add`\s are collected to try join
       constants so `e^x (C_1 \cos(x) + C_2 \cos(x))` will simplify to `e^x
       C_1 \cos(x)`;

    2. powers with exponents that are :py:class:`~sympy.core.add.Add`\s are
       expanded so `e^{C_1 + x}` will be simplified to `C_1 e^x`.

    Use :py:meth:`~sympy.solvers.ode.ode.constant_renumber` to renumber constants
    after simplification or else arbitrary numbers on constants may appear,
    e.g. `C_1 + C_3 x`.

    In rare cases, a single constant can be "simplified" into two constants.
    Every differential equation solution should have as many arbitrary
    constants as the order of the differential equation.  The result here will
    be technically correct, but it may, for example, have `C_1` and `C_2` in
    an expression, when `C_1` is actually equal to `C_2`.  Use your discretion
    in such situations, and also take advantage of the ability to use hints in
    :py:meth:`~sympy.solvers.ode.dsolve`.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.solvers.ode.ode import constantsimp
    >>> C1, C2, C3, x, y = symbols('C1, C2, C3, x, y')
    >>> constantsimp(2*C1*x, {C1, C2, C3})
    C1*x
    >>> constantsimp(C1 + 2 + x, {C1, C2, C3})
    C1 + x
    >>> constantsimp(C1*C2 + 2 + C2 + C3*x, {C1, C2, C3})
    C1 + C3*x

    """
    # This function works recursively.  The idea is that, for Mul,
    # Add, Pow, and Function, if the class has a constant in it, then
    # we can simplify it, which we do by recursing down and
    # simplifying up.  Otherwise, we can skip that part of the
    # expression.

    Cs = constants

    orig_expr = expr

    constant_subexprs = _get_constant_subexpressions(expr, Cs)
    for xe in constant_subexprs:
        xes = list(xe.free_symbols)
        if not xes:
            continue
        if all(expr.count(c) == xe.count(c) for c in xes):
            xes.sort(key=str)
            expr = expr.subs(xe, xes[0])

    # try to perform common sub-expression elimination of constant terms
    try:
        commons, rexpr = cse(expr)
        commons.reverse()
        rexpr = rexpr[0]
        for s in commons:
            cs = list(s[1].atoms(Symbol))
            if len(cs) == 1 and cs[0] in Cs and \
                cs[0] not in rexpr.atoms(Symbol) and \
                not any(cs[0] in ex for ex in commons if ex != s):
                rexpr = rexpr.subs(s[0], cs[0])
            else:
                rexpr = rexpr.subs(*s)
        expr = rexpr
    except IndexError:
        pass
    expr = __remove_linear_redundancies(expr, Cs)

    def _conditional_term_factoring(expr):
        new_expr = terms_gcd(expr, clear=False, deep=True, expand=False)

        # we do not want to factor exponentials, so handle this separately
        if new_expr.is_Mul:
            infac = False
            asfac = False
            for m in new_expr.args:
                if isinstance(m, exp):
                    asfac = True
                elif m.is_Add:
                    infac = any(isinstance(fi, exp) for t in m.args
                        for fi in Mul.make_args(t))
                if asfac and infac:
                    new_expr = expr
                    break
        return new_expr

    expr = _conditional_term_factoring(expr)

    # call recursively if more simplification is possible
    if orig_expr != expr:
        return constantsimp(expr, Cs)
    return expr


def constant_renumber(expr, variables=None, newconstants=None):
    r"""
    Renumber arbitrary constants in ``expr`` to use the symbol names as given
    in ``newconstants``. In the process, this reorders expression terms in a
    standard way.

    If ``newconstants`` is not provided then the new constant names will be
    ``C1``, ``C2`` etc. Otherwise ``newconstants`` should be an iterable
    giving the new symbols to use for the constants in order.

    The ``variables`` argument is a list of non-constant symbols. All other
    free symbols found in ``expr`` are assumed to be constants and will be
    renumbered. If ``variables`` is not given then any numbered symbol
    beginning with ``C`` (e.g. ``C1``) is assumed to be a constant.

    Symbols are renumbered based on ``.sort_key()``, so they should be
    numbered roughly in the order that they appear in the final, printed
    expression.  Note that this ordering is based in part on hashes, so it can
    produce different results on different machines.

    The structure of this function is very similar to that of
    :py:meth:`~sympy.solvers.ode.constantsimp`.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.solvers.ode.ode import constant_renumber
    >>> x, C1, C2, C3 = symbols('x,C1:4')
    >>> expr = C3 + C2*x + C1*x**2
    >>> expr
    C1*x**2  + C2*x + C3
    >>> constant_renumber(expr)
    C1 + C2*x + C3*x**2

    The ``variables`` argument specifies which are constants so that the
    other symbols will not be renumbered:

    >>> constant_renumber(expr, [C1, x])
    C1*x**2  + C2 + C3*x

    The ``newconstants`` argument is used to specify what symbols to use when
    replacing the constants:

    >>> constant_renumber(expr, [x], newconstants=symbols('E1:4'))
    E1 + E2*x + E3*x**2

    """

    # System of expressions
    if isinstance(expr, (set, list, tuple)):
        return type(expr)(constant_renumber(Tuple(*expr),
                        variables=variables, newconstants=newconstants))

    # Symbols in solution but not ODE are constants
    if variables is not None:
        variables = set(variables)
        free_symbols = expr.free_symbols
        constantsymbols = list(free_symbols - variables)
    # Any Cn is a constant...
    else:
        variables = set()
        isconstant = lambda s: s.startswith('C') and s[1:].isdigit()
        constantsymbols = [sym for sym in expr.free_symbols if isconstant(sym.name)]

    # Find new constants checking that they aren't already in the ODE
    if newconstants is None:
        iter_constants = numbered_symbols(start=1, prefix='C', exclude=variables)
    else:
        iter_constants = (sym for sym in newconstants if sym not in variables)

    constants_found = []

    # make a mapping to send all constantsymbols to S.One and use
    # that to make sure that term ordering is not dependent on
    # the indexed value of C
    C_1 = [(ci, S.One) for ci in constantsymbols]
    sort_key=lambda arg: default_sort_key(arg.subs(C_1))

    def _constant_renumber(expr):
        r"""
        We need to have an internal recursive function
        """

        # For system of expressions
        if isinstance(expr, Tuple):
            renumbered = [_constant_renumber(e) for e in expr]
            return Tuple(*renumbered)

        if isinstance(expr, Equality):
            return Eq(
                _constant_renumber(expr.lhs),
                _constant_renumber(expr.rhs))

        if type(expr) not in (Mul, Add, Pow) and not expr.is_Function and \
                not expr.has(*constantsymbols):
            # Base case, as above.  Hope there aren't constants inside
            # of some other class, because they won't be renumbered.
            return expr
        elif expr.is_Piecewise:
            return expr
        elif expr in constantsymbols:
            if expr not in constants_found:
                constants_found.append(expr)
            return expr
        elif expr.is_Function or expr.is_Pow:
            return expr.func(
                *[_constant_renumber(x) for x in expr.args])
        else:
            sortedargs = list(expr.args)
            sortedargs.sort(key=sort_key)
            return expr.func(*[_constant_renumber(x) for x in sortedargs])
    expr = _constant_renumber(expr)

    # Don't renumber symbols present in the ODE.
    constants_found = [c for c in constants_found if c not in variables]

    # Renumbering happens here
    subs_dict = dict(zip(constants_found, iter_constants))
    expr = expr.subs(subs_dict, simultaneous=True)

    return expr


def _handle_Integral(expr, func, hint):
    r"""
    Converts a solution with Integrals in it into an actual solution.

    For most hints, this simply runs ``expr.doit()``.

    """
    if hint == "nth_linear_constant_coeff_homogeneous":
        sol = expr
    elif not hint.endswith("_Integral"):
        sol = expr.doit()
    else:
        sol = expr
    return sol


# XXX: Should this function maybe go somewhere else?


def homogeneous_order(eq, *symbols):
    r"""
    Returns the order `n` if `g` is homogeneous and ``None`` if it is not
    homogeneous.

    Determines if a function is homogeneous and if so of what order.  A
    function `f(x, y, \cdots)` is homogeneous of order `n` if `f(t x, t y,
    \cdots) = t^n f(x, y, \cdots)`.

    If the function is of two variables, `F(x, y)`, then `f` being homogeneous
    of any order is equivalent to being able to rewrite `F(x, y)` as `G(x/y)`
    or `H(y/x)`.  This fact is used to solve 1st order ordinary differential
    equations whose coefficients are homogeneous of the same order (see the
    docstrings of
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsDepDivIndep` and
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsIndepDivDep`).

    Symbols can be functions, but every argument of the function must be a
    symbol, and the arguments of the function that appear in the expression
    must match those given in the list of symbols.  If a declared function
    appears with different arguments than given in the list of symbols,
    ``None`` is returned.

    Examples
    ========

    >>> from sympy import Function, homogeneous_order, sqrt
    >>> from sympy.abc import x, y
    >>> f = Function('f')
    >>> homogeneous_order(f(x), f(x)) is None
    True
    >>> homogeneous_order(f(x,y), f(y, x), x, y) is None
    True
    >>> homogeneous_order(f(x), f(x), x)
    1
    >>> homogeneous_order(x**2*f(x)/sqrt(x**2+f(x)**2), x, f(x))
    2
    >>> homogeneous_order(x**2+f(x), x, f(x)) is None
    True

    """

    if not symbols:
        raise ValueError("homogeneous_order: no symbols were given.")
    symset = set(symbols)
    eq = sympify(eq)

    # The following are not supported
    if eq.has(Order, Derivative):
        return None

    # These are all constants
    if (eq.is_Number or
        eq.is_NumberSymbol or
        eq.is_number
            ):
        return S.Zero

    # Replace all functions with dummy variables
    dum = numbered_symbols(prefix='d', cls=Dummy)
    newsyms = set()
    for i in [j for j in symset if getattr(j, 'is_Function')]:
        iargs = set(i.args)
        if iargs.difference(symset):
            return None
        else:
            dummyvar = next(dum)
            eq = eq.subs(i, dummyvar)
            symset.remove(i)
            newsyms.add(dummyvar)
    symset.update(newsyms)

    if not eq.free_symbols & symset:
        return None

    # assuming order of a nested function can only be equal to zero
    if isinstance(eq, Function):
        return None if homogeneous_order(
            eq.args[0], *tuple(symset)) != 0 else S.Zero

    # make the replacement of x with x*t and see if t can be factored out
    t = Dummy('t', positive=True)  # It is sufficient that t > 0
    eqs = separatevars(eq.subs([(i, t*i) for i in symset]), [t], dict=True)[t]
    if eqs is S.One:
        return S.Zero  # there was no term with only t
    i, d = eqs.as_independent(t, as_Add=False)
    b, e = d.as_base_exp()
    if b == t:
        return e


def ode_2nd_power_series_ordinary(eq, func, order, match):
    r"""
    Gives a power series solution to a second order homogeneous differential
    equation with polynomial coefficients at an ordinary point. A homogeneous
    differential equation is of the form

    .. math :: P(x)\frac{d^2y}{dx^2} + Q(x)\frac{dy}{dx} + R(x) y(x) = 0

    For simplicity it is assumed that `P(x)`, `Q(x)` and `R(x)` are polynomials,
    it is sufficient that `\frac{Q(x)}{P(x)}` and `\frac{R(x)}{P(x)}` exists at
    `x_{0}`. A recurrence relation is obtained by substituting `y` as `\sum_{n=0}^\infty a_{n}x^{n}`,
    in the differential equation, and equating the nth term. Using this relation
    various terms can be generated.


    Examples
    ========

    >>> from sympy import dsolve, Function, pprint
    >>> from sympy.abc import x
    >>> f = Function("f")
    >>> eq = f(x).diff(x, 2) + f(x)
    >>> pprint(dsolve(eq, hint='2nd_power_series_ordinary'))
              / 4    2    \        /     2\
              |x    x     |        |    x |    / 6\
    f(x) = C2*|-- - -- + 1| + C1*x*|1 - --| + O\x /
              \24   2     /        \    6 /


    References
    ==========
    - https://tutorial.math.lamar.edu/Classes/DE/SeriesSolutions.aspx
    - George E. Simmons, "Differential Equations with Applications and
      Historical Notes", p.p 176 - 184

    """
    x = func.args[0]
    f = func.func
    C0, C1 = get_numbered_constants(eq, num=2)
    n = Dummy("n", integer=True)
    s = Wild("s")
    k = Wild("k", exclude=[x])
    x0 = match['x0']
    terms = match['terms']
    p = match[match['a3']]
    q = match[match['b3']]
    r = match[match['c3']]
    seriesdict = {}
    recurr = Function("r")

    # Generating the recurrence relation which works this way:
    # for the second order term the summation begins at n = 2. The coefficients
    # p is multiplied with an*(n - 1)*(n - 2)*x**n-2 and a substitution is made such that
    # the exponent of x becomes n.
    # For example, if p is x, then the second degree recurrence term is
    # an*(n - 1)*(n - 2)*x**n-1, substituting (n - 1) as n, it transforms to
    # an+1*n*(n - 1)*x**n.
    # A similar process is done with the first order and zeroth order term.

    coefflist = [(recurr(n), r), (n*recurr(n), q), (n*(n - 1)*recurr(n), p)]
    for index, coeff in enumerate(coefflist):
        if coeff[1]:
            f2 = powsimp(expand((coeff[1]*(x - x0)**(n - index)).subs(x, x + x0)))
            if f2.is_Add:
                addargs = f2.args
            else:
                addargs = [f2]
            for arg in addargs:
                powm = arg.match(s*x**k)
                term = coeff[0]*powm[s]
                if not powm[k].is_Symbol:
                    term = term.subs(n, n - powm[k].as_independent(n)[0])
                startind = powm[k].subs(n, index)
                # Seeing if the startterm can be reduced further.
                # If it vanishes for n lesser than startind, it is
                # equal to summation from n.
                if startind:
                    for i in reversed(range(startind)):
                        if not term.subs(n, i):
                            seriesdict[term] = i
                        else:
                            seriesdict[term] = i + 1
                            break
                else:
                    seriesdict[term] = S.Zero

    # Stripping of terms so that the sum starts with the same number.
    teq = S.Zero
    suminit = seriesdict.values()
    rkeys = seriesdict.keys()
    req = Add(*rkeys)
    if any(suminit):
        maxval = max(suminit)
        for term in seriesdict:
            val = seriesdict[term]
            if val != maxval:
                for i in range(val, maxval):
                    teq += term.subs(n, val)

    finaldict = {}
    if teq:
        fargs = teq.atoms(AppliedUndef)
        if len(fargs) == 1:
            finaldict[fargs.pop()] = 0
        else:
            maxf = max(fargs, key = lambda x: x.args[0])
            sol = solve(teq, maxf)
            if isinstance(sol, list):
                sol = sol[0]
            finaldict[maxf] = sol

    # Finding the recurrence relation in terms of the largest term.
    fargs = req.atoms(AppliedUndef)
    maxf = max(fargs, key = lambda x: x.args[0])
    minf = min(fargs, key = lambda x: x.args[0])
    if minf.args[0].is_Symbol:
        startiter = 0
    else:
        startiter = -minf.args[0].as_independent(n)[0]
    lhs = maxf
    rhs =  solve(req, maxf)
    if isinstance(rhs, list):
        rhs = rhs[0]

    # Checking how many values are already present
    tcounter = len([t for t in finaldict.values() if t])

    for _ in range(tcounter, terms - 3):  # Assuming c0 and c1 to be arbitrary
        check = rhs.subs(n, startiter)
        nlhs = lhs.subs(n, startiter)
        nrhs = check.subs(finaldict)
        finaldict[nlhs] = nrhs
        startiter += 1

    # Post processing
    series = C0 + C1*(x - x0)
    for term in finaldict:
        if finaldict[term]:
            fact = term.args[0]
            series += (finaldict[term].subs([(recurr(0), C0), (recurr(1), C1)])*(
                x - x0)**fact)
    series = collect(expand_mul(series), [C0, C1]) + Order(x**terms)
    return Eq(f(x), series)


def ode_2nd_power_series_regular(eq, func, order, match):
    r"""
    Gives a power series solution to a second order homogeneous differential
    equation with polynomial coefficients at a regular point. A second order
    homogeneous differential equation is of the form

    .. math :: P(x)\frac{d^2y}{dx^2} + Q(x)\frac{dy}{dx} + R(x) y(x) = 0

    A point is said to regular singular at `x0` if `x - x0\frac{Q(x)}{P(x)}`
    and `(x - x0)^{2}\frac{R(x)}{P(x)}` are analytic at `x0`. For simplicity
    `P(x)`, `Q(x)` and `R(x)` are assumed to be polynomials. The algorithm for
    finding the power series solutions is:

    1.  Try expressing `(x - x0)P(x)` and `((x - x0)^{2})Q(x)` as power series
        solutions about x0. Find `p0` and `q0` which are the constants of the
        power series expansions.
    2.  Solve the indicial equation `f(m) = m(m - 1) + m*p0 + q0`, to obtain the
        roots `m1` and `m2` of the indicial equation.
    3.  If `m1 - m2` is a non integer there exists two series solutions. If
        `m1 = m2`, there exists only one solution. If `m1 - m2` is an integer,
        then the existence of one solution is confirmed. The other solution may
        or may not exist.

    The power series solution is of the form `x^{m}\sum_{n=0}^\infty a_{n}x^{n}`. The
    coefficients are determined by the following recurrence relation.
    `a_{n} = -\frac{\sum_{k=0}^{n-1} q_{n-k} + (m + k)p_{n-k}}{f(m + n)}`. For the case
    in which `m1 - m2` is an integer, it can be seen from the recurrence relation
    that for the lower root `m`, when `n` equals the difference of both the
    roots, the denominator becomes zero. So if the numerator is not equal to zero,
    a second series solution exists.


    Examples
    ========

    >>> from sympy import dsolve, Function, pprint
    >>> from sympy.abc import x
    >>> f = Function("f")
    >>> eq = x*(f(x).diff(x, 2)) + 2*(f(x).diff(x)) + x*f(x)
    >>> pprint(dsolve(eq, hint='2nd_power_series_regular'))
                                  /   6     4    2    \
                                  |  x     x    x     |
              / 4     2    \   C1*|- --- + -- - -- + 1|
              |x     x     |      \  720   24   2     /    / 6\
    f(x) = C2*|--- - -- + 1| + ------------------------ + O\x /
              \120   6     /              x


    References
    ==========
    - George E. Simmons, "Differential Equations with Applications and
      Historical Notes", p.p 176 - 184

    """
    x = func.args[0]
    f = func.func
    C0, C1 = get_numbered_constants(eq, num=2)
    m = Dummy("m")  # for solving the indicial equation
    x0 = match['x0']
    terms = match['terms']
    p = match['p']
    q = match['q']

    # Generating the indicial equation
    indicial = []
    for term in [p, q]:
        if not term.has(x):
            indicial.append(term)
        else:
            term = series(term, x=x, n=1, x0=x0)
            if isinstance(term, Order):
                indicial.append(S.Zero)
            else:
                for arg in term.args:
                    if not arg.has(x):
                        indicial.append(arg)
                        break

    p0, q0 = indicial
    sollist = solve(m*(m - 1) + m*p0 + q0, m)
    if sollist and isinstance(sollist, list) and all(
        sol.is_real for sol in sollist):
        serdict1 = {}
        serdict2 = {}
        if len(sollist) == 1:
            # Only one series solution exists in this case.
            m1 = m2 = sollist.pop()
            if terms-m1-1 <= 0:
                return Eq(f(x), Order(terms))
            serdict1 = _frobenius(terms-m1-1, m1, p0, q0, p, q, x0, x, C0)

        else:
            m1 = sollist[0]
            m2 = sollist[1]
            if m1 < m2:
                m1, m2 = m2, m1
            # Irrespective of whether m1 - m2 is an integer or not, one
            # Frobenius series solution exists.
            serdict1 = _frobenius(terms-m1-1, m1, p0, q0, p, q, x0, x, C0)
            if not (m1 - m2).is_integer:
                # Second frobenius series solution exists.
                serdict2 = _frobenius(terms-m2-1, m2, p0, q0, p, q, x0, x, C1)
            else:
                # Check if second frobenius series solution exists.
                serdict2 = _frobenius(terms-m2-1, m2, p0, q0, p, q, x0, x, C1, check=m1)

        if serdict1:
            finalseries1 = C0
            for key in serdict1:
                power = int(key.name[1:])
                finalseries1 += serdict1[key]*(x - x0)**power
            finalseries1 = (x - x0)**m1*finalseries1
            finalseries2 = S.Zero
            if serdict2:
                for key in serdict2:
                    power = int(key.name[1:])
                    finalseries2 += serdict2[key]*(x - x0)**power
                finalseries2 += C1
                finalseries2 = (x - x0)**m2*finalseries2
            return Eq(f(x), collect(finalseries1 + finalseries2,
                [C0, C1]) + Order(x**terms))


def _frobenius(n, m, p0, q0, p, q, x0, x, c, check=None):
    r"""
    Returns a dict with keys as coefficients and values as their values in terms of C0
    """
    n = int(n)
    # In cases where m1 - m2 is not an integer
    m2 = check

    d = Dummy("d")
    numsyms = numbered_symbols("C", start=0)
    numsyms = [next(numsyms) for i in range(n + 1)]
    serlist = []
    for ser in [p, q]:
        # Order term not present
        if ser.is_polynomial(x) and Poly(ser, x).degree() <= n:
            if x0:
                ser = ser.subs(x, x + x0)
            dict_ = Poly(ser, x).as_dict()
        # Order term present
        else:
            tseries = series(ser, x=x0, n=n+1)
            # Removing order
            dict_ = Poly(list(ordered(tseries.args))[: -1], x).as_dict()
        # Fill in with zeros, if coefficients are zero.
        for i in range(n + 1):
            if (i,) not in dict_:
                dict_[(i,)] = S.Zero
        serlist.append(dict_)

    pseries = serlist[0]
    qseries = serlist[1]
    indicial = d*(d - 1) + d*p0 + q0
    frobdict = {}
    for i in range(1, n + 1):
        num = c*(m*pseries[(i,)] + qseries[(i,)])
        for j in range(1, i):
            sym = Symbol("C" + str(j))
            num += frobdict[sym]*((m + j)*pseries[(i - j,)] + qseries[(i - j,)])

        # Checking for cases when m1 - m2 is an integer. If num equals zero
        # then a second Frobenius series solution cannot be found. If num is not zero
        # then set constant as zero and proceed.
        if m2 is not None and i == m2 - m:
            if num:
                return False
            else:
                frobdict[numsyms[i]] = S.Zero
        else:
            frobdict[numsyms[i]] = -num/(indicial.subs(d, m+i))

    return frobdict

def _remove_redundant_solutions(eq, solns, order, var):
    r"""
    Remove redundant solutions from the set of solutions.

    This function is needed because otherwise dsolve can return
    redundant solutions. As an example consider:

        eq = Eq((f(x).diff(x, 2))*f(x).diff(x), 0)

    There are two ways to find solutions to eq. The first is to solve f(x).diff(x, 2) = 0
    leading to solution f(x)=C1 + C2*x. The second is to solve the equation f(x).diff(x) = 0
    leading to the solution f(x) = C1. In this particular case we then see
    that the second solution is a special case of the first and we do not
    want to return it.

    This does not always happen. If we have

        eq = Eq((f(x)**2-4)*(f(x).diff(x)-4), 0)

    then we get the algebraic solution f(x) = [-2, 2] and the integral solution
    f(x) = x + C1 and in this case the two solutions are not equivalent wrt
    initial conditions so both should be returned.
    """
    def is_special_case_of(soln1, soln2):
        return _is_special_case_of(soln1, soln2, eq, order, var)

    unique_solns = []
    for soln1 in solns:
        for soln2 in unique_solns.copy():
            if is_special_case_of(soln1, soln2):
                break
            elif is_special_case_of(soln2, soln1):
                unique_solns.remove(soln2)
        else:
            unique_solns.append(soln1)

    return unique_solns

def _is_special_case_of(soln1, soln2, eq, order, var):
    r"""
    True if soln1 is found to be a special case of soln2 wrt some value of the
    constants that appear in soln2. False otherwise.
    """
    # The solutions returned by dsolve may be given explicitly or implicitly.
    # We will equate the sol1=(soln1.rhs - soln1.lhs), sol2=(soln2.rhs - soln2.lhs)
    # of the two solutions.
    #
    # Since this is supposed to hold for all x it also holds for derivatives.
    # For an order n ode we should be able to differentiate
    # each solution n times to get n+1 equations.
    #
    # We then try to solve those n+1 equations for the integrations constants
    # in sol2. If we can find a solution that does not depend on x then it
    # means that some value of the constants in sol1 is a special case of
    # sol2 corresponding to a particular choice of the integration constants.

    # In case the solution is in implicit form we subtract the sides
    soln1 = soln1.rhs - soln1.lhs
    soln2 = soln2.rhs - soln2.lhs

    # Work for the series solution
    if soln1.has(Order) and soln2.has(Order):
        if soln1.getO() == soln2.getO():
            soln1 = soln1.removeO()
            soln2 = soln2.removeO()
        else:
            return False
    elif soln1.has(Order) or soln2.has(Order):
        return False

    constants1 = soln1.free_symbols.difference(eq.free_symbols)
    constants2 = soln2.free_symbols.difference(eq.free_symbols)

    constants1_new = get_numbered_constants(Tuple(soln1, soln2), len(constants1))
    if len(constants1) == 1:
        constants1_new = {constants1_new}
    for c_old, c_new in zip(constants1, constants1_new):
        soln1 = soln1.subs(c_old, c_new)

    # n equations for sol1 = sol2, sol1'=sol2', ...
    lhs = soln1
    rhs = soln2
    eqns = [Eq(lhs, rhs)]
    for n in range(1, order):
        lhs = lhs.diff(var)
        rhs = rhs.diff(var)
        eq = Eq(lhs, rhs)
        eqns.append(eq)

    # BooleanTrue/False awkwardly show up for trivial equations
    if any(isinstance(eq, BooleanFalse) for eq in eqns):
        return False
    eqns = [eq for eq in eqns if not isinstance(eq, BooleanTrue)]

    try:
        constant_solns = solve(eqns, constants2)
    except NotImplementedError:
        return False

    # Sometimes returns a dict and sometimes a list of dicts
    if isinstance(constant_solns, dict):
        constant_solns = [constant_solns]

    # after solving the issue 17418, maybe we don't need the following checksol code.
    for constant_soln in constant_solns:
        for eq in eqns:
            eq=eq.rhs-eq.lhs
            if checksol(eq, constant_soln) is not True:
                return False

    # If any solution gives all constants as expressions that don't depend on
    # x then there exists constants for soln2 that give soln1
    for constant_soln in constant_solns:
        if not any(c.has(var) for c in constant_soln.values()):
            return True

    return False


def ode_1st_power_series(eq, func, order, match):
    r"""
    The power series solution is a method which gives the Taylor series expansion
    to the solution of a differential equation.

    For a first order differential equation `\frac{dy}{dx} = h(x, y)`, a power
    series solution exists at a point `x = x_{0}` if `h(x, y)` is analytic at `x_{0}`.
    The solution is given by

    .. math:: y(x) = y(x_{0}) + \sum_{n = 1}^{\infty} \frac{F_{n}(x_{0},b)(x - x_{0})^n}{n!},

    where `y(x_{0}) = b` is the value of y at the initial value of `x_{0}`.
    To compute the values of the `F_{n}(x_{0},b)` the following algorithm is
    followed, until the required number of terms are generated.

    1. `F_1 = h(x_{0}, b)`
    2. `F_{n+1} = \frac{\partial F_{n}}{\partial x} + \frac{\partial F_{n}}{\partial y}F_{1}`

    Examples
    ========

    >>> from sympy import Function, pprint, exp, dsolve
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = exp(x)*(f(x).diff(x)) - f(x)
    >>> pprint(dsolve(eq, hint='1st_power_series'))
                           3       4       5
                       C1*x    C1*x    C1*x     / 6\
    f(x) = C1 + C1*x - ----- + ----- + ----- + O\x /
                         6       24      60


    References
    ==========

    - Travis W. Walker, Analytic power series technique for solving first-order
      differential equations, p.p 17, 18

    """
    x = func.args[0]
    y = match['y']
    f = func.func
    h = -match[match['d']]/match[match['e']]
    point = match['f0']
    value = match['f0val']
    terms = match['terms']

    # First term
    F = h
    if not h:
        return Eq(f(x), value)

    # Initialization
    series = value
    if terms > 1:
        hc = h.subs({x: point, y: value})
        if hc.has(oo) or hc.has(nan) or hc.has(zoo):
            # Derivative does not exist, not analytic
            return Eq(f(x), oo)
        elif hc:
            series += hc*(x - point)

    for factcount in range(2, terms):
        Fnew = F.diff(x) + F.diff(y)*h
        Fnewc = Fnew.subs({x: point, y: value})
        # Same logic as above
        if Fnewc.has(oo) or Fnewc.has(nan) or Fnewc.has(-oo) or Fnewc.has(zoo):
            return Eq(f(x), oo)
        series += Fnewc*((x - point)**factcount)/factorial(factcount)
        F = Fnew
    series += Order(x**terms)
    return Eq(f(x), series)


def checkinfsol(eq, infinitesimals, func=None, order=None):
    r"""
    This function is used to check if the given infinitesimals are the
    actual infinitesimals of the given first order differential equation.
    This method is specific to the Lie Group Solver of ODEs.

    As of now, it simply checks, by substituting the infinitesimals in the
    partial differential equation.


    .. math:: \frac{\partial \eta}{\partial x} + \left(\frac{\partial \eta}{\partial y}
                - \frac{\partial \xi}{\partial x}\right)*h
                - \frac{\partial \xi}{\partial y}*h^{2}
                - \xi\frac{\partial h}{\partial x} - \eta\frac{\partial h}{\partial y} = 0


    where `\eta`, and `\xi` are the infinitesimals and `h(x,y) = \frac{dy}{dx}`

    The infinitesimals should be given in the form of a list of dicts
    ``[{xi(x, y): inf, eta(x, y): inf}]``, corresponding to the
    output of the function infinitesimals. It returns a list
    of values of the form ``[(True/False, sol)]`` where ``sol`` is the value
    obtained after substituting the infinitesimals in the PDE. If it
    is ``True``, then ``sol`` would be 0.

    """
    if isinstance(eq, Equality):
        eq = eq.lhs - eq.rhs
    if not func:
        eq, func = _preprocess(eq)
    variables = func.args
    if len(variables) != 1:
        raise ValueError("ODE's have only one independent variable")
    else:
        x = variables[0]
        if not order:
            order = ode_order(eq, func)
        if order != 1:
            raise NotImplementedError("Lie groups solver has been implemented "
            "only for first order differential equations")
        else:
            df = func.diff(x)
            a = Wild('a', exclude = [df])
            b = Wild('b', exclude = [df])
            match = collect(expand(eq), df).match(a*df + b)

            if match:
                h = -simplify(match[b]/match[a])
            else:
                try:
                    sol = solve(eq, df)
                except NotImplementedError:
                    raise NotImplementedError("Infinitesimals for the "
                        "first order ODE could not be found")
                else:
                    h = sol[0]  # Find infinitesimals for one solution

            y = Dummy('y')
            h = h.subs(func, y)
            xi = Function('xi')(x, y)
            eta = Function('eta')(x, y)
            dxi = Function('xi')(x, func)
            deta = Function('eta')(x, func)
            pde = (eta.diff(x) + (eta.diff(y) - xi.diff(x))*h -
                (xi.diff(y))*h**2 - xi*(h.diff(x)) - eta*(h.diff(y)))
            soltup = []
            for sol in infinitesimals:
                tsol = {xi: S(sol[dxi]).subs(func, y),
                    eta: S(sol[deta]).subs(func, y)}
                sol = simplify(pde.subs(tsol).doit())
                if sol:
                    soltup.append((False, sol.subs(y, func)))
                else:
                    soltup.append((True, 0))
            return soltup


def sysode_linear_2eq_order1(match_):
    x = match_['func'][0].func
    y = match_['func'][1].func
    func = match_['func']
    fc = match_['func_coeff']
    eq = match_['eq']
    r = {}
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    for i in range(2):
        eq[i] = Add(*[terms/fc[i,func[i],1] for terms in Add.make_args(eq[i])])

    # for equations Eq(a1*diff(x(t),t), a*x(t) + b*y(t) + k1)
    # and Eq(a2*diff(x(t),t), c*x(t) + d*y(t) + k2)
    r['a'] = -fc[0,x(t),0]/fc[0,x(t),1]
    r['c'] = -fc[1,x(t),0]/fc[1,y(t),1]
    r['b'] = -fc[0,y(t),0]/fc[0,x(t),1]
    r['d'] = -fc[1,y(t),0]/fc[1,y(t),1]
    forcing = [S.Zero,S.Zero]
    for i in range(2):
        for j in Add.make_args(eq[i]):
            if not j.has(x(t), y(t)):
                forcing[i] += j
    if not (forcing[0].has(t) or forcing[1].has(t)):
        r['k1'] = forcing[0]
        r['k2'] = forcing[1]
    else:
        raise NotImplementedError("Only homogeneous problems are supported" +
                                  " (and constant inhomogeneity)")

    if match_['type_of_equation'] == 'type6':
        sol = _linear_2eq_order1_type6(x, y, t, r, eq)
    if match_['type_of_equation'] == 'type7':
        sol = _linear_2eq_order1_type7(x, y, t, r, eq)
    return sol

def _linear_2eq_order1_type6(x, y, t, r, eq):
    r"""
    The equations of this type of ode are .

    .. math:: x' = f(t) x + g(t) y

    .. math:: y' = a [f(t) + a h(t)] x + a [g(t) - h(t)] y

    This is solved by first multiplying the first equation by `-a` and adding
    it to the second equation to obtain

    .. math:: y' - a x' = -a h(t) (y - a x)

    Setting `U = y - ax` and integrating the equation we arrive at

    .. math:: y - ax = C_1 e^{-a \int h(t) \,dt}

    and on substituting the value of y in first equation give rise to first order ODEs. After solving for
    `x`, we can obtain `y` by substituting the value of `x` in second equation.

    """
    C1, C2, C3, C4 = get_numbered_constants(eq, num=4)
    p = 0
    q = 0
    p1 = cancel(r['c']/cancel(r['c']/r['d']).as_numer_denom()[0])
    p2 = cancel(r['a']/cancel(r['a']/r['b']).as_numer_denom()[0])
    for n, i in enumerate([p1, p2]):
        for j in Mul.make_args(collect_const(i)):
            if not j.has(t):
                q = j
            if q!=0 and n==0:
                if ((r['c']/j - r['a'])/(r['b'] - r['d']/j)) == j:
                    p = 1
                    s = j
                    break
            if q!=0 and n==1:
                if ((r['a']/j - r['c'])/(r['d'] - r['b']/j)) == j:
                    p = 2
                    s = j
                    break

    if p == 1:
        equ = diff(x(t),t) - r['a']*x(t) - r['b']*(s*x(t) + C1*exp(-s*Integral(r['b'] - r['d']/s, t)))
        hint1 = classify_ode(equ)[1]
        sol1 = dsolve(equ, hint=hint1+'_Integral').rhs
        sol2 = s*sol1 + C1*exp(-s*Integral(r['b'] - r['d']/s, t))
    elif p ==2:
        equ = diff(y(t),t) - r['c']*y(t) - r['d']*s*y(t) + C1*exp(-s*Integral(r['d'] - r['b']/s, t))
        hint1 = classify_ode(equ)[1]
        sol2 = dsolve(equ, hint=hint1+'_Integral').rhs
        sol1 = s*sol2 + C1*exp(-s*Integral(r['d'] - r['b']/s, t))
    return [Eq(x(t), sol1), Eq(y(t), sol2)]

def _linear_2eq_order1_type7(x, y, t, r, eq):
    r"""
    The equations of this type of ode are .

    .. math:: x' = f(t) x + g(t) y

    .. math:: y' = h(t) x + p(t) y

    Differentiating the first equation and substituting the value of `y`
    from second equation will give a second-order linear equation

    .. math:: g x'' - (fg + gp + g') x' + (fgp - g^{2} h + f g' - f' g) x = 0

    This above equation can be easily integrated if following conditions are satisfied.

    1. `fgp - g^{2} h + f g' - f' g = 0`

    2. `fgp - g^{2} h + f g' - f' g = ag, fg + gp + g' = bg`

    If first condition is satisfied then it is solved by current dsolve solver and in second case it becomes
    a constant coefficient differential equation which is also solved by current solver.

    Otherwise if the above condition fails then,
    a particular solution is assumed as `x = x_0(t)` and `y = y_0(t)`
    Then the general solution is expressed as

    .. math:: x = C_1 x_0(t) + C_2 x_0(t) \int \frac{g(t) F(t) P(t)}{x_0^{2}(t)} \,dt

    .. math:: y = C_1 y_0(t) + C_2 [\frac{F(t) P(t)}{x_0(t)} + y_0(t) \int \frac{g(t) F(t) P(t)}{x_0^{2}(t)} \,dt]

    where C1 and C2 are arbitrary constants and

    .. math:: F(t) = e^{\int f(t) \,dt}, P(t) = e^{\int p(t) \,dt}

    """
    C1, C2, C3, C4 = get_numbered_constants(eq, num=4)
    e1 = r['a']*r['b']*r['c'] - r['b']**2*r['c'] + r['a']*diff(r['b'],t) - diff(r['a'],t)*r['b']
    e2 = r['a']*r['c']*r['d'] - r['b']*r['c']**2 + diff(r['c'],t)*r['d'] - r['c']*diff(r['d'],t)
    m1 = r['a']*r['b'] + r['b']*r['d'] + diff(r['b'],t)
    m2 = r['a']*r['c'] + r['c']*r['d'] + diff(r['c'],t)
    if e1 == 0:
        sol1 = dsolve(r['b']*diff(x(t),t,t) - m1*diff(x(t),t)).rhs
        sol2 = dsolve(diff(y(t),t) - r['c']*sol1 - r['d']*y(t)).rhs
    elif e2 == 0:
        sol2 = dsolve(r['c']*diff(y(t),t,t) - m2*diff(y(t),t)).rhs
        sol1 = dsolve(diff(x(t),t) - r['a']*x(t) - r['b']*sol2).rhs
    elif not (e1/r['b']).has(t) and not (m1/r['b']).has(t):
        sol1 = dsolve(diff(x(t),t,t) - (m1/r['b'])*diff(x(t),t) - (e1/r['b'])*x(t)).rhs
        sol2 = dsolve(diff(y(t),t) - r['c']*sol1 - r['d']*y(t)).rhs
    elif not (e2/r['c']).has(t) and not (m2/r['c']).has(t):
        sol2 = dsolve(diff(y(t),t,t) - (m2/r['c'])*diff(y(t),t) - (e2/r['c'])*y(t)).rhs
        sol1 = dsolve(diff(x(t),t) - r['a']*x(t) - r['b']*sol2).rhs
    else:
        x0 = Function('x0')(t)    # x0 and y0 being particular solutions
        y0 = Function('y0')(t)
        F = exp(Integral(r['a'],t))
        P = exp(Integral(r['d'],t))
        sol1 = C1*x0 + C2*x0*Integral(r['b']*F*P/x0**2, t)
        sol2 = C1*y0 + C2*(F*P/x0 + y0*Integral(r['b']*F*P/x0**2, t))
    return [Eq(x(t), sol1), Eq(y(t), sol2)]


def sysode_nonlinear_2eq_order1(match_):
    func = match_['func']
    eq = match_['eq']
    fc = match_['func_coeff']
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    if match_['type_of_equation'] == 'type5':
        sol = _nonlinear_2eq_order1_type5(func, t, eq)
        return sol
    x = func[0].func
    y = func[1].func
    for i in range(2):
        eqs = 0
        for terms in Add.make_args(eq[i]):
            eqs += terms/fc[i,func[i],1]
        eq[i] = eqs
    if match_['type_of_equation'] == 'type1':
        sol = _nonlinear_2eq_order1_type1(x, y, t, eq)
    elif match_['type_of_equation'] == 'type2':
        sol = _nonlinear_2eq_order1_type2(x, y, t, eq)
    elif match_['type_of_equation'] == 'type3':
        sol = _nonlinear_2eq_order1_type3(x, y, t, eq)
    elif match_['type_of_equation'] == 'type4':
        sol = _nonlinear_2eq_order1_type4(x, y, t, eq)
    return sol


def _nonlinear_2eq_order1_type1(x, y, t, eq):
    r"""
    Equations:

    .. math:: x' = x^n F(x,y)

    .. math:: y' = g(y) F(x,y)

    Solution:

    .. math:: x = \varphi(y), \int \frac{1}{g(y) F(\varphi(y),y)} \,dy = t + C_2

    where

    if `n \neq 1`

    .. math:: \varphi = [C_1 + (1-n) \int \frac{1}{g(y)} \,dy]^{\frac{1}{1-n}}

    if `n = 1`

    .. math:: \varphi = C_1 e^{\int \frac{1}{g(y)} \,dy}

    where `C_1` and `C_2` are arbitrary constants.

    """
    C1, C2 = get_numbered_constants(eq, num=2)
    n = Wild('n', exclude=[x(t),y(t)])
    f = Wild('f')
    u, v = symbols('u, v')
    r = eq[0].match(diff(x(t),t) - x(t)**n*f)
    g = ((diff(y(t),t) - eq[1])/r[f]).subs(y(t),v)
    F = r[f].subs(x(t),u).subs(y(t),v)
    n = r[n]
    if n!=1:
        phi = (C1 + (1-n)*Integral(1/g, v))**(1/(1-n))
    else:
        phi = C1*exp(Integral(1/g, v))
    phi = phi.doit()
    sol2 = solve(Integral(1/(g*F.subs(u,phi)), v).doit() - t - C2, v)
    sol = []
    for sols in sol2:
        sol.append(Eq(x(t),phi.subs(v, sols)))
        sol.append(Eq(y(t), sols))
    return sol

def _nonlinear_2eq_order1_type2(x, y, t, eq):
    r"""
    Equations:

    .. math:: x' = e^{\lambda x} F(x,y)

    .. math:: y' = g(y) F(x,y)

    Solution:

    .. math:: x = \varphi(y), \int \frac{1}{g(y) F(\varphi(y),y)} \,dy = t + C_2

    where

    if `\lambda \neq 0`

    .. math:: \varphi = -\frac{1}{\lambda} log(C_1 - \lambda \int \frac{1}{g(y)} \,dy)

    if `\lambda = 0`

    .. math:: \varphi = C_1 + \int \frac{1}{g(y)} \,dy

    where `C_1` and `C_2` are arbitrary constants.

    """
    C1, C2 = get_numbered_constants(eq, num=2)
    n = Wild('n', exclude=[x(t),y(t)])
    f = Wild('f')
    u, v = symbols('u, v')
    r = eq[0].match(diff(x(t),t) - exp(n*x(t))*f)
    g = ((diff(y(t),t) - eq[1])/r[f]).subs(y(t),v)
    F = r[f].subs(x(t),u).subs(y(t),v)
    n = r[n]
    if n:
        phi = -1/n*log(C1 - n*Integral(1/g, v))
    else:
        phi = C1 + Integral(1/g, v)
    phi = phi.doit()
    sol2 = solve(Integral(1/(g*F.subs(u,phi)), v).doit() - t - C2, v)
    sol = []
    for sols in sol2:
        sol.append(Eq(x(t),phi.subs(v, sols)))
        sol.append(Eq(y(t), sols))
    return sol

def _nonlinear_2eq_order1_type3(x, y, t, eq):
    r"""
    Autonomous system of general form

    .. math:: x' = F(x,y)

    .. math:: y' = G(x,y)

    Assuming `y = y(x, C_1)` where `C_1` is an arbitrary constant is the general
    solution of the first-order equation

    .. math:: F(x,y) y'_x = G(x,y)

    Then the general solution of the original system of equations has the form

    .. math:: \int \frac{1}{F(x,y(x,C_1))} \,dx = t + C_1

    """
    C1, C2, C3, C4 = get_numbered_constants(eq, num=4)
    v = Function('v')
    u = Symbol('u')
    f = Wild('f')
    g = Wild('g')
    r1 = eq[0].match(diff(x(t),t) - f)
    r2 = eq[1].match(diff(y(t),t) - g)
    F = r1[f].subs(x(t), u).subs(y(t), v(u))
    G = r2[g].subs(x(t), u).subs(y(t), v(u))
    sol2r = dsolve(Eq(diff(v(u), u), G/F))
    if isinstance(sol2r, Equality):
        sol2r = [sol2r]
    for sol2s in sol2r:
        sol1 = solve(Integral(1/F.subs(v(u), sol2s.rhs), u).doit() - t - C2, u)
    sol = []
    for sols in sol1:
        sol.append(Eq(x(t), sols))
        sol.append(Eq(y(t), (sol2s.rhs).subs(u, sols)))
    return sol

def _nonlinear_2eq_order1_type4(x, y, t, eq):
    r"""
    Equation:

    .. math:: x' = f_1(x) g_1(y) \phi(x,y,t)

    .. math:: y' = f_2(x) g_2(y) \phi(x,y,t)

    First integral:

    .. math:: \int \frac{f_2(x)}{f_1(x)} \,dx - \int \frac{g_1(y)}{g_2(y)} \,dy = C

    where `C` is an arbitrary constant.

    On solving the first integral for `x` (resp., `y` ) and on substituting the
    resulting expression into either equation of the original solution, one
    arrives at a first-order equation for determining `y` (resp., `x` ).

    """
    C1, C2 = get_numbered_constants(eq, num=2)
    u, v = symbols('u, v')
    U, V = symbols('U, V', cls=Function)
    f = Wild('f')
    g = Wild('g')
    f1 = Wild('f1', exclude=[v,t])
    f2 = Wild('f2', exclude=[v,t])
    g1 = Wild('g1', exclude=[u,t])
    g2 = Wild('g2', exclude=[u,t])
    r1 = eq[0].match(diff(x(t),t) - f)
    r2 = eq[1].match(diff(y(t),t) - g)
    num, den = (
        (r1[f].subs(x(t),u).subs(y(t),v))/
        (r2[g].subs(x(t),u).subs(y(t),v))).as_numer_denom()
    R1 = num.match(f1*g1)
    R2 = den.match(f2*g2)
    phi = (r1[f].subs(x(t),u).subs(y(t),v))/num
    F1 = R1[f1]; F2 = R2[f2]
    G1 = R1[g1]; G2 = R2[g2]
    sol1r = solve(Integral(F2/F1, u).doit() - Integral(G1/G2,v).doit() - C1, u)
    sol2r = solve(Integral(F2/F1, u).doit() - Integral(G1/G2,v).doit() - C1, v)
    sol = []
    for sols in sol1r:
        sol.append(Eq(y(t), dsolve(diff(V(t),t) - F2.subs(u,sols).subs(v,V(t))*G2.subs(v,V(t))*phi.subs(u,sols).subs(v,V(t))).rhs))
    for sols in sol2r:
        sol.append(Eq(x(t), dsolve(diff(U(t),t) - F1.subs(u,U(t))*G1.subs(v,sols).subs(u,U(t))*phi.subs(v,sols).subs(u,U(t))).rhs))
    return set(sol)

def _nonlinear_2eq_order1_type5(func, t, eq):
    r"""
    Clairaut system of ODEs

    .. math:: x = t x' + F(x',y')

    .. math:: y = t y' + G(x',y')

    The following are solutions of the system

    `(i)` straight lines:

    .. math:: x = C_1 t + F(C_1, C_2), y = C_2 t + G(C_1, C_2)

    where `C_1` and `C_2` are arbitrary constants;

    `(ii)` envelopes of the above lines;

    `(iii)` continuously differentiable lines made up from segments of the lines
    `(i)` and `(ii)`.

    """
    C1, C2 = get_numbered_constants(eq, num=2)
    f = Wild('f')
    g = Wild('g')
    def check_type(x, y):
        r1 = eq[0].match(t*diff(x(t),t) - x(t) + f)
        r2 = eq[1].match(t*diff(y(t),t) - y(t) + g)
        if not (r1 and r2):
            r1 = eq[0].match(diff(x(t),t) - x(t)/t + f/t)
            r2 = eq[1].match(diff(y(t),t) - y(t)/t + g/t)
        if not (r1 and r2):
            r1 = (-eq[0]).match(t*diff(x(t),t) - x(t) + f)
            r2 = (-eq[1]).match(t*diff(y(t),t) - y(t) + g)
        if not (r1 and r2):
            r1 = (-eq[0]).match(diff(x(t),t) - x(t)/t + f/t)
            r2 = (-eq[1]).match(diff(y(t),t) - y(t)/t + g/t)
        return [r1, r2]
    for func_ in func:
        if isinstance(func_, list):
            x = func[0][0].func
            y = func[0][1].func
            [r1, r2] = check_type(x, y)
            if not (r1 and r2):
                [r1, r2] = check_type(y, x)
                x, y = y, x
    x1 = diff(x(t),t); y1 = diff(y(t),t)
    return {Eq(x(t), C1*t + r1[f].subs(x1,C1).subs(y1,C2)), Eq(y(t), C2*t + r2[g].subs(x1,C1).subs(y1,C2))}

def sysode_nonlinear_3eq_order1(match_):
    x = match_['func'][0].func
    y = match_['func'][1].func
    z = match_['func'][2].func
    eq = match_['eq']
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    if match_['type_of_equation'] == 'type1':
        sol = _nonlinear_3eq_order1_type1(x, y, z, t, eq)
    if match_['type_of_equation'] == 'type2':
        sol = _nonlinear_3eq_order1_type2(x, y, z, t, eq)
    if match_['type_of_equation'] == 'type3':
        sol = _nonlinear_3eq_order1_type3(x, y, z, t, eq)
    if match_['type_of_equation'] == 'type4':
        sol = _nonlinear_3eq_order1_type4(x, y, z, t, eq)
    if match_['type_of_equation'] == 'type5':
        sol = _nonlinear_3eq_order1_type5(x, y, z, t, eq)
    return sol

def _nonlinear_3eq_order1_type1(x, y, z, t, eq):
    r"""
    Equations:

    .. math:: a x' = (b - c) y z, \enspace b y' = (c - a) z x, \enspace c z' = (a - b) x y

    First Integrals:

    .. math:: a x^{2} + b y^{2} + c z^{2} = C_1

    .. math:: a^{2} x^{2} + b^{2} y^{2} + c^{2} z^{2} = C_2

    where `C_1` and `C_2` are arbitrary constants. On solving the integrals for `y` and
    `z` and on substituting the resulting expressions into the first equation of the
    system, we arrives at a separable first-order equation on `x`. Similarly doing that
    for other two equations, we will arrive at first order equation on `y` and `z` too.

    References
    ==========
    -https://eqworld.ipmnet.ru/en/solutions/sysode/sode0401.pdf

    """
    C1, C2 = get_numbered_constants(eq, num=2)
    u, v, w = symbols('u, v, w')
    p = Wild('p', exclude=[x(t), y(t), z(t), t])
    q = Wild('q', exclude=[x(t), y(t), z(t), t])
    s = Wild('s', exclude=[x(t), y(t), z(t), t])
    r = (diff(x(t),t) - eq[0]).match(p*y(t)*z(t))
    r.update((diff(y(t),t) - eq[1]).match(q*z(t)*x(t)))
    r.update((diff(z(t),t) - eq[2]).match(s*x(t)*y(t)))
    n1, d1 = r[p].as_numer_denom()
    n2, d2 = r[q].as_numer_denom()
    n3, d3 = r[s].as_numer_denom()
    val = solve([n1*u-d1*v+d1*w, d2*u+n2*v-d2*w, d3*u-d3*v-n3*w],[u,v])
    vals = [val[v], val[u]]
    c = lcm(vals[0].as_numer_denom()[1], vals[1].as_numer_denom()[1])
    b = vals[0].subs(w, c)
    a = vals[1].subs(w, c)
    y_x = sqrt(((c*C1-C2) - a*(c-a)*x(t)**2)/(b*(c-b)))
    z_x = sqrt(((b*C1-C2) - a*(b-a)*x(t)**2)/(c*(b-c)))
    z_y = sqrt(((a*C1-C2) - b*(a-b)*y(t)**2)/(c*(a-c)))
    x_y = sqrt(((c*C1-C2) - b*(c-b)*y(t)**2)/(a*(c-a)))
    x_z = sqrt(((b*C1-C2) - c*(b-c)*z(t)**2)/(a*(b-a)))
    y_z = sqrt(((a*C1-C2) - c*(a-c)*z(t)**2)/(b*(a-b)))
    sol1 = dsolve(a*diff(x(t),t) - (b-c)*y_x*z_x)
    sol2 = dsolve(b*diff(y(t),t) - (c-a)*z_y*x_y)
    sol3 = dsolve(c*diff(z(t),t) - (a-b)*x_z*y_z)
    return [sol1, sol2, sol3]


def _nonlinear_3eq_order1_type2(x, y, z, t, eq):
    r"""
    Equations:

    .. math:: a x' = (b - c) y z f(x, y, z, t)

    .. math:: b y' = (c - a) z x f(x, y, z, t)

    .. math:: c z' = (a - b) x y f(x, y, z, t)

    First Integrals:

    .. math:: a x^{2} + b y^{2} + c z^{2} = C_1

    .. math:: a^{2} x^{2} + b^{2} y^{2} + c^{2} z^{2} = C_2

    where `C_1` and `C_2` are arbitrary constants. On solving the integrals for `y` and
    `z` and on substituting the resulting expressions into the first equation of the
    system, we arrives at a first-order differential equations on `x`. Similarly doing
    that for other two equations we will arrive at first order equation on `y` and `z`.

    References
    ==========
    -https://eqworld.ipmnet.ru/en/solutions/sysode/sode0402.pdf

    """
    C1, C2 = get_numbered_constants(eq, num=2)
    u, v, w = symbols('u, v, w')
    p = Wild('p', exclude=[x(t), y(t), z(t), t])
    q = Wild('q', exclude=[x(t), y(t), z(t), t])
    s = Wild('s', exclude=[x(t), y(t), z(t), t])
    f = Wild('f')
    r1 = (diff(x(t),t) - eq[0]).match(y(t)*z(t)*f)
    r = collect_const(r1[f]).match(p*f)
    r.update(((diff(y(t),t) - eq[1])/r[f]).match(q*z(t)*x(t)))
    r.update(((diff(z(t),t) - eq[2])/r[f]).match(s*x(t)*y(t)))
    n1, d1 = r[p].as_numer_denom()
    n2, d2 = r[q].as_numer_denom()
    n3, d3 = r[s].as_numer_denom()
    val = solve([n1*u-d1*v+d1*w, d2*u+n2*v-d2*w, -d3*u+d3*v+n3*w],[u,v])
    vals = [val[v], val[u]]
    c = lcm(vals[0].as_numer_denom()[1], vals[1].as_numer_denom()[1])
    a = vals[0].subs(w, c)
    b = vals[1].subs(w, c)
    y_x = sqrt(((c*C1-C2) - a*(c-a)*x(t)**2)/(b*(c-b)))
    z_x = sqrt(((b*C1-C2) - a*(b-a)*x(t)**2)/(c*(b-c)))
    z_y = sqrt(((a*C1-C2) - b*(a-b)*y(t)**2)/(c*(a-c)))
    x_y = sqrt(((c*C1-C2) - b*(c-b)*y(t)**2)/(a*(c-a)))
    x_z = sqrt(((b*C1-C2) - c*(b-c)*z(t)**2)/(a*(b-a)))
    y_z = sqrt(((a*C1-C2) - c*(a-c)*z(t)**2)/(b*(a-b)))
    sol1 = dsolve(a*diff(x(t),t) - (b-c)*y_x*z_x*r[f])
    sol2 = dsolve(b*diff(y(t),t) - (c-a)*z_y*x_y*r[f])
    sol3 = dsolve(c*diff(z(t),t) - (a-b)*x_z*y_z*r[f])
    return [sol1, sol2, sol3]

def _nonlinear_3eq_order1_type3(x, y, z, t, eq):
    r"""
    Equations:

    .. math:: x' = c F_2 - b F_3, \enspace y' = a F_3 - c F_1, \enspace z' = b F_1 - a F_2

    where `F_n = F_n(x, y, z, t)`.

    1. First Integral:

    .. math:: a x + b y + c z = C_1,

    where C is an arbitrary constant.

    2. If we assume function `F_n` to be independent of `t`,i.e, `F_n` = `F_n (x, y, z)`
    Then, on eliminating `t` and `z` from the first two equation of the system, one
    arrives at the first-order equation

    .. math:: \frac{dy}{dx} = \frac{a F_3 (x, y, z) - c F_1 (x, y, z)}{c F_2 (x, y, z) -
                b F_3 (x, y, z)}

    where `z = \frac{1}{c} (C_1 - a x - b y)`

    References
    ==========
    -https://eqworld.ipmnet.ru/en/solutions/sysode/sode0404.pdf

    """
    C1 = get_numbered_constants(eq, num=1)
    u, v, w = symbols('u, v, w')
    fu, fv, fw = symbols('u, v, w', cls=Function)
    p = Wild('p', exclude=[x(t), y(t), z(t), t])
    q = Wild('q', exclude=[x(t), y(t), z(t), t])
    s = Wild('s', exclude=[x(t), y(t), z(t), t])
    F1, F2, F3 = symbols('F1, F2, F3', cls=Wild)
    r1 = (diff(x(t), t) - eq[0]).match(F2-F3)
    r = collect_const(r1[F2]).match(s*F2)
    r.update(collect_const(r1[F3]).match(q*F3))
    if eq[1].has(r[F2]) and not eq[1].has(r[F3]):
        r[F2], r[F3] = r[F3], r[F2]
        r[s], r[q] = -r[q], -r[s]
    r.update((diff(y(t), t) - eq[1]).match(p*r[F3] - r[s]*F1))
    a = r[p]; b = r[q]; c = r[s]
    F1 = r[F1].subs(x(t), u).subs(y(t),v).subs(z(t), w)
    F2 = r[F2].subs(x(t), u).subs(y(t),v).subs(z(t), w)
    F3 = r[F3].subs(x(t), u).subs(y(t),v).subs(z(t), w)
    z_xy = (C1-a*u-b*v)/c
    y_zx = (C1-a*u-c*w)/b
    x_yz = (C1-b*v-c*w)/a
    y_x = dsolve(diff(fv(u),u) - ((a*F3-c*F1)/(c*F2-b*F3)).subs(w,z_xy).subs(v,fv(u))).rhs
    z_x = dsolve(diff(fw(u),u) - ((b*F1-a*F2)/(c*F2-b*F3)).subs(v,y_zx).subs(w,fw(u))).rhs
    z_y = dsolve(diff(fw(v),v) - ((b*F1-a*F2)/(a*F3-c*F1)).subs(u,x_yz).subs(w,fw(v))).rhs
    x_y = dsolve(diff(fu(v),v) - ((c*F2-b*F3)/(a*F3-c*F1)).subs(w,z_xy).subs(u,fu(v))).rhs
    y_z = dsolve(diff(fv(w),w) - ((a*F3-c*F1)/(b*F1-a*F2)).subs(u,x_yz).subs(v,fv(w))).rhs
    x_z = dsolve(diff(fu(w),w) - ((c*F2-b*F3)/(b*F1-a*F2)).subs(v,y_zx).subs(u,fu(w))).rhs
    sol1 = dsolve(diff(fu(t),t) - (c*F2 - b*F3).subs(v,y_x).subs(w,z_x).subs(u,fu(t))).rhs
    sol2 = dsolve(diff(fv(t),t) - (a*F3 - c*F1).subs(u,x_y).subs(w,z_y).subs(v,fv(t))).rhs
    sol3 = dsolve(diff(fw(t),t) - (b*F1 - a*F2).subs(u,x_z).subs(v,y_z).subs(w,fw(t))).rhs
    return [sol1, sol2, sol3]

def _nonlinear_3eq_order1_type4(x, y, z, t, eq):
    r"""
    Equations:

    .. math:: x' = c z F_2 - b y F_3, \enspace y' = a x F_3 - c z F_1, \enspace z' = b y F_1 - a x F_2

    where `F_n = F_n (x, y, z, t)`

    1. First integral:

    .. math:: a x^{2} + b y^{2} + c z^{2} = C_1

    where `C` is an arbitrary constant.

    2. Assuming the function `F_n` is independent of `t`: `F_n = F_n (x, y, z)`. Then on
    eliminating `t` and `z` from the first two equations of the system, one arrives at
    the first-order equation

    .. math:: \frac{dy}{dx} = \frac{a x F_3 (x, y, z) - c z F_1 (x, y, z)}
                {c z F_2 (x, y, z) - b y F_3 (x, y, z)}

    where `z = \pm \sqrt{\frac{1}{c} (C_1 - a x^{2} - b y^{2})}`

    References
    ==========
    -https://eqworld.ipmnet.ru/en/solutions/sysode/sode0405.pdf

    """
    C1 = get_numbered_constants(eq, num=1)
    u, v, w = symbols('u, v, w')
    p = Wild('p', exclude=[x(t), y(t), z(t), t])
    q = Wild('q', exclude=[x(t), y(t), z(t), t])
    s = Wild('s', exclude=[x(t), y(t), z(t), t])
    F1, F2, F3 = symbols('F1, F2, F3', cls=Wild)
    r1 = eq[0].match(diff(x(t),t) - z(t)*F2 + y(t)*F3)
    r = collect_const(r1[F2]).match(s*F2)
    r.update(collect_const(r1[F3]).match(q*F3))
    if eq[1].has(r[F2]) and not eq[1].has(r[F3]):
        r[F2], r[F3] = r[F3], r[F2]
        r[s], r[q] = -r[q], -r[s]
    r.update((diff(y(t),t) - eq[1]).match(p*x(t)*r[F3] - r[s]*z(t)*F1))
    a = r[p]; b = r[q]; c = r[s]
    F1 = r[F1].subs(x(t),u).subs(y(t),v).subs(z(t),w)
    F2 = r[F2].subs(x(t),u).subs(y(t),v).subs(z(t),w)
    F3 = r[F3].subs(x(t),u).subs(y(t),v).subs(z(t),w)
    x_yz = sqrt((C1 - b*v**2 - c*w**2)/a)
    y_zx = sqrt((C1 - c*w**2 - a*u**2)/b)
    z_xy = sqrt((C1 - a*u**2 - b*v**2)/c)
    y_x = dsolve(diff(v(u),u) - ((a*u*F3-c*w*F1)/(c*w*F2-b*v*F3)).subs(w,z_xy).subs(v,v(u))).rhs
    z_x = dsolve(diff(w(u),u) - ((b*v*F1-a*u*F2)/(c*w*F2-b*v*F3)).subs(v,y_zx).subs(w,w(u))).rhs
    z_y = dsolve(diff(w(v),v) - ((b*v*F1-a*u*F2)/(a*u*F3-c*w*F1)).subs(u,x_yz).subs(w,w(v))).rhs
    x_y = dsolve(diff(u(v),v) - ((c*w*F2-b*v*F3)/(a*u*F3-c*w*F1)).subs(w,z_xy).subs(u,u(v))).rhs
    y_z = dsolve(diff(v(w),w) - ((a*u*F3-c*w*F1)/(b*v*F1-a*u*F2)).subs(u,x_yz).subs(v,v(w))).rhs
    x_z = dsolve(diff(u(w),w) - ((c*w*F2-b*v*F3)/(b*v*F1-a*u*F2)).subs(v,y_zx).subs(u,u(w))).rhs
    sol1 = dsolve(diff(u(t),t) - (c*w*F2 - b*v*F3).subs(v,y_x).subs(w,z_x).subs(u,u(t))).rhs
    sol2 = dsolve(diff(v(t),t) - (a*u*F3 - c*w*F1).subs(u,x_y).subs(w,z_y).subs(v,v(t))).rhs
    sol3 = dsolve(diff(w(t),t) - (b*v*F1 - a*u*F2).subs(u,x_z).subs(v,y_z).subs(w,w(t))).rhs
    return [sol1, sol2, sol3]

def _nonlinear_3eq_order1_type5(x, y, z, t, eq):
    r"""
    .. math:: x' = x (c F_2 - b F_3), \enspace y' = y (a F_3 - c F_1), \enspace z' = z (b F_1 - a F_2)

    where `F_n = F_n (x, y, z, t)` and are arbitrary functions.

    First Integral:

    .. math:: \left|x\right|^{a} \left|y\right|^{b} \left|z\right|^{c} = C_1

    where `C` is an arbitrary constant. If the function `F_n` is independent of `t`,
    then, by eliminating `t` and `z` from the first two equations of the system, one
    arrives at a first-order equation.

    References
    ==========
    -https://eqworld.ipmnet.ru/en/solutions/sysode/sode0406.pdf

    """
    C1 = get_numbered_constants(eq, num=1)
    u, v, w = symbols('u, v, w')
    fu, fv, fw = symbols('u, v, w', cls=Function)
    p = Wild('p', exclude=[x(t), y(t), z(t), t])
    q = Wild('q', exclude=[x(t), y(t), z(t), t])
    s = Wild('s', exclude=[x(t), y(t), z(t), t])
    F1, F2, F3 = symbols('F1, F2, F3', cls=Wild)
    r1 = eq[0].match(diff(x(t), t) - x(t)*F2 + x(t)*F3)
    r = collect_const(r1[F2]).match(s*F2)
    r.update(collect_const(r1[F3]).match(q*F3))
    if eq[1].has(r[F2]) and not eq[1].has(r[F3]):
        r[F2], r[F3] = r[F3], r[F2]
        r[s], r[q] = -r[q], -r[s]
    r.update((diff(y(t), t) - eq[1]).match(y(t)*(p*r[F3] - r[s]*F1)))
    a = r[p]; b = r[q]; c = r[s]
    F1 = r[F1].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    F2 = r[F2].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    F3 = r[F3].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    x_yz = (C1*v**-b*w**-c)**-a
    y_zx = (C1*w**-c*u**-a)**-b
    z_xy = (C1*u**-a*v**-b)**-c
    y_x = dsolve(diff(fv(u), u) - ((v*(a*F3 - c*F1))/(u*(c*F2 - b*F3))).subs(w, z_xy).subs(v, fv(u))).rhs
    z_x = dsolve(diff(fw(u), u) - ((w*(b*F1 - a*F2))/(u*(c*F2 - b*F3))).subs(v, y_zx).subs(w, fw(u))).rhs
    z_y = dsolve(diff(fw(v), v) - ((w*(b*F1 - a*F2))/(v*(a*F3 - c*F1))).subs(u, x_yz).subs(w, fw(v))).rhs
    x_y = dsolve(diff(fu(v), v) - ((u*(c*F2 - b*F3))/(v*(a*F3 - c*F1))).subs(w, z_xy).subs(u, fu(v))).rhs
    y_z = dsolve(diff(fv(w), w) - ((v*(a*F3 - c*F1))/(w*(b*F1 - a*F2))).subs(u, x_yz).subs(v, fv(w))).rhs
    x_z = dsolve(diff(fu(w), w) - ((u*(c*F2 - b*F3))/(w*(b*F1 - a*F2))).subs(v, y_zx).subs(u, fu(w))).rhs
    sol1 = dsolve(diff(fu(t), t) - (u*(c*F2 - b*F3)).subs(v, y_x).subs(w, z_x).subs(u, fu(t))).rhs
    sol2 = dsolve(diff(fv(t), t) - (v*(a*F3 - c*F1)).subs(u, x_y).subs(w, z_y).subs(v, fv(t))).rhs
    sol3 = dsolve(diff(fw(t), t) - (w*(b*F1 - a*F2)).subs(u, x_z).subs(v, y_z).subs(w, fw(t))).rhs
    return [sol1, sol2, sol3]


#This import is written at the bottom to avoid circular imports.
from .single import SingleODEProblem, SingleODESolver, solver_map
