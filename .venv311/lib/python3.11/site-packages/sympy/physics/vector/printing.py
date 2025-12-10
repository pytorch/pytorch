from sympy.core.function import Derivative
from sympy.core.function import UndefinedFunction, AppliedUndef
from sympy.core.symbol import Symbol
from sympy.interactive.printing import init_printing
from sympy.printing.latex import LatexPrinter
from sympy.printing.pretty.pretty import PrettyPrinter
from sympy.printing.pretty.pretty_symbology import center_accent
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import PRECEDENCE

__all__ = ['vprint', 'vsstrrepr', 'vsprint', 'vpprint', 'vlatex',
           'init_vprinting']


class VectorStrPrinter(StrPrinter):
    """String Printer for vector expressions. """

    def _print_Derivative(self, e):
        from sympy.physics.vector.functions import dynamicsymbols
        t = dynamicsymbols._t
        if (bool(sum(i == t for i in e.variables)) &
                isinstance(type(e.args[0]), UndefinedFunction)):
            ol = str(e.args[0].func)
            for i, v in enumerate(e.variables):
                ol += dynamicsymbols._str
            return ol
        else:
            return StrPrinter().doprint(e)

    def _print_Function(self, e):
        from sympy.physics.vector.functions import dynamicsymbols
        t = dynamicsymbols._t
        if isinstance(type(e), UndefinedFunction):
            return StrPrinter().doprint(e).replace("(%s)" % t, '')
        return e.func.__name__ + "(%s)" % self.stringify(e.args, ", ")


class VectorStrReprPrinter(VectorStrPrinter):
    """String repr printer for vector expressions."""
    def _print_str(self, s):
        return repr(s)


class VectorLatexPrinter(LatexPrinter):
    """Latex Printer for vector expressions. """

    def _print_Function(self, expr, exp=None):
        from sympy.physics.vector.functions import dynamicsymbols
        func = expr.func.__name__
        t = dynamicsymbols._t

        if (hasattr(self, '_print_' + func) and not
            isinstance(type(expr), UndefinedFunction)):
            return getattr(self, '_print_' + func)(expr, exp)
        elif isinstance(type(expr), UndefinedFunction) and (expr.args == (t,)):
            # treat this function like a symbol
            expr = Symbol(func)
            if exp is not None:
                # copied from LatexPrinter._helper_print_standard_power, which
                # we can't call because we only have exp as a string.
                base = self.parenthesize(expr, PRECEDENCE['Pow'])
                base = self.parenthesize_super(base)
                return r"%s^{%s}" % (base, exp)
            else:
                return super()._print(expr)
        else:
            return super()._print_Function(expr, exp)

    def _print_Derivative(self, der_expr):
        from sympy.physics.vector.functions import dynamicsymbols
        # make sure it is in the right form
        der_expr = der_expr.doit()
        if not isinstance(der_expr, Derivative):
            return r"\left(%s\right)" % self.doprint(der_expr)

        # check if expr is a dynamicsymbol
        t = dynamicsymbols._t
        expr = der_expr.expr
        red = expr.atoms(AppliedUndef)
        syms = der_expr.variables
        test1 = not all(True for i in red if i.free_symbols == {t})
        test2 = not all(t == i for i in syms)
        if test1 or test2:
            return super()._print_Derivative(der_expr)

        # done checking
        dots = len(syms)
        base = self._print_Function(expr)
        base_split = base.split('_', 1)
        base = base_split[0]
        if dots == 1:
            base = r"\dot{%s}" % base
        elif dots == 2:
            base = r"\ddot{%s}" % base
        elif dots == 3:
            base = r"\dddot{%s}" % base
        elif dots == 4:
            base = r"\ddddot{%s}" % base
        else:  # Fallback to standard printing
            return super()._print_Derivative(der_expr)
        if len(base_split) != 1:
            base += '_' + base_split[1]
        return base


class VectorPrettyPrinter(PrettyPrinter):
    """Pretty Printer for vectorialexpressions. """

    def _print_Derivative(self, deriv):
        from sympy.physics.vector.functions import dynamicsymbols
        # XXX use U('PARTIAL DIFFERENTIAL') here ?
        t = dynamicsymbols._t
        dot_i = 0
        syms = list(reversed(deriv.variables))

        while len(syms) > 0:
            if syms[-1] == t:
                syms.pop()
                dot_i += 1
            else:
                return super()._print_Derivative(deriv)

        if not (isinstance(type(deriv.expr), UndefinedFunction) and
                (deriv.expr.args == (t,))):
            return super()._print_Derivative(deriv)
        else:
            pform = self._print_Function(deriv.expr)

        # the following condition would happen with some sort of non-standard
        # dynamic symbol I guess, so we'll just print the SymPy way
        if len(pform.picture) > 1:
            return super()._print_Derivative(deriv)

        # There are only special symbols up to fourth-order derivatives
        if dot_i >= 5:
            return super()._print_Derivative(deriv)

        # Deal with special symbols
        dots = {0: "",
                1: "\N{COMBINING DOT ABOVE}",
                2: "\N{COMBINING DIAERESIS}",
                3: "\N{COMBINING THREE DOTS ABOVE}",
                4: "\N{COMBINING FOUR DOTS ABOVE}"}

        d = pform.__dict__
        # if unicode is false then calculate number of apostrophes needed and
        # add to output
        if not self._use_unicode:
            apostrophes = ""
            for i in range(0, dot_i):
                apostrophes += "'"
            d['picture'][0] += apostrophes + "(t)"
        else:
            d['picture'] = [center_accent(d['picture'][0], dots[dot_i])]
        return pform

    def _print_Function(self, e):
        from sympy.physics.vector.functions import dynamicsymbols
        t = dynamicsymbols._t
        # XXX works only for applied functions
        func = e.func
        args = e.args
        func_name = func.__name__
        pform = self._print_Symbol(Symbol(func_name))
        # If this function is an Undefined function of t, it is probably a
        # dynamic symbol, so we'll skip the (t). The rest of the code is
        # identical to the normal PrettyPrinter code
        if not (isinstance(func, UndefinedFunction) and (args == (t,))):
            return super()._print_Function(e)
        return pform


def vprint(expr, **settings):
    r"""Function for printing of expressions generated in the
    sympy.physics vector package.

    Extends SymPy's StrPrinter, takes the same setting accepted by SymPy's
    :func:`~.sstr`, and is equivalent to ``print(sstr(foo))``.

    Parameters
    ==========

    expr : valid SymPy object
        SymPy expression to print.
    settings : args
        Same as the settings accepted by SymPy's sstr().

    Examples
    ========

    >>> from sympy.physics.vector import vprint, dynamicsymbols
    >>> u1 = dynamicsymbols('u1')
    >>> print(u1)
    u1(t)
    >>> vprint(u1)
    u1

    """

    outstr = vsprint(expr, **settings)

    import builtins
    if (outstr != 'None'):
        builtins._ = outstr
        print(outstr)


def vsstrrepr(expr, **settings):
    """Function for displaying expression representation's with vector
    printing enabled.

    Parameters
    ==========

    expr : valid SymPy object
        SymPy expression to print.
    settings : args
        Same as the settings accepted by SymPy's sstrrepr().

    """
    p = VectorStrReprPrinter(settings)
    return p.doprint(expr)


def vsprint(expr, **settings):
    r"""Function for displaying expressions generated in the
    sympy.physics vector package.

    Returns the output of vprint() as a string.

    Parameters
    ==========

    expr : valid SymPy object
        SymPy expression to print
    settings : args
        Same as the settings accepted by SymPy's sstr().

    Examples
    ========

    >>> from sympy.physics.vector import vsprint, dynamicsymbols
    >>> u1, u2 = dynamicsymbols('u1 u2')
    >>> u2d = dynamicsymbols('u2', level=1)
    >>> print("%s = %s" % (u1, u2 + u2d))
    u1(t) = u2(t) + Derivative(u2(t), t)
    >>> print("%s = %s" % (vsprint(u1), vsprint(u2 + u2d)))
    u1 = u2 + u2'

    """

    string_printer = VectorStrPrinter(settings)
    return string_printer.doprint(expr)


def vpprint(expr, **settings):
    r"""Function for pretty printing of expressions generated in the
    sympy.physics vector package.

    Mainly used for expressions not inside a vector; the output of running
    scripts and generating equations of motion. Takes the same options as
    SymPy's :func:`~.pretty_print`; see that function for more information.

    Parameters
    ==========

    expr : valid SymPy object
        SymPy expression to pretty print
    settings : args
        Same as those accepted by SymPy's pretty_print.


    """

    pp = VectorPrettyPrinter(settings)

    # Note that this is copied from sympy.printing.pretty.pretty_print:

    # XXX: this is an ugly hack, but at least it works
    use_unicode = pp._settings['use_unicode']
    from sympy.printing.pretty.pretty_symbology import pretty_use_unicode
    uflag = pretty_use_unicode(use_unicode)

    try:
        return pp.doprint(expr)
    finally:
        pretty_use_unicode(uflag)


def vlatex(expr, **settings):
    r"""Function for printing latex representation of sympy.physics.vector
    objects.

    For latex representation of Vectors, Dyadics, and dynamicsymbols. Takes the
    same options as SymPy's :func:`~.latex`; see that function for more
    information;

    Parameters
    ==========

    expr : valid SymPy object
        SymPy expression to represent in LaTeX form
    settings : args
        Same as latex()

    Examples
    ========

    >>> from sympy.physics.vector import vlatex, ReferenceFrame, dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q1, q2 = dynamicsymbols('q1 q2')
    >>> q1d, q2d = dynamicsymbols('q1 q2', 1)
    >>> q1dd, q2dd = dynamicsymbols('q1 q2', 2)
    >>> vlatex(N.x + N.y)
    '\\mathbf{\\hat{n}_x} + \\mathbf{\\hat{n}_y}'
    >>> vlatex(q1 + q2)
    'q_{1} + q_{2}'
    >>> vlatex(q1d)
    '\\dot{q}_{1}'
    >>> vlatex(q1 * q2d)
    'q_{1} \\dot{q}_{2}'
    >>> vlatex(q1dd * q1 / q1d)
    '\\frac{q_{1} \\ddot{q}_{1}}{\\dot{q}_{1}}'

    """
    latex_printer = VectorLatexPrinter(settings)

    return latex_printer.doprint(expr)


def init_vprinting(**kwargs):
    """Initializes time derivative printing for all SymPy objects, i.e. any
    functions of time will be displayed in a more compact notation. The main
    benefit of this is for printing of time derivatives; instead of
    displaying as ``Derivative(f(t),t)``, it will display ``f'``. This is
    only actually needed for when derivatives are present and are not in a
    physics.vector.Vector or physics.vector.Dyadic object. This function is a
    light wrapper to :func:`~.init_printing`. Any keyword
    arguments for it are valid here.

    {0}

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> t, x = symbols('t, x')
    >>> omega = Function('omega')
    >>> omega(x).diff()
    Derivative(omega(x), x)
    >>> omega(t).diff()
    Derivative(omega(t), t)

    Now use the string printer:

    >>> from sympy.physics.vector import init_vprinting
    >>> init_vprinting(pretty_print=False)
    >>> omega(x).diff()
    Derivative(omega(x), x)
    >>> omega(t).diff()
    omega'

    """
    kwargs['str_printer'] = vsstrrepr
    kwargs['pretty_printer'] = vpprint
    kwargs['latex_printer'] = vlatex
    init_printing(**kwargs)


params = init_printing.__doc__.split('Examples\n    ========')[0]  # type: ignore
init_vprinting.__doc__ = init_vprinting.__doc__.format(params)  # type: ignore
