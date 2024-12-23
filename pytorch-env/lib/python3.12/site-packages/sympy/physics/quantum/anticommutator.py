"""The anti-commutator: ``{A,B} = A*B + B*A``."""

from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.printing.pretty.stringpict import prettyForm

from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.dagger import Dagger

__all__ = [
    'AntiCommutator'
]

#-----------------------------------------------------------------------------
# Anti-commutator
#-----------------------------------------------------------------------------


class AntiCommutator(Expr):
    """The standard anticommutator, in an unevaluated state.

    Explanation
    ===========

    Evaluating an anticommutator is defined [1]_ as: ``{A, B} = A*B + B*A``.
    This class returns the anticommutator in an unevaluated form.  To evaluate
    the anticommutator, use the ``.doit()`` method.

    Canonical ordering of an anticommutator is ``{A, B}`` for ``A < B``. The
    arguments of the anticommutator are put into canonical order using
    ``__cmp__``. If ``B < A``, then ``{A, B}`` is returned as ``{B, A}``.

    Parameters
    ==========

    A : Expr
        The first argument of the anticommutator {A,B}.
    B : Expr
        The second argument of the anticommutator {A,B}.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.quantum import AntiCommutator
    >>> from sympy.physics.quantum import Operator, Dagger
    >>> x, y = symbols('x,y')
    >>> A = Operator('A')
    >>> B = Operator('B')

    Create an anticommutator and use ``doit()`` to multiply them out.

    >>> ac = AntiCommutator(A,B); ac
    {A,B}
    >>> ac.doit()
    A*B + B*A

    The commutator orders it arguments in canonical order:

    >>> ac = AntiCommutator(B,A); ac
    {A,B}

    Commutative constants are factored out:

    >>> AntiCommutator(3*x*A,x*y*B)
    3*x**2*y*{A,B}

    Adjoint operations applied to the anticommutator are properly applied to
    the arguments:

    >>> Dagger(AntiCommutator(A,B))
    {Dagger(A),Dagger(B)}

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Commutator
    """
    is_commutative = False

    def __new__(cls, A, B):
        r = cls.eval(A, B)
        if r is not None:
            return r
        obj = Expr.__new__(cls, A, B)
        return obj

    @classmethod
    def eval(cls, a, b):
        if not (a and b):
            return S.Zero
        if a == b:
            return Integer(2)*a**2
        if a.is_commutative or b.is_commutative:
            return Integer(2)*a*b

        # [xA,yB]  ->  xy*[A,B]
        ca, nca = a.args_cnc()
        cb, ncb = b.args_cnc()
        c_part = ca + cb
        if c_part:
            return Mul(Mul(*c_part), cls(Mul._from_args(nca), Mul._from_args(ncb)))

        # Canonical ordering of arguments
        #The Commutator [A,B] is on canonical form if A < B.
        if a.compare(b) == 1:
            return cls(b, a)

    def doit(self, **hints):
        """ Evaluate anticommutator """
        A = self.args[0]
        B = self.args[1]
        if isinstance(A, Operator) and isinstance(B, Operator):
            try:
                comm = A._eval_anticommutator(B, **hints)
            except NotImplementedError:
                try:
                    comm = B._eval_anticommutator(A, **hints)
                except NotImplementedError:
                    comm = None
            if comm is not None:
                return comm.doit(**hints)
        return (A*B + B*A).doit(**hints)

    def _eval_adjoint(self):
        return AntiCommutator(Dagger(self.args[0]), Dagger(self.args[1]))

    def _sympyrepr(self, printer, *args):
        return "%s(%s,%s)" % (
            self.__class__.__name__, printer._print(
                self.args[0]), printer._print(self.args[1])
        )

    def _sympystr(self, printer, *args):
        return "{%s,%s}" % (
            printer._print(self.args[0]), printer._print(self.args[1]))

    def _pretty(self, printer, *args):
        pform = printer._print(self.args[0], *args)
        pform = prettyForm(*pform.right(prettyForm(',')))
        pform = prettyForm(*pform.right(printer._print(self.args[1], *args)))
        pform = prettyForm(*pform.parens(left='{', right='}'))
        return pform

    def _latex(self, printer, *args):
        return "\\left\\{%s,%s\\right\\}" % tuple([
            printer._print(arg, *args) for arg in self.args])
