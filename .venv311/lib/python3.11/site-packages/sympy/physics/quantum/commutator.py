"""The commutator: [A,B] = A*B - B*A."""

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.kind import KindDispatcher
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.printing.pretty.stringpict import prettyForm

from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.kind import _OperatorKind, OperatorKind


__all__ = [
    'Commutator'
]

#-----------------------------------------------------------------------------
# Commutator
#-----------------------------------------------------------------------------


class Commutator(Expr):
    """The standard commutator, in an unevaluated state.

    Explanation
    ===========

    Evaluating a commutator is defined [1]_ as: ``[A, B] = A*B - B*A``. This
    class returns the commutator in an unevaluated form. To evaluate the
    commutator, use the ``.doit()`` method.

    Canonical ordering of a commutator is ``[A, B]`` for ``A < B``. The
    arguments of the commutator are put into canonical order using ``__cmp__``.
    If ``B < A``, then ``[B, A]`` is returned as ``-[A, B]``.

    Parameters
    ==========

    A : Expr
        The first argument of the commutator [A,B].
    B : Expr
        The second argument of the commutator [A,B].

    Examples
    ========

    >>> from sympy.physics.quantum import Commutator, Dagger, Operator
    >>> from sympy.abc import x, y
    >>> A = Operator('A')
    >>> B = Operator('B')
    >>> C = Operator('C')

    Create a commutator and use ``.doit()`` to evaluate it:

    >>> comm = Commutator(A, B)
    >>> comm
    [A,B]
    >>> comm.doit()
    A*B - B*A

    The commutator orders it arguments in canonical order:

    >>> comm = Commutator(B, A); comm
    -[A,B]

    Commutative constants are factored out:

    >>> Commutator(3*x*A, x*y*B)
    3*x**2*y*[A,B]

    Using ``.expand(commutator=True)``, the standard commutator expansion rules
    can be applied:

    >>> Commutator(A+B, C).expand(commutator=True)
    [A,C] + [B,C]
    >>> Commutator(A, B+C).expand(commutator=True)
    [A,B] + [A,C]
    >>> Commutator(A*B, C).expand(commutator=True)
    [A,C]*B + A*[B,C]
    >>> Commutator(A, B*C).expand(commutator=True)
    [A,B]*C + B*[A,C]

    Adjoint operations applied to the commutator are properly applied to the
    arguments:

    >>> Dagger(Commutator(A, B))
    -[Dagger(A),Dagger(B)]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Commutator
    """
    is_commutative = False

    _kind_dispatcher = KindDispatcher("Commutator_kind_dispatcher", commutative=True)

    @property
    def kind(self):
        arg_kinds = (a.kind for a in self.args)
        return self._kind_dispatcher(*arg_kinds)

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
            return S.Zero
        if a.is_commutative or b.is_commutative:
            return S.Zero

        # [xA,yB]  ->  xy*[A,B]
        ca, nca = a.args_cnc()
        cb, ncb = b.args_cnc()
        c_part = ca + cb
        if c_part:
            return Mul(Mul(*c_part), cls(Mul._from_args(nca), Mul._from_args(ncb)))

        # Canonical ordering of arguments
        # The Commutator [A, B] is in canonical form if A < B.
        if a.compare(b) == 1:
            return S.NegativeOne*cls(b, a)

    def _expand_pow(self, A, B, sign):
        exp = A.exp
        if not exp.is_integer or not exp.is_constant() or abs(exp) <= 1:
            # nothing to do
            return self
        base = A.base
        if exp.is_negative:
            base = A.base**-1
            exp = -exp
        comm = Commutator(base, B).expand(commutator=True)

        result = base**(exp - 1) * comm
        for i in range(1, exp):
            result += base**(exp - 1 - i) * comm * base**i
        return sign*result.expand()

    def _eval_expand_commutator(self, **hints):
        A = self.args[0]
        B = self.args[1]

        if isinstance(A, Add):
            # [A + B, C]  ->  [A, C] + [B, C]
            sargs = []
            for term in A.args:
                comm = Commutator(term, B)
                if isinstance(comm, Commutator):
                    comm = comm._eval_expand_commutator()
                sargs.append(comm)
            return Add(*sargs)
        elif isinstance(B, Add):
            # [A, B + C]  ->  [A, B] + [A, C]
            sargs = []
            for term in B.args:
                comm = Commutator(A, term)
                if isinstance(comm, Commutator):
                    comm = comm._eval_expand_commutator()
                sargs.append(comm)
            return Add(*sargs)
        elif isinstance(A, Mul):
            # [A*B, C] -> A*[B, C] + [A, C]*B
            a = A.args[0]
            b = Mul(*A.args[1:])
            c = B
            comm1 = Commutator(b, c)
            comm2 = Commutator(a, c)
            if isinstance(comm1, Commutator):
                comm1 = comm1._eval_expand_commutator()
            if isinstance(comm2, Commutator):
                comm2 = comm2._eval_expand_commutator()
            first = Mul(a, comm1)
            second = Mul(comm2, b)
            return Add(first, second)
        elif isinstance(B, Mul):
            # [A, B*C] -> [A, B]*C + B*[A, C]
            a = A
            b = B.args[0]
            c = Mul(*B.args[1:])
            comm1 = Commutator(a, b)
            comm2 = Commutator(a, c)
            if isinstance(comm1, Commutator):
                comm1 = comm1._eval_expand_commutator()
            if isinstance(comm2, Commutator):
                comm2 = comm2._eval_expand_commutator()
            first = Mul(comm1, c)
            second = Mul(b, comm2)
            return Add(first, second)
        elif isinstance(A, Pow):
            # [A**n, C] -> A**(n - 1)*[A, C] + A**(n - 2)*[A, C]*A + ... + [A, C]*A**(n-1)
            return self._expand_pow(A, B, 1)
        elif isinstance(B, Pow):
            # [A, C**n] -> C**(n - 1)*[C, A] + C**(n - 2)*[C, A]*C + ... + [C, A]*C**(n-1)
            return self._expand_pow(B, A, -1)

        # No changes, so return self
        return self

    def doit(self, **hints):
        """ Evaluate commutator """
        # Keep the import of Operator here to avoid problems with
        # circular imports.
        from sympy.physics.quantum.operator import Operator
        A = self.args[0]
        B = self.args[1]
        if isinstance(A, Operator) and isinstance(B, Operator):
            try:
                comm = A._eval_commutator(B, **hints)
            except NotImplementedError:
                try:
                    comm = -1*B._eval_commutator(A, **hints)
                except NotImplementedError:
                    comm = None
            if comm is not None:
                return comm.doit(**hints)
        return (A*B - B*A).doit(**hints)

    def _eval_adjoint(self):
        return Commutator(Dagger(self.args[1]), Dagger(self.args[0]))

    def _sympyrepr(self, printer, *args):
        return "%s(%s,%s)" % (
            self.__class__.__name__, printer._print(
                self.args[0]), printer._print(self.args[1])
        )

    def _sympystr(self, printer, *args):
        return "[%s,%s]" % (
            printer._print(self.args[0]), printer._print(self.args[1]))

    def _pretty(self, printer, *args):
        pform = printer._print(self.args[0], *args)
        pform = prettyForm(*pform.right(prettyForm(',')))
        pform = prettyForm(*pform.right(printer._print(self.args[1], *args)))
        pform = prettyForm(*pform.parens(left='[', right=']'))
        return pform

    def _latex(self, printer, *args):
        return "\\left[%s,%s\\right]" % tuple([
            printer._print(arg, *args) for arg in self.args])


@Commutator._kind_dispatcher.register(_OperatorKind, _OperatorKind)
def find_op_kind(e1, e2):
    """Find the kind of an anticommutator of two OperatorKinds."""
    return OperatorKind
