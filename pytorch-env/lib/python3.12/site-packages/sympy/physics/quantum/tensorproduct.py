"""Abstract tensor product."""

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.sympify import sympify
from sympy.matrices.dense import DenseMatrix as Matrix
from sympy.matrices.immutable import ImmutableDenseMatrix as ImmutableMatrix
from sympy.printing.pretty.stringpict import prettyForm

from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.state import Ket, Bra
from sympy.physics.quantum.matrixutils import (
    numpy_ndarray,
    scipy_sparse_matrix,
    matrix_tensor_product
)
from sympy.physics.quantum.trace import Tr


__all__ = [
    'TensorProduct',
    'tensor_product_simp'
]

#-----------------------------------------------------------------------------
# Tensor product
#-----------------------------------------------------------------------------

_combined_printing = False


def combined_tensor_printing(combined):
    """Set flag controlling whether tensor products of states should be
    printed as a combined bra/ket or as an explicit tensor product of different
    bra/kets. This is a global setting for all TensorProduct class instances.

    Parameters
    ----------
    combine : bool
        When true, tensor product states are combined into one ket/bra, and
        when false explicit tensor product notation is used between each
        ket/bra.
    """
    global _combined_printing
    _combined_printing = combined


class TensorProduct(Expr):
    """The tensor product of two or more arguments.

    For matrices, this uses ``matrix_tensor_product`` to compute the Kronecker
    or tensor product matrix. For other objects a symbolic ``TensorProduct``
    instance is returned. The tensor product is a non-commutative
    multiplication that is used primarily with operators and states in quantum
    mechanics.

    Currently, the tensor product distinguishes between commutative and
    non-commutative arguments.  Commutative arguments are assumed to be scalars
    and are pulled out in front of the ``TensorProduct``. Non-commutative
    arguments remain in the resulting ``TensorProduct``.

    Parameters
    ==========

    args : tuple
        A sequence of the objects to take the tensor product of.

    Examples
    ========

    Start with a simple tensor product of SymPy matrices::

        >>> from sympy import Matrix
        >>> from sympy.physics.quantum import TensorProduct

        >>> m1 = Matrix([[1,2],[3,4]])
        >>> m2 = Matrix([[1,0],[0,1]])
        >>> TensorProduct(m1, m2)
        Matrix([
        [1, 0, 2, 0],
        [0, 1, 0, 2],
        [3, 0, 4, 0],
        [0, 3, 0, 4]])
        >>> TensorProduct(m2, m1)
        Matrix([
        [1, 2, 0, 0],
        [3, 4, 0, 0],
        [0, 0, 1, 2],
        [0, 0, 3, 4]])

    We can also construct tensor products of non-commutative symbols:

        >>> from sympy import Symbol
        >>> A = Symbol('A',commutative=False)
        >>> B = Symbol('B',commutative=False)
        >>> tp = TensorProduct(A, B)
        >>> tp
        AxB

    We can take the dagger of a tensor product (note the order does NOT reverse
    like the dagger of a normal product):

        >>> from sympy.physics.quantum import Dagger
        >>> Dagger(tp)
        Dagger(A)xDagger(B)

    Expand can be used to distribute a tensor product across addition:

        >>> C = Symbol('C',commutative=False)
        >>> tp = TensorProduct(A+B,C)
        >>> tp
        (A + B)xC
        >>> tp.expand(tensorproduct=True)
        AxC + BxC
    """
    is_commutative = False

    def __new__(cls, *args):
        if isinstance(args[0], (Matrix, ImmutableMatrix, numpy_ndarray,
                                                    scipy_sparse_matrix)):
            return matrix_tensor_product(*args)
        c_part, new_args = cls.flatten(sympify(args))
        c_part = Mul(*c_part)
        if len(new_args) == 0:
            return c_part
        elif len(new_args) == 1:
            return c_part * new_args[0]
        else:
            tp = Expr.__new__(cls, *new_args)
            return c_part * tp

    @classmethod
    def flatten(cls, args):
        # TODO: disallow nested TensorProducts.
        c_part = []
        nc_parts = []
        for arg in args:
            cp, ncp = arg.args_cnc()
            c_part.extend(list(cp))
            nc_parts.append(Mul._from_args(ncp))
        return c_part, nc_parts

    def _eval_adjoint(self):
        return TensorProduct(*[Dagger(i) for i in self.args])

    def _eval_rewrite(self, rule, args, **hints):
        return TensorProduct(*args).expand(tensorproduct=True)

    def _sympystr(self, printer, *args):
        length = len(self.args)
        s = ''
        for i in range(length):
            if isinstance(self.args[i], (Add, Pow, Mul)):
                s = s + '('
            s = s + printer._print(self.args[i])
            if isinstance(self.args[i], (Add, Pow, Mul)):
                s = s + ')'
            if i != length - 1:
                s = s + 'x'
        return s

    def _pretty(self, printer, *args):

        if (_combined_printing and
                (all(isinstance(arg, Ket) for arg in self.args) or
                 all(isinstance(arg, Bra) for arg in self.args))):

            length = len(self.args)
            pform = printer._print('', *args)
            for i in range(length):
                next_pform = printer._print('', *args)
                length_i = len(self.args[i].args)
                for j in range(length_i):
                    part_pform = printer._print(self.args[i].args[j], *args)
                    next_pform = prettyForm(*next_pform.right(part_pform))
                    if j != length_i - 1:
                        next_pform = prettyForm(*next_pform.right(', '))

                if len(self.args[i].args) > 1:
                    next_pform = prettyForm(
                        *next_pform.parens(left='{', right='}'))
                pform = prettyForm(*pform.right(next_pform))
                if i != length - 1:
                    pform = prettyForm(*pform.right(',' + ' '))

            pform = prettyForm(*pform.left(self.args[0].lbracket))
            pform = prettyForm(*pform.right(self.args[0].rbracket))
            return pform

        length = len(self.args)
        pform = printer._print('', *args)
        for i in range(length):
            next_pform = printer._print(self.args[i], *args)
            if isinstance(self.args[i], (Add, Mul)):
                next_pform = prettyForm(
                    *next_pform.parens(left='(', right=')')
                )
            pform = prettyForm(*pform.right(next_pform))
            if i != length - 1:
                if printer._use_unicode:
                    pform = prettyForm(*pform.right('\N{N-ARY CIRCLED TIMES OPERATOR}' + ' '))
                else:
                    pform = prettyForm(*pform.right('x' + ' '))
        return pform

    def _latex(self, printer, *args):

        if (_combined_printing and
                (all(isinstance(arg, Ket) for arg in self.args) or
                 all(isinstance(arg, Bra) for arg in self.args))):

            def _label_wrap(label, nlabels):
                return label if nlabels == 1 else r"\left\{%s\right\}" % label

            s = r", ".join([_label_wrap(arg._print_label_latex(printer, *args),
                                        len(arg.args)) for arg in self.args])

            return r"{%s%s%s}" % (self.args[0].lbracket_latex, s,
                                  self.args[0].rbracket_latex)

        length = len(self.args)
        s = ''
        for i in range(length):
            if isinstance(self.args[i], (Add, Mul)):
                s = s + '\\left('
            # The extra {} brackets are needed to get matplotlib's latex
            # rendered to render this properly.
            s = s + '{' + printer._print(self.args[i], *args) + '}'
            if isinstance(self.args[i], (Add, Mul)):
                s = s + '\\right)'
            if i != length - 1:
                s = s + '\\otimes '
        return s

    def doit(self, **hints):
        return TensorProduct(*[item.doit(**hints) for item in self.args])

    def _eval_expand_tensorproduct(self, **hints):
        """Distribute TensorProducts across addition."""
        args = self.args
        add_args = []
        for i in range(len(args)):
            if isinstance(args[i], Add):
                for aa in args[i].args:
                    tp = TensorProduct(*args[:i] + (aa,) + args[i + 1:])
                    c_part, nc_part = tp.args_cnc()
                    # Check for TensorProduct object: is the one object in nc_part, if any:
                    # (Note: any other object type to be expanded must be added here)
                    if len(nc_part) == 1 and isinstance(nc_part[0], TensorProduct):
                        nc_part = (nc_part[0]._eval_expand_tensorproduct(), )
                    add_args.append(Mul(*c_part)*Mul(*nc_part))
                break

        if add_args:
            return Add(*add_args)
        else:
            return self

    def _eval_trace(self, **kwargs):
        indices = kwargs.get('indices', None)
        exp = tensor_product_simp(self)

        if indices is None or len(indices) == 0:
            return Mul(*[Tr(arg).doit() for arg in exp.args])
        else:
            return Mul(*[Tr(value).doit() if idx in indices else value
                         for idx, value in enumerate(exp.args)])


def tensor_product_simp_Mul(e):
    """Simplify a Mul with TensorProducts.

    Current the main use of this is to simplify a ``Mul`` of ``TensorProduct``s
    to a ``TensorProduct`` of ``Muls``. It currently only works for relatively
    simple cases where the initial ``Mul`` only has scalars and raw
    ``TensorProduct``s, not ``Add``, ``Pow``, ``Commutator``s of
    ``TensorProduct``s.

    Parameters
    ==========

    e : Expr
        A ``Mul`` of ``TensorProduct``s to be simplified.

    Returns
    =======

    e : Expr
        A ``TensorProduct`` of ``Mul``s.

    Examples
    ========

    This is an example of the type of simplification that this function
    performs::

        >>> from sympy.physics.quantum.tensorproduct import \
                    tensor_product_simp_Mul, TensorProduct
        >>> from sympy import Symbol
        >>> A = Symbol('A',commutative=False)
        >>> B = Symbol('B',commutative=False)
        >>> C = Symbol('C',commutative=False)
        >>> D = Symbol('D',commutative=False)
        >>> e = TensorProduct(A,B)*TensorProduct(C,D)
        >>> e
        AxB*CxD
        >>> tensor_product_simp_Mul(e)
        (A*C)x(B*D)

    """
    # TODO: This won't work with Muls that have other composites of
    # TensorProducts, like an Add, Commutator, etc.
    # TODO: This only works for the equivalent of single Qbit gates.
    if not isinstance(e, Mul):
        return e
    c_part, nc_part = e.args_cnc()
    n_nc = len(nc_part)
    if n_nc == 0:
        return e
    elif n_nc == 1:
        if isinstance(nc_part[0], Pow):
            return  Mul(*c_part) * tensor_product_simp_Pow(nc_part[0])
        return e
    elif e.has(TensorProduct):
        current = nc_part[0]
        if not isinstance(current, TensorProduct):
            if isinstance(current, Pow):
                if isinstance(current.base, TensorProduct):
                    current = tensor_product_simp_Pow(current)
            else:
                raise TypeError('TensorProduct expected, got: %r' % current)
        n_terms = len(current.args)
        new_args = list(current.args)
        for next in nc_part[1:]:
            # TODO: check the hilbert spaces of next and current here.
            if isinstance(next, TensorProduct):
                if n_terms != len(next.args):
                    raise QuantumError(
                        'TensorProducts of different lengths: %r and %r' %
                        (current, next)
                    )
                for i in range(len(new_args)):
                    new_args[i] = new_args[i] * next.args[i]
            else:
                if isinstance(next, Pow):
                    if isinstance(next.base, TensorProduct):
                        new_tp = tensor_product_simp_Pow(next)
                        for i in range(len(new_args)):
                            new_args[i] = new_args[i] * new_tp.args[i]
                    else:
                        raise TypeError('TensorProduct expected, got: %r' % next)
                else:
                    raise TypeError('TensorProduct expected, got: %r' % next)
            current = next
        return Mul(*c_part) * TensorProduct(*new_args)
    elif e.has(Pow):
        new_args = [ tensor_product_simp_Pow(nc) for nc in nc_part ]
        return tensor_product_simp_Mul(Mul(*c_part) * TensorProduct(*new_args))
    else:
        return e

def tensor_product_simp_Pow(e):
    """Evaluates ``Pow`` expressions whose base is ``TensorProduct``"""
    if not isinstance(e, Pow):
        return e

    if isinstance(e.base, TensorProduct):
        return TensorProduct(*[ b**e.exp for b in e.base.args])
    else:
        return e

def tensor_product_simp(e, **hints):
    """Try to simplify and combine TensorProducts.

    In general this will try to pull expressions inside of ``TensorProducts``.
    It currently only works for relatively simple cases where the products have
    only scalars, raw ``TensorProducts``, not ``Add``, ``Pow``, ``Commutators``
    of ``TensorProducts``. It is best to see what it does by showing examples.

    Examples
    ========

    >>> from sympy.physics.quantum import tensor_product_simp
    >>> from sympy.physics.quantum import TensorProduct
    >>> from sympy import Symbol
    >>> A = Symbol('A',commutative=False)
    >>> B = Symbol('B',commutative=False)
    >>> C = Symbol('C',commutative=False)
    >>> D = Symbol('D',commutative=False)

    First see what happens to products of tensor products:

    >>> e = TensorProduct(A,B)*TensorProduct(C,D)
    >>> e
    AxB*CxD
    >>> tensor_product_simp(e)
    (A*C)x(B*D)

    This is the core logic of this function, and it works inside, powers, sums,
    commutators and anticommutators as well:

    >>> tensor_product_simp(e**2)
    (A*C)x(B*D)**2

    """
    if isinstance(e, Add):
        return Add(*[tensor_product_simp(arg) for arg in e.args])
    elif isinstance(e, Pow):
        if isinstance(e.base, TensorProduct):
            return tensor_product_simp_Pow(e)
        else:
            return tensor_product_simp(e.base) ** e.exp
    elif isinstance(e, Mul):
        return tensor_product_simp_Mul(e)
    elif isinstance(e, Commutator):
        return Commutator(*[tensor_product_simp(arg) for arg in e.args])
    elif isinstance(e, AntiCommutator):
        return AntiCommutator(*[tensor_product_simp(arg) for arg in e.args])
    else:
        return e
