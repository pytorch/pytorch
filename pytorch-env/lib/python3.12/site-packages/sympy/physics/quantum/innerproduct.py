"""Symbolic inner product."""

from sympy.core.expr import Expr
from sympy.functions.elementary.complexes import conjugate
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.state import KetBase, BraBase

__all__ = [
    'InnerProduct'
]


# InnerProduct is not an QExpr because it is really just a regular commutative
# number. We have gone back and forth about this, but we gain a lot by having
# it subclass Expr. The main challenges were getting Dagger to work
# (we use _eval_conjugate) and represent (we can use atoms and subs). Having
# it be an Expr, mean that there are no commutative QExpr subclasses,
# which simplifies the design of everything.

class InnerProduct(Expr):
    """An unevaluated inner product between a Bra and a Ket [1].

    Parameters
    ==========

    bra : BraBase or subclass
        The bra on the left side of the inner product.
    ket : KetBase or subclass
        The ket on the right side of the inner product.

    Examples
    ========

    Create an InnerProduct and check its properties:

        >>> from sympy.physics.quantum import Bra, Ket
        >>> b = Bra('b')
        >>> k = Ket('k')
        >>> ip = b*k
        >>> ip
        <b|k>
        >>> ip.bra
        <b|
        >>> ip.ket
        |k>

    In simple products of kets and bras inner products will be automatically
    identified and created::

        >>> b*k
        <b|k>

    But in more complex expressions, there is ambiguity in whether inner or
    outer products should be created::

        >>> k*b*k*b
        |k><b|*|k>*<b|

    A user can force the creation of a inner products in a complex expression
    by using parentheses to group the bra and ket::

        >>> k*(b*k)*b
        <b|k>*|k>*<b|

    Notice how the inner product <b|k> moved to the left of the expression
    because inner products are commutative complex numbers.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inner_product
    """
    is_complex = True

    def __new__(cls, bra, ket):
        if not isinstance(ket, KetBase):
            raise TypeError('KetBase subclass expected, got: %r' % ket)
        if not isinstance(bra, BraBase):
            raise TypeError('BraBase subclass expected, got: %r' % ket)
        obj = Expr.__new__(cls, bra, ket)
        return obj

    @property
    def bra(self):
        return self.args[0]

    @property
    def ket(self):
        return self.args[1]

    def _eval_conjugate(self):
        return InnerProduct(Dagger(self.ket), Dagger(self.bra))

    def _sympyrepr(self, printer, *args):
        return '%s(%s,%s)' % (self.__class__.__name__,
            printer._print(self.bra, *args), printer._print(self.ket, *args))

    def _sympystr(self, printer, *args):
        sbra = printer._print(self.bra)
        sket = printer._print(self.ket)
        return '%s|%s' % (sbra[:-1], sket[1:])

    def _pretty(self, printer, *args):
        # Print state contents
        bra = self.bra._print_contents_pretty(printer, *args)
        ket = self.ket._print_contents_pretty(printer, *args)
        # Print brackets
        height = max(bra.height(), ket.height())
        use_unicode = printer._use_unicode
        lbracket, _ = self.bra._pretty_brackets(height, use_unicode)
        cbracket, rbracket = self.ket._pretty_brackets(height, use_unicode)
        # Build innerproduct
        pform = prettyForm(*bra.left(lbracket))
        pform = prettyForm(*pform.right(cbracket))
        pform = prettyForm(*pform.right(ket))
        pform = prettyForm(*pform.right(rbracket))
        return pform

    def _latex(self, printer, *args):
        bra_label = self.bra._print_contents_latex(printer, *args)
        ket = printer._print(self.ket, *args)
        return r'\left\langle %s \right. %s' % (bra_label, ket)

    def doit(self, **hints):
        try:
            r = self.ket._eval_innerproduct(self.bra, **hints)
        except NotImplementedError:
            try:
                r = conjugate(
                    self.bra.dual._eval_innerproduct(self.ket.dual, **hints)
                )
            except NotImplementedError:
                r = None
        if r is not None:
            return r
        return self
