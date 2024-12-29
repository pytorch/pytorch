"""Hilbert spaces for quantum mechanics.

Authors:
* Brian Granger
* Matt Curry
"""

from functools import reduce

from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.sets.sets import Interval
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.qexpr import QuantumError


__all__ = [
    'HilbertSpaceError',
    'HilbertSpace',
    'TensorProductHilbertSpace',
    'TensorPowerHilbertSpace',
    'DirectSumHilbertSpace',
    'ComplexSpace',
    'L2',
    'FockSpace'
]

#-----------------------------------------------------------------------------
# Main objects
#-----------------------------------------------------------------------------


class HilbertSpaceError(QuantumError):
    pass

#-----------------------------------------------------------------------------
# Main objects
#-----------------------------------------------------------------------------


class HilbertSpace(Basic):
    """An abstract Hilbert space for quantum mechanics.

    In short, a Hilbert space is an abstract vector space that is complete
    with inner products defined [1]_.

    Examples
    ========

    >>> from sympy.physics.quantum.hilbert import HilbertSpace
    >>> hs = HilbertSpace()
    >>> hs
    H

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hilbert_space
    """

    def __new__(cls):
        obj = Basic.__new__(cls)
        return obj

    @property
    def dimension(self):
        """Return the Hilbert dimension of the space."""
        raise NotImplementedError('This Hilbert space has no dimension.')

    def __add__(self, other):
        return DirectSumHilbertSpace(self, other)

    def __radd__(self, other):
        return DirectSumHilbertSpace(other, self)

    def __mul__(self, other):
        return TensorProductHilbertSpace(self, other)

    def __rmul__(self, other):
        return TensorProductHilbertSpace(other, self)

    def __pow__(self, other, mod=None):
        if mod is not None:
            raise ValueError('The third argument to __pow__ is not supported \
            for Hilbert spaces.')
        return TensorPowerHilbertSpace(self, other)

    def __contains__(self, other):
        """Is the operator or state in this Hilbert space.

        This is checked by comparing the classes of the Hilbert spaces, not
        the instances. This is to allow Hilbert Spaces with symbolic
        dimensions.
        """
        if other.hilbert_space.__class__ == self.__class__:
            return True
        else:
            return False

    def _sympystr(self, printer, *args):
        return 'H'

    def _pretty(self, printer, *args):
        ustr = '\N{LATIN CAPITAL LETTER H}'
        return prettyForm(ustr)

    def _latex(self, printer, *args):
        return r'\mathcal{H}'


class ComplexSpace(HilbertSpace):
    """Finite dimensional Hilbert space of complex vectors.

    The elements of this Hilbert space are n-dimensional complex valued
    vectors with the usual inner product that takes the complex conjugate
    of the vector on the right.

    A classic example of this type of Hilbert space is spin-1/2, which is
    ``ComplexSpace(2)``. Generalizing to spin-s, the space is
    ``ComplexSpace(2*s+1)``.  Quantum computing with N qubits is done with the
    direct product space ``ComplexSpace(2)**N``.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.quantum.hilbert import ComplexSpace
    >>> c1 = ComplexSpace(2)
    >>> c1
    C(2)
    >>> c1.dimension
    2

    >>> n = symbols('n')
    >>> c2 = ComplexSpace(n)
    >>> c2
    C(n)
    >>> c2.dimension
    n

    """

    def __new__(cls, dimension):
        dimension = sympify(dimension)
        r = cls.eval(dimension)
        if isinstance(r, Basic):
            return r
        obj = Basic.__new__(cls, dimension)
        return obj

    @classmethod
    def eval(cls, dimension):
        if len(dimension.atoms()) == 1:
            if not (dimension.is_Integer and dimension > 0 or dimension is S.Infinity
            or dimension.is_Symbol):
                raise TypeError('The dimension of a ComplexSpace can only'
                                'be a positive integer, oo, or a Symbol: %r'
                                % dimension)
        else:
            for dim in dimension.atoms():
                if not (dim.is_Integer or dim is S.Infinity or dim.is_Symbol):
                    raise TypeError('The dimension of a ComplexSpace can only'
                                    ' contain integers, oo, or a Symbol: %r'
                                    % dim)

    @property
    def dimension(self):
        return self.args[0]

    def _sympyrepr(self, printer, *args):
        return "%s(%s)" % (self.__class__.__name__,
                           printer._print(self.dimension, *args))

    def _sympystr(self, printer, *args):
        return "C(%s)" % printer._print(self.dimension, *args)

    def _pretty(self, printer, *args):
        ustr = '\N{LATIN CAPITAL LETTER C}'
        pform_exp = printer._print(self.dimension, *args)
        pform_base = prettyForm(ustr)
        return pform_base**pform_exp

    def _latex(self, printer, *args):
        return r'\mathcal{C}^{%s}' % printer._print(self.dimension, *args)


class L2(HilbertSpace):
    """The Hilbert space of square integrable functions on an interval.

    An L2 object takes in a single SymPy Interval argument which represents
    the interval its functions (vectors) are defined on.

    Examples
    ========

    >>> from sympy import Interval, oo
    >>> from sympy.physics.quantum.hilbert import L2
    >>> hs = L2(Interval(0,oo))
    >>> hs
    L2(Interval(0, oo))
    >>> hs.dimension
    oo
    >>> hs.interval
    Interval(0, oo)

    """

    def __new__(cls, interval):
        if not isinstance(interval, Interval):
            raise TypeError('L2 interval must be an Interval instance: %r'
            % interval)
        obj = Basic.__new__(cls, interval)
        return obj

    @property
    def dimension(self):
        return S.Infinity

    @property
    def interval(self):
        return self.args[0]

    def _sympyrepr(self, printer, *args):
        return "L2(%s)" % printer._print(self.interval, *args)

    def _sympystr(self, printer, *args):
        return "L2(%s)" % printer._print(self.interval, *args)

    def _pretty(self, printer, *args):
        pform_exp = prettyForm('2')
        pform_base = prettyForm('L')
        return pform_base**pform_exp

    def _latex(self, printer, *args):
        interval = printer._print(self.interval, *args)
        return r'{\mathcal{L}^2}\left( %s \right)' % interval


class FockSpace(HilbertSpace):
    """The Hilbert space for second quantization.

    Technically, this Hilbert space is a infinite direct sum of direct
    products of single particle Hilbert spaces [1]_. This is a mess, so we have
    a class to represent it directly.

    Examples
    ========

    >>> from sympy.physics.quantum.hilbert import FockSpace
    >>> hs = FockSpace()
    >>> hs
    F
    >>> hs.dimension
    oo

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fock_space
    """

    def __new__(cls):
        obj = Basic.__new__(cls)
        return obj

    @property
    def dimension(self):
        return S.Infinity

    def _sympyrepr(self, printer, *args):
        return "FockSpace()"

    def _sympystr(self, printer, *args):
        return "F"

    def _pretty(self, printer, *args):
        ustr = '\N{LATIN CAPITAL LETTER F}'
        return prettyForm(ustr)

    def _latex(self, printer, *args):
        return r'\mathcal{F}'


class TensorProductHilbertSpace(HilbertSpace):
    """A tensor product of Hilbert spaces [1]_.

    The tensor product between Hilbert spaces is represented by the
    operator ``*`` Products of the same Hilbert space will be combined into
    tensor powers.

    A ``TensorProductHilbertSpace`` object takes in an arbitrary number of
    ``HilbertSpace`` objects as its arguments. In addition, multiplication of
    ``HilbertSpace`` objects will automatically return this tensor product
    object.

    Examples
    ========

    >>> from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace
    >>> from sympy import symbols

    >>> c = ComplexSpace(2)
    >>> f = FockSpace()
    >>> hs = c*f
    >>> hs
    C(2)*F
    >>> hs.dimension
    oo
    >>> hs.spaces
    (C(2), F)

    >>> c1 = ComplexSpace(2)
    >>> n = symbols('n')
    >>> c2 = ComplexSpace(n)
    >>> hs = c1*c2
    >>> hs
    C(2)*C(n)
    >>> hs.dimension
    2*n

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hilbert_space#Tensor_products
    """

    def __new__(cls, *args):
        r = cls.eval(args)
        if isinstance(r, Basic):
            return r
        obj = Basic.__new__(cls, *args)
        return obj

    @classmethod
    def eval(cls, args):
        """Evaluates the direct product."""
        new_args = []
        recall = False
        #flatten arguments
        for arg in args:
            if isinstance(arg, TensorProductHilbertSpace):
                new_args.extend(arg.args)
                recall = True
            elif isinstance(arg, (HilbertSpace, TensorPowerHilbertSpace)):
                new_args.append(arg)
            else:
                raise TypeError('Hilbert spaces can only be multiplied by \
                other Hilbert spaces: %r' % arg)
        #combine like arguments into direct powers
        comb_args = []
        prev_arg = None
        for new_arg in new_args:
            if prev_arg is not None:
                if isinstance(new_arg, TensorPowerHilbertSpace) and \
                    isinstance(prev_arg, TensorPowerHilbertSpace) and \
                        new_arg.base == prev_arg.base:
                    prev_arg = new_arg.base**(new_arg.exp + prev_arg.exp)
                elif isinstance(new_arg, TensorPowerHilbertSpace) and \
                        new_arg.base == prev_arg:
                    prev_arg = prev_arg**(new_arg.exp + 1)
                elif isinstance(prev_arg, TensorPowerHilbertSpace) and \
                        new_arg == prev_arg.base:
                    prev_arg = new_arg**(prev_arg.exp + 1)
                elif new_arg == prev_arg:
                    prev_arg = new_arg**2
                else:
                    comb_args.append(prev_arg)
                    prev_arg = new_arg
            elif prev_arg is None:
                prev_arg = new_arg
        comb_args.append(prev_arg)
        if recall:
            return TensorProductHilbertSpace(*comb_args)
        elif len(comb_args) == 1:
            return TensorPowerHilbertSpace(comb_args[0].base, comb_args[0].exp)
        else:
            return None

    @property
    def dimension(self):
        arg_list = [arg.dimension for arg in self.args]
        if S.Infinity in arg_list:
            return S.Infinity
        else:
            return reduce(lambda x, y: x*y, arg_list)

    @property
    def spaces(self):
        """A tuple of the Hilbert spaces in this tensor product."""
        return self.args

    def _spaces_printer(self, printer, *args):
        spaces_strs = []
        for arg in self.args:
            s = printer._print(arg, *args)
            if isinstance(arg, DirectSumHilbertSpace):
                s = '(%s)' % s
            spaces_strs.append(s)
        return spaces_strs

    def _sympyrepr(self, printer, *args):
        spaces_reprs = self._spaces_printer(printer, *args)
        return "TensorProductHilbertSpace(%s)" % ','.join(spaces_reprs)

    def _sympystr(self, printer, *args):
        spaces_strs = self._spaces_printer(printer, *args)
        return '*'.join(spaces_strs)

    def _pretty(self, printer, *args):
        length = len(self.args)
        pform = printer._print('', *args)
        for i in range(length):
            next_pform = printer._print(self.args[i], *args)
            if isinstance(self.args[i], (DirectSumHilbertSpace,
                          TensorProductHilbertSpace)):
                next_pform = prettyForm(
                    *next_pform.parens(left='(', right=')')
                )
            pform = prettyForm(*pform.right(next_pform))
            if i != length - 1:
                if printer._use_unicode:
                    pform = prettyForm(*pform.right(' ' + '\N{N-ARY CIRCLED TIMES OPERATOR}' + ' '))
                else:
                    pform = prettyForm(*pform.right(' x '))
        return pform

    def _latex(self, printer, *args):
        length = len(self.args)
        s = ''
        for i in range(length):
            arg_s = printer._print(self.args[i], *args)
            if isinstance(self.args[i], (DirectSumHilbertSpace,
                 TensorProductHilbertSpace)):
                arg_s = r'\left(%s\right)' % arg_s
            s = s + arg_s
            if i != length - 1:
                s = s + r'\otimes '
        return s


class DirectSumHilbertSpace(HilbertSpace):
    """A direct sum of Hilbert spaces [1]_.

    This class uses the ``+`` operator to represent direct sums between
    different Hilbert spaces.

    A ``DirectSumHilbertSpace`` object takes in an arbitrary number of
    ``HilbertSpace`` objects as its arguments. Also, addition of
    ``HilbertSpace`` objects will automatically return a direct sum object.

    Examples
    ========

    >>> from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace

    >>> c = ComplexSpace(2)
    >>> f = FockSpace()
    >>> hs = c+f
    >>> hs
    C(2)+F
    >>> hs.dimension
    oo
    >>> list(hs.spaces)
    [C(2), F]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hilbert_space#Direct_sums
    """
    def __new__(cls, *args):
        r = cls.eval(args)
        if isinstance(r, Basic):
            return r
        obj = Basic.__new__(cls, *args)
        return obj

    @classmethod
    def eval(cls, args):
        """Evaluates the direct product."""
        new_args = []
        recall = False
        #flatten arguments
        for arg in args:
            if isinstance(arg, DirectSumHilbertSpace):
                new_args.extend(arg.args)
                recall = True
            elif isinstance(arg, HilbertSpace):
                new_args.append(arg)
            else:
                raise TypeError('Hilbert spaces can only be summed with other \
                Hilbert spaces: %r' % arg)
        if recall:
            return DirectSumHilbertSpace(*new_args)
        else:
            return None

    @property
    def dimension(self):
        arg_list = [arg.dimension for arg in self.args]
        if S.Infinity in arg_list:
            return S.Infinity
        else:
            return reduce(lambda x, y: x + y, arg_list)

    @property
    def spaces(self):
        """A tuple of the Hilbert spaces in this direct sum."""
        return self.args

    def _sympyrepr(self, printer, *args):
        spaces_reprs = [printer._print(arg, *args) for arg in self.args]
        return "DirectSumHilbertSpace(%s)" % ','.join(spaces_reprs)

    def _sympystr(self, printer, *args):
        spaces_strs = [printer._print(arg, *args) for arg in self.args]
        return '+'.join(spaces_strs)

    def _pretty(self, printer, *args):
        length = len(self.args)
        pform = printer._print('', *args)
        for i in range(length):
            next_pform = printer._print(self.args[i], *args)
            if isinstance(self.args[i], (DirectSumHilbertSpace,
                          TensorProductHilbertSpace)):
                next_pform = prettyForm(
                    *next_pform.parens(left='(', right=')')
                )
            pform = prettyForm(*pform.right(next_pform))
            if i != length - 1:
                if printer._use_unicode:
                    pform = prettyForm(*pform.right(' \N{CIRCLED PLUS} '))
                else:
                    pform = prettyForm(*pform.right(' + '))
        return pform

    def _latex(self, printer, *args):
        length = len(self.args)
        s = ''
        for i in range(length):
            arg_s = printer._print(self.args[i], *args)
            if isinstance(self.args[i], (DirectSumHilbertSpace,
                 TensorProductHilbertSpace)):
                arg_s = r'\left(%s\right)' % arg_s
            s = s + arg_s
            if i != length - 1:
                s = s + r'\oplus '
        return s


class TensorPowerHilbertSpace(HilbertSpace):
    """An exponentiated Hilbert space [1]_.

    Tensor powers (repeated tensor products) are represented by the
    operator ``**`` Identical Hilbert spaces that are multiplied together
    will be automatically combined into a single tensor power object.

    Any Hilbert space, product, or sum may be raised to a tensor power. The
    ``TensorPowerHilbertSpace`` takes two arguments: the Hilbert space; and the
    tensor power (number).

    Examples
    ========

    >>> from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace
    >>> from sympy import symbols

    >>> n = symbols('n')
    >>> c = ComplexSpace(2)
    >>> hs = c**n
    >>> hs
    C(2)**n
    >>> hs.dimension
    2**n

    >>> c = ComplexSpace(2)
    >>> c*c
    C(2)**2
    >>> f = FockSpace()
    >>> c*f*f
    C(2)*F**2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hilbert_space#Tensor_products
    """

    def __new__(cls, *args):
        r = cls.eval(args)
        if isinstance(r, Basic):
            return r
        return Basic.__new__(cls, *r)

    @classmethod
    def eval(cls, args):
        new_args = args[0], sympify(args[1])
        exp = new_args[1]
        #simplify hs**1 -> hs
        if exp is S.One:
            return args[0]
        #simplify hs**0 -> 1
        if exp is S.Zero:
            return S.One
        #check (and allow) for hs**(x+42+y...) case
        if len(exp.atoms()) == 1:
            if not (exp.is_Integer and exp >= 0 or exp.is_Symbol):
                raise ValueError('Hilbert spaces can only be raised to \
                positive integers or Symbols: %r' % exp)
        else:
            for power in exp.atoms():
                if not (power.is_Integer or power.is_Symbol):
                    raise ValueError('Tensor powers can only contain integers \
                    or Symbols: %r' % power)
        return new_args

    @property
    def base(self):
        return self.args[0]

    @property
    def exp(self):
        return self.args[1]

    @property
    def dimension(self):
        if self.base.dimension is S.Infinity:
            return S.Infinity
        else:
            return self.base.dimension**self.exp

    def _sympyrepr(self, printer, *args):
        return "TensorPowerHilbertSpace(%s,%s)" % (printer._print(self.base,
        *args), printer._print(self.exp, *args))

    def _sympystr(self, printer, *args):
        return "%s**%s" % (printer._print(self.base, *args),
        printer._print(self.exp, *args))

    def _pretty(self, printer, *args):
        pform_exp = printer._print(self.exp, *args)
        if printer._use_unicode:
            pform_exp = prettyForm(*pform_exp.left(prettyForm('\N{N-ARY CIRCLED TIMES OPERATOR}')))
        else:
            pform_exp = prettyForm(*pform_exp.left(prettyForm('x')))
        pform_base = printer._print(self.base, *args)
        return pform_base**pform_exp

    def _latex(self, printer, *args):
        base = printer._print(self.base, *args)
        exp = printer._print(self.exp, *args)
        return r'{%s}^{\otimes %s}' % (base, exp)
