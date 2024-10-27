"""Quantum mechanical angular momemtum."""

from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.numbers import int_valued
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.simplify.simplify import simplify
from sympy.matrices import zeros
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.printing.pretty.pretty_symbology import pretty_symbol

from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.operator import (HermitianOperator, Operator,
                                            UnitaryOperator)
from sympy.physics.quantum.state import Bra, Ket, State
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.hilbert import ComplexSpace, DirectSumHilbertSpace
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.qapply import qapply


__all__ = [
    'm_values',
    'Jplus',
    'Jminus',
    'Jx',
    'Jy',
    'Jz',
    'J2',
    'Rotation',
    'WignerD',
    'JxKet',
    'JxBra',
    'JyKet',
    'JyBra',
    'JzKet',
    'JzBra',
    'JzOp',
    'J2Op',
    'JxKetCoupled',
    'JxBraCoupled',
    'JyKetCoupled',
    'JyBraCoupled',
    'JzKetCoupled',
    'JzBraCoupled',
    'couple',
    'uncouple'
]


def m_values(j):
    j = sympify(j)
    size = 2*j + 1
    if not size.is_Integer or not size > 0:
        raise ValueError(
            'Only integer or half-integer values allowed for j, got: : %r' % j
        )
    return size, [j - i for i in range(int(2*j + 1))]


#-----------------------------------------------------------------------------
# Spin Operators
#-----------------------------------------------------------------------------


class SpinOpBase:
    """Base class for spin operators."""

    @classmethod
    def _eval_hilbert_space(cls, label):
        # We consider all j values so our space is infinite.
        return ComplexSpace(S.Infinity)

    @property
    def name(self):
        return self.args[0]

    def _print_contents(self, printer, *args):
        return '%s%s' % (self.name, self._coord)

    def _print_contents_pretty(self, printer, *args):
        a = stringPict(str(self.name))
        b = stringPict(self._coord)
        return self._print_subscript_pretty(a, b)

    def _print_contents_latex(self, printer, *args):
        return r'%s_%s' % ((self.name, self._coord))

    def _represent_base(self, basis, **options):
        j = options.get('j', S.Half)
        size, mvals = m_values(j)
        result = zeros(size, size)
        for p in range(size):
            for q in range(size):
                me = self.matrix_element(j, mvals[p], j, mvals[q])
                result[p, q] = me
        return result

    def _apply_op(self, ket, orig_basis, **options):
        state = ket.rewrite(self.basis)
        # If the state has only one term
        if isinstance(state, State):
            ret = (hbar*state.m)*state
        # state is a linear combination of states
        elif isinstance(state, Sum):
            ret = self._apply_operator_Sum(state, **options)
        else:
            ret = qapply(self*state)
        if ret == self*state:
            raise NotImplementedError
        return ret.rewrite(orig_basis)

    def _apply_operator_JxKet(self, ket, **options):
        return self._apply_op(ket, 'Jx', **options)

    def _apply_operator_JxKetCoupled(self, ket, **options):
        return self._apply_op(ket, 'Jx', **options)

    def _apply_operator_JyKet(self, ket, **options):
        return self._apply_op(ket, 'Jy', **options)

    def _apply_operator_JyKetCoupled(self, ket, **options):
        return self._apply_op(ket, 'Jy', **options)

    def _apply_operator_JzKet(self, ket, **options):
        return self._apply_op(ket, 'Jz', **options)

    def _apply_operator_JzKetCoupled(self, ket, **options):
        return self._apply_op(ket, 'Jz', **options)

    def _apply_operator_TensorProduct(self, tp, **options):
        # Uncoupling operator is only easily found for coordinate basis spin operators
        # TODO: add methods for uncoupling operators
        if not isinstance(self, (JxOp, JyOp, JzOp)):
            raise NotImplementedError
        result = []
        for n in range(len(tp.args)):
            arg = []
            arg.extend(tp.args[:n])
            arg.append(self._apply_operator(tp.args[n]))
            arg.extend(tp.args[n + 1:])
            result.append(tp.__class__(*arg))
        return Add(*result).expand()

    # TODO: move this to qapply_Mul
    def _apply_operator_Sum(self, s, **options):
        new_func = qapply(self*s.function)
        if new_func == self*s.function:
            raise NotImplementedError
        return Sum(new_func, *s.limits)

    def _eval_trace(self, **options):
        #TODO: use options to use different j values
        #For now eval at default basis

        # is it efficient to represent each time
        # to do a trace?
        return self._represent_default_basis().trace()


class JplusOp(SpinOpBase, Operator):
    """The J+ operator."""

    _coord = '+'

    basis = 'Jz'

    def _eval_commutator_JminusOp(self, other):
        return 2*hbar*JzOp(self.name)

    def _apply_operator_JzKet(self, ket, **options):
        j = ket.j
        m = ket.m
        if m.is_Number and j.is_Number:
            if m >= j:
                return S.Zero
        return hbar*sqrt(j*(j + S.One) - m*(m + S.One))*JzKet(j, m + S.One)

    def _apply_operator_JzKetCoupled(self, ket, **options):
        j = ket.j
        m = ket.m
        jn = ket.jn
        coupling = ket.coupling
        if m.is_Number and j.is_Number:
            if m >= j:
                return S.Zero
        return hbar*sqrt(j*(j + S.One) - m*(m + S.One))*JzKetCoupled(j, m + S.One, jn, coupling)

    def matrix_element(self, j, m, jp, mp):
        result = hbar*sqrt(j*(j + S.One) - mp*(mp + S.One))
        result *= KroneckerDelta(m, mp + 1)
        result *= KroneckerDelta(j, jp)
        return result

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(basis, **options)

    def _eval_rewrite_as_xyz(self, *args, **kwargs):
        return JxOp(args[0]) + I*JyOp(args[0])


class JminusOp(SpinOpBase, Operator):
    """The J- operator."""

    _coord = '-'

    basis = 'Jz'

    def _apply_operator_JzKet(self, ket, **options):
        j = ket.j
        m = ket.m
        if m.is_Number and j.is_Number:
            if m <= -j:
                return S.Zero
        return hbar*sqrt(j*(j + S.One) - m*(m - S.One))*JzKet(j, m - S.One)

    def _apply_operator_JzKetCoupled(self, ket, **options):
        j = ket.j
        m = ket.m
        jn = ket.jn
        coupling = ket.coupling
        if m.is_Number and j.is_Number:
            if m <= -j:
                return S.Zero
        return hbar*sqrt(j*(j + S.One) - m*(m - S.One))*JzKetCoupled(j, m - S.One, jn, coupling)

    def matrix_element(self, j, m, jp, mp):
        result = hbar*sqrt(j*(j + S.One) - mp*(mp - S.One))
        result *= KroneckerDelta(m, mp - 1)
        result *= KroneckerDelta(j, jp)
        return result

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(basis, **options)

    def _eval_rewrite_as_xyz(self, *args, **kwargs):
        return JxOp(args[0]) - I*JyOp(args[0])


class JxOp(SpinOpBase, HermitianOperator):
    """The Jx operator."""

    _coord = 'x'

    basis = 'Jx'

    def _eval_commutator_JyOp(self, other):
        return I*hbar*JzOp(self.name)

    def _eval_commutator_JzOp(self, other):
        return -I*hbar*JyOp(self.name)

    def _apply_operator_JzKet(self, ket, **options):
        jp = JplusOp(self.name)._apply_operator_JzKet(ket, **options)
        jm = JminusOp(self.name)._apply_operator_JzKet(ket, **options)
        return (jp + jm)/Integer(2)

    def _apply_operator_JzKetCoupled(self, ket, **options):
        jp = JplusOp(self.name)._apply_operator_JzKetCoupled(ket, **options)
        jm = JminusOp(self.name)._apply_operator_JzKetCoupled(ket, **options)
        return (jp + jm)/Integer(2)

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        jp = JplusOp(self.name)._represent_JzOp(basis, **options)
        jm = JminusOp(self.name)._represent_JzOp(basis, **options)
        return (jp + jm)/Integer(2)

    def _eval_rewrite_as_plusminus(self, *args, **kwargs):
        return (JplusOp(args[0]) + JminusOp(args[0]))/2


class JyOp(SpinOpBase, HermitianOperator):
    """The Jy operator."""

    _coord = 'y'

    basis = 'Jy'

    def _eval_commutator_JzOp(self, other):
        return I*hbar*JxOp(self.name)

    def _eval_commutator_JxOp(self, other):
        return -I*hbar*J2Op(self.name)

    def _apply_operator_JzKet(self, ket, **options):
        jp = JplusOp(self.name)._apply_operator_JzKet(ket, **options)
        jm = JminusOp(self.name)._apply_operator_JzKet(ket, **options)
        return (jp - jm)/(Integer(2)*I)

    def _apply_operator_JzKetCoupled(self, ket, **options):
        jp = JplusOp(self.name)._apply_operator_JzKetCoupled(ket, **options)
        jm = JminusOp(self.name)._apply_operator_JzKetCoupled(ket, **options)
        return (jp - jm)/(Integer(2)*I)

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        jp = JplusOp(self.name)._represent_JzOp(basis, **options)
        jm = JminusOp(self.name)._represent_JzOp(basis, **options)
        return (jp - jm)/(Integer(2)*I)

    def _eval_rewrite_as_plusminus(self, *args, **kwargs):
        return (JplusOp(args[0]) - JminusOp(args[0]))/(2*I)


class JzOp(SpinOpBase, HermitianOperator):
    """The Jz operator."""

    _coord = 'z'

    basis = 'Jz'

    def _eval_commutator_JxOp(self, other):
        return I*hbar*JyOp(self.name)

    def _eval_commutator_JyOp(self, other):
        return -I*hbar*JxOp(self.name)

    def _eval_commutator_JplusOp(self, other):
        return hbar*JplusOp(self.name)

    def _eval_commutator_JminusOp(self, other):
        return -hbar*JminusOp(self.name)

    def matrix_element(self, j, m, jp, mp):
        result = hbar*mp
        result *= KroneckerDelta(m, mp)
        result *= KroneckerDelta(j, jp)
        return result

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(basis, **options)


class J2Op(SpinOpBase, HermitianOperator):
    """The J^2 operator."""

    _coord = '2'

    def _eval_commutator_JxOp(self, other):
        return S.Zero

    def _eval_commutator_JyOp(self, other):
        return S.Zero

    def _eval_commutator_JzOp(self, other):
        return S.Zero

    def _eval_commutator_JplusOp(self, other):
        return S.Zero

    def _eval_commutator_JminusOp(self, other):
        return S.Zero

    def _apply_operator_JxKet(self, ket, **options):
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def _apply_operator_JxKetCoupled(self, ket, **options):
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def _apply_operator_JyKet(self, ket, **options):
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def _apply_operator_JyKetCoupled(self, ket, **options):
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def _apply_operator_JzKet(self, ket, **options):
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def _apply_operator_JzKetCoupled(self, ket, **options):
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def matrix_element(self, j, m, jp, mp):
        result = (hbar**2)*j*(j + 1)
        result *= KroneckerDelta(m, mp)
        result *= KroneckerDelta(j, jp)
        return result

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(basis, **options)

    def _print_contents_pretty(self, printer, *args):
        a = prettyForm(str(self.name))
        b = prettyForm('2')
        return a**b

    def _print_contents_latex(self, printer, *args):
        return r'%s^2' % str(self.name)

    def _eval_rewrite_as_xyz(self, *args, **kwargs):
        return JxOp(args[0])**2 + JyOp(args[0])**2 + JzOp(args[0])**2

    def _eval_rewrite_as_plusminus(self, *args, **kwargs):
        a = args[0]
        return JzOp(a)**2 + \
            S.Half*(JplusOp(a)*JminusOp(a) + JminusOp(a)*JplusOp(a))


class Rotation(UnitaryOperator):
    """Wigner D operator in terms of Euler angles.

    Defines the rotation operator in terms of the Euler angles defined by
    the z-y-z convention for a passive transformation. That is the coordinate
    axes are rotated first about the z-axis, giving the new x'-y'-z' axes. Then
    this new coordinate system is rotated about the new y'-axis, giving new
    x''-y''-z'' axes. Then this new coordinate system is rotated about the
    z''-axis. Conventions follow those laid out in [1]_.

    Parameters
    ==========

    alpha : Number, Symbol
        First Euler Angle
    beta : Number, Symbol
        Second Euler angle
    gamma : Number, Symbol
        Third Euler angle

    Examples
    ========

    A simple example rotation operator:

        >>> from sympy import pi
        >>> from sympy.physics.quantum.spin import Rotation
        >>> Rotation(pi, 0, pi/2)
        R(pi,0,pi/2)

    With symbolic Euler angles and calculating the inverse rotation operator:

        >>> from sympy import symbols
        >>> a, b, c = symbols('a b c')
        >>> Rotation(a, b, c)
        R(a,b,c)
        >>> Rotation(a, b, c).inverse()
        R(-c,-b,-a)

    See Also
    ========

    WignerD: Symbolic Wigner-D function
    D: Wigner-D function
    d: Wigner small-d function

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    """

    @classmethod
    def _eval_args(cls, args):
        args = QExpr._eval_args(args)
        if len(args) != 3:
            raise ValueError('3 Euler angles required, got: %r' % args)
        return args

    @classmethod
    def _eval_hilbert_space(cls, label):
        # We consider all j values so our space is infinite.
        return ComplexSpace(S.Infinity)

    @property
    def alpha(self):
        return self.label[0]

    @property
    def beta(self):
        return self.label[1]

    @property
    def gamma(self):
        return self.label[2]

    def _print_operator_name(self, printer, *args):
        return 'R'

    def _print_operator_name_pretty(self, printer, *args):
        if printer._use_unicode:
            return prettyForm('\N{SCRIPT CAPITAL R}' + ' ')
        else:
            return prettyForm("R ")

    def _print_operator_name_latex(self, printer, *args):
        return r'\mathcal{R}'

    def _eval_inverse(self):
        return Rotation(-self.gamma, -self.beta, -self.alpha)

    @classmethod
    def D(cls, j, m, mp, alpha, beta, gamma):
        """Wigner D-function.

        Returns an instance of the WignerD class corresponding to the Wigner-D
        function specified by the parameters.

        Parameters
        ===========

        j : Number
            Total angular momentum
        m : Number
            Eigenvalue of angular momentum along axis after rotation
        mp : Number
            Eigenvalue of angular momentum along rotated axis
        alpha : Number, Symbol
            First Euler angle of rotation
        beta : Number, Symbol
            Second Euler angle of rotation
        gamma : Number, Symbol
            Third Euler angle of rotation

        Examples
        ========

        Return the Wigner-D matrix element for a defined rotation, both
        numerical and symbolic:

            >>> from sympy.physics.quantum.spin import Rotation
            >>> from sympy import pi, symbols
            >>> alpha, beta, gamma = symbols('alpha beta gamma')
            >>> Rotation.D(1, 1, 0,pi, pi/2,-pi)
            WignerD(1, 1, 0, pi, pi/2, -pi)

        See Also
        ========

        WignerD: Symbolic Wigner-D function

        """
        return WignerD(j, m, mp, alpha, beta, gamma)

    @classmethod
    def d(cls, j, m, mp, beta):
        """Wigner small-d function.

        Returns an instance of the WignerD class corresponding to the Wigner-D
        function specified by the parameters with the alpha and gamma angles
        given as 0.

        Parameters
        ===========

        j : Number
            Total angular momentum
        m : Number
            Eigenvalue of angular momentum along axis after rotation
        mp : Number
            Eigenvalue of angular momentum along rotated axis
        beta : Number, Symbol
            Second Euler angle of rotation

        Examples
        ========

        Return the Wigner-D matrix element for a defined rotation, both
        numerical and symbolic:

            >>> from sympy.physics.quantum.spin import Rotation
            >>> from sympy import pi, symbols
            >>> beta = symbols('beta')
            >>> Rotation.d(1, 1, 0, pi/2)
            WignerD(1, 1, 0, 0, pi/2, 0)

        See Also
        ========

        WignerD: Symbolic Wigner-D function

        """
        return WignerD(j, m, mp, 0, beta, 0)

    def matrix_element(self, j, m, jp, mp):
        result = self.__class__.D(
            jp, m, mp, self.alpha, self.beta, self.gamma
        )
        result *= KroneckerDelta(j, jp)
        return result

    def _represent_base(self, basis, **options):
        j = sympify(options.get('j', S.Half))
        # TODO: move evaluation up to represent function/implement elsewhere
        evaluate = sympify(options.get('doit'))
        size, mvals = m_values(j)
        result = zeros(size, size)
        for p in range(size):
            for q in range(size):
                me = self.matrix_element(j, mvals[p], j, mvals[q])
                if evaluate:
                    result[p, q] = me.doit()
                else:
                    result[p, q] = me
        return result

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(basis, **options)

    def _apply_operator_uncoupled(self, state, ket, *, dummy=True, **options):
        a = self.alpha
        b = self.beta
        g = self.gamma
        j = ket.j
        m = ket.m
        if j.is_number:
            s = []
            size = m_values(j)
            sz = size[1]
            for mp in sz:
                r = Rotation.D(j, m, mp, a, b, g)
                z = r.doit()
                s.append(z*state(j, mp))
            return Add(*s)
        else:
            if dummy:
                mp = Dummy('mp')
            else:
                mp = symbols('mp')
            return Sum(Rotation.D(j, m, mp, a, b, g)*state(j, mp), (mp, -j, j))

    def _apply_operator_JxKet(self, ket, **options):
        return self._apply_operator_uncoupled(JxKet, ket, **options)

    def _apply_operator_JyKet(self, ket, **options):
        return self._apply_operator_uncoupled(JyKet, ket, **options)

    def _apply_operator_JzKet(self, ket, **options):
        return self._apply_operator_uncoupled(JzKet, ket, **options)

    def _apply_operator_coupled(self, state, ket, *, dummy=True, **options):
        a = self.alpha
        b = self.beta
        g = self.gamma
        j = ket.j
        m = ket.m
        jn = ket.jn
        coupling = ket.coupling
        if j.is_number:
            s = []
            size = m_values(j)
            sz = size[1]
            for mp in sz:
                r = Rotation.D(j, m, mp, a, b, g)
                z = r.doit()
                s.append(z*state(j, mp, jn, coupling))
            return Add(*s)
        else:
            if dummy:
                mp = Dummy('mp')
            else:
                mp = symbols('mp')
            return Sum(Rotation.D(j, m, mp, a, b, g)*state(
                j, mp, jn, coupling), (mp, -j, j))

    def _apply_operator_JxKetCoupled(self, ket, **options):
        return self._apply_operator_coupled(JxKetCoupled, ket, **options)

    def _apply_operator_JyKetCoupled(self, ket, **options):
        return self._apply_operator_coupled(JyKetCoupled, ket, **options)

    def _apply_operator_JzKetCoupled(self, ket, **options):
        return self._apply_operator_coupled(JzKetCoupled, ket, **options)

class WignerD(Expr):
    r"""Wigner-D function

    The Wigner D-function gives the matrix elements of the rotation
    operator in the jm-representation. For the Euler angles `\alpha`,
    `\beta`, `\gamma`, the D-function is defined such that:

    .. math ::
        <j,m| \mathcal{R}(\alpha, \beta, \gamma ) |j',m'> = \delta_{jj'} D(j, m, m', \alpha, \beta, \gamma)

    Where the rotation operator is as defined by the Rotation class [1]_.

    The Wigner D-function defined in this way gives:

    .. math ::
        D(j, m, m', \alpha, \beta, \gamma) = e^{-i m \alpha} d(j, m, m', \beta) e^{-i m' \gamma}

    Where d is the Wigner small-d function, which is given by Rotation.d.

    The Wigner small-d function gives the component of the Wigner
    D-function that is determined by the second Euler angle. That is the
    Wigner D-function is:

    .. math ::
        D(j, m, m', \alpha, \beta, \gamma) = e^{-i m \alpha} d(j, m, m', \beta) e^{-i m' \gamma}

    Where d is the small-d function. The Wigner D-function is given by
    Rotation.D.

    Note that to evaluate the D-function, the j, m and mp parameters must
    be integer or half integer numbers.

    Parameters
    ==========

    j : Number
        Total angular momentum
    m : Number
        Eigenvalue of angular momentum along axis after rotation
    mp : Number
        Eigenvalue of angular momentum along rotated axis
    alpha : Number, Symbol
        First Euler angle of rotation
    beta : Number, Symbol
        Second Euler angle of rotation
    gamma : Number, Symbol
        Third Euler angle of rotation

    Examples
    ========

    Evaluate the Wigner-D matrix elements of a simple rotation:

        >>> from sympy.physics.quantum.spin import Rotation
        >>> from sympy import pi
        >>> rot = Rotation.D(1, 1, 0, pi, pi/2, 0)
        >>> rot
        WignerD(1, 1, 0, pi, pi/2, 0)
        >>> rot.doit()
        sqrt(2)/2

    Evaluate the Wigner-d matrix elements of a simple rotation

        >>> rot = Rotation.d(1, 1, 0, pi/2)
        >>> rot
        WignerD(1, 1, 0, 0, pi/2, 0)
        >>> rot.doit()
        -sqrt(2)/2

    See Also
    ========

    Rotation: Rotation operator

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    """

    is_commutative = True

    def __new__(cls, *args, **hints):
        if not len(args) == 6:
            raise ValueError('6 parameters expected, got %s' % args)
        args = sympify(args)
        evaluate = hints.get('evaluate', False)
        if evaluate:
            return Expr.__new__(cls, *args)._eval_wignerd()
        return Expr.__new__(cls, *args)

    @property
    def j(self):
        return self.args[0]

    @property
    def m(self):
        return self.args[1]

    @property
    def mp(self):
        return self.args[2]

    @property
    def alpha(self):
        return self.args[3]

    @property
    def beta(self):
        return self.args[4]

    @property
    def gamma(self):
        return self.args[5]

    def _latex(self, printer, *args):
        if self.alpha == 0 and self.gamma == 0:
            return r'd^{%s}_{%s,%s}\left(%s\right)' % \
                (
                    printer._print(self.j), printer._print(
                        self.m), printer._print(self.mp),
                    printer._print(self.beta) )
        return r'D^{%s}_{%s,%s}\left(%s,%s,%s\right)' % \
            (
                printer._print(
                    self.j), printer._print(self.m), printer._print(self.mp),
                printer._print(self.alpha), printer._print(self.beta), printer._print(self.gamma) )

    def _pretty(self, printer, *args):
        top = printer._print(self.j)

        bot = printer._print(self.m)
        bot = prettyForm(*bot.right(','))
        bot = prettyForm(*bot.right(printer._print(self.mp)))

        pad = max(top.width(), bot.width())
        top = prettyForm(*top.left(' '))
        bot = prettyForm(*bot.left(' '))
        if pad > top.width():
            top = prettyForm(*top.right(' '*(pad - top.width())))
        if pad > bot.width():
            bot = prettyForm(*bot.right(' '*(pad - bot.width())))
        if self.alpha == 0 and self.gamma == 0:
            args = printer._print(self.beta)
            s = stringPict('d' + ' '*pad)
        else:
            args = printer._print(self.alpha)
            args = prettyForm(*args.right(','))
            args = prettyForm(*args.right(printer._print(self.beta)))
            args = prettyForm(*args.right(','))
            args = prettyForm(*args.right(printer._print(self.gamma)))

            s = stringPict('D' + ' '*pad)

        args = prettyForm(*args.parens())
        s = prettyForm(*s.above(top))
        s = prettyForm(*s.below(bot))
        s = prettyForm(*s.right(args))
        return s

    def doit(self, **hints):
        hints['evaluate'] = True
        return WignerD(*self.args, **hints)

    def _eval_wignerd(self):
        j = self.j
        m = self.m
        mp = self.mp
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        if alpha == 0 and beta == 0 and gamma == 0:
            return KroneckerDelta(m, mp)
        if not j.is_number:
            raise ValueError(
                'j parameter must be numerical to evaluate, got %s' % j)
        r = 0
        if beta == pi/2:
            # Varshalovich Equation (5), Section 4.16, page 113, setting
            # alpha=gamma=0.
            for k in range(2*j + 1):
                if k > j + mp or k > j - m or k < mp - m:
                    continue
                r += (S.NegativeOne)**k*binomial(j + mp, k)*binomial(j - mp, k + m - mp)
            r *= (S.NegativeOne)**(m - mp) / 2**j*sqrt(factorial(j + m) *
                    factorial(j - m) / (factorial(j + mp)*factorial(j - mp)))
        else:
            # Varshalovich Equation(5), Section 4.7.2, page 87, where we set
            # beta1=beta2=pi/2, and we get alpha=gamma=pi/2 and beta=phi+pi,
            # then we use the Eq. (1), Section 4.4. page 79, to simplify:
            # d(j, m, mp, beta+pi) = (-1)**(j-mp)*d(j, m, -mp, beta)
            # This happens to be almost the same as in Eq.(10), Section 4.16,
            # except that we need to substitute -mp for mp.
            size, mvals = m_values(j)
            for mpp in mvals:
                r += Rotation.d(j, m, mpp, pi/2).doit()*(cos(-mpp*beta) + I*sin(-mpp*beta))*\
                    Rotation.d(j, mpp, -mp, pi/2).doit()
            # Empirical normalization factor so results match Varshalovich
            # Tables 4.3-4.12
            # Note that this exact normalization does not follow from the
            # above equations
            r = r*I**(2*j - m - mp)*(-1)**(2*m)
            # Finally, simplify the whole expression
            r = simplify(r)
        r *= exp(-I*m*alpha)*exp(-I*mp*gamma)
        return r


Jx = JxOp('J')
Jy = JyOp('J')
Jz = JzOp('J')
J2 = J2Op('J')
Jplus = JplusOp('J')
Jminus = JminusOp('J')


#-----------------------------------------------------------------------------
# Spin States
#-----------------------------------------------------------------------------


class SpinState(State):
    """Base class for angular momentum states."""

    _label_separator = ','

    def __new__(cls, j, m):
        j = sympify(j)
        m = sympify(m)
        if j.is_number:
            if 2*j != int(2*j):
                raise ValueError(
                    'j must be integer or half-integer, got: %s' % j)
            if j < 0:
                raise ValueError('j must be >= 0, got: %s' % j)
        if m.is_number:
            if 2*m != int(2*m):
                raise ValueError(
                    'm must be integer or half-integer, got: %s' % m)
        if j.is_number and m.is_number:
            if abs(m) > j:
                raise ValueError('Allowed values for m are -j <= m <= j, got j, m: %s, %s' % (j, m))
            if int(j - m) != j - m:
                raise ValueError('Both j and m must be integer or half-integer, got j, m: %s, %s' % (j, m))
        return State.__new__(cls, j, m)

    @property
    def j(self):
        return self.label[0]

    @property
    def m(self):
        return self.label[1]

    @classmethod
    def _eval_hilbert_space(cls, label):
        return ComplexSpace(2*label[0] + 1)

    def _represent_base(self, **options):
        j = self.j
        m = self.m
        alpha = sympify(options.get('alpha', 0))
        beta = sympify(options.get('beta', 0))
        gamma = sympify(options.get('gamma', 0))
        size, mvals = m_values(j)
        result = zeros(size, 1)
        # breaks finding angles on L930
        for p, mval in enumerate(mvals):
            if m.is_number:
                result[p, 0] = Rotation.D(
                    self.j, mval, self.m, alpha, beta, gamma).doit()
            else:
                result[p, 0] = Rotation.D(self.j, mval,
                                          self.m, alpha, beta, gamma)
        return result

    def _eval_rewrite_as_Jx(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jx, JxBra, **options)
        return self._rewrite_basis(Jx, JxKet, **options)

    def _eval_rewrite_as_Jy(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jy, JyBra, **options)
        return self._rewrite_basis(Jy, JyKet, **options)

    def _eval_rewrite_as_Jz(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jz, JzBra, **options)
        return self._rewrite_basis(Jz, JzKet, **options)

    def _rewrite_basis(self, basis, evect, **options):
        from sympy.physics.quantum.represent import represent
        j = self.j
        args = self.args[2:]
        if j.is_number:
            if isinstance(self, CoupledSpinState):
                if j == int(j):
                    start = j**2
                else:
                    start = (2*j - 1)*(2*j + 1)/4
            else:
                start = 0
            vect = represent(self, basis=basis, **options)
            result = Add(
                *[vect[start + i]*evect(j, j - i, *args) for i in range(2*j + 1)])
            if isinstance(self, CoupledSpinState) and options.get('coupled') is False:
                return uncouple(result)
            return result
        else:
            i = 0
            mi = symbols('mi')
            # make sure not to introduce a symbol already in the state
            while self.subs(mi, 0) != self:
                i += 1
                mi = symbols('mi%d' % i)
                break
            # TODO: better way to get angles of rotation
            if isinstance(self, CoupledSpinState):
                test_args = (0, mi, (0, 0))
            else:
                test_args = (0, mi)
            if isinstance(self, Ket):
                angles = represent(
                    self.__class__(*test_args), basis=basis)[0].args[3:6]
            else:
                angles = represent(self.__class__(
                    *test_args), basis=basis)[0].args[0].args[3:6]
            if angles == (0, 0, 0):
                return self
            else:
                state = evect(j, mi, *args)
                lt = Rotation.D(j, mi, self.m, *angles)
                return Sum(lt*state, (mi, -j, j))

    def _eval_innerproduct_JxBra(self, bra, **hints):
        result = KroneckerDelta(self.j, bra.j)
        if bra.dual_class() is not self.__class__:
            result *= self._represent_JxOp(None)[bra.j - bra.m]
        else:
            result *= KroneckerDelta(
                self.j, bra.j)*KroneckerDelta(self.m, bra.m)
        return result

    def _eval_innerproduct_JyBra(self, bra, **hints):
        result = KroneckerDelta(self.j, bra.j)
        if bra.dual_class() is not self.__class__:
            result *= self._represent_JyOp(None)[bra.j - bra.m]
        else:
            result *= KroneckerDelta(
                self.j, bra.j)*KroneckerDelta(self.m, bra.m)
        return result

    def _eval_innerproduct_JzBra(self, bra, **hints):
        result = KroneckerDelta(self.j, bra.j)
        if bra.dual_class() is not self.__class__:
            result *= self._represent_JzOp(None)[bra.j - bra.m]
        else:
            result *= KroneckerDelta(
                self.j, bra.j)*KroneckerDelta(self.m, bra.m)
        return result

    def _eval_trace(self, bra, **hints):

        # One way to implement this method is to assume the basis set k is
        # passed.
        # Then we can apply the discrete form of Trace formula here
        # Tr(|i><j| ) = \Sum_k <k|i><j|k>
        #then we do qapply() on each each inner product and sum over them.

        # OR

        # Inner product of |i><j| = Trace(Outer Product).
        # we could just use this unless there are cases when this is not true

        return (bra*self).doit()


class JxKet(SpinState, Ket):
    """Eigenket of Jx.

    See JzKet for the usage of spin eigenstates.

    See Also
    ========

    JzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        return JxBra

    @classmethod
    def coupled_class(self):
        return JxKetCoupled

    def _represent_default_basis(self, **options):
        return self._represent_JxOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        return self._represent_base(**options)

    def _represent_JyOp(self, basis, **options):
        return self._represent_base(alpha=pi*Rational(3, 2), **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(beta=pi/2, **options)


class JxBra(SpinState, Bra):
    """Eigenbra of Jx.

    See JzKet for the usage of spin eigenstates.

    See Also
    ========

    JzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        return JxKet

    @classmethod
    def coupled_class(self):
        return JxBraCoupled


class JyKet(SpinState, Ket):
    """Eigenket of Jy.

    See JzKet for the usage of spin eigenstates.

    See Also
    ========

    JzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        return JyBra

    @classmethod
    def coupled_class(self):
        return JyKetCoupled

    def _represent_default_basis(self, **options):
        return self._represent_JyOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        return self._represent_base(gamma=pi/2, **options)

    def _represent_JyOp(self, basis, **options):
        return self._represent_base(**options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(alpha=pi*Rational(3, 2), beta=-pi/2, gamma=pi/2, **options)


class JyBra(SpinState, Bra):
    """Eigenbra of Jy.

    See JzKet for the usage of spin eigenstates.

    See Also
    ========

    JzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        return JyKet

    @classmethod
    def coupled_class(self):
        return JyBraCoupled


class JzKet(SpinState, Ket):
    """Eigenket of Jz.

    Spin state which is an eigenstate of the Jz operator. Uncoupled states,
    that is states representing the interaction of multiple separate spin
    states, are defined as a tensor product of states.

    Parameters
    ==========

    j : Number, Symbol
        Total spin angular momentum
    m : Number, Symbol
        Eigenvalue of the Jz spin operator

    Examples
    ========

    *Normal States:*

    Defining simple spin states, both numerical and symbolic:

        >>> from sympy.physics.quantum.spin import JzKet, JxKet
        >>> from sympy import symbols
        >>> JzKet(1, 0)
        |1,0>
        >>> j, m = symbols('j m')
        >>> JzKet(j, m)
        |j,m>

    Rewriting the JzKet in terms of eigenkets of the Jx operator:
    Note: that the resulting eigenstates are JxKet's

        >>> JzKet(1,1).rewrite("Jx")
        |1,-1>/2 - sqrt(2)*|1,0>/2 + |1,1>/2

    Get the vector representation of a state in terms of the basis elements
    of the Jx operator:

        >>> from sympy.physics.quantum.represent import represent
        >>> from sympy.physics.quantum.spin import Jx, Jz
        >>> represent(JzKet(1,-1), basis=Jx)
        Matrix([
        [      1/2],
        [sqrt(2)/2],
        [      1/2]])

    Apply innerproducts between states:

        >>> from sympy.physics.quantum.innerproduct import InnerProduct
        >>> from sympy.physics.quantum.spin import JxBra
        >>> i = InnerProduct(JxBra(1,1), JzKet(1,1))
        >>> i
        <1,1|1,1>
        >>> i.doit()
        1/2

    *Uncoupled States:*

    Define an uncoupled state as a TensorProduct between two Jz eigenkets:

        >>> from sympy.physics.quantum.tensorproduct import TensorProduct
        >>> j1,m1,j2,m2 = symbols('j1 m1 j2 m2')
        >>> TensorProduct(JzKet(1,0), JzKet(1,1))
        |1,0>x|1,1>
        >>> TensorProduct(JzKet(j1,m1), JzKet(j2,m2))
        |j1,m1>x|j2,m2>

    A TensorProduct can be rewritten, in which case the eigenstates that make
    up the tensor product is rewritten to the new basis:

        >>> TensorProduct(JzKet(1,1),JxKet(1,1)).rewrite('Jz')
        |1,1>x|1,-1>/2 + sqrt(2)*|1,1>x|1,0>/2 + |1,1>x|1,1>/2

    The represent method for TensorProduct's gives the vector representation of
    the state. Note that the state in the product basis is the equivalent of the
    tensor product of the vector representation of the component eigenstates:

        >>> represent(TensorProduct(JzKet(1,0),JzKet(1,1)))
        Matrix([
        [0],
        [0],
        [0],
        [1],
        [0],
        [0],
        [0],
        [0],
        [0]])
        >>> represent(TensorProduct(JzKet(1,1),JxKet(1,1)), basis=Jz)
        Matrix([
        [      1/2],
        [sqrt(2)/2],
        [      1/2],
        [        0],
        [        0],
        [        0],
        [        0],
        [        0],
        [        0]])

    See Also
    ========

    JzKetCoupled: Coupled eigenstates
    sympy.physics.quantum.tensorproduct.TensorProduct: Used to specify uncoupled states
    uncouple: Uncouples states given coupling parameters
    couple: Couples uncoupled states

    """

    @classmethod
    def dual_class(self):
        return JzBra

    @classmethod
    def coupled_class(self):
        return JzKetCoupled

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        return self._represent_base(beta=pi*Rational(3, 2), **options)

    def _represent_JyOp(self, basis, **options):
        return self._represent_base(alpha=pi*Rational(3, 2), beta=pi/2, gamma=pi/2, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(**options)


class JzBra(SpinState, Bra):
    """Eigenbra of Jz.

    See the JzKet for the usage of spin eigenstates.

    See Also
    ========

    JzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        return JzKet

    @classmethod
    def coupled_class(self):
        return JzBraCoupled


# Method used primarily to create coupled_n and coupled_jn by __new__ in
# CoupledSpinState
# This same method is also used by the uncouple method, and is separated from
# the CoupledSpinState class to maintain consistency in defining coupling
def _build_coupled(jcoupling, length):
    n_list = [ [n + 1] for n in range(length) ]
    coupled_jn = []
    coupled_n = []
    for n1, n2, j_new in jcoupling:
        coupled_jn.append(j_new)
        coupled_n.append( (n_list[n1 - 1], n_list[n2 - 1]) )
        n_sort = sorted(n_list[n1 - 1] + n_list[n2 - 1])
        n_list[n_sort[0] - 1] = n_sort
    return coupled_n, coupled_jn


class CoupledSpinState(SpinState):
    """Base class for coupled angular momentum states."""

    def __new__(cls, j, m, jn, *jcoupling):
        # Check j and m values using SpinState
        SpinState(j, m)
        # Build and check coupling scheme from arguments
        if len(jcoupling) == 0:
            # Use default coupling scheme
            jcoupling = []
            for n in range(2, len(jn)):
                jcoupling.append( (1, n, Add(*[jn[i] for i in range(n)])) )
            jcoupling.append( (1, len(jn), j) )
        elif len(jcoupling) == 1:
            # Use specified coupling scheme
            jcoupling = jcoupling[0]
        else:
            raise TypeError("CoupledSpinState only takes 3 or 4 arguments, got: %s" % (len(jcoupling) + 3) )
        # Check arguments have correct form
        if not isinstance(jn, (list, tuple, Tuple)):
            raise TypeError('jn must be Tuple, list or tuple, got %s' %
                            jn.__class__.__name__)
        if not isinstance(jcoupling, (list, tuple, Tuple)):
            raise TypeError('jcoupling must be Tuple, list or tuple, got %s' %
                            jcoupling.__class__.__name__)
        if not all(isinstance(term, (list, tuple, Tuple)) for term in jcoupling):
            raise TypeError(
                'All elements of jcoupling must be list, tuple or Tuple')
        if not len(jn) - 1 == len(jcoupling):
            raise ValueError('jcoupling must have length of %d, got %d' %
                             (len(jn) - 1, len(jcoupling)))
        if not all(len(x) == 3 for x in jcoupling):
            raise ValueError('All elements of jcoupling must have length 3')
        # Build sympified args
        j = sympify(j)
        m = sympify(m)
        jn = Tuple( *[sympify(ji) for ji in jn] )
        jcoupling = Tuple( *[Tuple(sympify(
            n1), sympify(n2), sympify(ji)) for (n1, n2, ji) in jcoupling] )
        # Check values in coupling scheme give physical state
        if any(2*ji != int(2*ji) for ji in jn if ji.is_number):
            raise ValueError('All elements of jn must be integer or half-integer, got: %s' % jn)
        if any(n1 != int(n1) or n2 != int(n2) for (n1, n2, _) in jcoupling):
            raise ValueError('Indices in jcoupling must be integers')
        if any(n1 < 1 or n2 < 1 or n1 > len(jn) or n2 > len(jn) for (n1, n2, _) in jcoupling):
            raise ValueError('Indices must be between 1 and the number of coupled spin spaces')
        if any(2*ji != int(2*ji) for (_, _, ji) in jcoupling if ji.is_number):
            raise ValueError('All coupled j values in coupling scheme must be integer or half-integer')
        coupled_n, coupled_jn = _build_coupled(jcoupling, len(jn))
        jvals = list(jn)
        for n, (n1, n2) in enumerate(coupled_n):
            j1 = jvals[min(n1) - 1]
            j2 = jvals[min(n2) - 1]
            j3 = coupled_jn[n]
            if sympify(j1).is_number and sympify(j2).is_number and sympify(j3).is_number:
                if j1 + j2 < j3:
                    raise ValueError('All couplings must have j1+j2 >= j3, '
                        'in coupling number %d got j1,j2,j3: %d,%d,%d' % (n + 1, j1, j2, j3))
                if abs(j1 - j2) > j3:
                    raise ValueError("All couplings must have |j1+j2| <= j3, "
                        "in coupling number %d got j1,j2,j3: %d,%d,%d" % (n + 1, j1, j2, j3))
                if int_valued(j1 + j2):
                    pass
            jvals[min(n1 + n2) - 1] = j3
        if len(jcoupling) > 0 and jcoupling[-1][2] != j:
            raise ValueError('Last j value coupled together must be the final j of the state')
        # Return state
        return State.__new__(cls, j, m, jn, jcoupling)

    def _print_label(self, printer, *args):
        label = [printer._print(self.j), printer._print(self.m)]
        for i, ji in enumerate(self.jn, start=1):
            label.append('j%d=%s' % (
                i, printer._print(ji)
            ))
        for jn, (n1, n2) in zip(self.coupled_jn[:-1], self.coupled_n[:-1]):
            label.append('j(%s)=%s' % (
                ','.join(str(i) for i in sorted(n1 + n2)), printer._print(jn)
            ))
        return ','.join(label)

    def _print_label_pretty(self, printer, *args):
        label = [self.j, self.m]
        for i, ji in enumerate(self.jn, start=1):
            symb = 'j%d' % i
            symb = pretty_symbol(symb)
            symb = prettyForm(symb + '=')
            item = prettyForm(*symb.right(printer._print(ji)))
            label.append(item)
        for jn, (n1, n2) in zip(self.coupled_jn[:-1], self.coupled_n[:-1]):
            n = ','.join(pretty_symbol("j%d" % i)[-1] for i in sorted(n1 + n2))
            symb = prettyForm('j' + n + '=')
            item = prettyForm(*symb.right(printer._print(jn)))
            label.append(item)
        return self._print_sequence_pretty(
            label, self._label_separator, printer, *args
        )

    def _print_label_latex(self, printer, *args):
        label = [
            printer._print(self.j, *args),
            printer._print(self.m, *args)
        ]
        for i, ji in enumerate(self.jn, start=1):
            label.append('j_{%d}=%s' % (i, printer._print(ji, *args)) )
        for jn, (n1, n2) in zip(self.coupled_jn[:-1], self.coupled_n[:-1]):
            n = ','.join(str(i) for i in sorted(n1 + n2))
            label.append('j_{%s}=%s' % (n, printer._print(jn, *args)) )
        return self._label_separator.join(label)

    @property
    def jn(self):
        return self.label[2]

    @property
    def coupling(self):
        return self.label[3]

    @property
    def coupled_jn(self):
        return _build_coupled(self.label[3], len(self.label[2]))[1]

    @property
    def coupled_n(self):
        return _build_coupled(self.label[3], len(self.label[2]))[0]

    @classmethod
    def _eval_hilbert_space(cls, label):
        j = Add(*label[2])
        if j.is_number:
            return DirectSumHilbertSpace(*[ ComplexSpace(x) for x in range(int(2*j + 1), 0, -2) ])
        else:
            # TODO: Need hilbert space fix, see issue 5732
            # Desired behavior:
            #ji = symbols('ji')
            #ret = Sum(ComplexSpace(2*ji + 1), (ji, 0, j))
            # Temporary fix:
            return ComplexSpace(2*j + 1)

    def _represent_coupled_base(self, **options):
        evect = self.uncoupled_class()
        if not self.j.is_number:
            raise ValueError(
                'State must not have symbolic j value to represent')
        if not self.hilbert_space.dimension.is_number:
            raise ValueError(
                'State must not have symbolic j values to represent')
        result = zeros(self.hilbert_space.dimension, 1)
        if self.j == int(self.j):
            start = self.j**2
        else:
            start = (2*self.j - 1)*(1 + 2*self.j)/4
        result[start:start + 2*self.j + 1, 0] = evect(
            self.j, self.m)._represent_base(**options)
        return result

    def _eval_rewrite_as_Jx(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jx, JxBraCoupled, **options)
        return self._rewrite_basis(Jx, JxKetCoupled, **options)

    def _eval_rewrite_as_Jy(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jy, JyBraCoupled, **options)
        return self._rewrite_basis(Jy, JyKetCoupled, **options)

    def _eval_rewrite_as_Jz(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jz, JzBraCoupled, **options)
        return self._rewrite_basis(Jz, JzKetCoupled, **options)


class JxKetCoupled(CoupledSpinState, Ket):
    """Coupled eigenket of Jx.

    See JzKetCoupled for the usage of coupled spin eigenstates.

    See Also
    ========

    JzKetCoupled: Usage of coupled spin states

    """

    @classmethod
    def dual_class(self):
        return JxBraCoupled

    @classmethod
    def uncoupled_class(self):
        return JxKet

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        return self._represent_coupled_base(**options)

    def _represent_JyOp(self, basis, **options):
        return self._represent_coupled_base(alpha=pi*Rational(3, 2), **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_coupled_base(beta=pi/2, **options)


class JxBraCoupled(CoupledSpinState, Bra):
    """Coupled eigenbra of Jx.

    See JzKetCoupled for the usage of coupled spin eigenstates.

    See Also
    ========

    JzKetCoupled: Usage of coupled spin states

    """

    @classmethod
    def dual_class(self):
        return JxKetCoupled

    @classmethod
    def uncoupled_class(self):
        return JxBra


class JyKetCoupled(CoupledSpinState, Ket):
    """Coupled eigenket of Jy.

    See JzKetCoupled for the usage of coupled spin eigenstates.

    See Also
    ========

    JzKetCoupled: Usage of coupled spin states

    """

    @classmethod
    def dual_class(self):
        return JyBraCoupled

    @classmethod
    def uncoupled_class(self):
        return JyKet

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        return self._represent_coupled_base(gamma=pi/2, **options)

    def _represent_JyOp(self, basis, **options):
        return self._represent_coupled_base(**options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_coupled_base(alpha=pi*Rational(3, 2), beta=-pi/2, gamma=pi/2, **options)


class JyBraCoupled(CoupledSpinState, Bra):
    """Coupled eigenbra of Jy.

    See JzKetCoupled for the usage of coupled spin eigenstates.

    See Also
    ========

    JzKetCoupled: Usage of coupled spin states

    """

    @classmethod
    def dual_class(self):
        return JyKetCoupled

    @classmethod
    def uncoupled_class(self):
        return JyBra


class JzKetCoupled(CoupledSpinState, Ket):
    r"""Coupled eigenket of Jz

    Spin state that is an eigenket of Jz which represents the coupling of
    separate spin spaces.

    The arguments for creating instances of JzKetCoupled are ``j``, ``m``,
    ``jn`` and an optional ``jcoupling`` argument. The ``j`` and ``m`` options
    are the total angular momentum quantum numbers, as used for normal states
    (e.g. JzKet).

    The other required parameter in ``jn``, which is a tuple defining the `j_n`
    angular momentum quantum numbers of the product spaces. So for example, if
    a state represented the coupling of the product basis state
    `\left|j_1,m_1\right\rangle\times\left|j_2,m_2\right\rangle`, the ``jn``
    for this state would be ``(j1,j2)``.

    The final option is ``jcoupling``, which is used to define how the spaces
    specified by ``jn`` are coupled, which includes both the order these spaces
    are coupled together and the quantum numbers that arise from these
    couplings. The ``jcoupling`` parameter itself is a list of lists, such that
    each of the sublists defines a single coupling between the spin spaces. If
    there are N coupled angular momentum spaces, that is ``jn`` has N elements,
    then there must be N-1 sublists. Each of these sublists making up the
    ``jcoupling`` parameter have length 3. The first two elements are the
    indices of the product spaces that are considered to be coupled together.
    For example, if we want to couple `j_1` and `j_4`, the indices would be 1
    and 4. If a state has already been coupled, it is referenced by the
    smallest index that is coupled, so if `j_2` and `j_4` has already been
    coupled to some `j_{24}`, then this value can be coupled by referencing it
    with index 2. The final element of the sublist is the quantum number of the
    coupled state. So putting everything together, into a valid sublist for
    ``jcoupling``, if `j_1` and `j_2` are coupled to an angular momentum space
    with quantum number `j_{12}` with the value ``j12``, the sublist would be
    ``(1,2,j12)``, N-1 of these sublists are used in the list for
    ``jcoupling``.

    Note the ``jcoupling`` parameter is optional, if it is not specified, the
    default coupling is taken. This default value is to coupled the spaces in
    order and take the quantum number of the coupling to be the maximum value.
    For example, if the spin spaces are `j_1`, `j_2`, `j_3`, `j_4`, then the
    default coupling couples `j_1` and `j_2` to `j_{12}=j_1+j_2`, then,
    `j_{12}` and `j_3` are coupled to `j_{123}=j_{12}+j_3`, and finally
    `j_{123}` and `j_4` to `j=j_{123}+j_4`. The jcoupling value that would
    correspond to this is:

        ``((1,2,j1+j2),(1,3,j1+j2+j3))``

    Parameters
    ==========

    args : tuple
        The arguments that must be passed are ``j``, ``m``, ``jn``, and
        ``jcoupling``. The ``j`` value is the total angular momentum. The ``m``
        value is the eigenvalue of the Jz spin operator. The ``jn`` list are
        the j values of argular momentum spaces coupled together. The
        ``jcoupling`` parameter is an optional parameter defining how the spaces
        are coupled together. See the above description for how these coupling
        parameters are defined.

    Examples
    ========

    Defining simple spin states, both numerical and symbolic:

        >>> from sympy.physics.quantum.spin import JzKetCoupled
        >>> from sympy import symbols
        >>> JzKetCoupled(1, 0, (1, 1))
        |1,0,j1=1,j2=1>
        >>> j, m, j1, j2 = symbols('j m j1 j2')
        >>> JzKetCoupled(j, m, (j1, j2))
        |j,m,j1=j1,j2=j2>

    Defining coupled spin states for more than 2 coupled spaces with various
    coupling parameters:

        >>> JzKetCoupled(2, 1, (1, 1, 1))
        |2,1,j1=1,j2=1,j3=1,j(1,2)=2>
        >>> JzKetCoupled(2, 1, (1, 1, 1), ((1,2,2),(1,3,2)) )
        |2,1,j1=1,j2=1,j3=1,j(1,2)=2>
        >>> JzKetCoupled(2, 1, (1, 1, 1), ((2,3,1),(1,2,2)) )
        |2,1,j1=1,j2=1,j3=1,j(2,3)=1>

    Rewriting the JzKetCoupled in terms of eigenkets of the Jx operator:
    Note: that the resulting eigenstates are JxKetCoupled

        >>> JzKetCoupled(1,1,(1,1)).rewrite("Jx")
        |1,-1,j1=1,j2=1>/2 - sqrt(2)*|1,0,j1=1,j2=1>/2 + |1,1,j1=1,j2=1>/2

    The rewrite method can be used to convert a coupled state to an uncoupled
    state. This is done by passing coupled=False to the rewrite function:

        >>> JzKetCoupled(1, 0, (1, 1)).rewrite('Jz', coupled=False)
        -sqrt(2)*|1,-1>x|1,1>/2 + sqrt(2)*|1,1>x|1,-1>/2

    Get the vector representation of a state in terms of the basis elements
    of the Jx operator:

        >>> from sympy.physics.quantum.represent import represent
        >>> from sympy.physics.quantum.spin import Jx
        >>> from sympy import S
        >>> represent(JzKetCoupled(1,-1,(S(1)/2,S(1)/2)), basis=Jx)
        Matrix([
        [        0],
        [      1/2],
        [sqrt(2)/2],
        [      1/2]])

    See Also
    ========

    JzKet: Normal spin eigenstates
    uncouple: Uncoupling of coupling spin states
    couple: Coupling of uncoupled spin states

    """

    @classmethod
    def dual_class(self):
        return JzBraCoupled

    @classmethod
    def uncoupled_class(self):
        return JzKet

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        return self._represent_coupled_base(beta=pi*Rational(3, 2), **options)

    def _represent_JyOp(self, basis, **options):
        return self._represent_coupled_base(alpha=pi*Rational(3, 2), beta=pi/2, gamma=pi/2, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_coupled_base(**options)


class JzBraCoupled(CoupledSpinState, Bra):
    """Coupled eigenbra of Jz.

    See the JzKetCoupled for the usage of coupled spin eigenstates.

    See Also
    ========

    JzKetCoupled: Usage of coupled spin states

    """

    @classmethod
    def dual_class(self):
        return JzKetCoupled

    @classmethod
    def uncoupled_class(self):
        return JzBra

#-----------------------------------------------------------------------------
# Coupling/uncoupling
#-----------------------------------------------------------------------------


def couple(expr, jcoupling_list=None):
    """ Couple a tensor product of spin states

    This function can be used to couple an uncoupled tensor product of spin
    states. All of the eigenstates to be coupled must be of the same class. It
    will return a linear combination of eigenstates that are subclasses of
    CoupledSpinState determined by Clebsch-Gordan angular momentum coupling
    coefficients.

    Parameters
    ==========

    expr : Expr
        An expression involving TensorProducts of spin states to be coupled.
        Each state must be a subclass of SpinState and they all must be the
        same class.

    jcoupling_list : list or tuple
        Elements of this list are sub-lists of length 2 specifying the order of
        the coupling of the spin spaces. The length of this must be N-1, where N
        is the number of states in the tensor product to be coupled. The
        elements of this sublist are the same as the first two elements of each
        sublist in the ``jcoupling`` parameter defined for JzKetCoupled. If this
        parameter is not specified, the default value is taken, which couples
        the first and second product basis spaces, then couples this new coupled
        space to the third product space, etc

    Examples
    ========

    Couple a tensor product of numerical states for two spaces:

        >>> from sympy.physics.quantum.spin import JzKet, couple
        >>> from sympy.physics.quantum.tensorproduct import TensorProduct
        >>> couple(TensorProduct(JzKet(1,0), JzKet(1,1)))
        -sqrt(2)*|1,1,j1=1,j2=1>/2 + sqrt(2)*|2,1,j1=1,j2=1>/2


    Numerical coupling of three spaces using the default coupling method, i.e.
    first and second spaces couple, then this couples to the third space:

        >>> couple(TensorProduct(JzKet(1,1), JzKet(1,1), JzKet(1,0)))
        sqrt(6)*|2,2,j1=1,j2=1,j3=1,j(1,2)=2>/3 + sqrt(3)*|3,2,j1=1,j2=1,j3=1,j(1,2)=2>/3

    Perform this same coupling, but we define the coupling to first couple
    the first and third spaces:

        >>> couple(TensorProduct(JzKet(1,1), JzKet(1,1), JzKet(1,0)), ((1,3),(1,2)) )
        sqrt(2)*|2,2,j1=1,j2=1,j3=1,j(1,3)=1>/2 - sqrt(6)*|2,2,j1=1,j2=1,j3=1,j(1,3)=2>/6 + sqrt(3)*|3,2,j1=1,j2=1,j3=1,j(1,3)=2>/3

    Couple a tensor product of symbolic states:

        >>> from sympy import symbols
        >>> j1,m1,j2,m2 = symbols('j1 m1 j2 m2')
        >>> couple(TensorProduct(JzKet(j1,m1), JzKet(j2,m2)))
        Sum(CG(j1, m1, j2, m2, j, m1 + m2)*|j,m1 + m2,j1=j1,j2=j2>, (j, m1 + m2, j1 + j2))

    """
    a = expr.atoms(TensorProduct)
    for tp in a:
        # Allow other tensor products to be in expression
        if not all(isinstance(state, SpinState) for state in tp.args):
            continue
        # If tensor product has all spin states, raise error for invalid tensor product state
        if not all(state.__class__ is tp.args[0].__class__ for state in tp.args):
            raise TypeError('All states must be the same basis')
        expr = expr.subs(tp, _couple(tp, jcoupling_list))
    return expr


def _couple(tp, jcoupling_list):
    states = tp.args
    coupled_evect = states[0].coupled_class()

    # Define default coupling if none is specified
    if jcoupling_list is None:
        jcoupling_list = []
        for n in range(1, len(states)):
            jcoupling_list.append( (1, n + 1) )

    # Check jcoupling_list valid
    if not len(jcoupling_list) == len(states) - 1:
        raise TypeError('jcoupling_list must be length %d, got %d' %
                        (len(states) - 1, len(jcoupling_list)))
    if not all( len(coupling) == 2 for coupling in jcoupling_list):
        raise ValueError('Each coupling must define 2 spaces')
    if any(n1 == n2 for n1, n2 in jcoupling_list):
        raise ValueError('Spin spaces cannot couple to themselves')
    if all(sympify(n1).is_number and sympify(n2).is_number for n1, n2 in jcoupling_list):
        j_test = [0]*len(states)
        for n1, n2 in jcoupling_list:
            if j_test[n1 - 1] == -1 or j_test[n2 - 1] == -1:
                raise ValueError('Spaces coupling j_n\'s are referenced by smallest n value')
            j_test[max(n1, n2) - 1] = -1

    # j values of states to be coupled together
    jn = [state.j for state in states]
    mn = [state.m for state in states]

    # Create coupling_list, which defines all the couplings between all
    # the spaces from jcoupling_list
    coupling_list = []
    n_list = [ [i + 1] for i in range(len(states)) ]
    for j_coupling in jcoupling_list:
        # Least n for all j_n which is coupled as first and second spaces
        n1, n2 = j_coupling
        # List of all n's coupled in first and second spaces
        j1_n = list(n_list[n1 - 1])
        j2_n = list(n_list[n2 - 1])
        coupling_list.append( (j1_n, j2_n) )
        # Set new j_n to be coupling of all j_n in both first and second spaces
        n_list[ min(n1, n2) - 1 ] = sorted(j1_n + j2_n)

    if all(state.j.is_number and state.m.is_number for state in states):
        # Numerical coupling
        # Iterate over difference between maximum possible j value of each coupling and the actual value
        diff_max = [ Add( *[ jn[n - 1] - mn[n - 1] for n in coupling[0] +
                         coupling[1] ] ) for coupling in coupling_list ]
        result = []
        for diff in range(diff_max[-1] + 1):
            # Determine available configurations
            n = len(coupling_list)
            tot = binomial(diff + n - 1, diff)

            for config_num in range(tot):
                diff_list = _confignum_to_difflist(config_num, diff, n)

                # Skip the configuration if non-physical
                # This is a lazy check for physical states given the loose restrictions of diff_max
                if any(d > m for d, m in zip(diff_list, diff_max)):
                    continue

                # Determine term
                cg_terms = []
                coupled_j = list(jn)
                jcoupling = []
                for (j1_n, j2_n), coupling_diff in zip(coupling_list, diff_list):
                    j1 = coupled_j[ min(j1_n) - 1 ]
                    j2 = coupled_j[ min(j2_n) - 1 ]
                    j3 = j1 + j2 - coupling_diff
                    coupled_j[ min(j1_n + j2_n) - 1 ] = j3
                    m1 = Add( *[ mn[x - 1] for x in j1_n] )
                    m2 = Add( *[ mn[x - 1] for x in j2_n] )
                    m3 = m1 + m2
                    cg_terms.append( (j1, m1, j2, m2, j3, m3) )
                    jcoupling.append( (min(j1_n), min(j2_n), j3) )
                # Better checks that state is physical
                if any(abs(term[5]) > term[4] for term in cg_terms):
                    continue
                if any(term[0] + term[2] < term[4] for term in cg_terms):
                    continue
                if any(abs(term[0] - term[2]) > term[4] for term in cg_terms):
                    continue
                coeff = Mul( *[ CG(*term).doit() for term in cg_terms] )
                state = coupled_evect(j3, m3, jn, jcoupling)
                result.append(coeff*state)
        return Add(*result)
    else:
        # Symbolic coupling
        cg_terms = []
        jcoupling = []
        sum_terms = []
        coupled_j = list(jn)
        for j1_n, j2_n in coupling_list:
            j1 = coupled_j[ min(j1_n) - 1 ]
            j2 = coupled_j[ min(j2_n) - 1 ]
            if len(j1_n + j2_n) == len(states):
                j3 = symbols('j')
            else:
                j3_name = 'j' + ''.join(["%s" % n for n in j1_n + j2_n])
                j3 = symbols(j3_name)
            coupled_j[ min(j1_n + j2_n) - 1 ] = j3
            m1 = Add( *[ mn[x - 1] for x in j1_n] )
            m2 = Add( *[ mn[x - 1] for x in j2_n] )
            m3 = m1 + m2
            cg_terms.append( (j1, m1, j2, m2, j3, m3) )
            jcoupling.append( (min(j1_n), min(j2_n), j3) )
            sum_terms.append((j3, m3, j1 + j2))
        coeff = Mul( *[ CG(*term) for term in cg_terms] )
        state = coupled_evect(j3, m3, jn, jcoupling)
        return Sum(coeff*state, *sum_terms)


def uncouple(expr, jn=None, jcoupling_list=None):
    """ Uncouple a coupled spin state

    Gives the uncoupled representation of a coupled spin state. Arguments must
    be either a spin state that is a subclass of CoupledSpinState or a spin
    state that is a subclass of SpinState and an array giving the j values
    of the spaces that are to be coupled

    Parameters
    ==========

    expr : Expr
        The expression containing states that are to be coupled. If the states
        are a subclass of SpinState, the ``jn`` and ``jcoupling`` parameters
        must be defined. If the states are a subclass of CoupledSpinState,
        ``jn`` and ``jcoupling`` will be taken from the state.

    jn : list or tuple
        The list of the j-values that are coupled. If state is a
        CoupledSpinState, this parameter is ignored. This must be defined if
        state is not a subclass of CoupledSpinState. The syntax of this
        parameter is the same as the ``jn`` parameter of JzKetCoupled.

    jcoupling_list : list or tuple
        The list defining how the j-values are coupled together. If state is a
        CoupledSpinState, this parameter is ignored. This must be defined if
        state is not a subclass of CoupledSpinState. The syntax of this
        parameter is the same as the ``jcoupling`` parameter of JzKetCoupled.

    Examples
    ========

    Uncouple a numerical state using a CoupledSpinState state:

        >>> from sympy.physics.quantum.spin import JzKetCoupled, uncouple
        >>> from sympy import S
        >>> uncouple(JzKetCoupled(1, 0, (S(1)/2, S(1)/2)))
        sqrt(2)*|1/2,-1/2>x|1/2,1/2>/2 + sqrt(2)*|1/2,1/2>x|1/2,-1/2>/2

    Perform the same calculation using a SpinState state:

        >>> from sympy.physics.quantum.spin import JzKet
        >>> uncouple(JzKet(1, 0), (S(1)/2, S(1)/2))
        sqrt(2)*|1/2,-1/2>x|1/2,1/2>/2 + sqrt(2)*|1/2,1/2>x|1/2,-1/2>/2

    Uncouple a numerical state of three coupled spaces using a CoupledSpinState state:

        >>> uncouple(JzKetCoupled(1, 1, (1, 1, 1), ((1,3,1),(1,2,1)) ))
        |1,-1>x|1,1>x|1,1>/2 - |1,0>x|1,0>x|1,1>/2 + |1,1>x|1,0>x|1,0>/2 - |1,1>x|1,1>x|1,-1>/2

    Perform the same calculation using a SpinState state:

        >>> uncouple(JzKet(1, 1), (1, 1, 1), ((1,3,1),(1,2,1)) )
        |1,-1>x|1,1>x|1,1>/2 - |1,0>x|1,0>x|1,1>/2 + |1,1>x|1,0>x|1,0>/2 - |1,1>x|1,1>x|1,-1>/2

    Uncouple a symbolic state using a CoupledSpinState state:

        >>> from sympy import symbols
        >>> j,m,j1,j2 = symbols('j m j1 j2')
        >>> uncouple(JzKetCoupled(j, m, (j1, j2)))
        Sum(CG(j1, m1, j2, m2, j, m)*|j1,m1>x|j2,m2>, (m1, -j1, j1), (m2, -j2, j2))

    Perform the same calculation using a SpinState state

        >>> uncouple(JzKet(j, m), (j1, j2))
        Sum(CG(j1, m1, j2, m2, j, m)*|j1,m1>x|j2,m2>, (m1, -j1, j1), (m2, -j2, j2))

    """
    a = expr.atoms(SpinState)
    for state in a:
        expr = expr.subs(state, _uncouple(state, jn, jcoupling_list))
    return expr


def _uncouple(state, jn, jcoupling_list):
    if isinstance(state, CoupledSpinState):
        jn = state.jn
        coupled_n = state.coupled_n
        coupled_jn = state.coupled_jn
        evect = state.uncoupled_class()
    elif isinstance(state, SpinState):
        if jn is None:
            raise ValueError("Must specify j-values for coupled state")
        if not isinstance(jn, (list, tuple)):
            raise TypeError("jn must be list or tuple")
        if jcoupling_list is None:
            # Use default
            jcoupling_list = []
            for i in range(1, len(jn)):
                jcoupling_list.append(
                    (1, 1 + i, Add(*[jn[j] for j in range(i + 1)])) )
        if not isinstance(jcoupling_list, (list, tuple)):
            raise TypeError("jcoupling must be a list or tuple")
        if not len(jcoupling_list) == len(jn) - 1:
            raise ValueError("Must specify 2 fewer coupling terms than the number of j values")
        coupled_n, coupled_jn = _build_coupled(jcoupling_list, len(jn))
        evect = state.__class__
    else:
        raise TypeError("state must be a spin state")
    j = state.j
    m = state.m
    coupling_list = []
    j_list = list(jn)

    # Create coupling, which defines all the couplings between all the spaces
    for j3, (n1, n2) in zip(coupled_jn, coupled_n):
        # j's which are coupled as first and second spaces
        j1 = j_list[n1[0] - 1]
        j2 = j_list[n2[0] - 1]
        # Build coupling list
        coupling_list.append( (n1, n2, j1, j2, j3) )
        # Set new value in j_list
        j_list[min(n1 + n2) - 1] = j3

    if j.is_number and m.is_number:
        diff_max = [ 2*x for x in jn ]
        diff = Add(*jn) - m

        n = len(jn)
        tot = binomial(diff + n - 1, diff)

        result = []
        for config_num in range(tot):
            diff_list = _confignum_to_difflist(config_num, diff, n)
            if any(d > p for d, p in zip(diff_list, diff_max)):
                continue

            cg_terms = []
            for coupling in coupling_list:
                j1_n, j2_n, j1, j2, j3 = coupling
                m1 = Add( *[ jn[x - 1] - diff_list[x - 1] for x in j1_n ] )
                m2 = Add( *[ jn[x - 1] - diff_list[x - 1] for x in j2_n ] )
                m3 = m1 + m2
                cg_terms.append( (j1, m1, j2, m2, j3, m3) )
            coeff = Mul( *[ CG(*term).doit() for term in cg_terms ] )
            state = TensorProduct(
                *[ evect(j, j - d) for j, d in zip(jn, diff_list) ] )
            result.append(coeff*state)
        return Add(*result)
    else:
        # Symbolic coupling
        m_str = "m1:%d" % (len(jn) + 1)
        mvals = symbols(m_str)
        cg_terms = [(j1, Add(*[mvals[n - 1] for n in j1_n]),
                     j2, Add(*[mvals[n - 1] for n in j2_n]),
                     j3, Add(*[mvals[n - 1] for n in j1_n + j2_n])) for j1_n, j2_n, j1, j2, j3 in coupling_list[:-1] ]
        cg_terms.append(*[(j1, Add(*[mvals[n - 1] for n in j1_n]),
                           j2, Add(*[mvals[n - 1] for n in j2_n]),
                           j, m) for j1_n, j2_n, j1, j2, j3 in [coupling_list[-1]] ])
        cg_coeff = Mul(*[CG(*cg_term) for cg_term in cg_terms])
        sum_terms = [ (m, -j, j) for j, m in zip(jn, mvals) ]
        state = TensorProduct( *[ evect(j, m) for j, m in zip(jn, mvals) ] )
        return Sum(cg_coeff*state, *sum_terms)


def _confignum_to_difflist(config_num, diff, list_len):
    # Determines configuration of diffs into list_len number of slots
    diff_list = []
    for n in range(list_len):
        prev_diff = diff
        # Number of spots after current one
        rem_spots = list_len - n - 1
        # Number of configurations of distributing diff among the remaining spots
        rem_configs = binomial(diff + rem_spots - 1, diff)
        while config_num >= rem_configs:
            config_num -= rem_configs
            diff -= 1
            rem_configs = binomial(diff + rem_spots - 1, diff)
        diff_list.append(prev_diff - diff)
    return diff_list
