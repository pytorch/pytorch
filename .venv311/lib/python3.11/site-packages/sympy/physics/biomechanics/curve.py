"""Implementations of characteristic curves for musculotendon models."""

from dataclasses import dataclass

from sympy.core.expr import UnevaluatedExpr
from sympy.core.function import ArgumentIndexError, Function
from sympy.core.numbers import Float, Integer
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import cosh, sinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.printing.precedence import PRECEDENCE


__all__ = [
    'CharacteristicCurveCollection',
    'CharacteristicCurveFunction',
    'FiberForceLengthActiveDeGroote2016',
    'FiberForceLengthPassiveDeGroote2016',
    'FiberForceLengthPassiveInverseDeGroote2016',
    'FiberForceVelocityDeGroote2016',
    'FiberForceVelocityInverseDeGroote2016',
    'TendonForceLengthDeGroote2016',
    'TendonForceLengthInverseDeGroote2016',
]


class CharacteristicCurveFunction(Function):
    """Base class for all musculotendon characteristic curve functions."""

    @classmethod
    def eval(cls):
        msg = (
            f'Cannot directly instantiate {cls.__name__!r}, instances of '
            f'characteristic curves must be of a concrete subclass.'

        )
        raise TypeError(msg)

    def _print_code(self, printer):
        """Print code for the function defining the curve using a printer.

        Explanation
        ===========

        The order of operations may need to be controlled as constant folding
        the numeric terms within the equations of a musculotendon
        characteristic curve can sometimes results in a numerically-unstable
        expression.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print a string representation of the
            characteristic curve as valid code in the target language.

        """
        return printer._print(printer.parenthesize(
            self.doit(deep=False, evaluate=False), PRECEDENCE['Atom'],
        ))

    _ccode = _print_code
    _cupycode = _print_code
    _cxxcode = _print_code
    _fcode = _print_code
    _jaxcode = _print_code
    _lambdacode = _print_code
    _mpmathcode = _print_code
    _octave = _print_code
    _pythoncode = _print_code
    _numpycode = _print_code
    _scipycode = _print_code


class TendonForceLengthDeGroote2016(CharacteristicCurveFunction):
    r"""Tendon force-length curve based on De Groote et al., 2016 [1]_.

    Explanation
    ===========

    Gives the normalized tendon force produced as a function of normalized
    tendon length.

    The function is defined by the equation:

    $fl^T = c_0 \exp{c_3 \left( \tilde{l}^T - c_1 \right)} - c_2$

    with constant values of $c_0 = 0.2$, $c_1 = 0.995$, $c_2 = 0.25$, and
    $c_3 = 33.93669377311689$.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces no
    force when the tendon is in an unstrained state. It also produces a force
    of 1 normalized unit when the tendon is under a 5% strain.

    Examples
    ========

    The preferred way to instantiate :class:`TendonForceLengthDeGroote2016` is using
    the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized tendon length. We'll create a
    :class:`~.Symbol` called ``l_T_tilde`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import TendonForceLengthDeGroote2016
    >>> l_T_tilde = Symbol('l_T_tilde')
    >>> fl_T = TendonForceLengthDeGroote2016.with_defaults(l_T_tilde)
    >>> fl_T
    TendonForceLengthDeGroote2016(l_T_tilde, 0.2, 0.995, 0.25,
    33.93669377311689)

    It's also possible to populate the four constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3 = symbols('c0 c1 c2 c3')
    >>> fl_T = TendonForceLengthDeGroote2016(l_T_tilde, c0, c1, c2, c3)
    >>> fl_T
    TendonForceLengthDeGroote2016(l_T_tilde, c0, c1, c2, c3)

    You don't just have to use symbols as the arguments, it's also possible to
    use expressions. Let's create a new pair of symbols, ``l_T`` and
    ``l_T_slack``, representing tendon length and tendon slack length
    respectively. We can then represent ``l_T_tilde`` as an expression, the
    ratio of these.

    >>> l_T, l_T_slack = symbols('l_T l_T_slack')
    >>> l_T_tilde = l_T/l_T_slack
    >>> fl_T = TendonForceLengthDeGroote2016.with_defaults(l_T_tilde)
    >>> fl_T
    TendonForceLengthDeGroote2016(l_T/l_T_slack, 0.2, 0.995, 0.25,
    33.93669377311689)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> fl_T.doit(evaluate=False)
    -0.25 + 0.2*exp(33.93669377311689*(l_T/l_T_slack - 0.995))

    The function can also be differentiated. We'll differentiate with respect
    to l_T using the ``diff`` method on an instance with the single positional
    argument ``l_T``.

    >>> fl_T.diff(l_T)
    6.787338754623378*exp(33.93669377311689*(l_T/l_T_slack - 0.995))/l_T_slack

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod
    def with_defaults(cls, l_T_tilde):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the tendon force-length function using the
        four constant values specified in the original publication.

        These have the values:

        $c_0 = 0.2$
        $c_1 = 0.995$
        $c_2 = 0.25$
        $c_3 = 33.93669377311689$

        Parameters
        ==========

        l_T_tilde : Any (sympifiable)
            Normalized tendon length.

        """
        c0 = Float('0.2')
        c1 = Float('0.995')
        c2 = Float('0.25')
        c3 = Float('33.93669377311689')
        return cls(l_T_tilde, c0, c1, c2, c3)

    @classmethod
    def eval(cls, l_T_tilde, c0, c1, c2, c3):
        """Evaluation of basic inputs.

        Parameters
        ==========

        l_T_tilde : Any (sympifiable)
            Normalized tendon length.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.2``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``0.995``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``0.25``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``33.93669377311689``.

        """
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``l_T_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        l_T_tilde, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            l_T_tilde = l_T_tilde.doit(deep=deep, **hints)
            c0, c1, c2, c3 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1, c2, c3 = constants

        if evaluate:
            return c0*exp(c3*(l_T_tilde - c1)) - c2

        return c0*exp(c3*UnevaluatedExpr(l_T_tilde - c1)) - c2

    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
        l_T_tilde, c0, c1, c2, c3 = self.args
        if argindex == 1:
            return c0*c3*exp(c3*UnevaluatedExpr(l_T_tilde - c1))
        elif argindex == 2:
            return exp(c3*UnevaluatedExpr(l_T_tilde - c1))
        elif argindex == 3:
            return -c0*c3*exp(c3*UnevaluatedExpr(l_T_tilde - c1))
        elif argindex == 4:
            return Integer(-1)
        elif argindex == 5:
            return c0*(l_T_tilde - c1)*exp(c3*UnevaluatedExpr(l_T_tilde - c1))

        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
        return TendonForceLengthInverseDeGroote2016

    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        l_T_tilde = self.args[0]
        _l_T_tilde = printer._print(l_T_tilde)
        return r'\operatorname{fl}^T \left( %s \right)' % _l_T_tilde


class TendonForceLengthInverseDeGroote2016(CharacteristicCurveFunction):
    r"""Inverse tendon force-length curve based on De Groote et al., 2016 [1]_.

    Explanation
    ===========

    Gives the normalized tendon length that produces a specific normalized
    tendon force.

    The function is defined by the equation:

    ${fl^T}^{-1} = frac{\log{\frac{fl^T + c_2}{c_0}}}{c_3} + c_1$

    with constant values of $c_0 = 0.2$, $c_1 = 0.995$, $c_2 = 0.25$, and
    $c_3 = 33.93669377311689$. This function is the exact analytical inverse
    of the related tendon force-length curve ``TendonForceLengthDeGroote2016``.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces no
    force when the tendon is in an unstrained state. It also produces a force
    of 1 normalized unit when the tendon is under a 5% strain.

    Examples
    ========

    The preferred way to instantiate :class:`TendonForceLengthInverseDeGroote2016` is
    using the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized tendon force-length, which is
    equal to the tendon force. We'll create a :class:`~.Symbol` called ``fl_T`` to
    represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import TendonForceLengthInverseDeGroote2016
    >>> fl_T = Symbol('fl_T')
    >>> l_T_tilde = TendonForceLengthInverseDeGroote2016.with_defaults(fl_T)
    >>> l_T_tilde
    TendonForceLengthInverseDeGroote2016(fl_T, 0.2, 0.995, 0.25,
    33.93669377311689)

    It's also possible to populate the four constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3 = symbols('c0 c1 c2 c3')
    >>> l_T_tilde = TendonForceLengthInverseDeGroote2016(fl_T, c0, c1, c2, c3)
    >>> l_T_tilde
    TendonForceLengthInverseDeGroote2016(fl_T, c0, c1, c2, c3)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> l_T_tilde.doit(evaluate=False)
    c1 + log((c2 + fl_T)/c0)/c3

    The function can also be differentiated. We'll differentiate with respect
    to l_T using the ``diff`` method on an instance with the single positional
    argument ``l_T``.

    >>> l_T_tilde.diff(fl_T)
    1/(c3*(c2 + fl_T))

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod
    def with_defaults(cls, fl_T):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the inverse tendon force-length function
        using the four constant values specified in the original publication.

        These have the values:

        $c_0 = 0.2$
        $c_1 = 0.995$
        $c_2 = 0.25$
        $c_3 = 33.93669377311689$

        Parameters
        ==========

        fl_T : Any (sympifiable)
            Normalized tendon force as a function of tendon length.

        """
        c0 = Float('0.2')
        c1 = Float('0.995')
        c2 = Float('0.25')
        c3 = Float('33.93669377311689')
        return cls(fl_T, c0, c1, c2, c3)

    @classmethod
    def eval(cls, fl_T, c0, c1, c2, c3):
        """Evaluation of basic inputs.

        Parameters
        ==========

        fl_T : Any (sympifiable)
            Normalized tendon force as a function of tendon length.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.2``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``0.995``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``0.25``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``33.93669377311689``.

        """
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``l_T_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        fl_T, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            fl_T = fl_T.doit(deep=deep, **hints)
            c0, c1, c2, c3 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1, c2, c3 = constants

        if evaluate:
            return log((fl_T + c2)/c0)/c3 + c1

        return log(UnevaluatedExpr((fl_T + c2)/c0))/c3 + c1

    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
        fl_T, c0, c1, c2, c3 = self.args
        if argindex == 1:
            return 1/(c3*(fl_T + c2))
        elif argindex == 2:
            return -1/(c0*c3)
        elif argindex == 3:
            return Integer(1)
        elif argindex == 4:
            return 1/(c3*(fl_T + c2))
        elif argindex == 5:
            return -log(UnevaluatedExpr((fl_T + c2)/c0))/c3**2

        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
        return TendonForceLengthDeGroote2016

    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        fl_T = self.args[0]
        _fl_T = printer._print(fl_T)
        return r'\left( \operatorname{fl}^T \right)^{-1} \left( %s \right)' % _fl_T


class FiberForceLengthPassiveDeGroote2016(CharacteristicCurveFunction):
    r"""Passive muscle fiber force-length curve based on De Groote et al., 2016
    [1]_.

    Explanation
    ===========

    The function is defined by the equation:

    $fl^M_{pas} = \frac{\frac{\exp{c_1 \left(\tilde{l^M} - 1\right)}}{c_0} - 1}{\exp{c_1} - 1}$

    with constant values of $c_0 = 0.6$ and $c_1 = 4.0$.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    passive fiber force very close to 0 for all normalized fiber lengths
    between 0 and 1.

    Examples
    ========

    The preferred way to instantiate :class:`FiberForceLengthPassiveDeGroote2016` is
    using the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized muscle fiber length. We'll
    create a :class:`~.Symbol` called ``l_M_tilde`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceLengthPassiveDeGroote2016
    >>> l_M_tilde = Symbol('l_M_tilde')
    >>> fl_M = FiberForceLengthPassiveDeGroote2016.with_defaults(l_M_tilde)
    >>> fl_M
    FiberForceLengthPassiveDeGroote2016(l_M_tilde, 0.6, 4.0)

    It's also possible to populate the two constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1 = symbols('c0 c1')
    >>> fl_M = FiberForceLengthPassiveDeGroote2016(l_M_tilde, c0, c1)
    >>> fl_M
    FiberForceLengthPassiveDeGroote2016(l_M_tilde, c0, c1)

    You don't just have to use symbols as the arguments, it's also possible to
    use expressions. Let's create a new pair of symbols, ``l_M`` and
    ``l_M_opt``, representing muscle fiber length and optimal muscle fiber
    length respectively. We can then represent ``l_M_tilde`` as an expression,
    the ratio of these.

    >>> l_M, l_M_opt = symbols('l_M l_M_opt')
    >>> l_M_tilde = l_M/l_M_opt
    >>> fl_M = FiberForceLengthPassiveDeGroote2016.with_defaults(l_M_tilde)
    >>> fl_M
    FiberForceLengthPassiveDeGroote2016(l_M/l_M_opt, 0.6, 4.0)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> fl_M.doit(evaluate=False)
    0.0186573603637741*(-1 + exp(6.66666666666667*(l_M/l_M_opt - 1)))

    The function can also be differentiated. We'll differentiate with respect
    to l_M using the ``diff`` method on an instance with the single positional
    argument ``l_M``.

    >>> fl_M.diff(l_M)
    0.12438240242516*exp(6.66666666666667*(l_M/l_M_opt - 1))/l_M_opt

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod
    def with_defaults(cls, l_M_tilde):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the muscle fiber passive force-length
        function using the four constant values specified in the original
        publication.

        These have the values:

        $c_0 = 0.6$
        $c_1 = 4.0$

        Parameters
        ==========

        l_M_tilde : Any (sympifiable)
            Normalized muscle fiber length.

        """
        c0 = Float('0.6')
        c1 = Float('4.0')
        return cls(l_M_tilde, c0, c1)

    @classmethod
    def eval(cls, l_M_tilde, c0, c1):
        """Evaluation of basic inputs.

        Parameters
        ==========

        l_M_tilde : Any (sympifiable)
            Normalized muscle fiber length.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.6``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``4.0``.

        """
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``l_T_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        l_M_tilde, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            l_M_tilde = l_M_tilde.doit(deep=deep, **hints)
            c0, c1 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1 = constants

        if evaluate:
            return (exp((c1*(l_M_tilde - 1))/c0) - 1)/(exp(c1) - 1)

        return (exp((c1*UnevaluatedExpr(l_M_tilde - 1))/c0) - 1)/(exp(c1) - 1)

    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
        l_M_tilde, c0, c1 = self.args
        if argindex == 1:
            return c1*exp(c1*UnevaluatedExpr(l_M_tilde - 1)/c0)/(c0*(exp(c1) - 1))
        elif argindex == 2:
            return (
                -c1*exp(c1*UnevaluatedExpr(l_M_tilde - 1)/c0)
                *UnevaluatedExpr(l_M_tilde - 1)/(c0**2*(exp(c1) - 1))
            )
        elif argindex == 3:
            return (
                -exp(c1)*(-1 + exp(c1*UnevaluatedExpr(l_M_tilde - 1)/c0))/(exp(c1) - 1)**2
                + exp(c1*UnevaluatedExpr(l_M_tilde - 1)/c0)*(l_M_tilde - 1)/(c0*(exp(c1) - 1))
            )

        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
        return FiberForceLengthPassiveInverseDeGroote2016

    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        l_M_tilde = self.args[0]
        _l_M_tilde = printer._print(l_M_tilde)
        return r'\operatorname{fl}^M_{pas} \left( %s \right)' % _l_M_tilde


class FiberForceLengthPassiveInverseDeGroote2016(CharacteristicCurveFunction):
    r"""Inverse passive muscle fiber force-length curve based on De Groote et
    al., 2016 [1]_.

    Explanation
    ===========

    Gives the normalized muscle fiber length that produces a specific normalized
    passive muscle fiber force.

    The function is defined by the equation:

    ${fl^M_{pas}}^{-1} = \frac{c_0 \log{\left(\exp{c_1} - 1\right)fl^M_pas + 1}}{c_1} + 1$

    with constant values of $c_0 = 0.6$ and $c_1 = 4.0$. This function is the
    exact analytical inverse of the related tendon force-length curve
    ``FiberForceLengthPassiveDeGroote2016``.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    passive fiber force very close to 0 for all normalized fiber lengths
    between 0 and 1.

    Examples
    ========

    The preferred way to instantiate
    :class:`FiberForceLengthPassiveInverseDeGroote2016` is using the
    :meth:`~.with_defaults` constructor because this will automatically populate the
    constants within the characteristic curve equation with the floating point
    values from the original publication. This constructor takes a single
    argument corresponding to the normalized passive muscle fiber length-force
    component of the muscle fiber force. We'll create a :class:`~.Symbol` called
    ``fl_M_pas`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceLengthPassiveInverseDeGroote2016
    >>> fl_M_pas = Symbol('fl_M_pas')
    >>> l_M_tilde = FiberForceLengthPassiveInverseDeGroote2016.with_defaults(fl_M_pas)
    >>> l_M_tilde
    FiberForceLengthPassiveInverseDeGroote2016(fl_M_pas, 0.6, 4.0)

    It's also possible to populate the two constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1 = symbols('c0 c1')
    >>> l_M_tilde = FiberForceLengthPassiveInverseDeGroote2016(fl_M_pas, c0, c1)
    >>> l_M_tilde
    FiberForceLengthPassiveInverseDeGroote2016(fl_M_pas, c0, c1)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> l_M_tilde.doit(evaluate=False)
    c0*log(1 + fl_M_pas*(exp(c1) - 1))/c1 + 1

    The function can also be differentiated. We'll differentiate with respect
    to fl_M_pas using the ``diff`` method on an instance with the single positional
    argument ``fl_M_pas``.

    >>> l_M_tilde.diff(fl_M_pas)
    c0*(exp(c1) - 1)/(c1*(fl_M_pas*(exp(c1) - 1) + 1))

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod
    def with_defaults(cls, fl_M_pas):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the inverse muscle fiber passive force-length
        function using the four constant values specified in the original
        publication.

        These have the values:

        $c_0 = 0.6$
        $c_1 = 4.0$

        Parameters
        ==========

        fl_M_pas : Any (sympifiable)
            Normalized passive muscle fiber force as a function of muscle fiber
            length.

        """
        c0 = Float('0.6')
        c1 = Float('4.0')
        return cls(fl_M_pas, c0, c1)

    @classmethod
    def eval(cls, fl_M_pas, c0, c1):
        """Evaluation of basic inputs.

        Parameters
        ==========

        fl_M_pas : Any (sympifiable)
            Normalized passive muscle fiber force.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.6``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``4.0``.

        """
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``l_T_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        fl_M_pas, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            fl_M_pas = fl_M_pas.doit(deep=deep, **hints)
            c0, c1 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1 = constants

        if evaluate:
            return c0*log(fl_M_pas*(exp(c1) - 1) + 1)/c1 + 1

        return c0*log(UnevaluatedExpr(fl_M_pas*(exp(c1) - 1)) + 1)/c1 + 1

    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
        fl_M_pas, c0, c1 = self.args
        if argindex == 1:
            return c0*(exp(c1) - 1)/(c1*(fl_M_pas*(exp(c1) - 1) + 1))
        elif argindex == 2:
            return log(fl_M_pas*(exp(c1) - 1) + 1)/c1
        elif argindex == 3:
            return (
                c0*fl_M_pas*exp(c1)/(c1*(fl_M_pas*(exp(c1) - 1) + 1))
                - c0*log(fl_M_pas*(exp(c1) - 1) + 1)/c1**2
            )

        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
        return FiberForceLengthPassiveDeGroote2016

    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        fl_M_pas = self.args[0]
        _fl_M_pas = printer._print(fl_M_pas)
        return r'\left( \operatorname{fl}^M_{pas} \right)^{-1} \left( %s \right)' % _fl_M_pas


class FiberForceLengthActiveDeGroote2016(CharacteristicCurveFunction):
    r"""Active muscle fiber force-length curve based on De Groote et al., 2016
    [1]_.

    Explanation
    ===========

    The function is defined by the equation:

    $fl_{\text{act}}^M = c_0 \exp\left(-\frac{1}{2}\left(\frac{\tilde{l}^M - c_1}{c_2 + c_3 \tilde{l}^M}\right)^2\right)
    + c_4 \exp\left(-\frac{1}{2}\left(\frac{\tilde{l}^M - c_5}{c_6 + c_7 \tilde{l}^M}\right)^2\right)
    + c_8 \exp\left(-\frac{1}{2}\left(\frac{\tilde{l}^M - c_9}{c_{10} + c_{11} \tilde{l}^M}\right)^2\right)$

    with constant values of $c0 = 0.814$, $c1 = 1.06$, $c2 = 0.162$,
    $c3 = 0.0633$, $c4 = 0.433$, $c5 = 0.717$, $c6 = -0.0299$, $c7 = 0.2$,
    $c8 = 0.1$, $c9 = 1.0$, $c10 = 0.354$, and $c11 = 0.0$.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    active fiber force of 1 at a normalized fiber length of 1, and an active
    fiber force of 0 at normalized fiber lengths of 0 and 2.

    Examples
    ========

    The preferred way to instantiate :class:`FiberForceLengthActiveDeGroote2016` is
    using the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized muscle fiber length. We'll
    create a :class:`~.Symbol` called ``l_M_tilde`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceLengthActiveDeGroote2016
    >>> l_M_tilde = Symbol('l_M_tilde')
    >>> fl_M = FiberForceLengthActiveDeGroote2016.with_defaults(l_M_tilde)
    >>> fl_M
    FiberForceLengthActiveDeGroote2016(l_M_tilde, 0.814, 1.06, 0.162, 0.0633,
    0.433, 0.717, -0.0299, 0.2, 0.1, 1.0, 0.354, 0.0)

    It's also possible to populate the two constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = symbols('c0:12')
    >>> fl_M = FiberForceLengthActiveDeGroote2016(l_M_tilde, c0, c1, c2, c3,
    ...     c4, c5, c6, c7, c8, c9, c10, c11)
    >>> fl_M
    FiberForceLengthActiveDeGroote2016(l_M_tilde, c0, c1, c2, c3, c4, c5, c6,
    c7, c8, c9, c10, c11)

    You don't just have to use symbols as the arguments, it's also possible to
    use expressions. Let's create a new pair of symbols, ``l_M`` and
    ``l_M_opt``, representing muscle fiber length and optimal muscle fiber
    length respectively. We can then represent ``l_M_tilde`` as an expression,
    the ratio of these.

    >>> l_M, l_M_opt = symbols('l_M l_M_opt')
    >>> l_M_tilde = l_M/l_M_opt
    >>> fl_M = FiberForceLengthActiveDeGroote2016.with_defaults(l_M_tilde)
    >>> fl_M
    FiberForceLengthActiveDeGroote2016(l_M/l_M_opt, 0.814, 1.06, 0.162, 0.0633,
    0.433, 0.717, -0.0299, 0.2, 0.1, 1.0, 0.354, 0.0)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> fl_M.doit(evaluate=False)
    0.814*exp(-(l_M/l_M_opt
    - 1.06)**2/(2*(0.0633*l_M/l_M_opt + 0.162)**2))
    + 0.433*exp(-(l_M/l_M_opt - 0.717)**2/(2*(0.2*l_M/l_M_opt - 0.0299)**2))
    + 0.1*exp(-3.98991349867535*(l_M/l_M_opt - 1.0)**2)

    The function can also be differentiated. We'll differentiate with respect
    to l_M using the ``diff`` method on an instance with the single positional
    argument ``l_M``.

    >>> fl_M.diff(l_M)
    ((-0.79798269973507*l_M/l_M_opt
    + 0.79798269973507)*exp(-3.98991349867535*(l_M/l_M_opt - 1.0)**2)
    + (0.433*(-l_M/l_M_opt + 0.717)/(0.2*l_M/l_M_opt - 0.0299)**2
    + 0.0866*(l_M/l_M_opt - 0.717)**2/(0.2*l_M/l_M_opt
    - 0.0299)**3)*exp(-(l_M/l_M_opt - 0.717)**2/(2*(0.2*l_M/l_M_opt - 0.0299)**2))
    + (0.814*(-l_M/l_M_opt + 1.06)/(0.0633*l_M/l_M_opt
    + 0.162)**2 + 0.0515262*(l_M/l_M_opt
    - 1.06)**2/(0.0633*l_M/l_M_opt
    + 0.162)**3)*exp(-(l_M/l_M_opt
    - 1.06)**2/(2*(0.0633*l_M/l_M_opt + 0.162)**2)))/l_M_opt

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod
    def with_defaults(cls, l_M_tilde):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the inverse muscle fiber act force-length
        function using the four constant values specified in the original
        publication.

        These have the values:

        $c0 = 0.814$
        $c1 = 1.06$
        $c2 = 0.162$
        $c3 = 0.0633$
        $c4 = 0.433$
        $c5 = 0.717$
        $c6 = -0.0299$
        $c7 = 0.2$
        $c8 = 0.1$
        $c9 = 1.0$
        $c10 = 0.354$
        $c11 = 0.0$

        Parameters
        ==========

        fl_M_act : Any (sympifiable)
            Normalized passive muscle fiber force as a function of muscle fiber
            length.

        """
        c0 = Float('0.814')
        c1 = Float('1.06')
        c2 = Float('0.162')
        c3 = Float('0.0633')
        c4 = Float('0.433')
        c5 = Float('0.717')
        c6 = Float('-0.0299')
        c7 = Float('0.2')
        c8 = Float('0.1')
        c9 = Float('1.0')
        c10 = Float('0.354')
        c11 = Float('0.0')
        return cls(l_M_tilde, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11)

    @classmethod
    def eval(cls, l_M_tilde, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11):
        """Evaluation of basic inputs.

        Parameters
        ==========

        l_M_tilde : Any (sympifiable)
            Normalized muscle fiber length.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.814``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``1.06``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``0.162``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``0.0633``.
        c4 : Any (sympifiable)
            The fifth constant in the characteristic equation. The published
            value is ``0.433``.
        c5 : Any (sympifiable)
            The sixth constant in the characteristic equation. The published
            value is ``0.717``.
        c6 : Any (sympifiable)
            The seventh constant in the characteristic equation. The published
            value is ``-0.0299``.
        c7 : Any (sympifiable)
            The eighth constant in the characteristic equation. The published
            value is ``0.2``.
        c8 : Any (sympifiable)
            The ninth constant in the characteristic equation. The published
            value is ``0.1``.
        c9 : Any (sympifiable)
            The tenth constant in the characteristic equation. The published
            value is ``1.0``.
        c10 : Any (sympifiable)
            The eleventh constant in the characteristic equation. The published
            value is ``0.354``.
        c11 : Any (sympifiable)
            The tweflth constant in the characteristic equation. The published
            value is ``0.0``.

        """
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``l_M_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        l_M_tilde, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            l_M_tilde = l_M_tilde.doit(deep=deep, **hints)
            constants = [c.doit(deep=deep, **hints) for c in constants]
        c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = constants

        if evaluate:
            return (
                c0*exp(-(((l_M_tilde - c1)/(c2 + c3*l_M_tilde))**2)/2)
                + c4*exp(-(((l_M_tilde - c5)/(c6 + c7*l_M_tilde))**2)/2)
                + c8*exp(-(((l_M_tilde - c9)/(c10 + c11*l_M_tilde))**2)/2)
            )

        return (
            c0*exp(-((UnevaluatedExpr(l_M_tilde - c1)/(c2 + c3*l_M_tilde))**2)/2)
            + c4*exp(-((UnevaluatedExpr(l_M_tilde - c5)/(c6 + c7*l_M_tilde))**2)/2)
            + c8*exp(-((UnevaluatedExpr(l_M_tilde - c9)/(c10 + c11*l_M_tilde))**2)/2)
        )

    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
        l_M_tilde, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = self.args
        if argindex == 1:
            return (
                c0*(
                    c3*(l_M_tilde - c1)**2/(c2 + c3*l_M_tilde)**3
                    + (c1 - l_M_tilde)/((c2 + c3*l_M_tilde)**2)
                )*exp(-(l_M_tilde - c1)**2/(2*(c2 + c3*l_M_tilde)**2))
                + c4*(
                    c7*(l_M_tilde - c5)**2/(c6 + c7*l_M_tilde)**3
                    + (c5 - l_M_tilde)/((c6 + c7*l_M_tilde)**2)
                )*exp(-(l_M_tilde - c5)**2/(2*(c6 + c7*l_M_tilde)**2))
                + c8*(
                    c11*(l_M_tilde - c9)**2/(c10 + c11*l_M_tilde)**3
                    + (c9 - l_M_tilde)/((c10 + c11*l_M_tilde)**2)
                )*exp(-(l_M_tilde - c9)**2/(2*(c10 + c11*l_M_tilde)**2))
            )
        elif argindex == 2:
            return exp(-(l_M_tilde - c1)**2/(2*(c2 + c3*l_M_tilde)**2))
        elif argindex == 3:
            return (
                c0*(l_M_tilde - c1)/(c2 + c3*l_M_tilde)**2
                *exp(-(l_M_tilde - c1)**2 /(2*(c2 + c3*l_M_tilde)**2))
            )
        elif argindex == 4:
            return (
                c0*(l_M_tilde - c1)**2/(c2 + c3*l_M_tilde)**3
                *exp(-(l_M_tilde - c1)**2/(2*(c2 + c3*l_M_tilde)**2))
            )
        elif argindex == 5:
            return (
                c0*l_M_tilde*(l_M_tilde - c1)**2/(c2 + c3*l_M_tilde)**3
                *exp(-(l_M_tilde - c1)**2/(2*(c2 + c3*l_M_tilde)**2))
            )
        elif argindex == 6:
            return exp(-(l_M_tilde - c5)**2/(2*(c6 + c7*l_M_tilde)**2))
        elif argindex == 7:
            return (
                c4*(l_M_tilde - c5)/(c6 + c7*l_M_tilde)**2
                *exp(-(l_M_tilde - c5)**2 /(2*(c6 + c7*l_M_tilde)**2))
            )
        elif argindex == 8:
            return (
                c4*(l_M_tilde - c5)**2/(c6 + c7*l_M_tilde)**3
                *exp(-(l_M_tilde - c5)**2/(2*(c6 + c7*l_M_tilde)**2))
            )
        elif argindex == 9:
            return (
                c4*l_M_tilde*(l_M_tilde - c5)**2/(c6 + c7*l_M_tilde)**3
                *exp(-(l_M_tilde - c5)**2/(2*(c6 + c7*l_M_tilde)**2))
            )
        elif argindex == 10:
            return exp(-(l_M_tilde - c9)**2/(2*(c10 + c11*l_M_tilde)**2))
        elif argindex == 11:
            return (
                c8*(l_M_tilde - c9)/(c10 + c11*l_M_tilde)**2
                *exp(-(l_M_tilde - c9)**2 /(2*(c10 + c11*l_M_tilde)**2))
            )
        elif argindex == 12:
            return (
                c8*(l_M_tilde - c9)**2/(c10 + c11*l_M_tilde)**3
                *exp(-(l_M_tilde - c9)**2/(2*(c10 + c11*l_M_tilde)**2))
            )
        elif argindex == 13:
            return (
                c8*l_M_tilde*(l_M_tilde - c9)**2/(c10 + c11*l_M_tilde)**3
                *exp(-(l_M_tilde - c9)**2/(2*(c10 + c11*l_M_tilde)**2))
            )

        raise ArgumentIndexError(self, argindex)

    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        l_M_tilde = self.args[0]
        _l_M_tilde = printer._print(l_M_tilde)
        return r'\operatorname{fl}^M_{act} \left( %s \right)' % _l_M_tilde


class FiberForceVelocityDeGroote2016(CharacteristicCurveFunction):
    r"""Muscle fiber force-velocity curve based on De Groote et al., 2016 [1]_.

    Explanation
    ===========

    Gives the normalized muscle fiber force produced as a function of
    normalized tendon velocity.

    The function is defined by the equation:

    $fv^M = c_0 \log{\left(c_1 \tilde{v}_m + c_2\right) + \sqrt{\left(c_1 \tilde{v}_m + c_2\right)^2 + 1}} + c_3$

    with constant values of $c_0 = -0.318$, $c_1 = -8.149$, $c_2 = -0.374$, and
    $c_3 = 0.886$.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    normalized muscle fiber force of 1 when the muscle fibers are contracting
    isometrically (they have an extension rate of 0).

    Examples
    ========

    The preferred way to instantiate :class:`FiberForceVelocityDeGroote2016` is using
    the :meth:`~.with_defaults` constructor because this will automatically populate
    the constants within the characteristic curve equation with the floating
    point values from the original publication. This constructor takes a single
    argument corresponding to normalized muscle fiber extension velocity. We'll
    create a :class:`~.Symbol` called ``v_M_tilde`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceVelocityDeGroote2016
    >>> v_M_tilde = Symbol('v_M_tilde')
    >>> fv_M = FiberForceVelocityDeGroote2016.with_defaults(v_M_tilde)
    >>> fv_M
    FiberForceVelocityDeGroote2016(v_M_tilde, -0.318, -8.149, -0.374, 0.886)

    It's also possible to populate the four constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3 = symbols('c0 c1 c2 c3')
    >>> fv_M = FiberForceVelocityDeGroote2016(v_M_tilde, c0, c1, c2, c3)
    >>> fv_M
    FiberForceVelocityDeGroote2016(v_M_tilde, c0, c1, c2, c3)

    You don't just have to use symbols as the arguments, it's also possible to
    use expressions. Let's create a new pair of symbols, ``v_M`` and
    ``v_M_max``, representing muscle fiber extension velocity and maximum
    muscle fiber extension velocity respectively. We can then represent
    ``v_M_tilde`` as an expression, the ratio of these.

    >>> v_M, v_M_max = symbols('v_M v_M_max')
    >>> v_M_tilde = v_M/v_M_max
    >>> fv_M = FiberForceVelocityDeGroote2016.with_defaults(v_M_tilde)
    >>> fv_M
    FiberForceVelocityDeGroote2016(v_M/v_M_max, -0.318, -8.149, -0.374, 0.886)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> fv_M.doit(evaluate=False)
    0.886 - 0.318*log(-8.149*v_M/v_M_max - 0.374 + sqrt(1 + (-8.149*v_M/v_M_max
    - 0.374)**2))

    The function can also be differentiated. We'll differentiate with respect
    to v_M using the ``diff`` method on an instance with the single positional
    argument ``v_M``.

    >>> fv_M.diff(v_M)
    2.591382*(1 + (-8.149*v_M/v_M_max - 0.374)**2)**(-1/2)/v_M_max

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod
    def with_defaults(cls, v_M_tilde):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the muscle fiber force-velocity function
        using the four constant values specified in the original publication.

        These have the values:

        $c_0 = -0.318$
        $c_1 = -8.149$
        $c_2 = -0.374$
        $c_3 = 0.886$

        Parameters
        ==========

        v_M_tilde : Any (sympifiable)
            Normalized muscle fiber extension velocity.

        """
        c0 = Float('-0.318')
        c1 = Float('-8.149')
        c2 = Float('-0.374')
        c3 = Float('0.886')
        return cls(v_M_tilde, c0, c1, c2, c3)

    @classmethod
    def eval(cls, v_M_tilde, c0, c1, c2, c3):
        """Evaluation of basic inputs.

        Parameters
        ==========

        v_M_tilde : Any (sympifiable)
            Normalized muscle fiber extension velocity.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``-0.318``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``-8.149``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``-0.374``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``0.886``.

        """
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``v_M_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        v_M_tilde, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            v_M_tilde = v_M_tilde.doit(deep=deep, **hints)
            c0, c1, c2, c3 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1, c2, c3 = constants

        if evaluate:
            return c0*log(c1*v_M_tilde + c2 + sqrt((c1*v_M_tilde + c2)**2 + 1)) + c3

        return c0*log(c1*v_M_tilde + c2 + sqrt(UnevaluatedExpr(c1*v_M_tilde + c2)**2 + 1)) + c3

    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
        v_M_tilde, c0, c1, c2, c3 = self.args
        if argindex == 1:
            return c0*c1/sqrt(UnevaluatedExpr(c1*v_M_tilde + c2)**2 + 1)
        elif argindex == 2:
            return log(
                c1*v_M_tilde + c2
                + sqrt(UnevaluatedExpr(c1*v_M_tilde + c2)**2 + 1)
            )
        elif argindex == 3:
            return c0*v_M_tilde/sqrt(UnevaluatedExpr(c1*v_M_tilde + c2)**2 + 1)
        elif argindex == 4:
            return c0/sqrt(UnevaluatedExpr(c1*v_M_tilde + c2)**2 + 1)
        elif argindex == 5:
            return Integer(1)

        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
        return FiberForceVelocityInverseDeGroote2016

    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        v_M_tilde = self.args[0]
        _v_M_tilde = printer._print(v_M_tilde)
        return r'\operatorname{fv}^M \left( %s \right)' % _v_M_tilde


class FiberForceVelocityInverseDeGroote2016(CharacteristicCurveFunction):
    r"""Inverse muscle fiber force-velocity curve based on De Groote et al.,
    2016 [1]_.

    Explanation
    ===========

    Gives the normalized muscle fiber velocity that produces a specific
    normalized muscle fiber force.

    The function is defined by the equation:

    ${fv^M}^{-1} = \frac{\sinh{\frac{fv^M - c_3}{c_0}} - c_2}{c_1}$

    with constant values of $c_0 = -0.318$, $c_1 = -8.149$, $c_2 = -0.374$, and
    $c_3 = 0.886$. This function is the exact analytical inverse of the related
    muscle fiber force-velocity curve ``FiberForceVelocityDeGroote2016``.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    normalized muscle fiber force of 1 when the muscle fibers are contracting
    isometrically (they have an extension rate of 0).

    Examples
    ========

    The preferred way to instantiate :class:`FiberForceVelocityInverseDeGroote2016`
    is using the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized muscle fiber force-velocity
    component of the muscle fiber force. We'll create a :class:`~.Symbol` called
    ``fv_M`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceVelocityInverseDeGroote2016
    >>> fv_M = Symbol('fv_M')
    >>> v_M_tilde = FiberForceVelocityInverseDeGroote2016.with_defaults(fv_M)
    >>> v_M_tilde
    FiberForceVelocityInverseDeGroote2016(fv_M, -0.318, -8.149, -0.374, 0.886)

    It's also possible to populate the four constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3 = symbols('c0 c1 c2 c3')
    >>> v_M_tilde = FiberForceVelocityInverseDeGroote2016(fv_M, c0, c1, c2, c3)
    >>> v_M_tilde
    FiberForceVelocityInverseDeGroote2016(fv_M, c0, c1, c2, c3)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> v_M_tilde.doit(evaluate=False)
    (-c2 + sinh((-c3 + fv_M)/c0))/c1

    The function can also be differentiated. We'll differentiate with respect
    to fv_M using the ``diff`` method on an instance with the single positional
    argument ``fv_M``.

    >>> v_M_tilde.diff(fv_M)
    cosh((-c3 + fv_M)/c0)/(c0*c1)

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod
    def with_defaults(cls, fv_M):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the inverse muscle fiber force-velocity
        function using the four constant values specified in the original
        publication.

        These have the values:

        $c_0 = -0.318$
        $c_1 = -8.149$
        $c_2 = -0.374$
        $c_3 = 0.886$

        Parameters
        ==========

        fv_M : Any (sympifiable)
            Normalized muscle fiber extension velocity.

        """
        c0 = Float('-0.318')
        c1 = Float('-8.149')
        c2 = Float('-0.374')
        c3 = Float('0.886')
        return cls(fv_M, c0, c1, c2, c3)

    @classmethod
    def eval(cls, fv_M, c0, c1, c2, c3):
        """Evaluation of basic inputs.

        Parameters
        ==========

        fv_M : Any (sympifiable)
            Normalized muscle fiber force as a function of muscle fiber
            extension velocity.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``-0.318``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``-8.149``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``-0.374``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``0.886``.

        """
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``fv_M`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        fv_M, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            fv_M = fv_M.doit(deep=deep, **hints)
            c0, c1, c2, c3 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1, c2, c3 = constants

        if evaluate:
            return (sinh((fv_M - c3)/c0) - c2)/c1

        return (sinh(UnevaluatedExpr(fv_M - c3)/c0) - c2)/c1

    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
        fv_M, c0, c1, c2, c3 = self.args
        if argindex == 1:
            return cosh((fv_M - c3)/c0)/(c0*c1)
        elif argindex == 2:
            return (c3 - fv_M)*cosh((fv_M - c3)/c0)/(c0**2*c1)
        elif argindex == 3:
            return (c2 - sinh((fv_M - c3)/c0))/c1**2
        elif argindex == 4:
            return -1/c1
        elif argindex == 5:
            return -cosh((fv_M - c3)/c0)/(c0*c1)

        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
        return FiberForceVelocityDeGroote2016

    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        fv_M = self.args[0]
        _fv_M = printer._print(fv_M)
        return r'\left( \operatorname{fv}^M \right)^{-1} \left( %s \right)' % _fv_M


@dataclass(frozen=True)
class CharacteristicCurveCollection:
    """Simple data container to group together related characteristic curves."""
    tendon_force_length: CharacteristicCurveFunction
    tendon_force_length_inverse: CharacteristicCurveFunction
    fiber_force_length_passive: CharacteristicCurveFunction
    fiber_force_length_passive_inverse: CharacteristicCurveFunction
    fiber_force_length_active: CharacteristicCurveFunction
    fiber_force_velocity: CharacteristicCurveFunction
    fiber_force_velocity_inverse: CharacteristicCurveFunction

    def __iter__(self):
        """Iterator support for ``CharacteristicCurveCollection``."""
        yield self.tendon_force_length
        yield self.tendon_force_length_inverse
        yield self.fiber_force_length_passive
        yield self.fiber_force_length_passive_inverse
        yield self.fiber_force_length_active
        yield self.fiber_force_velocity
        yield self.fiber_force_velocity_inverse
