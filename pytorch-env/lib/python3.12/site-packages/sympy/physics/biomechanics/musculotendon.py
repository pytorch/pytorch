"""Implementations of musculotendon models.

Musculotendon models are a critical component of biomechanical models, one that
differentiates them from pure multibody systems. Musculotendon models produce a
force dependent on their level of activation, their length, and their
extension velocity. Length- and extension velocity-dependent force production
are governed by force-length and force-velocity characteristics.
These are normalized functions that are dependent on the musculotendon's state
and are specific to a given musculotendon model.

"""

from abc import abstractmethod
from enum import IntEnum, unique

from sympy.core.numbers import Float, Integer
from sympy.core.symbol import Symbol, symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices.dense import MutableDenseMatrix as Matrix, diag, eye, zeros
from sympy.physics.biomechanics.activation import ActivationBase
from sympy.physics.biomechanics.curve import (
    CharacteristicCurveCollection,
    FiberForceLengthActiveDeGroote2016,
    FiberForceLengthPassiveDeGroote2016,
    FiberForceLengthPassiveInverseDeGroote2016,
    FiberForceVelocityDeGroote2016,
    FiberForceVelocityInverseDeGroote2016,
    TendonForceLengthDeGroote2016,
    TendonForceLengthInverseDeGroote2016,
)
from sympy.physics.biomechanics._mixin import _NamedMixin
from sympy.physics.mechanics.actuator import ForceActuator
from sympy.physics.vector.functions import dynamicsymbols


__all__ = [
    'MusculotendonBase',
    'MusculotendonDeGroote2016',
    'MusculotendonFormulation',
]


@unique
class MusculotendonFormulation(IntEnum):
    """Enumeration of types of musculotendon dynamics formulations.

    Explanation
    ===========

    An (integer) enumeration is used as it allows for clearer selection of the
    different formulations of musculotendon dynamics.

    Members
    =======

    RIGID_TENDON : 0
        A rigid tendon model.
    FIBER_LENGTH_EXPLICIT : 1
        An explicit elastic tendon model with the muscle fiber length (l_M) as
        the state variable.
    TENDON_FORCE_EXPLICIT : 2
        An explicit elastic tendon model with the tendon force (F_T) as the
        state variable.
    FIBER_LENGTH_IMPLICIT : 3
        An implicit elastic tendon model with the muscle fiber length (l_M) as
        the state variable and the muscle fiber velocity as an additional input
        variable.
    TENDON_FORCE_IMPLICIT : 4
        An implicit elastic tendon model with the tendon force (F_T) as the
        state variable as the muscle fiber velocity as an additional input
        variable.

    """

    RIGID_TENDON = 0
    FIBER_LENGTH_EXPLICIT = 1
    TENDON_FORCE_EXPLICIT = 2
    FIBER_LENGTH_IMPLICIT = 3
    TENDON_FORCE_IMPLICIT = 4

    def __str__(self):
        """Returns a string representation of the enumeration value.

        Notes
        =====

        This hard coding is required due to an incompatibility between the
        ``IntEnum`` implementations in Python 3.10 and Python 3.11
        (https://github.com/python/cpython/issues/84247). From Python 3.11
        onwards, the ``__str__`` method uses ``int.__str__``, whereas prior it
        used ``Enum.__str__``. Once Python 3.11 becomes the minimum version
        supported by SymPy, this method override can be removed.

        """
        return str(self.value)


_DEFAULT_MUSCULOTENDON_FORMULATION = MusculotendonFormulation.RIGID_TENDON


class MusculotendonBase(ForceActuator, _NamedMixin):
    r"""Abstract base class for all musculotendon classes to inherit from.

    Explanation
    ===========

    A musculotendon generates a contractile force based on its activation,
    length, and shortening velocity. This abstract base class is to be inherited
    by all musculotendon subclasses that implement different characteristic
    musculotendon curves. Characteristic musculotendon curves are required for
    the tendon force-length, passive fiber force-length, active fiber force-
    length, and fiber force-velocity relationships.

    Parameters
    ==========

    name : str
        The name identifier associated with the musculotendon. This name is used
        as a suffix when automatically generated symbols are instantiated. It
        must be a string of nonzero length.
    pathway : PathwayBase
        The pathway that the actuator follows. This must be an instance of a
        concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.
    activation_dynamics : ActivationBase
        The activation dynamics that will be modeled within the musculotendon.
        This must be an instance of a concrete subclass of ``ActivationBase``,
        e.g. ``FirstOrderActivationDeGroote2016``.
    musculotendon_dynamics : MusculotendonFormulation | int
        The formulation of musculotendon dynamics that should be used
        internally, i.e. rigid or elastic tendon model, the choice of
        musculotendon state etc. This must be a member of the integer
        enumeration ``MusculotendonFormulation`` or an integer that can be cast
        to a member. To use a rigid tendon formulation, set this to
        ``MusculotendonFormulation.RIGID_TENDON`` (or the integer value ``0``,
        which will be cast to the enumeration member). There are four possible
        formulations for an elastic tendon model. To use an explicit formulation
        with the fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_EXPLICIT`` (or the integer value
        ``1``). To use an explicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_EXPLICIT``
        (or the integer value ``2``). To use an implicit formulation with the
        fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_IMPLICIT`` (or the integer value
        ``3``). To use an implicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_IMPLICIT``
        (or the integer value ``4``). The default is
        ``MusculotendonFormulation.RIGID_TENDON``, which corresponds to a rigid
        tendon formulation.
    tendon_slack_length : Expr | None
        The length of the tendon when the musculotendon is in its unloaded
        state. In a rigid tendon model the tendon length is the tendon slack
        length. In all musculotendon models, tendon slack length is used to
        normalize tendon length to give
        :math:`\tilde{l}^T = \frac{l^T}{l^T_{slack}}`.
    peak_isometric_force : Expr | None
        The maximum force that the muscle fiber can produce when it is
        undergoing an isometric contraction (no lengthening velocity). In all
        musculotendon models, peak isometric force is used to normalized tendon
        and muscle fiber force to give
        :math:`\tilde{F}^T = \frac{F^T}{F^M_{max}}`.
    optimal_fiber_length : Expr | None
        The muscle fiber length at which the muscle fibers produce no passive
        force and their maximum active force. In all musculotendon models,
        optimal fiber length is used to normalize muscle fiber length to give
        :math:`\tilde{l}^M = \frac{l^M}{l^M_{opt}}`.
    maximal_fiber_velocity : Expr | None
        The fiber velocity at which, during muscle fiber shortening, the muscle
        fibers are unable to produce any active force. In all musculotendon
        models, maximal fiber velocity is used to normalize muscle fiber
        extension velocity to give :math:`\tilde{v}^M = \frac{v^M}{v^M_{max}}`.
    optimal_pennation_angle : Expr | None
        The pennation angle when muscle fiber length equals the optimal fiber
        length.
    fiber_damping_coefficient : Expr | None
        The coefficient of damping to be used in the damping element in the
        muscle fiber model.
    with_defaults : bool
        Whether ``with_defaults`` alternate constructors should be used when
        automatically constructing child classes. Default is ``False``.

    """

    def __init__(
        self,
        name,
        pathway,
        activation_dynamics,
        *,
        musculotendon_dynamics=_DEFAULT_MUSCULOTENDON_FORMULATION,
        tendon_slack_length=None,
        peak_isometric_force=None,
        optimal_fiber_length=None,
        maximal_fiber_velocity=None,
        optimal_pennation_angle=None,
        fiber_damping_coefficient=None,
        with_defaults=False,
    ):
        self.name = name

        # Supply a placeholder force to the super initializer, this will be
        # replaced later
        super().__init__(Symbol('F'), pathway)

        # Activation dynamics
        if not isinstance(activation_dynamics, ActivationBase):
            msg = (
                f'Can\'t set attribute `activation_dynamics` to '
                f'{activation_dynamics} as it must be of type '
                f'`ActivationBase`, not {type(activation_dynamics)}.'
            )
            raise TypeError(msg)
        self._activation_dynamics = activation_dynamics
        self._child_objects = (self._activation_dynamics, )

        # Constants
        if tendon_slack_length is not None:
            self._l_T_slack = tendon_slack_length
        else:
            self._l_T_slack = Symbol(f'l_T_slack_{self.name}')
        if peak_isometric_force is not None:
            self._F_M_max = peak_isometric_force
        else:
            self._F_M_max = Symbol(f'F_M_max_{self.name}')
        if optimal_fiber_length is not None:
            self._l_M_opt = optimal_fiber_length
        else:
            self._l_M_opt = Symbol(f'l_M_opt_{self.name}')
        if maximal_fiber_velocity is not None:
            self._v_M_max = maximal_fiber_velocity
        else:
            self._v_M_max = Symbol(f'v_M_max_{self.name}')
        if optimal_pennation_angle is not None:
            self._alpha_opt = optimal_pennation_angle
        else:
            self._alpha_opt = Symbol(f'alpha_opt_{self.name}')
        if fiber_damping_coefficient is not None:
            self._beta = fiber_damping_coefficient
        else:
            self._beta = Symbol(f'beta_{self.name}')

        # Musculotendon dynamics
        self._with_defaults = with_defaults
        if musculotendon_dynamics == MusculotendonFormulation.RIGID_TENDON:
            self._rigid_tendon_musculotendon_dynamics()
        elif musculotendon_dynamics == MusculotendonFormulation.FIBER_LENGTH_EXPLICIT:
            self._fiber_length_explicit_musculotendon_dynamics()
        elif musculotendon_dynamics == MusculotendonFormulation.TENDON_FORCE_EXPLICIT:
            self._tendon_force_explicit_musculotendon_dynamics()
        elif musculotendon_dynamics == MusculotendonFormulation.FIBER_LENGTH_IMPLICIT:
            self._fiber_length_implicit_musculotendon_dynamics()
        elif musculotendon_dynamics == MusculotendonFormulation.TENDON_FORCE_IMPLICIT:
            self._tendon_force_implicit_musculotendon_dynamics()
        else:
            msg = (
                f'Musculotendon dynamics {repr(musculotendon_dynamics)} '
                f'passed to `musculotendon_dynamics` was of type '
                f'{type(musculotendon_dynamics)}, must be '
                f'{MusculotendonFormulation}.'
            )
            raise TypeError(msg)
        self._musculotendon_dynamics = musculotendon_dynamics

        # Must override the placeholder value in `self._force` now that the
        # actual force has been calculated by
        # `self._<MUSCULOTENDON FORMULATION>_musculotendon_dynamics`.
        # Note that `self._force` assumes forces are expansile, musculotendon
        # forces are contractile hence the minus sign preceeding `self._F_T`
        # (the tendon force).
        self._force = -self._F_T

    @classmethod
    def with_defaults(
        cls,
        name,
        pathway,
        activation_dynamics,
        *,
        musculotendon_dynamics=_DEFAULT_MUSCULOTENDON_FORMULATION,
        tendon_slack_length=None,
        peak_isometric_force=None,
        optimal_fiber_length=None,
        maximal_fiber_velocity=Float('10.0'),
        optimal_pennation_angle=Float('0.0'),
        fiber_damping_coefficient=Float('0.1'),
    ):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the musculotendon class using recommended
        values for ``v_M_max``, ``alpha_opt``, and ``beta``. The values are:

            :math:`v^M_{max} = 10`
            :math:`\alpha_{opt} = 0`
            :math:`\beta = \frac{1}{10}`

        The musculotendon curves are also instantiated using the constants from
        the original publication.

        Parameters
        ==========

        name : str
            The name identifier associated with the musculotendon. This name is
            used as a suffix when automatically generated symbols are
            instantiated. It must be a string of nonzero length.
        pathway : PathwayBase
            The pathway that the actuator follows. This must be an instance of a
            concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.
        activation_dynamics : ActivationBase
            The activation dynamics that will be modeled within the
            musculotendon. This must be an instance of a concrete subclass of
            ``ActivationBase``, e.g. ``FirstOrderActivationDeGroote2016``.
        musculotendon_dynamics : MusculotendonFormulation | int
            The formulation of musculotendon dynamics that should be used
            internally, i.e. rigid or elastic tendon model, the choice of
            musculotendon state etc. This must be a member of the integer
            enumeration ``MusculotendonFormulation`` or an integer that can be
            cast to a member. To use a rigid tendon formulation, set this to
            ``MusculotendonFormulation.RIGID_TENDON`` (or the integer value
            ``0``, which will be cast to the enumeration member). There are four
            possible formulations for an elastic tendon model. To use an
            explicit formulation with the fiber length as the state, set this to
            ``MusculotendonFormulation.FIBER_LENGTH_EXPLICIT`` (or the integer
            value ``1``). To use an explicit formulation with the tendon force
            as the state, set this to
            ``MusculotendonFormulation.TENDON_FORCE_EXPLICIT`` (or the integer
            value ``2``). To use an implicit formulation with the fiber length
            as the state, set this to
            ``MusculotendonFormulation.FIBER_LENGTH_IMPLICIT`` (or the integer
            value ``3``). To use an implicit formulation with the tendon force
            as the state, set this to
            ``MusculotendonFormulation.TENDON_FORCE_IMPLICIT`` (or the integer
            value ``4``). The default is
            ``MusculotendonFormulation.RIGID_TENDON``, which corresponds to a
            rigid tendon formulation.
        tendon_slack_length : Expr | None
            The length of the tendon when the musculotendon is in its unloaded
            state. In a rigid tendon model the tendon length is the tendon slack
            length. In all musculotendon models, tendon slack length is used to
            normalize tendon length to give
            :math:`\tilde{l}^T = \frac{l^T}{l^T_{slack}}`.
        peak_isometric_force : Expr | None
            The maximum force that the muscle fiber can produce when it is
            undergoing an isometric contraction (no lengthening velocity). In
            all musculotendon models, peak isometric force is used to normalized
            tendon and muscle fiber force to give
            :math:`\tilde{F}^T = \frac{F^T}{F^M_{max}}`.
        optimal_fiber_length : Expr | None
            The muscle fiber length at which the muscle fibers produce no
            passive force and their maximum active force. In all musculotendon
            models, optimal fiber length is used to normalize muscle fiber
            length to give :math:`\tilde{l}^M = \frac{l^M}{l^M_{opt}}`.
        maximal_fiber_velocity : Expr | None
            The fiber velocity at which, during muscle fiber shortening, the
            muscle fibers are unable to produce any active force. In all
            musculotendon models, maximal fiber velocity is used to normalize
            muscle fiber extension velocity to give
            :math:`\tilde{v}^M = \frac{v^M}{v^M_{max}}`.
        optimal_pennation_angle : Expr | None
            The pennation angle when muscle fiber length equals the optimal
            fiber length.
        fiber_damping_coefficient : Expr | None
            The coefficient of damping to be used in the damping element in the
            muscle fiber model.

        """
        return cls(
            name,
            pathway,
            activation_dynamics=activation_dynamics,
            musculotendon_dynamics=musculotendon_dynamics,
            tendon_slack_length=tendon_slack_length,
            peak_isometric_force=peak_isometric_force,
            optimal_fiber_length=optimal_fiber_length,
            maximal_fiber_velocity=maximal_fiber_velocity,
            optimal_pennation_angle=optimal_pennation_angle,
            fiber_damping_coefficient=fiber_damping_coefficient,
            with_defaults=True,
        )

    @abstractmethod
    def curves(cls):
        """Return a ``CharacteristicCurveCollection`` of the curves related to
        the specific model."""
        pass

    @property
    def tendon_slack_length(self):
        r"""Symbol or value corresponding to the tendon slack length constant.

        Explanation
        ===========

        The length of the tendon when the musculotendon is in its unloaded
        state. In a rigid tendon model the tendon length is the tendon slack
        length. In all musculotendon models, tendon slack length is used to
        normalize tendon length to give
        :math:`\tilde{l}^T = \frac{l^T}{l^T_{slack}}`.

        The alias ``l_T_slack`` can also be used to access the same attribute.

        """
        return self._l_T_slack

    @property
    def l_T_slack(self):
        r"""Symbol or value corresponding to the tendon slack length constant.

        Explanation
        ===========

        The length of the tendon when the musculotendon is in its unloaded
        state. In a rigid tendon model the tendon length is the tendon slack
        length. In all musculotendon models, tendon slack length is used to
        normalize tendon length to give
        :math:`\tilde{l}^T = \frac{l^T}{l^T_{slack}}`.

        The alias ``tendon_slack_length`` can also be used to access the same
        attribute.

        """
        return self._l_T_slack

    @property
    def peak_isometric_force(self):
        r"""Symbol or value corresponding to the peak isometric force constant.

        Explanation
        ===========

        The maximum force that the muscle fiber can produce when it is
        undergoing an isometric contraction (no lengthening velocity). In all
        musculotendon models, peak isometric force is used to normalized tendon
        and muscle fiber force to give
        :math:`\tilde{F}^T = \frac{F^T}{F^M_{max}}`.

        The alias ``F_M_max`` can also be used to access the same attribute.

        """
        return self._F_M_max

    @property
    def F_M_max(self):
        r"""Symbol or value corresponding to the peak isometric force constant.

        Explanation
        ===========

        The maximum force that the muscle fiber can produce when it is
        undergoing an isometric contraction (no lengthening velocity). In all
        musculotendon models, peak isometric force is used to normalized tendon
        and muscle fiber force to give
        :math:`\tilde{F}^T = \frac{F^T}{F^M_{max}}`.

        The alias ``peak_isometric_force`` can also be used to access the same
        attribute.

        """
        return self._F_M_max

    @property
    def optimal_fiber_length(self):
        r"""Symbol or value corresponding to the optimal fiber length constant.

        Explanation
        ===========

        The muscle fiber length at which the muscle fibers produce no passive
        force and their maximum active force. In all musculotendon models,
        optimal fiber length is used to normalize muscle fiber length to give
        :math:`\tilde{l}^M = \frac{l^M}{l^M_{opt}}`.

        The alias ``l_M_opt`` can also be used to access the same attribute.

        """
        return self._l_M_opt

    @property
    def l_M_opt(self):
        r"""Symbol or value corresponding to the optimal fiber length constant.

        Explanation
        ===========

        The muscle fiber length at which the muscle fibers produce no passive
        force and their maximum active force. In all musculotendon models,
        optimal fiber length is used to normalize muscle fiber length to give
        :math:`\tilde{l}^M = \frac{l^M}{l^M_{opt}}`.

        The alias ``optimal_fiber_length`` can also be used to access the same
        attribute.

        """
        return self._l_M_opt

    @property
    def maximal_fiber_velocity(self):
        r"""Symbol or value corresponding to the maximal fiber velocity constant.

        Explanation
        ===========

        The fiber velocity at which, during muscle fiber shortening, the muscle
        fibers are unable to produce any active force. In all musculotendon
        models, maximal fiber velocity is used to normalize muscle fiber
        extension velocity to give :math:`\tilde{v}^M = \frac{v^M}{v^M_{max}}`.

        The alias ``v_M_max`` can also be used to access the same attribute.

        """
        return self._v_M_max

    @property
    def v_M_max(self):
        r"""Symbol or value corresponding to the maximal fiber velocity constant.

        Explanation
        ===========

        The fiber velocity at which, during muscle fiber shortening, the muscle
        fibers are unable to produce any active force. In all musculotendon
        models, maximal fiber velocity is used to normalize muscle fiber
        extension velocity to give :math:`\tilde{v}^M = \frac{v^M}{v^M_{max}}`.

        The alias ``maximal_fiber_velocity`` can also be used to access the same
        attribute.

        """
        return self._v_M_max

    @property
    def optimal_pennation_angle(self):
        """Symbol or value corresponding to the optimal pennation angle
        constant.

        Explanation
        ===========

        The pennation angle when muscle fiber length equals the optimal fiber
        length.

        The alias ``alpha_opt`` can also be used to access the same attribute.

        """
        return self._alpha_opt

    @property
    def alpha_opt(self):
        """Symbol or value corresponding to the optimal pennation angle
        constant.

        Explanation
        ===========

        The pennation angle when muscle fiber length equals the optimal fiber
        length.

        The alias ``optimal_pennation_angle`` can also be used to access the
        same attribute.

        """
        return self._alpha_opt

    @property
    def fiber_damping_coefficient(self):
        """Symbol or value corresponding to the fiber damping coefficient
        constant.

        Explanation
        ===========

        The coefficient of damping to be used in the damping element in the
        muscle fiber model.

        The alias ``beta`` can also be used to access the same attribute.

        """
        return self._beta

    @property
    def beta(self):
        """Symbol or value corresponding to the fiber damping coefficient
        constant.

        Explanation
        ===========

        The coefficient of damping to be used in the damping element in the
        muscle fiber model.

        The alias ``fiber_damping_coefficient`` can also be used to access the
        same attribute.

        """
        return self._beta

    @property
    def activation_dynamics(self):
        """Activation dynamics model governing this musculotendon's activation.

        Explanation
        ===========

        Returns the instance of a subclass of ``ActivationBase`` that governs
        the relationship between excitation and activation that is used to
        represent the activation dynamics of this musculotendon.

        """
        return self._activation_dynamics

    @property
    def excitation(self):
        """Dynamic symbol representing excitation.

        Explanation
        ===========

        The alias ``e`` can also be used to access the same attribute.

        """
        return self._activation_dynamics._e

    @property
    def e(self):
        """Dynamic symbol representing excitation.

        Explanation
        ===========

        The alias ``excitation`` can also be used to access the same attribute.

        """
        return self._activation_dynamics._e

    @property
    def activation(self):
        """Dynamic symbol representing activation.

        Explanation
        ===========

        The alias ``a`` can also be used to access the same attribute.

        """
        return self._activation_dynamics._a

    @property
    def a(self):
        """Dynamic symbol representing activation.

        Explanation
        ===========

        The alias ``activation`` can also be used to access the same attribute.

        """
        return self._activation_dynamics._a

    @property
    def musculotendon_dynamics(self):
        """The choice of rigid or type of elastic tendon musculotendon dynamics.

        Explanation
        ===========

        The formulation of musculotendon dynamics that should be used
        internally, i.e. rigid or elastic tendon model, the choice of
        musculotendon state etc. This must be a member of the integer
        enumeration ``MusculotendonFormulation`` or an integer that can be cast
        to a member. To use a rigid tendon formulation, set this to
        ``MusculotendonFormulation.RIGID_TENDON`` (or the integer value ``0``,
        which will be cast to the enumeration member). There are four possible
        formulations for an elastic tendon model. To use an explicit formulation
        with the fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_EXPLICIT`` (or the integer value
        ``1``). To use an explicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_EXPLICIT``
        (or the integer value ``2``). To use an implicit formulation with the
        fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_IMPLICIT`` (or the integer value
        ``3``). To use an implicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_IMPLICIT``
        (or the integer value ``4``). The default is
        ``MusculotendonFormulation.RIGID_TENDON``, which corresponds to a rigid
        tendon formulation.

        """
        return self._musculotendon_dynamics

    def _rigid_tendon_musculotendon_dynamics(self):
        """Rigid tendon musculotendon."""
        self._l_MT = self.pathway.length
        self._v_MT = self.pathway.extension_velocity
        self._l_T = self._l_T_slack
        self._l_T_tilde = Integer(1)
        self._l_M = sqrt((self._l_MT - self._l_T)**2 + (self._l_M_opt*sin(self._alpha_opt))**2)
        self._l_M_tilde = self._l_M/self._l_M_opt
        self._v_M = self._v_MT*(self._l_MT - self._l_T_slack)/self._l_M
        self._v_M_tilde = self._v_M/self._v_M_max
        if self._with_defaults:
            self._fl_T = self.curves.tendon_force_length.with_defaults(self._l_T_tilde)
            self._fl_M_pas = self.curves.fiber_force_length_passive.with_defaults(self._l_M_tilde)
            self._fl_M_act = self.curves.fiber_force_length_active.with_defaults(self._l_M_tilde)
            self._fv_M = self.curves.fiber_force_velocity.with_defaults(self._v_M_tilde)
        else:
            fl_T_constants = symbols(f'c_0:4_fl_T_{self.name}')
            self._fl_T = self.curves.tendon_force_length(self._l_T_tilde, *fl_T_constants)
            fl_M_pas_constants = symbols(f'c_0:2_fl_M_pas_{self.name}')
            self._fl_M_pas = self.curves.fiber_force_length_passive(self._l_M_tilde, *fl_M_pas_constants)
            fl_M_act_constants = symbols(f'c_0:12_fl_M_act_{self.name}')
            self._fl_M_act = self.curves.fiber_force_length_active(self._l_M_tilde, *fl_M_act_constants)
            fv_M_constants = symbols(f'c_0:4_fv_M_{self.name}')
            self._fv_M = self.curves.fiber_force_velocity(self._v_M_tilde, *fv_M_constants)
        self._F_M_tilde = self.a*self._fl_M_act*self._fv_M + self._fl_M_pas + self._beta*self._v_M_tilde
        self._F_T_tilde = self._F_M_tilde
        self._F_M = self._F_M_tilde*self._F_M_max
        self._cos_alpha = cos(self._alpha_opt)
        self._F_T = self._F_M*self._cos_alpha

        # Containers
        self._state_vars = zeros(0, 1)
        self._input_vars = zeros(0, 1)
        self._state_eqns = zeros(0, 1)
        self._curve_constants = Matrix(
            fl_T_constants
            + fl_M_pas_constants
            + fl_M_act_constants
            + fv_M_constants
        ) if not self._with_defaults else zeros(0, 1)

    def _fiber_length_explicit_musculotendon_dynamics(self):
        """Elastic tendon musculotendon using `l_M_tilde` as a state."""
        self._l_M_tilde = dynamicsymbols(f'l_M_tilde_{self.name}')
        self._l_MT = self.pathway.length
        self._v_MT = self.pathway.extension_velocity
        self._l_M = self._l_M_tilde*self._l_M_opt
        self._l_T = self._l_MT - sqrt(self._l_M**2 - (self._l_M_opt*sin(self._alpha_opt))**2)
        self._l_T_tilde = self._l_T/self._l_T_slack
        self._cos_alpha = (self._l_MT - self._l_T)/self._l_M
        if self._with_defaults:
            self._fl_T = self.curves.tendon_force_length.with_defaults(self._l_T_tilde)
            self._fl_M_pas = self.curves.fiber_force_length_passive.with_defaults(self._l_M_tilde)
            self._fl_M_act = self.curves.fiber_force_length_active.with_defaults(self._l_M_tilde)
        else:
            fl_T_constants = symbols(f'c_0:4_fl_T_{self.name}')
            self._fl_T = self.curves.tendon_force_length(self._l_T_tilde, *fl_T_constants)
            fl_M_pas_constants = symbols(f'c_0:2_fl_M_pas_{self.name}')
            self._fl_M_pas = self.curves.fiber_force_length_passive(self._l_M_tilde, *fl_M_pas_constants)
            fl_M_act_constants = symbols(f'c_0:12_fl_M_act_{self.name}')
            self._fl_M_act = self.curves.fiber_force_length_active(self._l_M_tilde, *fl_M_act_constants)
        self._F_T_tilde = self._fl_T
        self._F_T = self._F_T_tilde*self._F_M_max
        self._F_M = self._F_T/self._cos_alpha
        self._F_M_tilde = self._F_M/self._F_M_max
        self._fv_M = (self._F_M_tilde - self._fl_M_pas)/(self.a*self._fl_M_act)
        if self._with_defaults:
            self._v_M_tilde = self.curves.fiber_force_velocity_inverse.with_defaults(self._fv_M)
        else:
            fv_M_constants = symbols(f'c_0:4_fv_M_{self.name}')
            self._v_M_tilde = self.curves.fiber_force_velocity_inverse(self._fv_M, *fv_M_constants)
        self._dl_M_tilde_dt = (self._v_M_max/self._l_M_opt)*self._v_M_tilde

        self._state_vars = Matrix([self._l_M_tilde])
        self._input_vars = zeros(0, 1)
        self._state_eqns = Matrix([self._dl_M_tilde_dt])
        self._curve_constants = Matrix(
            fl_T_constants
            + fl_M_pas_constants
            + fl_M_act_constants
            + fv_M_constants
        ) if not self._with_defaults else zeros(0, 1)

    def _tendon_force_explicit_musculotendon_dynamics(self):
        """Elastic tendon musculotendon using `F_T_tilde` as a state."""
        self._F_T_tilde = dynamicsymbols(f'F_T_tilde_{self.name}')
        self._l_MT = self.pathway.length
        self._v_MT = self.pathway.extension_velocity
        self._fl_T = self._F_T_tilde
        if self._with_defaults:
            self._fl_T_inv = self.curves.tendon_force_length_inverse.with_defaults(self._fl_T)
        else:
            fl_T_constants = symbols(f'c_0:4_fl_T_{self.name}')
            self._fl_T_inv = self.curves.tendon_force_length_inverse(self._fl_T, *fl_T_constants)
        self._l_T_tilde = self._fl_T_inv
        self._l_T = self._l_T_tilde*self._l_T_slack
        self._l_M = sqrt((self._l_MT - self._l_T)**2 + (self._l_M_opt*sin(self._alpha_opt))**2)
        self._l_M_tilde = self._l_M/self._l_M_opt
        if self._with_defaults:
            self._fl_M_pas = self.curves.fiber_force_length_passive.with_defaults(self._l_M_tilde)
            self._fl_M_act = self.curves.fiber_force_length_active.with_defaults(self._l_M_tilde)
        else:
            fl_M_pas_constants = symbols(f'c_0:2_fl_M_pas_{self.name}')
            self._fl_M_pas = self.curves.fiber_force_length_passive(self._l_M_tilde, *fl_M_pas_constants)
            fl_M_act_constants = symbols(f'c_0:12_fl_M_act_{self.name}')
            self._fl_M_act = self.curves.fiber_force_length_active(self._l_M_tilde, *fl_M_act_constants)
        self._cos_alpha = (self._l_MT - self._l_T)/self._l_M
        self._F_T = self._F_T_tilde*self._F_M_max
        self._F_M = self._F_T/self._cos_alpha
        self._F_M_tilde = self._F_M/self._F_M_max
        self._fv_M = (self._F_M_tilde - self._fl_M_pas)/(self.a*self._fl_M_act)
        if self._with_defaults:
            self._fv_M_inv = self.curves.fiber_force_velocity_inverse.with_defaults(self._fv_M)
        else:
            fv_M_constants = symbols(f'c_0:4_fv_M_{self.name}')
            self._fv_M_inv = self.curves.fiber_force_velocity_inverse(self._fv_M, *fv_M_constants)
        self._v_M_tilde = self._fv_M_inv
        self._v_M = self._v_M_tilde*self._v_M_max
        self._v_T = self._v_MT - (self._v_M/self._cos_alpha)
        self._v_T_tilde = self._v_T/self._l_T_slack
        if self._with_defaults:
            self._fl_T = self.curves.tendon_force_length.with_defaults(self._l_T_tilde)
        else:
            self._fl_T = self.curves.tendon_force_length(self._l_T_tilde, *fl_T_constants)
        self._dF_T_tilde_dt = self._fl_T.diff(dynamicsymbols._t).subs({self._l_T_tilde.diff(dynamicsymbols._t): self._v_T_tilde})

        self._state_vars = Matrix([self._F_T_tilde])
        self._input_vars = zeros(0, 1)
        self._state_eqns = Matrix([self._dF_T_tilde_dt])
        self._curve_constants = Matrix(
            fl_T_constants
            + fl_M_pas_constants
            + fl_M_act_constants
            + fv_M_constants
        ) if not self._with_defaults else zeros(0, 1)

    def _fiber_length_implicit_musculotendon_dynamics(self):
        raise NotImplementedError

    def _tendon_force_implicit_musculotendon_dynamics(self):
        raise NotImplementedError

    @property
    def state_vars(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        The alias ``x`` can also be used to access the same attribute.

        """
        state_vars = [self._state_vars]
        for child in self._child_objects:
            state_vars.append(child.state_vars)
        return Matrix.vstack(*state_vars)

    @property
    def x(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        The alias ``state_vars`` can also be used to access the same attribute.

        """
        state_vars = [self._state_vars]
        for child in self._child_objects:
            state_vars.append(child.state_vars)
        return Matrix.vstack(*state_vars)

    @property
    def input_vars(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        The alias ``r`` can also be used to access the same attribute.

        """
        input_vars = [self._input_vars]
        for child in self._child_objects:
            input_vars.append(child.input_vars)
        return Matrix.vstack(*input_vars)

    @property
    def r(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        The alias ``input_vars`` can also be used to access the same attribute.

        """
        input_vars = [self._input_vars]
        for child in self._child_objects:
            input_vars.append(child.input_vars)
        return Matrix.vstack(*input_vars)

    @property
    def constants(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Explanation
        ===========

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        The alias ``p`` can also be used to access the same attribute.

        """
        musculotendon_constants = [
            self._l_T_slack,
            self._F_M_max,
            self._l_M_opt,
            self._v_M_max,
            self._alpha_opt,
            self._beta,
        ]
        musculotendon_constants = [
            c for c in musculotendon_constants if not c.is_number
        ]
        constants = [
            Matrix(musculotendon_constants)
            if musculotendon_constants
            else zeros(0, 1)
        ]
        for child in self._child_objects:
            constants.append(child.constants)
        constants.append(self._curve_constants)
        return Matrix.vstack(*constants)

    @property
    def p(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Explanation
        ===========

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        The alias ``constants`` can also be used to access the same attribute.

        """
        musculotendon_constants = [
            self._l_T_slack,
            self._F_M_max,
            self._l_M_opt,
            self._v_M_max,
            self._alpha_opt,
            self._beta,
        ]
        musculotendon_constants = [
            c for c in musculotendon_constants if not c.is_number
        ]
        constants = [
            Matrix(musculotendon_constants)
            if musculotendon_constants
            else zeros(0, 1)
        ]
        for child in self._child_objects:
            constants.append(child.constants)
        constants.append(self._curve_constants)
        return Matrix.vstack(*constants)

    @property
    def M(self):
        """Ordered square matrix of coefficients on the LHS of ``M x' = F``.

        Explanation
        ===========

        The square matrix that forms part of the LHS of the linear system of
        ordinary differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        As zeroth-order activation dynamics have no state variables, this
        linear system has dimension 0 and therefore ``M`` is an empty square
        ``Matrix`` with shape (0, 0).

        """
        M = [eye(len(self._state_vars))]
        for child in self._child_objects:
            M.append(child.M)
        return diag(*M)

    @property
    def F(self):
        """Ordered column matrix of equations on the RHS of ``M x' = F``.

        Explanation
        ===========

        The column matrix that forms the RHS of the linear system of ordinary
        differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        As zeroth-order activation dynamics have no state variables, this
        linear system has dimension 0 and therefore ``F`` is an empty column
        ``Matrix`` with shape (0, 1).

        """
        F = [self._state_eqns]
        for child in self._child_objects:
            F.append(child.F)
        return Matrix.vstack(*F)

    def rhs(self):
        """Ordered column matrix of equations for the solution of ``M x' = F``.

        Explanation
        ===========

        The solution to the linear system of ordinary differential equations
        governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        As zeroth-order activation dynamics have no state variables, this
        linear has dimension 0 and therefore this method returns an empty
        column ``Matrix`` with shape (0, 1).

        """
        is_explicit = (
            MusculotendonFormulation.FIBER_LENGTH_EXPLICIT,
            MusculotendonFormulation.TENDON_FORCE_EXPLICIT,
        )
        if self.musculotendon_dynamics is MusculotendonFormulation.RIGID_TENDON:
            child_rhs = [child.rhs() for child in self._child_objects]
            return Matrix.vstack(*child_rhs)
        elif self.musculotendon_dynamics in is_explicit:
            rhs = self._state_eqns
            child_rhs = [child.rhs() for child in self._child_objects]
            return Matrix.vstack(rhs, *child_rhs)
        return self.M.solve(self.F)

    def __repr__(self):
        """Returns a string representation to reinstantiate the model."""
        return (
            f'{self.__class__.__name__}({self.name!r}, '
            f'pathway={self.pathway!r}, '
            f'activation_dynamics={self.activation_dynamics!r}, '
            f'musculotendon_dynamics={self.musculotendon_dynamics}, '
            f'tendon_slack_length={self._l_T_slack!r}, '
            f'peak_isometric_force={self._F_M_max!r}, '
            f'optimal_fiber_length={self._l_M_opt!r}, '
            f'maximal_fiber_velocity={self._v_M_max!r}, '
            f'optimal_pennation_angle={self._alpha_opt!r}, '
            f'fiber_damping_coefficient={self._beta!r})'
        )

    def __str__(self):
        """Returns a string representation of the expression for musculotendon
        force."""
        return str(self.force)


class MusculotendonDeGroote2016(MusculotendonBase):
    r"""Musculotendon model using the curves of De Groote et al., 2016 [1]_.

    Examples
    ========

    This class models the musculotendon actuator parametrized by the
    characteristic curves described in De Groote et al., 2016 [1]_. Like all
    musculotendon models in SymPy's biomechanics module, it requires a pathway
    to define its line of action. We'll begin by creating a simple
    ``LinearPathway`` between two points that our musculotendon will follow.
    We'll create a point ``O`` to represent the musculotendon's origin and
    another ``I`` to represent its insertion.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (LinearPathway, Point,
    ...     ReferenceFrame, dynamicsymbols)

    >>> N = ReferenceFrame('N')
    >>> O, I = O, P = symbols('O, I', cls=Point)
    >>> q, u = dynamicsymbols('q, u', real=True)
    >>> I.set_pos(O, q*N.x)
    >>> O.set_vel(N, 0)
    >>> I.set_vel(N, u*N.x)
    >>> pathway = LinearPathway(O, I)
    >>> pathway.attachments
    (O, I)
    >>> pathway.length
    Abs(q(t))
    >>> pathway.extension_velocity
    sign(q(t))*Derivative(q(t), t)

    A musculotendon also takes an instance of an activation dynamics model as
    this will be used to provide symbols for the activation in the formulation
    of the musculotendon dynamics. We'll use an instance of
    ``FirstOrderActivationDeGroote2016`` to represent first-order activation
    dynamics. Note that a single name argument needs to be provided as SymPy
    will use this as a suffix.

    >>> from sympy.physics.biomechanics import FirstOrderActivationDeGroote2016

    >>> activation = FirstOrderActivationDeGroote2016('muscle')
    >>> activation.x
    Matrix([[a_muscle(t)]])
    >>> activation.r
    Matrix([[e_muscle(t)]])
    >>> activation.p
    Matrix([
    [tau_a_muscle],
    [tau_d_muscle],
    [    b_muscle]])
    >>> activation.rhs()
    Matrix([[((1/2 - tanh(b_muscle*(-a_muscle(t) + e_muscle(t)))/2)*(3*...]])

    The musculotendon class requires symbols or values to be passed to represent
    the constants in the musculotendon dynamics. We'll use SymPy's ``symbols``
    function to create symbols for the maximum isometric force ``F_M_max``,
    optimal fiber length ``l_M_opt``, tendon slack length ``l_T_slack``, maximum
    fiber velocity ``v_M_max``, optimal pennation angle ``alpha_opt, and fiber
    damping coefficient ``beta``.

    >>> F_M_max = symbols('F_M_max', real=True)
    >>> l_M_opt = symbols('l_M_opt', real=True)
    >>> l_T_slack = symbols('l_T_slack', real=True)
    >>> v_M_max = symbols('v_M_max', real=True)
    >>> alpha_opt = symbols('alpha_opt', real=True)
    >>> beta = symbols('beta', real=True)

    We can then import the class ``MusculotendonDeGroote2016`` from the
    biomechanics module and create an instance by passing in the various objects
    we have previously instantiated. By default, a musculotendon model with
    rigid tendon musculotendon dynamics will be created.

    >>> from sympy.physics.biomechanics import MusculotendonDeGroote2016

    >>> rigid_tendon_muscle = MusculotendonDeGroote2016(
    ...     'muscle',
    ...     pathway,
    ...     activation,
    ...     tendon_slack_length=l_T_slack,
    ...     peak_isometric_force=F_M_max,
    ...     optimal_fiber_length=l_M_opt,
    ...     maximal_fiber_velocity=v_M_max,
    ...     optimal_pennation_angle=alpha_opt,
    ...     fiber_damping_coefficient=beta,
    ... )

    We can inspect the various properties of the musculotendon, including
    getting the symbolic expression describing the force it produces using its
    ``force`` attribute.

    >>> rigid_tendon_muscle.force
    -F_M_max*(beta*(-l_T_slack + Abs(q(t)))*sign(q(t))*Derivative(q(t), t)...

    When we created the musculotendon object, we passed in an instance of an
    activation dynamics object that governs the activation within the
    musculotendon. SymPy makes a design choice here that the activation dynamics
    instance will be treated as a child object of the musculotendon dynamics.
    Therefore, if we want to inspect the state and input variables associated
    with the musculotendon model, we will also be returned the state and input
    variables associated with the child object, or the activation dynamics in
    this case. As the musculotendon model that we created here uses rigid tendon
    dynamics, no additional states or inputs relating to the musculotendon are
    introduces. Consequently, the model has a single state associated with it,
    the activation, and a single input associated with it, the excitation. The
    states and inputs can be inspected using the ``x`` and ``r`` attributes
    respectively. Note that both ``x`` and ``r`` have the alias attributes of
    ``state_vars`` and ``input_vars``.

    >>> rigid_tendon_muscle.x
    Matrix([[a_muscle(t)]])
    >>> rigid_tendon_muscle.r
    Matrix([[e_muscle(t)]])

    To see which constants are symbolic in the musculotendon model, we can use
    the ``p`` or ``constants`` attribute. This returns a ``Matrix`` populated
    by the constants that are represented by a ``Symbol`` rather than a numeric
    value.

    >>> rigid_tendon_muscle.p
    Matrix([
    [           l_T_slack],
    [             F_M_max],
    [             l_M_opt],
    [             v_M_max],
    [           alpha_opt],
    [                beta],
    [        tau_a_muscle],
    [        tau_d_muscle],
    [            b_muscle],
    [     c_0_fl_T_muscle],
    [     c_1_fl_T_muscle],
    [     c_2_fl_T_muscle],
    [     c_3_fl_T_muscle],
    [ c_0_fl_M_pas_muscle],
    [ c_1_fl_M_pas_muscle],
    [ c_0_fl_M_act_muscle],
    [ c_1_fl_M_act_muscle],
    [ c_2_fl_M_act_muscle],
    [ c_3_fl_M_act_muscle],
    [ c_4_fl_M_act_muscle],
    [ c_5_fl_M_act_muscle],
    [ c_6_fl_M_act_muscle],
    [ c_7_fl_M_act_muscle],
    [ c_8_fl_M_act_muscle],
    [ c_9_fl_M_act_muscle],
    [c_10_fl_M_act_muscle],
    [c_11_fl_M_act_muscle],
    [     c_0_fv_M_muscle],
    [     c_1_fv_M_muscle],
    [     c_2_fv_M_muscle],
    [     c_3_fv_M_muscle]])

    Finally, we can call the ``rhs`` method to return a ``Matrix`` that
    contains as its elements the righthand side of the ordinary differential
    equations corresponding to each of the musculotendon's states. Like the
    method with the same name on the ``Method`` classes in SymPy's mechanics
    module, this returns a column vector where the number of rows corresponds to
    the number of states. For our example here, we have a single state, the
    dynamic symbol ``a_muscle(t)``, so the returned value is a 1-by-1
    ``Matrix``.

    >>> rigid_tendon_muscle.rhs()
    Matrix([[((1/2 - tanh(b_muscle*(-a_muscle(t) + e_muscle(t)))/2)*(3*...]])

    The musculotendon class supports elastic tendon musculotendon models in
    addition to rigid tendon ones. You can choose to either use the fiber length
    or tendon force as an additional state. You can also specify whether an
    explicit or implicit formulation should be used. To select a formulation,
    pass a member of the ``MusculotendonFormulation`` enumeration to the
    ``musculotendon_dynamics`` parameter when calling the constructor. This
    enumeration is an ``IntEnum``, so you can also pass an integer, however it
    is recommended to use the enumeration as it is clearer which formulation you
    are actually selecting. Below, we'll use the ``FIBER_LENGTH_EXPLICIT``
    member to create a musculotendon with an elastic tendon that will use the
    (normalized) muscle fiber length as an additional state and will produce
    the governing ordinary differential equation in explicit form.

    >>> from sympy.physics.biomechanics import MusculotendonFormulation

    >>> elastic_tendon_muscle = MusculotendonDeGroote2016(
    ...     'muscle',
    ...     pathway,
    ...     activation,
    ...     musculotendon_dynamics=MusculotendonFormulation.FIBER_LENGTH_EXPLICIT,
    ...     tendon_slack_length=l_T_slack,
    ...     peak_isometric_force=F_M_max,
    ...     optimal_fiber_length=l_M_opt,
    ...     maximal_fiber_velocity=v_M_max,
    ...     optimal_pennation_angle=alpha_opt,
    ...     fiber_damping_coefficient=beta,
    ... )

    >>> elastic_tendon_muscle.force
    -F_M_max*TendonForceLengthDeGroote2016((-sqrt(l_M_opt**2*...
    >>> elastic_tendon_muscle.x
    Matrix([
    [l_M_tilde_muscle(t)],
    [        a_muscle(t)]])
    >>> elastic_tendon_muscle.r
    Matrix([[e_muscle(t)]])
    >>> elastic_tendon_muscle.p
    Matrix([
    [           l_T_slack],
    [             F_M_max],
    [             l_M_opt],
    [             v_M_max],
    [           alpha_opt],
    [                beta],
    [        tau_a_muscle],
    [        tau_d_muscle],
    [            b_muscle],
    [     c_0_fl_T_muscle],
    [     c_1_fl_T_muscle],
    [     c_2_fl_T_muscle],
    [     c_3_fl_T_muscle],
    [ c_0_fl_M_pas_muscle],
    [ c_1_fl_M_pas_muscle],
    [ c_0_fl_M_act_muscle],
    [ c_1_fl_M_act_muscle],
    [ c_2_fl_M_act_muscle],
    [ c_3_fl_M_act_muscle],
    [ c_4_fl_M_act_muscle],
    [ c_5_fl_M_act_muscle],
    [ c_6_fl_M_act_muscle],
    [ c_7_fl_M_act_muscle],
    [ c_8_fl_M_act_muscle],
    [ c_9_fl_M_act_muscle],
    [c_10_fl_M_act_muscle],
    [c_11_fl_M_act_muscle],
    [     c_0_fv_M_muscle],
    [     c_1_fv_M_muscle],
    [     c_2_fv_M_muscle],
    [     c_3_fv_M_muscle]])
    >>> elastic_tendon_muscle.rhs()
    Matrix([
    [v_M_max*FiberForceVelocityInverseDeGroote2016((l_M_opt*...],
    [ ((1/2 - tanh(b_muscle*(-a_muscle(t) + e_muscle(t)))/2)*(3*...]])

    It is strongly recommended to use the alternate ``with_defaults``
    constructor when creating an instance because this will ensure that the
    published constants are used in the musculotendon characteristic curves.

    >>> elastic_tendon_muscle = MusculotendonDeGroote2016.with_defaults(
    ...     'muscle',
    ...     pathway,
    ...     activation,
    ...     musculotendon_dynamics=MusculotendonFormulation.FIBER_LENGTH_EXPLICIT,
    ...     tendon_slack_length=l_T_slack,
    ...     peak_isometric_force=F_M_max,
    ...     optimal_fiber_length=l_M_opt,
    ... )

    >>> elastic_tendon_muscle.x
    Matrix([
    [l_M_tilde_muscle(t)],
    [        a_muscle(t)]])
    >>> elastic_tendon_muscle.r
    Matrix([[e_muscle(t)]])
    >>> elastic_tendon_muscle.p
    Matrix([
    [   l_T_slack],
    [     F_M_max],
    [     l_M_opt],
    [tau_a_muscle],
    [tau_d_muscle],
    [    b_muscle]])

    Parameters
    ==========

    name : str
        The name identifier associated with the musculotendon. This name is used
        as a suffix when automatically generated symbols are instantiated. It
        must be a string of nonzero length.
    pathway : PathwayBase
        The pathway that the actuator follows. This must be an instance of a
        concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.
    activation_dynamics : ActivationBase
        The activation dynamics that will be modeled within the musculotendon.
        This must be an instance of a concrete subclass of ``ActivationBase``,
        e.g. ``FirstOrderActivationDeGroote2016``.
    musculotendon_dynamics : MusculotendonFormulation | int
        The formulation of musculotendon dynamics that should be used
        internally, i.e. rigid or elastic tendon model, the choice of
        musculotendon state etc. This must be a member of the integer
        enumeration ``MusculotendonFormulation`` or an integer that can be cast
        to a member. To use a rigid tendon formulation, set this to
        ``MusculotendonFormulation.RIGID_TENDON`` (or the integer value ``0``,
        which will be cast to the enumeration member). There are four possible
        formulations for an elastic tendon model. To use an explicit formulation
        with the fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_EXPLICIT`` (or the integer value
        ``1``). To use an explicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_EXPLICIT``
        (or the integer value ``2``). To use an implicit formulation with the
        fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_IMPLICIT`` (or the integer value
        ``3``). To use an implicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_IMPLICIT``
        (or the integer value ``4``). The default is
        ``MusculotendonFormulation.RIGID_TENDON``, which corresponds to a rigid
        tendon formulation.
    tendon_slack_length : Expr | None
        The length of the tendon when the musculotendon is in its unloaded
        state. In a rigid tendon model the tendon length is the tendon slack
        length. In all musculotendon models, tendon slack length is used to
        normalize tendon length to give
        :math:`\tilde{l}^T = \frac{l^T}{l^T_{slack}}`.
    peak_isometric_force : Expr | None
        The maximum force that the muscle fiber can produce when it is
        undergoing an isometric contraction (no lengthening velocity). In all
        musculotendon models, peak isometric force is used to normalized tendon
        and muscle fiber force to give
        :math:`\tilde{F}^T = \frac{F^T}{F^M_{max}}`.
    optimal_fiber_length : Expr | None
        The muscle fiber length at which the muscle fibers produce no passive
        force and their maximum active force. In all musculotendon models,
        optimal fiber length is used to normalize muscle fiber length to give
        :math:`\tilde{l}^M = \frac{l^M}{l^M_{opt}}`.
    maximal_fiber_velocity : Expr | None
        The fiber velocity at which, during muscle fiber shortening, the muscle
        fibers are unable to produce any active force. In all musculotendon
        models, maximal fiber velocity is used to normalize muscle fiber
        extension velocity to give :math:`\tilde{v}^M = \frac{v^M}{v^M_{max}}`.
    optimal_pennation_angle : Expr | None
        The pennation angle when muscle fiber length equals the optimal fiber
        length.
    fiber_damping_coefficient : Expr | None
        The coefficient of damping to be used in the damping element in the
        muscle fiber model.
    with_defaults : bool
        Whether ``with_defaults`` alternate constructors should be used when
        automatically constructing child classes. Default is ``False``.

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    curves = CharacteristicCurveCollection(
        tendon_force_length=TendonForceLengthDeGroote2016,
        tendon_force_length_inverse=TendonForceLengthInverseDeGroote2016,
        fiber_force_length_passive=FiberForceLengthPassiveDeGroote2016,
        fiber_force_length_passive_inverse=FiberForceLengthPassiveInverseDeGroote2016,
        fiber_force_length_active=FiberForceLengthActiveDeGroote2016,
        fiber_force_velocity=FiberForceVelocityDeGroote2016,
        fiber_force_velocity_inverse=FiberForceVelocityInverseDeGroote2016,
    )
