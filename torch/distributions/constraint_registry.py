r"""
PyTorch provides two global :class:`ConstraintRegistry` objects that link
:class:`~torch.distributions.constraints.Constraint` objects to
:class:`~torch.distributions.transforms.Transform` objects. These objects both
input constraints and return transforms, but they have different guarantees on
bijectivity.

1. ``biject_to(constraint)`` looks up a bijective
   :class:`~torch.distributions.transforms.Transform` from ``constraints.real``
   to the given ``constraint``. The returned transform is guaranteed to have
   ``.bijective = True`` and should implement ``.log_abs_det_jacobian()``.
2. ``transform_to(constraint)`` looks up a not-necessarily bijective
   :class:`~torch.distributions.transforms.Transform` from ``constraints.real``
   to the given ``constraint``. The returned transform is not guaranteed to
   implement ``.log_abs_det_jacobian()``.

The ``transform_to()`` registry is useful for performing unconstrained
optimization on constrained parameters of probability distributions, which are
indicated by each distribution's ``.arg_constraints`` dict. These transforms often
overparameterize a space in order to avoid rotation; they are thus more
suitable for coordinate-wise optimization algorithms like Adam::

    loc = torch.zeros(100, requires_grad=True)
    unconstrained = torch.zeros(100, requires_grad=True)
    scale = transform_to(Normal.arg_constraints["scale"])(unconstrained)
    loss = -Normal(loc, scale).log_prob(data).sum()

The ``biject_to()`` registry is useful for Hamiltonian Monte Carlo, where
samples from a probability distribution with constrained ``.support`` are
propagated in an unconstrained space, and algorithms are typically rotation
invariant.::

    dist = Exponential(rate)
    unconstrained = torch.zeros(100, requires_grad=True)
    sample = biject_to(dist.support)(unconstrained)
    potential_energy = -dist.log_prob(sample).sum()

.. note::

    An example where ``transform_to`` and ``biject_to`` differ is
    ``constraints.simplex``: ``transform_to(constraints.simplex)`` returns a
    :class:`~torch.distributions.transforms.SoftmaxTransform` that simply
    exponentiates and normalizes its inputs; this is a cheap and mostly
    coordinate-wise operation appropriate for algorithms like SVI. In
    contrast, ``biject_to(constraints.simplex)`` returns a
    :class:`~torch.distributions.transforms.StickBreakingTransform` that
    bijects its input down to a one-fewer-dimensional space; this a more
    expensive less numerically stable transform but is needed for algorithms
    like HMC.

The ``biject_to`` and ``transform_to`` objects can be extended by user-defined
constraints and transforms using their ``.register()`` method either as a
function on singleton constraints::

    transform_to.register(my_constraint, my_transform)

or as a decorator on parameterized constraints::

    @transform_to.register(MyConstraintClass)
    def my_factory(constraint):
        assert isinstance(constraint, MyConstraintClass)
        return MyTransform(constraint.param1, constraint.param2)

You can create your own registry by creating a new :class:`ConstraintRegistry`
object.
"""

from typing import Callable, Optional, overload, Union
from typing_extensions import Never, TypeAlias, TypeVar

from torch.distributions import constraints, transforms
from torch.distributions.constraints import (
    Cat,
    Constraint,
    CorrCholesky,
    GreaterThan,
    GreaterThanEq,
    HalfOpenInterval,
    Independent,
    Interval,
    LessThan,
    LowerCholesky,
    NonNegative,
    Positive,
    PositiveDefinite,
    PositiveSemidefinite,
    Real,
    Simplex,
    Stack,
)
from torch.distributions.transforms import Transform
from torch.types import _Number


__all__ = [
    "ConstraintRegistry",
    "biject_to",
    "transform_to",
]

Con = TypeVar("Con", bound=Constraint)
Factory: TypeAlias = Callable[[Con], Transform]
# Note: Technically, `F` should be lower-bounded by `Con`, but higher-kinded
#    type-variables are not supported at the time of writing this.
F = TypeVar("F", bound=Callable[[Never], Transform])


class ConstraintRegistry:
    """
    Registry to link constraints to transforms.
    """

    def __init__(self) -> None:
        self._registry: dict[type[Constraint], Factory] = {}
        super().__init__()

    @overload
    def register(
        self,
        constraint: Union[Con, type[Con]],
        factory: F,
    ) -> F: ...

    @overload  # decorator usage
    def register(
        self,
        constraint: Union[Con, type[Con]],
        factory: None = ...,
    ) -> Callable[[F], F]: ...

    def register(
        self,
        constraint: Union[Con, type[Con]],
        factory: Optional[F] = None,
    ) -> Union[F, Callable[[F], F]]:
        """
        Registers a :class:`~torch.distributions.constraints.Constraint`
        subclass in this registry. Usage::

            @my_registry.register(MyConstraintClass)
            def construct_transform(constraint):
                assert isinstance(constraint, MyConstraint)
                return MyTransform(constraint.arg_constraints)

        Args:
            constraint (subclass of :class:`~torch.distributions.constraints.Constraint`):
                A subclass of :class:`~torch.distributions.constraints.Constraint`, or
                a singleton object of the desired class.
            factory (Callable): A callable that inputs a constraint object and returns
                a  :class:`~torch.distributions.transforms.Transform` object.
        """
        # Support use as decorator.
        if factory is None:
            return lambda fac: self.register(constraint, fac)

        # Support calling on singleton instances.
        if isinstance(constraint, Constraint):
            constraint = type(constraint)  # type: ignore[assignment]

        if not isinstance(constraint, type) or not issubclass(constraint, Constraint):
            raise TypeError(
                f"Expected constraint to be either a Constraint subclass or instance, but got {constraint}"
            )

        self._registry[constraint] = factory
        return factory

    def __call__(self, constraint: Constraint) -> Transform:
        """
        Looks up a transform to constrained space, given a constraint object.
        Usage::

            constraint = Normal.arg_constraints["scale"]
            scale = transform_to(constraint)(torch.zeros(1))  # constrained
            u = transform_to(constraint).inv(scale)  # unconstrained

        Args:
            constraint (:class:`~torch.distributions.constraints.Constraint`):
                A constraint object.

        Returns:
            A :class:`~torch.distributions.transforms.Transform` object.

        Raises:
            `NotImplementedError` if no transform has been registered.
        """
        # Look up by Constraint subclass.
        try:
            factory = self._registry[type(constraint)]
        except KeyError:
            raise NotImplementedError(
                f"Cannot transform {type(constraint).__name__} constraints"
            ) from None
        return factory(constraint)


biject_to = ConstraintRegistry()
transform_to = ConstraintRegistry()


################################################################################
# Registration Table
################################################################################


@biject_to.register(constraints.real)
@transform_to.register(constraints.real)
def _transform_to_real(constraint: Real) -> Transform:
    return transforms.identity_transform


@biject_to.register(constraints.independent)
def _biject_to_independent(constraint: Independent) -> Transform:
    base_transform = biject_to(constraint.base_constraint)
    return transforms.IndependentTransform(
        base_transform, constraint.reinterpreted_batch_ndims
    )


@transform_to.register(constraints.independent)
def _transform_to_independent(constraint: Independent) -> Transform:
    base_transform = transform_to(constraint.base_constraint)
    return transforms.IndependentTransform(
        base_transform, constraint.reinterpreted_batch_ndims
    )


@biject_to.register(constraints.positive)
@biject_to.register(constraints.nonnegative)
@transform_to.register(constraints.positive)
@transform_to.register(constraints.nonnegative)
def _transform_to_positive(constraint: Union[Positive, NonNegative]) -> Transform:
    return transforms.ExpTransform()


@biject_to.register(constraints.greater_than)
@biject_to.register(constraints.greater_than_eq)
@transform_to.register(constraints.greater_than)
@transform_to.register(constraints.greater_than_eq)
def _transform_to_greater_than(
    constraint: Union[GreaterThan, GreaterThanEq],
) -> Transform:
    return transforms.ComposeTransform(
        [
            transforms.ExpTransform(),
            transforms.AffineTransform(constraint.lower_bound, 1),
        ]
    )


@biject_to.register(constraints.less_than)
@transform_to.register(constraints.less_than)
def _transform_to_less_than(constraint: LessThan) -> Transform:
    return transforms.ComposeTransform(
        [
            transforms.ExpTransform(),
            transforms.AffineTransform(constraint.upper_bound, -1),
        ]
    )


@biject_to.register(constraints.interval)
@biject_to.register(constraints.half_open_interval)
@transform_to.register(constraints.interval)
@transform_to.register(constraints.half_open_interval)
def _transform_to_interval(constraint: Union[Interval, HalfOpenInterval]) -> Transform:
    # Handle the special case of the unit interval.
    lower_is_0 = (
        isinstance(constraint.lower_bound, _Number) and constraint.lower_bound == 0
    )
    upper_is_1 = (
        isinstance(constraint.upper_bound, _Number) and constraint.upper_bound == 1
    )
    if lower_is_0 and upper_is_1:
        return transforms.SigmoidTransform()

    loc = constraint.lower_bound
    scale = constraint.upper_bound - constraint.lower_bound
    return transforms.ComposeTransform(
        [transforms.SigmoidTransform(), transforms.AffineTransform(loc, scale)]
    )


@biject_to.register(constraints.simplex)
def _biject_to_simplex(constraint: Simplex) -> Transform:
    return transforms.StickBreakingTransform()


@transform_to.register(constraints.simplex)
def _transform_to_simplex(constraint: Simplex) -> Transform:
    return transforms.SoftmaxTransform()


# TODO define a bijection for LowerCholeskyTransform
@transform_to.register(constraints.lower_cholesky)
def _transform_to_lower_cholesky(constraint: LowerCholesky) -> Transform:
    return transforms.LowerCholeskyTransform()


@transform_to.register(constraints.positive_definite)
@transform_to.register(constraints.positive_semidefinite)
def _transform_to_positive_definite(
    constraint: Union[PositiveDefinite, PositiveSemidefinite],
) -> Transform:
    return transforms.PositiveDefiniteTransform()


@biject_to.register(constraints.corr_cholesky)
@transform_to.register(constraints.corr_cholesky)
def _transform_to_corr_cholesky(constraint: CorrCholesky) -> Transform:
    return transforms.CorrCholeskyTransform()


@biject_to.register(constraints.cat)
def _biject_to_cat(constraint: Cat) -> Transform:
    return transforms.CatTransform(
        [biject_to(c) for c in constraint.cseq], constraint.dim, constraint.lengths
    )


@transform_to.register(constraints.cat)
def _transform_to_cat(constraint: Cat) -> Transform:
    return transforms.CatTransform(
        [transform_to(c) for c in constraint.cseq], constraint.dim, constraint.lengths
    )


@biject_to.register(constraints.stack)
def _biject_to_stack(constraint: Stack) -> Transform:
    return transforms.StackTransform(
        [biject_to(c) for c in constraint.cseq], constraint.dim
    )


@transform_to.register(constraints.stack)
def _transform_to_stack(constraint: Stack) -> Transform:
    return transforms.StackTransform(
        [transform_to(c) for c in constraint.cseq], constraint.dim
    )
