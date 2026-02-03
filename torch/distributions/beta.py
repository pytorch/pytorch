# mypy: allow-untyped-defs

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["Beta"]

# Threshold below which we use specialized sampling for numerical stability.
# When both concentration parameters are below this threshold, the standard
# Gamma-based sampling becomes numerically unstable due to floating point
# underflow. See https://github.com/pytorch/pytorch/issues/136532
_SMALL_CONCENTRATION_THRESHOLD = 0.1


class Beta(ExponentialFamily):
    r"""
    Beta distribution parameterized by :attr:`concentration1` and :attr:`concentration0`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Beta(torch.tensor([0.5]), torch.tensor([0.5]))
        >>> m.sample()  # Beta distributed with concentration concentration1 and concentration0
        tensor([ 0.1046])

    Args:
        concentration1 (float or Tensor): 1st concentration parameter of the distribution
            (often referred to as alpha)
        concentration0 (float or Tensor): 2nd concentration parameter of the distribution
            (often referred to as beta)
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
    }
    support = constraints.unit_interval
    has_rsample = True

    def __init__(
        self,
        concentration1: Tensor | float,
        concentration0: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        if isinstance(concentration1, _Number) and isinstance(concentration0, _Number):
            concentration1_concentration0 = torch.tensor(
                [float(concentration1), float(concentration0)]
            )
        else:
            concentration1, concentration0 = broadcast_all(
                concentration1, concentration0
            )
            concentration1_concentration0 = torch.stack(
                [concentration1, concentration0], -1
            )
        self._dirichlet = Dirichlet(
            concentration1_concentration0, validate_args=validate_args
        )
        super().__init__(self._dirichlet._batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Beta, _instance)
        batch_shape = torch.Size(batch_shape)
        new._dirichlet = self._dirichlet.expand(batch_shape)
        super(Beta, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        return self.concentration1 / (self.concentration1 + self.concentration0)

    @property
    def mode(self) -> Tensor:
        return self._dirichlet.mode[..., 0]

    @property
    def variance(self) -> Tensor:
        total = self.concentration1 + self.concentration0
        return self.concentration1 * self.concentration0 / (total.pow(2) * (total + 1))

    def rsample(self, sample_shape: _size = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        c1 = self.concentration1.expand(shape)
        c0 = self.concentration0.expand(shape)

        # Check if all concentrations are small (below threshold)
        # In this case, use specialized sampling to avoid numerical instability
        # from Gamma-based sampling which underflows for small concentrations.
        use_small_conc_sampling = (
            (c1 < _SMALL_CONCENTRATION_THRESHOLD).all()
            and (c0 < _SMALL_CONCENTRATION_THRESHOLD).all()
        )

        if use_small_conc_sampling:
            return self._rsample_small_concentration(shape, c1, c0)
        else:
            return self._dirichlet.rsample(sample_shape).select(-1, 0)

    def _rsample_small_concentration(
        self, shape: torch.Size, c1: Tensor, c0: Tensor
    ) -> Tensor:
        """
        Sample from Beta distribution using Johnk's algorithm for small concentrations.

        This algorithm is numerically stable for concentration parameters < 1 where
        the standard Gamma-based approach suffers from floating point underflow.

        Johnk's algorithm (rejection sampling):
        1. Generate U, V ~ Uniform(0, 1)
        2. X = U^(1/a), Y = V^(1/b)
        3. If X + Y <= 1 and X + Y > 0, accept and return X / (X + Y)
        4. Otherwise, reject and repeat

        See: https://github.com/pytorch/pytorch/issues/136532
        """
        dtype = c1.dtype
        device = c1.device

        # Initialize result tensor
        result = torch.empty(shape, dtype=dtype, device=device)

        # Track which samples have been accepted
        accepted = torch.zeros(shape, dtype=torch.bool, device=device)

        # Precompute reciprocals for power operation
        inv_c1 = 1.0 / c1
        inv_c0 = 1.0 / c0

        # Maximum iterations to prevent infinite loops (very unlikely to be reached)
        max_iterations = 1000
        iteration = 0

        while not accepted.all() and iteration < max_iterations:
            # Generate uniform samples for positions that haven't been accepted
            mask = ~accepted
            n_remaining = mask.sum().item()

            if n_remaining == 0:
                break

            # Generate uniform samples
            u = torch.rand(shape, dtype=dtype, device=device)
            v = torch.rand(shape, dtype=dtype, device=device)

            # Transform: X = U^(1/a), Y = V^(1/b)
            x = u.pow(inv_c1)
            y = v.pow(inv_c0)

            # Accept if X + Y <= 1 and X + Y > 0 (avoid division by zero)
            sum_xy = x + y
            accept_condition = (sum_xy <= 1.0) & (sum_xy > 0.0) & mask

            # Store accepted samples: result = X / (X + Y)
            result[accept_condition] = x[accept_condition] / sum_xy[accept_condition]

            # Update accepted mask
            accepted = accepted | accept_condition
            iteration += 1

        # For any remaining unaccepted samples (very rare), fall back to
        # clamped values at the distribution mode
        if not accepted.all():
            # For unaccepted samples, use the mode as a fallback
            # Mode of Beta(a, b) when a, b < 1 is either 0 or 1 (at boundaries)
            # Use a simple heuristic: closer to 0 if a < b, closer to 1 if a > b
            unaccepted = ~accepted
            # Use tiny values instead of exact 0 or 1 to stay in open interval
            tiny = torch.finfo(dtype).tiny
            result[unaccepted & (c1 <= c0)] = tiny
            result[unaccepted & (c1 > c0)] = 1.0 - tiny

        return result

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        heads_tails = torch.stack([value, 1.0 - value], -1)
        return self._dirichlet.log_prob(heads_tails)

    def entropy(self):
        return self._dirichlet.entropy()

    @property
    def concentration1(self) -> Tensor:
        result = self._dirichlet.concentration[..., 0]
        if isinstance(result, _Number):
            return torch.tensor([result])
        else:
            return result

    @property
    def concentration0(self) -> Tensor:
        result = self._dirichlet.concentration[..., 1]
        if isinstance(result, _Number):
            return torch.tensor([result])
        else:
            return result

    @property
    def _natural_params(self) -> tuple[Tensor, Tensor]:
        return (self.concentration1, self.concentration0)

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x, y):
        return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)
