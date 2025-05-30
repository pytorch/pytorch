from typing_extensions import assert_type

import torch
from torch import distributions, Tensor
from torch.distributions import Pareto
from torch.distributions.constraints import Constraint, GreaterThanEq


dist = distributions.Normal(0, 1)
assert_type(dist.mean, Tensor)

dist = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
assert_type(dist.covariance_matrix, Tensor)


def _check_transformed_distributions(td: distributions.TransformedDistribution) -> None:
    assert_type(td.support, Constraint)


def _check_pareto_distribution(p: Pareto) -> None:
    assert_type(p.support, GreaterThanEq)
