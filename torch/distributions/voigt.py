from typing import Any, Dict, Optional

from torch import Size
from torch.distributions.constraints import Constraint
from torch.distributions.distribution import Distribution


class Voigt(Distribution):
    @property
    def arg_constraints(self) -> Dict[str, Constraint]:
        return {}

    @property
    def mean(self):
        return 0.0

    @property
    def mode(self):
        return 0.0

    @property
    def support(self) -> Optional[Any]:
        return 0.0

    @property
    def variance(self):
        return 0.0

    def cdf(self, value):
        pass

    def entropy(self):
        pass

    def enumerate_support(self, expand=True):
        pass

    def expand(self, batch_shape, _instance=None):
        pass

    def icdf(self, value):
        pass

    def log_prob(self, value):
        pass

    def rsample(self, sample_shape=Size()):
        pass
