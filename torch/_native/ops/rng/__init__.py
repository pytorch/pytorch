# Importing registers CuTeDSL uniform_ and normal_ overrides.

from ... import cutedsl_utils
from ...cutedsl.config import POINTWISE_ARCH_MAJORS
from .impl import register_rng_overrides


cutedsl_utils.register_cuda_overrides(register_rng_overrides, POINTWISE_ARCH_MAJORS)
