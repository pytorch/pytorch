# Importing registers CuTeDSL pointwise overrides.

from ... import cutedsl_utils
from ...cutedsl.config import POINTWISE_ARCH_MAJORS
from .overrides import register_pointwise_overrides


cutedsl_utils.register_cuda_overrides(
    register_pointwise_overrides, POINTWISE_ARCH_MAJORS
)
