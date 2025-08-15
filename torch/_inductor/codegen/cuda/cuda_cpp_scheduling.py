# mypy: allow-untyped-defs
from ..cutlass.scheduling import CUTLASSScheduling


class CUDACPPScheduling(CUTLASSScheduling):
    """
    Partial Scheduling implementation for CUDA C++ Kernels.
    This class is intended to be used in combination with TritonScheduling,
    and delegated to by CombinedScheduling.

    It handles fusion decisions and CUDA C++ specific template code generation.
    """
