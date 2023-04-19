from subprocess import CalledProcessError

from torch._inductor.codecache import CppCodeCache
from torch._inductor.utils import has_triton
from torch.testing._internal.common_utils import (
    LazyBoolean,
    IS_FBCODE,
    TEST_WITH_ROCM,
)
import torch

def test_cpu():
    try:
        CppCodeCache.load("")
        return not IS_FBCODE
    except (
        CalledProcessError,
        OSError,
        torch._inductor.exc.InvalidCxxCompiler,
        torch._inductor.exc.CppCompileError,
    ):
        return False

HAS_CPU = LazyBoolean(test_cpu)

HAS_CUDA = has_triton() and not TEST_WITH_ROCM
