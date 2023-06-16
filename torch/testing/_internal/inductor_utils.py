from subprocess import CalledProcessError

from torch._inductor.codecache import CppCodeCache
from torch._inductor.utils import has_triton
from torch.testing._internal.common_utils import (
    LazyVal,
    IS_FBCODE,
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

HAS_CPU = LazyVal(test_cpu)

HAS_CUDA = has_triton()
