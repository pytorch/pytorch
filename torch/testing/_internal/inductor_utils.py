from subprocess import CalledProcessError

from torch._inductor.codecache import CppCodeCache
from torch._inductor.utils import has_triton
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    TEST_WITH_ROCM,
)
import torch

HAS_CPU = False
try:
    CppCodeCache.load("")
    HAS_CPU = not IS_FBCODE
except (
    CalledProcessError,
    OSError,
    torch._inductor.exc.InvalidCxxCompiler,
    torch._inductor.exc.CppCompileError,
):
    pass

HAS_CUDA = has_triton() and not TEST_WITH_ROCM
