# Owner(s): ["module: inductor"]
import math
import sys
import unittest

import torch
import torch._dynamo.config as dynamo_config
import torch.backends.cuda
import torch.nn.functional as F
from torch import nn
from torch._dynamo.debug_utils import same_two_models
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import same
from torch._inductor import config
from torch._inductor.compile_fx import compile_fx_inner
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    freeze_rng_state,
    IS_FBCODE,
    skipIfRocm,
    TEST_WITH_ASAN,
)

try:
    try:
        import triton
        from triton import language as tl
    except ImportError:
        raise unittest.SkipTest("requires triton")  # noqa: TRY200

    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


TestCase = test_torchinductor.TestCase
ToTuple = test_torchinductor.ToTuple
check_model_cuda = test_torchinductor.check_model_cuda
aten = torch.ops.aten


class StorageOffsetTests(TestCase):
    def test_symbolic_storage_offset(self):
        # Make sure that nn.Parameters with unaligned storage_offset works
        device = "cuda"
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.x = torch.nn.Parameter(
                    torch.rand((20,), device=device).as_strided((4, 4), (4, 1), 3)
                )

            def forward(self, x):
                return self.x + x

        mod = MyModule()

        x = torch.rand((4, 4), device=device)
        expected = mod(x)
        opt_mod = torch.compile(mod)
        actual = opt_mod(x)
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests
    from torch.testing._internal.inductor_utils import HAS_CUDA

    if HAS_CUDA and not TEST_WITH_ASAN:
        run_tests(needs="filelock")
