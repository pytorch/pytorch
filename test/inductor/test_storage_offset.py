# Owner(s): ["module: inductor"]
import sys
import unittest

import torch
import torch.backends.cuda
from torch.testing._internal.common_utils import TEST_WITH_ASAN
from torch.testing._internal.inductor_utils import HAS_CUDA

try:
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
    @unittest.skipIf(not HAS_CUDA, "Requires CUDA")
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

    if HAS_CUDA and not TEST_WITH_ASAN:
        run_tests(needs="filelock")
