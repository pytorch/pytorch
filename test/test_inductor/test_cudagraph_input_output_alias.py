import torch
import torch._dynamo
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_cuda import SM70OrLater
from torch.testing._internal.inductor_utils import skipCUDAIf
from torch._inductor.test_case import TestCase


class TestCUDAGraphInputOutputAlias(TestCase):
    @skipCUDAIf(not SM70OrLater, "Requires sm70")
    def test_reused_input_as_output(self):
        x = torch.randn(10, 10, device='cuda')

        def fn(x):
            return x  # just returns input directly (shared input/output)

        compiled_fn = torch.compile(fn, fullgraph=True, backend='inductor')

        for _ in range(5):
            out = compiled_fn(x)
            self.assertTrue(torch.allclose(out, x))

    @skipCUDAIf(not SM70OrLater, "Requires sm70")
    def test_memory_does_not_leak(self):
        torch.cuda.empty_cache()
        x = torch.randn(1024, 1024, device='cuda')

        def fn(x):
            return x + 1

        compiled_fn = torch.compile(fn, fullgraph=True, backend='inductor')
        torch.cuda.reset_peak_memory_stats()

        for _ in range(50):
            _ = compiled_fn(x)

        peak = torch.cuda.max_memory_reserved()
        self.assertLess(peak, 100 * 1024 * 1024)  # <100MB

if __name__ == "__main__":
    run_tests()
