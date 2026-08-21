# Owner(s): ["module: inductor", "module: optimizer"]

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class SymmetricMMTest(TestCase):
    @parametrize("shape", [(4096, 4096), (5120, 8192)])
    @parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_quack_symmetric_mm(self, device, shape, dtype):
        if torch.cuda.get_device_capability(device)[0] not in (10, 11):
            self.skipTest("requires SM100 or SM110")

        def fn(x):
            return x @ x.T

        from torch._vendor.quack.gemm_symmetric import _AUTOTUNE_CACHE

        x = torch.randn(shape, device=device, dtype=dtype)
        _AUTOTUNE_CACHE.clear()
        torch._dynamo.reset()
        compiled = torch.compile(fn, fullgraph=True)
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            actual = compiled(x)
            expected = fn(x)
        stream.synchronize()
        self.assertTrue(
            any(key[2:5] == (shape[0], shape[1], 1) for key in _AUTOTUNE_CACHE)
        )
        self.assertEqual(actual, expected)
        self.assertEqual(actual, actual.T)

    def test_quack_grouped_symmetric_mm(self, device):
        if torch.cuda.get_device_capability(device)[0] not in (10, 11):
            self.skipTest("requires SM100 or SM110")

        from torch._vendor.quack.gemm_symmetric import gemm_symmetric

        x = torch.randn(2, 512, 1024, device=device, dtype=torch.bfloat16)
        gram = torch.empty(2, 512, 512, device=device, dtype=torch.bfloat16)
        gemm_symmetric(x, gram)
        self.assertEqual(gram, torch.bmm(x, x.mT))

        update = torch.empty_like(gram)
        gemm_symmetric(gram, update, C=gram, alpha=2.0315, beta=-4.775)
        expected = torch.baddbmm(gram, gram, gram, beta=-4.775, alpha=2.0315)
        self.assertEqual(update, expected, rtol=2e-2, atol=5e-1)
        self.assertEqual(update, update.mT)


instantiate_device_type_tests(SymmetricMMTest, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
