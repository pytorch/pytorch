# Owner(s): ["oncall: distributed"]

import torch
from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import run_tests, TestCase


bf16, fp16, fp32 = torch.bfloat16, torch.float16, torch.float32


class TestChunkCatMixedDtype(TestCase):
    def test_numerics(self, device):
        # Dim-0 sizes that need padding, multi-dim, and one large enough to
        # span several blocks per chunk
        sizes = [(5, 3), (7,), (2049, 2, 2)]
        # On CUDA: the fused kernel, then two composite fallbacks
        for input_dtypes, noncontiguous in (
            ((bf16, fp32, bf16), False),
            ((bf16, fp32, bf16), True),
            ((fp16, fp32, fp16), False),
        ):
            tensors = [
                torch.randn(size, device=device, dtype=dtype)
                for size, dtype in zip(sizes, input_dtypes)
            ]
            if noncontiguous:
                tensors = [torch.cat([t, t], dim=-1)[..., ::2] for t in tensors]
                self.assertFalse(tensors[0].is_contiguous())
            expected = torch._chunk_cat([t.to(fp32) for t in tensors], 0, 4)
            out = torch.empty_like(expected)
            torch.ops.fsdp.chunk_cat_mixed_dtype(tensors, 0, 4, out=out)
            self.assertEqual(out, expected, atol=0, rtol=0)

    @onlyCUDA
    def test_kernels(self, device):
        tensors = [torch.randn(8, 3, device=device, dtype=d) for d in (bf16, fp32)]
        out = torch.empty(2, 24, device=device)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            torch.ops.fsdp.chunk_cat_mixed_dtype(tensors, 0, 2, out=out)
            torch.cuda.synchronize()
        kernels = [
            event.name
            for event in prof.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
            and not event.name.startswith("Memcpy")
        ]
        # One launch copies the fp32 input and a second casts the bf16 one
        self.assertEqual(len(kernels), 2, kernels)
        for kernel in kernels:
            self.assertIn("chunk_cat_cuda_kernel", kernel)


instantiate_device_type_tests(
    TestChunkCatMixedDtype, globals(), only_for=("cpu", "cuda")
)

if __name__ == "__main__":
    run_tests()
