# Owner(s): ["oncall: distributed"]

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


bf16, fp16, fp32 = torch.bfloat16, torch.float16, torch.float32


def chunk_cat_mixed_dtype(tensors, dim, num_chunks, out_dtype):
    expected = torch._chunk_cat([t.to(out_dtype) for t in tensors], dim, num_chunks)
    out = torch.empty_like(expected)
    torch.ops.fsdp.chunk_cat_mixed_dtype(tensors, dim, num_chunks, out=out)
    return out, expected


class TestChunkCatMixedDtype(TestCase):
    @parametrize(
        "input_dtypes,out_dtype",
        [
            ((bf16, fp32, bf16), fp32),
            ((bf16, bf16, bf16), fp32),
            ((fp32, fp32, fp32), fp32),
            ((fp32, fp32, fp32), bf16),
            ((bf16, fp32, bf16), bf16),
            ((fp16, fp32, fp16), fp32),
        ],
    )
    @parametrize("num_chunks", [1, 4])
    @parametrize("noncontiguous", [False, True])
    def test_numerics(self, device, input_dtypes, out_dtype, num_chunks, noncontiguous):
        # Dim-0 sizes that need padding, multi-dim, and one large enough to
        # span several blocks per chunk
        sizes = [(5, 3), (7,), (2049, 2, 2)]
        tensors = [
            torch.randn(size, device=device, dtype=dtype)
            for size, dtype in zip(sizes, input_dtypes)
        ]
        if noncontiguous:
            tensors = [torch.cat([t, t], dim=-1)[..., ::2] for t in tensors]
            self.assertFalse(tensors[0].is_contiguous())
        out, expected = chunk_cat_mixed_dtype(tensors, 0, num_chunks, out_dtype)
        self.assertEqual(out, expected, atol=0, rtol=0)

    @parametrize("input_dtypes", [(bf16, fp32, bf16), (bf16, bf16, bf16)])
    def test_nonzero_dim(self, device, input_dtypes):
        sizes = [(4, 5), (4, 3, 2), (4, 7)]
        tensors = [
            torch.randn(size, device=device, dtype=dtype)
            for size, dtype in zip(sizes, input_dtypes)
        ]
        out, expected = chunk_cat_mixed_dtype(tensors, 1, 2, fp32)
        self.assertEqual(out, expected, atol=0, rtol=0)

    def test_version_counter(self, device):
        tensors = [torch.randn(4, device=device, dtype=dtype) for dtype in (bf16, fp32)]
        out = torch.empty(2, 4, device=device)
        torch.ops.fsdp.chunk_cat_mixed_dtype(tensors, 0, 2, out=out)
        self.assertEqual(out._version, 1)

    def test_invalid_inputs(self, device):
        out = torch.empty(2, 8, device=device)
        valid = torch.randn(8, device=device, dtype=bf16)
        with self.assertRaisesRegex(RuntimeError, "cast"):
            torch.ops.fsdp.chunk_cat_mixed_dtype(
                [valid, torch.randn(8, device=device, dtype=torch.complex64)],
                0,
                2,
                out=out,
            )
        with self.assertRaisesRegex(RuntimeError, "positive num_chunks"):
            torch.ops.fsdp.chunk_cat_mixed_dtype([valid], 0, 0, out=out)
        with self.assertRaisesRegex(RuntimeError, "non-empty tensor"):
            torch.ops.fsdp.chunk_cat_mixed_dtype(
                [valid, torch.empty(0, device=device)], 0, 2, out=out
            )

    def test_fake_tensor(self, device):
        with FakeTensorMode():
            tensors = [torch.empty(5, 3, device=device, dtype=bf16)]
            tensors.append(torch.empty(7, device=device))
            out = torch.empty(4, 8, device=device)
            torch.ops.fsdp.chunk_cat_mixed_dtype(tensors, 0, 4, out=out)
        self.assertEqual(out.shape, (4, 8))

    def test_compile(self, device):
        def fn(tensors):
            out = torch.empty(4, 8, device=device)
            torch.ops.fsdp.chunk_cat_mixed_dtype(tensors, 0, 4, out=out)
            return out * 2

        tensors = [
            torch.randn(5, 3, device=device, dtype=bf16),
            torch.randn(7, device=device),
        ]
        self.assertEqual(torch.compile(fn, fullgraph=True)(tensors), fn(tensors))

    @onlyCUDA
    @parametrize(
        "input_dtypes,num_kernels",
        [((bf16, fp32, bf16), 2), ((bf16, bf16), 1), ((fp32, fp32), 1)],
    )
    def test_kernels(self, device, input_dtypes, num_kernels):
        tensors = [torch.randn(8, 3, device=device, dtype=d) for d in input_dtypes]
        out = torch.empty(2, 12 * len(tensors), device=device)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            torch.ops.fsdp.chunk_cat_mixed_dtype(tensors, 0, 2, out=out)
            torch.cuda.synchronize()
        kernels = [
            event.name
            for event in prof.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
            and not event.name.startswith("Memcpy")
        ]
        self.assertEqual(len(kernels), num_kernels, kernels)
        for kernel in kernels:
            self.assertIn("chunk_cat_cuda_kernel", kernel)


instantiate_device_type_tests(
    TestChunkCatMixedDtype, globals(), only_for=("cpu", "cuda")
)

if __name__ == "__main__":
    run_tests()
