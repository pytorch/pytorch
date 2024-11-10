# Owner(s): ["module: inductor"]

import functools
import unittest

import torch
from torch._dynamo import config as dynamo_config
from torch._inductor import config as inductor_config
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import is_big_gpu
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    skipGPUIf,
)
from torch.testing._internal.common_utils import IS_LINUX, parametrize
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CUDA,
    HAS_GPU,
    requires_gpu,
)


class TestUnbackedSymints(InductorTestCase):
    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_expand(self, device):
        def fn(x, y):
            nz = torch.nonzero(x)
            # unbacked symint in nz.size
            x_exp = nz.expand([-1, 128])
            # unbacked symint in target sizes
            y_exp = y.expand([-1, nz.size(0)])
            return x_exp, y_exp

        example_inputs = (
            torch.randn((32), device=device),
            torch.randn((32, 1), device=device),
        )

        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)

        torch.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_expand_ok_with_runtime_assert(self, device):
        def fn(x):
            nz = x.nonzero()
            torch._check(nz.size(0) == 128)
            return nz.expand([128, -1, 2])

        x = make_tensor(32, 4, device=device, dtype=torch.float32, exclude_zero=True)
        actual = torch.compile(fn, fullgraph=True)(x)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_broadcast_tensors(self, device):
        def fn(x):
            nz = x.nonzero()
            a = torch.zeros([nz.size(0), 512])
            b = torch.ones([nz.size(0), 1])
            return a * b

        x = torch.randn(32, 4, device=device)
        actual = torch.compile(fn, fullgraph=True)(x)
        expected = fn(x)
        torch.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_autotuning(self, device):
        def fn(x, y):
            nz = torch.nonzero(x)
            # unbacked symint in the GEMM input shape
            a = x.new_ones([nz.size(0), y.size(0)])
            return a @ y

        example_inputs = (
            torch.randn((64), device=device),
            torch.randn((32, 16), device=device),
        )

        with inductor_config.patch(
            {
                "max_autotune_gemm": True,
            }
        ):
            actual = torch.compile(fn, fullgraph=True)(*example_inputs)
            expected = fn(*example_inputs)

        torch.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_split_with_sizes(self, device):
        def fn(x, y):
            l = y.tolist()
            s = torch.split(x, l)
            d = l[0] + l[1] + l[2]
            return s[0].sum(), d

        example_inputs = (torch.randn((32), device=device), torch.tensor((7, 16, 9)))

        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)

        torch.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_view_of_slice(self, device):
        # Tests View.create(slice, size_with_unbacked_symint)
        def fn(x):
            nz = torch.nonzero(x)  # introduce unbacked symint
            squared = nz * nz  # avoid ReinterpretView when lowering Slice
            sliced = torch.ops.aten.slice.Tensor(squared, dim=1, start=-2, end=None)
            view = sliced.unsqueeze(dim=0)
            return view.squeeze(
                dim=0
            )  # make sure no unbacked symint in output's stride

        example_inputs = (torch.randn(1, 1, 1, 1, device=device),)
        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @requires_gpu()
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_triton_kernel_grid(self, device):
        if device == "cpu":
            raise unittest.SkipTest("Triton kernel requires GPU")

        from torch.testing._internal.triton_utils import add_kernel

        def fn(x):
            maxlen = max(x.item(), 512)
            a = torch.ones(maxlen, device=device)
            b = torch.ones(maxlen, device=device)
            out = torch.zeros_like(a)
            # unbacked symint in grid
            add_kernel[(1, 1, maxlen)](a, b, out, maxlen, 32)
            return out

        example_inputs = (torch.randint(high=1024, size=(1,), device=device),)
        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_nonzero_in_inference_mode(self, device):
        def fn(x):
            return torch.nonzero(x)

        example_inputs = (torch.randint(0, 2, (128,), device=device),)

        with torch.inference_mode():
            actual = torch.compile(fn, fullgraph=True)(*example_inputs)
            expected = fn(*example_inputs)

        torch.testing.assert_close(actual, expected)

    @inductor_config.patch({"max_autotune": True})
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_equivalent_backed_unbacked(self, device):
        # Tests scenario when there are two equivalent backed & unbacked symints,
        # but when we look-up a size hint on the unbacked symint, we ignorantly
        # use the default fallback hint.

        def fn(x, w, a, b):
            # Make tensors where 1st dim is unbacked/backed.
            u0, s0 = a.item(), b.size(0)
            unbacked = x.expand(u0, *x.shape)
            backed = x.expand(s0, *x.shape)

            # The cat unifies u0 and s0 -- i.e. u0 == s0.
            cat = torch.cat([backed, unbacked, unbacked], dim=1)  # [s0, 30, 16]
            mat1 = torch.permute(cat, [0, 2, 1])  # [s0, 16, 30]
            mat2 = w.expand(u0, *w.shape)  # [u0, 30, 32]
            bmm = torch.ops.aten.bmm(mat1, mat2)
            return bmm

        example_inputs = (
            torch.randn((10, 16), dtype=torch.float32, device=device),
            torch.randn((30, 32), dtype=torch.float32, device=device),
            torch.tensor(7, device=device),
            backed := torch.randn((7,), device=device),
        )
        torch._dynamo.mark_dynamic(backed, 0)  # create backed symint

        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @requires_gpu()
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_vertical_pointwise_reduction_fusion(self, device):
        # reset in case we run both cpu and cuda tests
        torch._inductor.metrics.reset()

        # Tests fusing a pointwise & reduction op with unbacked numel/rnumel.
        def fn(x, y, repeats):
            u0 = repeats.item()
            unbacked = y.expand(u0, *y.shape)  # [u0, 1, 16]

            # Note: We add x to both pointwise and reduction. Otherwise, the
            # scheduler will refuse to fuse ops whose only common buffer has
            # unbacked symints.
            pointwise = unbacked + x
            reduction = torch.sum(pointwise + x)
            return pointwise, reduction

        example_inputs = (
            torch.randn(32, 16).to(GPU_TYPE),
            torch.randn(1, 16).to(GPU_TYPE),
            torch.tensor(32).to(GPU_TYPE),
        )

        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @dynamo_config.patch({"capture_scalar_outputs": True})
    @parametrize(
        "torch_fn", [torch.mm, torch.bmm, torch.addmm], name_fn=lambda fn: fn.__name__
    )
    @parametrize("coordinate_descent_tuning", [True, False], name_fn=str)
    def test_mm_and_friends(self, device, torch_fn, coordinate_descent_tuning):
        if torch_fn == torch.addmm:
            torch_fn = functools.partial(torch_fn, torch.ones(1, device=device))

        def fn(x, w, repeats, is_bmm):
            u0 = repeats.item()
            torch._check_is_size(u0)

            x_unbacked = x.expand(u0, 32)
            w_unbacked = w.expand(32, u0)
            if is_bmm:
                # Make sure inputs are batched.
                x_unbacked = x_unbacked.expand(10, *x_unbacked.shape)
                w_unbacked = w_unbacked.expand(10, *w_unbacked.shape)

            return torch_fn(x_unbacked, w_unbacked)

        example_inputs = (
            torch.randn(1, 32, device=device),
            torch.randn(32, 1, device=device),
            torch.tensor(100, device=device),
            torch_fn == torch.bmm,
        )
        with inductor_config.patch(
            {
                # coordinate_descent_tuning has its own path during decomp
                "coordinate_descent_tuning": coordinate_descent_tuning,
            }
        ):
            actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_range_tree_divisor(self, device):
        def fn(x, num):
            u0 = num.item()
            torch._check_is_size(u0)
            zeros = torch.zeros(u0, device=device, dtype=torch.int)
            return (torch.ops.aten.index(x, [None, zeros]),)

        example_inputs = (
            torch.randn(16, 16, device=device),
            torch.tensor(3, device=device),
        )

        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_unbacked_masked_scatter(self, device):
        def fn(value, mask):
            u0 = mask.count_nonzero()
            source = torch.ones(u0, dtype=torch.float32, device=device)
            return torch.masked_scatter(value, mask, source)

        value = make_tensor(10, 10, dtype=torch.float32, device=device)
        mask = make_tensor(10, 10, dtype=torch.bool, device=device)
        example_inputs = (value, mask)

        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)


instantiate_device_type_tests(TestUnbackedSymints, globals(), allow_xpu=True)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if IS_LINUX and HAS_GPU and (not HAS_CUDA or is_big_gpu(0)):
        run_tests()
