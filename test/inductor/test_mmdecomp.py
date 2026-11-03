# Owner(s): ["module: nn"]

import math
import unittest

import torch
from torch._inductor import config
from torch._inductor.decomposition import addmm as decomp_addmm, bmm as decomp_bmm, mm
from torch._inductor.utils import fresh_cache
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)
from torch.testing._internal.common_cuda import SM80OrLater
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
)
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize,
    run_tests,
    TEST_XPU,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


default_atol = {
    torch.float16: 1e-3,
    torch.bfloat16: float("infinity"),
    torch.float32: 1e-5,
}
default_rtol = {
    torch.float16: 1e-3,
    torch.bfloat16: float("infinity"),
    torch.float32: 1.3e-6,
}


def rand_math_tensor(
    shape: tuple[int | list[int]],
    device: str,
    dtype: torch.dtype,
    requires_grad: bool = False,
    packed: bool = False,
) -> torch.Tensor:
    """Creates rand dense or nested tensor with given shape and type.

    Args:
        shape (Tuple[int]): Shape of Tensor to construct
        device (str): which device to create tensor on
        dtype (torch.dtype): Tensors' dtype
        requires_grad (bool, optional): Tensors grad status. Defaults to False.
        packed (bool, optional): Whether to create a single QKV packed or not. Defaults to False.

    Returns:
        torch.Tensor: A new tensor
    """
    return torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)


def init_tensor(tensor_list, **kwargs) -> torch.Tensor:
    return torch.Tensor(tensor_list).to(**kwargs)


def run_comp_nocomp(function, *inputs, **kwargs):
    c_function = torch.compile(function)

    f_res = function(*inputs)
    cf_res = c_function(*inputs)

    if not (math.isinf(kwargs.get("atol", 0.0)) or math.isinf(kwargs.get("rtol", 0.0))):
        torch.testing.assert_close(f_res, cf_res, **kwargs)


# The test functions are used by several tests
def torch_mm(a, b):
    return torch.mm(a, b)


def torch_addmm(add, b, c):
    return torch.addmm(add, b, c)


def torch_bmm(a, b):
    return torch.bmm(a, b)


def torch_baddbmm(add, b, c, alpha, beta):
    return torch.baddbmm(add, b, c, alpha=alpha, beta=beta)


def create_fake_tensor_with_dynamic_size(x, fake_mode):
    with fake_mode:
        dynamic_sizes = [DimDynamic.DYNAMIC for _ in range(x.dim())]
        dynamic_strides = [DimDynamic.INFER_STRIDE for _ in range(x.dim())]
        return fake_mode.from_tensor(
            x,
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=dynamic_sizes,
                dynamic_strides=dynamic_strides,
            ),
        )


# The shapes we test on
ts_list = [
    (1, 32, 32, 1),
    (1, 10, 10, 1),
    (1, 3, 3, 1),
    (32, 1, 1, 32),
    (3, 1, 1, 3),
    (4, 1, 1, 9),
    (9, 1, 1, 4),
]


class TestDecomp(NNTestCase):
    _do_cuda_memory_leak_check = GPU_TYPE == "cuda"
    _do_cuda_non_default_stream = GPU_TYPE == "cuda"

    @unittest.skipIf(not HAS_GPU, "GPU tests require triton")
    @parametrize("dtype", [torch.float, torch.bfloat16])
    def test_simple_mm(self, device, dtype):
        fudge = 10
        rtol = default_rtol[dtype] * fudge
        atol = default_atol[dtype] * fudge

        for t_size in ts_list:
            ((a1_0, a1_1, a2_0, a2_1)) = t_size

            t1 = rand_math_tensor((a1_0, a1_1), dtype=dtype, device=device)
            t2 = rand_math_tensor((a2_0, a2_1), dtype=dtype, device=device)
            tadd = rand_math_tensor((a1_0, a2_1), dtype=dtype, device=device)

            run_comp_nocomp(torch_mm, t1, t2, rtol=rtol, atol=atol)
            run_comp_nocomp(torch_addmm, tadd, t1, t2, rtol=rtol, atol=atol)

    @unittest.skipIf(not HAS_GPU, "GPU tests require triton")
    @parametrize(
        "dtype",
        [torch.float, torch.float16, torch.bfloat16]
        if SM80OrLater or TEST_XPU
        else [torch.float],
    )
    @parametrize("m", [256, 4096])
    @parametrize("k,n", [(2, 3), (3, 4), (4, 4)])
    def test_small_mm_pointwise(self, device, dtype, m, k, n):
        if device == "cpu":
            self.skipTest("small-dim mm pointwise is GPU-only")
        from torch._dynamo.utils import counters

        counters.clear()
        torch._dynamo.reset()
        atol = {torch.float32: 1e-4, torch.float16: 1e-2, torch.bfloat16: 1e-1}[dtype]
        rtol = {torch.float32: 1.3e-5, torch.float16: 1e-2, torch.bfloat16: 1.6e-2}[
            dtype
        ]
        t1 = rand_math_tensor((m, k), dtype=dtype, device=device)
        t2 = rand_math_tensor((k, n), dtype=dtype, device=device)
        with fresh_cache():
            run_comp_nocomp(torch_mm, t1, t2, rtol=rtol, atol=atol)
        self.assertEqual(counters["inductor"]["decompose_mm_pointwise"], 1)

    @unittest.skipIf(not HAS_GPU, "GPU tests require triton")
    @config.patch(shape_padding=True)
    def test_small_mm_pointwise_skips_padding(self, device):
        if device == "cpu":
            self.skipTest("small-dim mm pointwise is GPU-only")
        from unittest import mock

        from torch._dynamo.utils import counters

        counters.clear()
        torch._dynamo.reset()
        a = torch.ones(64, 3, device=device)
        b = torch.ones(3, 3, device=device)
        with mock.patch(
            "torch._inductor.fx_passes.pad_mm._should_pad", return_value=True
        ) as should_pad:
            with fresh_cache():
                run_comp_nocomp(torch_mm, a, b)
        self.assertEqual(should_pad.call_count, 0)
        self.assertEqual(counters["inductor"]["decompose_mm_pointwise"], 1)

    @unittest.skipIf(not HAS_GPU, "GPU tests require triton")
    @parametrize(
        "dtype",
        [torch.float, torch.float16, torch.bfloat16]
        if SM80OrLater or TEST_XPU
        else [torch.float],
    )
    @parametrize("m", [256, 4096])
    @parametrize("k,n", [(2, 3), (3, 4), (4, 4)])
    @parametrize("transpose", ["lhs", "rhs", "both"])
    def test_small_mm_pointwise_transposed(self, device, dtype, m, k, n, transpose):
        if device == "cpu":
            self.skipTest("small-dim mm pointwise is GPU-only")
        from torch._dynamo.utils import counters

        counters.clear()
        torch._dynamo.reset()
        atol = {torch.float32: 1e-4, torch.float16: 1e-2, torch.bfloat16: 1e-1}[dtype]
        rtol = {torch.float32: 1.3e-5, torch.float16: 1e-2, torch.bfloat16: 1.6e-2}[
            dtype
        ]
        if transpose == "lhs":
            t1 = rand_math_tensor((k, m), dtype=dtype, device=device)
            t2 = rand_math_tensor((k, n), dtype=dtype, device=device)

            def fn(a, b):
                return torch.mm(a.T, b)

        elif transpose == "rhs":
            t1 = rand_math_tensor((m, k), dtype=dtype, device=device)
            t2 = rand_math_tensor((n, k), dtype=dtype, device=device)

            def fn(a, b):
                return torch.mm(a, b.T)

        else:
            t1 = rand_math_tensor((k, m), dtype=dtype, device=device)
            t2 = rand_math_tensor((n, k), dtype=dtype, device=device)

            def fn(a, b):
                return torch.mm(a.T, b.T)

        with fresh_cache():
            run_comp_nocomp(fn, t1, t2, rtol=rtol, atol=atol)
        self.assertEqual(counters["inductor"]["decompose_mm_pointwise"], 1)

    @unittest.skipIf(not HAS_GPU, "GPU tests require triton")
    def test_small_mm_no_pointwise_for_large_dims(self, device):
        if device == "cpu":
            self.skipTest("GPU-only test")
        from torch._dynamo.utils import counters

        counters.clear()

        def fn(a, b):
            return torch.mm(a, b)

        a = torch.randn(256, 5, device=device)
        b = torch.randn(5, 5, device=device)
        torch.compile(fn)(a, b)
        self.assertEqual(counters["inductor"]["decompose_mm_pointwise"], 0)

    @unittest.skipIf(not HAS_GPU, "GPU tests require triton")
    @config.patch(max_autotune_gemm=True)
    def test_small_mm_no_pointwise_under_max_autotune(self, device):
        if device == "cpu":
            self.skipTest("GPU-only test")
        from torch._dynamo.utils import counters

        counters.clear()

        def fn(a, b):
            return torch.mm(a, b)

        a = torch.randn(256, 3, device=device)
        b = torch.randn(3, 4, device=device)
        torch.compile(fn)(a, b)
        self.assertEqual(counters["inductor"]["decompose_mm_pointwise"], 0)

    @unittest.skipIf(not HAS_GPU, "GPU tests require triton")
    @parametrize(
        "dtype",
        [torch.float, torch.bfloat16] if SM80OrLater or TEST_XPU else [torch.float],
    )
    @parametrize("bs", [1, 2, 4, 10])
    def test_batched_mm(self, device, dtype, bs):
        fudge = 3
        rtol = default_rtol[dtype] * fudge
        atol = default_atol[dtype] * fudge

        for t_size in ts_list:
            ((a1_0, a1_1, a2_0, a2_1)) = t_size

            t1 = rand_math_tensor((bs, a1_0, a1_1), dtype=dtype, device=device)
            t2 = rand_math_tensor((bs, a2_0, a2_1), dtype=dtype, device=device)
            tadd = rand_math_tensor((bs, a1_0, a2_1), dtype=dtype, device=device)

            run_comp_nocomp(torch_bmm, t1, t2, rtol=rtol, atol=atol)

            for alpha in (0, 1, -1, 0.5, -0.5):
                for beta in (0, 1, -1, 0.5, -0.5):
                    run_comp_nocomp(
                        torch_baddbmm, tadd, t1, t2, alpha, beta, rtol=rtol, atol=atol
                    )

    @unittest.skipIf(not HAS_GPU, "GPU tests require triton")
    @config.patch(coordinate_descent_tuning=True)
    def test_bmm_batch2_last_dim_size_is_one(self, device):
        fudge = 3
        rtol = default_rtol[torch.float32] * fudge
        atol = default_atol[torch.float32] * fudge

        t1 = torch.randn(1, 32, 2, device=device)
        t2 = torch.randn(1, 2, 1, device=device)

        run_comp_nocomp(torch_bmm, t1, t2, rtol=rtol, atol=atol)

    @config.patch(coordinate_descent_tuning=False)
    def test_bmm_outer_product_k_is_one(self, device):
        t1 = torch.randn(32, 8, 1, device=device)
        t2 = torch.randn(32, 1, 256, device=device)
        expected = torch.bmm(t1, t2)

        out = decomp_bmm(t1, t2)

        self.assertIsNot(out, NotImplemented)
        self.assertEqual(expected, out)

    @unittest.skipIf(not HAS_GPU, "GPU tests require triton")
    def test_bmm_outer_product_k_is_one_with_unbacked_k(self, device):
        if device == "cpu":
            self.skipTest("unbacked symints require GPU fake tensors")

        shape_env = ShapeEnv()
        with FakeTensorMode(shape_env=shape_env):
            b, m, n = [shape_env.create_unbacked_symint() for _ in range(3)]
            lhs_k_unbacked, rhs_k_unbacked = [
                shape_env.create_unbacked_symint() for _ in range(2)
            ]

            lhs_static_k = torch.empty((b, m, 1), device=device)
            rhs_static_k = torch.empty((b, 1, n), device=device)
            lhs_unbacked_k = torch.empty((b, m, lhs_k_unbacked), device=device)
            rhs_unbacked_k = torch.empty((b, rhs_k_unbacked, n), device=device)

            self.assertIsNot(
                decomp_bmm(lhs_static_k, rhs_static_k),
                NotImplemented,
            )
            self.assertIs(
                decomp_bmm(lhs_static_k, rhs_unbacked_k),
                NotImplemented,
            )
            self.assertIs(
                decomp_bmm(lhs_unbacked_k, rhs_static_k),
                NotImplemented,
            )
            self.assertIs(
                decomp_bmm(lhs_unbacked_k, rhs_unbacked_k),
                NotImplemented,
            )

    @config.patch(coordinate_descent_tuning=False)
    def test_bmm_outer_product_permuted_inputs(self, device):
        B, M, N = 4, 8, 16

        cases = [
            # LHS: batch dim permuted
            (
                torch.randn(M, B, 1, device=device).permute(1, 0, 2),
                torch.randn(B, 1, N, device=device),
            ),
            # RHS: batch dim permuted
            (
                torch.randn(B, M, 1, device=device),
                torch.randn(N, B, 1, device=device).permute(1, 2, 0),
            ),
            # Both permuted
            (
                torch.randn(M, B, 1, device=device).permute(1, 0, 2),
                torch.randn(N, B, 1, device=device).permute(1, 2, 0),
            ),
            # LHS: fully transposed [1, M, B] -> [B, M, 1]
            (
                torch.randn(1, M, B, device=device).permute(2, 1, 0),
                torch.randn(B, 1, N, device=device),
            ),
        ]

        for t1, t2 in cases:
            expected = torch.bmm(t1, t2)
            out = decomp_bmm(t1, t2)
            self.assertIsNot(out, NotImplemented)
            self.assertEqual(expected, out, exact_stride=True)

    @unittest.skipIf(not HAS_GPU, "GPU tests require triton")
    @parametrize("dtype", [torch.float, torch.bfloat16, torch.int])
    def test_some(self, device, dtype):
        # this Pytorch data type is not fully supported on cuda today
        # - unfortunately we can't skipIf because we don't see the actual params in skipIf
        if device.startswith(GPU_TYPE) and dtype == torch.int:
            return

        run_comp_nocomp(
            torch_mm,
            init_tensor([[1], [2], [3], [4]], dtype=dtype, device=device),
            init_tensor([[1, 2, 3, 4]], dtype=dtype, device=device),
        )
        run_comp_nocomp(
            torch_mm,
            init_tensor([[1, 2, 3, 4]], dtype=dtype, device=device),
            init_tensor([[1], [2], [3], [4]], dtype=dtype, device=device),
        )

    @unittest.skipIf(not HAS_GPU, "GPU tests require triton")
    @parametrize("dtype", [torch.float, torch.bfloat16, torch.int])
    @parametrize("bs", [1, 2, 4, 10])
    def test_some_batched(self, device, dtype, bs):
        # this Pytorch data type is not fully supported on cuda today
        # - unfortunately we can't skipIf because we don't see the actual params in skipIf
        if device.startswith(GPU_TYPE) and dtype == torch.int:
            return

        run_comp_nocomp(
            torch_bmm,
            init_tensor([[[1], [2], [3], [4]]] * bs, dtype=dtype, device=device),
            init_tensor([[[1, 2, 3, 4]]] * bs, dtype=dtype, device=device),
        )
        run_comp_nocomp(
            torch_bmm,
            init_tensor([[[1, 2, 3, 4]]] * bs, dtype=dtype, device=device),
            init_tensor([[[1], [2], [3], [4]]] * bs, dtype=dtype, device=device),
        )

    # Shapes hit each addmm decomp branch (CPU dot, CPU small mat2, non-CPU
    # k == 1) and the lowering.
    @parametrize("m,k,n", [(1, 4, 1), (1, 4, 4), (4, 1, 4), (4, 4, 4)])
    @parametrize("alpha", [1, 0.5])
    def test_addmm_zero_beta_drops_nan_bias(self, device, m, k, n, alpha):
        from torch._dynamo.utils import counters

        def fn(x, a, b):
            return torch.addmm(x, a, b, beta=0, alpha=alpha)

        counters.clear()
        torch._dynamo.reset()
        x = torch.full((n,), float("nan"), device=device)
        a = torch.randn(m, k, device=device)
        b = torch.randn(k, n, device=device)
        with fresh_cache():
            out = torch.compile(fn, fullgraph=True)(x, a, b)
        self.assertFalse(out.isnan().any())
        self.assertEqual(out, fn(x, a, b), equal_nan=False)
        # CUDA sends beta == 0 to the addmm lowering instead.
        device_type = torch.device(device).type
        decomposed = m == 1 if device_type == "cpu" else device_type == "xpu" and k == 1
        self.assertEqual(counters["inductor"]["decompose_addmm"], int(decomposed))

    @parametrize("op", [torch.addmm, torch.baddbmm])
    def test_zero_beta_fp16_alpha_overflow(self, device, op):
        def fn(x, a, b):
            return op(x, a, b, beta=0, alpha=0.25)

        # Each dot product is 160000, past the fp16 max; alpha * acc is not.
        shape = (2, 16, 16) if op is torch.baddbmm else (16, 16)
        x = torch.tensor(float("nan"), dtype=torch.float16, device=device)
        a = torch.full(shape, 100.0, dtype=torch.float16, device=device)
        self.assertEqual(torch.compile(fn, fullgraph=True)(x, a, a), fn(x, a, a))

    @onlyCPU
    @parametrize("op", [torch.addmm, torch.baddbmm])
    def test_zero_beta_aten_reads_no_zeros(self, device, op):
        from torch._inductor import metrics

        def fn(x, a, b):
            return op(x, a, b, beta=0, alpha=0.5)

        shape = (2, 16, 16) if op is torch.baddbmm else (16, 16)
        x = torch.full((16,), float("nan"))
        a = torch.randn(shape)
        metrics.reset()
        self.assertEqual(torch.compile(fn, fullgraph=True)(x, a, a), fn(x, a, a))
        self.assertEqual(metrics.generated_kernel_count, 0)

    @onlyCPU
    @parametrize("beta,m", [(0, 1), (0, 32), (1, 1)])
    def test_addmm_checks_bias_shape(self, device, beta, m):
        def fn(x, a, b):
            return torch.addmm(x, a, b, beta=beta)

        args = (torch.zeros(1, 1, 4), torch.randn(m, 4), torch.randn(4, 4))
        with self.assertRaisesRegex(RuntimeError, "expand"):
            fn(*args)
        with self.assertRaisesRegex(RuntimeError, "expand"):
            torch.compile(fn, fullgraph=True)(*args)

    @onlyCPU
    def test_addmm_unbacked_alpha(self, device):
        shape_env = ShapeEnv()
        with FakeTensorMode(shape_env=shape_env):
            alpha = shape_env.create_unbacked_symint()
            bias = torch.empty(4, device="xpu")
            a = torch.empty(4, 1, device="xpu")
            b = torch.empty(1, 4, device="xpu")
            out = decomp_addmm(bias, a, b, alpha=alpha)
        self.assertEqual(out.shape, (4, 4))

    @onlyCPU
    @parametrize(
        "bias_device,bias_shape,msg",
        [
            ("xpu", (3,), "expand"),
            ("xpu", (1, 1, 4), "expand"),
            ("cpu", (), "device"),
        ],
    )
    def test_addmm_checks_xpu_metadata(self, device, bias_device, bias_shape, msg):
        with FakeTensorMode():
            bias = torch.empty(bias_shape, device=bias_device)
            a = torch.empty(4, 1, device="xpu")
            b = torch.empty(1, 4, device="xpu")
            with self.assertRaisesRegex(RuntimeError, msg):
                decomp_addmm(bias, a, b, beta=0)

    @parametrize("dtype", [torch.float, torch.bfloat16])
    def test_dynamic_shape_mm(self, device, dtype):
        # Test that the mm decomp does not evaluate expressions for dynamic shapes

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)

        # Only test decomp for cpu to match fake tensors from dynamo
        if device != "cpu":
            return

        for t_size in ts_list:
            ((a1_0, a1_1, a2_0, a2_1)) = t_size

            # Create the fake tensors
            t1 = create_fake_tensor_with_dynamic_size(
                rand_math_tensor((a1_0, a1_1), dtype=dtype, device=device),
                fake_mode,
            )
            t2 = create_fake_tensor_with_dynamic_size(
                rand_math_tensor((a2_0, a2_1), dtype=dtype, device=device),
                fake_mode,
            )

            # Save the expression types to check if any symints are evaluated
            og_t1_expr_types = [
                type(d.node.expr) if type(d) is torch.SymInt else int for d in t1.size()
            ]
            og_t2_expr_types = [
                type(d.node.expr) if type(d) is torch.SymInt else int for d in t2.size()
            ]

            r = mm(t1, t2)

            # Make sure all symints are not evaluated
            new_t1_expr_types = [
                type(d.node.expr) if type(d) is torch.SymInt else int for d in t1.size()
            ]
            new_t2_expr_types = [
                type(d.node.expr) if type(d) is torch.SymInt else int for d in t2.size()
            ]
            self.assertTrue(
                all(
                    og_t1_expr_types[i] == new_t1_expr_types[i]
                    for i in range(len(og_t1_expr_types))
                )
            )
            self.assertTrue(
                all(
                    og_t2_expr_types[i] == new_t2_expr_types[i]
                    for i in range(len(og_t2_expr_types))
                )
            )

            if r is not NotImplemented:
                # Check that the output is well formed
                self.assertEqual(t1.size(0), r.size(0))
                self.assertEqual(t2.size(1), r.size(1))
                r_expr_types = [
                    type(d.node.expr) if type(d) is torch.SymInt else int
                    for d in r.size()
                ]
                self.assertTrue(r_expr_types[0] == og_t1_expr_types[0])
                self.assertTrue(r_expr_types[1] == og_t2_expr_types[1])


class TestAddmmZeroAlpha(NNTestCase):
    @parametrize("alpha", [0, 0.0, -0.0])
    @parametrize("shape", [(1, 4, 1), (1, 4, 4), (4, 1, 4)])
    def test_zero_alpha_decomposition(self, device, alpha, shape):
        from torch._inductor.decomposition import addmm

        m, k, n = shape
        args = (
            torch.ones(n, device=device),
            torch.ones(m, k, device=device),
            torch.ones(k, n, device=device),
        )
        self.assertIs(addmm(*args, alpha=alpha), NotImplemented)

    @parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
    @parametrize(
        "shape,operand,value,beta",
        [
            ((1, 4, 1), 1, float("nan"), 1),
            ((1, 4, 4), 2, float("inf"), -2),
            ((4, 1, 4), 1, -float("inf"), 0),
            ((8, 32, 8), 0, float("nan"), 0),
            ((8, 32, 8), 0, float("nan"), 1),
            ((8, 32, 8), 2, float("nan"), 0),
        ],
    )
    def test_zero_alpha_numerics(self, device, dtype, shape, operand, value, beta):
        from torch._inductor.utils import run_and_get_code

        def fn(inp, a, b):
            return torch.addmm(inp, a, b, alpha=0, beta=beta)

        m, k, n = shape
        args = (
            torch.ones(n, device=device, dtype=dtype),
            torch.ones(m, k, device=device, dtype=dtype),
            torch.ones(k, n, device=device, dtype=dtype),
        )
        args[operand].fill_(value)
        expected = fn(*args)
        with config.patch(cpp_wrapper=False):
            actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), *args)
        self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertIn("torch.ops.aten.addmm.default", "\n".join(code))

    @parametrize("bias_kind", ["scalar", "vector", "expanded", "transposed"])
    @parametrize("k", [0, 3])
    def test_zero_alpha_shapes(self, device, bias_kind, k):
        def fn(inp, a, b):
            return torch.addmm(inp, a, b, alpha=0, beta=-2)

        a = torch.ones(k, 2, device=device).t()
        b = torch.ones(4, k, device=device).t()

        if bias_kind == "scalar":
            inp = torch.tensor(2.0, device=device)
        elif bias_kind == "vector":
            inp = torch.arange(8.0, device=device)[::2]
        elif bias_kind == "expanded":
            inp = torch.ones(1, 4, device=device).expand(2, 4)
        else:
            inp = torch.arange(8.0, device=device).view(4, 2).t()

        expected = fn(inp, a, b)
        actual = torch.compile(fn, fullgraph=True)(inp, a, b)
        self.assertEqual(actual, expected, exact_stride=True)

    def test_zero_alpha_dynamic_backward(self, device):
        def fn(inp, a, b):
            return torch.addmm(inp, a, b, alpha=0, beta=-2).square().sum()

        compiled = torch.compile(fn, fullgraph=True, dynamic=True)

        for m, k, n in [(2, 3, 4), (5, 6, 7)]:
            args = (
                torch.randn(n, device=device, requires_grad=True),
                torch.randn(m, k, device=device, requires_grad=True),
                torch.randn(k, n, device=device, requires_grad=True),
            )
            expected = fn(*args)
            expected_grad = torch.autograd.grad(expected, args)

            actual = compiled(*args)
            actual_grad = torch.autograd.grad(actual, args)

            self.assertEqual(actual, expected)
            self.assertEqual(actual_grad, expected_grad)

    @parametrize("bad", ["matrices", "bias"])
    def test_zero_alpha_invalid_shapes(self, device, bad):
        def fn(inp, a, b):
            return torch.addmm(inp, a, b, alpha=0)

        inp = torch.ones(5 if bad == "bias" else 4, device=device)
        a = torch.ones(2, 3, device=device)
        b = torch.ones(5 if bad == "matrices" else 3, 4, device=device)

        with self.assertRaisesRegex(RuntimeError, "shape|size|dim|expand"):
            fn(inp, a, b)
        with self.assertRaisesRegex(RuntimeError, "shape|size|dim|expand"):
            torch.compile(fn, fullgraph=True)(inp, a, b)

    @onlyCPU
    @parametrize("alpha,beta", [(1, 1), (0, 1), (2, 1), (1, 0.5), (2, 0.5), (1, 0)])
    def test_addmm_mem_bound_scalars(self, device, alpha, beta):
        from torch._inductor.fx_passes import decompose_mem_bound_mm
        from torch._inductor.fx_passes.split_cat import construct_pattern_matcher_pass
        from torch.fx.experimental.proxy_tensor import make_fx

        def fn(inp, a, b):
            return torch.addmm(inp, a, b, alpha=alpha, beta=beta)

        args = (torch.ones(4), torch.ones(1, 4), torch.ones(4, 4))
        gm = make_fx(fn, tracing_mode="fake")(*args)
        expected = gm(*args)

        with config.patch(post_grad_fusion_options={"decompose_mm_pass": {}}):
            node = next(
                n for n in gm.graph.nodes if n.target == torch.ops.aten.addmm.default
            )
            self.assertTrue(
                decompose_mem_bound_mm.should_decompose_mm(node.args[1], node.args[2])
            )
            construct_pattern_matcher_pass("decompose_mm_pass").apply(gm)

        gm.graph.lint()
        gm.recompile()
        addmms = [
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.aten.addmm.default
        ]
        self.assertEqual(len(addmms), int((alpha, beta) != (1, 1)))
        self.assertEqual(gm(*args), expected)

    @parametrize("alpha,beta", [(1, 1), (0, 1), (2, 1), (1, 0.5), (2, 0.5), (1, 0)])
    def test_addmm_binary_folding_scalars(self, device, alpha, beta):
        from torch._inductor.fx_passes.binary_folding import binary_folding_init
        from torch._inductor.fx_passes.freezing_patterns import binary_folding_pass
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.passes.fake_tensor_prop import FakeTensorProp

        bias = torch.ones(4, device=device)
        weight = torch.ones(3, 4, device=device)
        other = torch.ones(4, device=device)

        def fn(a):
            return torch.addmm(bias, a, weight, alpha=alpha, beta=beta) + other

        a = torch.ones(2, 3, device=device)
        gm = make_fx(fn)(a)
        FakeTensorProp(gm, mode=FakeTensorMode(allow_non_fake_inputs=True)).propagate(a)
        expected = gm(a)

        binary_folding_init()
        with config.patch(enable_linear_binary_folding=True):
            matches = binary_folding_pass.apply(gm)

        gm.graph.lint()
        gm.recompile()
        self.assertEqual(matches > 0, (alpha, beta) == (1, 1))
        self.assertEqual(gm(a), expected)

    @parametrize(
        "alpha,beta,mixed",
        [
            (1, 1, False),
            (0, 1, False),
            (2, 1, False),
            (1, 0.5, False),
            (2, 0.5, False),
            (0, 1, True),
            (2, 1, True),
            (1, 0.5, True),
        ],
    )
    def test_addmm_concat_scalars(self, device, alpha, beta, mixed):
        from unittest import mock

        from torch._inductor.fx_passes import freezing_patterns
        from torch._inductor.pattern_matcher import (
            PatternMatcherPass,
            stable_topological_sort,
        )
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.passes.fake_tensor_prop import FakeTensorProp

        weights = tuple(
            torch.full((4, 4), float(i + 1), device=device) for i in range(3)
        )
        biases = tuple(torch.full((4,), float(i + 1), device=device) for i in range(3))
        coefficients = [(alpha, beta), (1, 1), (1, 1)] if mixed else [(alpha, beta)] * 3

        def fn(x):
            return tuple(
                torch.addmm(bias, x, weight, alpha=a, beta=b)
                for bias, weight, (a, b) in zip(biases, weights, coefficients)
            )

        x = torch.arange(16.0, device=device).reshape(4, 4)
        gm = make_fx(fn)(x)
        expected = gm(x)
        FakeTensorProp(gm, mode=FakeTensorMode(allow_non_fake_inputs=True)).propagate(x)
        concat_pass = PatternMatcherPass()
        with config.patch({"cpp.enable_concat_linear": True}):
            # Register the real concat rules without unrelated freezing passes.
            with mock.patch.object(freezing_patterns, "pass_patterns", [concat_pass]):
                with FakeTensorMode():
                    freezing_patterns.addmm_patterns_init.__wrapped__()
            matches = concat_pass.apply(gm)
        stable_topological_sort(gm.graph)
        gm.graph.lint()
        gm.recompile()
        default_scalars = (alpha, beta) == (1, 1)
        self.assertEqual(matches, int(default_scalars))
        self.assertEqual(
            sum(n.target == torch.ops.aten.addmm.default for n in gm.graph.nodes),
            1 if default_scalars else 3,
        )
        self.assertEqual(gm(x), expected, atol=0, rtol=0)

    def test_zero_alpha_pointwise(self, device):
        from torch._inductor.utils import run_and_get_code

        def fn(inp, a, b):
            return torch.addmm(inp, a, b, alpha=0).relu()

        args = (
            torch.ones(4, device=device),
            torch.full((2, 3), float("nan"), device=device),
            torch.ones(3, 4, device=device),
        )
        expected = fn(*args)
        with config.patch(cpp_wrapper=False):
            actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), *args)
        self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertIn("torch.ops.aten.addmm.default", "\n".join(code))


class TestAddmmZeroAlphaGuards(NNTestCase):
    @parametrize("alpha", [0, 1])
    def test_zero_alpha_unfuse_guard(self, alpha):
        from torch._inductor.fx_passes.post_grad import should_prefer_unfused_addmm
        from torch._inductor.pattern_matcher import Arg, CallFunction, KeywordArg, Match

        graph = torch.fx.Graph()
        inp, a, b = [graph.placeholder(name) for name in ("inp", "a", "b")]
        out = graph.call_function(
            torch.ops.aten.addmm.default,
            (inp, a, b),
            {"alpha": alpha, "beta": 1},
        )
        graph.output(graph.call_function(torch.ops.aten.relu.default, (out,)))
        with FakeTensorMode():
            inp.meta["val"] = torch.empty(4, device="cuda")
            a.meta["val"] = torch.empty(2, 3, device="cuda")
            b.meta["val"] = torch.empty(3, 4, device="cuda")
            out.meta["val"] = torch.empty(2, 4, device="cuda")
        pattern = CallFunction(
            torch.ops.aten.addmm.default,
            KeywordArg("inp"),
            Arg(),
            Arg(),
            alpha=KeywordArg("alpha"),
            beta=KeywordArg("beta"),
        )
        match = pattern.match(out)
        self.assertIsInstance(match, Match)
        self.assertEqual(should_prefer_unfused_addmm(match), alpha != 0)


instantiate_parametrized_tests(TestAddmmZeroAlphaGuards)
instantiate_device_type_tests(
    TestAddmmZeroAlpha, globals(), only_for=("cpu", GPU_TYPE), allow_xpu=True
)


device_types = ("cpu", GPU_TYPE)
instantiate_device_type_tests(
    TestDecomp, globals(), only_for=device_types, allow_xpu=True
)

if __name__ == "__main__":
    # We don't support torch.compile() on Windows
    if not IS_WINDOWS:
        run_tests()
