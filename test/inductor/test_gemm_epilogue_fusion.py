# Owner(s): ["module: inductor"]

from unittest import mock

import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from torch._C import FileCheck
from torch._higher_order_ops import (
    addmm_epilogue,
    baddbmm_epilogue,
    bmm_epilogue,
    gemm_epilogue_fusion,
    grouped_mm_epilogue,
    matmul_epilogue,
    mm_epilogue,
)
from torch._higher_order_ops.gemm_epilogue import (
    mx_e8m0_scale,
    nvfp4_e2m1_pack,
    nvfp4_e4m3_scale,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.fx.experimental.proxy_tensor import make_fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_cuda import (
    _get_torch_cuda_version,
    PLATFORM_SUPPORTS_FP8,
    PLATFORM_SUPPORTS_MX_GEMM,
)
from torch.testing._internal.common_quantized import (
    _bfloat16_to_float4_e2m1fn_x2,
    ceil_div,
    to_blocked,
)
from torch.testing._internal.inductor_utils import (
    _quantize_blockwise,
    _quantize_tensorwise,
)
from torch.testing._internal.triton_utils import requires_cuda_and_triton
from torch.utils._triton import has_triton_tma_device


class GemmEpilogueFusionTests(TestCase):
    def _grouped_mm_inputs(self, op, device):
        dtype = torch.bfloat16
        match op:
            case "2d/2d":
                m, n, k, groups = 64, 128, 64, 2
                a = torch.randn(m, k * groups, device=device, dtype=dtype)
                b = torch.randn(n, k * groups, device=device, dtype=dtype).t()
                offs = torch.arange(
                    k, k * groups + 1, k, device=device, dtype=torch.int32
                )
            case "2d/3d":
                groups, m, n, k = 2, 64, 128, 64
                a = torch.randn(m * groups, k, device=device, dtype=dtype)
                b = torch.randn(groups, n, k, device=device, dtype=dtype).transpose(
                    -2, -1
                )
                offs = torch.arange(
                    m, m * groups + 1, m, device=device, dtype=torch.int32
                )
            case "3d/2d":
                groups, m, n, k = 2, 64, 128, 64
                a = torch.randn(groups, m, k, device=device, dtype=dtype)
                b = torch.randn(n * groups, k, device=device, dtype=dtype).t()
                offs = torch.arange(
                    n, n * groups + 1, n, device=device, dtype=torch.int32
                )
            case "3d/3d":
                groups, m, n, k = 2, 64, 128, 64
                a = torch.randn(groups, m, k, device=device, dtype=dtype)
                b = torch.randn(groups, n, k, device=device, dtype=dtype).transpose(
                    -2, -1
                )
                offs = None
            case _:
                raise AssertionError(f"invalid grouped_mm op shape: {op}")
        return a, b, offs

    def _check_code(self, code, checks=(), check_nots=()):
        file_check = FileCheck()
        for pattern in checks:
            file_check = file_check.check(pattern)
        for pattern in check_nots:
            file_check = file_check.check_not(pattern)
        file_check.run(code)

    def _assert_compiled_matches(
        self,
        fn,
        *args,
        atol=1e-2,
        rtol=1e-2,
        checks=(),
        check_nots=(),
        contains=(),
        assert_fn=None,
    ):
        actual, codes = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), *args
        )
        expected = fn(*args)
        if assert_fn is None:
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        else:
            assert_fn(actual, expected)
        code = "\n".join(codes)
        self._check_code(code, checks=checks, check_nots=check_nots)
        for pattern in contains:
            self.assertIn(pattern, code)
        return actual, expected, code

    def _quack_codegen_invariants(self, *checks, extern="extern_kernels.mm"):
        return ("@cute.jit", "gemm_epilogue(", *checks), (
            "call_quack_gemm_epilogue",
            "torch.ops.flex_gemm",
            extern,
        )

    def _local_reduce_invariants(self, *, group, dim, op=None, feeds_main=False):
        checks = [f"local_reduce_group={group}", f"local_reduce_dim={dim}"]
        if op is not None:
            checks.append(f"local_reduce_op='{op}'")
        if feeds_main:
            checks.append("local_reduce_feeds_main=True")
        return tuple(checks)

    def _tensorssa_grouped_n_invariants(self, group):
        return (f"acc.reshape(((1, {group}, {32 // group}), 1, 1))",)

    def _block_scaled_numel(self, rows, reduction_size, block_size):
        num_k_blocks = ceil_div(reduction_size, block_size)
        return 128 * ceil_div(rows, 128) * (ceil_div(num_k_blocks, 4) * 4)

    def _assert_quack_tuple_epilogue(
        self,
        fn,
        *args,
        main_atol=1e-2,
        main_rtol=1e-2,
        aux_atol=1e-1,
        aux_rtol=1e-2,
        checks=(),
        check_nots=("extern_kernels.mm",),
    ):
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), *args
        )
        expected = fn(*args)

        torch.testing.assert_close(actual[0], expected[0], atol=main_atol, rtol=main_rtol)
        torch.testing.assert_close(actual[1], expected[1], atol=aux_atol, rtol=aux_rtol)
        self._check_code(code, checks=checks, check_nots=check_nots)
        return actual, expected, code

    def test_eager_matches_reference(self):
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)

        actual = gemm_epilogue_fusion(
            torch.ops.aten.mm.default,
            (a, b),
            lambda acc: acc.relu(),
        )

        torch.testing.assert_close(actual, (a @ b).relu())

    def test_rejects_tensor_gemm_kwargs(self):
        bias = torch.randn(2, 4)
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)

        with self.assertRaisesRegex(RuntimeError, "gemm_kwargs must not contain tensor"):
            gemm_epilogue_fusion(
                torch.ops.aten.addmm.default,
                (bias, a, b),
                lambda acc: acc.relu(),
                gemm_kwargs={"alpha": torch.tensor(1.0)},
            )

    def test_scaled_mm_v2_rejects_duplicate_kwarg_aliases(self):
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)
        scale_a = torch.ones(())
        scale_b = torch.ones(())

        with self.assertRaisesRegex(RuntimeError, "cannot specify both"):
            gemm_epilogue_fusion(
                torch.ops.aten._scaled_mm_v2.default,
                (a, b, scale_a, scale_b),
                lambda acc: acc.relu(),
                gemm_kwargs={
                    "out_dtype": torch.float16,
                    "output_dtype": torch.bfloat16,
                },
            )

    def test_shorthand_mm_eager_matches_reference(self):
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)

        actual = gemm_epilogue_fusion(
            torch.mm,
            (a, b),
            lambda acc: acc.relu(),
        )

        torch.testing.assert_close(actual, (a @ b).relu())

    def test_rejects_unsupported_cutedsl_backend(self):
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)

        with self.assertRaisesRegex(RuntimeError, "unsupported GEMM epilogue backend"):
            gemm_epilogue_fusion(
                torch.mm,
                (a, b),
                lambda acc: acc.relu(),
                kernel_options={"backend": "CUTEDSL"},
            )

    def test_tuple_epilogue_eager_matches_reference(self):
        M = 2
        a = torch.randn(M, 3)
        b = torch.randn(3, 64)

        actual = gemm_epilogue_fusion(
            torch.ops.aten.mm.default,
            (a, b),
            lambda acc: (acc.relu(), acc.float().view(M, -1, 32).sum(-1)),
        )
        expected = a @ b

        torch.testing.assert_close(actual[0], expected.relu())
        torch.testing.assert_close(actual[1], expected.float().view(M, -1, 32).sum(-1))

    def test_addmm_eager_matches_reference(self):
        bias = torch.randn(2, 4)
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)

        actual = gemm_epilogue_fusion(
            torch.ops.aten.addmm.default,
            (bias, a, b),
            lambda acc: acc.relu(),
        )

        torch.testing.assert_close(actual, torch.addmm(bias, a, b).relu())

    def test_bmm_eager_matches_reference(self):
        a = torch.randn(5, 2, 3)
        b = torch.randn(5, 3, 4)

        actual = gemm_epilogue_fusion(
            torch.ops.aten.bmm.default,
            (a, b),
            lambda acc: acc.relu(),
        )

        torch.testing.assert_close(actual, torch.bmm(a, b).relu())

    def test_baddbmm_eager_matches_reference(self):
        bias = torch.randn(5, 2, 4)
        a = torch.randn(5, 2, 3)
        b = torch.randn(5, 3, 4)

        actual = gemm_epilogue_fusion(
            torch.ops.aten.baddbmm.default,
            (bias, a, b),
            lambda acc: acc.relu(),
        )

        torch.testing.assert_close(actual, torch.baddbmm(bias, a, b).relu())

    def test_convenience_wrappers_eager_match_reference(self):
        bias = torch.randn(2, 4)
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)
        batch_bias = torch.randn(5, 2, 4)
        batch_a = torch.randn(5, 2, 3)
        batch_b = torch.randn(5, 3, 4)

        torch.testing.assert_close(
            mm_epilogue(a, b, lambda acc: acc.relu()), (a @ b).relu()
        )
        torch.testing.assert_close(
            addmm_epilogue(bias, a, b, lambda acc: acc.relu(), alpha=0.5, beta=0.25),
            torch.addmm(bias, a, b, alpha=0.5, beta=0.25).relu(),
        )
        torch.testing.assert_close(
            bmm_epilogue(batch_a, batch_b, lambda acc: acc.relu()),
            torch.bmm(batch_a, batch_b).relu(),
        )
        torch.testing.assert_close(
            baddbmm_epilogue(
                batch_bias,
                batch_a,
                batch_b,
                lambda acc: acc.relu(),
                alpha=0.5,
                beta=0.25,
            ),
            torch.baddbmm(batch_bias, batch_a, batch_b, alpha=0.5, beta=0.25).relu(),
        )
        torch.testing.assert_close(
            matmul_epilogue(a, b, lambda acc: acc.relu()), torch.matmul(a, b).relu()
        )
        torch.testing.assert_close(
            matmul_epilogue(batch_a, batch_b, lambda acc: acc.relu()),
            torch.matmul(batch_a, batch_b).relu(),
        )

    def test_matmul_epilogue_rejects_non_gemm_shapes(self):
        with self.assertRaisesRegex(NotImplementedError, "2D mm and 3D bmm"):
            matmul_epilogue(torch.randn(3), torch.randn(3), lambda acc: acc.relu())

    def test_grouped_mm_epilogue_eager_matches_reference(self):
        for op in ("2d/2d", "2d/3d", "3d/2d", "3d/3d"):
            with self.subTest(op=op):
                a, b, offs = self._grouped_mm_inputs(op, "cpu")

                actual = grouped_mm_epilogue(a, b, lambda acc: acc.relu(), offs=offs)

                torch.testing.assert_close(actual, F.grouped_mm(a, b, offs=offs).relu())

    def test_make_fx_preserves_gemm_kwargs(self):
        def fn(bias, a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.addmm.default,
                (bias, a, b),
                lambda acc: acc.relu(),
                gemm_kwargs={"alpha": 0.5, "beta": 0.25},
            )

        bias = torch.randn(2, 4)
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)

        actual = make_fx(fn)(bias, a, b)(bias, a, b)

        torch.testing.assert_close(
            actual, torch.addmm(bias, a, b, alpha=0.5, beta=0.25).relu()
        )

    def test_make_fx_preserves_tuple_epilogue(self):
        M = 2

        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: (acc.relu(), acc.float().view(M, -1, 32).sum(-1)),
            )

        a = torch.randn(M, 3)
        b = torch.randn(3, 64)

        gm = make_fx(fn)(a, b)
        actual = gm(a, b)
        expected = a @ b

        torch.testing.assert_close(actual[0], expected.relu())
        torch.testing.assert_close(actual[1], expected.float().view(M, -1, 32).sum(-1))
        body = next(
            module
            for name, module in gm.named_modules()
            if name.startswith("gemm_epilogue_fusion_body_graph")
        )
        FileCheck().check("aten.mm.default").check("relu").check("view").check(
            "sum"
        ).run(body.code)

    def test_exports_as_gemm_epilogue_fusion_region(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: acc.relu(),
            )

        gm, _ = torch._dynamo.export(fn, torch.randn(2, 3), torch.randn(3, 4))

        self.assertIn("gemm_epilogue_fusion", gm.code)
        self.assertIn("{'backend': 'TRITON', 'SPLIT_K': False}", gm.code)
        self.assertIn("aten.mm.default", gm.gemm_epilogue_fusion_body_0.code)
        self.assertIn("relu", gm.gemm_epilogue_fusion_body_0.code)

    def test_shorthand_mm_exports_as_gemm_epilogue_fusion_region(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.mm,
                (a, b),
                lambda acc: acc.relu(),
            )

        gm, _ = torch._dynamo.export(fn, torch.randn(2, 3), torch.randn(3, 4))

        self.assertIn("gemm_epilogue_fusion", gm.code)
        self.assertIn("aten.mm.default", gm.code)
        self.assertIn("aten.mm.default", gm.gemm_epilogue_fusion_body_0.code)

    @requires_cuda_and_triton
    @inductor_config.patch(max_autotune_gemm=False)
    def test_cuda_inductor_grouped_mm_epilogue_fuses(self):
        if torch.cuda.get_device_capability() < (9, 0):
            self.skipTest("TRITON grouped_mm template requires SM90+")

        def fn(a, b, offs):
            return grouped_mm_epilogue(a, b, lambda acc: acc.relu(), offs=offs)

        for op in ("2d/2d", "2d/3d", "3d/2d", "3d/3d"):
            with self.subTest(op=op):
                a, b, offs = self._grouped_mm_inputs(op, "cuda")

                actual, (code,) = run_and_get_code(
                    torch.compile(fn, backend="inductor", fullgraph=True), a, b, offs
                )

                torch.testing.assert_close(actual, fn(a, b, offs), atol=1e-2, rtol=1e-2)
                FileCheck().check("triton_").check("grouped_mm").check_not(
                    "extern_kernels._grouped_mm"
                ).run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_routes_grouped_mm_relu_to_quack_hook(self):
        def fn(a, b, offs):
            return grouped_mm_epilogue(
                a,
                b,
                lambda acc: acc.relu(),
                offs=offs,
                kernel_options={"backend": "QUACK"},
            )

        ops = ["2d/3d", "2d/2d"]
        if torch.cuda.get_device_capability() >= (10, 0):
            ops.append("3d/2d")
        for op in ops:
            with self.subTest(op=op):
                a, b, offs = self._grouped_mm_inputs(op, "cuda")
                if op == "2d/2d":
                    a = a.t().contiguous().t()
                    b = b.contiguous()
                elif op == "3d/2d":
                    b = b.contiguous()

                actual, (code,) = run_and_get_code(
                    torch.compile(fn, backend="inductor", fullgraph=True), a, b, offs
                )

                torch.testing.assert_close(actual, fn(a, b, offs), atol=1e-2, rtol=1e-2)
                FileCheck().check("@cute.jit").check("gemm_epilogue(").check(
                    "offs="
                ).check_not("extern_kernels._grouped_mm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_routes_3d_2d_grouped_mm_partial_n(self):
        if torch.cuda.get_device_capability() < (10, 0):
            self.skipTest("QUACK 3D/2D grouped_mm requires SM100+")

        def fn(a, b, offs):
            return grouped_mm_epilogue(
                a,
                b,
                lambda acc: acc.relu(),
                offs=offs,
                kernel_options={"backend": "QUACK"},
            )

        groups, m, n, k = 2, 64, 128, 64
        a = torch.randn(groups, m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n * groups + 64, device="cuda", dtype=torch.bfloat16)
        offs = torch.tensor([96, n * groups], device="cuda", dtype=torch.int32)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, offs
        )

        expected = fn(a, b, offs)
        covered_n = int(offs[-1].item())
        torch.testing.assert_close(
            actual[:, :covered_n], expected[:, :covered_n], atol=1e-2, rtol=1e-2
        )
        FileCheck().check("@cute.jit").check("gemm_epilogue(").check(
            "offs="
        ).check_not("extern_kernels._grouped_mm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_routes_3d_2d_grouped_mm_symbolic_layout(self):
        if torch.cuda.get_device_capability() < (10, 0):
            self.skipTest("QUACK 3D/2D grouped_mm requires SM100+")

        def fn(a, b, offs):
            return grouped_mm_epilogue(
                a,
                b,
                lambda acc: acc.relu(),
                offs=offs,
                kernel_options={"backend": "QUACK"},
            )

        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)
        for groups, m, n, k in ((2, 64, 128, 64), (4, 128, 128, 128)):
            with self.subTest(groups=groups, m=m, k=k):
                a = torch.randn(groups, m, k, device="cuda", dtype=torch.bfloat16)
                b = torch.randn(k, n * groups, device="cuda", dtype=torch.bfloat16)
                offs = torch.arange(n, n * groups + 1, n, device="cuda", dtype=torch.int32)

                actual = compiled(a, b, offs)

                torch.testing.assert_close(actual, fn(a, b, offs), atol=1e-2, rtol=1e-2)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_rejects_3d_2d_grouped_mm_unaligned_n(self):
        if torch.cuda.get_device_capability() < (10, 0):
            self.skipTest("QUACK 3D/2D grouped_mm requires SM100+")

        def fn(a, b, offs):
            return grouped_mm_epilogue(
                a,
                b,
                lambda acc: acc.relu(),
                offs=offs,
                kernel_options={"backend": "QUACK"},
            )

        groups, m, n, k = 2, 64, 128, 64
        a = torch.randn(groups, m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n * groups, device="cuda", dtype=torch.bfloat16)
        offs = torch.tensor([65, n * groups], device="cuda", dtype=torch.int32)

        with self.assertRaisesRegex(Exception, "tile_n-aligned"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b, offs)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_rejects_3d_2d_grouped_mm_unsupported_layout(self):
        if torch.cuda.get_device_capability() < (10, 0):
            self.skipTest("QUACK 3D/2D grouped_mm requires SM100+")

        def fn(a, b, offs):
            return grouped_mm_epilogue(
                a,
                b,
                lambda acc: acc.relu(),
                offs=offs,
                kernel_options={"backend": "QUACK"},
            )

        a, b, offs = self._grouped_mm_inputs("3d/2d", "cuda")

        with self.assertRaisesRegex(Exception, "contiguous N"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b, offs)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_grouped_mm_grouped_contract_fuses(self):
        M = 64
        N = 64
        group = 2

        def fn(a, b, offs):
            def epilogue(acc):
                lanes = acc.float().view(M * 2, -1, group)
                return (torch.relu(lanes[..., 0]) * lanes[..., 1]).to(acc.dtype)

            return grouped_mm_epilogue(
                a,
                b,
                epilogue,
                offs=offs,
                kernel_options={"backend": "QUACK"},
            )

        a, b, offs = self._grouped_mm_inputs("2d/3d", "cuda")

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, offs
        )
        expected = fn(a, b, offs)

        self.assertEqual(actual.shape, (M * 2, N))
        torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-1)
        FileCheck().check("offs=").check("main_output_transform='grouped_n_contract'").check(
            "main_output_transform_group=2"
        ).check_not("extern_kernels._grouped_mm").run(code)

    def _run_split_k_epilogue_test(self, backend, expected_kernel):
        if backend == "QUACK":
            try:
                import quack  # noqa: F401
            except ImportError:
                self.skipTest("QuACK is not available")

        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: acc.relu(),
                kernel_options={"backend": backend, "SPLIT_K": True},
            )

        a = torch.randn(8, 8192, device="cuda", dtype=torch.float16)
        b = torch.randn(8192, 8, device="cuda", dtype=torch.float16)

        actual, codes = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        torch.testing.assert_close(actual, fn(a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check(expected_kernel).run("\n".join(codes))

    @requires_cuda_and_triton
    @inductor_config.patch("triton.num_decompose_k_splits", 2)
    @inductor_config.patch("triton.decompose_k_threshold", 0)
    def test_cuda_inductor_triton_epilogue_split_k_allows_decompose_k(self):
        self._run_split_k_epilogue_test("TRITON", "decompose_k_fp32_mm")

    @requires_cuda_and_triton
    @inductor_config.patch("triton.num_decompose_k_splits", 2)
    @inductor_config.patch("triton.decompose_k_threshold", 0)
    def test_cuda_inductor_quack_epilogue_split_k_uses_quack_bmm(self):
        self._run_split_k_epilogue_test("QUACK", "quack_gemm")

    def _run_split_k_addmm_epilogue_test(self, backend, expected_kernel):
        if backend == "QUACK":
            try:
                import quack  # noqa: F401
            except ImportError:
                self.skipTest("QuACK is not available")

        def fn(bias, a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.addmm.default,
                (bias, a, b),
                lambda acc: acc.relu(),
                gemm_kwargs={"alpha": 0.5, "beta": 1.25},
                kernel_options={"backend": backend, "SPLIT_K": True},
            )

        bias = torch.randn(8, 8, device="cuda", dtype=torch.float16)
        a = torch.randn(8, 8192, device="cuda", dtype=torch.float16)
        b = torch.randn(8192, 8, device="cuda", dtype=torch.float16)

        actual, codes = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), bias, a, b
        )

        torch.testing.assert_close(actual, fn(bias, a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check(expected_kernel).run("\n".join(codes))

    @requires_cuda_and_triton
    @inductor_config.patch("triton.num_decompose_k_splits", 2)
    @inductor_config.patch("triton.decompose_k_threshold", 0)
    def test_cuda_inductor_triton_addmm_epilogue_split_k(self):
        self._run_split_k_addmm_epilogue_test("TRITON", "decompose_k_fp32_mm")

    @requires_cuda_and_triton
    @inductor_config.patch("triton.num_decompose_k_splits", 2)
    @inductor_config.patch("triton.decompose_k_threshold", 0)
    def test_cuda_inductor_quack_addmm_epilogue_split_k(self):
        self._run_split_k_addmm_epilogue_test("QUACK", "quack_gemm")

    @requires_cuda_and_triton
    def test_cuda_inductor_triton_epilogue_reads_closure_tensor(self):
        def fn(a, b, bias):
            return mm_epilogue(
                a,
                b,
                lambda acc: (acc + bias).relu(),
                kernel_options={"backend": "TRITON"},
            )

        a = torch.randn(16, 32, device="cuda", dtype=torch.float16)
        b = torch.randn(32, 16, device="cuda", dtype=torch.float16)
        bias = torch.randn(16, 16, device="cuda", dtype=torch.float16)

        actual, codes = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, bias
        )

        torch.testing.assert_close(actual, fn(a, b, bias), atol=1e-2, rtol=1e-2)
        FileCheck().check("triton_tem_fused").check("tl.load").check_not(
            "extern_kernels.mm"
        ).run("\n".join(codes))

    @requires_cuda_and_triton
    def test_cuda_inductor_triton_epilogue_reads_broadcast_closure_tensor(self):
        def fn(a, b, bias):
            return mm_epilogue(
                a,
                b,
                lambda acc: (acc + bias).relu(),
                kernel_options={"backend": "TRITON"},
            )

        a = torch.randn(16, 32, device="cuda", dtype=torch.float16)
        b = torch.randn(32, 16, device="cuda", dtype=torch.float16)
        bias = torch.randn(16, device="cuda", dtype=torch.float16)

        actual, codes = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, bias
        )

        torch.testing.assert_close(actual, fn(a, b, bias), atol=1e-2, rtol=1e-2)
        FileCheck().check("triton_tem_fused").check("tl.load").check_not(
            "extern_kernels.mm"
        ).run("\n".join(codes))

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_epilogue_reads_full_tile_closure_tensor(self):
        try:
            import quack  # noqa: F401
        except ImportError:
            self.skipTest("QuACK is not available")

        def fn(a, b, scale):
            return mm_epilogue(
                a,
                b,
                lambda acc: acc * scale,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(16, 32, device="cuda", dtype=torch.float16)
        b = torch.randn(32, 16, device="cuda", dtype=torch.float16)
        scale = torch.randn(16, 16, device="cuda", dtype=torch.float16)

        actual, codes = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, scale
        )

        torch.testing.assert_close(actual, fn(a, b, scale), atol=1e-2, rtol=1e-2)
        FileCheck().check("@cute.jit").check("aux0").check(
            "epilogue_args=("
        ).check("epilogue_arg_kinds=('tile',)").run("\n".join(codes))

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_epilogue_fast_math_codegen(self):
        try:
            import quack  # noqa: F401
        except ImportError:
            self.skipTest("QuACK is not available")

        def fn(a, b):
            return mm_epilogue(
                a,
                b,
                lambda acc: (
                    torch.tanh(acc)
                    + torch.sigmoid(acc)
                    + F.silu(acc)
                    + F.gelu(acc)
                    + F.softplus(acc)
                    + torch.log1p(acc.abs())
                ),
                kernel_options={"backend": "QUACK", "fast_math": True},
            )

        a = torch.randn(16, 32, device="cuda", dtype=torch.float16)
        b = torch.randn(32, 16, device="cuda", dtype=torch.float16)

        actual, codes = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        torch.testing.assert_close(actual, fn(a, b), atol=2e-1, rtol=2e-2)
        FileCheck().check("@cute.jit").check("fastmath=True").check(
            "cute.math.exp"
        ).check("cute.math.log").check("cute.math.tanh").check_not("cute.math.erf").run(
            "\n".join(codes)
        )

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_epilogue_fast_math_quick_gelu_uses_tanh_identity(self):
        try:
            import quack  # noqa: F401
        except ImportError:
            self.skipTest("QuACK is not available")

        def fn(a, b):
            return mm_epilogue(
                a,
                b,
                lambda acc: acc * torch.sigmoid(1.702 * acc),
                kernel_options={"backend": "QUACK", "fast_math": True},
            )

        a = torch.randn(16, 32, device="cuda", dtype=torch.float16)
        b = torch.randn(32, 16, device="cuda", dtype=torch.float16)

        actual, codes = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        torch.testing.assert_close(actual, fn(a, b), atol=2e-1, rtol=2e-2)
        FileCheck().check("@cute.jit").check("cute.math.tanh").check(
            "fastmath=True"
        ).check_not("cute.math.exp").run("\n".join(codes))

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_epilogue_fast_math_silu_uses_tanh_identity(self):
        try:
            import quack  # noqa: F401
        except ImportError:
            self.skipTest("QuACK is not available")

        def fn(a, b):
            return mm_epilogue(
                a,
                b,
                lambda acc: F.silu(acc),
                kernel_options={"backend": "QUACK", "fast_math": True},
            )

        a = torch.randn(16, 32, device="cuda", dtype=torch.float16)
        b = torch.randn(32, 16, device="cuda", dtype=torch.float16)

        actual, codes = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        torch.testing.assert_close(actual, fn(a, b), atol=2e-1, rtol=2e-2)
        FileCheck().check("@cute.jit").check("cute.math.tanh").check(
            "fastmath=True"
        ).check_not("cute.math.exp").run("\n".join(codes))

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_epilogue_tuned_kernel_option_codegen(self):
        try:
            import quack  # noqa: F401
        except ImportError:
            self.skipTest("QuACK is not available")

        a = torch.randn(16, 32, device="cuda", dtype=torch.float16)
        b = torch.randn(32, 16, device="cuda", dtype=torch.float16)

        for tuned in (False, True):
            with self.subTest(tuned=tuned):

                def fn(a, b):
                    return mm_epilogue(
                        a,
                        b,
                        lambda acc: acc.relu(),
                        kernel_options={"backend": "QUACK", "tuned": tuned},
                    )

                actual, codes = run_and_get_code(
                    torch.compile(fn, backend="inductor", fullgraph=True), a, b
                )

                torch.testing.assert_close(actual, fn(a, b), atol=1e-2, rtol=1e-2)
                FileCheck().check("@cute.jit").check("gemm_epilogue(").check(
                    f"tuned={tuned}"
                ).run("\n".join(codes))

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_epilogue_reads_broadcast_closure_tensor(self):
        try:
            import quack  # noqa: F401
        except ImportError:
            self.skipTest("QuACK is not available")

        def fn(a, b, scale):
            return mm_epilogue(
                a,
                b,
                lambda acc: (acc * scale).relu(),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(16, 32, device="cuda", dtype=torch.float16)
        b = torch.randn(32, 16, device="cuda", dtype=torch.float16)
        cases = (
            ("row", torch.randn(1, 16, device="cuda", dtype=torch.float16)),
            ("col", torch.randn(16, 1, device="cuda", dtype=torch.float16)),
        )

        for kind, scale in cases:
            with self.subTest(kind=kind):
                actual, codes = run_and_get_code(
                    torch.compile(fn, backend="inductor", fullgraph=True), a, b, scale
                )

                torch.testing.assert_close(actual, fn(a, b, scale), atol=1e-2, rtol=1e-2)
                FileCheck().check("@cute.jit").check("aux0").check(
                    "epilogue_args=("
                ).check(f"epilogue_arg_kinds=('{kind}',)").run("\n".join(codes))

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_epilogue_reads_many_broadcast_closure_tensors(self):
        try:
            import quack  # noqa: F401
        except ImportError:
            self.skipTest("QuACK is not available")

        def fn(a, b, row0, col1, row2, col3, row4):
            return mm_epilogue(
                a,
                b,
                lambda acc: (((acc.float() * row0 + col1) * row2 + col3) * row4).relu(),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(16, 32, device="cuda", dtype=torch.float16)
        b = torch.randn(32, 16, device="cuda", dtype=torch.float16)
        row0 = torch.randn(1, 16, device="cuda", dtype=torch.float32)
        col1 = torch.randn(16, 1, device="cuda", dtype=torch.float32)
        row2 = torch.randn(1, 16, device="cuda", dtype=torch.float32)
        col3 = torch.randn(16, 1, device="cuda", dtype=torch.float32)
        row4 = torch.randn(1, 16, device="cuda", dtype=torch.float32)

        actual, codes = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True),
            a,
            b,
            row0,
            col1,
            row2,
            col3,
            row4,
        )

        torch.testing.assert_close(
            actual, fn(a, b, row0, col1, row2, col3, row4), atol=1e-2, rtol=1e-2
        )
        FileCheck().check("@cute.jit").check("aux0").check("aux4").check(
            "epilogue_args=("
        ).check("epilogue_arg_kinds=('row', 'col', 'row', 'col', 'row')").check_not(
            "extern_kernels.mm"
        ).run("\n".join(codes))

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_epilogue_reads_two_tile_closure_tensors(self):
        try:
            import quack  # noqa: F401
        except ImportError:
            self.skipTest("QuACK is not available")

        def fn(a, b, tile0, tile1):
            return mm_epilogue(
                a,
                b,
                lambda acc: (acc.float() + tile0 * tile1).relu(),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(16, 32, device="cuda", dtype=torch.float16)
        b = torch.randn(32, 16, device="cuda", dtype=torch.float16)
        tile0 = torch.randn(16, 16, device="cuda", dtype=torch.float32)
        tile1 = torch.randn(16, 16, device="cuda", dtype=torch.float32)

        actual, codes = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, tile0, tile1
        )

        torch.testing.assert_close(actual, fn(a, b, tile0, tile1), atol=1e-2, rtol=1e-2)
        FileCheck().check("@cute.jit").check("aux0").check("aux1").check(
            "epilogue_args=("
        ).check("epilogue_arg_kinds=('tile', 'tile')").check_not(
            "extern_kernels.mm"
        ).run("\n".join(codes))

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_epilogue_contiguous_gated_concat_layout(self):
        try:
            import quack  # noqa: F401
        except ImportError:
            self.skipTest("QuACK is not available")

        m, n, k = 16, 16, 32

        def view_select_fn(a, b):
            return mm_epilogue(
                a,
                b,
                lambda acc: F.silu(acc.view(m, 2, n).select(1, 0))
                * acc.view(m, 2, n).select(1, 1),
                kernel_options={"backend": "QUACK", "fast_math": True},
            )

        def chunk_fn(a, b):
            def ep(acc):
                gate, up = acc.chunk(2, dim=-1)
                return F.silu(gate) * up

            return mm_epilogue(
                a,
                b,
                ep,
                kernel_options={"backend": "QUACK", "fast_math": True},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(2 * n, k, device="cuda", dtype=torch.float16).t()

        for fn in (view_select_fn, chunk_fn):
            with self.subTest(fn=fn.__name__):
                actual, codes = run_and_get_code(
                    torch.compile(fn, backend="inductor", fullgraph=True), a, b
                )

                torch.testing.assert_close(actual, fn(a, b), atol=2e-1, rtol=2e-2)
                FileCheck().check("main_output_transform='grouped_n_contract'").check(
                    "concat_layout=('B',)"
                ).check("@cute.jit").check("cute.math.tanh").run("\n".join(codes))

    @requires_cuda_and_triton
    def test_cuda_inductor_triton_epilogue_reads_mask_closure_tensor(self):
        def fn(a, b, mask):
            return mm_epilogue(
                a,
                b,
                lambda acc: torch.where(mask, acc, torch.zeros_like(acc)),
                kernel_options={"backend": "TRITON"},
            )

        a = torch.randn(16, 32, device="cuda", dtype=torch.float16)
        b = torch.randn(32, 16, device="cuda", dtype=torch.float16)
        mask = torch.randn(16, 16, device="cuda") > 0

        actual, codes = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, mask
        )

        torch.testing.assert_close(actual, fn(a, b, mask), atol=1e-2, rtol=1e-2)
        FileCheck().check("triton_tem_fused").check("tl.load").check_not(
            "extern_kernels.mm"
        ).run("\n".join(codes))

    def _run_split_k_epilogue_closure_read_test(self, backend, expected_kernel):
        if backend == "QUACK":
            try:
                import quack  # noqa: F401
            except ImportError:
                self.skipTest("QuACK is not available")

        def fn(a, b, row_scale):
            return mm_epilogue(
                a,
                b,
                lambda acc: acc * row_scale,
                kernel_options={"backend": backend, "SPLIT_K": True},
            )

        a = torch.randn(8, 8192, device="cuda", dtype=torch.float16)
        b = torch.randn(8192, 8, device="cuda", dtype=torch.float16)
        row_scale = torch.randn(8, 1, device="cuda", dtype=torch.float16)

        actual, codes = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, row_scale
        )

        torch.testing.assert_close(actual, fn(a, b, row_scale), atol=1e-1, rtol=1e-2)
        FileCheck().check(expected_kernel).check("tl.load").run("\n".join(codes))

    @requires_cuda_and_triton
    @inductor_config.patch("triton.num_decompose_k_splits", 2)
    @inductor_config.patch("triton.decompose_k_threshold", 0)
    def test_cuda_inductor_triton_split_k_epilogue_reads_closure_tensor(self):
        self._run_split_k_epilogue_closure_read_test("TRITON", "decompose_k_fp32_mm")

    @requires_cuda_and_triton
    @inductor_config.patch("triton.num_decompose_k_splits", 2)
    @inductor_config.patch("triton.decompose_k_threshold", 0)
    def test_cuda_inductor_quack_split_k_epilogue_reads_closure_tensor(self):
        self._run_split_k_epilogue_closure_read_test("QUACK", "quack_gemm")

    @requires_cuda_and_triton
    @inductor_config.patch(max_autotune_gemm=False)
    @inductor_config.patch("triton.num_decompose_k_splits", 2)
    @inductor_config.patch("triton.decompose_k_threshold", 0)
    def test_cuda_inductor_triton_epilogue_disables_decompose_k(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: acc.relu(),
            )

        a = torch.randn(16, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, 16, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        torch.testing.assert_close(actual, fn(a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("triton_").check_not("decompose_k_mm").run(code)

    @requires_cuda_and_triton
    @inductor_config.patch(max_autotune_gemm=False)
    def test_cuda_inductor_forces_template_epilogue_fusion(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: acc.relu(),
            )

        a = torch.randn(128, 128, device="cuda")
        b = torch.randn(128, 128, device="cuda")

        actual = torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

        torch.testing.assert_close(actual, fn(a, b))
        self.assertFalse(inductor_config.max_autotune_gemm)

    @requires_cuda_and_triton
    @inductor_config.patch(max_autotune_gemm=False)
    def test_cuda_inductor_addmm_epilogue_fuses(self):
        def fn(bias, a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.addmm.default,
                (bias, a, b),
                lambda acc: acc.relu(),
            )

        bias = torch.randn(128, 64, device="cuda", dtype=torch.float16)
        a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), bias, a, b
        )

        torch.testing.assert_close(actual, fn(bias, a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("triton_").check_not("extern_kernels.addmm").run(code)

    @requires_cuda_and_triton
    @inductor_config.patch(max_autotune_gemm=False)
    def test_cuda_inductor_addmm_alpha_beta_epilogue_fuses(self):
        def fn(bias, a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.addmm.default,
                (bias, a, b),
                lambda acc: acc.relu(),
                gemm_kwargs={"alpha": 0.5, "beta": 0.25},
            )

        bias = torch.randn(128, 64, device="cuda", dtype=torch.float16)
        a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), bias, a, b
        )

        torch.testing.assert_close(actual, fn(bias, a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("triton_").check_not("extern_kernels.addmm").run(code)

    @requires_cuda_and_triton
    @inductor_config.patch(max_autotune_gemm=False)
    def test_cuda_inductor_bmm_epilogue_fuses(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.bmm.default,
                (a, b),
                lambda acc: acc.relu(),
            )

        a = torch.randn(4, 64, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(4, 128, 32, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        torch.testing.assert_close(actual, fn(a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("triton_").check_not("extern_kernels.bmm").run(code)

    @requires_cuda_and_triton
    @inductor_config.patch(max_autotune_gemm=False)
    def test_cuda_inductor_baddbmm_epilogue_fuses(self):
        def fn(bias, a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.baddbmm.default,
                (bias, a, b),
                lambda acc: acc.relu(),
            )

        bias = torch.randn(4, 64, 32, device="cuda", dtype=torch.float16)
        a = torch.randn(4, 64, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(4, 128, 32, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), bias, a, b
        )

        torch.testing.assert_close(actual, fn(bias, a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("triton_").check_not("extern_kernels.baddbmm").run(code)

    @requires_cuda_and_triton
    @inductor_config.patch(max_autotune_gemm_search_space="DEFAULT")
    def test_cuda_inductor_scaled_mm_epilogue_fuses(self):
        if not PLATFORM_SUPPORTS_FP8:
            self.skipTest("FP8 is not supported")

        def fn(a, b, scale_a, scale_b):
            return gemm_epilogue_fusion(
                torch.ops.aten._scaled_mm.default,
                (a, b, scale_a, scale_b),
                lambda acc: acc.relu(),
                gemm_kwargs={"out_dtype": torch.bfloat16},
            )

        x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        a, scale_a = _quantize_tensorwise(x, torch.float8_e4m3fn)
        w_fp8, scale_b = _quantize_tensorwise(w, torch.float8_e4m3fn)
        b = w_fp8.t()

        from torch._inductor.template_heuristics.triton import (
            CUDAScaledMMTemplateConfigHeuristic,
            GemmConfig,
        )

        original_init = CUDAScaledMMTemplateConfigHeuristic.__init__

        def init_with_single_config(self):
            original_init(self)
            self.mm_configs = [GemmConfig(64, 32, 32, 3, 4)]

        with mock.patch.object(
            CUDAScaledMMTemplateConfigHeuristic, "__init__", init_with_single_config
        ), mock.patch.object(
            CUDAScaledMMTemplateConfigHeuristic,
            "_get_acc_type",
            lambda self, dtype: "tl.float32",
        ):
            actual, (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True),
                a,
                b,
                scale_a,
                scale_b,
            )

        torch.testing.assert_close(
            actual, fn(a, b, scale_a, scale_b), atol=0.05, rtol=1e-2
        )
        FileCheck().check("triton_tem_fused__scaled_mm").check("maximum").check_not(
            "extern_kernels._scaled_mm"
        ).run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_scaled_mm_blockwise_epilogue_fuses(self):
        if not PLATFORM_SUPPORTS_FP8:
            self.skipTest("FP8 is not supported")
        if not has_triton_tma_device():
            self.skipTest("Need device-side TMA support in Triton")
        if _get_torch_cuda_version() < (12, 9):
            self.skipTest("cuBLAS blockwise scaling added in CUDA 12.9")
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("cuBLAS 128-element blockwise scaling is SM90-only")

        def fn(a, b, scale_a, scale_b):
            return gemm_epilogue_fusion(
                torch.ops.aten._scaled_mm.default,
                (a, b, scale_a, scale_b),
                lambda acc: acc.relu(),
                gemm_kwargs={"out_dtype": torch.bfloat16},
            )

        m, k, n = 128, 256, 128
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        a, scale_a = _quantize_blockwise(
            x, torch.float8_e4m3fn, block_outer=1, block_inner=128
        )
        w_fp8, scale_b = _quantize_blockwise(
            w, torch.float8_e4m3fn, block_outer=128, block_inner=128
        )
        b = w_fp8.t()
        scale_b = scale_b.t()

        with inductor_config.patch(
            {
                "triton.enable_persistent_tma_matmul": True,
                "test_configs.autotune_choice_name_regex": "triton_scaled_mm_device_tma",
                "max_autotune_gemm_backends": "TRITON",
                "max_autotune": True,
            }
        ):
            actual, (code,) = run_and_get_code(
                torch.compile(
                    fn, backend="inductor", fullgraph=True, mode="max-autotune"
                ),
                a,
                b,
                scale_a,
                scale_b,
            )

        torch.testing.assert_close(
            actual, fn(a, b, scale_a, scale_b), atol=0.05, rtol=1e-2
        )
        FileCheck().check("triton_scaled_mm_device_tma").check(
            f"SCALE_RECIPE_A : tl.constexpr = {F.ScalingType.BlockWise1x128.value}"
        ).check(
            f"SCALE_RECIPE_B : tl.constexpr = {F.ScalingType.BlockWise128x128.value}"
        ).check_not("extern_kernels._scaled_mm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_scaled_mm_v2_epilogue_applies_row_scale(self):
        if not PLATFORM_SUPPORTS_FP8:
            self.skipTest("FP8 is not supported")

        def fn(a, b, scale_a, scale_b, row_scale):
            return gemm_epilogue_fusion(
                torch.ops.aten._scaled_mm_v2.default,
                (a, b, scale_a, scale_b),
                lambda acc: acc * row_scale,
                gemm_kwargs={
                    "scale_recipe_a": F.ScalingType.TensorWise,
                    "scale_recipe_b": F.ScalingType.TensorWise,
                    "output_dtype": torch.bfloat16,
                },
            )

        x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        a, scale_a = _quantize_tensorwise(x, torch.float8_e4m3fn)
        w_fp8, scale_b = _quantize_tensorwise(w, torch.float8_e4m3fn)
        row_scale = torch.randn(64, 1, device="cuda", dtype=torch.float32)
        b = w_fp8.t()

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True),
            a,
            b,
            scale_a,
            scale_b,
            row_scale,
        )

        torch.testing.assert_close(
            actual, fn(a, b, scale_a, scale_b, row_scale), atol=0.05, rtol=1e-2
        )
        FileCheck().check("_scaled_mm_v2").check("triton_").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_scaled_mm_v2_mxfp8_block_scaled_epilogue_fuses(self):
        if not PLATFORM_SUPPORTS_MX_GEMM:
            self.skipTest("MX GEMM is not supported")

        swizzle = (
            F.SwizzleType.NO_SWIZZLE
            if torch.version.hip
            else F.SwizzleType.SWIZZLE_32_4_4
        )

        def fn(a, b, scale_a, scale_b):
            return gemm_epilogue_fusion(
                torch.ops.aten._scaled_mm_v2.default,
                (a, b, scale_a, scale_b),
                lambda acc: acc.relu(),
                gemm_kwargs={
                    "scale_recipe_a": F.ScalingType.BlockWise1x32,
                    "scale_recipe_b": F.ScalingType.BlockWise1x32,
                    "swizzle_a": swizzle,
                    "swizzle_b": swizzle,
                    "output_dtype": torch.bfloat16,
                },
            )

        m, k, n = 128, 32, 128
        a = torch.eye(m, k, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        b = (
            torch.eye(n, k, device="cuda", dtype=torch.bfloat16)
            .to(torch.float8_e4m3fn)
            .t()
        )
        scale_a = torch.full(
            (self._block_scaled_numel(m, k, 32),),
            1.0,
            device="cuda",
            dtype=torch.float8_e8m0fnu,
        )
        scale_b = torch.full(
            (self._block_scaled_numel(n, k, 32),),
            1.0,
            device="cuda",
            dtype=torch.float8_e8m0fnu,
        )

        self._assert_compiled_matches(
            fn,
            a,
            b,
            scale_a,
            scale_b,
            checks=("_scaled_mm_v2", "triton_"),
            check_nots=("extern_kernels._scaled_mm",),
        )

    @requires_cuda_and_triton
    def test_cuda_inductor_scaled_mm_v2_nvfp4_block_scaled_epilogue_fuses(self):
        if not PLATFORM_SUPPORTS_MX_GEMM or torch.version.hip:
            self.skipTest("NVFP4 GEMM is not supported")

        def fn(a, b, scale_a, global_scale, scale_b):
            return gemm_epilogue_fusion(
                torch.ops.aten._scaled_mm_v2.default,
                (a, b, [scale_a, global_scale], [scale_b, global_scale]),
                lambda acc: acc.relu(),
                gemm_kwargs={
                    "scale_recipe_a": [
                        F.ScalingType.BlockWise1x16,
                        F.ScalingType.TensorWise,
                    ],
                    "scale_recipe_b": [
                        F.ScalingType.BlockWise1x16,
                        F.ScalingType.TensorWise,
                    ],
                    "swizzle_a": [
                        F.SwizzleType.SWIZZLE_32_4_4,
                        F.SwizzleType.NO_SWIZZLE,
                    ],
                    "swizzle_b": [
                        F.SwizzleType.SWIZZLE_32_4_4,
                        F.SwizzleType.NO_SWIZZLE,
                    ],
                    "output_dtype": torch.bfloat16,
                },
            )

        m, k, n = 128, 32, 128
        a = _bfloat16_to_float4_e2m1fn_x2(
            torch.eye(m, k, device="cuda", dtype=torch.bfloat16)
        )
        b = _bfloat16_to_float4_e2m1fn_x2(
            torch.eye(n, k, device="cuda", dtype=torch.bfloat16)
        ).t()
        scale_a = torch.full(
            (self._block_scaled_numel(m, k, 16),),
            1.0,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        scale_b = torch.full(
            (self._block_scaled_numel(n, k, 16),),
            1.0,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        global_scale = torch.ones((1,), device="cuda", dtype=torch.float32)

        self._assert_compiled_matches(
            fn,
            a,
            b,
            scale_a,
            global_scale,
            scale_b,
            checks=("_scaled_mm_v2", "triton_"),
            check_nots=("extern_kernels._scaled_mm",),
        )

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_routes_scaled_mm_v2_mxfp8_to_quack(self):
        if not PLATFORM_SUPPORTS_MX_GEMM:
            self.skipTest("MX GEMM is not supported")

        swizzle = (
            F.SwizzleType.NO_SWIZZLE
            if torch.version.hip
            else F.SwizzleType.SWIZZLE_32_4_4
        )

        def fn(a, b, scale_a, scale_b):
            return gemm_epilogue_fusion(
                torch.ops.aten._scaled_mm_v2.default,
                (a, b, scale_a, scale_b),
                lambda acc: acc.relu(),
                gemm_kwargs={
                    "scale_recipe_a": F.ScalingType.BlockWise1x32,
                    "scale_recipe_b": F.ScalingType.BlockWise1x32,
                    "swizzle_a": swizzle,
                    "swizzle_b": swizzle,
                    "output_dtype": torch.bfloat16,
                },
                kernel_options={"backend": "QUACK"},
            )

        m, k, n = 128, 32, 128
        a = torch.eye(m, k, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        b = (
            torch.eye(n, k, device="cuda", dtype=torch.bfloat16)
            .to(torch.float8_e4m3fn)
            .t()
        )
        scale_a = torch.full(
            (self._block_scaled_numel(m, k, 32),),
            1.0,
            device="cuda",
            dtype=torch.float8_e8m0fnu,
        )
        scale_b = torch.full(
            (self._block_scaled_numel(n, k, 32),),
            1.0,
            device="cuda",
            dtype=torch.float8_e8m0fnu,
        )

        checks, check_nots = self._quack_codegen_invariants("scale_a=", "scale_b=")
        self._assert_compiled_matches(
            fn,
            a,
            b,
            scale_a,
            scale_b,
            checks=checks,
            check_nots=(*check_nots, "extern_kernels._scaled_mm"),
        )

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_scaled_mm_v2_fuzzes_main_epilogues(self):
        if not PLATFORM_SUPPORTS_MX_GEMM:
            self.skipTest("MX GEMM is not supported")

        swizzle = (
            F.SwizzleType.NO_SWIZZLE
            if torch.version.hip
            else F.SwizzleType.SWIZZLE_32_4_4
        )

        def fn(a, b, scale_a, scale_b, row_bias, col_scale, tile_bias, epilogue):
            return gemm_epilogue_fusion(
                torch.ops.aten._scaled_mm_v2.default,
                (a, b, scale_a, scale_b),
                lambda acc: epilogue(acc.float(), row_bias, col_scale, tile_bias),
                gemm_kwargs={
                    "scale_recipe_a": F.ScalingType.BlockWise1x32,
                    "scale_recipe_b": F.ScalingType.BlockWise1x32,
                    "swizzle_a": swizzle,
                    "swizzle_b": swizzle,
                    "output_dtype": torch.bfloat16,
                },
                kernel_options={"backend": "QUACK", "fast_math": True},
            )

        cases = (
            (
                "affine_relu",
                lambda acc, row_bias, col_scale, tile_bias: (
                    (acc + row_bias) * col_scale + tile_bias
                ).relu(),
            ),
            (
                "silu_with_broadcasts",
                lambda acc, row_bias, col_scale, tile_bias: F.silu(
                    acc * col_scale + row_bias + tile_bias
                ),
            ),
            (
                "gelu_residual",
                lambda acc, row_bias, col_scale, tile_bias: F.gelu(acc + tile_bias)
                + row_bias * col_scale,
            ),
        )

        for seed, (name, epilogue) in enumerate(cases):
            with self.subTest(name=name):
                torch.manual_seed(seed)
                m, k, n = 128, 32, 128
                a = (0.5 * torch.randn(m, k, device="cuda", dtype=torch.bfloat16)).to(
                    torch.float8_e4m3fn
                )
                b = (
                    0.5 * torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
                ).to(torch.float8_e4m3fn).t()
                scale_a = torch.full(
                    (self._block_scaled_numel(m, k, 32),),
                    1.0,
                    device="cuda",
                    dtype=torch.float8_e8m0fnu,
                )
                scale_b = torch.full(
                    (self._block_scaled_numel(n, k, 32),),
                    1.0,
                    device="cuda",
                    dtype=torch.float8_e8m0fnu,
                )
                row_bias = torch.randn(m, 1, device="cuda", dtype=torch.float32) * 0.1
                col_scale = torch.randn(1, n, device="cuda", dtype=torch.float32) * 0.1
                tile_bias = torch.randn(m, n, device="cuda", dtype=torch.float32) * 0.1

                checks, check_nots = self._quack_codegen_invariants(
                    "scale_a=", "scale_b=", "epilogue_args=", "aux0", "aux2"
                )
                self._assert_compiled_matches(
                    lambda a, b, scale_a, scale_b, row_bias, col_scale, tile_bias: fn(
                        a,
                        b,
                        scale_a,
                        scale_b,
                        row_bias,
                        col_scale,
                        tile_bias,
                        epilogue,
                    ),
                    a,
                    b,
                    scale_a,
                    scale_b,
                    row_bias,
                    col_scale,
                    tile_bias,
                    atol=2e-1,
                    rtol=5e-2,
                    checks=checks,
                    check_nots=(*check_nots, "extern_kernels._scaled_mm"),
                )

    def _mxfp8_scaled_mm_v2_inputs(self, *, m=128, k=32, n=128):
        a = torch.eye(m, k, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        b = (
            torch.eye(n, k, device="cuda", dtype=torch.bfloat16)
            .to(torch.float8_e4m3fn)
            .t()
        )
        scale_a = torch.full(
            (self._block_scaled_numel(m, k, 32),),
            1.0,
            device="cuda",
            dtype=torch.float8_e8m0fnu,
        )
        scale_b = torch.full(
            (self._block_scaled_numel(n, k, 32),),
            1.0,
            device="cuda",
            dtype=torch.float8_e8m0fnu,
        )
        return a, b, scale_a, scale_b

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_rejects_scaled_mm_v2_tensorwise(self):
        if not PLATFORM_SUPPORTS_FP8:
            self.skipTest("FP8 is not supported")

        def fn(a, b, scale_a, scale_b):
            return gemm_epilogue_fusion(
                torch.ops.aten._scaled_mm_v2.default,
                (a, b, scale_a, scale_b),
                lambda acc: acc.relu(),
                gemm_kwargs={
                    "scale_recipe_a": F.ScalingType.TensorWise,
                    "scale_recipe_b": F.ScalingType.TensorWise,
                    "output_dtype": torch.bfloat16,
                },
                kernel_options={"backend": "QUACK"},
            )

        x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        a, scale_a = _quantize_tensorwise(x, torch.float8_e4m3fn)
        w_fp8, scale_b = _quantize_tensorwise(w, torch.float8_e4m3fn)

        with self.assertRaisesRegex(Exception, "MXFP8-like.*BlockWise1x32"):
            torch.compile(fn, backend="inductor", fullgraph=True)(
                a, w_fp8.t(), scale_a, scale_b
            )

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_routes_scaled_mm_v2_nvfp4_to_quack(self):
        if not PLATFORM_SUPPORTS_MX_GEMM or torch.version.hip:
            self.skipTest("NVFP4 GEMM is not supported")

        def fn(a, b, scale_a, global_scale, scale_b):
            return gemm_epilogue_fusion(
                torch.ops.aten._scaled_mm_v2.default,
                (a, b, [scale_a, global_scale], [scale_b, global_scale]),
                lambda acc: acc.relu(),
                gemm_kwargs={
                    "scale_recipe_a": [
                        F.ScalingType.BlockWise1x16,
                        F.ScalingType.TensorWise,
                    ],
                    "scale_recipe_b": [
                        F.ScalingType.BlockWise1x16,
                        F.ScalingType.TensorWise,
                    ],
                    "swizzle_a": [
                        F.SwizzleType.SWIZZLE_32_4_4,
                        F.SwizzleType.NO_SWIZZLE,
                    ],
                    "swizzle_b": [
                        F.SwizzleType.SWIZZLE_32_4_4,
                        F.SwizzleType.NO_SWIZZLE,
                    ],
                    "output_dtype": torch.bfloat16,
                },
                kernel_options={"backend": "QUACK"},
            )

        m, k, n = 128, 256, 192
        a = _bfloat16_to_float4_e2m1fn_x2(
            torch.eye(m, k, device="cuda", dtype=torch.bfloat16)
        )
        b = _bfloat16_to_float4_e2m1fn_x2(
            torch.eye(n, k, device="cuda", dtype=torch.bfloat16)
        ).t()
        from quack.blockscaled_gemm_utils import pack_scale_2d_to_blocked_contig

        sf_k = k // 16
        scale_a_2d = (
            0.5
            + (torch.arange(m, device="cuda")[:, None] % 7).float() * 0.125
            + (torch.arange(sf_k, device="cuda")[None, :] % 5).float() * 0.0625
        ).to(torch.float8_e4m3fn)
        scale_b_2d = (
            0.75
            + (torch.arange(n, device="cuda")[:, None] % 5).float() * 0.125
            + (torch.arange(sf_k, device="cuda")[None, :] % 7).float() * 0.0625
        ).to(torch.float8_e4m3fn)
        scale_a = pack_scale_2d_to_blocked_contig(scale_a_2d).flatten()
        scale_b = pack_scale_2d_to_blocked_contig(scale_b_2d).flatten()
        global_scale = torch.ones((1,), device="cuda", dtype=torch.float32)

        result = torch.compile(fn, backend="inductor", fullgraph=True)(
            a, b, scale_a, global_scale, scale_b
        )
        expected = torch.ops.aten._scaled_mm_v2.default(
            a,
            b,
            [scale_a, global_scale],
            [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            [scale_b, global_scale],
            [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            None,
            torch.bfloat16,
            [],
            False,
        ).relu()
        logical_expected = torch.zeros((m, n), device="cuda", dtype=torch.float32)
        diagonal = torch.arange(m, device="cuda")
        logical_expected[diagonal, diagonal] = (
            scale_a_2d.float()[diagonal, diagonal // 16]
            * scale_b_2d.float()[diagonal, diagonal // 16]
        )
        self.assertEqual(expected, logical_expected.to(torch.bfloat16))
        self.assertEqual(result, expected)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_rejects_scaled_mm_v2_fast_accum(self):
        if not PLATFORM_SUPPORTS_MX_GEMM:
            self.skipTest("MX GEMM is not supported")

        swizzle = (
            F.SwizzleType.NO_SWIZZLE
            if torch.version.hip
            else F.SwizzleType.SWIZZLE_32_4_4
        )

        def fn(a, b, scale_a, scale_b):
            return gemm_epilogue_fusion(
                torch.ops.aten._scaled_mm_v2.default,
                (a, b, scale_a, scale_b),
                lambda acc: acc.relu(),
                gemm_kwargs={
                    "scale_recipe_a": F.ScalingType.BlockWise1x32,
                    "scale_recipe_b": F.ScalingType.BlockWise1x32,
                    "swizzle_a": swizzle,
                    "swizzle_b": swizzle,
                    "output_dtype": torch.bfloat16,
                    "use_fast_accum": True,
                },
                kernel_options={"backend": "QUACK"},
            )

        with self.assertRaisesRegex(Exception, "use_fast_accum"):
            torch.compile(fn, backend="inductor", fullgraph=True)(
                *self._mxfp8_scaled_mm_v2_inputs()
            )

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_rejects_scaled_mm_v2_contraction_dim(self):
        if not PLATFORM_SUPPORTS_MX_GEMM:
            self.skipTest("MX GEMM is not supported")

        swizzle = (
            F.SwizzleType.NO_SWIZZLE
            if torch.version.hip
            else F.SwizzleType.SWIZZLE_32_4_4
        )

        def fn(a, b, scale_a, scale_b):
            return gemm_epilogue_fusion(
                torch.ops.aten._scaled_mm_v2.default,
                (a, b, scale_a, scale_b),
                lambda acc: acc.relu(),
                gemm_kwargs={
                    "scale_recipe_a": F.ScalingType.BlockWise1x32,
                    "scale_recipe_b": F.ScalingType.BlockWise1x32,
                    "swizzle_a": swizzle,
                    "swizzle_b": swizzle,
                    "output_dtype": torch.bfloat16,
                    "contraction_dim": [1],
                },
                kernel_options={"backend": "QUACK"},
            )

        with self.assertRaisesRegex(Exception, "contraction_dim"):
            torch.compile(fn, backend="inductor", fullgraph=True)(
                *self._mxfp8_scaled_mm_v2_inputs()
            )

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_rejects_scaled_mm_v2_bias(self):
        if not PLATFORM_SUPPORTS_MX_GEMM:
            self.skipTest("MX GEMM is not supported")

        swizzle = (
            F.SwizzleType.NO_SWIZZLE
            if torch.version.hip
            else F.SwizzleType.SWIZZLE_32_4_4
        )

        def fn(a, b, scale_a, scale_b, bias):
            return gemm_epilogue_fusion(
                torch.ops.aten._scaled_mm_v2.default,
                (a, b, scale_a, scale_b),
                lambda acc: acc.relu(),
                gemm_kwargs={
                    "scale_recipe_a": F.ScalingType.BlockWise1x32,
                    "scale_recipe_b": F.ScalingType.BlockWise1x32,
                    "swizzle_a": swizzle,
                    "swizzle_b": swizzle,
                    "bias": bias,
                    "output_dtype": torch.bfloat16,
                },
                kernel_options={"backend": "QUACK"},
            )

        a, b, scale_a, scale_b = self._mxfp8_scaled_mm_v2_inputs()
        bias = torch.ones((a.shape[0], b.shape[1]), device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(Exception, "bias"):
            torch.compile(fn, backend="inductor", fullgraph=True)(
                a, b, scale_a, scale_b, bias
            )

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_rejects_scaled_mm_v2_wrong_swizzle(self):
        if not PLATFORM_SUPPORTS_MX_GEMM:
            self.skipTest("MX GEMM is not supported")

        wrong_swizzle = (
            F.SwizzleType.SWIZZLE_32_4_4
            if torch.version.hip
            else F.SwizzleType.NO_SWIZZLE
        )

        def fn(a, b, scale_a, scale_b):
            return gemm_epilogue_fusion(
                torch.ops.aten._scaled_mm_v2.default,
                (a, b, scale_a, scale_b),
                lambda acc: acc.relu(),
                gemm_kwargs={
                    "scale_recipe_a": F.ScalingType.BlockWise1x32,
                    "scale_recipe_b": F.ScalingType.BlockWise1x32,
                    "swizzle_a": wrong_swizzle,
                    "swizzle_b": wrong_swizzle,
                    "output_dtype": torch.bfloat16,
                },
                kernel_options={"backend": "QUACK"},
            )

        with self.assertRaisesRegex(Exception, "swizzle"):
            torch.compile(fn, backend="inductor", fullgraph=True)(
                *self._mxfp8_scaled_mm_v2_inputs()
            )

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_routes_scaled_mm_relu_to_quack_hook(self):
        if not PLATFORM_SUPPORTS_MX_GEMM:
            self.skipTest("MX GEMM is not supported")

        def fn(a, b, scale_a, scale_b):
            return gemm_epilogue_fusion(
                torch.ops.aten._scaled_mm.default,
                (a, b, scale_a, scale_b),
                lambda acc: acc.relu(),
                gemm_kwargs={"out_dtype": torch.bfloat16},
                kernel_options={"backend": "QUACK"},
            )

        m, k, n = 128, 32, 128
        a = torch.eye(m, k, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        b = (
            torch.eye(n, k, device="cuda", dtype=torch.bfloat16)
            .to(torch.float8_e4m3fn)
            .t()
        )
        scale_a = to_blocked(
            torch.full(
                (m, ceil_div(k, 32)),
                1.0,
                device="cuda",
                dtype=torch.float8_e8m0fnu,
            )
        )
        scale_b = to_blocked(
            torch.full(
                (n, ceil_div(k, 32)),
                1.0,
                device="cuda",
                dtype=torch.float8_e8m0fnu,
            )
        )

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True),
            a,
            b,
            scale_a,
            scale_b,
        )

        torch.testing.assert_close(
            actual, fn(a, b, scale_a, scale_b), atol=0.05, rtol=1e-2
        )
        FileCheck().check("@cute.jit").check("gemm_epilogue(").check("scale_a=").check(
            "scale_b="
        ).check_not("extern_kernels._scaled_mm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_rejects_scaled_mm_non_mxfp8_scales(self):
        if not PLATFORM_SUPPORTS_MX_GEMM:
            self.skipTest("MX GEMM is not supported")

        def fn(a, b, scale_a, scale_b):
            return gemm_epilogue_fusion(
                torch.ops.aten._scaled_mm.default,
                (a, b, scale_a, scale_b),
                lambda acc: acc.relu(),
                gemm_kwargs={"out_dtype": torch.bfloat16},
                kernel_options={"backend": "QUACK"},
            )

        m, k, n = 128, 32, 128
        a = torch.eye(m, k, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        b = (
            torch.eye(n, k, device="cuda", dtype=torch.bfloat16)
            .to(torch.float8_e4m3fn)
            .t()
        )

        with self.assertRaisesRegex(Exception, "MXFP8-like.*BlockWise1x32"):
            torch.compile(fn, backend="inductor", fullgraph=True)(
                a,
                b,
                torch.ones((), device="cuda", dtype=torch.float32),
                torch.ones((), device="cuda", dtype=torch.float32),
            )

        scale_a = torch.full(
            (m, ceil_div(k, 32)),
            1.0,
            device="cuda",
            dtype=torch.float8_e8m0fnu,
        )
        scale_b = torch.full(
            (n, ceil_div(k, 32)),
            1.0,
            device="cuda",
            dtype=torch.float8_e8m0fnu,
        )
        with self.assertRaisesRegex(Exception, "Invalid blockwise scaling configuration"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b, scale_a, scale_b)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_routes_addmm_relu_to_quack_hook(self):
        def fn(bias, a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.addmm.default,
                (bias, a, b),
                lambda acc: acc.relu(),
                kernel_options={"backend": "QUACK"},
            )

        bias = torch.randn(128, 64, device="cuda", dtype=torch.float16)
        a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), bias, a, b
        )

        torch.testing.assert_close(actual, fn(bias, a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("@cute.jit").check("gemm_epilogue(").check(
            "C=arg0_1"
        ).check_not("extern_kernels.addmm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_routes_addmm_alpha_beta_to_quack_hook(self):
        def fn(bias, a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.addmm.default,
                (bias, a, b),
                lambda acc: acc.relu(),
                gemm_kwargs={"alpha": 0.5, "beta": 0.25},
                kernel_options={"backend": "QUACK"},
            )

        bias = torch.randn(128, 64, device="cuda", dtype=torch.float16)
        a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), bias, a, b
        )

        torch.testing.assert_close(actual, fn(bias, a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("gemm_epilogue(").check("alpha=0.5").check(
            "beta=0.25"
        ).check_not("extern_kernels.addmm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_routes_bmm_relu_to_quack_hook(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.bmm.default,
                (a, b),
                lambda acc: acc.relu(),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(4, 64, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(4, 128, 32, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        torch.testing.assert_close(actual, fn(a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("@cute.jit").check("gemm_epilogue(").check_not(
            "extern_kernels.bmm"
        ).run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_routes_baddbmm_alpha_beta_to_quack_hook(self):
        def fn(bias, a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.baddbmm.default,
                (bias, a, b),
                lambda acc: acc.relu(),
                gemm_kwargs={"alpha": 0.5, "beta": 0.25},
                kernel_options={"backend": "QUACK"},
            )

        bias = torch.randn(4, 64, 32, device="cuda", dtype=torch.float16)
        a = torch.randn(4, 64, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(4, 128, 32, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), bias, a, b
        )

        torch.testing.assert_close(actual, fn(bias, a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("gemm_epilogue(").check("alpha=0.5").check(
            "beta=0.25"
        ).check_not("extern_kernels.baddbmm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_routes_baddbmm_relu_to_quack_hook(self):
        def fn(bias, a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.baddbmm.default,
                (bias, a, b),
                lambda acc: acc.relu(),
                kernel_options={"backend": "QUACK"},
            )

        bias = torch.randn(4, 64, 32, device="cuda", dtype=torch.float16)
        a = torch.randn(4, 64, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(4, 128, 32, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), bias, a, b
        )

        torch.testing.assert_close(actual, fn(bias, a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("@cute.jit").check("gemm_epilogue(").check(
            "C=arg0_1"
        ).check_not("extern_kernels.baddbmm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_routes_relu_to_quack_hook(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: acc.relu(),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.float16)

        checks, check_nots = self._quack_codegen_invariants(
            "def flex_gemm_quack_epilogue_"
        )
        self._assert_compiled_matches(
            fn, a, b, checks=checks, check_nots=check_nots
        )

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_generates_pointwise_epilogue(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: (acc + 1.0).relu(),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.float16)

        checks, check_nots = self._quack_codegen_invariants(
            "tmp0 = (acc + cute.full_like(acc, 1.0))"
        )
        self._assert_compiled_matches(
            fn, a, b, checks=checks, check_nots=check_nots
        )

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_positional_clamp_matches_reference(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: acc.clamp(-0.5, 0.5),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.float16)

        checks, check_nots = self._quack_codegen_invariants("-0.5", "0.5")
        self._assert_compiled_matches(
            fn, a, b, checks=checks, check_nots=check_nots
        )

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_local_reduce_feeds_main_groups(self):
        M = 32
        N = 64

        for group in (2, 4, 8, 16, 32):
            with self.subTest(group=group):

                def fn(a, b):
                    def epilogue(acc):
                        x = acc.float().view(M, -1, group)
                        denom = x.sum(dim=-1, keepdim=True)
                        return (x * denom.reciprocal()).view(M, N)

                    return gemm_epilogue_fusion(
                        torch.ops.aten.mm.default,
                        (a, b),
                        epilogue,
                        kernel_options={"backend": "QUACK"},
                    )

                a = torch.rand(M, 64, device="cuda", dtype=torch.float16)
                b = torch.rand(64, N, device="cuda", dtype=torch.float16)

                checks, check_nots = self._quack_codegen_invariants(
                    "cute.ReductionOp.ADD",
                    "broadcast_to",
                )
                self._assert_compiled_matches(
                    fn,
                    a,
                    b,
                    atol=5e-2,
                    rtol=5e-2,
                    checks=checks,
                    check_nots=check_nots,
                    contains=self._tensorssa_grouped_n_invariants(group),
                    assert_fn=lambda actual, expected: torch.testing.assert_close(
                        actual.float(), expected, atol=5e-2, rtol=5e-2
                    ),
                )

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_local_amax_feeds_main_groups(self):
        M = 32
        N = 64

        for group in (2, 4, 8, 16, 32):
            with self.subTest(group=group):

                def fn(a, b):
                    def epilogue(acc):
                        x = acc.float().view(M, -1, group)
                        denom = x.abs().amax(dim=-1, keepdim=True)
                        return (x * denom.reciprocal()).view(M, N)

                    return gemm_epilogue_fusion(
                        torch.ops.aten.mm.default,
                        (a, b),
                        epilogue,
                        kernel_options={"backend": "QUACK"},
                    )

                a = torch.randn(M, 64, device="cuda", dtype=torch.float16)
                b = torch.randn(64, N, device="cuda", dtype=torch.float16)

                checks, check_nots = self._quack_codegen_invariants(
                    "cute.ReductionOp.MAX",
                    "broadcast_to",
                )
                self._assert_compiled_matches(
                    fn,
                    a,
                    b,
                    atol=5e-2,
                    rtol=5e-2,
                    checks=checks,
                    check_nots=check_nots,
                    contains=self._tensorssa_grouped_n_invariants(group),
                    assert_fn=lambda actual, expected: torch.testing.assert_close(
                        actual.float(), expected, atol=5e-2, rtol=5e-2
                    ),
                )

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_rejects_amax_without_abs_feeding_main(self):
        M = 32
        N = 64
        group = 16

        def fn(a, b):
            def epilogue(acc):
                x = acc.float().view(M, -1, group)
                denom = x.amax(-1, keepdim=True)
                return (x * denom.reciprocal()).view(M, N)

            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(M, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, N, device="cuda", dtype=torch.float16)

        with self.assertRaisesRegex(Exception, "requires an abs/nonnegative input"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_row_reduce_feeds_main_groups(self):
        M = 128
        N = 64

        for group in (2, 4, 8, 16):
            with self.subTest(group=group):

                def fn(a, b):
                    def epilogue(acc):
                        x = acc.float().view(-1, group, N)
                        denom = x.sum(1, keepdim=True)
                        return (x * denom.reciprocal()).view(M, N)

                    return gemm_epilogue_fusion(
                        torch.ops.aten.mm.default,
                        (a, b),
                        epilogue,
                        kernel_options={"backend": "QUACK"},
                    )

                a = torch.rand(M, 64, device="cuda", dtype=torch.float16)
                b = torch.rand(64, N, device="cuda", dtype=torch.float16)

                checks, check_nots = self._quack_codegen_invariants(
                    *self._local_reduce_invariants(
                        group=group, dim=0, feeds_main=True
                    )
                )
                self._assert_compiled_matches(
                    fn,
                    a,
                    b,
                    atol=5e-2,
                    rtol=5e-2,
                    checks=checks,
                    check_nots=check_nots,
                    assert_fn=lambda actual, expected: torch.testing.assert_close(
                        actual.float(), expected.float(), atol=5e-2, rtol=5e-2
                    ),
                )

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_rejects_large_group_reduce_feeding_main(self):
        M = 64
        N = 64
        group = 64

        def fn(a, b):
            def epilogue(acc):
                x = acc.float().view(M, -1, group)
                denom = x.sum(-1, keepdim=True)
                return (x * denom.reciprocal()).view(M, N)

            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(M, 64, device="cuda", dtype=torch.float16)
        b = torch.rand(64, N, device="cuda", dtype=torch.float16)

        with self.assertRaisesRegex(Exception, "same-fragment N width 32"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_rejects_large_n_group_tuple_aux(self):
        M = 128
        N = 512
        K = 128
        group = 256

        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: (acc.relu(), acc.float().view(M, -1, group).sum(-1)),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(M, K, device="cuda", dtype=torch.float16)
        b = torch.rand(K, N, device="cuda", dtype=torch.float16)

        with self.assertRaisesRegex(Exception, "divisible by group, got group=256"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_tuple_epilogue_row_groups_fuse(self):
        M = 128
        N = 64
        K = 64

        for dtype in (torch.float16, torch.bfloat16):
            for group in (2, 4, 8, 16):
                with self.subTest(dtype=dtype, group=group):

                    def fn(a, b):
                        return gemm_epilogue_fusion(
                            torch.ops.aten.mm.default,
                            (a, b),
                            lambda acc: (
                                acc.relu(),
                                acc.float().view(-1, group, N).sum(1),
                            ),
                            kernel_options={"backend": "QUACK"},
                        )

                    a = torch.rand(M, K, device="cuda", dtype=dtype)
                    b = torch.rand(K, N, device="cuda", dtype=dtype)

                    actual, (code,) = run_and_get_code(
                        torch.compile(fn, backend="inductor", fullgraph=True), a, b
                    )
                    expected = fn(a, b)

                    torch.testing.assert_close(
                        actual[0], expected[0], atol=1e-2, rtol=1e-2
                    )
                    torch.testing.assert_close(
                        actual[1], expected[1], atol=5e-1, rtol=1e-2
                    )
                    FileCheck().check(f"local_reduce_group={group}").check(
                        "local_reduce_dim=0"
                    ).check("local_reduce_out=").check_not("extern_kernels.mm").run(
                        code
                    )

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_tuple_epilogue_row_reshape_fuses(self):
        M = 256
        N = 128
        group = 4

        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: (
                    acc.relu(),
                    acc.float().reshape(-1, group, N).sum(1),
                ),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(M, 64, device="cuda", dtype=torch.float16)
        b = torch.rand(64, N, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected = fn(a, b)

        torch.testing.assert_close(actual[0], expected[0], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(actual[1], expected[1], atol=1e-1, rtol=1e-2)
        FileCheck().check(f"local_reduce_group={group}").check(
            "local_reduce_dim=0"
        ).check("local_reduce_out=").check_not("extern_kernels.mm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_rejects_mis_shaped_row_reduce(self):
        M = 128
        N = 64

        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: (
                    acc.relu(),
                    acc.float().view(-1, 2, N // 2).sum(1),
                ),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(M, 64, device="cuda", dtype=torch.float16)
        b = torch.rand(64, N, device="cuda", dtype=torch.float16)

        with self.assertRaisesRegex(Exception, "view\\(-1, group, N\\).sum\\(1\\)"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_rejects_row_group2_tail_m(self):
        M = 64
        N = 64

        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: (
                    acc.relu(),
                    acc.float().view(-1, 2, N).sum(1),
                ),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(M, 64, device="cuda", dtype=torch.float16)
        b = torch.rand(64, N, device="cuda", dtype=torch.float16)

        with self.assertRaisesRegex(Exception, "multiple of tile_m=128"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_tuple_epilogue_small_group_fuses(self):
        M = 32

        for group in (2, 4, 8):
            with self.subTest(group=group):

                def fn(a, b):
                    return gemm_epilogue_fusion(
                        torch.ops.aten.mm.default,
                        (a, b),
                        lambda acc: (
                            acc.relu(),
                            acc.float().view(M, -1, group).sum(-1),
                        ),
                        kernel_options={"backend": "QUACK"},
                    )

                a = torch.rand(M, 64, device="cuda", dtype=torch.float16)
                b = torch.rand(64, 64, device="cuda", dtype=torch.float16)

                actual, (code,) = run_and_get_code(
                    torch.compile(fn, backend="inductor", fullgraph=True), a, b
                )
                expected = fn(a, b)

                torch.testing.assert_close(actual[0], expected[0], atol=1e-2, rtol=1e-2)
                torch.testing.assert_close(actual[1], expected[1], atol=1e-1, rtol=1e-2)
                FileCheck().check(f"local_reduce_group={group}").check(
                    "local_reduce_dim=1"
                ).check("local_reduce_out=").check_not("extern_kernels.mm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_tuple_epilogue_group16_fuses(self):
        M = 128

        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: (acc.relu(), acc.float().view(M, -1, 16).sum(-1)),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(M, 128, device="cuda", dtype=torch.float16)
        b = torch.rand(128, 64, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected = fn(a, b)

        torch.testing.assert_close(actual[0], expected[0], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(actual[1], expected[1], atol=1e-1, rtol=1e-2)
        FileCheck().check("local_reduce_group=16").check("local_reduce_dim=1").check(
            "local_reduce_out="
        ).check_not("extern_kernels.mm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_tuple_epilogue_large_n_groups_fuse(self):
        M = 128
        N = 512
        K = 128

        for group in (32, 64, 128):
            with self.subTest(group=group):

                def fn(a, b):
                    return gemm_epilogue_fusion(
                        torch.ops.aten.mm.default,
                        (a, b),
                        lambda acc: (
                            acc.relu(),
                            acc.float().view(M, -1, group).sum(-1),
                        ),
                        kernel_options={"backend": "QUACK"},
                    )

                a = torch.rand(M, K, device="cuda", dtype=torch.float16)
                b = torch.rand(K, N, device="cuda", dtype=torch.float16)

                actual, (code,) = run_and_get_code(
                    torch.compile(fn, backend="inductor", fullgraph=True), a, b
                )
                expected = fn(a, b)

                torch.testing.assert_close(
                    actual[0], expected[0], atol=1e-2, rtol=1e-2
                )
                torch.testing.assert_close(
                    actual[1], expected[1], atol=5e-1, rtol=5e-2
                )
                FileCheck().check(f"local_reduce_group={group}").check(
                    "local_reduce_dim=1"
                ).check("local_reduce_out=").check_not("extern_kernels.mm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_local_reduce_feeds_main(self):
        M = 64
        N = 64

        def fn(a, b):
            def epilogue(acc):
                x = acc.float().view(M, -1, 32)
                denom = x.sum(-1, keepdim=True)
                return (x * denom.reciprocal()).view(M, N)

            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(M, 64, device="cuda", dtype=torch.float16)
        b = torch.rand(64, N, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected = fn(a, b)

        torch.testing.assert_close(actual.float(), expected, atol=5e-2, rtol=5e-2)
        FileCheck().check("@cute.jit").check("cute.ReductionOp.ADD").check(
            "gemm_epilogue("
        ).check_not("extern_kernels.mm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_tuple_epilogue_fuses(self):
        M = 128

        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: (acc.relu(), acc.float().view(M, -1, 32).sum(-1)),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(M, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.float16)

        self._assert_quack_tuple_epilogue(
            fn,
            a,
            b,
            checks=("local_reduce_group=32", "local_reduce_dim=1", "local_reduce_out="),
        )

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_tuple_epilogue_generic_aux_fuses(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: (acc.relu(), acc.float().abs()),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.float16)

        self._assert_quack_tuple_epilogue(
            fn,
            a,
            b,
            checks=("aux_out=",),
            check_nots=("local_reduce_out=", "extern_kernels.mm"),
        )

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_tuple_epilogue_generic_aux_only_is_live(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: (acc.relu(), acc.float().abs()),
                kernel_options={"backend": "QUACK"},
            )[1]

        a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        torch.testing.assert_close(actual, fn(a, b), atol=1e-1, rtol=1e-2)
        FileCheck().check("aux_out=").check_not("extern_kernels.mm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_tuple_epilogue_local_amax_aux_fuses(self):
        M = 32
        N = 64

        for group in (2, 4, 8, 16, 32):
            with self.subTest(group=group):

                def fn(a, b):
                    def epilogue(acc):
                        x = acc.float().view(M, -1, group)
                        return acc.relu(), x.abs().amax(-1)

                    return gemm_epilogue_fusion(
                        torch.ops.aten.mm.default,
                        (a, b),
                        epilogue,
                        kernel_options={"backend": "QUACK"},
                    )

                a = torch.randn(M, 64, device="cuda", dtype=torch.float16)
                b = torch.randn(64, N, device="cuda", dtype=torch.float16)

                self._assert_quack_tuple_epilogue(
                    fn,
                    a,
                    b,
                    checks=(
                        f"local_reduce_group={group}",
                        "local_reduce_dim=1",
                        "local_reduce_out=",
                        "local_reduce_op='amax_abs'",
                    ),
                    check_nots=("aux_out=", "extern_kernels.mm"),
                )

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_tuple_epilogue_local_amax_scale_aux_fuses(self):
        M = 32
        N = 64
        group = 16

        def fn(a, b):
            def epilogue(acc):
                x = acc.float().view(M, -1, group)
                return acc.relu(), x.abs().amax(-1) / 448.0

            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(M, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, N, device="cuda", dtype=torch.float16)

        self._assert_quack_tuple_epilogue(
            fn,
            a,
            b,
            aux_atol=1e-3,
            aux_rtol=1e-3,
            checks=(
                f"local_reduce_group={group}",
                "local_reduce_dim=1",
                "local_reduce_out=",
                "local_reduce_op='amax_abs'",
                "local_reduce_scale=",
            ),
            check_nots=("aux_out=", "extern_kernels.mm"),
        )

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_tuple_epilogue_local_amax_scale_main_and_aux_fuses(self):
        M = 64
        N = 64
        group = 16

        def fn(a, b):
            def epilogue(acc):
                x = acc.float().view(M, -1, group)
                scale = x.abs().amax(-1, keepdim=True) / 448.0
                q = (x * scale.reciprocal()).view(M, N)
                # Recompute the reduced scale for aux output so the main-output
                # TensorSSA expression and store-only aux reduction can be lowered
                # independently for now.
                return q, x.abs().amax(-1) / 448.0

            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(M, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, N, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected = fn(a, b)

        torch.testing.assert_close(actual[0].float(), expected[0], atol=1.0, rtol=2e-3)
        torch.testing.assert_close(actual[1], expected[1], atol=1e-3, rtol=1e-3)
        FileCheck().check("cute.ReductionOp.MAX").check(
            f"local_reduce_group={group}"
        ).check("local_reduce_out=").check("local_reduce_op='amax_abs'").check(
            "local_reduce_scale="
        ).check_not("extern_kernels.mm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_tuple_epilogue_shared_local_amax_scale_main_and_aux_fuses(self):
        M = 64
        N = 64
        group = 16

        def fn(a, b):
            def epilogue(acc):
                x = acc.float().view(M, -1, group)
                scale = x.abs().amax(-1, keepdim=True) / 448.0
                q = (x * scale.reciprocal()).view(M, N)
                return q, scale.view(M, -1)

            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(M, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, N, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected = fn(a, b)

        torch.testing.assert_close(actual[0].float(), expected[0], atol=1.0, rtol=2e-3)
        torch.testing.assert_close(actual[1], expected[1], atol=1e-3, rtol=1e-3)
        FileCheck().check("cute.ReductionOp.MAX").check(
            f"local_reduce_group={group}"
        ).check("local_reduce_out=").check("local_reduce_op='copy'").check(
            "local_reduce_source_from_epilogue=True"
        ).check_not("extern_kernels.mm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_tuple_epilogue_local_amax_scale_fp8_main_fuses(self):
        M = 64
        N = 64
        group = 16

        def fn(a, b):
            def epilogue(acc):
                x = acc.float().view(M, -1, group)
                scale = x.abs().amax(-1, keepdim=True) / 448.0
                q = (x * scale.reciprocal()).clamp(min=-448.0, max=448.0).view(M, N)
                return q.to(torch.float8_e4m3fn), scale.view(M, -1)

            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(M, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, N, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected = fn(a, b)

        self.assertEqual(actual[0].dtype, torch.float8_e4m3fn)
        # CUTLASS/CuTe FP8 conversion can differ by one coarse FP8 bin from
        # eager PyTorch's cast near saturation boundaries.
        torch.testing.assert_close(
            actual[0].float(), expected[0].float(), atol=32.0, rtol=1.25e-1
        )
        torch.testing.assert_close(actual[1], expected[1], atol=1e-3, rtol=1e-3)
        FileCheck().check("out_dtype=torch.float8_e4m3fn").check(
            f"local_reduce_group={group}"
        ).check("local_reduce_out=").check("local_reduce_op='copy'").check(
            "local_reduce_source_from_epilogue=True"
        ).check_not("extern_kernels.mm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_tuple_epilogue_mxfp8_like_main_and_e8m0_scale_fuses(self):
        M = 64
        N = 64
        group = 16

        def fn(a, b):
            def epilogue(acc):
                x = acc.float().view(M, -1, group)
                scale = x.abs().amax(-1, keepdim=True) / 448.0
                q = (x * scale.reciprocal()).clamp(min=-448.0, max=448.0).view(M, N)
                return q.to(torch.float8_e4m3fn), scale.view(M, -1).to(
                    torch.float8_e8m0fnu
                )

            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(M, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, N, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected = fn(a, b)

        self.assertEqual(actual[0].dtype, torch.float8_e4m3fn)
        self.assertEqual(actual[1].dtype, torch.float8_e8m0fnu)
        torch.testing.assert_close(
            actual[0].float(), expected[0].float(), atol=32.0, rtol=1.25e-1
        )
        torch.testing.assert_close(
            actual[1].float(), expected[1].float(), atol=6.25e-2, rtol=1.25e-1
        )
        FileCheck().check("out_dtype=torch.float8_e4m3fn").check(
            f"local_reduce_group={group}"
        ).check("local_reduce_out=").check("local_reduce_op='copy'").check(
            "local_reduce_source_from_epilogue=True"
        ).check_not("extern_kernels.mm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_tuple_epilogue_mxfp8_scale_op_fuses(self):
        M = 64
        N = 64
        group = 32

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):

                def fn(a, b):
                    def epilogue(acc):
                        x = acc.float().view(M, -1, group)
                        scale_e8m0 = mx_e8m0_scale(x.abs().amax(-1, keepdim=True))
                        q = (x * scale_e8m0.float().reciprocal()).clamp(
                            min=-448.0, max=448.0
                        ).view(M, N)
                        return q.to(torch.float8_e4m3fn), scale_e8m0.view(M, -1)

                    return gemm_epilogue_fusion(
                        torch.ops.aten.mm.default,
                        (a, b),
                        epilogue,
                        kernel_options={"backend": "QUACK"},
                    )

                a = torch.randn(M, 64, device="cuda", dtype=dtype)
                b = torch.randn(64, N, device="cuda", dtype=dtype)

                actual, (code,) = run_and_get_code(
                    torch.compile(fn, backend="inductor", fullgraph=True), a, b
                )
                expected = fn(a, b)

                self.assertEqual(actual[0].dtype, torch.float8_e4m3fn)
                self.assertEqual(actual[1].dtype, torch.float8_e8m0fnu)
                torch.testing.assert_close(
                    actual[0].float(), expected[0].float(), atol=32.0, rtol=1.25e-1
                )
                torch.testing.assert_close(
                    actual[1].float(), expected[1].float(), atol=0.0, rtol=0.0
                )
                FileCheck().check("out_dtype=torch.float8_e4m3fn").check(
                    f"local_reduce_group={group}"
                ).check("local_reduce_out=").check(
                    "local_reduce_op='mx_e8m0_scale'"
                ).check_not("extern_kernels.mm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_tuple_epilogue_keepdim_scale_aux_only_fuses(self):
        M = 64
        N = 64
        group = 16

        def fn(a, b):
            def epilogue(acc):
                x = acc.float().view(M, -1, group)
                scale = x.abs().amax(-1, keepdim=True) / 448.0
                return acc.relu(), scale.view(M, -1)

            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(M, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, N, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected = fn(a, b)

        torch.testing.assert_close(actual[0], expected[0], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(actual[1], expected[1], atol=1e-3, rtol=1e-3)
        FileCheck().check(f"local_reduce_group={group}").check(
            "local_reduce_out="
        ).check("local_reduce_op='amax_abs'").check_not("extern_kernels.mm").run(
            code
        )

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_tuple_epilogue_bias_local_amax_scale_fuses(self):
        M = 64
        N = 64
        group = 32

        def fn(a, b, bias):
            def epilogue(acc):
                x = (acc + bias).float().view(M, -1, group)
                scale = x.abs().amax(-1, keepdim=True)
                q = (x * scale.reciprocal()).view(M, N)
                return q, scale.view(M, -1)

            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(M, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, N, device="cuda", dtype=torch.float16)
        bias = torch.randn(1, N, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, bias
        )
        expected = fn(a, b, bias)

        torch.testing.assert_close(actual[0].float(), expected[0], atol=1.0, rtol=2e-3)
        torch.testing.assert_close(actual[1], expected[1], atol=1e-3, rtol=1e-3)
        FileCheck().check(f"local_reduce_group={group}").check(
            "local_reduce_out="
        ).check("local_reduce_op='copy'").check(
            "local_reduce_source_from_epilogue=True"
        ).check_not("extern_kernels.mm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_tuple_epilogue_relu_local_amax_scale_fuses(self):
        M = 64
        N = 64
        group = 32

        def fn(a, b):
            def epilogue(acc):
                x = acc.relu().float().view(M, -1, group)
                scale = x.abs().amax(-1, keepdim=True)
                q = (x * scale.reciprocal()).view(M, N)
                return q, scale.view(M, -1)

            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(M, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, N, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected = fn(a, b)

        torch.testing.assert_close(actual[0].float(), expected[0], atol=1.0, rtol=2e-3)
        torch.testing.assert_close(actual[1], expected[1], atol=1e-3, rtol=1e-3)
        FileCheck().check(f"local_reduce_group={group}").check(
            "local_reduce_out="
        ).check("local_reduce_op='copy'").check_not("math.absi").check_not(
            "extern_kernels.mm"
        ).run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_tuple_epilogue_mx_scale_aux_only_fuses(self):
        M = 64
        N = 64
        group = 16

        def fn(a, b):
            def epilogue(acc):
                x = acc.float().view(M, -1, group)
                scale = mx_e8m0_scale(x.abs().amax(-1, keepdim=True))
                return acc.relu(), scale.view(M, -1)

            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(M, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, N, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected = fn(a, b)

        torch.testing.assert_close(actual[0], expected[0], atol=1e-2, rtol=1e-2)
        self.assertEqual(actual[1].dtype, torch.float8_e8m0fnu)
        torch.testing.assert_close(actual[1].float(), expected[1].float(), atol=0.0, rtol=0.0)
        FileCheck().check(f"local_reduce_group={group}").check(
            "local_reduce_out="
        ).check("local_reduce_op='mx_e8m0_scale'").check_not("extern_kernels.mm").run(
            code
        )

    def test_mx_e8m0_scale_edge_values(self):
        values = torch.tensor(
            [0.0, 1.0, 2.0, 8.0, 16.0, 448.0, float("nan")],
            dtype=torch.float32,
        )

        actual = mx_e8m0_scale(values)
        self.assertEqual(actual.dtype, torch.float8_e8m0fnu)
        self.assertEqual(
            actual.view(torch.uint8),
            torch.tensor([0, 119, 120, 122, 123, 127, 255], dtype=torch.uint8),
        )

        actual_max_power_2 = mx_e8m0_scale(values, max_power=2)
        self.assertEqual(
            actual_max_power_2.view(torch.uint8),
            torch.tensor([0, 125, 126, 128, 129, 133, 255], dtype=torch.uint8),
        )

        with FakeTensorMode():
            fake_input = torch.empty_strided((2, 3), (3, 1))
            fake_actual = mx_e8m0_scale(fake_input)
            self.assertEqual(fake_actual.shape, fake_input.shape)
            self.assertEqual(fake_actual.stride(), fake_input.stride())
            self.assertEqual(fake_actual.dtype, torch.float8_e8m0fnu)

    def test_nvfp4_e4m3_scale_edge_values(self):
        values = torch.tensor(
            [0.0, 1e-6, 6.0, 12.0, 448.0 * 6.0, 1e9], dtype=torch.float32
        )

        actual = nvfp4_e4m3_scale(values)
        self.assertEqual(actual.dtype, torch.float8_e4m3fn)
        self.assertEqual(
            actual.float(),
            torch.tensor([1 / 64, 1 / 64, 1.0, 2.0, 448.0, 448.0]),
        )

        with FakeTensorMode():
            fake_input = torch.empty_strided((2, 3), (3, 1))
            fake_actual = nvfp4_e4m3_scale(fake_input)
            self.assertEqual(fake_actual.shape, fake_input.shape)
            self.assertEqual(fake_actual.stride(), fake_input.stride())
            self.assertEqual(fake_actual.dtype, torch.float8_e4m3fn)

    def test_nvfp4_e2m1_pack_matches_reference(self):
        from torch.testing._internal.common_quantized import (
            _bfloat16_to_float4_e2m1fn_x2,
        )

        x = torch.linspace(-6.0, 6.0, 64, dtype=torch.float32).reshape(4, 16)
        actual = nvfp4_e2m1_pack(x)
        expected = _bfloat16_to_float4_e2m1fn_x2(x.to(torch.bfloat16))
        self.assertEqual(actual.dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(actual.shape, (4, 8))
        self.assertEqual(actual.stride(), (8, 1))
        self.assertEqual(actual.view(torch.uint8), expected.view(torch.uint8))

        with FakeTensorMode():
            fake_actual = nvfp4_e2m1_pack(torch.empty(4, 16))
            self.assertEqual(fake_actual.stride(), actual.stride())

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_grouped_contract_shape_changing_main_fuses(self):
        cases = (
            (64, 64, 64, torch.float16, 2, "silu", True),
            (96, 96, 64, torch.bfloat16, 2, "relu", False),
            (64, 128, 96, torch.float16, 2, "sub", False),
            (64, 64, 32, torch.float16, 4, "sum4", False),
        )
        for M, K, N, dtype, group, expr, cast_acc in cases:
            with self.subTest(shape=(M, K, N), dtype=dtype, group=group, expr=expr):

                def fn(a, b):
                    def epilogue(acc):
                        lanes = (acc.float() if cast_acc else acc).view(M, -1, group)
                        if expr == "silu":
                            out = torch.nn.functional.silu(lanes[..., 0]) * lanes[..., 1]
                        elif expr == "relu":
                            out = torch.relu(lanes[..., 0]) * lanes[..., 1]
                        elif expr == "sub":
                            out = lanes[..., 0] - lanes[..., 1]
                        else:
                            out = lanes[..., 0] + lanes[..., 1] + lanes[..., 2] + lanes[..., 3]
                        return out.to(acc.dtype) if cast_acc else out

                    return gemm_epilogue_fusion(
                        torch.ops.aten.mm.default,
                        (a, b),
                        epilogue,
                        kernel_options={"backend": "QUACK"},
                    )

                a = torch.randn(M, K, device="cuda", dtype=dtype)
                weights = [torch.randn(K, N, device="cuda", dtype=dtype) for _ in range(group)]
                b = torch.stack(weights, dim=-1).reshape(K, group * N).contiguous()

                actual, (code,) = run_and_get_code(
                    torch.compile(fn, backend="inductor", fullgraph=True), a, b
                )
                expected = fn(a, b)

                self.assertEqual(actual.shape, (M, N))
                torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-1)
                FileCheck().check("main_output_transform='grouped_n_contract'").check(
                    f"main_output_transform_group={group}"
                ).check_not("activation='").check_not("extern_kernels.mm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_bmm_reads_full_tile_closure_tensor(self):
        def fn(a, b, scale):
            return bmm_epilogue(
                a,
                b,
                lambda acc: acc * scale,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(2, 16, 32, device="cuda", dtype=torch.float16)
        b = torch.randn(2, 32, 16, device="cuda", dtype=torch.float16)
        scale = torch.randn(2, 16, 16, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, scale
        )
        expected = fn(a, b, scale)

        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        FileCheck().check("@cute.jit").check("aux0").check("epilogue_args=(").check_not(
            "extern_kernels.bmm"
        ).run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_bmm_fp8_main_output_fuses(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.bmm.default,
                (a, b),
                lambda acc: acc.clamp(min=-448.0, max=448.0).to(torch.float8_e4m3fn),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(2, 32, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(2, 64, 48, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected = fn(a, b)

        self.assertEqual(actual.dtype, torch.float8_e4m3fn)
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=32.0, rtol=1.25e-1
        )
        FileCheck().check("out_dtype=torch.float8_e4m3fn").check_not(
            "extern_kernels.bmm"
        ).run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_bmm_grouped_contract_fuses(self):
        cases = (
            (2, 32, 64, 32, 2, "relu"),
            (2, 16, 64, 16, 4, "sum4"),
        )
        for B, M, K, N, group, expr in cases:
            with self.subTest(shape=(B, M, K, N), group=group, expr=expr):

                def fn(a, b):
                    def epilogue(acc):
                        lanes = acc.view(B, M, -1, group)
                        if expr == "relu":
                            return torch.relu(lanes[..., 0]) * lanes[..., 1]
                        return lanes[..., 0] + lanes[..., 1] + lanes[..., 2] + lanes[..., 3]

                    return gemm_epilogue_fusion(
                        torch.ops.aten.bmm.default,
                        (a, b),
                        epilogue,
                        kernel_options={"backend": "QUACK"},
                    )

                a = torch.randn(B, M, K, device="cuda", dtype=torch.float16)
                weights = [
                    torch.randn(B, K, N, device="cuda", dtype=torch.float16)
                    for _ in range(group)
                ]
                b = torch.stack(weights, dim=-1).reshape(B, K, group * N).contiguous()

                actual, (code,) = run_and_get_code(
                    torch.compile(fn, backend="inductor", fullgraph=True), a, b
                )
                expected = fn(a, b)

                self.assertEqual(actual.shape, (B, M, N))
                torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-1)
                FileCheck().check("main_output_transform='grouped_n_contract'").check(
                    f"main_output_transform_group={group}"
                ).check_not("extern_kernels.bmm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_bmm_local_n_sum_aux_fuses(self):
        B = 2
        M = 32
        K = 64
        N = 64
        group = 4

        def fn(a, b):
            def epilogue(acc):
                x = acc.float().view(B, M, -1, group)
                return acc.relu(), x.sum(-1)

            return gemm_epilogue_fusion(
                torch.ops.aten.bmm.default,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(B, M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(B, K, N, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected = fn(a, b)

        torch.testing.assert_close(actual[0], expected[0], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(actual[1], expected[1], atol=1e-2, rtol=1e-2)
        FileCheck().check(f"local_reduce_group={group}").check(
            "local_reduce_out="
        ).check_not("extern_kernels.bmm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_bmm_local_m_reduce_feeding_main_fuses(self):
        B = 2
        M = 128
        K = 64
        N = 64
        group = 2

        def fn(a, b):
            def epilogue(acc):
                x = acc.float().view(B, -1, group, N)
                denom = x.sum(2, keepdim=True)
                return (x * denom.reciprocal()).view(B, M, N)

            return gemm_epilogue_fusion(
                torch.ops.aten.bmm.default,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(B, M, K, device="cuda", dtype=torch.float16)
        b = torch.rand(B, K, N, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected = fn(a, b)

        torch.testing.assert_close(actual.float(), expected.float(), atol=5e-2, rtol=5e-2)
        FileCheck().check(f"local_reduce_group={group}").check(
            "local_reduce_dim=0"
        ).check("local_reduce_feeds_main=True").check_not("extern_kernels.bmm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_bmm_local_m_sum_aux_fuses(self):
        B = 2
        M = 128
        K = 64
        N = 64
        group = 2

        def fn(a, b):
            def epilogue(acc):
                x = acc.float().view(B, -1, group, N)
                return acc.relu(), x.sum(2)

            return gemm_epilogue_fusion(
                torch.ops.aten.bmm.default,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(B, M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(B, K, N, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected = fn(a, b)

        torch.testing.assert_close(actual[0], expected[0], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(actual[1], expected[1], atol=1e-2, rtol=1e-2)
        FileCheck().check(f"local_reduce_group={group}").check(
            "local_reduce_dim=0"
        ).check("local_reduce_out=").check_not("extern_kernels.bmm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_bmm_local_amax_aux_fuses(self):
        B = 2
        M = 32
        K = 64
        N = 64
        group = 16

        cases = (
            (
                "amax_abs",
                lambda x: x.abs().amax(-1),
                torch.float32,
                "local_reduce_op='amax_abs'",
            ),
            (
                "mx_e8m0_scale",
                lambda x: mx_e8m0_scale(x.abs().amax(-1, keepdim=True)).view(
                    B, M, -1
                ),
                torch.float8_e8m0fnu,
                "local_reduce_op='mx_e8m0_scale'",
            ),
            (
                "nvfp4_e4m3_scale",
                lambda x: nvfp4_e4m3_scale(x.abs().amax(-1, keepdim=True)).view(
                    B, M, -1
                ),
                torch.float8_e4m3fn,
                "local_reduce_op='nvfp4_e4m3_scale'",
            ),
        )
        for name, aux_fn, aux_dtype, check_op in cases:
            with self.subTest(name=name):

                def fn(a, b):
                    def epilogue(acc):
                        x = acc.float().view(B, M, -1, group)
                        return acc.relu(), aux_fn(x)

                    return gemm_epilogue_fusion(
                        torch.ops.aten.bmm.default,
                        (a, b),
                        epilogue,
                        kernel_options={"backend": "QUACK"},
                    )

                a = torch.randn(B, M, K, device="cuda", dtype=torch.float16)
                b = torch.randn(B, K, N, device="cuda", dtype=torch.float16)

                actual, (code,) = run_and_get_code(
                    torch.compile(fn, backend="inductor", fullgraph=True), a, b
                )
                expected = fn(a, b)

                torch.testing.assert_close(actual[0], expected[0], atol=1e-2, rtol=1e-2)
                self.assertEqual(actual[1].dtype, aux_dtype)
                torch.testing.assert_close(
                    actual[1].float(), expected[1].float(), atol=1e-1, rtol=1.25e-1
                )
                FileCheck().check(f"local_reduce_group={group}").check(
                    "local_reduce_out="
                ).check(check_op).check_not("extern_kernels.bmm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_bmm_mxfp8_like_main_and_scale_fuses(self):
        B = 2
        M = 32
        K = 64
        N = 64
        group = 32

        def fn(a, b):
            def epilogue(acc):
                x = acc.float().view(B, M, -1, group)
                scale = mx_e8m0_scale(x.abs().amax(-1, keepdim=True))
                q = (x * scale.float().reciprocal()).clamp(min=-448.0, max=448.0).view(B, M, N)
                return q.to(torch.float8_e4m3fn), scale.view(B, M, -1)

            return gemm_epilogue_fusion(
                torch.ops.aten.bmm.default,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(B, M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(B, K, N, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected = fn(a, b)

        self.assertEqual(actual[0].dtype, torch.float8_e4m3fn)
        self.assertEqual(actual[1].dtype, torch.float8_e8m0fnu)
        torch.testing.assert_close(
            actual[0].float(), expected[0].float(), atol=32.0, rtol=1.25e-1
        )
        torch.testing.assert_close(actual[1].float(), expected[1].float(), atol=0.0, rtol=0.0)
        FileCheck().check("out_dtype=torch.float8_e4m3fn").check(
            f"local_reduce_group={group}"
        ).check("local_reduce_out=").check("local_reduce_op='mx_e8m0_scale'").check_not(
            "extern_kernels.bmm"
        ).run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_bmm_local_n_feeds_main_fuses(self):
        B = 2
        M = 32
        K = 64
        N = 64
        group = 16

        cases = (
            (
                "sum",
                lambda x: x.sum(-1, keepdim=True),
                "cute.ReductionOp.ADD",
            ),
            (
                "amax_abs",
                lambda x: x.abs().amax(-1, keepdim=True),
                "cute.ReductionOp.MAX",
            ),
        )
        for name, reduce_fn, check_op in cases:
            with self.subTest(name=name):

                def fn(a, b):
                    def epilogue(acc):
                        x = acc.float().view(B, M, -1, group)
                        denom = reduce_fn(x)
                        return (x * denom.reciprocal()).view(B, M, N)

                    return gemm_epilogue_fusion(
                        torch.ops.aten.bmm.default,
                        (a, b),
                        epilogue,
                        kernel_options={"backend": "QUACK"},
                    )

                a = torch.randn(B, M, K, device="cuda", dtype=torch.float16)
                b = torch.randn(B, K, N, device="cuda", dtype=torch.float16)

                actual, (code,) = run_and_get_code(
                    torch.compile(fn, backend="inductor", fullgraph=True), a, b
                )
                expected = fn(a, b)

                torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1.25e-1)
                FileCheck().check(check_op).check("broadcast_to").check_not(
                    "extern_kernels.bmm"
                ).run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_bmm_tuple_epilogue_generic_aux_fuses(self):
        def fn(a, b):
            def epilogue(acc):
                return acc.relu(), acc.float().abs()

            return gemm_epilogue_fusion(
                torch.ops.aten.bmm.default,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(2, 32, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(2, 64, 48, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected = fn(a, b)

        torch.testing.assert_close(actual[0], expected[0], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(actual[1], expected[1], atol=1e-2, rtol=1e-2)
        FileCheck().check("aux_out=").check_not("extern_kernels.bmm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_rejects_unsupported_shape_changing_main(self):
        M = 32
        K = 64
        N = 32

        def chunk_fn(a, b):
            def epilogue(acc):
                gate, up = acc.chunk(2, dim=-1)
                return torch.nn.functional.silu(gate) * up

            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        gate_w = torch.randn(K, N, device="cuda", dtype=torch.float16)
        up_w = torch.randn(K, N, device="cuda", dtype=torch.float16)
        b = torch.cat((gate_w, up_w), dim=1)
        with self.assertRaisesRegex(Exception, "shape-changing main epilogues"):
            torch.compile(chunk_fn, backend="inductor", fullgraph=True)(a, b)

        def group3_fn(a, b):
            def epilogue(acc):
                x = acc.view(M, -1, 3)
                return x[..., 0] + x[..., 1]

            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, 3 * N, device="cuda", dtype=torch.float16)
        with self.assertRaisesRegex(Exception, "shape-changing main epilogues"):
            torch.compile(group3_fn, backend="inductor", fullgraph=True)(a, b)

        def group8_fn(a, b):
            def epilogue(acc):
                x = acc.view(M, -1, 8)
                return x[..., 0] + x[..., 1]

            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        b = torch.randn(K, 8 * N, device="cuda", dtype=torch.float16)
        with self.assertRaisesRegex(Exception, "supports only groups 2 and 4"):
            torch.compile(group8_fn, backend="inductor", fullgraph=True)(a, b)

        def group_m_fn(a, b):
            def epilogue(acc):
                x = acc.view(-1, 2, N)
                return x[:, 0, :] + x[:, 1, :]

            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(2 * M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        with self.assertRaisesRegex(Exception, "M-mode shape-changing main epilogues"):
            torch.compile(group_m_fn, backend="inductor", fullgraph=True)(a, b)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_rejects_malformed_local_n_reduce_aux(self):
        M = 32
        N = 64
        group = 16

        def fn(a, b):
            def epilogue(acc):
                bad_aux = acc.float().view(1, -1, group).sum(-1)
                return acc.relu(), bad_aux

            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(M, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, N, device="cuda", dtype=torch.float16)

        with self.assertRaisesRegex(Exception, "generic aux tuple epilogues"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_tuple_epilogue_nvfp4_scale_aux_fuses(self):
        M = 64
        N = 64
        group = 16

        def fn(a, b):
            def epilogue(acc):
                x = acc.float().view(M, -1, group)
                scale = nvfp4_e4m3_scale(x.abs().amax(-1, keepdim=True))
                return acc.relu(), scale.view(M, -1)

            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(M, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, N, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected = fn(a, b)

        torch.testing.assert_close(actual[0], expected[0], atol=1e-2, rtol=1e-2)
        self.assertEqual(actual[1].dtype, torch.float8_e4m3fn)
        torch.testing.assert_close(
            actual[1].float(), expected[1].float(), atol=1e-1, rtol=1.25e-1
        )
        FileCheck().check(f"local_reduce_group={group}").check(
            "local_reduce_out="
        ).check("local_reduce_op='nvfp4_e4m3_scale'").check_not(
            "extern_kernels.mm"
        ).run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_rejects_nvfp4_scale_feeding_main(self):
        M = 64
        N = 64
        group = 16

        def fn(a, b):
            def epilogue(acc):
                x = acc.float().view(M, -1, group)
                scale = nvfp4_e4m3_scale(x.abs().amax(-1, keepdim=True))
                q = (x * scale.float().reciprocal()).clamp(min=-6.0, max=6.0).view(M, N)
                return q, scale.view(M, -1)

            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(M, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, N, device="cuda", dtype=torch.float16)

        with self.assertRaisesRegex(Exception, "E4M3 rounding semantics"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_addmm_fp8_main_output_fuses(self):
        def fn(c, a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.addmm.default,
                (c, a, b),
                lambda acc: acc.clamp(min=-448.0, max=448.0).to(torch.float8_e4m3fn),
                kernel_options={"backend": "QUACK"},
            )

        c = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        a = torch.randn(64, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), c, a, b
        )
        expected = fn(c, a, b)

        self.assertEqual(actual.dtype, torch.float8_e4m3fn)
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=32.0, rtol=1.25e-1
        )
        FileCheck().check("out_dtype=torch.float8_e4m3fn").check_not(
            "extern_kernels.mm"
        ).run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_honors_epilogue_add_alpha(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: torch.add(acc, 1.0, alpha=2.0).relu(),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        torch.testing.assert_close(actual, fn(a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("@cute.jit").check("gemm_epilogue(").check_not(
            "call_quack_gemm_epilogue"
        ).run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_generates_cutedsl_math_epilogue(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: torch.abs(acc),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        torch.testing.assert_close(actual, fn(a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("from cutlass._mlir.dialects import math as mlir_math").check(
            "mlir_math.absf"
        ).check("gemm_epilogue(").check_not("call_quack_gemm_epilogue").run(code)


if __name__ == "__main__":
    run_tests(needs="filelock")
