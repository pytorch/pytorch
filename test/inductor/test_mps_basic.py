# Owner(s): ["module: mps"]
import importlib
import os
import sys
import unittest

import numpy as np

import torch
from torch.testing import FileCheck, make_tensor
from torch.testing._internal.common_dtype import get_all_dtypes
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    MACOS_VERSION,
    parametrize,
)


MPS_UNSUPPORTED_TYPES = [torch.double, torch.cdouble] + (
    [torch.bfloat16] if MACOS_VERSION < 14.0 else []
)
MPS_DTYPES = [t for t in get_all_dtypes() if t not in MPS_UNSUPPORTED_TYPES]

importlib.import_module("filelock")

pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model_gpu,
    CommonTemplate,
    TestCase,
)


# TODO: Remove this file.
# This tests basic MPS compile functionality


@unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
@instantiate_parametrized_tests
class MPSBasicTests(TestCase):
    is_dtype_supported = CommonTemplate.is_dtype_supported
    common = check_model_gpu
    device = "mps"

    @parametrize("dtype", MPS_DTYPES)
    def test_add(self, dtype):
        self.common(
            lambda a, b: a + b,
            (
                make_tensor(1024, dtype=dtype, device=self.device),
                make_tensor(1024, dtype=dtype, device=self.device),
            ),
            check_lowp=False,
        )

    def test_log(self):
        self.common(lambda x: x.log(), (torch.rand(1024),))

    def test_acos(self):
        self.common(lambda x: x.acos(), (torch.rand(1024),))

    def test_atanh(self):
        self.common(lambda x: x.atanh(), (torch.rand(1024),))

    def test_tanh(self):
        self.common(lambda x: x.tanh(), (torch.rand(1024),))

    def test_tanh_large_values(self):
        # Test that tanh handles large values correctly (should saturate to ±1)
        x = torch.tensor([-100.0, -50.0, -15.0, 0.0, 15.0, 50.0, 100.0], device="mps")

        @torch.compile
        def fn(x):
            return x.tanh()

        result = fn(x)
        if not torch.allclose(result[0], torch.tensor(-1.0, device="mps")):
            raise AssertionError("tanh(-100) should be -1")
        if not torch.allclose(result[-1], torch.tensor(1.0, device="mps")):
            raise AssertionError("tanh(100) should be +1")
        if torch.isnan(result).any():
            raise AssertionError("tanh should not produce NaN for large values")

    def test_floor(self):
        self.common(lambda x: x.floor(), (torch.rand(1024),))

    def test_sign(self):
        self.common(lambda x: x.sign(), (torch.rand(1024),))

    def test_sliced_input(self):
        self.common(
            lambda x: x[:, ::2].sin() + x[:, 1::2].cos(), (torch.rand(32, 1024),)
        )

    def test_where(self):
        def foo(x):
            rc = x.abs().sqrt()
            rc[x < 0] = -5
            return rc

        self.common(foo, (torch.rand(1024),))

    @parametrize("dtype", MPS_DTYPES)
    def test_cast(self, dtype):
        self.common(lambda a: a.to(dtype), (torch.rand(1024),))

    def test_broadcast(self):
        self.common(torch.add, (torch.rand(32, 1024), torch.rand(1024)))

    def test_inplace(self):
        def inc_(x):
            x += 1
            return x

        self.common(inc_, (torch.rand(1024),))

    def test_rms_norm_nograd(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/150629
        def fn(x, w):
            with torch.no_grad():
                return torch.nn.functional.rms_norm(x, x.shape, w)

        self.common(fn, (torch.rand(10), torch.ones(10)))

    def test_batchnorm_train_running_stats(self):
        # Regression test: missing closing threadgroup_barrier in
        # threadgroup_welford_{reduce,combine}
        torch.manual_seed(0)
        xs = [torch.randn(16, 8, 4, 4) for _ in range(10)]

        def run(device, compile_):
            torch.manual_seed(0)
            bn = torch.nn.BatchNorm2d(8).to(device).train()
            f = torch.compile(bn) if compile_ else bn
            for x in xs:
                f(x.to(device))
            return bn.running_mean.cpu(), bn.running_var.cpu()

        m_ref, v_ref = run("cpu", False)
        m_mps, v_mps = run("mps", True)
        self.assertEqual(m_mps, m_ref)
        self.assertEqual(v_mps, v_ref)

    def test_compile_numpy_scalar(self):
        def fn(x, y):
            return x / y

        self.common(fn, (torch.rand(10), np.exp(0.3)))

    def test_conv_transpose_channels_last(self):
        def fn(x, y):
            return torch.nn.functional.conv_transpose2d(x, y, stride=1, padding=1)

        self.common(
            fn,
            (
                torch.rand(1, 1, 16, 16).to(memory_format=torch.channels_last),
                torch.rand(1, 4, 8, 8),
            ),
        )

    def test_conv_train(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/161905
        def fn(x, y):
            return torch.nn.functional.conv2d(x, y, None, 1, 1, 1)

        self.common(
            fn,
            (
                torch.rand(4, 512, 7, 7, requires_grad=True),
                torch.rand(512, 512, 3, 3),
            ),
            check_gradient=True,
        )

    def test_cholesky(self):
        def fn(x):
            return (
                torch.linalg.cholesky(x, upper=False),
                torch.linalg.cholesky(x, upper=True),
            )

        self.common(fn, (torch.eye(64),), check_lowp=False)

    def test_reduced_max(self):
        # inductor test do not validate that max of say 16K half elements can be computed
        self.common(torch.max, (torch.rand(16384, dtype=torch.half),), check_lowp=False)

    def test_linalg_inv(self):
        def fn(x):
            return torch.linalg.inv(torch.linalg.cholesky(x))

        A = torch.diag(torch.tensor([20.0, 0.5, 5.0], dtype=torch.float32) ** 2)
        self.common(fn, (A,), check_lowp=False)

    def test_large_reduction(self):
        def fn(a, b):
            return (a[:, None] - b[None, :]).sum()

        a = torch.randn(32, device="mps")
        b = torch.randn(64, device="mps")
        self.common(
            fn,
            (
                a,
                b,
            ),
        )

    @parametrize("shape", [(4, 5000), (3, 1023), (7, 1025), (5, 32), (1, 30000)])
    def test_welford_reduction_dynamic_shape(self, shape):
        # (5, 32): single-stage welford_reduce
        # (3, 1023), (4, 5000), (7, 1025): multistage welford_reduce
        # (1, 30000): split reduction -> welford_combine
        @torch.compile(dynamic=True)
        def fn(x):
            return x.var(dim=-1)

        x = torch.randn(*shape, device=self.device)
        torch._dynamo.mark_dynamic(x, 1)
        self.assertEqual(fn(x), x.var(dim=-1))

    def test_while_loop_kernel_naming(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/187852
        # while_loop compiles cond and body as separate MetalScheduling instances,
        # each of which used to reset _kernel_fn_counter to 0, producing duplicate
        # "generated_kernel_0" names that caused a Metal mangled-name collision.
        def fn(iterations):
            def cond(i):
                return i < iterations

            def body(i):
                return (i + 2,)

            (out_i,) = torch._higher_order_ops.while_loop(
                cond, body, (torch.tensor(0, dtype=torch.int32, device=self.device),)
            )
            return out_i

        iters = torch.tensor(4, dtype=torch.int32, device=self.device)
        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(iters)
        self.assertEqual(result, torch.tensor(4, dtype=torch.int32, device=self.device))

    def test_welford_multistage_sibling_redeclare(self):
        # Regression test: BatchNorm2d-train emits two codegen passes on
        # the same multistage reduction root (welford + running-stats
        # update). Sibling indices (r0_1, r0_2) declared via the
        # root_already_processed branch must be redeclared in the second
        # loop scope; otherwise Metal compilation fails with
        # "use of undeclared identifier 'r0_2'".
        torch.manual_seed(0)
        bn_ref = torch.nn.BatchNorm2d(8).train()
        bn_mps = torch.nn.BatchNorm2d(8).to(self.device).train()
        bn_mps.load_state_dict(bn_ref.state_dict())
        x = torch.randn(4, 8, 32, 32)
        y_ref = bn_ref(x)
        y_mps = torch.compile(bn_mps)(x.to(self.device))
        self.assertEqual(y_mps.cpu(), y_ref)

    def test_sdpa_split_qkv(self):
        # regression test for metal compiler bug where fused (x / A) % B
        # produces wrong results, causing incorrect reads from non-contiguous.
        n_head, n_embd, seq_len = 6, 384, 1024
        x = torch.randn(16, seq_len, n_embd, device="mps")
        c_attn = torch.nn.Linear(n_embd, 3 * n_embd).to("mps").eval()
        qkv = c_attn(x)
        q, k, v = qkv.split(n_embd, dim=2)
        q = q.view(16, seq_len, n_head, n_embd // n_head).transpose(1, 2)
        k = k.view(16, seq_len, n_head, n_embd // n_head).transpose(1, 2)
        v = v.view(16, seq_len, n_head, n_embd // n_head).transpose(1, 2)

        def fn(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )

        self.common(fn, (q, k, v), atol=1e-4, rtol=1e-4, check_lowp=False)

    @parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_sdpa_prefill_strided(self, dtype):
        torch.manual_seed(0)
        B, H, S, D = 1, 16, 1179, 128

        def fn(q, k, v, mask):
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )

        q, k, v = (
            torch.randn(B, S, H, D, device=self.device, dtype=dtype) for _ in range(3)
        )
        mask = torch.zeros(B, 1, S, S, device=self.device, dtype=dtype)
        self.assertEqual(torch.compile(fn)(q, k, v, mask), fn(q, k, v, mask))

    def test_nested_masked_cat(self):
        # Regression test for YOLOv3 compilation failure on MPS.
        # See https://github.com/pytorch/pytorch/actions/runs/23477894502
        # YOLOv3 detection heads do view/permute/clone, then in-place slice
        # assignment (sigmoid+grid, exp*anchor, sigmoid) followed by cat across
        # scales. The slice_scatter decomposition fused with cat produces nested
        # ops.masked calls in Metal codegen. Without depth-aware variable
        # prefixes, inner scoped variables shadow outer ones, causing:
        #   "variable 'tmp_scoped_1' declared with deduced type 'auto'
        #    cannot appear in its own initializer"
        na, no = 3, 5

        def head(p, grid, anchor_wh):
            bs, _, ny, nx = p.shape
            p = p.view(bs, na, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            io = p.clone()
            io[..., :2] = torch.sigmoid(io[..., :2]) + grid
            io[..., 2:4] = torch.exp(io[..., 2:4]) * anchor_wh
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, no)

        def fn(p1, p2, grid1, grid2, anchor_wh1, anchor_wh2):
            return torch.cat(
                [head(p1, grid1, anchor_wh1), head(p2, grid2, anchor_wh2)], dim=1
            )

        self.common(
            fn,
            (
                torch.randn(1, na * no, 4, 4, device="mps"),
                torch.randn(1, na * no, 8, 8, device="mps"),
                torch.randn(1, 1, 4, 4, 2, device="mps"),
                torch.randn(1, 1, 8, 8, 2, device="mps"),
                torch.randn(1, na, 1, 1, 2, device="mps"),
                torch.randn(1, na, 1, 1, 2, device="mps"),
            ),
        )


@unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
class MPSBasicTestsAOTI(TestCase):
    def check_model(self, m, inp, dynamic_shapes=None):
        res2 = m(*inp)
        ep = torch.export.export(m, inp, dynamic_shapes=dynamic_shapes)
        path = torch._inductor.aoti_compile_and_package(ep)
        m = torch._inductor.aoti_load_package(path)
        res = m(*inp)
        if not torch.allclose(res, res2):
            raise AssertionError

    def test_add_mps(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        inp = (torch.ones(3, 3, device="mps"), torch.ones(3, 3, device="mps"))
        m = M().to("mps")
        self.check_model(m, inp)

    def test_tanh_codegen(self):
        # Verify that tanh uses metal::precise::tanh in generated Metal shader
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.tanh()

        example_inputs = (torch.randn(1024, device="mps"),)
        model = Model()

        ep = torch.export.export(model, example_inputs)
        package_path = torch._export.aot_compile(ep.module(), example_inputs)

        with open(os.path.splitext(package_path)[0] + ".cpp") as cpp:
            src_code = cpp.read()
            # Verify metal::precise::tanh is used (not clamped version)
            FileCheck().check("metal::precise::tanh").run(src_code)

    def test_fallback_mps(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.nn.functional.linear(x, y)

        inp = (
            torch.randn(10, 10, device="mps"),
            torch.randn(10, 10, device="mps"),
        )
        m = M().to("mps")
        self.check_model(m, inp)

    def test_c10(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return torch.cat(tensors=torch.split(x, 4, dim=1), dim=-2)

        inp = (torch.randn(2, 8, device="mps"),)
        m = M().to("mps")
        self.check_model(m, inp)

    def test_two_const(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.y = torch.ones(3, 3, device="mps")
                self.z = torch.full((3, 3), 2, device="mps")

            def forward(self, x):
                return x + self.y + self.z

        inp = (torch.ones(3, 3, device="mps"),)
        m = Model().to(device="mps")
        self.check_model(m, inp)

    def test_simple_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                add_0 = x + y
                return torch.nn.functional.relu(input=add_0, inplace=False)

        x = torch.randn(128, 2048, device="mps")
        y = torch.randn(128, 2048, device="mps")
        inp = (x, y)

        m = Model().to(device="mps")
        dim0_x = torch.export.Dim("dim0_x", min=1, max=2048)
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_x}}

        self.check_model(m, inp, dynamic_shapes)

    def test_dynamic_cat(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                return torch.cat([a, b], dim=0)

        a = torch.randn(2, 4, device="mps")
        b = torch.randn(3, 4, device="mps")
        inp = (a, b)
        m = Model().to(device="mps")

        dim0_a = torch.export.Dim("dim0_a", min=1, max=10)
        dim0_b = torch.export.Dim("dim0_b", min=1, max=20)
        dynamic_shapes = {"a": {0: dim0_a}, "b": {0: dim0_b}}
        self.check_model(m, inp, dynamic_shapes)

    def test_reuse_kernel(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                a = torch.sin(x)
                b = torch.mm(a, y)
                c = torch.sin(b)
                d = torch.mm(b, c)
                return d

        example_inputs = (
            torch.randn(87, 87, device="mps"),
            torch.randn(87, 87, device="mps"),
        )
        model = Model()

        ep = torch.export.export(model, example_inputs)
        package_path = torch._export.aot_compile(ep.module(), example_inputs)

        target_str = "aoti_torch_mps_get_kernel_function("
        target_count = 1

        with open(os.path.splitext(package_path)[0] + ".cpp") as cpp:
            src_code = cpp.read()
            FileCheck().check_count(
                target_str,
                target_count,
                exactly=True,
            ).run(src_code)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if torch.backends.mps.is_available():
        run_tests(needs="filelock")
