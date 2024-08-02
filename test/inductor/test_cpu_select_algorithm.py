# Owner(s): ["oncall: cpu inductor"]
import contextlib
import functools
import sys
import unittest
from typing import Optional
from unittest.mock import patch

import torch
import torch._dynamo.config
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config
import torch._inductor.select_algorithm as select_algorithm
from torch._dynamo.utils import counters
from torch._inductor.cpu_vec_isa import VecAMX
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_quantization import _generate_qdq_quantized_model
from torch.testing._internal.common_utils import IS_MACOS, parametrize, TEST_MKL


try:
    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise

check_model = test_torchinductor.check_model

aten = torch.ops.aten


def patches(fn):
    def skip_cache(self, choices, name, key, benchmark):
        if benchmark is None:
            return {}
        timings = benchmark(choices)
        for choice, timing in timings.items():
            if isinstance(choice, select_algorithm.ExternKernelCaller):
                # we intentionally make ATEN kernel slower to cover the cases
                # where template kernels are always chosen with fusions applied
                # and correctness checks at runtime.
                timings[choice] = timing * 1000
        return timings

    for patcher in [
        dynamo_config.patch(verbose=True),
        # Fails due to https://github.com/pytorch/pytorch/issues/131929
        dynamo_config.patch(inline_inbuilt_nn_modules=False),
        inductor_config.patch(
            debug=True,
            max_autotune=True,
            epilogue_fusion=True,
            max_autotune_gemm_backends="CPP,ATEN",
        ),
        patch.object(select_algorithm, "VERIFY", dict(atol=1e-4, rtol=1e-4)),
        patch.object(select_algorithm.AlgorithmSelectorCache, "lookup", skip_cache),
    ]:
        fn = patcher(fn)

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        counters.clear()
        torch.manual_seed(12345)
        return fn(*args, **kwargs)

    return wrapped


@contextlib.contextmanager
def verify(dtype):
    # For bfloat16 and half, we have to relax the tolerance
    # due to the difference associave orders in different
    # kernel implementations
    atol, rtol = 1e-4, 1e-4
    if dtype == torch.half or dtype == torch.bfloat16:
        atol, rtol = 1e-2, 1e-2
    with patch.object(select_algorithm, "VERIFY", dict(atol=atol, rtol=rtol)):
        yield atol, rtol


def _get_epilogue(epilogue: str, other: Optional[torch.Tensor] = None):
    if epilogue == "none":
        return lambda x: x
    elif epilogue == "relu":
        return torch.nn.ReLU()
    elif epilogue == "gelu":
        return torch.nn.GELU()
    elif epilogue == "silu":
        return torch.nn.SiLU()
    elif epilogue == "sigmoid":
        return torch.nn.Sigmoid()
    elif epilogue == "tanh":
        return torch.nn.Tanh()
    elif epilogue == "hardswish":
        return torch.nn.Hardswish()
    elif epilogue == "hardsigmoid":
        return torch.nn.Hardsigmoid()
    elif epilogue == "leaky_relu":
        return torch.nn.LeakyReLU()
    elif epilogue == "hardtanh":
        return torch.nn.Hardtanh()
    elif epilogue == "add":
        return lambda x: x + other
    elif epilogue == "sub":
        return lambda x: x - other
    elif epilogue == "mul":
        return lambda x: x * other
    elif epilogue == "div":
        return lambda x: x / other


class BaseTestSelectAlgorithm(TestCase):
    def _check_amx_counter(self, vec_amx):
        if vec_amx:
            self.assertTrue(counters["inductor"]["cpp_micro_gemm_amx_counter"] > 0)
        else:
            self.assertEqual(counters["inductor"]["cpp_micro_gemm_amx_counter"], 0)


class TestSelectAlgorithm(BaseTestSelectAlgorithm):
    common = check_model

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("batch_size", (1, 2, 1000))
    @parametrize("in_features", (1, 1000))
    @parametrize("out_features", (1, 1024))
    @parametrize("bias", (True, False))
    @parametrize("input_3d", (True, False))
    @dtypes(torch.float, torch.bfloat16, torch.half)
    def test_linear_static_shapes(
        self, batch_size, in_features, out_features, bias, input_3d, dtype
    ):
        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)

            def forward(self, x):
                return self.linear(x)

        counters.clear()
        mod = M(bias=bias).to(dtype=dtype).eval()
        B = (2, batch_size) if input_3d else (batch_size,)
        v = torch.randn(*B, in_features).to(dtype=dtype)
        with verify(dtype) as (atol, rtol):
            self.common(mod, (v,), atol=atol, rtol=rtol)
        if (
            counters["inductor"]["decompose_mm"] > 0
            or counters["inductor"]["decompose_addmm"] > 0
        ):
            # This is a special case where we go directly with vectorized codegen
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 0)
        else:
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("bias", (True, False))
    @dtypes(torch.float)
    def test_linear_input_transpose(self, bias, dtype):
        batch_size = 384
        in_features = 196
        out_features = 384

        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)

            @torch.compile
            def forward(self, x):
                return self.linear(x)

        counters.clear()
        mod = M(bias=bias).to(dtype=dtype).eval()
        v = torch.randn(in_features, batch_size).to(dtype=dtype)
        self.common(mod, (v.transpose(0, 1),))
        # TODO(jgong5): support transposed input
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 0)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("batch_size", (384,))
    @parametrize("in_features", (196,))
    @parametrize("out_features", (384, 385))
    @parametrize("bias", (True, False))
    @parametrize(
        "epilogue",
        (
            "relu",
            "gelu",
            "silu",
            "sigmoid",
            "tanh",
            "hardswish",
            "hardsigmoid",
            "leaky_relu",
            "hardtanh",
            "add",
            "sub",
            "mul",
            "div",
        ),
    )
    @dtypes(torch.float, torch.bfloat16, torch.half)
    def test_linear_with_pointwise(
        self, batch_size, in_features, out_features, bias, epilogue, dtype
    ):
        class M(torch.nn.Module):
            def __init__(self, bias, epilogue, other):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)
                self.epilogue = _get_epilogue(epilogue, other)

            def forward(self, x):
                return self.epilogue(self.linear(x))

        counters.clear()
        v = torch.randn(batch_size, in_features).to(dtype=dtype)
        u = torch.randn(batch_size, out_features).to(dtype=dtype)
        mod = M(bias=bias, epilogue=epilogue, other=u).to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (v,), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)
        if (
            (
                dtype == torch.bfloat16
                or (
                    dtype == torch.float16
                    and torch.ops.mkldnn._is_mkldnn_fp16_supported()
                )
            )
            and epilogue != "mul"
            and epilogue != "div"
            or (dtype == torch.half and epilogue == "add" and not bias)
        ):
            # Several scenarios where epilogue fusion is not counted in:
            # 1. For bfloat16, the epilogue fusion is part of the template,
            #    not fused via scheduler. This will also be true for float16 when
            #    hardware has the float16 instruction. The exception is mul or
            #    div fusion which is not supported for oneDNN linear.
            # 2. For float16, since oneDNN linear is not applied, linear w/o bias
            #    plus epilogue add is treated as linear w/ bias.
            self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 0)
        else:
            self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("batch_size", (384,))
    @parametrize("in_features", (196,))
    @parametrize("out_features", (128, 129))
    @parametrize("bias", (True, False))
    @parametrize(
        "epilogue",
        (
            "none",
            "relu",
            "add",
            "sub",
            "mul",
        ),
    )
    @dtypes(torch.float, torch.bfloat16, torch.half)
    def test_linear_with_transpose(
        self, batch_size, in_features, out_features, bias, epilogue, dtype
    ):
        class M(torch.nn.Module):
            def __init__(self, bias, epilogue, other):
                super().__init__()
                self.epilogue = _get_epilogue(epilogue, other)
                self.linear = torch.nn.Linear(in_features, out_features, bias)

            def forward(self, x, y):
                return self.epilogue(self.linear(x)).transpose(0, 1) + y

        counters.clear()
        v = torch.randn(batch_size, in_features).to(dtype=dtype)
        u = torch.randn(out_features, batch_size).to(dtype=dtype)
        other = torch.randn(batch_size, out_features).to(dtype=dtype)
        mod = M(bias=bias, epilogue=epilogue, other=other).to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (v, u), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)
        self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("batch_size", (384,))
    @parametrize("in_features", (196,))
    @parametrize("out_features", (384, 385))
    @parametrize("bias", (True, False))
    @parametrize(
        "unary",
        ("relu",),
    )
    @parametrize(
        "binary",
        (
            "add",
            "sub",
            "mul",
            "div",
        ),
    )
    @dtypes(torch.float, torch.bfloat16, torch.half)
    def test_linear_with_unary_binary(
        self, batch_size, in_features, out_features, bias, unary, binary, dtype
    ):
        class M(torch.nn.Module):
            def __init__(self, bias, unary, binary, other):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)
                self.unary = _get_epilogue(unary)
                self.binary = _get_epilogue(binary, other)

            def forward(self, x):
                return self.binary(self.unary(self.linear(x)))

        counters.clear()
        v = torch.randn(batch_size, in_features).to(dtype=dtype)
        u = torch.randn(batch_size, out_features).to(dtype=dtype)
        mod = M(bias=bias, unary=unary, binary=binary, other=u).to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (v,), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)
        self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @parametrize("batch_size", (1024,))
    @parametrize("in_features", (1024,))
    @parametrize("out_features", (1024, 1025))
    @parametrize("bias", (True, False))
    @dtypes(torch.bfloat16)
    def test_linear_amx(self, batch_size, in_features, out_features, bias, dtype):
        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)

            def forward(self, x):
                return self.linear(x)

        counters.clear()
        v = torch.randn(batch_size, in_features).to(dtype=dtype)
        mod = M(bias=bias).to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (v,), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)
        vec_amx = VecAMX()
        self._check_amx_counter(vec_amx)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @parametrize("batch_size", (384,))
    @parametrize("in_features", (196,))
    @parametrize("out_features", (384,))
    @parametrize("bias", (True, False))
    @dtypes(torch.bfloat16)
    def test_linear_with_embedding(
        self, batch_size, in_features, out_features, bias, dtype
    ):
        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias).to(
                    dtype=dtype
                )
                self.emb = torch.nn.Embedding(64, out_features)

            def forward(self, idx, x):
                return self.emb(idx) + self.linear(x)

        idx = torch.randint(0, 64, (batch_size,))
        x = torch.randn(batch_size, in_features).to(dtype=dtype)
        mod = M(bias=bias).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (idx, x), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)
        self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("batch_size", (32,))
    @parametrize("in_features", (128,))
    @parametrize("out_features", (64, 65))
    @parametrize("bias", (False, True))
    @parametrize("input_3d", (False, True))
    @dtypes(torch.float32, torch.bfloat16)
    @parametrize(
        "epilogue",
        (
            "none",
            "relu",
            "gelu",
        ),
    )
    def test_quantized_linear_with_pointwise(
        self, batch_size, in_features, out_features, bias, input_3d, dtype, epilogue
    ):
        B = (2, batch_size) if input_3d else (batch_size,)
        input = torch.randn(*B, in_features).to(dtype=torch.float32)

        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)
                self.epilogue = _get_epilogue(epilogue)
                self.linear2 = torch.nn.Linear(out_features, out_features, bias)
                self.epilogue2 = _get_epilogue(epilogue)

            def forward(self, x):
                res = self.epilogue(self.linear(x))
                res = self.epilogue2(self.linear2(res))
                return res

        counters.clear()
        ref_quantized_mod = _generate_qdq_quantized_model(
            M(bias=bias).eval(),
            (input,),
        )

        atol, rtol = 1e-3, 1e-3
        if dtype == torch.bfloat16:
            atol, rtol = 5e-2, 5e-2

        with patch.object(
            select_algorithm, "VERIFY", dict(atol=atol, rtol=rtol)
        ), torch.no_grad(), torch.autocast(
            "cpu", enabled=(dtype == torch.bfloat16), dtype=dtype
        ):
            ref_res = ref_quantized_mod(input)
            cfn = torch.compile(ref_quantized_mod)
            res = cfn(input)
            self.assertEqual(
                res,
                ref_res,
                atol=atol,
                rtol=rtol,
                equal_nan=True,
                exact_dtype=True,
            )
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 2)
            self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 0)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("batch_size", (32,))
    @parametrize("in_features", (128,))
    @parametrize("out_features", (64, 65))
    @parametrize("bias", (False, True))
    @parametrize("input_3d", (False, True))
    @parametrize("int8_mixed_bf16", (False, True))
    @dtypes(torch.float32, torch.bfloat16)
    @parametrize(
        "epilogue",
        (
            "none",
            "relu",
        ),
    )
    def test_quantized_linear_with_pointwise_binary(
        self,
        batch_size,
        in_features,
        out_features,
        bias,
        input_3d,
        int8_mixed_bf16,
        dtype,
        epilogue,
    ):
        if not int8_mixed_bf16 and dtype == torch.bfloat16:
            return
        B = (2, batch_size) if input_3d else (batch_size,)
        input = torch.randn(*B, in_features).to(dtype=torch.float32)

        other = torch.randn(*B, out_features).to(dtype=dtype)
        # Avoid hiting qlinear inplace sum fusion
        if input_3d:
            other2 = torch.randn(B[0] * B[1], out_features).to(dtype=dtype)
        else:
            other2 = torch.randn(1, *B, out_features).to(dtype=dtype)

        class M(torch.nn.Module):
            def __init__(self, bias, input_3d):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)
                self.epilogue = _get_epilogue(epilogue)
                self.linear2 = torch.nn.Linear(out_features, out_features, bias)
                self.epilogue2 = _get_epilogue(epilogue)
                self.input_3d = input_3d

            def forward(self, x, other, other2):
                res = self.epilogue(self.linear(x) + other)
                # Avoid hiting qlinear inplace sum fusion
                if self.input_3d:
                    other2 = other2.view(2, other2.size(0) // 2, other2.size(1))
                else:
                    other2 = other2.view(other2.size(1), other2.size(2))
                res = self.epilogue2(self.linear2(res) + other2)
                return res

        counters.clear()
        ref_quantized_mod = _generate_qdq_quantized_model(
            M(bias=bias, input_3d=input_3d).eval(),
            (input, other, other2),
        )
        atol, rtol = 5e-2, 5e-2
        with patch.object(
            select_algorithm, "VERIFY", dict(atol=atol, rtol=rtol)
        ), torch.no_grad(), torch.autocast(
            "cpu", enabled=int8_mixed_bf16, dtype=torch.bfloat16
        ):
            ref_res = ref_quantized_mod(input, other, other2)
            cfn = torch.compile(ref_quantized_mod)
            res = cfn(input, other, other2)
            self.assertEqual(
                res,
                ref_res,
                atol=atol,
                rtol=rtol,
                equal_nan=True,
                exact_dtype=True,
            )
            self.assertEqual(
                counters["inductor"]["select_algorithm_autotune"],
                2,
            )
            self.assertEqual(
                counters["inductor"]["cpp_epilogue_fusion_counter"],
                0,
            )

    @inductor_config.patch({"freezing": True})
    @patches
    @torch._dynamo.config.patch(inline_inbuilt_nn_modules=True)
    @torch.no_grad
    @parametrize("batch_size", (3, 16, 32, 49))
    @parametrize("in_features", (4, 68, 128))  # k should be a multiple of 4
    @parametrize("out_features", (64, 65))
    @parametrize("bias", (True, False))
    def test_quantized_linear_amx(self, batch_size, in_features, out_features, bias):
        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)

            def forward(self, x):
                return self.linear(x)

        counters.clear()
        v = torch.randn(batch_size, in_features).to(dtype=torch.float32)
        ref_quantized_mod = _generate_qdq_quantized_model(
            M(bias=bias).eval(),
            (v,),
        )
        atol, rtol = 1e-2, 1e-2
        with patch.object(select_algorithm, "VERIFY", dict(atol=atol, rtol=rtol)):
            self.common(ref_quantized_mod, (v,), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)
        vec_amx = VecAMX()
        self._check_amx_counter(vec_amx)

    @inductor_config.patch({"freezing": True})
    @inductor_config.patch({"cpp.gemm_max_k_slices": 0})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("batch_size", (2,))
    @parametrize("in_features", (1000,))
    @parametrize("out_features", (2,))
    @parametrize("bias", (True, False))
    @parametrize(
        "epilogue",
        (
            "none",
            "relu",
        ),
    )
    @dtypes(torch.float, torch.bfloat16, torch.half)
    def test_linear_k_slicing(
        self, batch_size, in_features, out_features, bias, epilogue, dtype
    ):
        class M(torch.nn.Module):
            def __init__(self, bias, epilogue, other):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)
                self.epilogue = _get_epilogue(epilogue, other)

            def forward(self, x):
                return self.epilogue(self.linear(x))

        counters.clear()
        v = torch.randn(batch_size, in_features).to(dtype=dtype)
        u = torch.randn(batch_size, out_features).to(dtype=dtype)
        mod = M(bias=bias, epilogue=epilogue, other=u).to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (v,), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)


@dynamo_config.patch({"dynamic_shapes": True, "assume_static_by_default": False})
class _DynamicShapesTestBase(BaseTestSelectAlgorithm):
    pass


class TestSelectAlgorithmDynamicShapes(_DynamicShapesTestBase):
    common = check_model
    test_linear_dynamic_shapes = TestSelectAlgorithm.test_linear_static_shapes
    test_linear_with_pointwise_dynamic_shapes = (
        TestSelectAlgorithm.test_linear_with_pointwise
    )
    test_linear_with_transpose_dynamic_shapes = (
        TestSelectAlgorithm.test_linear_with_transpose
    )
    test_linear_with_unary_binary_dynamic_shapes = (
        TestSelectAlgorithm.test_linear_with_unary_binary
    )
    test_linear_amx_dynamic_shapes = TestSelectAlgorithm.test_linear_amx
    test_linear_with_embedding_dynamic_shapes = (
        TestSelectAlgorithm.test_linear_with_embedding
    )
    test_quantized_linear_with_pointwise_dynamic_shapes = (
        TestSelectAlgorithm.test_quantized_linear_with_pointwise
    )
    test_quantized_linear_with_pointwise_binary_dynamic_shapes = (
        TestSelectAlgorithm.test_quantized_linear_with_pointwise_binary
    )
    test_quantized_linear_amx_dynamic_shapes = (
        TestSelectAlgorithm.test_quantized_linear_amx
    )


instantiate_device_type_tests(TestSelectAlgorithm, globals(), only_for="cpu")
instantiate_device_type_tests(
    TestSelectAlgorithmDynamicShapes, globals(), only_for="cpu"
)


if __name__ == "__main__":
    from torch.testing._internal.inductor_utils import HAS_CPU

    if HAS_CPU and not IS_MACOS:
        run_tests()
