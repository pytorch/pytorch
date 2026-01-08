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
import torch._inductor.cpu_vec_isa
import torch._inductor.select_algorithm as select_algorithm
from torch._dynamo.utils import counters
from torch._inductor import test_operators
from torch._inductor.cpu_vec_isa import VecAMX
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_quantized import (
    _calculate_dynamic_per_channel_qparams,
)
from torch.testing._internal.common_utils import (
    IS_MACOS,
    IS_WINDOWS,
    parametrize,
    TEST_MKL,
)


try:
    try:
        from . import test_cpu_repro, test_torchinductor
    except ImportError:
        import test_cpu_repro  # @manual=fbcode//caffe2/test/inductor:test_cpu_repro-library
        import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise

check_model = test_torchinductor.check_model
set_num_threads = test_cpu_repro.set_num_threads
run_and_get_cpp_code = test_torchinductor.run_and_get_cpp_code

aten = torch.ops.aten


def patches(fn):
    def skip_cache(self, choices, name, key, benchmark, hint_override=None):
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
        dynamo_config.patch(inline_inbuilt_nn_modules=True),
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

    def _check_brgemm_counter(self, vec_amx):
        if vec_amx and torch.cpu._is_amx_fp16_supported():
            self.assertTrue(counters["inductor"]["cpp_micro_brgemm_counter"] > 0)
        else:
            self.assertEqual(counters["inductor"]["cpp_micro_brgemm_counter"], 0)


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
            self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 0)
        else:
            self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("in_features", (1000,))
    @parametrize("out_features", (1024,))
    @parametrize("bias", (True,))
    @dtypes(
        torch.float,
    )
    def test_linear_wgt_multi_users(self, in_features, out_features, bias, dtype):
        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.embeddings = torch.nn.Embedding(out_features, in_features)
                self.linear = torch.nn.Linear(in_features, out_features, bias)
                self.linear.weight = self.embeddings.weight

            def forward(self, x):
                x = self.embeddings(x)
                return self.linear(x)

        counters.clear()
        mod = M(bias=bias).to(dtype=dtype).eval()
        v = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        with verify(dtype) as (atol, rtol):
            self.common(mod, (v,), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)

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
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 0)

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
    @torch.fx.experimental._config.patch(use_duck_shape=False)
    @torch._dynamo.config.patch(specialize_float=True)
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
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)
        if (
            (
                (
                    dtype == torch.bfloat16
                    and torch.ops.mkldnn._is_mkldnn_bf16_supported()
                )
                or (
                    dtype == torch.float16
                    and torch.ops.mkldnn._is_mkldnn_fp16_supported()
                )
                or (
                    dtype == torch.float32
                    and not dynamo_config.assume_static_by_default
                )
            )
            and epilogue != "mul"
            and epilogue != "div"
            or (
                dtype in (torch.float16, torch.bfloat16)
                and epilogue == "add"
                and not bias
            )
        ):
            # Several scenarios where epilogue fusion is not counted in:
            # 1. For bfloat16, the epilogue fusion is part of the template,
            #    not fused via scheduler. This will also be true for float16 when
            #    hardware has the float16 instruction. And this will also be true
            #    for float32 dynamic mode. The exception is mul or div fusion
            #    which is not supported for oneDNN linear.
            # 2. For bfloat16/float16, when oneDNN linear is not applied, linear w/o bias
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
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)
        self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @parametrize("batch_size", (1,))
    @parametrize("in_features", (16,))
    @parametrize("image_size", (18,))
    @parametrize("out_features", (32,))
    @parametrize(
        "bias",
        (
            False,
            True,
        ),
    )
    @parametrize(
        "has_non_epilogue_users",
        (
            True,
            False,
        ),
    )
    @dtypes(torch.bfloat16)
    def test_linear_with_permute(
        self,
        batch_size,
        in_features,
        image_size,
        out_features,
        bias,
        has_non_epilogue_users,
        dtype,
    ):
        # Reproducer from the convnext model in timm
        class M(torch.nn.Module):
            def __init__(self, bias, has_non_epilogue_users):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)
                self._frozen_param398 = torch.randn(batch_size, out_features, 1, 1)
                self.conv = torch.nn.Conv2d(
                    out_features,
                    out_features,
                    kernel_size=7,
                    padding=3,
                    groups=out_features,
                )
                self.linear2 = torch.nn.Linear(out_features, out_features, bias)
                self._frozen_param400 = torch.randn(batch_size, out_features, 1, 1)
                self.has_non_epilogue_users = has_non_epilogue_users

            def forward(self, mul_272, _convolution_pointwise_default_31):
                out1 = torch.ops.prims.convert_element_type.default(
                    mul_272, torch.bfloat16
                )
                mul_272 = None

                _linear_pointwise_default_131 = self.linear(out1)
                permute_188 = torch.ops.aten.permute.default(
                    _linear_pointwise_default_131, [0, 3, 1, 2]
                )

                mul_273 = torch.ops.aten.mul.Tensor(permute_188, self._frozen_param398)
                add_187 = torch.ops.aten.add.Tensor(
                    mul_273, _convolution_pointwise_default_31
                )
                convert_element_type_847 = torch.ops.prims.convert_element_type.default(
                    add_187, torch.bfloat16
                )
                _convolution_pointwise_default_29 = self.conv(convert_element_type_847)
                permute_189 = torch.ops.aten.permute.default(
                    _convolution_pointwise_default_29, [0, 2, 3, 1]
                )
                permute_189 = self.linear2(permute_189)
                permute_189 = torch.ops.aten.permute.default(permute_189, [0, 3, 1, 2])
                permute_189 = torch.ops.aten.mul.Tensor(
                    permute_189, self._frozen_param400
                )
                # If template_buffer will be used by nodes other than the epilogue nodes,
                # we can't alias the template_buffer with the Y buffer.
                if self.has_non_epilogue_users:
                    add_191 = torch.ops.aten.add.Tensor(permute_189, add_187)
                    return add_191
                return permute_189

        view_12 = torch.randn(batch_size, image_size, image_size, in_features)
        _convolution_pointwise_default_31 = torch.randn(
            batch_size, out_features, image_size, image_size
        ).to(memory_format=torch.channels_last)

        mod = M(bias=bias, has_non_epilogue_users=has_non_epilogue_users).eval()
        with verify(dtype) as (atol, rtol), torch.cpu.amp.autocast():
            self.common(
                mod,
                (
                    view_12,
                    _convolution_pointwise_default_31,
                ),
                atol=atol,
                rtol=rtol,
            )
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 2)
        self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 2)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("batch_size", (8,))
    @parametrize("in_features", (3,))
    @parametrize("linear_in_features", (384,))
    @parametrize("out_features", (196,))
    @parametrize("bias", (True,))
    @dtypes(torch.float)
    def test_linear_with_input_of_flexible_layout(
        self, batch_size, in_features, linear_in_features, out_features, bias, dtype
    ):
        # Reproducer from the resmlp_12_224 model in timm
        flatten_BS = int(batch_size * linear_in_features)

        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_features,
                    linear_in_features,
                    kernel_size=16,
                    padding=0,
                    stride=16,
                    dilation=1,
                    groups=1,
                )
                self._frozen_param151 = torch.randn(1, 1, linear_in_features)
                self._frozen_param3 = torch.randn(1, 1, linear_in_features)
                self._frozen_param2 = torch.randn(linear_in_features)

                self.linear = torch.nn.Linear(out_features, out_features, bias)

            def forward(self, arg150_1):
                _convolution_pointwise_default = self.conv(arg150_1)
                view_73 = torch.ops.aten.reshape.default(
                    _convolution_pointwise_default,
                    [batch_size, linear_in_features, out_features],
                )
                _convolution_pointwise_default = None
                permute_62 = torch.ops.aten.permute.default(view_73, [0, 2, 1])
                view_73 = None
                mul_111 = torch.ops.aten.mul.Tensor(self._frozen_param151, permute_62)
                add_73 = torch.ops.aten.add.Tensor(self._frozen_param3, mul_111)
                permute_63 = torch.ops.aten.permute.default(add_73, [0, 2, 1])
                add_73 = None
                view_74 = torch.ops.aten.reshape.default(
                    permute_63, [flatten_BS, out_features]
                )
                permute_63 = None
                _mkl_linear_36 = self.linear(view_74)
                view_75 = torch.ops.aten.reshape.default(
                    _mkl_linear_36, [batch_size, linear_in_features, out_features]
                )
                _mkl_linear_36 = None
                permute_65 = torch.ops.aten.permute.default(view_75, [0, 2, 1])
                view_75 = None
                mul_112 = torch.ops.aten.mul.Tensor(self._frozen_param2, permute_65)
                _frozen_param2 = permute_65 = None
                add_74 = torch.ops.aten.add.Tensor(permute_62, mul_112)
                permute_62 = mul_112 = None
                return add_74

        v = torch.randn(batch_size, in_features, 224, 224).to(dtype=dtype)
        mod = M(bias=bias).to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (v,), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 1)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("batch_size", (8,))
    @parametrize("in_features", (128,))
    @parametrize("size_0", (4,))
    @parametrize("size_1", (14,))
    @parametrize("out_features", (512,))
    @parametrize("out_features_conv", (256,))
    @parametrize(
        "bias",
        (
            False,
            True,
        ),
    )
    @parametrize(
        "epilogue",
        (
            False,
            True,
        ),
    )
    @dtypes(torch.float32)
    def test_linear_unsupported_epilogue_fusion(
        self,
        batch_size,
        in_features,
        size_0,
        size_1,
        out_features,
        out_features_conv,
        bias,
        epilogue,
        dtype,
    ):
        img_size_0 = int(size_0 * size_0)
        img_size_1 = int(size_1 * size_1)
        conv_shape = int(size_0 * size_1)
        flatten_BS = int(batch_size * size_0 * size_0 * size_1 * size_1)

        # Reproducer from the jx_nest_base model in timm
        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear1 = torch.nn.Linear(in_features, in_features, bias=bias)
                self.linear2 = torch.nn.Linear(out_features, in_features, bias=bias)
                self.conv = torch.nn.Conv2d(
                    in_features,
                    out_features_conv,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    dilation=1,
                    groups=1,
                )
                self.epilogue = epilogue

            def forward(self, mul_239, view_425, add_184):
                _mkl_linear_91 = self.linear1(view_425)
                view_426 = torch.ops.aten.reshape.default(
                    _mkl_linear_91, [batch_size, img_size_0, img_size_1, in_features]
                )
                _mkl_linear_91 = None
                add_187 = torch.ops.aten.add.Tensor(add_184, view_426)
                add_184 = view_426 = None
                view_429 = torch.ops.aten.reshape.default(
                    mul_239, [flatten_BS, out_features]
                )
                mul_239 = None

                _mkl_linear_89 = self.linear2(view_429)
                if self.epilogue:
                    _mkl_linear_89 = torch.pow(_mkl_linear_89, 2)
                    _mkl_linear_89 = test_operators.realize(_mkl_linear_89)

                view_430 = torch.ops.aten.reshape.default(
                    _mkl_linear_89, [batch_size, img_size_0, img_size_1, in_features]
                )
                _mkl_linear_89 = None

                add_191 = torch.ops.aten.add.Tensor(add_187, view_430)
                add_187 = view_430 = None

                view_431 = torch.ops.aten.reshape.default(
                    add_191, [batch_size, size_0, size_0, size_1, size_1, in_features]
                )
                add_191 = None
                permute_203 = torch.ops.aten.permute.default(
                    view_431, [0, 1, 3, 2, 4, 5]
                )
                view_431 = None
                clone_188 = torch.ops.aten.clone.default(
                    permute_203, memory_format=torch.contiguous_format
                )
                permute_203 = None
                view_432 = torch.ops.aten.reshape.default(
                    clone_188, [batch_size, conv_shape, conv_shape, in_features]
                )
                clone_188 = None
                permute_204 = torch.ops.aten.permute.default(view_432, [0, 3, 1, 2])
                view_432 = None

                _convolution_pointwise_default_1 = self.conv(permute_204)

                return _convolution_pointwise_default_1

        mul_239 = torch.randn(batch_size, img_size_0, img_size_1, out_features)
        view_425 = torch.randn(flatten_BS, in_features)
        add_184 = torch.randn(batch_size, img_size_0, img_size_1, in_features)
        mod = M(bias=bias).eval()
        with (
            verify(dtype) as (atol, rtol),
            torch.cpu.amp.autocast(enabled=dtype == torch.bfloat16),
        ):
            self.common(
                mod,
                (
                    mul_239,
                    view_425,
                    add_184,
                ),
                atol=atol,
                rtol=rtol,
            )
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 2)
        # TODO: change cpp_epilogue_fusion_counter to 1 once supported
        self.assertEqual(
            counters["inductor"]["cpp_epilogue_fusion_counter"], 1 if epilogue else 0
        )

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
        self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 1)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("batch_size", (384,))
    @parametrize("in_features", (196,))
    @parametrize("out_features", (384,))
    @parametrize("bias", (True, False))
    @parametrize(
        "binary",
        ("add",),
    )
    @dtypes(torch.float, torch.bfloat16, torch.half)
    def test_linear_with_binary_input_3d(
        self, batch_size, in_features, out_features, bias, binary, dtype
    ):
        class M(torch.nn.Module):
            def __init__(self, bias, binary, other):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)
                self.binary = _get_epilogue(binary, other)

            def forward(self, x):
                return self.binary(self.linear(x))

        counters.clear()
        B = (2, batch_size)
        v = torch.randn(*B, in_features).to(dtype=dtype)
        u = torch.randn(*B, out_features).to(dtype=dtype)
        mod = M(bias=bias, binary=binary, other=u).to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (v,), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @set_num_threads(1)
    @dynamo_config.patch({"dynamic_shapes": True, "assume_static_by_default": False})
    @parametrize("batch_size", (256,))
    @parametrize("in_features", (3,))
    @parametrize("out_features", (1024,))
    @parametrize("out_features2", (2,))
    @parametrize("bias", (True, False))
    @dtypes(torch.float)
    def test_linear_local_and_global_buffer_dynamic_shapes(
        self, batch_size, in_features, out_features, out_features2, bias, dtype
    ):
        # Reproducer from soft_actor_critic
        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)
                self.linear1 = torch.nn.Linear(out_features, out_features, bias)
                self.linear2 = torch.nn.Linear(out_features, out_features2, bias)

            def forward(self, arg7_1):
                addmm_3 = self.linear(arg7_1)
                relu_2 = torch.ops.aten.relu.default(addmm_3)

                addmm_4 = self.linear1(relu_2)
                relu_3 = torch.ops.aten.relu.default(addmm_4)

                addmm_5 = self.linear2(relu_3)

                split_1 = torch.ops.aten.split.Tensor(addmm_5, 1, 1)
                getitem_2 = split_1[0]
                getitem_3 = split_1[1]

                tanh_1 = torch.ops.aten.tanh.default(getitem_3)

                add_62 = torch.ops.aten.add.Tensor(tanh_1, 1)

                mul_36 = torch.ops.aten.mul.Tensor(add_62, 6.0)
                add_69 = torch.ops.aten.add.Tensor(mul_36, -10.0)

                exp_1 = torch.ops.aten.exp.default(add_69)
                return (getitem_2, exp_1)

        counters.clear()
        v = torch.randn(batch_size, in_features).to(dtype=dtype)
        mod = M(bias=bias).to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (v,), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 3)
        self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 0)

    @unittest.skipIf(
        not torch._C._cpu._is_amx_tile_supported(), "AMX ISA support is required"
    )
    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @parametrize("batch_size", (1024,))
    @parametrize("in_features", (1024,))
    @parametrize("out_features", (1024, 1025))
    @parametrize("bias", (True, False))
    @dtypes(torch.bfloat16, torch.half)
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
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)
        vec_amx = VecAMX()
        # Currently brgemm config is only added for half
        if dtype == torch.half and not vec_amx.is_amx_fp16_supported():
            self._check_brgemm_counter(vec_amx)
        else:
            self._check_amx_counter(vec_amx)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("batch_size", (8,))
    @parametrize("in_features", (128,))
    @parametrize("in_features_2", (196,))
    @parametrize("out_features", (256,))
    @parametrize(
        "bias",
        (True,),
    )
    @dtypes(torch.float32)
    def test_linear_with_multiple_reindexers(
        self,
        batch_size,
        in_features,
        in_features_2,
        out_features,
        bias,
        dtype,
    ):
        flatten_BS = int(batch_size * in_features_2)

        # Reproducer from the levit_128 model in timm
        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    64,
                    128,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    dilation=1,
                    groups=1,
                )
                self.linear = torch.nn.Linear(in_features, out_features, bias=False)
                self._frozen_param221 = torch.randn(out_features)
                self._frozen_param389 = torch.randn(out_features)
                self._frozen_param20 = torch.randn(out_features)
                self._frozen_param21 = torch.randn(out_features)

            def forward(self, view_368):
                _mkl_linear_57 = self.linear(view_368)
                view_369 = torch.ops.aten.reshape.default(
                    _mkl_linear_57, [batch_size, in_features_2, out_features]
                )
                _mkl_linear_57 = None

                view_370 = torch.ops.aten.reshape.default(
                    view_369, [flatten_BS, out_features]
                )
                view_369 = None
                sub_85 = torch.ops.aten.sub.Tensor(view_370, self._frozen_param221)
                view_370 = _frozen_param221 = None
                mul_261 = torch.ops.aten.mul.Tensor(sub_85, self._frozen_param389)
                sub_85 = _frozen_param389 = None
                mul_262 = torch.ops.aten.mul.Tensor(mul_261, self._frozen_param20)
                mul_261 = _frozen_param20 = None
                add_219 = torch.ops.aten.add.Tensor(mul_262, self._frozen_param21)
                mul_262 = _frozen_param21 = None
                view_371 = torch.ops.aten.reshape.default(
                    add_219, [batch_size, in_features_2, out_features]
                )
                add_219 = None

                add_220 = torch.ops.aten.add.Tensor(view_371, 3)
                clamp_min_35 = torch.ops.aten.clamp_min.default(add_220, 0)
                add_220 = None
                clamp_max_35 = torch.ops.aten.clamp_max.default(clamp_min_35, 6)
                clamp_min_35 = None
                mul_263 = torch.ops.aten.mul.Tensor(view_371, clamp_max_35)
                view_371 = clamp_max_35 = None
                div_51 = torch.ops.aten.div.Tensor(mul_263, 6)
                mul_263 = None

                return div_51

        view_368 = torch.randn(flatten_BS, in_features)

        mod = M(bias=bias).eval()
        with verify(dtype) as (atol, rtol):
            self.common(
                mod,
                (view_368,),
                atol=atol,
                rtol=rtol,
            )
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)
        self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 2)

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
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)
        self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @parametrize("batch_size", (2,))
    @parametrize("in_features", (16,))
    @parametrize("seq_lens", (128,))
    @parametrize("out_features", (32,))
    @parametrize("bias", (True,))
    @dtypes(torch.bfloat16)
    def test_linear_with_indirect_indexing(
        self, batch_size, in_features, seq_lens, out_features, bias, dtype
    ):
        # Reproducer from the GPT2ForSequenceClassification model in HuggingFace
        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.wte = torch.nn.Embedding(128, seq_lens)
                self.wpe = torch.nn.Embedding(in_features, seq_lens)
                self.linear = torch.nn.Linear(out_features, seq_lens, bias)

            def forward(self, view_12, input_ids, view_9):
                inputs_embeds = self.wte(input_ids)

                position_ids = torch.arange(0, in_features, dtype=torch.long)
                position_ids = position_ids.unsqueeze(0)
                position_embeds = self.wpe(position_ids)

                add = inputs_embeds + position_embeds
                add_4 = view_9 + add

                _linear_pointwise_default_45 = self.linear(view_12)

                view_13 = torch.ops.aten.reshape.default(
                    _linear_pointwise_default_45, [batch_size, in_features, seq_lens]
                )
                out = torch.ops.aten.add.Tensor(add_4, view_13)

                return out

        view_12 = torch.randn(batch_size * in_features, out_features)
        input_ids = torch.randint(0, 128, (batch_size, in_features))
        view_9 = torch.randn(batch_size, in_features, seq_lens)
        mod = M(bias=bias).eval()
        with verify(dtype) as (atol, rtol), torch.cpu.amp.autocast():
            self.common(
                mod,
                (
                    view_12,
                    input_ids,
                    view_9,
                ),
                atol=atol,
                rtol=rtol,
            )
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)
        self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @parametrize("batch_size", (8,))
    @parametrize("in_features", (3,))
    @parametrize("in_features2", (192,))
    @parametrize("image_size", (224,))
    @parametrize("out_features", (64,))
    @parametrize(
        "bias",
        (True,),
    )
    @dtypes(torch.float32)
    def test_linear_with_in_out_buffer(
        self,
        batch_size,
        in_features,
        in_features2,
        image_size,
        out_features,
        bias,
        dtype,
    ):
        # Reproducer from the coat_lite_mini model in timm
        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self._frozen_param398 = torch.randn(batch_size, out_features, 1, 1)
                self.conv = torch.nn.Conv2d(
                    in_features,
                    out_features,
                    kernel_size=4,
                    padding=0,
                    stride=4,
                    dilation=1,
                    groups=1,
                )
                self.conv2 = torch.nn.Conv2d(
                    out_features,
                    out_features,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    dilation=1,
                    groups=out_features,
                )

                self.conv3 = torch.nn.Conv2d(
                    16,
                    16,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    dilation=1,
                    groups=16,
                )

                self.conv4 = torch.nn.Conv2d(
                    24,
                    24,
                    kernel_size=5,
                    padding=2,
                    stride=1,
                    dilation=1,
                    groups=24,
                )

                self.conv5 = torch.nn.Conv2d(
                    24,
                    24,
                    kernel_size=7,
                    padding=3,
                    stride=1,
                    dilation=1,
                    groups=24,
                )

                self.linear = torch.nn.Linear(out_features, in_features2, bias)

                self.linear2 = torch.nn.Linear(out_features, out_features, bias)
                self._frozen_param2 = torch.randn(out_features)
                self._frozen_param3 = torch.randn(out_features)
                self._frozen_param7 = torch.randn(out_features)
                self._frozen_param8 = torch.randn(out_features)
                self._frozen_param153 = torch.randn(batch_size, 1, out_features)

            def forward(self, arg152_1):
                _convolution_pointwise_default_35 = self.conv(arg152_1)
                arg152_1 = None

                view_168 = torch.ops.aten.reshape.default(
                    _convolution_pointwise_default_35, [8, 64, 3136]
                )
                _convolution_pointwise_default_35 = None
                permute_97 = torch.ops.aten.permute.default(view_168, [0, 2, 1])
                view_168 = None
                clone_65 = torch.ops.aten.clone.default(
                    permute_97, memory_format=torch.contiguous_format
                )
                permute_97 = None
                var_mean_21 = torch.ops.aten.var_mean.correction(
                    clone_65, [2], correction=0, keepdim=True
                )
                getitem_90 = var_mean_21[0]
                getitem_91 = var_mean_21[1]
                var_mean_21 = None
                add_82 = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
                getitem_90 = None
                rsqrt_21 = torch.ops.aten.rsqrt.default(add_82)
                add_82 = None
                sub_29 = torch.ops.aten.sub.Tensor(clone_65, getitem_91)
                clone_65 = getitem_91 = None
                mul_82 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_21)
                sub_29 = rsqrt_21 = None
                mul_83 = torch.ops.aten.mul.Tensor(mul_82, self._frozen_param2)
                mul_82 = None
                add_83 = torch.ops.aten.add.Tensor(mul_83, self._frozen_param3)
                mul_83 = None
                _frozen_param153 = self._frozen_param153
                cat_20 = torch.ops.aten.cat.default([_frozen_param153, add_83], 1)
                _frozen_param153 = add_83 = None
                slice_111 = torch.ops.aten.slice.Tensor(cat_20, 1, 0, 1)
                slice_113 = torch.ops.aten.slice.Tensor(
                    cat_20, 1, 1, 9223372036854775807
                )
                cat_20 = None
                permute_98 = torch.ops.aten.permute.default(slice_113, [0, 2, 1])
                slice_113 = None
                view_169 = torch.ops.aten.reshape.default(permute_98, [8, 64, 56, 56])
                permute_98 = None
                _convolution_pointwise_default_34 = self.conv2(view_169)

                add_84 = torch.ops.aten.add.Tensor(
                    _convolution_pointwise_default_34, view_169
                )
                _convolution_pointwise_default_34 = view_169 = None
                view_170 = torch.ops.aten.reshape.default(add_84, [8, 64, 3136])
                add_84 = None
                permute_99 = torch.ops.aten.permute.default(view_170, [0, 2, 1])
                view_170 = None
                cat_21 = torch.ops.aten.cat.default([slice_111, permute_99], 1)
                slice_111 = permute_99 = None
                var_mean_22 = torch.ops.aten.var_mean.correction(
                    cat_21, [2], correction=0, keepdim=True
                )
                getitem_92 = var_mean_22[0]
                getitem_93 = var_mean_22[1]
                var_mean_22 = None
                add_85 = torch.ops.aten.add.Tensor(getitem_92, 1e-06)
                getitem_92 = None
                rsqrt_22 = torch.ops.aten.rsqrt.default(add_85)
                add_85 = None
                sub_30 = torch.ops.aten.sub.Tensor(cat_21, getitem_93)
                getitem_93 = None
                mul_84 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_22)
                sub_30 = rsqrt_22 = None
                mul_85 = torch.ops.aten.mul.Tensor(mul_84, self._frozen_param7)
                mul_84 = None
                add_86 = torch.ops.aten.add.Tensor(mul_85, self._frozen_param8)
                mul_85 = None
                view_171 = torch.ops.aten.reshape.default(add_86, [25096, 64])
                add_86 = None

                _mkl_linear_32 = self.linear(view_171)
                view_171 = None

                view_172 = torch.ops.aten.reshape.default(
                    _mkl_linear_32, [8, 3137, 192]
                )
                _mkl_linear_32 = None
                view_173 = torch.ops.aten.reshape.default(view_172, [8, 3137, 3, 8, 8])
                view_172 = None
                permute_101 = torch.ops.aten.permute.default(view_173, [2, 0, 3, 1, 4])
                view_173 = None
                unbind_8 = torch.ops.aten.unbind.int(permute_101)
                permute_101 = None
                getitem_94 = unbind_8[0]
                getitem_95 = unbind_8[1]
                getitem_96 = unbind_8[2]
                unbind_8 = None
                clone_66 = torch.ops.aten.clone.default(
                    getitem_95, memory_format=torch.contiguous_format
                )
                getitem_95 = None
                amax_8 = torch.ops.aten.amax.default(clone_66, [2], True)
                sub_31 = torch.ops.aten.sub.Tensor(clone_66, amax_8)
                clone_66 = amax_8 = None
                exp_8 = torch.ops.aten.exp.default(sub_31)
                sub_31 = None
                sum_9 = torch.ops.aten.sum.dim_IntList(exp_8, [2], True)
                div_8 = torch.ops.aten.div.Tensor(exp_8, sum_9)
                exp_8 = sum_9 = None
                permute_102 = torch.ops.aten.permute.default(div_8, [0, 1, 3, 2])
                div_8 = None
                expand_37 = torch.ops.aten.expand.default(permute_102, [8, 8, 8, 3137])
                permute_102 = None
                view_174 = torch.ops.aten.reshape.default(expand_37, [64, 8, 3137])
                expand_37 = None
                expand_38 = torch.ops.aten.expand.default(getitem_96, [8, 8, 3137, 8])
                clone_67 = torch.ops.aten.clone.default(
                    expand_38, memory_format=torch.contiguous_format
                )
                expand_38 = None
                view_175 = torch.ops.aten.reshape.default(clone_67, [64, 3137, 8])
                clone_67 = None
                bmm_16 = torch.ops.aten.bmm.default(view_174, view_175)
                view_174 = view_175 = None
                view_176 = torch.ops.aten.reshape.default(bmm_16, [8, 8, 8, 8])
                bmm_16 = None
                expand_39 = torch.ops.aten.expand.default(getitem_94, [8, 8, 3137, 8])
                clone_68 = torch.ops.aten.clone.default(
                    expand_39, memory_format=torch.contiguous_format
                )
                expand_39 = None
                view_177 = torch.ops.aten.reshape.default(clone_68, [64, 3137, 8])
                clone_68 = None
                expand_40 = torch.ops.aten.expand.default(view_176, [8, 8, 8, 8])
                view_176 = None
                view_178 = torch.ops.aten.reshape.default(expand_40, [64, 8, 8])
                expand_40 = None
                bmm_17 = torch.ops.aten.bmm.default(view_177, view_178)
                view_177 = view_178 = None
                view_179 = torch.ops.aten.reshape.default(bmm_17, [8, 8, 3137, 8])
                bmm_17 = None
                slice_116 = torch.ops.aten.slice.Tensor(
                    getitem_94, 2, 1, 9223372036854775807
                )
                getitem_94 = None
                slice_120 = torch.ops.aten.slice.Tensor(
                    getitem_96, 2, 1, 9223372036854775807
                )
                getitem_96 = None
                permute_103 = torch.ops.aten.permute.default(slice_120, [0, 1, 3, 2])
                slice_120 = None
                view_180 = torch.ops.aten.reshape.default(permute_103, [8, 64, 56, 56])
                permute_103 = None
                split_with_sizes_8 = torch.ops.aten.split_with_sizes.default(
                    view_180, [16, 24, 24], 1
                )
                view_180 = None
                getitem_97 = split_with_sizes_8[0]
                getitem_98 = split_with_sizes_8[1]
                getitem_99 = split_with_sizes_8[2]
                split_with_sizes_8 = None

                _convolution_pointwise_default_33 = self.conv3(getitem_97)
                _convolution_pointwise_default_32 = self.conv4(getitem_98)
                _convolution_pointwise_default_31 = self.conv5(getitem_99)

                cat_22 = torch.ops.aten.cat.default(
                    [
                        _convolution_pointwise_default_33,
                        _convolution_pointwise_default_32,
                        _convolution_pointwise_default_31,
                    ],
                    1,
                )
                _convolution_pointwise_default_33 = (
                    _convolution_pointwise_default_32
                ) = _convolution_pointwise_default_31 = None
                view_181 = torch.ops.aten.reshape.default(cat_22, [8, 8, 8, 3136])
                cat_22 = None
                permute_104 = torch.ops.aten.permute.default(view_181, [0, 1, 3, 2])
                view_181 = None

                mul_86 = torch.ops.aten.mul.Tensor(slice_116, permute_104)
                slice_116 = permute_104 = None
                constant_pad_nd_8 = torch.ops.aten.constant_pad_nd.default(
                    mul_86, [0, 0, 1, 0, 0, 0], 0.0
                )
                mul_86 = None
                mul_87 = torch.ops.aten.mul.Tensor(view_179, 0.3535533905932738)
                view_179 = None
                add_87 = torch.ops.aten.add.Tensor(mul_87, constant_pad_nd_8)
                mul_87 = constant_pad_nd_8 = None
                return add_87

        view_12 = torch.randn(batch_size, in_features, image_size, image_size)

        mod = M(bias=bias).eval()
        with verify(dtype) as (atol, rtol):
            self.common(
                mod,
                (view_12,),
                atol=atol,
                rtol=rtol,
            )
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 2)
        self.assertEqual(
            counters["inductor"]["cpp_epilogue_fusion_counter"], 2 if TEST_MKL else 1
        )

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @dtypes(torch.float32, torch.bfloat16)
    def test_qlinear_pointwise_int8_layout(self, dtype):
        class M(torch.nn.Module):
            def __init__(self, N, K):
                super().__init__()
                qw = torch.randint(-128, 127, (N, K), dtype=torch.int8)
                self.w_scales = torch.Tensor([0.8] * N)
                self.w_zps = torch.zeros(N).to(dtype=torch.int)
                self.qw_packed = torch.ops.onednn.qlinear_prepack(qw, None)
                self.b = torch.rand((N,), dtype=dtype)
                self.y_scale = 0.5
                self.y_zp = 0
                # set int8 output to check int8 layout path
                self.output_dtype = torch.int8
                self.post_op = "none"
                self.unary_post_op_args = ()
                self.post_op_algo = "none"

            def forward(self, qx, x_scale, x_zp):
                return torch.ops.onednn.qlinear_pointwise(
                    qx,
                    x_scale,
                    x_zp,
                    self.qw_packed,
                    self.w_scales,
                    self.w_zps,
                    self.b,
                    self.y_scale,
                    self.y_zp,
                    self.output_dtype,
                    self.post_op,
                    self.unary_post_op_args,
                    self.post_op_algo,
                )

        x = torch.rand((32, 64), dtype=dtype)
        x_scale, x_zp = torch.ops.quantized_decomposed.choose_qparams.tensor(
            x, -128, 127, torch.Tensor([torch.finfo(torch.float32).eps]), torch.int8
        )
        qx = torch.ops.quantized_decomposed.quantize_per_tensor.tensor(
            x, x_scale, x_zp, -128, 127, torch.int8
        )
        mod = M(64, 64).eval()
        counters.clear()
        self.common(mod, (qx, x_scale.item(), x_zp.item()))
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @dtypes(torch.bfloat16)
    @parametrize(
        "batch_size",
        (
            1,
            17,
            32,
        ),
    )
    @parametrize(
        "mid_dim",
        (
            1,
            8,
        ),
    )
    @parametrize("in_features", (128, 144, 1024))
    @parametrize("out_features", (64, 65, 1024))
    def test_int8_woq_mm(self, dtype, batch_size, mid_dim, in_features, out_features):
        def _convert_weight_to_int8pack(w):
            scale, zp = _calculate_dynamic_per_channel_qparams(
                w.to(torch.float), torch.int8
            )
            scale = torch.from_numpy(scale)
            zp = torch.from_numpy(zp)
            w_int8 = torch.ao.quantization.fx._decomposed.quantize_per_channel(
                input=w,
                scales=scale,
                zero_points=zp,
                axis=0,
                quant_min=-128,
                quant_max=127,
                dtype=torch.int8,
            )
            return w_int8, scale.to(torch.bfloat16)

        class M(torch.nn.Module):
            def __init__(self, w):
                super().__init__()
                self.linear_weight = torch.nn.Parameter(w, requires_grad=False)

            def forward(self, x, scale):
                return (
                    torch.nn.functional.linear(x, self.linear_weight.to(x.dtype))
                    * scale
                )

        counters.clear()
        # Currently, the corresponding torch.fx pattern only supports 3D x
        # Add 2D X case once the corresponding pattern-matcher pattern is added
        x = torch.rand((batch_size, mid_dim, in_features), dtype=dtype)
        w = torch.rand((out_features, in_features), dtype=dtype)
        w_int8pack, w_scales = _convert_weight_to_int8pack(w)
        mod = M(w_int8pack).eval()
        self.common(mod, (x, w_scales))
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)
        if batch_size * mid_dim >= 16:
            vec_amx = VecAMX()
            self._check_amx_counter(vec_amx)

    @inductor_config.patch({"freezing": True, "cpp.enable_concat_linear": True})
    @patches
    @torch.no_grad
    @dtypes(torch.bfloat16)
    @parametrize(
        "batch_size",
        (
            1,
            32,
        ),
    )
    @parametrize(
        "mid_dim",
        (
            1,
            8,
        ),
    )
    @parametrize("in_features", (128,))
    @parametrize("out_features", (64,))
    def test_int8_woq_mm_concat(
        self, dtype, batch_size, mid_dim, in_features, out_features
    ):
        def _convert_weight_to_int8pack(w):
            scale, zp = _calculate_dynamic_per_channel_qparams(
                w.to(torch.float), torch.int8
            )
            scale = torch.from_numpy(scale)
            zp = torch.from_numpy(zp)
            w_int8 = torch.ao.quantization.fx._decomposed.quantize_per_channel(
                input=w,
                scales=scale,
                zero_points=zp,
                axis=0,
                quant_min=-128,
                quant_max=127,
                dtype=torch.int8,
            )
            return w_int8, scale.to(torch.bfloat16)

        class M(torch.nn.Module):
            def __init__(self, w1, w2, w3):
                super().__init__()
                self.w1 = torch.nn.Parameter(w1, requires_grad=False)
                self.w2 = torch.nn.Parameter(w2, requires_grad=False)
                self.w3 = torch.nn.Parameter(w3, requires_grad=False)

            def forward(self, x, scale1, scale2, scale3):
                # Ref: _linear_fp_act_int8_weight_impl in torchao/dtypes/uintx/plain_layout.py
                y1 = (
                    torch.mm(x.reshape(-1, x.shape[-1]), self.w1.t().to(x.dtype))
                    * scale1
                )
                y2 = (
                    torch.mm(x.reshape(-1, x.shape[-1]), self.w2.t().to(x.dtype))
                    * scale2
                )
                y3 = (
                    torch.mm(x.reshape(-1, x.shape[-1]), self.w3.t().to(x.dtype))
                    * scale3
                )
                return (
                    y1.reshape(*x.shape[:-1], y1.shape[-1]),
                    y2.reshape(*x.shape[:-1], y2.shape[-1]),
                    y3.reshape(*x.shape[:-1], y3.shape[-1]),
                )

        counters.clear()
        # Currently, the corresponding torch.fx pattern only supports 3D x
        # Add 2D X case once the corresponding pattern-matcher pattern is added
        x = torch.rand((batch_size, mid_dim, in_features), dtype=dtype)
        w1 = torch.rand((out_features, in_features), dtype=dtype)
        w2 = torch.rand((out_features, in_features), dtype=dtype)
        w3 = torch.rand((out_features, in_features), dtype=dtype)
        w1_int8pack, w1_scales = _convert_weight_to_int8pack(w1)
        w2_int8pack, w2_scales = _convert_weight_to_int8pack(w2)
        w3_int8pack, w3_scales = _convert_weight_to_int8pack(w3)
        mod = M(w1_int8pack, w2_int8pack, w3_int8pack).eval()
        self.common(mod, (x, w1_scales, w2_scales, w3_scales))
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)
        if batch_size * mid_dim >= 16:
            vec_amx = VecAMX()
            self._check_amx_counter(vec_amx)

    @unittest.skipIf(
        not torch._C._cpu._is_amx_tile_supported(), "AMX ISA support is required"
    )
    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    # We set allow_ignore_mark_dynamic to True because Dynamo may end up specializing M dimension
    # despite it being marked as dynamic with mark_dynamic.
    @dynamo_config.patch({"allow_ignore_mark_dynamic": True})
    @parametrize("has_bias", [True, False])
    @parametrize("dtype", [torch.float, torch.bfloat16])
    @parametrize("per_channel_quant", [True, False])
    @parametrize("reshape_a", [True, False])
    @parametrize("expand_a_scale", [True, False])
    @parametrize("dynamic", [True, False])
    @parametrize("M", [1, 32])
    def test_da8w8_sym_act_sym_wgt_with_int_mm(
        self, has_bias, dtype, per_channel_quant, reshape_a, expand_a_scale, dynamic, M
    ):
        r"""
        This testcase check if we can match the int8_dynamic_activation_int8_weight int8 linear pattern from torchao,
        when activation is symmetrically quantized dynamically & weights are symmetrically quantized (statically)
        The pattern is:
            (no bias) _int_mm -> convert_element_type -> ([maybe_expand_a_scale] -> mul) -> mul
        or
            (with bias) pattern_no_bias -> add
        Expansion of the scale of activation is optional.
        The pattern depiction doesn't mean that convert_element_type output is fed into expand_a as input,
        but simply that activation scale may be applied after an expand operation on it.
        """
        if dtype == torch.bfloat16 and not torch.ops.mkldnn._is_mkldnn_bf16_supported():
            return
        in_feature = 48
        out_feature = 64
        q_min, q_max = -32, 31

        class Mod(torch.nn.Module):
            def __init__(self, dtype: torch.dtype, has_bias: bool):
                super().__init__()
                self.dtype = dtype
                self.has_bias = has_bias
                self.b = torch.randint(
                    q_min, q_max, [in_feature, out_feature], dtype=torch.int8
                )
                self.per_channel_quant = per_channel_quant
                a_scale_per_tensor = torch.rand([1], dtype=dtype) * 0.01 + 0.01
                a_scale_per_channel = torch.rand([M, 1], dtype=dtype) * 0.01 + 0.01
                self.a_scale = (
                    a_scale_per_channel if per_channel_quant else a_scale_per_tensor
                )
                self.b_scale = torch.rand([out_feature]) * 0.01 + 0.01
                self.b_scale = self.b_scale.to(dtype)
                self.bias = torch.rand([out_feature], dtype=dtype) if has_bias else None

            def forward(self, a):
                if reshape_a:
                    a_reshaped = a.reshape(-1, a.size(-1))
                else:
                    a_reshaped = a
                c = torch._int_mm(a_reshaped, self.b)
                c = c.to(self.dtype)
                if not expand_a_scale:
                    a_scale = self.a_scale
                else:
                    a_scale = self.a_scale.expand(c.shape)
                c = c * a_scale
                c = c * self.b_scale
                if self.has_bias:
                    c = c + self.bias
                return c

        mod = Mod(dtype, has_bias).eval()
        a = torch.randint(q_min, q_max, [M, in_feature], dtype=torch.int8)
        if dynamic:
            torch._dynamo.mark_dynamic(a, 0)
            torch._dynamo.mark_static(a, 1)
        self.common(
            mod,
            (a,),
            atol=1e-2 if dtype is torch.bfloat16 else None,
            rtol=1e-2 if dtype is torch.bfloat16 else None,
        )

        vec_amx = VecAMX()
        self._check_amx_counter(vec_amx)
        if torch._C._cpu._is_amx_tile_supported():
            # Only AMX ISA based micro-kernel is currently supported for da8w8
            self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @dtypes(torch.bfloat16)
    @parametrize("batch_size", (1,))
    @parametrize("in_features", (128, 256))
    @parametrize("out_features", (64, 128))
    @parametrize("group_size", (32, 64))
    def test_int4_woq_mm_avx512(
        self, dtype, batch_size, in_features, out_features, group_size
    ):
        class M(torch.nn.Module):
            def __init__(self, K, N, group_size):
                super().__init__()
                self.linear_weight = torch.randint(
                    0, 15, (N, K // 2), dtype=torch.uint8
                )
                self.qscale_and_zeros = torch.rand(K // group_size, N, 2, dtype=dtype)
                self.group_size = group_size

            def forward(self, x):
                x_shape = x.shape
                x = x.reshape(-1, x_shape[-1])
                y = torch._weight_int4pack_mm_for_cpu(
                    x, self.linear_weight, self.group_size, self.qscale_and_zeros
                )
                return y.reshape(*x_shape[:-1], out_features)

        counters.clear()
        seq_len = 4
        x = torch.rand((batch_size, seq_len, in_features), dtype=dtype)
        mod = M(in_features, out_features, group_size).eval()
        self.common(mod, (x,), reference_in_float=False)
        available_isa = torch._inductor.cpu_vec_isa.pick_vec_isa()
        avx512_available = "avx512" in str(available_isa)
        autotune_count = 1 if avx512_available else 0
        self.assertEqual(
            counters["inductor"]["select_algorithm_autotune"], autotune_count
        )

    @unittest.skipIf(
        not torch._C._cpu._is_amx_tile_supported(), "AMX ISA support is required"
    )
    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @dtypes(torch.bfloat16)
    @parametrize("batch_size", (64,))
    @parametrize("in_features", (14336,))
    @parametrize("out_features", (96,))
    @parametrize("group_size", (128,))
    @set_num_threads(1)
    def test_int4_woq_mm_amx_Nc_larger_than_one(
        self, dtype, batch_size, in_features, out_features, group_size
    ):
        """
        Note:
        `torch._weight_int4pack_mm_for_cpu` computes with float32, while the AMX-based GEMM
        template computes with bfloat16. So, the difference of computation results may be big.
        But we need `_weight_int4pack_mm_for_cpu` for its pattern.
        Therefore, we define module M1 for its pattern and parameters and define module M2 for
        the reference computation. M2's forward function gets the dequantized and unpacked weight
        in bfloat16 then computes GEMM with bfloat16.
        Besides, we need to skip the VERIFY patch and cannot use self.common for testing.
        """

        class M1(torch.nn.Module):
            def __init__(self, K, N, group_size):
                super().__init__()
                self.linear_weight = torch.randint(
                    0, 255, (N, K // 2), dtype=torch.uint8
                )
                self.qscale_and_zeros = torch.rand(K // group_size, N, 2, dtype=dtype)
                self.group_size = group_size

            def forward(self, x):
                x_shape = x.shape
                x = x.reshape(-1, x_shape[-1])
                y = torch._weight_int4pack_mm_for_cpu(
                    x, self.linear_weight, self.group_size, self.qscale_and_zeros
                )
                return y.reshape(*x_shape[:-1], out_features)

        class M2(torch.nn.Module):
            def __init__(self, mod: M1):
                super().__init__()
                self.mod = mod

            def forward(self, x):
                x_eye = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
                dq_w = self.mod(x_eye).T.contiguous()
                return torch.nn.functional.linear(x, dq_w)

        counters.clear()
        seq_len = 8
        x = torch.rand((batch_size, seq_len, in_features), dtype=dtype)
        mod = M1(in_features, out_features, group_size).eval()
        mod2 = M2(mod)
        # Skip VERIFY during torch.compile and don't use self.common. See explanation above.
        with patch.object(select_algorithm, "VERIFY", None):
            m = torch.compile(mod)
            y_ref = mod2(x)
            y = m(x)
            self.assertEqual(
                y,
                y_ref,
                atol=1e-2,
                rtol=1e-2,
            )
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @unittest.skipIf(
        not torch._C._cpu._is_amx_tile_supported(), "AMX ISA support is required"
    )
    @inductor_config.patch({"freezing": True})
    @inductor_config.patch({"cpp.use_small_dequant_buffer": True})
    @patches
    @torch.no_grad
    @dtypes(torch.bfloat16)
    @parametrize("batch_size", (16,))
    @parametrize("in_features", (14336,))
    @parametrize("out_features", (96,))
    @parametrize("group_size", (128,))
    @set_num_threads(1)
    def test_int4_woq_mm_with_small_buffer_config(
        self, dtype, batch_size, in_features, out_features, group_size
    ):
        class M1(torch.nn.Module):
            def __init__(self, K, N, group_size):
                super().__init__()
                self.linear_weight = torch.randint(
                    0, 255, (N, K // 2), dtype=torch.uint8
                )
                self.qscale_and_zeros = torch.rand(K // group_size, N, 2, dtype=dtype)
                self.group_size = group_size

            def forward(self, x):
                x_shape = x.shape
                x = x.reshape(-1, x_shape[-1])
                y = torch._weight_int4pack_mm_for_cpu(
                    x, self.linear_weight, self.group_size, self.qscale_and_zeros
                )
                return y.reshape(*x_shape[:-1], out_features)

        counters.clear()
        seq_len = 1
        x = torch.rand((batch_size, seq_len, in_features), dtype=dtype)
        mod = M1(in_features, out_features, group_size).eval()
        with patch.object(select_algorithm, "VERIFY", None):
            m = torch.compile(mod)
            _, code = run_and_get_cpp_code(m, x)
            kr = 32  # only kr=32 supported in woq int4 amx kernel
            _target_code_check = f"constexpr int64_t Kc_blocks = {group_size // kr};"
            torch._C.FileCheck().check(_target_code_check).run(code)

    @unittest.skipIf(
        not torch._C._cpu._is_amx_tile_supported(), "AMX ISA support is required"
    )
    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @dtypes(torch.bfloat16)
    @parametrize("batch_size", (1, 4, 6))
    @parametrize("in_features", (128, 1024))
    @parametrize("out_features", (128, 1024))
    @parametrize("group_size", (32, 64, 128))
    def test_int4_woq_mm_amx(
        self, dtype, batch_size, in_features, out_features, group_size
    ):
        """
        Note:
        `torch._weight_int4pack_mm_for_cpu` computes with float32, while the AMX-based GEMM
        template computes with bfloat16. So, the difference of computation results may be big.
        But we need `_weight_int4pack_mm_for_cpu` for its pattern.
        Therefore, we define module M1 for its pattern and parameters and define module M2 for
        the reference computation. M2's forward function gets the dequantized and unpacked weight
        in bfloat16 then computes GEMM with bfloat16.
        Besides, we need to skip the VERIFY patch and cannot use self.common for testing.
        """

        class M1(torch.nn.Module):
            def __init__(self, K, N, group_size):
                super().__init__()
                self.linear_weight = torch.randint(
                    0, 255, (N, K // 2), dtype=torch.uint8
                )
                self.qscale_and_zeros = torch.rand(K // group_size, N, 2, dtype=dtype)
                self.group_size = group_size

            def forward(self, x):
                x_shape = x.shape
                x = x.reshape(-1, x_shape[-1])
                y = torch._weight_int4pack_mm_for_cpu(
                    x, self.linear_weight, self.group_size, self.qscale_and_zeros
                )
                return y.reshape(*x_shape[:-1], out_features)

        class M2(torch.nn.Module):
            def __init__(self, mod: M1):
                super().__init__()
                self.mod = mod

            def forward(self, x):
                x_eye = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
                dq_w = self.mod(x_eye).T.contiguous()
                return torch.nn.functional.linear(x, dq_w)

        counters.clear()
        seq_len = 8
        x = torch.rand((batch_size, seq_len, in_features), dtype=dtype)
        mod = M1(in_features, out_features, group_size).eval()
        mod2 = M2(mod)
        # Skip VERIFY during torch.compile and don't use self.common. See explanation above.
        with patch.object(select_algorithm, "VERIFY", None):
            m = torch.compile(mod)
            y_ref = mod2(x)
            y = m(x)
            self.assertEqual(
                y,
                y_ref,
                atol=1e-2,
                rtol=1e-2,
            )
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @unittest.skipIf(
        not torch._C._cpu._is_amx_tile_supported(), "AMX ISA support is required"
    )
    @inductor_config.patch({"freezing": True})
    @inductor_config.patch({"cpp.enable_concat_linear": True})
    @patches
    @torch.no_grad
    @dtypes(torch.bfloat16)
    @parametrize("batch_size", (4,))
    @parametrize("in_features", (256,))
    @parametrize("out_features", ((512, 256, 256), (512, 512)))
    @parametrize("group_size", (32, 128))
    def test_int4_concat_woq_mm(
        self, dtype, batch_size, in_features, out_features, group_size
    ):
        class M1(torch.nn.Module):
            def __init__(self, K, out_features, group_size):
                super().__init__()
                self.linear_weight = [
                    torch.randint(0, 255, (N, K // 2), dtype=torch.uint8)
                    for N in out_features
                ]
                self.qscale_and_zeros = [
                    torch.rand(K // group_size, N, 2, dtype=dtype) for N in out_features
                ]
                self.group_size = group_size
                self.out_features = out_features

            def forward(self, x):
                x_shape = x.shape
                x = x.reshape(-1, x_shape[-1])
                y = [
                    torch._weight_int4pack_mm_for_cpu(
                        x,
                        self.linear_weight[idx],
                        self.group_size,
                        self.qscale_and_zeros[idx],
                    )
                    for idx in range(len(self.out_features))
                ]
                return [
                    y[idx].reshape(*x_shape[:-1], self.out_features[idx])
                    for idx in range(len(self.out_features))
                ]

        class M2(torch.nn.Module):
            def __init__(self, mod: M1):
                super().__init__()
                self.mod = mod

            def forward(self, x):
                x_eye = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
                dq_w_list = []
                for idx in range(len(self.mod.out_features)):
                    x_shape = x_eye.shape
                    dq_w = torch._weight_int4pack_mm_for_cpu(
                        x_eye,
                        self.mod.linear_weight[idx],
                        self.mod.group_size,
                        self.mod.qscale_and_zeros[idx],
                    )
                    dq_w_list.append(
                        dq_w.reshape(
                            *x_shape[:-1], self.mod.out_features[idx]
                        ).T.contiguous()
                    )

                return [torch.nn.functional.linear(x, dq_w) for dq_w in dq_w_list]

        counters.clear()
        seq_len = 8
        x = torch.rand((batch_size, seq_len, in_features), dtype=dtype)
        mod = M1(in_features, out_features, group_size).eval()
        mod2 = M2(mod)
        # Skip VERIFY during torch.compile and don't use self.common. See explanation above.
        with patch.object(select_algorithm, "VERIFY", None):
            y_ref = mod2(x)
            m = torch.compile(mod)
            y = m(x)
            self.assertEqual(
                y,
                y_ref,
                atol=1e-2,
                rtol=1e-2,
            )
            # Only do once tuning, since the wgt has been concat
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

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
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @inductor_config.patch({"cpp.gemm_cache_blocking": "2,2,2"})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @set_num_threads(1)
    @parametrize("batch_size", (512,))
    @parametrize("in_features", (1024,))
    @parametrize("out_features", (1024,))
    @parametrize("bias", (True, False))
    @dtypes(torch.float, torch.bfloat16, torch.half)
    def test_linear_cache_blocking(
        self, batch_size, in_features, out_features, bias, dtype
    ):
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
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @inductor_config.patch({"cpp.gemm_thread_factors": "4,2,7"})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @set_num_threads(56)
    @parametrize("batch_size", (1024,))
    @parametrize("in_features", (1024,))
    @parametrize("out_features", (1024,))
    @parametrize("bias", (True, False))
    @dtypes(torch.float, torch.bfloat16, torch.half)
    def test_linear_thread_factors(
        self, batch_size, in_features, out_features, bias, dtype
    ):
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
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)

    @inductor_config.patch({"freezing": False})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("batch_size", (16,))
    @parametrize("in_features", (128,))
    @parametrize("out_features", (64,))
    @parametrize("bias", (True,))
    @dtypes(
        torch.float,
    )
    def test_aoti_linear(self, batch_size, in_features, out_features, bias, dtype):
        try:
            try:
                from . import test_aot_inductor_utils
            except ImportError:
                import test_aot_inductor_utils
        except Exception:
            # skip this UT if import failed
            return

        class M(torch.nn.Module):
            def __init__(self, bias=bias) -> None:
                super().__init__()
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(in_features, out_features, bias=bias),
                    torch.nn.ReLU(),
                )

            def forward(self, x):
                return self.mlp(x)

        assert torch._inductor.config.freezing is False

        counters.clear()
        v = torch.randn(batch_size, in_features).to(dtype=dtype)
        mod = M(bias=bias).to(dtype=dtype).eval()
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        torch.manual_seed(0)
        with verify(dtype) as (atol, rtol), torch.no_grad():
            expected = mod(v)
            actual = test_aot_inductor_utils.AOTIRunnerUtil.run(
                mod,
                (v,),
            )
            self.assertEqual(actual, expected, atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @inductor_config.patch({"cpp.enable_grouped_gemm_template": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("batch_size", (16,))
    @parametrize("in_features", (52,))
    @parametrize("out_features", (32,))
    @parametrize("gemm_num", (2, 3))
    def test_grouped_linear_invalid(
        self,
        batch_size,
        in_features,
        out_features,
        gemm_num,
    ):
        class M(torch.nn.Module):
            def __init__(self, in_feature, out_feature, gemm_num):
                super().__init__()
                self.linears = [
                    torch.nn.Linear(in_feature, out_feature + gemm_idx, bias=False)
                    for gemm_idx in range(gemm_num)
                ]

            def forward(self, x):
                return [linear(x) for linear in self.linears]

        # each linear has different num of out features, thus invalid grouped gemm
        dtypes = []
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        for dtype in dtypes:
            torch._dynamo.reset()
            torch._inductor.metrics.reset()
            counters.clear()
            mod = M(in_features, out_features, gemm_num).eval()
            v = torch.randn(batch_size, in_features).to(dtype)
            with (
                verify(dtype) as (atol, rtol),
                torch.autocast(device_type="cpu", dtype=dtype),
                torch.no_grad(),
            ):
                self.common(mod, (v,), atol=atol, rtol=rtol)
            # gemm_num independent template instead of grouped gemm template
            self.assertEqual(
                counters["inductor"]["cpp_templated_kernel_counter"], gemm_num
            )
            self.assertEqual(counters["inductor"]["cpp_grouped_gemm_template"], 0)

    @inductor_config.patch({"freezing": True})
    @inductor_config.patch({"cpp.enable_grouped_gemm_template": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("batch_size", (16,))
    @parametrize("in_features", (52,))
    @parametrize("out_features", (32,))
    @parametrize("input_3d", (False, True))
    @parametrize("gemm_num", (2, 3))
    def test_grouped_linear(
        self,
        batch_size,
        in_features,
        out_features,
        input_3d,
        gemm_num,
    ):
        class M(torch.nn.Module):
            def __init__(self, in_feature, out_feature, gemm_num):
                super().__init__()
                self.linears = [
                    torch.nn.Linear(in_feature, out_feature, bias=False)
                    for _ in range(gemm_num)
                ]

            def forward(self, x):
                return [linear(x) for linear in self.linears]

        dtypes = []
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        for dtype in dtypes:
            if dtype == torch.float16 and input_3d:
                # reduce the number of tests
                continue
            torch._dynamo.reset()
            torch._inductor.metrics.reset()
            counters.clear()
            mod = M(in_features, out_features, gemm_num).eval()
            B = (2, batch_size) if input_3d else (batch_size,)
            v = torch.randn(*B, in_features).to(dtype)
            with (
                verify(dtype) as (atol, rtol),
                torch.autocast(device_type="cpu", dtype=dtype),
                torch.no_grad(),
            ):
                self.common(mod, (v,), atol=atol, rtol=rtol)
            self.assertEqual(counters["inductor"]["cpp_grouped_gemm_template"], 1)

    @inductor_config.patch({"freezing": True})
    @inductor_config.patch({"cpp.enable_grouped_gemm_template": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("batch_size", (16,))
    @parametrize("in_features", (52,))
    @parametrize("out_features", (32,))
    @parametrize("input_3d", (True, False))
    @parametrize(
        "bias",
        (
            [True, True],
            [True, False],
            [False, True],
            [False, False],
        ),
    )
    @parametrize(
        "epilogue",
        (
            ["none", "none"],
            ["relu", "none"],
            ["none", "relu"],
            ["relu", "relu"],
            ["silu", "mul"],
        ),
    )
    def test_grouped_linear_epilogue(
        self,
        batch_size,
        in_features,
        out_features,
        input_3d,
        bias,
        epilogue,
    ):
        class M(torch.nn.Module):
            def __init__(self, in_feature, out_feature, bias, epilogue):
                super().__init__()
                self.linear0 = torch.nn.Linear(in_feature, out_feature, bias=bias[0])
                self.linear1 = torch.nn.Linear(in_feature, out_feature, bias=bias[1])
                self.epilogue0 = epilogue[0]
                self.epilogue1 = epilogue[1]

            def forward(self, x):
                res0 = self.linear0(x)
                res1 = self.linear1(x)
                if self.epilogue0 == "silu" and self.epilogue1 == "mul":
                    return torch.nn.functional.silu(res0) * res1
                else:
                    if self.epilogue0 == "relu":
                        res0 = torch.nn.functional.relu(res0)
                    if self.epilogue1 == "relu":
                        res1 = torch.nn.functional.relu(res1)
                    return res0, res1

        dtypes = []
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        for dtype in dtypes:
            if input_3d and dtype == torch.float16:
                # Reduce the number of test cases
                continue
            torch._dynamo.reset()
            torch._inductor.metrics.reset()
            counters.clear()
            mod = M(in_features, out_features, bias, epilogue).eval()
            B = (2, batch_size) if input_3d else (batch_size,)
            v = torch.randn(*B, in_features).to(dtype)
            with (
                verify(dtype) as (atol, rtol),
                torch.autocast(device_type="cpu", dtype=dtype),
                torch.no_grad(),
            ):
                self.common(mod, (v,), atol=atol, rtol=rtol)
            self.assertEqual(counters["inductor"]["cpp_grouped_gemm_template"], 1)
            if any(e != "none" for e in epilogue):
                self.assertGreater(
                    counters["inductor"]["cpp_epilogue_fusion_counter"], 0
                )

    @inductor_config.patch({"freezing": False})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("batch_size", (16,))
    @parametrize("in_features", (128,))
    @parametrize("out_features", (64,))
    @dtypes(
        torch.float,
    )
    def test_aoti_linear_multi_view_operations(
        self, batch_size, in_features, out_features, dtype
    ):
        try:
            try:
                from . import test_aot_inductor_utils
            except ImportError:
                import test_aot_inductor_utils
        except Exception:
            # skip this UT if import failed
            return

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bias = torch.randn(out_features)
                self.weight = torch.randn(out_features // 2, 2, in_features)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                tmp = torch.addmm(
                    self.bias,
                    x,
                    self.weight.permute(2, 0, 1).view(in_features, out_features),
                )
                return self.relu(tmp)

        assert torch._inductor.config.freezing is False

        counters.clear()
        v = torch.randn(batch_size, in_features).to(dtype=dtype)
        mod = M().to(dtype=dtype).eval()
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        torch.manual_seed(0)
        with verify(dtype) as (atol, rtol), torch.no_grad():
            expected = mod(v)
            actual = test_aot_inductor_utils.AOTIRunnerUtil.run(
                mod,
                (v,),
            )
            self.assertEqual(actual, expected, atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @inductor_config.patch({"coordinate_descent_tuning": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    def test_cpp_coordinate_descent_tuning(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(512, 1024, bias=False)

            def forward(self, x):
                return self.linear(x)

        v = torch.randn(1, 512)
        mod = M().eval()
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        counters.clear()
        with verify(torch.bfloat16) as (atol, rtol), torch.autocast(device_type="cpu"):
            self.common(mod, (v,), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("batch_size", (2,))
    @parametrize("in_features", (128,))
    @parametrize("out_features", (64,))
    @parametrize("bias", (True, False))
    def test_linear_to_lowp_fp(self, batch_size, in_features, out_features, bias):
        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)

            def forward(self, x):
                return self.linear(x).to(torch.float16)

        counters.clear()
        dtype = torch.float32
        mod = M(bias=bias).to(dtype=dtype).eval()
        B = (batch_size,)
        v = torch.randn(*B, in_features).to(dtype=dtype)
        with verify(dtype) as (atol, rtol):
            self.common(mod, (v,), atol=atol, rtol=rtol)
            self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    def test_cpp_weight_prune(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(32, 128, bias=False)

            def forward(self, x):
                return self.linear(x)

        v = torch.randn(2, 32).to(torch.bfloat16)
        mod = M().eval().to(torch.bfloat16)
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        counters.clear()
        with verify(torch.bfloat16) as (atol, rtol):
            self.common(mod, (v,), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)
        self.assertEqual(counters["inductor"]["select_algorithm_weight_prune"], 1)

    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("bs", (1, 50))
    @parametrize("Mdim", (192,))
    @parametrize("Kdim", (196,))
    @parametrize("Ndim", (84, 385))
    @dtypes(torch.float, torch.bfloat16, torch.half)
    def test_bmm(self, dtype, bs, Mdim, Kdim, Ndim):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x @ y

        counters.clear()
        u = torch.randn(bs, Mdim, Kdim).to(dtype=dtype)
        v = torch.randn(bs, Kdim, Ndim).to(dtype=dtype)
        mod = M().to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (u, v), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)

    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("bs", (2,))
    @parametrize("Mdim", (16, 32))
    @parametrize("Kdim", (32,))
    @parametrize("Ndim", (3, 16, 32, 48, 128, 1024, 1025))
    @dtypes(torch.bfloat16, torch.half)
    def test_bmm_amx(self, dtype, bs, Mdim, Kdim, Ndim):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x @ y

        counters.clear()
        u = torch.randn(bs, Mdim, Kdim).to(dtype=dtype)
        v = torch.randn(bs, Kdim, Ndim).to(dtype=dtype)
        mod = M().to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (u, v), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)
        vec_amx = VecAMX()
        # Currently brgemm config is only added for half
        if dtype == torch.half:
            self._check_brgemm_counter(vec_amx)
        else:
            self._check_amx_counter(vec_amx)

    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("bs", (1,))
    @parametrize("Mdim", (192,))
    @parametrize("Kdim", (196,))
    @parametrize("Ndim", (84,))
    @dtypes(torch.float, torch.bfloat16, torch.half)
    def test_bmm_amp(self, dtype, bs, Mdim, Kdim, Ndim):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x @ y

        counters.clear()
        u = torch.randn(bs, Mdim, Kdim).to(dtype=dtype)
        v = torch.randn(bs, Kdim, Ndim).to(dtype=dtype)
        mod = M().to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol), torch.amp.autocast("cpu"):
            self.common(mod, (u, v), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("bs", (1,))
    @parametrize("Mdim", (192,))
    @parametrize("Kdim", (196,))
    @parametrize("Ndim", (64, 65))
    @dtypes(torch.float, torch.bfloat16, torch.half)
    def test_bmm_freezing(self, dtype, bs, Mdim, Kdim, Ndim):
        class M(torch.nn.Module):
            def __init__(self, w):
                super().__init__()
                self.w = torch.nn.Parameter(w, requires_grad=False)

            def forward(self, x):
                return x @ self.w

        counters.clear()
        u = torch.randn(bs, Mdim, Kdim).to(dtype=dtype)
        v = torch.randn(bs, Kdim, Ndim).to(dtype=dtype)
        mod = M(v).to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (u,), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)

    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("Ndim", (64, 61))
    @parametrize(
        "order",
        (
            ((0, 1, 2), (0, 2, 1)),  # First BMM in hf_Reformer
            ((0, 1, 2), (1, 2, 0)),  # First BMM in hf_DistilBert
            ((0, 1, 2), (1, 0, 2)),  # Second BMM in hf_DistilBert, hf_T5
            ((1, 0, 2), (0, 1, 2)),  # Third BMM in hf_Reformer
            ((1, 0, 2), (1, 2, 0)),  # First in hf_T5
        ),
    )
    @dtypes(torch.float, torch.bfloat16, torch.half)
    @unittest.skip(
        "Skip for now since it's flaky: https://github.com/pytorch/pytorch/actions/runs/19916391966/job/57096613514?pr=169151"
    )
    def test_bmm_2d_permute(self, Ndim, order, dtype):
        # TODO: Support bmm with transposed X
        bs = 12
        Mdim = 10
        Kdim = 62
        x_args = (bs, Mdim, Kdim)
        w_args = (bs, Kdim, Ndim)
        inverse_order = [torch.argsort(torch.tensor(o)).tolist() for o in order]

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, w):
                if order[0] != (0, 1, 2):
                    x_order = [x_args[i] for i in inverse_order[0]]
                    x = x.reshape(x_order[0], x_order[1] * x_order[2]).clone()
                    x = x.reshape(*x_order).permute(*order[0])
                if order[1] != (0, 1, 2):
                    w_order = [w_args[i] for i in inverse_order[1]]
                    w = w.reshape(w_order[0], w_order[1] * w_order[2]).clone()
                    w = w.reshape(*w_order).permute(*order[1])
                y = x @ w
                return y

        counters.clear()
        u = torch.randn(bs, Mdim, Kdim).to(dtype=dtype)
        v = torch.randn(bs, Kdim, Ndim).to(dtype=dtype)
        mod = M().to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (u, v), atol=atol, rtol=rtol)
        self.assertEqual(
            counters["inductor"]["cpp_templated_kernel_counter"],
            1 if order[0] == (0, 1, 2) else 0,
        )

    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("bs", (5,))
    @parametrize("Mdim", (64,))
    @parametrize("Kdim", (96,))
    @dtypes(torch.float, torch.float16, torch.bfloat16)
    def test_bmm_self_permute(self, bs, Mdim, Kdim, dtype):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x @ x.permute(0, 2, 1)

        counters.clear()
        u = torch.randn(bs, Mdim, Kdim).to(dtype=dtype)
        mod = M().to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (u,), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)

    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("bs", (5,))
    @parametrize("Mdim", (3, 64))  # Test small Mdim which uses reshaped weights
    @dtypes(torch.float)
    def test_bmm_self_square(self, bs, Mdim, dtype):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x @ x

        counters.clear()
        u = torch.randn(bs, Mdim, Mdim).to(dtype=dtype)
        mod = M().to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (u,), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)

    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("bs", (5,))
    @parametrize("Mdim", (16,))
    @parametrize("Kdim", (32,))
    @parametrize("Ndim", (64,))
    @dtypes(torch.float)
    def test_bmm_with_broadcasted_mat1(self, bs, Mdim, Kdim, Ndim, dtype):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, w):
                assert x.dim() == 2, f"Expected x to be 2D, got {x.dim()}D"
                x_expanded = x.unsqueeze(0).expand(bs, -1, -1)
                return x_expanded @ w

        counters.clear()
        u = torch.randn(Mdim, Kdim).to(dtype=dtype)
        v = torch.randn(bs, Kdim, Ndim).to(dtype=dtype)
        mod = M().to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (u, v), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)

    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("bs", (5,))
    @parametrize("Mdim", (384,))
    @parametrize("Kdim", (96,))
    @parametrize("Ndim", (64, 65))
    @parametrize(
        "epilogue",
        (
            "relu",
            "add",
            "sub",
            "mul",
            "div",
        ),
    )
    @dtypes(torch.float32, torch.bfloat16, torch.half)
    def test_bmm_with_pointwise(self, bs, Mdim, Kdim, Ndim, epilogue, dtype):
        class M(torch.nn.Module):
            def __init__(self, epilogue, other):
                super().__init__()
                self.epilogue = _get_epilogue(epilogue, other)

            def forward(self, x, w):
                return self.epilogue(x @ w)

        counters.clear()
        x = torch.randn(bs, Mdim, Kdim).to(dtype=dtype)
        w = torch.randn(bs, Kdim, Ndim).to(dtype=dtype)
        other = torch.randn(bs, Mdim, Ndim).to(dtype=dtype)
        mod = M(epilogue, other).to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (x, w), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)
        self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 1)

    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @dtypes(torch.float32, torch.bfloat16, torch.half)
    def test_bmm_with_fused_epilogues(self, dtype):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mul = torch.randn(8, 8, 3136, 8).as_strided(
                    (8, 8, 3136, 8), (200704, 8, 64, 1)
                )

            def forward(self, x, w):
                x = torch.ops.aten.reshape.default(x, [64, 3137, 8])
                w = torch.ops.aten.reshape.default(w, [64, 8, 8])
                bmm = torch.ops.aten.bmm.default(x, w)
                bmm = torch.ops.aten.reshape.default(bmm, [8, 8, 3137, 8])
                constant_pad_nd = torch.ops.aten.constant_pad_nd.default(
                    self.mul, [0, 0, 1, 0, 0, 0], 0.0
                )
                mul_2 = torch.ops.aten.mul.Tensor(bmm, 0.3535533905932738)
                add = torch.ops.aten.add.Tensor(mul_2, constant_pad_nd)
                return add

        counters.clear()
        x = torch.randn(8, 8, 3137, 8).to(dtype=dtype)
        w = torch.randn(8, 8, 8, 8).to(dtype=dtype)
        mod = M().to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (x, w), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)
        self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 1)

    @patches
    @torch.no_grad
    @parametrize("bs", (1, 50))
    @parametrize("Mdim", (192,))
    @parametrize("Kdim", (196,))
    @parametrize("Ndim", (84, 385))
    @dtypes(torch.float, torch.bfloat16, torch.half)
    def test_bmm_with_y_storage_offset(self, dtype, bs, Mdim, Kdim, Ndim):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                # y_with_offset: contiguous, but has non-zero storage offset
                y_with_offset = torch.empty(
                    (3, *y.shape), dtype=y.dtype, device=y.device
                )[2].copy_(y)
                return x @ y_with_offset

        counters.clear()
        u = torch.randn(bs, Mdim, Kdim).to(dtype=dtype)
        v = torch.randn(bs, Kdim, Ndim).to(dtype=dtype)
        mod = M().to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (u, v), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)

    @patches
    @torch.no_grad
    @dtypes(torch.float)
    def test_aoti_bmm_unique_identifiers(self, dtype):
        try:
            try:
                from . import test_aot_inductor_utils
            except ImportError:
                import test_aot_inductor_utils
        except Exception:
            # skip this UT if import failed
            return

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, w):
                y = x @ w
                return y @ w

        counters.clear()
        x = torch.randn(3, 64, 64).to(dtype=dtype)
        w = torch.randn(3, 64, 64).to(dtype=dtype)
        mod = M().to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol), torch.no_grad():
            expected = mod(x, w)
            actual = test_aot_inductor_utils.AOTIRunnerUtil.run(
                mod,
                (x, w),
            )
            self.assertEqual(actual, expected, atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 2)

    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @set_num_threads(1)  # avoid k_slicing to make the test deterministic
    @parametrize(
        "out_features1",
        (
            8,
            16,
            24,
            32,
            48,
        ),
    )
    @dtypes(torch.float)
    def test_local_and_global_accumulator(self, out_features1, dtype):
        batch_size = 256
        in_features = 64
        out_features = 129
        in_features1 = 128
        bias = True
        try:
            try:
                from . import test_aot_inductor_utils
            except ImportError:
                import test_aot_inductor_utils
        except Exception:
            # skip this UT if import failed
            return

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.linear = torch.nn.Linear(in_features, out_features, bias)
                self.linear1 = torch.nn.Linear(in_features1, out_features1, bias)

            def forward(self, x):
                y = self.linear(x)
                view = torch.ops.aten.view.default(y, [-1, in_features1])
                return self.linear1(view)

        counters.clear()
        x = torch.randn(batch_size, in_features).to(dtype=dtype)
        mod = M().to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol), torch.no_grad():
            expected = mod(
                x,
            )
            actual = test_aot_inductor_utils.AOTIRunnerUtil.run(
                mod,
                (x,),
            )
            self.assertEqual(actual, expected, atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 2)

    @patches
    @inductor_config.patch(freezing=True)
    @unittest.skipIf(not torch._C._has_mkldnn, "MKLDNN is not enabled")
    def test_bmm_flexible_layout(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, u, v):
                view_3 = torch.ops.aten.reshape.default(u, [-1, 512, 64])
                clone_1 = torch.ops.aten.clone.default(
                    v, memory_format=torch.contiguous_format
                )
                view_7 = torch.ops.aten.reshape.default(clone_1, [-1, 512, 64])
                permute_6 = torch.ops.aten.permute.default(view_7, [0, 2, 1])
                div = torch.ops.aten.div.Tensor(permute_6, 8.0)
                # view_3 is a ReinterpretView and div is a FlexibleLayout which will become FixedLayout
                bmm = torch.ops.aten.bmm.default(view_3, div)
                return bmm

        mod = M().eval()
        u = torch.randn(2, 24, 512, 64)
        v = torch.randn(48, 512, 64)
        with verify(u.dtype) as (atol, rtol):
            self.common(mod, (u, v))

    @unittest.skipIf(
        not torch._C._cpu._is_amx_tile_supported(), "AMX ISA support is required"
    )
    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @parametrize("batch_size", (1024,))
    @parametrize("in_features", (1024,))
    @parametrize("out_features", (2048,))
    @dtypes(torch.bfloat16)
    def test_linear_reuse_kernels(self, batch_size, in_features, out_features, dtype):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_x = torch.nn.Linear(in_features, out_features)
                self.linear_y = torch.nn.Linear(out_features, in_features)
                self.linear_z = torch.nn.Linear(in_features, out_features)

            def forward(self, x):
                out = self.linear_x(x)
                out = self.linear_y(out)
                out = self.linear_z(out)
                return out

        x = torch.randn(batch_size, in_features).to(dtype=dtype)
        mod = M().to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            ref_res = mod(x)
            m = torch.compile(mod)
            res, code = run_and_get_cpp_code(m, x)
            self.assertEqual(
                res,
                ref_res,
                atol=atol,
                rtol=rtol,
                equal_nan=True,
                exact_dtype=True,
            )
            # Check that only 2 kernels are in the generated code
            assert code.count("AMXState amx_state") == 2


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
    test_grouped_linear_dynamic_shapes = TestSelectAlgorithm.test_grouped_linear
    test_grouped_linear_epilogue_dynamic_shapes = (
        TestSelectAlgorithm.test_grouped_linear_epilogue
    )
    test_linear_k_slicing_dynamic_shapes = TestSelectAlgorithm.test_linear_k_slicing
    test_linear_cache_blocking_dynamic_shapes = (
        TestSelectAlgorithm.test_linear_cache_blocking
    )
    test_linear_thread_factors_dynamic_shapes = (
        TestSelectAlgorithm.test_linear_thread_factors
    )

    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("bs", (5,))
    @parametrize("Mdim", (384,))
    @parametrize("Kdim", (96,))
    @parametrize("Ndim", (64, 65))
    @dtypes(torch.float, torch.bfloat16, torch.half)
    def test_bmm_with_pointwise_dynamic_shapes(self, bs, Mdim, Kdim, Ndim, dtype):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.epilogue = torch.nn.ReLU()

            def forward(self, x, other):
                return self.epilogue(x @ other)

        counters.clear()
        u = torch.randn(bs, Mdim, Kdim).to(dtype=dtype)
        v = torch.randn(bs, Kdim, Ndim).to(dtype=dtype)
        torch._dynamo.mark_dynamic(u, 0)
        torch._dynamo.mark_dynamic(u, 1)
        torch._dynamo.mark_static(u, 2)
        torch._dynamo.mark_static(v, 2)
        mod = M().to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (u, v), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)
        self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 1)

    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("bs", (5,))
    @parametrize("Mdim", (384,))
    @parametrize("Kdim", (96,))
    @parametrize("Ndim", (64, 65))
    @dtypes(torch.float, torch.bfloat16, torch.half)
    def test_bmm_with_pointwise_with_reshape_dynamic_shapes(
        self, bs, Mdim, Kdim, Ndim, dtype
    ):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.epilogue = torch.nn.ReLU()

            def forward(self, x, other, noise):
                result = x.reshape(-1, Mdim, Kdim) @ other.reshape(-1, Kdim, Ndim)
                return self.epilogue(result) + noise

        counters.clear()

        u = torch.randn(bs, 8, Mdim, Kdim).to(dtype=dtype)
        v = torch.randn(bs, 8, Kdim, Ndim).to(dtype=dtype)
        noise = torch.randn(bs * 8, Mdim, Ndim).to(dtype=dtype)
        torch._dynamo.mark_dynamic(u, 0)
        torch._dynamo.mark_dynamic(u, 1)
        torch._dynamo.mark_static(u, 2)
        torch._dynamo.mark_static(u, 3)
        torch._dynamo.mark_static(v, 2)
        torch._dynamo.mark_static(v, 3)
        mod = M().to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (u, v, noise), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)
        self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 1)

    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @dtypes(torch.float, torch.bfloat16)
    def test_bmm_epilogue_dynamic_reshape(self, dtype):
        bs = 5

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.epilogue = torch.nn.ReLU()

            def forward(self, x, w, arg5_1):
                arg131_1 = x.shape[0]
                mul_91 = arg131_1 * 8
                view_422 = torch.ops.aten.reshape.default(x, [mul_91, 512, 64])
                view_423 = torch.ops.aten.reshape.default(w, [mul_91, 64, 512])
                bmm_36 = torch.ops.aten.bmm.default(view_422, view_423)
                view_424 = torch.ops.aten.reshape.default(
                    bmm_36, [arg131_1, 8, 512, 512]
                )
                abs_2 = torch.ones(512, 512, dtype=torch.int64)
                lt_562 = torch.ops.aten.lt.Scalar(abs_2, 8)
                add_5084 = torch.ones(512, 512, dtype=torch.int64)
                add_5085 = torch.ones(512, 512, dtype=torch.int64)
                full_default_1 = torch.ops.aten.full.default(
                    [512, 512], 15, dtype=torch.int64, layout=torch.strided
                )
                minimum_3 = torch.ops.aten.minimum.default(add_5085, full_default_1)
                where_2 = torch.ops.aten.where.self(lt_562, abs_2, minimum_3)
                add_5086 = torch.ops.aten.add.Tensor(add_5084, where_2)
                embedding_5 = torch.ops.aten.embedding.default(arg5_1, add_5086)
                permute_196 = torch.ops.aten.permute.default(embedding_5, [2, 0, 1])
                unsqueeze_21 = torch.ops.aten.unsqueeze.default(permute_196, 0)
                full_default = torch.ops.aten.full.default(
                    [arg131_1, 1, 1, 512],
                    -0.0,
                    dtype=torch.float32,
                    layout=torch.strided,
                )
                add_5087 = torch.ops.aten.add.Tensor(unsqueeze_21, full_default)
                add_5103 = torch.ops.aten.add.Tensor(view_424, add_5087)
                return add_5103

        counters.clear()
        u = torch.randn(bs, 8, 512, 64).to(dtype=dtype)
        v = torch.randn(bs, 8, 64, 512).to(dtype=dtype)
        arg5 = torch.randn(32, 8)
        torch._dynamo.mark_dynamic(u, 0)
        torch._dynamo.mark_static(u, 1)
        torch._dynamo.mark_static(u, 2)
        torch._dynamo.mark_static(u, 3)
        torch._dynamo.mark_static(v, 2)
        torch._dynamo.mark_static(v, 3)
        mod = M().to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (u, v, arg5), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)
        self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 1)

    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    def test_bmm_dynamic_bm_stride(self):
        bs = 8
        Mdim = 256
        Kdim = 64
        dtype = torch.float

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, weight):
                return x @ weight.permute(2, 0, 1)

        counters.clear()
        u = torch.randn(bs, Mdim, Kdim).to(dtype=dtype)
        v = torch.randn(Kdim, Mdim, bs).to(dtype=dtype)
        torch._dynamo.mark_dynamic(u, 0)
        torch._dynamo.mark_dynamic(u, 1)
        torch._dynamo.mark_static(u, 2)
        torch._dynamo.mark_static(v, 0)
        torch._dynamo.mark_static(v, 1)
        mod = M().to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (u, v), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["cpp_templated_kernel_counter"], 1)


instantiate_device_type_tests(TestSelectAlgorithm, globals(), only_for="cpu")
instantiate_device_type_tests(
    TestSelectAlgorithmDynamicShapes, globals(), only_for="cpu"
)


if __name__ == "__main__":
    from torch.testing._internal.inductor_utils import HAS_CPU

    if HAS_CPU and not (IS_MACOS or IS_WINDOWS):
        run_tests()
