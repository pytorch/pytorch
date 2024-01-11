# Owner(s): ["oncall: cpu inductor"]
import contextlib
import copy
import itertools
import unittest

import torch
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq

from torch._dynamo import config as dynamo_config
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch._export import capture_pre_autograd_graph
from torch._inductor import config
from torch._inductor.utils import run_and_get_code
from torch.ao.quantization.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch.nn import functional as F
from torch.testing._internal.common_quantization import (
    skipIfNoDynamoSupport,
    skipIfNoONEDNN,
    skipIfNoONEDNNBF16,
)
from torch.testing._internal.common_utils import IS_LINUX, skipIfRocm, TEST_MKL
from torch.testing._internal.inductor_utils import _check_has_dynamic_shape, HAS_CPU


# The dict value is match_nodes(computation_op+unary_op)

unary_list = {
    torch.nn.ReLU(): 2,
    torch.nn.Sigmoid(): 2,
    torch.nn.Tanh(): 2,
    torch.nn.Hardswish(): 6,
    torch.nn.LeakyReLU(0.1, inplace=False): 4,
    torch.nn.Hardtanh(min_val=-0.5, max_val=4, inplace=False): 3,
    torch.nn.Hardtanh(min_val=-0.5, max_val=float("inf"), inplace=False): 3,
    torch.nn.GELU(approximate="none"): 6,
    torch.nn.GELU(approximate="tanh"): 10,
    torch.nn.ReLU6(): 3,
    torch.nn.SiLU(): 3,
    torch.nn.Hardsigmoid(): 5,
}

non_decomposed_unary_list = [
    torch.nn.ReLU,
    torch.nn.Sigmoid,
    torch.nn.Tanh,
]

# The dict value is (match_count, match_nodes, inplace)
binary_list = {
    lambda x, y: torch.add(x, y): (1, 2, False),  # call_function
    lambda x, y: torch.add(y, x): (1, 2, False),  # call_function
    lambda x, y: x.add(y): (1, 2, False),  # call_method
    lambda x, y: x.add_(y): (1, 2, True),  # call_method
    lambda x, y: torch.sub(x, y): (1, 2, False),  # call_function
    lambda x, y: x.sub(y): (1, 2, False),  # call_method
    lambda x, y: x.sub_(y): (1, 2, True),  # call_method
}

quantization_add_fn_list = [
    lambda x, y: torch.add(x, y),
    lambda x, y: x.add(y),
]

quantization_inplace_add_fn_list = [
    lambda x, y: x.add_(y),
]


@config.patch({"freezing": True})
class TestPatternMatcherBase(TestCase):
    def _check_unary_is_decomposed(self, unary_fn):
        return not any(
            isinstance(unary_fn, fn)
            for fn in [torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh]
        )

    def _clone_inputs(self, inputs):
        def clone(x):
            if not isinstance(x, torch.Tensor):
                return x
            return x.clone()

        return tuple(clone(x) for x in inputs)

    def _generate_qdq_quantized_model(self, mod, inputs, is_qat=False):
        maybe_no_grad = contextlib.nullcontext() if is_qat else torch.no_grad()
        with maybe_no_grad:
            export_model = capture_pre_autograd_graph(
                mod,
                inputs,
            )
            quantizer = X86InductorQuantizer()
            quantizer.set_global(
                xiq.get_default_x86_inductor_quantization_config(is_qat=is_qat)
            )
            prepare_model = (
                prepare_qat_pt2e(export_model, quantizer)
                if is_qat
                else prepare_pt2e(export_model, quantizer)
            )
            prepare_model(*inputs)
            convert_model = convert_pt2e(prepare_model, fold_quantize=True)
            torch.ao.quantization.move_exported_model_to_eval(convert_model)
            return convert_model

    def _test_common(
        self,
        mod,
        inputs,
        matcher_count=None,
        matcher_nodes=None,
        atol=1e-5,
        rtol=1.3e-6,
        check_autocast=torch.float32,
        check_quantization=False,
        is_qat=False,
        matcher_check_fn=None,
        dtype=None,
    ):
        counters.clear()
        torch._dynamo.reset()
        assert matcher_check_fn is not None or (
            matcher_count is not None and matcher_nodes is not None
        )
        if (
            check_autocast == torch.bfloat16
            and torch.ops.mkldnn._is_mkldnn_bf16_supported()
        ):
            maybe_autocast = torch.cpu.amp.autocast(dtype=torch.bfloat16)
            atol, rtol = 1e-2, 1e-2
        elif (
            check_autocast == torch.float16
            and torch.ops.mkldnn._is_mkldnn_fp16_supported()
        ):
            maybe_autocast = torch.cpu.amp.autocast(dtype=torch.float16)
            atol, rtol = 1e-2, 1e-2
        else:
            assert check_autocast == torch.float32
            maybe_autocast = contextlib.nullcontext()

        if check_quantization:
            convert_model = self._generate_qdq_quantized_model(mod, inputs, is_qat)
            with torch.no_grad(), maybe_autocast:
                _ = torch.compile(convert_model)(*inputs)
                if matcher_count is not None:
                    self.assertEqual(
                        counters["inductor"]["pattern_matcher_count"], matcher_count
                    )
                if matcher_nodes is not None:
                    self.assertEqual(
                        counters["inductor"]["pattern_matcher_nodes"],
                        matcher_nodes,
                    )
                if matcher_check_fn is not None:
                    matcher_check_fn()
        else:
            with torch.no_grad(), maybe_autocast:
                clone_inputs = self._clone_inputs(inputs)
                expected = mod(*inputs)
                actual = torch.compile(mod)(*clone_inputs)
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                self.assertEqual(
                    counters["inductor"]["pattern_matcher_count"], matcher_count
                )
                self.assertEqual(
                    counters["inductor"]["pattern_matcher_nodes"],
                    matcher_nodes,
                )

    def _test_code_common(
        self,
        mod,
        inputs,
        include_ops,
        exclude_ops,
        atol=1e-5,
        rtol=1.3e-6,
        check_quantization=False,
        check_dynamic=None,
    ):
        with torch.no_grad():
            clone_inputs = self._clone_inputs(inputs)
            if check_quantization:
                mod = self._generate_qdq_quantized_model(mod, inputs)
            expected = mod(*inputs)
            actual, (source_code,) = run_and_get_code(
                torch.compile(mod, fullgraph=True, dynamic=check_dynamic),
                *clone_inputs,
            )
            for op in include_ops:
                self.assertIn(op, source_code)
            for op in exclude_ops:
                self.assertNotIn(op, source_code)
            if check_dynamic is not None:
                _check_has_dynamic_shape(self, source_code)
            if not check_quantization:
                # Skip due to reduce range setting for Quantization on preCI system.
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


class TestPatternMatcher(TestPatternMatcherBase):
    def test_conv2d_unary_cpu(self):
        class M(torch.nn.Module):
            def __init__(
                self,
                unary_fn,
                **kwargs,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                self.unary_fn = unary_fn

            def forward(self, x):
                x = self.conv(x)
                return self.unary_fn(x)

        dtypes = [
            torch.float,
        ]
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        options = itertools.product(
            unary_list.keys(),
            [torch.contiguous_format, torch.channels_last],
            dtypes,
        )

        for (
            unary_fn,
            memory_format,
            dtype,
        ) in options:
            x_shape = (1, 3, 56, 56)
            mod = M(unary_fn).to(memory_format=memory_format).eval()

            v = (
                torch.randn(x_shape, dtype=torch.float32)
                .add(1)
                .to(memory_format=memory_format)
            )
            # Add 1 for weight packing pass.
            match_nodes = unary_list[unary_fn] + 1
            if dtype in (
                torch.float16,
                torch.bfloat16,
            ) and self._check_unary_is_decomposed(unary_fn):
                # Has extra dtype conversion nodes for autocast.
                match_nodes += 2
            self._test_common(mod, (v,), 2, match_nodes, check_autocast=dtype)

    def test_linear_unary(self):
        class M(torch.nn.Module):
            def __init__(
                self,
                unary_fn,
                in_features,
                out_features,
                bias,
                **kwargs,
            ):
                super().__init__()
                self.linear = torch.nn.Linear(
                    in_features,
                    out_features,
                    bias,
                    **kwargs,
                )
                self.unary_fn = unary_fn

            def forward(self, x):
                x = self.linear(x)
                return self.unary_fn(x)

        dtypes = []
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        options = itertools.product(unary_list, [True, False], dtypes)
        for unary_fn, bias, dtype in options:
            mod = M(unary_fn, 10, 30, bias=bias).eval()
            # only fuse for linear when the dtype is bf16
            mod = mod
            v = torch.randn(2, 10)
            # packing pass + unary fusion.
            matcher_count = 2
            # Add 1 for weight packing pass.
            matcher_nodes = unary_list[unary_fn] + 1
            if self._check_unary_is_decomposed(unary_fn):
                # Has extra dtype conversion nodes for autocast.
                matcher_nodes += 2
            self._test_common(
                mod, (v,), matcher_count, matcher_nodes, check_autocast=dtype
            )

    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    def test_linear_fp32(self):
        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = torch.nn.Linear(10, 30, bias)

            def forward(self, x):
                return self.linear(x)

        for bias in [True, False]:
            mod = M(bias=bias).eval()
            v = torch.randn(2, 10)
            # packing pass.
            matcher_count = 1
            matcher_nodes = 1
            self._test_common(mod, (v,), matcher_count, matcher_nodes)

    def test_conv_transpose2d_unary(self):
        class M(torch.nn.Module):
            def __init__(
                self,
                unary_fn,
                **kwargs,
            ):
                super().__init__()
                self.conv_transpose2d = torch.nn.ConvTranspose2d(
                    3, 16, 3, stride=2, padding=1
                )
                self.unary_fn = unary_fn

            def forward(self, x):
                x = self.conv_transpose2d(x)
                return self.unary_fn(x)

        dtypes = [
            torch.float,
        ]
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)

        options = itertools.product(
            unary_list,
            [torch.contiguous_format, torch.channels_last],
            dtypes,
        )

        for unary_fn, memory_format, dtype in options:
            x_shape = (1, 3, 28, 28)
            mod = M(unary_fn).eval()

            v = torch.randn(x_shape, dtype=torch.float32).to(
                memory_format=memory_format
            )
            # Add 1 for weight packing pass.
            match_nodes = unary_list[unary_fn] + 1
            if dtype in (
                torch.float16,
                torch.bfloat16,
            ) and self._check_unary_is_decomposed(unary_fn):
                # Has extra dtype conversion nodes for autocast.
                match_nodes += 2
            self._test_common(mod, (v,), 2, match_nodes, check_autocast=dtype)

    def test_conv2d_binary(self):
        class M(torch.nn.Module):
            def __init__(
                self,
                binary_fn,
                has_relu,
                **kwargs,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                self.binary_fn = binary_fn
                self.has_relu = has_relu

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                if has_relu:
                    return self.binary_fn(x1, x2).relu()
                else:
                    return self.binary_fn(x1, x2)

        test_memory_format = [torch.contiguous_format, torch.channels_last]
        options = itertools.product(
            binary_list,
            [True, False],
            test_memory_format,
        )

        for (
            binary_fn,
            has_relu,
            memory_format,
        ) in options:
            x_shape = (1, 3, 56, 56)
            mod = M(binary_fn, has_relu).eval()
            v = (
                torch.randn(x_shape, dtype=torch.float32, requires_grad=True)
                .add(1)
                .to(memory_format=memory_format)
            )
            match_count = binary_list[binary_fn][0] + 2
            match_nodes = binary_list[binary_fn][1]
            if has_relu:
                match_nodes += 1
            self._test_common(mod, (v,), match_count, match_nodes + 2)

    def test_linear_binary(self):
        class M(torch.nn.Module):
            def __init__(self, binary_fn, in_channels, out_channels, bias, **kwargs):
                super().__init__()
                self.linear = torch.nn.Linear(
                    in_channels, out_channels, bias=bias, **kwargs
                )
                self.binary_fn = binary_fn

            def forward(self, x, y):
                x = self.linear(x)
                x = self.binary_fn(x, y.clone())
                return x

        dtypes = []
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        options = itertools.product(
            binary_list, [[2, 3, 10], [2, 10]], [True, False], dtypes
        )
        out_feature = 30
        for binary_fn, input_shape, bias, dtype in options:
            torch._dynamo.reset()
            # addmm(mm) + (linear+add)
            match_count = 2
            match_nodes = 3
            if len(input_shape) == 3:
                is_inplace = binary_list[binary_fn][2]
                # view + linear + view(joint_graph+freeze pass)
                match_count = match_count + 5 if is_inplace else match_count + 3
                match_nodes = match_nodes + 7 if is_inplace else match_nodes + 5
            mod = M(binary_fn, input_shape[-1], out_feature, bias).to(dtype).eval()
            v = torch.randn(input_shape).to(dtype)
            other = torch.randn(input_shape[:-1] + [out_feature]).to(dtype)
            mod_c = torch.compile(mod)
            out, code = run_and_get_code(mod_c, v, other)
            self.assertEqual(out, mod(v, other), rtol=1e-2, atol=1e-2)
            # TODO - assert fusions work code

    def test_multi_linear_share_same_input(self):
        # llama pattern.
        class M(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.w1 = torch.nn.Linear(16, 16, bias=False)
                self.w2 = torch.nn.Linear(16, 16, bias=False)

            def forward(self, x):
                return F.silu(self.w1(x)) * F.relu(self.w2(x))

        dtypes = []
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        for dtype in dtypes:
            mod = M().to(dtype).eval()
            v = torch.randn(2, 4, 16).to(dtype)
            # 1. view(match_count=4, match_nodes=4).
            # 2. mm to packed linear(match_count=2, match_nodes=2).
            # 3. view+linear+view to linear(match_count=2, match_nodes=6).
            # 4. linear+silu fusion(match_count=1, match_nodes=5)
            # 5. linear+relu fusion(match_count=1, match_nodes=2)

            match_count = 10
            match_nodes = 19
            self._test_common(mod, (v,), match_count, match_nodes, rtol=1e-2, atol=1e-2)

    def _qconv2d_cpu_test_helper(self, int8_mixed_bf16=False):
        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1)

            def forward(self, x):
                return self.conv2(self.conv(x))

        mod = M().eval()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)

        def matcher_check_fn():
            # 1. Dequant-Conv2D pattern matched in QConv2D weight prepack * 1
            #    int8_mixed_fp32: [convert_element_type_1, sub, mul_1, dequantize_per_channel, clone, convolution]
            #    int8_mixed_bf16: [convert_element_type_1, sub, mul_1, optional(convert_element_type_4),
            #     dequantize_per_channel, optional(convert_element_type_3), clone, convolution]
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_count"], 2
            )
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_nodes"],
                16 if int8_mixed_bf16 else 12,
            )

        self._test_common(
            mod,
            (v,),
            check_quantization=True,
            check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
            matcher_check_fn=matcher_check_fn,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_cpu(self):
        r"""
        This testcase will quantize a single Conv2d module.
        """
        self._qconv2d_cpu_test_helper()

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_int8_mixed_bf16(self):
        r"""
        This testcase will quantize a single Conv2d module with int8_mixed_bf16 quantization.
        """
        self._qconv2d_cpu_test_helper(int8_mixed_bf16=True)

    def _qconv2d_unary_cpu_test_helper(
        self,
        int8_mixed_bf16=False,
        unary_op=torch.nn.ReLU(),
    ):
        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1)
                self.unary_fn = copy.deepcopy(unary_op)
                self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1)
                self.unary_fn2 = copy.deepcopy(unary_op)

            def forward(self, x):
                tmp = self.unary_fn(self.conv(x))
                return self.unary_fn2(self.conv2(tmp))

        mod = M().eval()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)

        def matcher_check_fn():
            # 1. Dequant-Conv2D pattern matched in quantization weight prepack * 2
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_count"], 2
            )
            # 2. QConv2D Unary fusion in post-grad fusion pass * 2
            self.assertEqual(counters["inductor"]["qconv2d_unary_matcher_count"], 2)

        self._test_common(
            mod,
            (v,),
            check_quantization=True,
            check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
            matcher_check_fn=matcher_check_fn,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_relu_cpu(self):
        r"""
        This testcase will quantize Conv2d->ReLU pattern.
        """
        self._qconv2d_unary_cpu_test_helper()

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_relu_int8_mixed_bf16(self):
        r"""
        This testcase will quantize Conv2d->ReLU pattern with int8_mixed_bf16 quantization.
        """
        self._qconv2d_unary_cpu_test_helper(int8_mixed_bf16=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_relu6_cpu(self):
        r"""
        This testcase will quantize Conv2d->ReLU6 pattern.
        """
        self._qconv2d_unary_cpu_test_helper(unary_op=torch.nn.ReLU6())

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_hardtanh_cpu(self):
        r"""
        This testcase will quantize Conv2d->Hardtanh pattern.
        """
        self._qconv2d_unary_cpu_test_helper(unary_op=torch.nn.Hardtanh())

    def _qconv2d_add_cpu_test_helper(self, use_relu=False, int8_mixed_bf16=False):
        r"""
        This testcase will quantize a Conv2d->Add pattern as:
                 X
               /   \
        Conv1(X)   Conv2(X)
               \   /
                Add
                 |
           Optional(relu)
                 |
                 Y
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                add_fn,
                use_relu,
                **kwargs,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.add_fn = add_fn
                self.relu = torch.nn.ReLU()
                self.conv3 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.conv4 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.add_fn2 = add_fn
                self.relu2 = torch.nn.ReLU()
                self.use_relu = use_relu

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                tmp = self.add_fn(x1, x2)
                if self.use_relu:
                    tmp = self.relu(tmp)
                tmp1 = self.conv3(tmp)
                tmp2 = self.conv4(tmp)
                res = self.add_fn2(tmp1, tmp2)
                if self.use_relu:
                    res = self.relu2(res)
                return res

        for add_fn in quantization_add_fn_list + quantization_inplace_add_fn_list:
            mod = M(add_fn, use_relu).eval()
            v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(
                1
            )

            def matcher_check_fn():
                # 1. Dequant-Conv2D pattern matched in quantization weight prepack * 4
                self.assertEqual(
                    counters["inductor"]["qconv2d_weight_prepack_matcher_count"], 4
                )
                # 2. Qconv2d Binary Unary fusion in post-grad fusion pass * 2
                self.assertEqual(
                    counters["inductor"]["qconv2d_binary_matcher_count"], 2
                )

            self._test_common(
                mod,
                (v,),
                check_quantization=True,
                check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
                matcher_check_fn=matcher_check_fn,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_add_cpu(self):
        self._qconv2d_add_cpu_test_helper()

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_add_int8_mixed_bf16(self):
        self._qconv2d_add_cpu_test_helper(int8_mixed_bf16=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_add_relu_cpu(self):
        self._qconv2d_add_cpu_test_helper(use_relu=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_add_relu_int8_mixed_bf16(self):
        self._qconv2d_add_cpu_test_helper(use_relu=True, int8_mixed_bf16=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_add_broadcast_shapes_cpu(self):
        r"""
        This testcase will quantize Conv2d->add pattern using broadcast shape inputs.
        Conv2d->Add fusion will fail for the broadcast shape inputs case.
        """

        class M(torch.nn.Module):
            def __init__(self, use_bias):
                super().__init__()
                self.conv = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1)

            def forward(self, x1, x2):
                return torch.add(self.conv(x1), x2)

        bias_list = [True, False]
        for bias in bias_list:
            mod = M(bias).eval()
            x1 = torch.randn((2, 32, 9, 9))
            x2 = torch.randn((2, 32, 1, 1))

            def matcher_check_fn():
                # 1. Dequant-Conv2D pattern matched in quantization weight prepack * 1
                self.assertEqual(
                    counters["inductor"]["qconv2d_weight_prepack_matcher_count"], 1
                )
                # 2. Qconv2d Binary Unary fusion in post-grad fusion pass * 0
                self.assertEqual(
                    counters["inductor"]["qconv2d_binary_matcher_count"], 0
                )

            self._test_common(
                mod,
                (x1, x2),
                check_quantization=True,
                matcher_check_fn=matcher_check_fn,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_add_2(self):
        r"""
        This testcase prevents this pattern be matched as a conv_binary fusion by mistake.
                Conv(X)  3
                    \   /
                     Add
        We see this pattern in Mobilenet v3 large which add is decomposed from torch.nn.Hardswish or torch.nn.Hardsigmoid.
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                post_op,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.post_op = post_op

            def forward(self, x):
                return self.post_op(self.conv(x))

        for post_op in [
            torch.nn.Hardswish(inplace=True),
            torch.nn.Hardsigmoid(inplace=True),
        ]:
            mod = M(post_op).eval()
            v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(
                1
            )

            def matcher_check_fn():
                # Shouldn't hit conv binary fusion
                self.assertEqual(
                    counters["inductor"]["qconv2d_binary_matcher_count"], 0
                )

            self._test_common(
                mod,
                (v,),
                check_quantization=True,
                matcher_check_fn=matcher_check_fn,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_add_3(self):
        r"""
        This testcase will test below model:
             x
           /   \
        conv1  maxpool
          \    /   \
           add    conv2
            \     /
              cat
        Based on default recipe of x86InductorQuantizer, we will see this pattern after convert:
        qconv1    maxpool
         \           |
          \         q1
           \       /   \
            \     dq1  qconv2
             \   /
              add
               |
               q2
        Since q1 has 2 users and qconv2 is not ancestor node of qconv1, we shouldn't fuse:
                int8
                 /
        qconv1 dq1
           \   /
            add
             |
             q2
             |
            int8
        Instead we can match and fuse this pattern into qconv_binary:
        qconv1  fp32
            \   /
             add
              |
             fp32
        """

        class M(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
                self.maxpool = torch.nn.MaxPool2d(
                    kernel_size=3, stride=1, padding=0, dilation=1
                )

            def forward(self, x):
                tmp1 = self.conv1(x)
                tmp2 = self.maxpool(x)
                add = torch.add(tmp1, tmp2)
                tmp3 = self.conv2(tmp2)
                return torch.cat((add, tmp3), dim=1)

        mod = M().eval()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)

        def matcher_check_fn():
            self.assertEqual(counters["inductor"]["qconv2d_binary_matcher_count"], 1)
            # The matched qconv binary pattern should have 2 nodes [qconv, add]
            # instead of 11 which has dequant in binary input and output quant
            self.assertEqual(counters["inductor"]["qconv2d_binary_matcher_nodes"], 2)

        self._test_common(
            mod,
            (v,),
            check_quantization=True,
            matcher_check_fn=matcher_check_fn,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qat_qconv2d(self):
        r"""
        This testcase will quantize a single Conv2d module with qat flow.
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1)
                self.bn = torch.nn.BatchNorm2d(128)

            def forward(self, x):
                return self.bn(self.conv(x))

        mod = M().train()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=True).add(1)

        def matcher_check_fn():
            # 1. Dequant-conv pattern matched in quantization weight prepack * 1
            #    [convert_element_type_1, sub, mul_1, dequantize_per_channel, clone, convolution]
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_count"], 1
            )
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_nodes"], 6
            )
            # 2. QConv2D Unary fusion in post-grad fusion pass * 1
            #    [qconv2d_pointwise_default, div_1, round_2, add_1, clamp_min_1, clamp_max_1, convert_element_type_2]
            self.assertEqual(counters["inductor"]["qconv2d_unary_matcher_count"], 1)
            self.assertEqual(counters["inductor"]["qconv2d_unary_matcher_nodes"], 7)

        self._test_common(
            mod,
            (v,),
            check_quantization=True,
            is_qat=True,
            matcher_check_fn=matcher_check_fn,
        )

    def _qat_qconv2d_unary_cpu_test_helper(
        self,
        unary_op=torch.nn.ReLU(),
    ):
        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1)
                self.unary_fn = copy.deepcopy(unary_op)
                self.bn = torch.nn.BatchNorm2d(3)
                self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1)
                self.unary_fn2 = copy.deepcopy(unary_op)
                self.bn2 = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                tmp = self.unary_fn(self.bn(self.conv(x)))
                return self.unary_fn2(self.bn2(self.conv2(tmp)))

        mod = M()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=True).add(1)

        def matcher_check_fn():
            # 1. Dequant-conv pattern matched in quantization weight prepack * 1
            #    [convert_element_type_1, sub, mul_1, dequantize_per_channel, clone, convolution]
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_count"], 2
            )
            # 2. QConv2D Unary fusion in post-grad fusion pass * 1
            #    [qconv2d_pointwise_default, relu, div_1, round_2, add_1, clamp_min_1, clamp_max_1, convert_element_type_2]
            self.assertEqual(counters["inductor"]["qconv2d_unary_matcher_count"], 2)

        self._test_common(
            mod,
            (v,),
            check_quantization=True,
            is_qat=True,
            matcher_check_fn=matcher_check_fn,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qat_qconv2d_relu(self):
        r"""
        This testcase will quantize Conv2d->ReLU pattern with qat flow.
        """

        self._qat_qconv2d_unary_cpu_test_helper()

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qat_qconv2d_relu6(self):
        r"""
        This testcase will quantize Conv2d->ReLU6 pattern with qat flow.
        """

        self._qat_qconv2d_unary_cpu_test_helper(unary_op=torch.nn.ReLU6())

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qat_qconv2d_hardtanh(self):
        r"""
        This testcase will quantize Conv2d->Hardtanh pattern with qat flow.
        """

        self._qat_qconv2d_unary_cpu_test_helper(unary_op=torch.nn.Hardtanh())

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qat_qconv2d_add(self):
        r"""
        This testcase will quantize a Conv2d->Add pattern as:
                 X
               /   \
        Conv1(X)   Conv2(X)
               \   /
                Add
                 |
                 Y
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.bn1 = torch.nn.BatchNorm2d(6)
                self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.bn2 = torch.nn.BatchNorm2d(6)

            def forward(self, x):
                x1 = self.bn1(self.conv1(x))
                x2 = self.bn2(self.conv2(x))
                return x1 + x2

        mod = M().train()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=True).add(1)

        def matcher_check_fn():
            # 1. Dequant-conv pattern matched in quantization weight prepack * 2
            #    [convert_element_type_1, sub, mul_1, dequantize_per_channel, clone, convolution]
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_count"], 2
            )
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_nodes"], 12
            )
            # 2. Qconv2d Binary fusion in post-grad fusion pass * 1
            #    [qconv2d_pointwise_default_1, convert_element_type_5, sub_2, mul_5, add_3, mul_6, round_4, add_4,
            #     clamp_min_3, clamp_max_3, convert_element_type_6]
            self.assertEqual(counters["inductor"]["qconv2d_binary_matcher_count"], 1)
            self.assertEqual(counters["inductor"]["qconv2d_binary_matcher_nodes"], 11)

        self._test_common(
            mod,
            (v,),
            check_quantization=True,
            is_qat=True,
            matcher_check_fn=matcher_check_fn,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qat_qconv2d_add_relu(self):
        r"""
        This testcase will quantize a Conv2d->Add->ReLU pattern as:
                 X
               /   \
        Conv1(X)   Conv2(X)
               \   /
                Add
                 |
                ReLU
                 |
                 Y
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.bn1 = torch.nn.BatchNorm2d(6)
                self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.bn2 = torch.nn.BatchNorm2d(6)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x1 = self.bn1(self.conv1(x))
                x2 = self.bn2(self.conv2(x))
                return self.relu(x1 + x2)

        mod = M().train()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=True).add(1)

        def matcher_check_fn():
            # 1. Dequant-conv pattern matched in quantization weight prepack * 2
            #    [convert_element_type_1, sub, mul_1, dequantize_per_channel, clone, convolution]
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_count"], 2
            )
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_nodes"], 12
            )
            # 2. Qconv2d Binary fusion in post-grad fusion pass * 1
            #    [qconv2d_pointwise_default_1, convert_element_type_5, sub_2, mul_5, add_3, relu, mul_6, round_4, add_4,
            #     clamp_min_3, clamp_max_3, convert_element_type_6]
            self.assertEqual(counters["inductor"]["qconv2d_binary_matcher_count"], 1)
            self.assertEqual(counters["inductor"]["qconv2d_binary_matcher_nodes"], 12)

        self._test_common(
            mod,
            (v,),
            check_quantization=True,
            is_qat=True,
            matcher_check_fn=matcher_check_fn,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_dequant_promotion_cpu(self):
        r"""
        This testcase tests if dequant node before conv2d is promoted correctly:
                 X
                 |
              Conv1(X)
               /   \
        Conv2(X)   Conv3(X)
               \   /
                Add
                 |
                 Y
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.conv3 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)

            def forward(self, x):
                temp = self.conv1(x)
                temp = self.conv2(temp) + self.conv3(temp)
                return temp

        mod = M().eval()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)

        def matcher_check_fn():
            # 1. Dequant pattern matcher for dequant promotion * 1
            #    [convert_element_type_3, sub_1, mul_3]
            self.assertEqual(counters["inductor"]["dequant_promotion_matcher_count"], 1)
            self.assertEqual(counters["inductor"]["dequant_promotion_matcher_nodes"], 3)
            # 2. Dequant-conv pattern matched in quantization weight prepack * 3
            #    [convert_element_type_1, sub, mul_1, dequantize_per_channel, clone, convolution]
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_count"], 3
            )
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_nodes"], 18
            )
            # 3. Qconv2d Binary fusion in post-grad fusion pass * 1
            #    [qconv2d_pointwise_default_1, add_3]
            self.assertEqual(counters["inductor"]["qconv2d_binary_matcher_count"], 1)
            self.assertEqual(counters["inductor"]["qconv2d_binary_matcher_nodes"], 2)

        self._test_common(
            mod,
            (v,),
            check_quantization=True,
            matcher_check_fn=matcher_check_fn,
        )

    def _qlinear_cpu_test_helper(
        self,
        inputs,
        int8_mixed_bf16=False,
        do_permute=False,
        matcher_check_fn=None,
        bias=True,
    ):
        class M(torch.nn.Module):
            def __init__(self, use_bias, do_permute=False):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4, use_bias)
                self.linear2 = torch.nn.Linear(4, 4, use_bias)
                self.do_permute = do_permute

            def forward(self, x):
                if self.do_permute:
                    x = torch.reshape(torch.permute(x, (0, 2, 3, 1)), (2, 12, 4))
                return self.linear2(self.linear(x))

        mod = M(bias, do_permute=do_permute).eval()

        def _default_matcher_check_fn():
            self.assertEqual(
                counters["inductor"]["qlinear_weight_prepack_matcher_count"], 2
            )

        self._test_common(
            mod,
            inputs,
            check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
            check_quantization=True,
            matcher_check_fn=matcher_check_fn
            if matcher_check_fn is not None
            else _default_matcher_check_fn,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qlinear_cpu(self):
        r"""
        This testcase will quantize a single Linear Moduel.
        """
        for bias in [True, False]:
            self._qlinear_cpu_test_helper((torch.randn((2, 4)),), bias=bias)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qlinear_int8_mixed_bf16(self):
        r"""
        This testcase will quantize a single Linear Moduel with int8_mixed_bf16 quantization.
        """
        for bias in [True, False]:
            self._qlinear_cpu_test_helper(
                (torch.randn((2, 4)),), int8_mixed_bf16=True, bias=bias
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qlinear_input_dim_exceeds_2(self):
        r"""
        This testcase will quantize a single Linear Moduel.
        """
        for bias in [True, False]:
            self._qlinear_cpu_test_helper((torch.randn((2, 3, 4)),), bias=bias)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qlinear_int8_mixed_bf16_input_dim_exceeds_2(self):
        r"""
        This testcase will quantize a single Linear Moduel with int8_mixed_bf16 quantization.
        """
        for bias in [True, False]:
            self._qlinear_cpu_test_helper(
                (torch.randn((2, 3, 4)),), int8_mixed_bf16=True, bias=bias
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qlinear_input_dim_exceeds_2_and_not_contiguous(self):
        r"""
        This testcase will quantize a single Linear Module.
        * Input dim exceeds 2
        * Input not contiguous
        """
        for bias in [True, False]:

            def matcher_check_fn():
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_count"], 2
                )
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_nodes"],
                    17 if bias else 16,
                )

            self._qlinear_cpu_test_helper(
                (torch.randn((2, 4, 3, 4)),),
                do_permute=True,
                matcher_check_fn=matcher_check_fn,
                bias=bias,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qlinear_int8_mixed_bf16_input_dim_exceeds_2_and_not_contiguous(self):
        r"""
        This testcase will quantize a single Linear Module for int8_bf16.
        * Input dim exceeds 2
        * Input not contiguous
        """
        for bias in [True, False]:

            def matcher_check_fn():
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_count"], 2
                )
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_nodes"],
                    21 if bias else 20,
                )

            self._qlinear_cpu_test_helper(
                (torch.randn((2, 4, 3, 4)),),
                int8_mixed_bf16=True,
                do_permute=True,
                matcher_check_fn=matcher_check_fn,
                bias=bias,
            )

    def _qlinear_unary_cpu_test_helper(self, inputs, int8_mixed_bf16=False):
        class M(torch.nn.Module):
            def __init__(self, use_bias):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4, use_bias)
                self.unary_fn = torch.nn.ReLU()
                self.linear2 = torch.nn.Linear(4, 4, use_bias)
                self.unary_fn2 = torch.nn.ReLU()

            def forward(self, x):
                tmp = self.unary_fn(self.linear(x))
                return self.unary_fn2(self.linear2(tmp))

        bias_list = [True, False]
        for bias in bias_list:
            mod = M(bias).eval()

            def matcher_check_fn():
                # 1. dequant-linear pattern matched in quantization weight prepack
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_count"], 2
                )
                # 2. QLinear Unary fusion in post-grad fusion pass
                self.assertEqual(counters["inductor"]["qlinear_unary_matcher_count"], 2)

            self._test_common(
                mod,
                inputs,
                check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
                check_quantization=True,
                matcher_check_fn=matcher_check_fn,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qlinear_relu_cpu(self):
        r"""
        This testcase will quantize a Linear->ReLU pattern.
        """
        self._qlinear_unary_cpu_test_helper((torch.randn((2, 4)),))

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qlinear_relu_int8_mixed_bf16(self):
        r"""
        This testcase will quantize a Linear->ReLU pattern with int8_mixed_bf16 quantization.
        """
        self._qlinear_unary_cpu_test_helper(
            (torch.randn((2, 4)),), int8_mixed_bf16=True
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qlinear_relu_input_dim_exceeds_2(self):
        r"""
        This testcase will quantize a Linear->ReLU pattern.
        """
        self._qlinear_unary_cpu_test_helper((torch.randn((2, 3, 4)),))

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qlinear_relu_int8_mixed_bf16_input_dim_exceeds_2(self):
        r"""
        This testcase will quantize a Linear->ReLU pattern with int8_mixed_bf16 quantization.
        """
        self._qlinear_unary_cpu_test_helper(
            (torch.randn((2, 3, 4)),), int8_mixed_bf16=True
        )

    def _qlinear_dequant_promotion_cpu_test_helper(self, inputs, int8_mixed_bf16=False):
        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.linear1 = torch.nn.Linear(4, 4)
                self.linear2 = torch.nn.Linear(4, 4)
                self.linear3 = torch.nn.Linear(4, 4)

            def forward(self, x):
                temp = self.linear1(x)
                temp = self.linear2(temp) + self.linear3(temp)
                return temp

        mod = M().eval()

        def matcher_check_fn():
            # 1. Dequant pattern matcher for dequant promotion * 1
            self.assertEqual(counters["inductor"]["dequant_promotion_matcher_count"], 1)
            # 2. dequant-linear pattern matched in quantization weight prepack * 3
            self.assertEqual(
                counters["inductor"]["qlinear_weight_prepack_matcher_count"], 3
            )
            # 3. QLinear Unary fusion in post-grad fusion pass * 1
            self.assertEqual(counters["inductor"]["qlinear_unary_matcher_count"], 1)

        self._test_common(
            mod,
            inputs,
            check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
            check_quantization=True,
            matcher_check_fn=matcher_check_fn,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qlinear_dequant_promotion_cpu(self):
        r"""
        This testcase test if dequant node before linear is promoted correctly:
                  X
                  |
               Linear1(X)
                /   \
        Linear2(X)   Linear3(X)
                \   /
                 Add
                  |
                  Y
        """
        self._qlinear_dequant_promotion_cpu_test_helper((torch.randn((2, 4)),))

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qlinear_dequant_promotion_int8_mixed_bf16(self):
        r"""
        Test with int8_mixed_bf16 quantization.
        This testcase test if dequant node before linear is promoted correctly:
                  X
                  |
               Linear1(X)
                /   \
        Linear2(X)   Linear3(X)
                \   /
                 Add
                  |
                  Y
        """
        self._qlinear_dequant_promotion_cpu_test_helper(
            (torch.randn((2, 4)),), int8_mixed_bf16=True
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qlinear_dequant_promotion_cpu_input_dim_exceeds_2(self):
        r"""
        This testcase test if dequant node before linear is promoted correctly:
                  X
                  |
               Linear1(X)
                /   \
        Linear2(X)   Linear3(X)
                \   /
                 Add
                  |
                  Y
        """
        self._qlinear_dequant_promotion_cpu_test_helper((torch.randn((2, 3, 4)),))

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qlinear_dequant_promotion_int8_mixed_bf16_input_dim_exceeds_2(self):
        r"""
        Test with int8_mixed_bf16 quantization.
        This testcase test if dequant node before linear is promoted correctly:
                  X
                  |
               Linear1(X)
                /   \
        Linear2(X)   Linear3(X)
                \   /
                 Add
                  |
                  Y
        """
        self._qlinear_dequant_promotion_cpu_test_helper(
            (torch.randn((2, 3, 4)),), int8_mixed_bf16=True
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qlinear_mul_cpu(self):
        r"""
        This testcase will quantize a Linear->Mul pattern.
        """

        class M(torch.nn.Module):
            def __init__(self, use_bias):
                super().__init__()
                self.linear = torch.nn.Linear(4, 5, use_bias)

            def forward(self, x1, x2):
                return torch.mul(self.linear(x1), x2)

        bias_list = [True, False]
        for bias in bias_list:
            mod = M(bias).eval()
            x1 = torch.randn((2, 4))
            x2 = torch.randn((2, 5))

            self._test_common(
                mod,
                (x1, x2),
                2,
                8,
                check_quantization=True,
            )

    @skipIfNoDynamoSupport
    @skipIfRocm
    def test_qmaxpool2d(self):
        r"""
        This testcase will quantize Conv2d->ReLU->MaxPool2d pattern.
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                kwargs,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 64, 7, bias=True, stride=2, padding=3, dilation=1
                )
                self.relu = torch.nn.ReLU()
                self.maxpool = torch.nn.MaxPool2d(3, **kwargs)

            def forward(self, x):
                return self.maxpool(self.relu(self.conv(x)))

        kwargs_list = [
            {"stride": 2},
            {"stride": 2, "padding": 1},
            {"stride": 2, "padding": 1, "dilation": 1},
            {"stride": 2, "padding": 1, "dilation": 1, "ceil_mode": False},
        ]
        for kwargs in kwargs_list:
            mod = M(kwargs).eval()
            v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(
                1
            )
            # Totally 6 pattern_matcher_count, 31 pattern_matcher_nodes
            # 1. Pair of to_int8 and to_fp32 * 3, matched in pointless_convert pass at
            #    torch/_inductor/fx_passes/joint_graph.py: [convert_element_type, convert_element_type_1]
            # 2. Dequant-conv pattern matched in quantization weight prepack * 1
            #    [convert_element_type_1, sub, mul_1, dequantize_per_channel, clone, convolution]
            # 3. qconv2d_relu fusion in post-grad fusion pass * 1
            #    [qconv2d_pointwise_default, relu, mul_2, round_2, add_1, clamp_min_1, clamp_max_1, convert_element_type_2]
            # 4. qmaxpool2d * 1
            #    [convert_element_type_3, sub_1, mul_3, max_pool2d_with_indices, getitem, mul_4, round_3, add_2,
            #    clamp_min_2, clamp_max_2, convert_element_type_4]
            self._test_common(
                mod,
                (v,),
                6,
                31,
                check_quantization=True,
            )

    @skipIfNoDynamoSupport
    @skipIfRocm
    def test_qflatten(self):
        r"""
        This testcase will quantize Conv2d->AdaptiveAvgPool2d->flatten pattern.
        """

        class M(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 64, 7, bias=True, stride=2, padding=3, dilation=1
                )
                self.relu = torch.nn.ReLU()
                self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                return torch.flatten(
                    self.adaptive_avg_pool2d(self.relu(self.conv(x))), 1
                )

        mod = M().eval()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)

        def matcher_check_fn():
            self.assertEqual(counters["inductor"]["qreshape_matcher_count"], 1)

        self._test_common(
            mod,
            (v,),
            check_quantization=True,
            matcher_check_fn=matcher_check_fn,
        )

    @skipIfNoDynamoSupport
    @skipIfRocm
    def test_qcat(self):
        r"""
        This testcase will quantize cat based pattern:
                X
             /     \
        Conv1(X)  Pow(x)
            \        \
             \     Conv2(X)
              \    /
               Cat
                |
                Y
        """

        class M(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 64, 7, bias=True, stride=2, padding=3, dilation=1
                )
                self.conv2 = torch.nn.Conv2d(
                    3, 64, 7, bias=True, stride=2, padding=3, dilation=1
                )

            def forward(self, x):
                temp1 = self.conv(x)
                temp2 = self.conv2(torch.pow(x, 2))
                return torch.cat((temp1, temp2), 1)

        mod = M().eval()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)
        # Totally 10 pattern_matcher_count, 49 pattern_matcher_nodes
        # 1. Pair of to_int8 and to_fp32 * 5, matched in pointless_convert pass at
        #    torch/_inductor/fx_passes/joint_graph.py: [convert_element_type, convert_element_type_1]
        # 2. Dequant-conv pattern matched in quantization weight prepack * 2
        #    [convert_element_type_1, sub, mul_1, dequantize_per_channel, clone, convolution]
        # 3. qconv2d fusion in post-grad fusion pass * 2
        #    [qconv2d_pointwise_default, mul_2, round_2, add_1, clamp_min_1, clamp_max_1, convert_element_type_2]
        # 4. qcat * 1
        #    [convert_element_type_3, sub_1, mul_3, convert_element_type_7, sub_3, mul_7, cat, mul_8, round_5,
        #    add_4, clamp_min_4, clamp_max_4, convert_element_type_8]
        self._test_common(
            mod,
            (v,),
            10,
            49,
            check_quantization=True,
        )

    # https://github.com/pytorch/pytorch/issues/99841.
    def test_hardtanh_pattern_fallback(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_transpose = torch.nn.ConvTranspose2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x, min_value, max_value):
                conv_transpose_output = self.conv_transpose(x)
                clamp_min_output = torch.clamp_min(conv_transpose_output, min_value)
                clamp_max_output = torch.clamp_max(clamp_min_output, max_value)
                return clamp_max_output

        # check works for min_value > max_value.
        min_values = [3, torch.randn(1, 32, 28, 28)]
        max_values = [0, torch.randn(1, 32, 28, 28)]
        v = torch.randn(1, 3, 28, 28)
        for min_value, max_value in zip(min_values, max_values):
            mod = Model().eval()
            self._test_common(mod, (v, min_value, max_value), 2, 4)

    def test_leaky_relu_pattern_fallback(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x, negative_slope):
                conv_out = self.conv(x)
                return torch.where(conv_out > 0, conv_out, conv_out * negative_slope)

        negative_slopes = [0.1, torch.randn(1, 32, 28, 28)]
        with torch.no_grad():
            v = torch.randn(1, 3, 28, 28)
            for negative_slope in negative_slopes:
                mod = Model().eval()
                self._test_common(mod, (v, negative_slope), 2, 5)

    # https://github.com/pytorch/pytorch/issues/99838.
    def test_conv2d_add_scalar(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x):
                out_conv = self.conv(x)
                out = torch.add(out_conv, 1.0)
                return out

        with torch.no_grad():
            mod = Model().eval()
            v = torch.randn(1, 3, 28, 28)
            self._test_common(mod, (v,), 1, 1)

    def test_conv2d_binary_inplace_fusion_pass_cpu(
        self, include_ops=None, exclude_ops=None
    ):
        class Model_v1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x, other):
                conv_out = self.conv(x)
                return torch.add(conv_out, other.relu())

        class Model_v2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )
                self.conv2 = torch.nn.Conv2d(
                    in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
                )
                self.conv3 = torch.nn.Conv2d(
                    in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x, _):
                conv_out1 = self.conv(x)
                pow_out = torch.pow(conv_out1, 2)
                conv_out2 = self.conv2(pow_out)
                conv_out3 = self.conv3(conv_out2)
                res = torch.add(conv_out3, pow_out)
                return res

        input = torch.randn(1, 3, 28, 28).to(memory_format=torch.channels_last)
        others = [
            torch.randn(1, 32, 28, 28).to(memory_format=torch.channels_last),
            torch.randn(1, 32, 28, 28).to(memory_format=torch.channels_last),
        ]
        mod_v1 = Model_v1().to(memory_format=torch.channels_last).eval()
        mod_v2 = Model_v2().to(memory_format=torch.channels_last).eval()

        if include_ops is None:
            include_ops = ["mkldnn._convolution_pointwise_.binary"]
        if exclude_ops is None:
            exclude_ops = ["mkldnn._convolution_pointwise.binary"]

        for other, mod in zip(others, [mod_v1, mod_v2]):
            self._test_code_common(mod, (input, other), include_ops, exclude_ops)

    def test_conv2d_binary_inplace_fusion_failed_cpu(
        self, include_ops=None, exclude_ops=None
    ):
        # Written buffer is graph input, we can't fuse inplace.
        class Model_v1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x, other):
                conv_out = self.conv(x)
                return torch.add(conv_out, other)

        # Written buffer is an alias tensor, we can't fuse inplace.
        class Model_v2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x, other):
                conv_out = self.conv(x)
                return torch.add(conv_out, other[1:2, :, :, :]), other

        class Model_v3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )
                self.conv2 = torch.nn.Conv2d(
                    in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x, _):
                pow_out = torch.pow(self.conv(x), 2)
                other2 = F.relu(pow_out)
                conv_out2 = self.conv2(pow_out)
                res = torch.add(conv_out2, pow_out)
                res = res + other2
                return res

        input = torch.randn(1, 3, 28, 28).to(memory_format=torch.channels_last)
        others = [
            torch.randn(1, 32, 28, 28).to(memory_format=torch.channels_last),
            torch.randn(2, 32, 28, 28).to(memory_format=torch.channels_last),
            torch.randn(1, 32, 28, 28).to(memory_format=torch.channels_last),
        ]
        mod_v1 = Model_v1().to(memory_format=torch.channels_last).eval()
        mod_v2 = Model_v2().to(memory_format=torch.channels_last).eval()
        mod_v3 = Model_v3().to(memory_format=torch.channels_last).eval()

        if include_ops is None:
            include_ops = ["mkldnn._convolution_pointwise.binary"]
        if exclude_ops is None:
            exclude_ops = ["mkldnn._convolution_pointwise_.binary"]

        for other, mod in zip(others, [mod_v1, mod_v2, mod_v3]):
            self._test_code_common(mod, (input, other), include_ops, exclude_ops)

    def test_conv2d_binary_fusion_failed(self):
        # we don't support alpha !=1 case or other has different size with conv's output.
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x, other, alpha):
                conv_out = self.conv(x)
                return torch.add(conv_out, other, alpha=alpha)

        # https://github.com/pytorch/pytorch/issues/100802.
        # we can't do the fusion when add's inputs are same tensor.
        class Model2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x):
                out = self.conv(x)
                out = torch.add(out, out)
                return out

        # https://github.com/pytorch/pytorch/issues/101374.
        # we can't do the fusion when add's inputs are mixed dtype.
        class Model3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x):
                temp = self.conv(x)
                other = torch.ones(temp.shape, dtype=torch.double)
                out = torch.add(temp, other)
                return out

        input = torch.randn(1, 3, 28, 28).to(memory_format=torch.channels_last)
        others = [
            torch.randn(1, 32, 28, 28).to(memory_format=torch.channels_last),
            torch.randn(32, 28, 28),
        ]
        include_ops = ["mkldnn._convolution_pointwise"]
        exclude_ops = [
            "mkldnn._convolution_pointwise.binary",
            "mkldnn._convolution_pointwise_.binary",
        ]

        # case1
        for other, alpha in zip(others, [0.1, 1.0]):
            mod = Model().to(memory_format=torch.channels_last).eval()
            self._test_code_common(mod, (input, other, alpha), include_ops, exclude_ops)
        # case2:
        mod = Model2().to(memory_format=torch.channels_last).eval()
        self._test_code_common(mod, (input,), include_ops, exclude_ops)
        # case3:
        mod = Model3().to(memory_format=torch.channels_last).eval()
        self._test_code_common(mod, (input,), include_ops, exclude_ops)

    def test_reproduce_99842_issue(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

            def forward(self, input_tensor):
                x = self.conv(input_tensor)
                x = F.relu(x + torch.ones(x.size()))
                return x

        input = torch.randn(1, 3, 14, 14)
        mod = Model().eval()
        include_ops = ["mkldnn._convolution_pointwise_.binary"]
        self._test_code_common(mod, (input,), include_ops, [])

    def test_reproduce_113440_issue_1(self):
        class Mod(torch.nn.Module):
            def __init__(
                self,
                add_fn,
                **kwargs,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.add_fn = add_fn
                self.relu = torch.nn.ReLU(inplace=True)
                self.conv3 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.conv4 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.add_fn2 = add_fn
                self.relu2 = torch.nn.ReLU(inplace=True)
                self.use_relu = True

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                tmp = self.add_fn(x1, x2)
                if self.use_relu:
                    tmp = self.relu(tmp)
                tmp1 = self.conv3(tmp)
                tmp2 = self.conv4(tmp)
                res = self.add_fn2(tmp1, tmp2)
                if self.use_relu:
                    res = self.relu2(res)
                return res

        with torch.no_grad():
            example_inputs = (
                torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(
                    1
                ),
            )
            example_inputs[0].get_device()
            m = Mod(
                lambda x, y: x.add_(y),
            ).eval()
            om = torch.compile(m)
            om(*example_inputs)
            om(*example_inputs)

    def test_reproduce_113440_issue_2(self):
        class Mod(torch.nn.Module):
            def __init__(
                self,
                add_fn,
                **kwargs,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.add_fn = add_fn
                self.relu = torch.nn.ReLU(inplace=True)
                self.conv3 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.conv4 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.add_fn2 = add_fn
                self.relu2 = torch.nn.ReLU(inplace=True)

                self.conv5 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.conv6 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.conv7 = torch.nn.Conv2d(6, 6, kernel_size=1, stride=1)
                self.add_fn3 = add_fn
                self.relu3 = torch.nn.ReLU(inplace=True)

                self.use_relu = True

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                tmp = self.add_fn(x1, x2)
                if self.use_relu:
                    tmp = self.relu(tmp)

                tmp1 = self.conv3(tmp)
                res = self.relu2(tmp1)

                return res

        with torch.no_grad():
            example_inputs = (
                torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(
                    1
                ),
            )
            m = Mod(
                lambda x, y: x.add_(y),
            ).eval()
            om = torch.compile(m)
            om(*example_inputs)
            om(*example_inputs)


@dynamo_config.patch({"dynamic_shapes": True, "assume_static_by_default": False})
class TestDynamicPatternMatcher(TestPatternMatcherBase):
    test_conv2d_unary_dynamic_shapes = TestPatternMatcher.test_conv2d_unary_cpu
    test_conv2d_binary_dynamic_shapes = TestPatternMatcher.test_conv2d_binary
    test_linear_unary_dynamic_shapes = TestPatternMatcher.test_linear_unary

    def test_conv_transpose2d_dynamic_shapes(self):
        # We don't support conv_transpose2d for now.
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_transpose2d = torch.nn.ConvTranspose2d(
                    3, 16, 3, stride=2, padding=1
                )

            def forward(self, x):
                return self.conv_transpose2d(x)

        x_shape = (1, 3, 28, 28)
        mod = M().eval()
        v = torch.randn(x_shape, dtype=torch.float32)
        self._test_common(mod, (v,), 0, 0)

    def test_multi_linear_share_same_input_dynamic(self):
        # llama pattern.
        class M(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.w1 = torch.nn.Linear(16, 16, bias=False)
                self.w2 = torch.nn.Linear(16, 16, bias=False)

            def forward(self, x):
                return F.silu(self.w1(x)) * F.relu(self.w2(x))

        dtypes = []
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        for dtype in dtypes:
            mod = M().to(dtype).eval()
            v = torch.randn(2, 4, 16).to(dtype)
            # 1. view(match_count=4, match_nodes=4).
            # 2. mm to packed linear(match_count=2, match_nodes=2).
            # 3. view+linear+view to linear(match_count=2, match_nodes=6).

            match_count = 8
            match_nodes = 12
            self._test_common(mod, (v,), match_count, match_nodes, rtol=1e-2, atol=1e-2)

    def test_qconv2d_maxpool2d_linear_dynamic_cpu(self, include_ops=None):
        r"""
        This testcase will quantize a single Conv2d->Maxpool2d->Linear module
        with dynamic batch size input.
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 16, (2, 2), stride=(1, 1), padding=(1, 1)
                )
                self.relu = torch.nn.ReLU()
                self.maxpool2d = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.linear = torch.nn.Linear(16, 16)

            def forward(self, x):
                temp = self.relu(self.conv(x))
                temp = self.maxpool2d(temp)
                temp = self.avgpool(temp)
                temp = torch.flatten(temp, 1)
                return self.linear(temp)

        mod = M().eval()
        v = torch.randn((2, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)
        if include_ops is None:
            include_ops = [
                "torch.ops.onednn.qconv2d_pointwise",
                "torch.ops.quantized.max_pool2d",
                "torch.ops.onednn.qlinear_pointwise",
            ]
        exclude_ops = []
        self._test_code_common(
            mod,
            (v,),
            include_ops,
            exclude_ops,
            check_quantization=True,
            check_dynamic=True,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qat_bn_conv2d(self):
        r"""
        This testcase will quantize a single BN Conv2d module with qat flow.
        """

        class M(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.bn1 = torch.nn.BatchNorm2d(3)
                self.bn2 = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.conv(self.bn1(x))
                return self.bn2(x)

        mod = M().train()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=True).add(1)

        def matcher_check_fn():
            self.assertEqual(
                counters["inductor"]["qconv2d_weight_prepack_matcher_count"], 1
            )

        self._test_common(
            mod,
            (v,),
            check_quantization=True,
            is_qat=True,
            matcher_check_fn=matcher_check_fn,
        )


if __name__ == "__main__":
    if IS_LINUX and HAS_CPU and torch.backends.mkldnn.is_available():
        run_tests()
