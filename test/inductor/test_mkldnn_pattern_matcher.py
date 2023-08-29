# Owner(s): ["module: inductor"]
import contextlib
import copy
import itertools

import torch
import torch._dynamo as torchdynamo
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq

from torch._dynamo import config as dynamo_config
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.utils import run_and_get_code
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch.nn import functional as F
from torch.testing._internal.common_quantization import (
    skipIfNoDynamoSupport,
    skipIfNoONEDNN,
)
from torch.testing._internal.common_utils import IS_LINUX, skipIfRocm
from torch.testing._internal.inductor_utils import HAS_CPU

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

    def _test_common(
        self,
        mod,
        inputs,
        matcher_count,
        matcher_nodes,
        atol=1e-5,
        rtol=1.3e-6,
        check_autocast=False,
        check_quantization=False,
    ):
        counters.clear()
        maybe_autocast = contextlib.nullcontext()
        if check_autocast and torch.ops.mkldnn._is_mkldnn_bf16_supported():
            maybe_autocast = torch.cpu.amp.autocast()
            atol, rtol = 1e-2, 1e-2
        if check_quantization:
            with torch.no_grad():
                export_model, guards = torchdynamo.export(
                    mod,
                    *copy.deepcopy(inputs),
                    aten_graph=True,
                )
                quantizer = X86InductorQuantizer()
                quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
                prepare_model = prepare_pt2e(export_model, quantizer)
                prepare_model(*inputs)
                convert_model = convert_pt2e(prepare_model).eval()
                _ = torch.compile(convert_model)(*inputs)
                self.assertEqual(
                    counters["inductor"]["pattern_matcher_count"], matcher_count
                )
                self.assertEqual(
                    counters["inductor"]["pattern_matcher_nodes"],
                    matcher_nodes,
                )
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
        self, mod, inputs, include_ops, exclude_ops, atol=1e-5, rtol=1.3e-6
    ):
        with torch.no_grad():
            clone_inputs = self._clone_inputs(inputs)
            expected = mod(*inputs)
            actual, (source_code,) = run_and_get_code(
                torch.compile(mod, fullgraph=True), *clone_inputs
            )
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            for op in include_ops:
                self.assertIn(op, source_code)
            for op in exclude_ops:
                self.assertNotIn(op, source_code)


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

        options = itertools.product(
            unary_list.keys(),
            [torch.contiguous_format, torch.channels_last],
            [True, False] if torch.ops.mkldnn._is_mkldnn_bf16_supported() else [False],
        )

        for (
            unary_fn,
            memory_format,
            check_autocast,
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
            if check_autocast and self._check_unary_is_decomposed(unary_fn):
                # Has extra dtype conversion nodes for autocast.
                match_nodes += 2
            self._test_common(mod, (v,), 2, match_nodes, check_autocast=check_autocast)

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

        options = itertools.product(unary_list, [True, False])
        dtype = torch.bfloat16
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            for unary_fn, bias in options:
                mod = M(unary_fn, 10, 30, bias=bias).eval()
                # only fuse for linear when the dtype is bf16
                mod = mod.to(dtype)
                v = torch.randn(2, 10).to(dtype)
                # packing pass + unary fusion.
                matcher_count = 2
                # Add 1 for weight packing pass.
                matcher_nodes = unary_list[unary_fn] + 1
                if self._check_unary_is_decomposed(unary_fn):
                    # Has extra dtype conversion nodes for autocast.
                    matcher_nodes += 2
                self._test_common(
                    mod, (v,), matcher_count, matcher_nodes, check_autocast=True
                )

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

        options = itertools.product(
            unary_list,
            [torch.contiguous_format, torch.channels_last],
            [True, False] if torch.ops.mkldnn._is_mkldnn_bf16_supported() else [False],
        )

        for unary_fn, memory_format, check_autocast in options:
            x_shape = (1, 3, 28, 28)
            mod = M(unary_fn).eval()

            v = torch.randn(x_shape, dtype=torch.float32).to(
                memory_format=memory_format
            )
            # Add 1 for weight packing pass.
            match_nodes = unary_list[unary_fn] + 1
            if check_autocast and self._check_unary_is_decomposed(unary_fn):
                # Has extra dtype conversion nodes for autocast.
                match_nodes += 2
            self._test_common(mod, (v,), 2, match_nodes, check_autocast=check_autocast)

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

        options = itertools.product(binary_list, [[2, 3, 10], [2, 10]], [True, False])
        dtype = torch.bfloat16
        out_feature = 30
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            for binary_fn, input_shape, bias in options:
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

        mod = M().to(torch.bfloat16).eval()
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            v = torch.randn(2, 4, 16).to(torch.bfloat16)
            # 1. view(match_count=4, match_nodes=4).
            # 2. mm to packed linear(match_count=2, match_nodes=2).
            # 3. view+linear+view to linear(match_count=2, match_nodes=6).
            # 4. linear+silu fusion(match_count=1, match_nodes=5)
            # 5. linear+relu fusion(match_count=1, match_nodes=2)

            match_count = 10
            match_nodes = 19
            self._test_common(mod, (v,), match_count, match_nodes, rtol=1e-2, atol=1e-2)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_add(self):
        class M(torch.nn.Module):
            def __init__(
                self,
                add_fn,
                **kwargs,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.add_fn = add_fn

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                return self.add_fn(x1, x2)

        for add_fn in quantization_add_fn_list:
            mod = M(add_fn).eval()
            v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(
                1
            )
            # Totally 8 pattern_matcher_count, 39 pattern_matcher_nodes
            # 1. Pair of to_int8 and to_fp32 at conv input * 1, extra input of add * 1, and graph output * 1
            #    matched in pointless_convert pass at
            #    torch/_inductor/fx_passes/joint_graph.py: [convert_element_type, convert_element_type_1]
            # 2. Dequant pattern matcher for dequant promotion * 1
            #    [convert_element_type_3, sub_1, mul_3]
            # 3. Dequant-conv pattern matched in quantization weight prepack * 2
            #    [convert_element_type_1, sub, mul_1, dequantize_per_channel, clone, convolution]
            # 4. Quantization fusion in post-grad fusion pass * 1
            #    [qconv2d_pointwise_default, div_1, round_2, add_1, clamp_min_1, clamp_max_1, convert_element_type_2]
            # 5. Qconv2d_add * 1
            #    [qconv2d_pointwise_default_1, convert_element_type_5, sub_2, mul_5, add_3,
            #     mul_6, round_4, add_4, clamp_min_3, clamp_max_3, convert_element_type_6]
            self._test_common(
                mod,
                (v,),
                8,
                39,
                check_quantization=True,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_add_relu(self):
        class M(torch.nn.Module):
            def __init__(
                self,
                add_fn,
                **kwargs,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.add_fn = add_fn
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                return self.relu(self.add_fn(x1, x2))

        for add_fn in quantization_add_fn_list:
            mod = M(add_fn).eval()
            v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(
                1
            )
            # Totally 8 pattern_matcher_count, 40 pattern_matcher_nodes
            # 1. Pair of to_int8 and to_fp32 at conv input * 1, extra input of add * 1, and graph output * 1
            #    matched in pointless_convert pass at
            #    torch/_inductor/fx_passes/joint_graph.py: [convert_element_type, convert_element_type_1]
            # 2. Dequant pattern matcher for dequant promotion * 1
            #    [convert_element_type_3, sub_1, mul_3]
            # 3. Dequant-conv pattern matched in quantization weight prepack * 2
            #    [convert_element_type_1, sub, mul_1, dequantize_per_channel, clone, convolution]
            # 4. Quantization fusion in post-grad fusion pass * 1
            #    [qconv2d_pointwise_default, div_1, round_2, add_1, clamp_min_1, clamp_max_1, convert_element_type_2]
            # 5. Qconv2d_add * 1
            #    [qconv2d_pointwise_default_1, convert_element_type_5, sub_2, mul_5, add_3, relu,
            #     mul_6, round_4, add_4, clamp_min_3, clamp_max_3, convert_element_type_6]
            self._test_common(
                mod,
                (v,),
                8,
                40,
                check_quantization=True,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d(self):
        class M(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1)

            def forward(self, x):
                return self.conv(x)

        mod = M().eval()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)

        # Totally pattern_matcher_count 4,
        # pattern_matcher_nodes 17
        # 1. pair of to_int8 and to_fp32 at conv input matched in pointless_convert pass
        #    at torch/_inductor/fx_passes/joint_graph.py: [convert_element_type, convert_element_type_1]
        # 2. dequant-conv pattern matched in quantization weight prepack
        #    [convert_element_type_1, sub, mul_1, dequantize_per_channel, clone, convolution]
        # 3. pair of to_int8 and to_fp32 at conv output matched in pointless_convert pass
        #    at torch/_inductor/fx_passes/joint_graph.py: [convert_element_type_2, convert_element_type_3]
        # 4. Quantization fusion in post-grad fusion pass
        #    [qconv2d_pointwise_default, div_1, round_2, add_1,
        #     clamp_min_1, clamp_max_1, convert_element_type_2]
        self._test_common(
            mod,
            (v,),
            4,
            17,
            check_quantization=True,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_relu(self):
        class M(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1)
                self.unary_fn = torch.nn.ReLU()

            def forward(self, x):
                return self.unary_fn(self.conv(x))

        mod = M().eval()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)

        # Totally pattern_matcher_count 4,
        # pattern_matcher_nodes 18
        # 1. pair of to_int8 and to_fp32 at conv input matched in pointless_convert pass
        #    at torch/_inductor/fx_passes/joint_graph.py: [convert_element_type, convert_element_type_1]
        # 2. dequant-conv pattern matched in quantization weight prepack
        #    [convert_element_type_1, sub, mul_1, dequantize_per_channel, clone, convolution]
        # 3. pair of to_int8 and to_fp32 at conv output matched in pointless_convert pass
        #    at torch/_inductor/fx_passes/joint_graph.py: [convert_element_type_2, convert_element_type_3]
        # 4. Quantization fusion in post-grad fusion pass
        #    [qconv2d_pointwise_default, relu, div_1, round_2, add_1,
        #     clamp_min_1, clamp_max_1, convert_element_type_2]
        self._test_common(
            mod,
            (v,),
            4,
            18,
            check_quantization=True,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_dequant_promotion(self):
        class M(torch.nn.Module):
            def __init__(
                self,
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

        # Totally 11 pattern_matcher_count, 54 pattern_matcher_nodes for conv
        # 1. Pair of to_int8 and to_fp32 at conv input * 2, extra input of add * 1, and graph output * 1
        #    matched in pointless_convert pass at
        #    torch/_inductor/fx_passes/joint_graph.py: [convert_element_type, convert_element_type_1]
        # 2. Dequant pattern matcher for dequant promotion * 1
        #    [convert_element_type_3, sub_1, mul_3]
        # 3. Dequant-conv pattern matched in quantization weight prepack * 3
        #    [convert_element_type_1, sub, mul_1, dequantize_per_channel, clone, convolution]
        # 4. Quantization fusion in post-grad fusion pass * 2
        #    [qconv2d_pointwise_default, div_1, round_2, add_1, clamp_min_1, clamp_max_1, convert_element_type_2]
        # 5. Qconv2d_add * 1
        #    [qconv2d_pointwise_default_1, convert_element_type_5, sub_2, mul_5, add_3, mul_6, round_4, add_4,
        #     clamp_min_3, clamp_max_3, convert_element_type_6]
        self._test_common(
            mod,
            (v,),
            11,
            54,
            check_quantization=True,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear(self):
        class M(torch.nn.Module):
            def __init__(self, use_bias):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4, use_bias)

            def forward(self, x):
                return self.linear(x)

        bias_list = [True, False]
        for bias in bias_list:
            mod = M(bias).eval()
            v = torch.randn((2, 4))

            # Totally pattern_matcher_count 4, pattern_matcher_nodes 17
            # 1. pair of to_int8 and to_fp32 at input matched in pointless_convert pass
            #    at torch/_inductor/fx_passes/joint_graph.py: [convert_element_type, convert_element_type_1]
            # 2. dequant-linear pattern matched in quantization weight prepack
            #    [convert_element_type_1, sub, mul_1, dequantize_per_channel, t, addmm/mm]
            # 3. pair of to_int8 and to_fp32 at output matched in pointless_convert pass
            #    at torch/_inductor/fx_passes/joint_graph.py: [convert_element_type_2, convert_element_type_3]
            # 4. Quantization fusion in post-grad fusion pass
            #    [qlinear_pointwise_default, div_1, round_2, add_1,
            #     clamp_min_1, clamp_max_1, convert_element_type_2]
            self._test_common(
                mod,
                (v,),
                4,
                17,
                check_quantization=True,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear_relu(self):
        class M(torch.nn.Module):
            def __init__(self, use_bias):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4, use_bias)
                self.unary_fn = torch.nn.ReLU()

            def forward(self, x):
                return self.unary_fn(self.linear(x))

        bias_list = [True, False]
        for bias in bias_list:
            mod = M(bias).eval()
            v = torch.randn((2, 4))

            # Totally pattern_matcher_count 4, pattern_matcher_nodes 18
            # 1. pair of to_int8 and to_fp32 at input matched in pointless_convert pass
            #    at torch/_inductor/fx_passes/joint_graph.py: [convert_element_type, convert_element_type_1]
            # 2. dequant-linear pattern matched in quantization weight prepack
            #    [convert_element_type_1, sub, mul_1, dequantize_per_channel, t, addmm/mm]
            # 3. pair of to_int8 and to_fp32 at output matched in pointless_convert pass
            #    at torch/_inductor/fx_passes/joint_graph.py: [convert_element_type_2, convert_element_type_3]
            # 4. Quantization fusion in post-grad fusion pass
            #    [qlinear_pointwise_default, relu, div_1, round_2, add_1,
            #     clamp_min_1, clamp_max_1, convert_element_type_2]
            self._test_common(
                mod,
                (v,),
                4,
                18,
                check_quantization=True,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qlinear_dequant_promotion(self):
        class M(torch.nn.Module):
            def __init__(
                self,
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
        v = torch.rand((2, 4))

        # Totally 11 pattern_matcher_count, 50 pattern_matcher_nodes for linear
        # 1. Pair of to_int8 and to_fp32 at linear input * 2, extra input of add * 1, and graph output * 1
        #    matched in pointless_convert pass at
        #    torch/_inductor/fx_passes/joint_graph.py: [convert_element_type, convert_element_type_1]
        # 2. Dequant pattern matcher for dequant promotion * 1
        #    [convert_element_type_3, sub_1, mul_3]
        # 3. Dequant-linear pattern matched in quantization weight prepack * 3
        #    [convert_element_type_1, sub, mul_1, dequantize_per_channel, permute, addmm]
        # 4. Quantization fusion in post-grad fusion pass * 3
        #    [qlinear_pointwise_default, mul_6, round_4, add_3, clamp_min_3, clamp_max_3, convert_element_type_6]
        self._test_common(
            mod,
            (v,),
            11,
            50,
            check_quantization=True,
        )

    @skipIfNoDynamoSupport
    @skipIfRocm
    def test_qmaxpool2d(self):
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
    def test_qcat(self):
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
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x, other):
                conv_out = self.conv(x)
                return torch.add(conv_out, other.relu())

        inputs = [
            torch.randn(1, 3, 28, 28).to(memory_format=torch.channels_last),
            torch.randn(1, 32, 28, 28).to(memory_format=torch.channels_last),
        ]
        mod = Model().to(memory_format=torch.channels_last).eval()

        if include_ops is None:
            include_ops = ["mkldnn._convolution_pointwise_.binary"]
        if exclude_ops is None:
            exclude_ops = ["mkldnn._convolution_pointwise.binary"]

        self._test_code_common(mod, inputs, include_ops, exclude_ops)

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

        input = torch.randn(1, 3, 28, 28).to(memory_format=torch.channels_last)
        others = [
            torch.randn(1, 32, 28, 28).to(memory_format=torch.channels_last),
            torch.randn(2, 32, 28, 28).to(memory_format=torch.channels_last),
        ]
        mod_v1 = Model_v1().to(memory_format=torch.channels_last).eval()
        mod_v2 = Model_v2().to(memory_format=torch.channels_last).eval()

        if include_ops is None:
            include_ops = ["mkldnn._convolution_pointwise.binary"]
        if exclude_ops is None:
            exclude_ops = ["mkldnn._convolution_pointwise_.binary"]

        for other, mod in zip(others, [mod_v1, mod_v2]):
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

        mod = M().to(torch.bfloat16).eval()
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            v = torch.randn(2, 4, 16).to(torch.bfloat16)
            # 1. view(match_count=4, match_nodes=4).
            # 2. mm to packed linear(match_count=2, match_nodes=2).
            # 3. view+linear+view to linear(match_count=2, match_nodes=6).

            match_count = 8
            match_nodes = 12
            self._test_common(mod, (v,), match_count, match_nodes, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    if IS_LINUX and HAS_CPU and torch.backends.mkldnn.is_available():
        run_tests()
