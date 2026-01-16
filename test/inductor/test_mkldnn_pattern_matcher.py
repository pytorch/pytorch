# Owner(s): ["oncall: cpu inductor"]
import contextlib
import copy
import itertools
import unittest

import torch
from torch._dynamo.utils import counters
from torch._inductor import config, metrics
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import (
    is_mkldnn_bf16_supported,
    is_mkldnn_fp16_supported,
    run_and_get_code,
)
from torch.nn import functional as F
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_mkldnn import reduced_f32_on_and_off
from torch.testing._internal.common_quantization import (
    skipIfNoDynamoSupport,
    skipIfNoONEDNN,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_LINUX,
    skipIfXpu,
    TEST_ACL,
    TEST_MKL,
)
from torch.testing._internal.inductor_utils import (
    _check_has_dynamic_shape,
    clone_preserve_strides_offset,
    HAS_CPU,
)


# The dict value is match_nodes(computation_op+unary_op)

unary_list = {
    torch.nn.ReLU(): 2,
    torch.nn.Sigmoid(): 2,
    torch.nn.Tanh(): 2,
    torch.nn.Hardswish(): 6,
    torch.nn.LeakyReLU(0.1, inplace=False): 4,
    # Use floats for min/max, otherwise they can get converted to symints
    torch.nn.Hardtanh(min_val=-0.5, max_val=4.0, inplace=False): 3,
    torch.nn.Hardtanh(min_val=-0.5, max_val=float("inf"), inplace=False): 3,
    torch.nn.GELU(approximate="none"): 6,
    torch.nn.GELU(approximate="tanh"): 10,
    torch.nn.ReLU6(): 3,
    torch.nn.SiLU(): 5,
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


def cal_conv_generated_kernel_number(mod, input, dtype, dim=4, device="cpu"):
    # this function is to decide how many kernels are generated
    # while testing conv2d/3d/deconv2d
    # the assumption is:
    #   (1) There will be a to_dtype kernel for input for lp
    #   (2) inductor always use channel_last format, there will
    #       be a to_channel_last format for input
    #   (3) to_dtype and to_channel_last for input can be fused
    #   (4) inductor always get channel last format from mkldnn_conv_pointwise(binary),
    #       and force the output to have same stride with eager.
    #       So there will be a to_contiguous for output if eager output is contiguouse
    mod = copy.deepcopy(mod)
    mod = mod.to(device=device)
    input = input.clone()
    input = input.to(device)

    if dtype == torch.float32:
        maybe_autocast = contextlib.nullcontext()
    else:
        maybe_autocast = torch.amp.autocast(device_type=device, dtype=dtype)
    with torch.no_grad(), maybe_autocast:
        output = mod(input)
    input_kernel, output_kernel = 0, 0
    if (
        input.is_contiguous(memory_format=torch.contiguous_format)
        or dtype != torch.float32
        or (TEST_ACL and dim == 4)
    ):
        input_kernel = 1
    if output.is_contiguous(memory_format=torch.contiguous_format) or (
        TEST_ACL and (dtype == torch.bfloat16 or dtype == torch.half)
    ):
        output_kernel = 1

    return input_kernel + output_kernel


class TestPatternMatcherBase(TestCase):
    def setUp(self):
        super().setUp()
        self.ctx_stack = contextlib.ExitStack()
        self.ctx_stack.enter_context(config.patch({"freezing": True}))

    def tearDown(self):
        TestCase.tearDown(self)
        self.ctx_stack.close()

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
        matcher_check_fn,
        atol=1e-5,
        rtol=1.3e-6,
        check_autocast=torch.float32,
        check_quantization=False,
        is_qat=False,
        dtype=None,
        is_dynamic=False,
        quantizer=None,
        compile_options={},  # noqa: B006
        quantization_with_autocast=False,
    ):
        if not hasattr(self, "device"):
            has_xpu = any(
                isinstance(input, torch.Tensor) and input.device.type == "xpu"
                for input in inputs
            )
            device = "xpu" if has_xpu else "cpu"
        else:
            device = self.device

        mod = mod.to(device=device)
        if device != "cpu":
            inputs = tuple(
                clone_preserve_strides_offset(x, device=device) for x in inputs
            )
        counters.clear()
        torch._dynamo.reset()
        if check_autocast == torch.bfloat16 and is_mkldnn_bf16_supported(device):
            maybe_autocast = torch.amp.autocast(
                device_type=device, dtype=torch.bfloat16
            )
            atol, rtol = 5e-2, 5e-2
        elif check_autocast == torch.float16 and (is_mkldnn_fp16_supported(device)):
            maybe_autocast = torch.amp.autocast(device_type=device, dtype=torch.float16)
            atol, rtol = 5e-2, 5e-2
        else:
            assert check_autocast == torch.float32
            maybe_autocast = contextlib.nullcontext()
        if check_quantization:
            raise NotImplementedError("not supported, please migrate to torchao")
            """
            if quantization_with_autocast:
                with maybe_autocast:
                    convert_model = _generate_qdq_quantized_model(
                        mod, inputs, is_qat, is_dynamic, quantizer
                    )
            else:
                convert_model = _generate_qdq_quantized_model(
                    mod, inputs, is_qat, is_dynamic, quantizer
                )
            with torch.no_grad(), maybe_autocast:
                _ = torch.compile(convert_model)(*inputs)
                matcher_check_fn()
            """
        else:
            with torch.no_grad(), maybe_autocast:
                clone_inputs = self._clone_inputs(inputs)
                expected = mod(*inputs)
                actual = torch.compile(mod, **compile_options)(*clone_inputs)
                if self.precision != 0:
                    torch.testing.assert_close(
                        actual, expected, atol=self.precision, rtol=self.precision
                    )
                else:
                    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                matcher_check_fn()

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
        num_include_ops=None,
        quantizer=None,
    ):
        with torch.no_grad():
            clone_inputs = self._clone_inputs(inputs)
            if check_quantization:
                raise NotImplementedError("not supported, please migrate to torchao")
                """
                mod = _generate_qdq_quantized_model(mod, inputs, quantizer=quantizer)
                """
            expected = mod(*inputs)
            actual, (source_code,) = run_and_get_code(
                torch.compile(mod, fullgraph=True, dynamic=check_dynamic),
                *clone_inputs,
            )
            assert_keywords = ["assert_size_stride", "assert_alignment"]
            filtered_lines = [
                line
                for line in source_code.splitlines()
                if not any(assert_key in line for assert_key in assert_keywords)
            ]
            source_code = "\n".join(filtered_lines)

            for op in include_ops:
                self.assertIn(op, source_code)
            if num_include_ops is not None:
                assert len(include_ops) == len(num_include_ops)
                for i in range(len(include_ops)):
                    self.assertEqual(
                        source_code.count(include_ops[i]), num_include_ops[i]
                    )
            for op in exclude_ops:
                self.assertNotIn(op, source_code)
            if check_dynamic is not None:
                _check_has_dynamic_shape(self, source_code)
            if not check_quantization:
                # Skip due to reduce range setting for Quantization on preCI system.
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


class TestPatternMatcherGeneric(TestPatternMatcherBase):
    def _test_conv_unary_base(self, dim=4):
        assert dim == 4 or dim == 5

        class M(torch.nn.Module):
            def __init__(
                self,
                unary_fn,
                **kwargs,
            ):
                super().__init__()
                if dim == 4:
                    self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                else:
                    self.conv = torch.nn.Conv3d(3, 16, kernel_size=3, stride=1)
                self.unary_fn = unary_fn

            def forward(self, x):
                x = self.conv(x)
                return self.unary_fn(x)

        dtypes = [
            torch.float,
        ]
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)
        cl_format = torch.channels_last if dim == 4 else torch.channels_last_3d
        options = itertools.product(
            unary_list.keys(),
            [torch.contiguous_format, cl_format],
            dtypes,
        )

        for (
            unary_fn,
            memory_format,
            dtype,
        ) in options:
            if (
                dtype != torch.float32
                and torch.backends.mkldnn.matmul.fp32_precision == "tf32"
            ):
                continue
            metrics.reset()
            if dim == 4:
                x_shape = (1, 3, 56, 56)
            else:
                x_shape = (1, 3, 20, 56, 56)
            mod = M(unary_fn).to(memory_format=memory_format).eval()

            v = (
                torch.randn(x_shape, dtype=torch.float32)
                .add(1)
                .to(memory_format=memory_format)
            )

            def matcher_check_fn():
                match_nodes = unary_list[unary_fn]
                if dtype in (
                    torch.float16,
                    torch.bfloat16,
                ) and self._check_unary_is_decomposed(unary_fn):
                    # Has extra dtype conversion nodes for autocast.
                    match_nodes += 2
                self.assertEqual(
                    counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"],
                    0 if TEST_ACL else match_nodes,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_conv_weight_pack_matcher_count"], 1
                )

            self._test_common(mod, (v,), matcher_check_fn, check_autocast=dtype)
            generated_kernel_count = cal_conv_generated_kernel_number(
                mod, v, dtype, dim, self.device
            )
            self.assertEqual(metrics.generated_kernel_count, generated_kernel_count)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @reduced_f32_on_and_off()
    def test_conv2d_unary(self, device):
        self.device = device
        self._test_conv_unary_base(dim=4)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @reduced_f32_on_and_off()
    def test_conv3d_unary(self, device):
        self.device = device
        self._test_conv_unary_base(dim=5)

    def _test_conv_transpose_unary_base(self, dim=4):
        assert dim == 4 or dim == 5

        class M(torch.nn.Module):
            def __init__(
                self,
                unary_fn,
                **kwargs,
            ):
                super().__init__()
                if dim == 4:
                    self.conv_transpose = torch.nn.ConvTranspose2d(
                        3, 16, 3, stride=2, padding=1
                    )
                else:
                    self.conv_transpose = torch.nn.ConvTranspose3d(
                        3, 16, 3, stride=2, padding=1
                    )
                self.unary_fn = unary_fn

            def forward(self, x):
                x = self.conv_transpose(x)
                return self.unary_fn(x)

        dtypes = [
            torch.float,
        ]
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)

        cl_format = torch.channels_last if dim == 4 else torch.channels_last_3d
        options = itertools.product(
            unary_list,
            [torch.contiguous_format, cl_format],
            dtypes,
        )

        for unary_fn, memory_format, dtype in options:
            metrics.reset()
            if dim == 4:
                x_shape = (1, 3, 28, 28)
            else:
                x_shape = (1, 3, 17, 28, 28)
            mod = M(unary_fn).eval()

            v = torch.randn(x_shape, dtype=torch.float32).to(
                memory_format=memory_format
            )

            def matcher_check_fn():
                match_nodes = unary_list[unary_fn]
                if dtype in (
                    torch.float16,
                    torch.bfloat16,
                ) and self._check_unary_is_decomposed(unary_fn):
                    # Has extra dtype conversion nodes for autocast.
                    match_nodes += 2
                self.assertEqual(
                    counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"],
                    0 if TEST_ACL else match_nodes,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_conv_weight_pack_matcher_count"], 1
                )

            self._test_common(mod, (v,), matcher_check_fn, check_autocast=dtype)
            generated_kernel_count = cal_conv_generated_kernel_number(
                mod, v, dtype, dim, self.device
            )
            self.assertEqual(metrics.generated_kernel_count, generated_kernel_count)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfXpu(
        msg="The operator 'mkldnn::_convolution_transpose_pointwise' is not currently implemented for the XPU device."
    )
    @reduced_f32_on_and_off()
    def test_conv_transpose2d_unary(self, device):
        self.device = device
        self._test_conv_transpose_unary_base(dim=4)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfXpu(
        msg="The operator 'mkldnn::_convolution_transpose_pointwise' is not currently implemented for the XPU device."
    )
    @reduced_f32_on_and_off()
    def test_conv_transpose3d_unary(self, device):
        self.device = device
        self._test_conv_transpose_unary_base(dim=5)

    def _test_conv_binary_base(self, dim=4):
        assert dim == 4 or dim == 5

        class M(torch.nn.Module):
            def __init__(
                self,
                binary_fn,
                has_relu,
                **kwargs,
            ):
                super().__init__()
                if dim == 4:
                    self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                    self.conv2 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                else:
                    self.conv1 = torch.nn.Conv3d(3, 16, kernel_size=3, stride=1)
                    self.conv2 = torch.nn.Conv3d(3, 16, kernel_size=3, stride=1)
                self.binary_fn = binary_fn
                self.has_relu = has_relu

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                if has_relu:
                    return self.binary_fn(x1, x2).relu()
                else:
                    return self.binary_fn(x1, x2)

        dtypes = [
            torch.float,
        ]
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)
        cl_format = torch.channels_last if dim == 4 else torch.channels_last_3d
        test_memory_format = [torch.contiguous_format, cl_format]
        options = itertools.product(
            binary_list,
            [True, False],
            test_memory_format,
            dtypes,
        )

        for (
            binary_fn,
            has_relu,
            memory_format,
            dtype,
        ) in options:
            if (
                dtype != torch.float32
                and torch.backends.mkldnn.matmul.fp32_precision == "tf32"
            ):
                continue
            metrics.reset()
            if dim == 4:
                x_shape = (1, 3, 56, 56)
            else:
                x_shape = (1, 3, 20, 56, 56)
            mod = M(binary_fn, has_relu).eval()
            v = (
                torch.randn(x_shape, dtype=torch.float32, requires_grad=True)
                .add(1)
                .to(memory_format=memory_format)
            )

            def matcher_check_fn():
                match_nodes = binary_list[binary_fn][1]
                if has_relu:
                    match_nodes += 1
                self.assertEqual(
                    counters["inductor"][
                        "mkldnn_conv_binary_unary_fusion_matcher_nodes"
                    ],
                    0 if TEST_ACL else match_nodes,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_conv_weight_pack_matcher_count"], 2
                )

            self._test_common(mod, (v,), matcher_check_fn, check_autocast=dtype)
            generated_kernel_count = cal_conv_generated_kernel_number(
                mod, v, dtype, dim, self.device
            )
            self.assertEqual(metrics.generated_kernel_count, generated_kernel_count)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @reduced_f32_on_and_off(0.02)
    def test_conv2d_binary(self, device):
        self.device = device
        self._test_conv_binary_base(dim=4)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @reduced_f32_on_and_off(0.02)
    def test_conv3d_binary(self, device):
        self.device = device
        self._test_conv_binary_base(dim=5)

    def _test_conv_binary_broadcast_shapes_base(self, dim=4):
        assert dim == 4 or dim == 5
        torch.manual_seed(12345)

        class M(torch.nn.Module):
            def __init__(
                self,
                binary_fn,
                has_relu,
                **kwargs,
            ):
                super().__init__()
                if dim == 4:
                    self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                else:
                    self.conv = torch.nn.Conv3d(3, 16, kernel_size=3, stride=1)
                self.binary_fn = binary_fn
                self.has_relu = has_relu

            def forward(self, x, x2):
                x1 = self.conv(x)
                if has_relu:
                    return self.binary_fn(x1, x2).relu()
                else:
                    return self.binary_fn(x1, x2)

        dtypes = [
            torch.float,
        ]
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)
        cl_format = torch.channels_last if dim == 4 else torch.channels_last_3d
        test_memory_format = [torch.contiguous_format, cl_format]
        if dim == 4:
            input_shapes = [
                [2, 3, 56, 56],
            ]
            other_shapes = [[2, 16, 1, 1], [1, 16, 1, 1], [1, 1, 1, 1]]
        else:
            input_shapes = [
                [2, 3, 20, 56, 56],
            ]
            other_shapes = [[2, 16, 1, 1, 1], [1, 16, 1, 1, 1], [1, 1, 1, 1, 1]]
        options = itertools.product(
            binary_list,
            input_shapes,
            other_shapes,
            [True, False],
            test_memory_format,
            dtypes,
        )

        for (
            binary_fn,
            x_shape,
            other_shape,
            has_relu,
            memory_format,
            dtype,
        ) in options:
            metrics.reset()
            mod = M(binary_fn, has_relu).eval()
            x = (
                torch.randn(x_shape, dtype=torch.float32, requires_grad=True)
                .add(1)
                .to(memory_format=memory_format)
            )
            other = (
                torch.randn(other_shape, dtype=torch.float32, requires_grad=True)
                .add(1)
                .to(memory_format=memory_format)
                .to(dtype)
            )

            def matcher_check_fn():
                match_nodes = binary_list[binary_fn][1]
                if has_relu:
                    match_nodes += 1
                self.assertEqual(
                    counters["inductor"][
                        "mkldnn_conv_binary_unary_fusion_matcher_nodes"
                    ],
                    0 if TEST_ACL else match_nodes,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_conv_weight_pack_matcher_nodes"], 1
                )

            self._test_common(mod, (x, other), matcher_check_fn, check_autocast=dtype)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @reduced_f32_on_and_off()
    def test_conv2d_binary_broadcast_shapes(self, device):
        self.device = device
        self._test_conv_binary_broadcast_shapes_base(dim=4)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @reduced_f32_on_and_off(bf32_precision=5e-2)
    def test_conv3d_binary_broadcast_shapes(self, device):
        self.device = device
        self._test_conv_binary_broadcast_shapes_base(dim=5)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @unittest.skipIf(IS_FBCODE, "Failing in fbcode")
    @reduced_f32_on_and_off()
    def test_conv2d_linear_add_broadcast_shapes(self, device):
        self.device = device

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                self.linear = torch.nn.Linear(3, 16)

            def forward(self, x1, x2):
                return self.conv(x1) + self.linear(x2)[:, :, None, None]

        metrics.reset()
        mod = M().eval()
        x1 = torch.randn(2, 3, 56, 56)
        x2 = torch.randn(2, 3)

        def matcher_check_fn():
            match_nodes = 0 if TEST_ACL else 2
            self.assertEqual(
                counters["inductor"]["mkldnn_conv_binary_unary_fusion_matcher_nodes"],
                match_nodes,
            )
            self.assertEqual(
                counters["inductor"]["mkldnn_conv_weight_pack_matcher_nodes"], 1
            )

        self._test_common(mod, (x1, x2), matcher_check_fn)

    @skipIfNoDynamoSupport
    def test_woq_int8(self, device):
        class M(torch.nn.Module):
            def __init__(self, is_permute):
                super().__init__()
                self.is_permute = is_permute

            def forward(self, x, weight, scales):
                if self.is_permute:
                    weight = weight.t()
                    m = torch.mm(
                        x.reshape(-1, x.shape[-1]),
                        weight.to(x.dtype),
                    )
                    y = m * scales.to(m.dtype)
                    y = y.reshape(*x.shape[:-1], y.shape[-1])
                    return y
                else:
                    return (
                        torch.nn.functional.linear(x, weight.to(dtype=x.dtype)) * scales
                    )

        x_shape = (1, 1, 256)
        s_shape = 12
        x_strides = [
            (256, 256, 1),  # linear dispatching to mm
            (256, 32, 1),  # linear dispatching to bmm
        ]
        is_permutes = [False, True]
        for x_stride, is_permute in itertools.product(x_strides, is_permutes):
            mod = M(is_permute=is_permute).eval()
            x = (
                torch.randn(x_shape, dtype=torch.bfloat16)
                .as_strided(x_shape, x_stride)
                .to(device)
            )

            w_shape = (12, 256)
            w = torch.randint(-128, 127, w_shape, dtype=torch.int8).to(device)
            s = torch.randn(s_shape, dtype=torch.bfloat16).to(device)

            def matcher_check_fn():
                self.assertEqual(counters["inductor"]["woq_matcher_count"], 1)

            self._test_common(
                mod,
                (x, w, s),
                matcher_check_fn,
                check_quantization=False,
                atol=0.001,
                rtol=0.07,
            )


class TestPatternMatcher(TestPatternMatcherBase):
    @reduced_f32_on_and_off()
    def test_linear_unary(self, device="cpu"):
        self.device = device

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
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)
        if torch.backends.mkldnn.matmul.fp32_precision in ["bf16", "tf32"]:
            dtypes.append(torch.float32)
        options = itertools.product(unary_list, [True, False], dtypes)
        for unary_fn, bias, dtype in options:
            if (
                dtype != torch.float32
                and torch.backends.mkldnn.matmul.fp32_precision == "tf32"
            ):
                continue
            metrics.reset()
            mod = M(unary_fn, 10, 30, bias=bias).eval()
            # only fuse for linear when the dtype is bf16
            v = torch.randn(2, 10)

            def matcher_check_fn():
                match_nodes = unary_list[unary_fn]
                if dtype != torch.float32 and self._check_unary_is_decomposed(unary_fn):
                    # Has extra dtype conversion nodes for autocast.
                    match_nodes += 2
                self.assertEqual(
                    counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"],
                    0 if TEST_ACL else match_nodes,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"], 1
                )

            self._test_common(mod, (v,), matcher_check_fn, check_autocast=dtype)
            # only generated 1 kernel for "to_dtype"
            expected_kernel_count = 2 if TEST_ACL else 1
            if dtype == torch.float32:
                # In BF32, input is float32, will not generate kernel for "to_dtype"
                expected_kernel_count -= 1
            self.assertEqual(metrics.generated_kernel_count, expected_kernel_count)

    @reduced_f32_on_and_off()
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    def test_linear_fp32(self, device="cpu"):
        self.device = device

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
            def matcher_check_fn():
                self.assertEqual(
                    counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"], 1
                )

            self._test_common(mod, (v,), matcher_check_fn)

    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    def test_linear_input_non_contiguous_3D_wo_bias(self, device="cpu"):
        self.device = device

        # Activation is 3D, non-contiguous and without Bias
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4096, 1024, bias=False)

            def forward(self, x):
                x = torch.ops.aten.permute.default(x, [0, 2, 1, 3])
                x = torch.ops.aten.reshape.default(x, [4, 1, 4096])
                return self.linear(x)

        mod = M().eval()
        v = torch.randn(4, 32, 1, 128)

        dtypes = [torch.float]
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)

        for dtype in dtypes:
            torch._dynamo.reset()
            autocast_enabled = dtype in [torch.bfloat16, torch.float16]
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type="cpu",
                    enabled=autocast_enabled,
                    dtype=dtype,
                ),
            ):
                expected = mod(v)
                actual, (source_code,) = run_and_get_code(
                    torch.compile(mod, fullgraph=True),
                    v,
                )
                self.assertIn(
                    "torch.ops.mkldnn._linear_pointwise.default"
                    if autocast_enabled
                    else "torch.ops.mkl._mkl_linear.default",
                    source_code,
                )
                torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    @skipIfXpu(
        msg="Different with CPU, two linears will be concat on XPU for better performance"
    )
    def test_linear_add_bias(self, device="cpu"):
        self.device = device

        class M(torch.nn.Module):
            def __init__(self, device, dtype, unary_fn, cast_bias):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 64, bias=False)
                self.bias1 = torch.randn(64, device=device)
                self.linear2 = torch.nn.Linear(10, 64, bias=False)
                self.bias2 = torch.randn(64, device=device)
                if cast_bias:
                    self.bias1 = self.bias1.to(dtype=dtype, device=device)
                    self.bias2 = self.bias2.to(dtype=dtype, device=device)
                self.unary_fn = unary_fn

            def forward(self, x):
                a = self.linear1(x) + self.bias1
                b = self.linear2(x) + self.bias2
                return self.unary_fn(a), self.unary_fn(b)

        dtypes = []
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)
        options = itertools.product(unary_list, dtypes)
        for unary_fn, dtype in options:
            metrics.reset()
            fold_mod = M(self.device, dtype, unary_fn, cast_bias=True).eval()
            v = torch.randn(2, 10)

            def folder_matcher_check_fn():
                match_nodes = unary_list[unary_fn]
                if self._check_unary_is_decomposed(unary_fn):
                    # Has extra dtype conversion nodes for autocast.
                    match_nodes += 2
                # we have 2 linears, so we double the matcher_count/nodes
                self.assertEqual(
                    counters["inductor"]["mkldnn_unary_fusion_matcher_count"],
                    0 if TEST_ACL else 2,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"],
                    0 if TEST_ACL else match_nodes * 2,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"], 2
                )

            self._test_common(
                fold_mod,
                (v,),
                folder_matcher_check_fn,
                check_autocast=dtype,
            )
            self.assertEqual(metrics.generated_kernel_count, 3 if TEST_ACL else 1)
            # we won't fold the bias if bias is not same dtype with weight
            # https://github.com/pytorch/pytorch/pull/129138
            metrics.reset()
            mod = M(self.device, dtype, unary_fn, cast_bias=False).eval()

            def matcher_check_fn():
                self.assertEqual(
                    counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"], 2
                )

            self._test_common(mod, (v,), matcher_check_fn, check_autocast=dtype)
            # 1 kernel for "to_lowp", 2 kernels for unary ops
            self.assertEqual(metrics.generated_kernel_count, 3)

    @reduced_f32_on_and_off()
    def test_linear_binary(self, device="cpu"):
        self.device = device

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
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)
        if torch.backends.mkldnn.matmul.fp32_precision in ["bf16", "tf32"]:
            dtypes.append(torch.float32)
        options = itertools.product(
            binary_list, [[2, 3, 10], [2, 10]], [True, False], dtypes
        )
        out_feature = 30

        for binary_fn, input_shape, bias, dtype in options:
            metrics.reset()
            if (
                dtype != torch.float32
                and torch.backends.mkldnn.matmul.fp32_precision == "tf32"
            ):
                continue

            def matcher_check_fn():
                self.assertEqual(
                    counters["inductor"][
                        "mkldnn_conv_binary_unary_fusion_matcher_nodes"
                    ],
                    0 if TEST_ACL else 2,
                )
                reshape_linear_reshape_match_nodes = 3 if len(input_shape) == 3 else 0
                self.assertEqual(
                    counters["inductor"]["mkldnn_reshape_linear_reshape_matcher_nodes"],
                    reshape_linear_reshape_match_nodes,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"], 1
                )

            mod = M(binary_fn, input_shape[-1], out_feature, bias).eval()
            v = torch.randn(input_shape)
            other = torch.randn(input_shape[:-1] + [out_feature]).to(dtype)
            self._test_common(
                mod,
                (
                    v,
                    other,
                ),
                matcher_check_fn,
                check_autocast=dtype,
            )
            # only generated 1 kernel for "to_dtype"
            expected_kernel_count = 2 if TEST_ACL else 1
            if dtype == torch.float32:
                # In BF32, input is float32, will not generate kernel for "to_dtype"
                expected_kernel_count -= 1
            self.assertEqual(metrics.generated_kernel_count, expected_kernel_count)

    def test_linear_binary_broadcast_shapes(self, device="cpu"):
        self.device = device

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
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)
        options = itertools.product(
            binary_list,
            (
                ([2, 3, 10], [1, 1, 30]),
                ([2, 10], [1, 30]),
            ),
            (True, False),
            dtypes,
        )
        out_feature = 30

        for binary_fn, (input_shape, other_shape), bias, dtype in options:
            metrics.reset()
            mod = M(binary_fn, input_shape[-1], out_feature, bias).eval()
            v = torch.randn(input_shape)
            other = torch.randn(other_shape).to(dtype)

            def matcher_check_fn():
                reshape_linear_reshape_match_nodes = 3 if len(input_shape) == 3 else 0
                self.assertEqual(
                    counters["inductor"]["mkldnn_reshape_linear_reshape_matcher_nodes"],
                    reshape_linear_reshape_match_nodes,
                )
                self.assertEqual(
                    counters["inductor"][
                        "mkldnn_conv_binary_unary_fusion_matcher_nodes"
                    ],
                    0 if TEST_ACL else 2,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_linear_weight_pack_matcher_nodes"], 1
                )

            self._test_common(
                mod,
                (
                    v,
                    other,
                ),
                matcher_check_fn,
                check_autocast=dtype,
            )
            self.assertEqual(metrics.generated_kernel_count, 2 if TEST_ACL else 1)

    @skipIfXpu(
        msg="Different with CPU, two linears will be concat on XPU for better performance"
    )
    def test_multi_linear_share_same_input(self, device="cpu"):
        self.device = device

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
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)

        def matcher_check_fn():
            # SiLU: 5 base nodes + 2 dtype conversion = 7
            # ReLU: 2 base nodes (non-decomposed, no dtype conversion)
            # Total: 7 + 2 = 9
            self.assertEqual(
                counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"],
                0 if TEST_ACL else 9,
            )
            self.assertEqual(
                counters["inductor"]["mkldnn_unary_fusion_matcher_count"],
                0 if TEST_ACL else 2,
            )
            self.assertEqual(
                counters["inductor"]["mkldnn_reshape_linear_reshape_matcher_nodes"], 6
            )
            self.assertEqual(
                counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"], 2
            )

        for dtype in dtypes:
            mod = M().to(dtype).eval()
            v = torch.randn(2, 4, 16).to(dtype)
            self._test_common(mod, (v,), matcher_check_fn, rtol=1e-2, atol=1e-2)


instantiate_device_type_tests(
    TestPatternMatcherGeneric, globals(), allow_xpu=True, only_for=("cpu", "xpu")
)
instantiate_parametrized_tests(TestPatternMatcher)
if __name__ == "__main__":
    if IS_LINUX and (HAS_CPU) and torch.backends.mkldnn.is_available():
        run_tests()
