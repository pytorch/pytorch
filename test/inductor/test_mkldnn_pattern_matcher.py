# Owner(s): ["module: inductor"]
import itertools

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import expectedFailureDynamicWrapper
from torch._dynamo.utils import counters
from torch._inductor.utils import run_and_get_code
from torch.nn import functional as F
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CPU

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
    lambda x: F.relu(x): 2,
    lambda x: F.sigmoid(x): 2,
    lambda x: F.tanh(x): 2,
    lambda x: F.hardswish(x): 6,
    lambda x: F.leaky_relu(x, 0.1): 4,
    lambda x: F.hardtanh(x, min_val=-0.5, max_val=4): 3,
    lambda x: F.gelu(x, approximate="none"): 6,
    lambda x: F.gelu(x, approximate="tanh"): 10,
    lambda x: F.relu6(x): 3,
    lambda x: F.silu(x): 3,
    lambda x: F.hardsigmoid(x): 5,
    lambda x: torch.relu(x): 2,
    lambda x: torch.sigmoid(x): 2,
    lambda x: torch.tanh(x): 2,
    lambda x: x.relu(): 2,
    lambda x: x.sigmoid(): 2,
    lambda x: x.tanh(): 2,
}

unary_list_bf16 = {
    torch.nn.ReLU(): 2,
    torch.nn.Sigmoid(): 2,
    torch.nn.Tanh(): 2,
    lambda x: F.relu(x): 2,
    lambda x: F.sigmoid(x): 2,
    lambda x: F.tanh(x): 2,
    lambda x: torch.relu(x): 2,
    lambda x: torch.sigmoid(x): 2,
    lambda x: torch.tanh(x): 2,
    lambda x: x.relu(): 2,
    lambda x: x.sigmoid(): 2,
    lambda x: x.tanh(): 2,
}

binary_list = {
    lambda x, y: torch.add(x, y): 2,  # call_function
    lambda x, y: torch.add(y, x): 2,  # call_function
    lambda x, y: x.add(y): 2,  # call_method
    lambda x, y: x.add_(y): 2,  # call_method
    lambda x, y: torch.sub(x, y): 2,  # call_function
    lambda x, y: x.sub(y): 2,  # call_method
    lambda x, y: x.sub_(y): 2,  # call_method
}


class TestPaternMatcher(TestCase):
    def _clone_inputs(self, inputs):
        def clone(x):
            if not isinstance(x, torch.Tensor):
                return x
            return x.clone()

        return tuple(clone(x) for x in inputs)

    def _test_common(
        self, mod, inputs, matcher_count, matcher_nodes, atol=1e-5, rtol=1.3e-6
    ):
        counters.clear()
        with torch.no_grad():
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

        test_memory_format = [torch.contiguous_format, torch.channels_last]
        options = itertools.product(
            unary_list.keys(),
            test_memory_format,
        )

        for (
            unary_fn,
            memory_format,
        ) in options:
            x_shape = (1, 3, 56, 56)
            mod = M(unary_fn).eval()

            # TODO: add bf16 test for cpu path?
            # TODO: this test fails when requires_grad=False
            v = (
                torch.randn(x_shape, dtype=torch.float32, requires_grad=True)
                .add(1)
                .to(memory_format=memory_format)
            )
            self._test_common(mod, (v,), 1, unary_list[unary_fn])

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

        options = itertools.product(unary_list_bf16, [True, False])
        dtype = torch.bfloat16
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            for unary_fn, bias in options:
                mod = M(unary_fn, 10, 30, bias=bias).eval()
                # only fuse for linear when the dtype is bf16
                mod = mod.to(dtype)
                v = torch.randn(2, 10).to(dtype)
                self._test_common(
                    mod, (v,), 1, unary_list_bf16[unary_fn], atol=1e-5, rtol=1.6e-2
                )

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

        test_memory_format = [torch.contiguous_format, torch.channels_last]
        options = itertools.product(
            unary_list,
            test_memory_format,
        )

        for unary_fn, memory_format in options:
            x_shape = (1, 3, 28, 28)
            mod = M(unary_fn).eval()

            v = torch.randn(x_shape, dtype=torch.float32).to(
                memory_format=memory_format
            )
            self._test_common(mod, (v,), 1, unary_list[unary_fn])

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
            match_nodes = (
                binary_list[binary_fn] + 1 if has_relu else binary_list[binary_fn]
            )
            self._test_common(mod, (v,), 1, match_nodes)

    @expectedFailureDynamicWrapper
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
                x = self.binary_fn(x, y)
                return x

        options = itertools.product(binary_list, [[2, 3, 10], [2, 10]], [True, False])
        dtype = torch.bfloat16
        out_feature = 30
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            for binary_fn, input_shape, bias in options:
                mod = M(binary_fn, input_shape[-1], out_feature, bias).to(dtype).eval()
                v = torch.randn(input_shape).to(dtype)
                other = torch.randn(input_shape[:-1] + [out_feature]).to(dtype)
                self._test_common(
                    mod, (v, other), 1, binary_list[binary_fn], rtol=1e-2, atol=1e-2
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
        mod = Model().eval()
        v = torch.randn(1, 3, 28, 28)
        for min_value, max_value in zip(min_values, max_values):
            self._test_common(mod, (v, min_value, max_value), 1, 3)

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
            mod = Model().eval()
            v = torch.randn(1, 3, 28, 28)
            for negative_slope in negative_slopes:
                self._test_common(mod, (v, negative_slope), 1, 4)

    # https://github.com/pytorch/pytorch/issues/99838.
    def test_conv2d_add_scalar(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
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
            self._test_common(mod, (v,), 0, 0)

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
                super(Model2, self).__init__()
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
                super(Model3, self).__init__()
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
        mod = Model().to(memory_format=torch.channels_last).eval()
        for other, alpha in zip(others, [0.1, 1.0]):
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
                super(Model, self).__init__()
                self.conv = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

            def forward(self, input_tensor):
                x = self.conv(input_tensor)
                x = F.relu(x + torch.ones(x.size()))
                return x

        input = torch.randn(1, 3, 14, 14)
        mod = Model().eval()
        include_ops = ["mkldnn._convolution_pointwise_.binary"]
        self._test_code_common(mod, (input,), include_ops, [])


if __name__ == "__main__":
    if IS_LINUX and HAS_CPU and torch.backends.mkldnn.is_available():
        run_tests()
