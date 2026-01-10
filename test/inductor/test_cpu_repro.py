# Owner(s): ["oncall: cpu inductor"]
import contextlib
import copy
import functools
import itertools
import math
import os
import platform
import sys
import unittest
from collections.abc import Callable
from unittest.mock import patch

import torch
from torch import nn
from torch._C import FileCheck
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import same
from torch._inductor import config, cpu_vec_isa, metrics, test_operators
from torch._inductor.codegen.cpp import CppOverrides, CppVecOverrides
from torch._inductor.compile_fx import (
    compile_fx,
    compile_fx_inner,
    complex_memory_overlap,
)
from torch._inductor.exc import InductorError
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import timed
from torch._prims_common import is_float_dtype
from torch.autograd.functional import vjp
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn import functional as F
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_MACOS,
    parametrize,
    skipIfRocm,
    slowTest,
    TEST_MKL,
    xfailIfS390X,
)
from torch.utils._python_dispatch import TorchDispatchMode


try:
    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


vec_dtypes = test_torchinductor.vec_dtypes
_lowp_fp_dtypes = (
    torch.bfloat16,
    torch.float16,
)
run_and_get_cpp_code = test_torchinductor.run_and_get_cpp_code
TestCase = test_torchinductor.TestCase
aten = torch.ops.aten
check_model = test_torchinductor.check_model

requires_vectorization = unittest.skipUnless(
    cpu_vec_isa.valid_vec_isa_list() and os.getenv("ATEN_CPU_CAPABILITY") != "default",
    "Does not support vectorization",
)


def _can_check_vec_metrics():
    return (
        cpu_vec_isa.valid_vec_isa_list()
        and os.getenv("ATEN_CPU_CAPABILITY") != "default"
        and config.cpp.simdlen != 1
    )


def check_metrics_vec_kernel_count(num_expected_vec_kernels):
    if _can_check_vec_metrics():
        assert metrics.generated_cpp_vec_kernel_count == num_expected_vec_kernels


def simd_lengths_to_test():
    """Returns a minimal list of simd lengths to cover common cases"""
    simdlens = [None, 1]
    valid_isa_list = cpu_vec_isa.valid_vec_isa_list()
    if valid_isa_list:
        simdlens.append(valid_isa_list[0].bit_width())
    return simdlens


@contextlib.contextmanager
def set_num_threads(num_threads):
    orig_num_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    yield
    torch.set_num_threads(orig_num_threads)


class LstmModule(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bias=True,
        bidirectional=False,
        batch_first=False,
    ):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )

    def forward(self, x, h=None):
        x, h = self.lstm(x, h)
        return x, h


@instantiate_parametrized_tests
class CPUReproTests(TestCase):
    common = check_model

    def test_torch_linalg_qr_tuple_slice(self):
        def fn(x):
            return torch.linalg.qr(x)[:1]

        x = torch.randn(4, 4)
        compiled = torch.compile(fn, backend="inductor")

        expected = fn(x)
        actual = compiled(x)

        self.assertIsInstance(actual, tuple)
        self.assertEqual(len(actual), 1)
        torch.testing.assert_close(actual[0], expected[0])

    @skipIfRocm
    def test_conv_stride_constraints(self):
        for fmt in [torch.contiguous_format, torch.channels_last]:
            # TorchDispatch doesn't work in our cuda invocation for some reason
            m = torch.nn.Conv2d(5, 6, [3, 3])

            def fn(inp, weight):
                return (
                    F.conv2d(
                        inp, weight, None, m.stride, m.padding, m.dilation, m.groups
                    ),
                )

            inp = torch.randn([2, 5, 16, 16])
            inps = [inp, m.weight.to(memory_format=fmt)]
            fn_fx = make_fx(fn)(*inps)
            fn_compiled = compile_fx_inner(fn_fx, inps)
            test_self = self
            conv_seen = False

            class RecordFunctions(TorchDispatchMode):
                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    kwargs = kwargs if kwargs else {}
                    if func == torch.ops.aten.convolution.default:
                        # For CPU and mkldnn enable, we always using channels last
                        nonlocal fmt
                        if (
                            torch.backends.mkldnn.enabled
                            and torch.backends.mkldnn.is_available()
                        ):
                            fmt = torch.channels_last
                        test_self.assertTrue(args[0].is_contiguous(memory_format=fmt))
                        test_self.assertTrue(args[1].is_contiguous(memory_format=fmt))
                        nonlocal conv_seen
                        conv_seen = True

                    return func(*args, **kwargs)

            with RecordFunctions():
                fn_compiled(inps)

            self.assertTrue(conv_seen)

    @patch("torch.cuda.is_available", lambda: False)
    def test_conv2d_bn_mixed_dtype(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3,
                    16,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    dtype=torch.bfloat16,
                )
                self.bn = torch.nn.BatchNorm2d(
                    16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
                )

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        v = torch.randn(1, 3, 64, 64, dtype=torch.bfloat16)
        mod = Model().eval()
        with torch.no_grad():
            self.common(
                mod,
                (v,),
            )

    def test_complex_cholesky_mh_view_fallback(self):
        torch.manual_seed(0)

        n = 8

        def fn(inp: torch.Tensor):
            I0 = torch.eye(n, dtype=inp.dtype, device=inp.device)
            I = I0.unsqueeze(0).expand(inp.shape[0], n, n).contiguous()
            hermitian = I + 0.5 * (inp @ inp.mH)
            chol = torch.linalg.cholesky(hermitian, upper=True)
            return chol.abs().sum()

        base = torch.randn(4, n, n, dtype=torch.complex64)

        def run(compiled_fn):
            inp = base.clone().detach().requires_grad_(True)
            loss = compiled_fn(inp)
            loss.backward()
            return loss.detach(), inp.grad.detach()

        expected_loss, expected_grad = run(fn)

        compiled = torch.compile(fn, backend="inductor")
        actual_loss, actual_grad = run(compiled)

        torch.testing.assert_close(actual_loss, expected_loss)
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_nn_fold(self):
        # Fix https://github.com/pytorch/pytorch/issues/147848

        class Model(torch.nn.Module):
            def __init__(self, output_size, kernel_size, stride) -> None:
                super().__init__()
                self.fold = torch.nn.Fold(
                    output_size=output_size, kernel_size=kernel_size, stride=stride
                )

            def forward(self, x):
                x = self.fold(x)
                return x

        output_sizes = [(64, 64), (64, 64)]
        kernel_sizes = [(32, 32), (32, 32)]
        strides = [(1, 1), (2, 2)]
        input_sizes = [(1, 32 * 32, 1089), (1, 64 * 64, 289)]

        for idx in range(len(output_sizes)):
            output_size = output_sizes[idx]
            kernel_size = kernel_sizes[idx]
            stride = strides[idx]
            input_size = input_sizes[idx]

            for num_threads in [1, None]:
                torch._dynamo.reset()
                metrics.reset()
                v = torch.randn(*input_size)
                mod = Model(output_size, kernel_size, stride).eval()
                with (
                    contextlib.nullcontext()
                    if (num_threads != 1)
                    else set_num_threads(1)
                ):
                    with torch.no_grad():
                        self.common(
                            mod,
                            (v,),
                        )

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    def test_conv2d_packed(self):
        options = itertools.product([[3, 56, 56]], [True, False], [0, (0,)])
        for x_shape, mode_train, padding in options:
            mod = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, 3, padding=padding)
            ).train(mode=mode_train)
            v = torch.randn(x_shape, dtype=torch.float32)

            with torch.no_grad():
                self.common(
                    mod,
                    (v,),
                )

    @patch("torch.cuda.is_available", lambda: False)
    def test_conv2d_autocast(self):
        v = torch.randn(1, 3, 28, 18, dtype=torch.float32)
        mod = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3, 3)).eval()
        with torch.no_grad(), torch.cpu.amp.autocast():
            self.common(
                mod,
                (v,),
            )

    def test_conv1d_strided_weight_torch_compile(self):
        def fn(x, w):
            wt = w.transpose(2, 1)
            y = F.conv1d(x, wt)
            return y.clone()

        x_eager = torch.randn(2, 3, 5, requires_grad=True)
        w_eager = torch.randn(4, 2, 3, requires_grad=True)

        out_eager = fn(x_eager, w_eager)
        grad = torch.randn_like(out_eager)
        out_eager_val = out_eager.detach()
        out_eager.backward(grad)
        grad_x_eager = x_eager.grad.detach().clone()
        grad_w_eager = w_eager.grad.detach().clone()

        x_comp = x_eager.detach().requires_grad_(True)
        w_comp = w_eager.detach().requires_grad_(True)
        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)
        out_comp = compiled(x_comp, w_comp)
        out_comp_val = out_comp.detach()
        out_comp.backward(grad)

        torch.testing.assert_close(out_comp_val, out_eager_val)
        torch.testing.assert_close(x_comp.grad, grad_x_eager)
        torch.testing.assert_close(w_comp.grad, grad_w_eager)

    @config.patch(freezing=True)
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @patch("torch.cuda.is_available", lambda: False)
    def test_mkl_linear(self):
        dtypes = [torch.float32]
        options = itertools.product([[2, 3, 10]], [2], [True, False], dtypes)
        for input_shape, out_dim, bias, dtype in options:
            mod = torch.nn.Sequential(
                torch.nn.Linear(input_shape[-1], out_dim, bias=bias)
            ).eval()

            v = torch.randn(input_shape)
            with torch.no_grad():
                self.common(
                    mod.to(dtype),
                    (v.to(dtype),),
                )

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    def test_unsupported_conv_transpose(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv_transpose = torch.nn.ConvTranspose2d(
                    3, 6, 3, stride=1, padding=1, output_padding=1
                )

            def forward(self, input_tensor):
                x = self.conv_transpose(input_tensor)
                output = torch.tanh(x)
                return output

        input = torch.randn(1, 3, 28, 28)
        m = Model().eval()

        with torch.no_grad():
            compiled_m = torch.compile(m)
            # The cpp_wrapper C-shim can't utilize the Python error API, so error
            # messages are printed to stderr directly, and the intercepted RuntimeError
            # is significantly less verbose.
            msg = (
                r"aoti_torch_cpu_convolution\(.*\) API call failed"
                if config.cpp_wrapper
                else "output padding must be smaller than either stride or dilation"
            )
            with self.assertRaisesRegex(RuntimeError, msg):
                compiled_m(input)

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    def test_conv_used_from_multiple_places(self):
        class M(torch.nn.Module):
            def __init__(self, conv_in_channel, conv_out_channel) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(conv_in_channel, conv_out_channel, (3, 3))

            def forward(self, x):
                res = self.conv(x)
                res = F.relu(res)
                res = self.conv(res)
                return res

        with torch.no_grad():
            mod = M(3, 3).eval()
            x = torch.randn(1, 3, 224, 224)
            self.common(
                mod,
                (x,),
            )

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    def test_linear_used_from_multiple_places(self):
        class M(torch.nn.Module):
            def __init__(self, in_channel, out_channel) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(in_channel, out_channel)

            def forward(self, x):
                res = self.linear(x)
                res = F.relu(res)
                res = self.linear(res)
                return res

        dtypes = []
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        for dtype in dtypes:
            with torch.no_grad():
                m = M(224, 224).to(dtype).eval()
                m_opt = torch.compile(m)
                x = torch.randn(224, 224, dtype=dtype)
                m_opt(x)
                self.assertEqual(m(x), m_opt(x))

    @config.patch(implicit_fallbacks=True)
    def test_multihead_attention_cpu(self):
        def fn(
            q,
            k,
            v,
            embed_dim,
            num_heads,
            qkv_weight,
            qkv_bias,
            proj_weight,
            proj_bias,
            mask,
            need_weights,
        ):
            return torch._native_multi_head_attention(
                q,
                k,
                v,
                embed_dim,
                num_heads,
                qkv_weight,
                qkv_bias,
                proj_weight,
                proj_bias,
                mask,
                need_weights,
            )

        B = 1
        T = 3
        embed_dim = 6
        num_heads = 2
        q = torch.randn([B, T, embed_dim])
        k = torch.randn([B, T, embed_dim])
        v = torch.randn([B, T, embed_dim])
        qkv_weight = torch.randn([3 * embed_dim, embed_dim])
        qkv_bias = torch.randn([3 * embed_dim])
        proj_weight = torch.randn([3 * embed_dim, embed_dim])
        proj_bias = torch.randn([3 * embed_dim])
        mask = None
        need_weights = False

        inps = [
            q,
            k,
            v,
            embed_dim,
            num_heads,
            qkv_weight,
            qkv_bias,
            proj_weight,
            proj_bias,
            mask,
            need_weights,
        ]
        self.common(fn, inps)

    @config.patch(freezing=True)
    def test_module_buffer_mutation(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = torch.nn.Buffer(torch.rand((3, 10)))

            def forward(self, x):
                lx = [x, x.clone(), x.clone()]
                y = []
                for i in range(3):
                    y.append(lx[i] + self.foo[i])
                return torch.cat(y, 1)

        with torch.no_grad():
            example_inputs = (torch.rand(1, 10),)
            self.common(Model(), example_inputs)

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    def test_linear_packed(self):
        dtypes = []
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        options = itertools.product(
            [[2, 3, 10], [2, 10], [10], [2, 0]], [3, 0], [True, False], dtypes
        )
        for input_shape, out_dim, bias, dtype in options:
            mod = torch.nn.Sequential(
                torch.nn.Linear(input_shape[-1], out_dim, bias=bias)
            ).eval()

            v = torch.randn(input_shape)
            with torch.no_grad():
                self.common(
                    mod.to(dtype),
                    (v.to(dtype),),
                )

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    def test_conv_transpose2d_packed_cpu(self):
        options = itertools.product([[1, 3, 28, 28], [3, 28, 28]], [0, (0,)])
        for x_shape, padding in options:
            mod = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(3, 64, 3, 3, padding=padding)
            ).eval()
            v = torch.randn(x_shape, dtype=torch.float32)
            with torch.no_grad():
                self.common(
                    mod,
                    (v,),
                )

    @torch._dynamo.config.patch(
        {"dynamic_shapes": True, "assume_static_by_default": False}
    )
    def test_full_boolean_dynamic_shape(self):
        def fn(n):
            x = torch.full((1024,), n >= 1024)
            return x, x + 1

        self.common(fn, (1024,))
        self.common(fn, (1023,))

    @config.patch(freezing=True)
    @unittest.skipIf(not torch._C._has_mkldnn, "MKLDNN is not enabled")
    @torch._dynamo.config.patch(dynamic_shapes=True)
    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_conv_in_channel_1_dynamic_shapes(self):
        class M(torch.nn.Module):
            def __init__(self, in_channel, out_channel) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(in_channel, out_channel, 3)

            def forward(self, x):
                res = self.conv(x)
                res = F.relu(res)
                return res

        # test the case where the channels dim of the input is 1
        # Reproducer from the maml_omniglot model in Torchbench
        in_channel = 1
        out_channel = 3
        amp_enabled_configs = [False]
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            # When amp is enabled here, the input to Conv is a FlexibleLayout.
            # While it's disabled, the input is a FixedLayout.
            amp_enabled_configs.append(True)
        for amp_enabled in amp_enabled_configs:
            mod = M(in_channel, out_channel).eval()
            v = torch.randn(5, in_channel, 15, 15)
            with torch.no_grad(), torch.cpu.amp.autocast(enabled=amp_enabled):
                self.common(
                    mod,
                    (v,),
                )

    @unittest.skipIf(not torch._C._has_mkldnn, "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    @torch._dynamo.config.patch(dynamic_shapes=True)
    @torch._dynamo.config.patch(assume_static_by_default=False)
    @torch._dynamo.config.patch(allow_rnn=True)
    @config.patch(freezing=True)
    def _test_lstm_packed(
        self,
        unbatched,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        bias,
        empty_state,
        batch_first,
        batch_size,
        seq_len,
        change_input_sizes=False,
    ):
        from torch._dynamo.utils import counters

        dtypes = [torch.float]
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        for dtype in dtypes:
            counters.clear()
            num_directions = 2 if bidirectional else 1

            seq_len_var = seq_len + 3
            if unbatched:
                v = torch.randn(seq_len, input_size)
                v_var = torch.randn(seq_len_var, input_size)
                h = torch.randn(num_layers * num_directions, hidden_size)
                c = torch.randn(num_layers * num_directions, hidden_size)
            else:
                if batch_first:
                    v = torch.randn(batch_size, seq_len, input_size)
                    v_var = torch.randn(batch_size, seq_len_var, input_size)
                else:
                    v = torch.randn(seq_len, batch_size, input_size)
                    v_var = torch.randn(seq_len_var, batch_size, input_size)
                h = torch.randn(num_layers * num_directions, batch_size, hidden_size)
                c = torch.randn(num_layers * num_directions, batch_size, hidden_size)

            mod = LstmModule(
                input_size,
                hidden_size,
                num_layers,
                bias,
                bidirectional,
                batch_first,
            ).eval()
            maybe_autocast = (
                torch.cpu.amp.autocast()
                if dtype == torch.bfloat16
                else contextlib.nullcontext()
            )

            with torch.no_grad(), maybe_autocast:
                inps = [v]
                if not empty_state:
                    inps.append((h, c))

                fn_opt = torch.compile(mod, backend="inductor")
                _, code = run_and_get_cpp_code(fn_opt, *inps)

                # Check that _flat_weights are not functional_tensor, otherwise
                # deepcopy will fail during recompilation.
                fn_opt_copy = copy.deepcopy(fn_opt)
                _flat_weights = fn_opt_copy.lstm._flat_weights
                for _flat_weight in _flat_weights:
                    self.assertFalse(torch._is_functional_tensor(_flat_weight))

                self.assertTrue("aten.mkldnn_rnn_layer" in code)
                self.assertEqual(fn_opt(*inps), mod(*inps))
                self.assertEqual(
                    counters["inductor"]["pattern_matcher_count"],
                    num_layers * num_directions
                    + 2,  # num of mkldnn_rnn_layer call + 2 view call on the concatenated hy, cy.
                )

                # Change input sizes
                if change_input_sizes:
                    inps_var = [v_var]
                    self.assertEqual(fn_opt(*inps_var), mod(*inps_var))

    @parametrize(
        "unbatched, input_size, hidden_size, num_layers, bidirectional, bias, empty_state, batch_first, batch_size, seq_len",
        itertools.product(
            *[
                [True, False],
                [1, 7],
                [7],
                [1, 7],
                [False, True],
                [False, True],
                [False, True],
                [True, False],
                [1, 7],
                [1, 7],
            ]
        ),
    )
    def test_lstm_packed(
        self,
        unbatched,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        bias,
        empty_state,
        batch_first,
        batch_size,
        seq_len,
    ):
        self._test_lstm_packed(
            unbatched,
            input_size,
            hidden_size,
            num_layers,
            bidirectional,
            bias,
            empty_state,
            batch_first,
            batch_size,
            seq_len,
        )

    _test_lstm_packed_change_input_sizes_cpu_params = list(
        itertools.product(
            *[
                [False],
                [2],
                [5],
                [3],
                [True],
                [True],
                [False],
                [False],
                [2],
                [3],
            ]
        )
    )

    @parametrize(
        "unbatched, input_size, hidden_size, num_layers, bidirectional, bias, empty_state, batch_first, batch_size, seq_len",
        _test_lstm_packed_change_input_sizes_cpu_params,
    )
    def test_lstm_packed_change_input_sizes_cpu(
        self,
        unbatched,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        bias,
        empty_state,
        batch_first,
        batch_size,
        seq_len,
    ):
        self._test_lstm_packed(
            unbatched,
            input_size,
            hidden_size,
            num_layers,
            bidirectional,
            bias,
            empty_state,
            batch_first,
            batch_size,
            seq_len,
            change_input_sizes=True,
        )

    def test_set_source_Tensor(self):
        class MaskedConv2d(torch.nn.Conv2d):
            def __init__(
                self,
                *,
                in_channels: int,
                out_channels: int,
                kernel_size: int,
                padding: int = 0,
            ) -> None:
                super().__init__(
                    in_channels, out_channels, kernel_size, padding=padding
                )
                mask = torch.zeros_like(self.weight)

                mask[:, :, : kernel_size // 2, :] = 1
                mask[:, :, kernel_size // 2, : kernel_size // 2] = 1
                self.register_buffer("mask", mask)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    self.weight.data *= self.mask
                return super().forward(x)

        class M(torch.nn.Module):
            def __init__(
                self, num_channels: int, num_colors: int, H: int, W: int
            ) -> None:
                super().__init__()
                self.num_channels = num_channels
                self.num_colors = num_colors
                self.H = H
                self.W = W
                kernel_size = 7
                padding = (kernel_size - 1) // 2
                # 1 7x7 Mask
                layers = [
                    MaskedConv2d(
                        in_channels=self.num_channels,
                        out_channels=64,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                ]
                self.model = nn.Sequential(*layers)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x.permute(0, 3, 1, 2)
                return self.model(x)

        model = M(H=32, W=32, num_channels=4, num_colors=2)
        fn_opt = torch.compile(model, backend="inductor")
        v = (torch.rand(10, 32, 32, 4) > 0.5).to(torch.float32)
        inp = v.clone()
        result, code = run_and_get_cpp_code(fn_opt, inp)
        self.assertIn(
            "aoti_torch_cpu_set__source_Tensor"
            if config.cpp_wrapper
            else "aten.set_.source_Tensor",
            code,
        )
        expected = model(inp)
        self.assertEqual(expected, result)

        # test cpp_wrapper_build_separate
        with config.patch(cpp_wrapper=True, cpp_wrapper_build_separate=True):
            result, code = run_and_get_cpp_code(fn_opt, inp)
            self.assertIn("kernel_src", code)
            self.assertEqual(expected, result)

        with config.patch(cpp_wrapper=True, cpp_wrapper_build_separate=False):
            result, code = run_and_get_cpp_code(fn_opt, inp)
            self.assertNotIn("kernel_src", code)
            self.assertEqual(expected, result)

    @torch._dynamo.config.patch(dynamic_shapes=True)
    @torch._dynamo.config.patch(assume_static_by_default=False)
    @torch._dynamo.config.patch(allow_rnn=True)
    def test_pack_padded_sequence_lstm(self):
        embedding_dim = 12
        hidden_dim = 10
        batch_size = 24
        num_layers = 1
        bidirectional = True
        num_direc = 2
        max_lens = 96

        sent = torch.randn(batch_size, max_lens, embedding_dim)
        hid_0 = torch.rand(num_layers * num_direc, batch_size, hidden_dim)
        hid_1 = torch.randn(num_layers * num_direc, batch_size, hidden_dim)

        sent_lens = torch.Tensor(
            [1, 2, 3, 4, 5, 1, 3, 2, 96, 5, 3, 1, 1, 2, 1, 2, 3, 6, 1, 2, 4, 6, 2, 1]
        )

        assert sent_lens.shape[0] == batch_size
        assert sent_lens.max().item() == max_lens

        hidden_0 = hid_0.clone().requires_grad_(False)
        hidden_1 = hid_1.clone().requires_grad_(False)
        embeds = torch.nn.utils.rnn.pack_padded_sequence(
            sent, sent_lens, batch_first=True, enforce_sorted=False
        )

        mod = LstmModule(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bias=True,
            bidirectional=bidirectional,
            batch_first=True,
        ).eval()

        with torch.no_grad():
            inps = [embeds, (hidden_0, hidden_1)]
            fn_opt = torch.compile(mod, backend="inductor")
            _, code = run_and_get_cpp_code(fn_opt, *inps)
            # This case is unsupported
            self.assertFalse("torch.ops.mkldnn._lstm" in code)
            self.assertEqual(fn_opt(*inps), mod(*inps))

    @patch("torch.cuda.is_available", lambda: False)
    def test_conv_transpose2d_has_output_size_input(self):
        # https://github.com/pytorch/pytorch/issues/100344.
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv_transpose = torch.nn.ConvTranspose2d(
                    in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x):
                return self.conv_transpose(x, output_size=(10, 10))

        mod = M().eval()
        v = torch.randn(1, 3, 10, 10, dtype=torch.float32)
        with torch.no_grad():
            self.common(
                mod,
                (v,),
            )

    def test_pad_with_nan_value(self):
        # https://github.com/pytorch/pytorch/issues/100988.
        class Model(torch.nn.Module):
            def forward(self, x):
                x = F.pad(x, (1, 1, 1, 1), value=float("nan"))
                return x

        mod = Model().eval()
        v = torch.randn(1, 3, 10, 10, dtype=torch.float32)
        with torch.no_grad():
            self.common(
                mod,
                (v,),
            )

    def test_masked_fill_with_inf_or_nan_value(self):
        def fn(value, mask):
            y1 = torch.masked_fill(value, mask, float("inf"))
            y2 = torch.masked_fill(value, mask, float("-inf"))
            y3 = torch.masked_fill(value, mask, float("nan"))
            return y1, y2, y3

        value = torch.randn((2, 17))
        mask = torch.randint(0, 1, size=(2, 17), dtype=torch.uint8).to(torch.bool)
        with torch.no_grad():
            self.common(
                fn,
                (value, mask),
            )

    def test_relu_with_inf_value(self):
        # https://github.com/pytorch/pytorch/issues/117544.

        def fn(out):
            out = torch.sinh(input=out)
            out = torch.relu(input=out)
            return out

        x = torch.Tensor([-572373.5000, 755109.1250, 330995.5625])
        with torch.no_grad():
            self.common(
                fn,
                (x,),
            )

    def test_acosh_with_negative_large_input(self):
        # https://github.com/pytorch/pytorch/issues/118267.

        def fn(input):
            out = torch.acosh(input)
            return out

        x = torch.Tensor(
            [
                [
                    -8493.9854,
                    431654.1250,
                    71741.5859,
                    608234.5000,
                    -103814.7500,
                    -699397.0000,
                    -910685.8125,
                    -832737.1875,
                    875343.5000,
                ]
            ]
        ).repeat(3, 9)

        for dtype in [torch.float32, torch.bfloat16, torch.double]:
            with torch.no_grad():
                torch._dynamo.reset()
                metrics.reset()
                _x = x.to(dtype)
                self.common(
                    fn,
                    (_x,),
                )

    @requires_vectorization
    def test_asinh_with_corner_inputs(self):
        # https://github.com/pytorch/pytorch/issues/142345

        def fn(input):
            out = torch.asinh(input)
            return out

        x = torch.tensor([0, 0, 0, -10000.1]).repeat(3, 4)

        bit_widths = [isa._bit_width for isa in cpu_vec_isa.valid_vec_isa_list()]
        for dtype in [torch.float32, torch.bfloat16, torch.float16, torch.double]:
            for simdlen in bit_widths:
                with torch.no_grad(), config.patch({"cpp.simdlen": simdlen}):
                    torch._dynamo.reset()
                    metrics.reset()
                    _x = x.to(dtype)
                    self.common(fn, (_x,))
                    check_metrics_vec_kernel_count(1)

    @config.patch(fallback_random=True)
    def test_require_stride_order_non_owning(self):
        def test_concat_with_conv():
            x1 = torch.randn(2, 3, 4, 4).to(memory_format=torch.channels_last)
            x2 = torch.randn(2, 5, 4, 4).to(memory_format=torch.channels_last)

            # First do the concatenation
            cat_result = torch.cat([x1, x2], dim=1)

            # Then use x1 (which was an input to the cat) in a conv
            conv_weight = torch.randn(4, 3, 3, 3).to(memory_format=torch.channels_last)
            x1_conv = torch.nn.functional.conv2d(x1, conv_weight, padding=1)

            return cat_result, x1_conv

        torch.manual_seed(1)
        f_c = torch.compile(test_concat_with_conv)
        out_result, code = run_and_get_cpp_code(f_c)

        torch.manual_seed(1)
        self.assertEqual(out_result, test_concat_with_conv())

        # both inputs to conv should be channels last
        if config.cpp_wrapper:
            FileCheck().check("{2L, 3L, 4L, 4L}").check("{128L, 1L, 32L, 8L}").check(
                "{4L, 3L, 3L, 3L}"
            ).check("{27L, 1L, 9L, 3L}").check("aoti_torch_empty_strided").run(code)
        else:
            FileCheck().check("(2, 3, 4, 4), (128, 1, 32, 8)").check(
                "empty_strided_cpu((4, 3, 3, 3), (27, 1, 9, 3)"
            ).run(code)

    @config.patch(implicit_fallbacks=True)
    def test_repeat_interleave(self):
        def fn(y):
            return torch.repeat_interleave(y, 2, output_size=8)

        a = torch.tensor([[1, 2], [3, 4]])
        self.common(
            fn,
            (a,),
        )

    def test_inplace_squeeze_needed(self):
        mod = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.LayerNorm(10),
            torch.nn.ReLU(),
        ).eval()

        def fn(x):
            return mod(x)

        v = torch.randn(10)
        # TODO: OMP parallel reduction order is not deterministic.
        # Hence, the accuracy might vary up and down. For short term,
        # we increase the tolerance and will fix it later by using
        # aten parallel.
        self.common(fn, (v,), atol=5e-1, rtol=5e-1)

    def test_parallel_reduction_vectorization(self):
        # Fix issue: https://github.com/pytorch/pytorch/issues/151523
        class Model(torch.nn.Module):
            def __init__(self, enable_masked_tail_vec):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3,
                    out_channels=16,
                    kernel_size=(1, 7),
                    stride=(2, 1),
                    padding=0,
                )
                self.enable_masked_tail_vec = enable_masked_tail_vec

            def forward(self, x, weight):
                x = self.conv(x)
                if not self.enable_masked_tail_vec:
                    x = F.hardshrink(x, lambd=0)
                x = x.view(x.size(0), -1)
                x = torch.mv(weight, x[0])
                return x

        for enable_masked_tail_vec in [True, False]:
            mod = Model(enable_masked_tail_vec).eval()
            x = torch.randn(2, 3, 127, 255)
            weight = torch.randn(10, 254976)
            # Use same criterion as test_inplace_squeeze_needed
            # for parallel reduction.
            self.common(mod, (x, weight), atol=5e-1, rtol=5e-1)

    def test_cat_mul(self):
        # https://github.com/pytorch/pytorch/issues/93365
        def fn(p0, p1):
            y1 = torch.cat([p0, p1], dim=0)
            y2 = torch.mul(y1, y1)
            return y1, y2

        p0 = torch.randn(3, 4)
        p1 = torch.randn(3, 4)
        self.common(fn, (p0, p1))

    def test_pow_cos(self):
        # https://github.com/pytorch/pytorch/issues/98149
        def fn(x):
            t = x.pow(5)
            return torch.cos(t)

        x = torch.tensor([4], dtype=torch.uint8)
        self.common(fn, (x,))

    def test_reduce_with_masked(self):
        # https://github.com/pytorch/pytorch/issues/96484
        def fn(a, b):
            a = torch.nn.functional.pad(a, (0, -1))
            c = a + b
            return c.min(0).values

        a = torch.randn([2])
        b = torch.randn([2])
        self.common(fn, (a, b))

    def test_scalar_sign_with_min(self):
        # https://github.com/pytorch/pytorch/issues/101340
        def fn(a):
            t1 = torch.tanh(a)
            t2 = torch.sign(t1)
            return torch.min(t1, t2)

        a = torch.randn(1, 3)
        self.common(fn, (a,))

    def test_tanh_atan2(self):
        # https://github.com/pytorch/pytorch/issues/148241
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shrink = nn.Tanhshrink()

            def forward(self, x):
                x = self.shrink(x)
                x = torch.atan2(x, x)
                return x

        x = torch.randn(1, 3, 64, 64)
        self.common(Model(), (x,))

    @unittest.skipIf(
        os.getenv("ATEN_CPU_CAPABILITY") == "default",
        "Failing in periodic nogpu_NO_AVX2 after added in #152542",
    )
    @config.patch("cpp.use_decompose_tanh", "1")
    def test_tanh_atan2_use_decompose_tanh(self):
        # https://github.com/pytorch/pytorch/issues/148241
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shrink = nn.Tanhshrink()

            def forward(self, x):
                x = self.shrink(x)
                x = torch.atan2(x, x)
                return x

        x = torch.randn(1, 3, 64, 64)
        with self.assertRaises(AssertionError):
            self.common(Model(), (x,))

    def test_index_propagation_issue_102065(self):
        def fn(x):
            x = torch.arange(x.numel())
            return (x.unsqueeze(0) - x.unsqueeze(1)) ** 2

        self.common(
            fn,
            (torch.randn(8),),
        )

    def test_low_fp_index_expr_issue_147279(self):
        # https://github.com/pytorch/pytorch/issues/147279
        def fn(start, end, dtype, dim):
            return torch.sum(
                torch.arange(start=start, end=end, dtype=dtype),
                dim=dim,
            )

        self.common(
            fn,
            (300, 400, torch.float16, (0,)),
        )

    def test_index_put(self):
        # https://github.com/pytorch/pytorch/issues/138908
        def fn(x, y):
            x = x + 10
            y[x] += y[x]

        x = torch.randint(-10, -9, (1, 2), dtype=torch.int64)
        y = torch.randn((2, 32), dtype=torch.float32)
        x_clone = x.clone()
        y_clone = y.clone()
        with torch.no_grad():
            fn(x, y)
            torch.compile(fn)(x_clone, y_clone)
            self.assertEqual(y, y_clone, atol=1e-3, rtol=1e-3)

    def test_index_put2(self):
        # https://github.com/pytorch/pytorch/issues/138908
        def fn(y, index0, index1):
            y[index1] += y[index0]

        y = torch.randn((2, 32), dtype=torch.float32)
        index0 = torch.tensor([[0, 1]])
        index1 = torch.tensor([[1, 0]])
        y_clone = y.clone()
        index0_clone = index0.clone()
        index1_clone = index1.clone()
        with torch.no_grad():
            fn(y, index0, index1)
            torch.compile(fn)(y_clone, index0_clone, index1_clone)
            self.assertEqual(y, y_clone, atol=1e-3, rtol=1e-3)

    def test_index_add(self):
        # https://github.com/pytorch/pytorch/issues/138908
        def fn(x, y, scale_y, index):
            values = x[index] + y * scale_y
            out = x.index_add_(dim=0, source=values, index=index)
            return out

        inp = (
            torch.randn(10, 10),
            torch.randn(5, 10),
            torch.randn(10),
            torch.randperm(10, device="cpu")[:5].to(torch.int32),
        )
        inp_clones = []
        for i in range(3):
            inp_clones.append(
                [
                    inp[0].clone(),
                    inp[1].clone(),
                    inp[2].clone(),
                    inp[3].clone()
                    if i == 0
                    else torch.zeros(10, device="cpu")[:5].to(torch.int32),
                ]
            )
        inp_clone, inp_clone2, inp_clone3 = inp_clones
        with torch.no_grad():
            cfn = torch.compile(fn)
            ref = fn(*inp)
            res = cfn(*inp_clone)
            self.assertEqual(ref, res, atol=1e-3, rtol=1e-3)
            ref = fn(*inp_clone2)
            res = cfn(*inp_clone3)
            self.assertEqual(ref, res, atol=1e-3, rtol=1e-3)

    def test_ModularIndexing_range_issue_103133(self):
        def fn(q, k):
            einsum = torch.einsum("bcxd,bcyd->bcxy", (q, k))
            constant_pad_nd = torch.ops.aten.constant_pad_nd.default(
                einsum, [0, 0, 0, 1], 0.0
            )
            view = torch.ops.aten.view.default(constant_pad_nd, [12, 1, 512, 513])
            y = view.new_zeros((12, 2, 256, 513))
            y[:, :-1, :, 256:] = view[:, :, :256, :257]
            return y

        self.common(
            fn,
            (
                torch.empty_strided((12, 1, 512, 64), (64, 196608, 768, 1)),
                torch.empty_strided((12, 1, 512, 64), (64, 196608, 768, 1)),
            ),
        )

    @patch("torch.cuda.is_available", lambda: False)
    def test_max_reduction_lowp_fp(self):
        def fn(x):
            return torch.ops.aten.max(x, 1, keepdim=True)[0].float()

        for dtype in _lowp_fp_dtypes:
            self.common(
                fn,
                (torch.randn(1, 32, 4, 4).to(dtype),),
            )

    @patch("torch.cuda.is_available", lambda: False)
    def test_vec_transpose_lowp_fp(self):
        for dtype in _lowp_fp_dtypes:

            def fn(x):
                return x.to(memory_format=torch.channels_last).to(dtype)

            self.common(
                fn,
                (torch.randn(2, 3, 4, 4),),
            )

    def test_load_inf_bf16(self):
        def fn1(x):
            return torch.where(x > 0, x, math.inf)

        def fn2(x):
            return torch.where(x > 0, x, -math.inf)

        for fn in [fn1, fn2]:
            self.common(
                fn,
                (torch.randn(1, 3, 16, 16),),
            )

    @patch("torch.cuda.is_available", lambda: False)
    def test_fp32_load_with_to_lowp_fp(self):
        # From llama model.
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.cache_k = torch.zeros(8, 4, 2, 2)

            def forward(self, x, xk):
                bsz, seqlen, _ = x.shape
                self.cache_k = self.cache_k.to(x)
                self.cache_k[:bsz, 1 : 1 + seqlen] = xk
                return self.cache_k

        for dtype in _lowp_fp_dtypes:
            ref_model = Model().eval()
            opt_model = torch.compile()(Model().eval())
            x = torch.randn(4, 2, 2).to(dtype)
            xk = torch.randn(4, 2, 2, 2).to(dtype)
            self.assertEqual(opt_model(x, xk), ref_model(x, xk))

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_sigmoid_with_reduction(self):
        def fn(x):
            x = torch.ops.aten.sigmoid.default(x)
            return torch.ops.aten.mean.dim(x, [-1, -2], True)

        x = torch.randn((1, 8, 8, 8))
        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, (x,))

    def test_slice_scatter_default_end_value(self):
        # From HF AllenaiLongformerBase.
        def fn(query, key, window_overlap):
            batch_size, seq_len, num_heads, head_dim = query.size()
            assert seq_len % (window_overlap * 2) == 0, (
                f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"
            )

            chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
            diagonal_chunked_attention_scores = key
            diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
                (
                    batch_size * num_heads,
                    chunks_count + 1,
                    window_overlap,
                    window_overlap * 2 + 1,
                )
            )
            diagonal_attention_scores[:, :3, :, window_overlap:] = (
                diagonal_chunked_attention_scores[
                    :, :, :window_overlap, : window_overlap + 1
                ]
            )
            return diagonal_attention_scores

        self.common(
            fn,
            (
                torch.randn(1, 1024, 12, 64),
                torch.randn(12, 3, 512, 513),
                256,
            ),
        )

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_to_uint8_rounding_method(self):
        def fn(x):
            return x.to(torch.uint8)

        numerical_testsuit = [4.4, 4.5, 4.6, 5.5]
        for numerical_number in numerical_testsuit:
            x = torch.ones(17) * numerical_number
            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x,))
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    def _test_decomposed_dequant_relu_quant_helper(self, dtype):
        def fn(
            x, scale, zero_point, use_dequant, use_quant, quant_min, quant_max, dtype
        ):
            # For quantized_decomposed.dequantize_per_tensor
            # Refer to torch/ao/quantization/fx/_decomposed.py
            if use_dequant:
                x = (x.to(torch.float32) - zero_point) * scale

            x = torch.relu(x)

            # For quantized_decomposed.quantize_per_tensor
            # Refer to torch/ao/quantization/fx/_decomposed.py
            if use_quant:
                inv_scale = 1.0 / scale
                x = torch.clamp(
                    torch.round(x * inv_scale) + zero_point, quant_min, quant_max
                ).to(dtype)
            return x

        assert dtype in [torch.uint8, torch.int8]
        quant_min = 0 if dtype == torch.uint8 else -128
        quant_max = 255 if dtype == torch.uint8 else 127

        use_dequant_list = [False, True]
        use_quant_list = [False, True]
        for use_dequant, use_quant in itertools.product(
            use_dequant_list, use_quant_list
        ):
            x = torch.clamp(
                torch.randn((1, 7, 7, 9), dtype=torch.float32) * 100,
                quant_min,
                quant_max,
            )
            if use_dequant:
                x = x.to(dtype)
            zero_point = 100
            scale = 0.01
            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(
                    fn,
                    (
                        x,
                        scale,
                        zero_point,
                        use_dequant,
                        use_quant,
                        quant_min,
                        quant_max,
                        dtype,
                    ),
                )
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    def test_decomposed_dequant_relu_quant_uint8(self):
        self._test_decomposed_dequant_relu_quant_helper(torch.uint8)

    @requires_vectorization
    def test_decomposed_dequant_relu_quant_int8(self):
        self._test_decomposed_dequant_relu_quant_helper(torch.int8)

    def _test_dequant_quant_lowering_helper(self, dtype, dequant_out_dtype=None):
        def fn(
            x,
            scale,
            zero_point,
            use_dequant,
            use_quant,
            quant_min,
            quant_max,
            dtype,
            dequant_out_dtype,
        ):
            if use_dequant:
                x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                    x,
                    scale,
                    zero_point,
                    quant_min,
                    quant_max,
                    dtype,
                    out_dtype=dequant_out_dtype,
                )

            x = torch.relu(x)

            if use_quant:
                x = torch.ops.quantized_decomposed.quantize_per_tensor(
                    x, scale, zero_point, quant_min, quant_max, dtype
                )
            return x

        use_dequant_list = [False, True]
        use_quant_list = [False, True]
        use_tensor_overload_list = [False, True]

        assert dtype in [
            torch.uint8,
            torch.int8,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]
        quant_min = 0 if dtype == torch.uint8 else -128
        quant_max = 255 if dtype == torch.uint8 else 127
        if dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            quant_min = int(torch.finfo(dtype).min)
            quant_max = int(torch.finfo(dtype).max)
            use_tensor_overload_list = [
                False,
            ]

        for (
            use_dequant,
            use_quant,
            use_tensor_overload,
        ) in itertools.product(
            use_dequant_list,
            use_quant_list,
            use_tensor_overload_list,
        ):
            x = torch.clamp(
                torch.randn((1, 7, 7, 9), dtype=torch.float32) * 100,
                quant_min,
                quant_max,
            )
            if use_dequant:
                x = x.to(dtype)
            zero_point = 100
            scale = 0.01
            if use_tensor_overload:
                zero_point = torch.tensor(zero_point, dtype=torch.int64)
                scale = torch.tensor(scale)
            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                inputs = (
                    x,
                    scale,
                    zero_point,
                    use_dequant,
                    use_quant,
                    quant_min,
                    quant_max,
                    dtype,
                    dequant_out_dtype,
                )
                self.common(fn, inputs)
                check_metrics_vec_kernel_count(1)

                # Check that both main and tail loops are vectorized
                if dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    compiled_fn = torch.compile(fn)
                    _, code = run_and_get_cpp_code(compiled_fn, *inputs)
                    FileCheck().check_count("loadu", 2, exactly=True).run(code)

    @requires_vectorization
    def test_dequant_quant_lowering_uint8(self):
        self._test_dequant_quant_lowering_helper(torch.uint8)
        self._test_dequant_quant_lowering_helper(
            torch.uint8, dequant_out_dtype=torch.bfloat16
        )

    @requires_vectorization
    def test_dequant_quant_lowering_int8(self):
        self._test_dequant_quant_lowering_helper(torch.int8)
        self._test_dequant_quant_lowering_helper(
            torch.int8, dequant_out_dtype=torch.bfloat16
        )

    @requires_vectorization
    def test_dequant_quant_lowering_fp8_e4m3(self):
        self._test_dequant_quant_lowering_helper(torch.float8_e4m3fn)

    @requires_vectorization
    def test_dequant_quant_lowering_fp8_e5m2(self):
        self._test_dequant_quant_lowering_helper(torch.float8_e5m2)

    def _test_dequant_maxpool2d_lowering_helper(self, dtype):
        def fn(x, scale, zero_point, quant_min, quant_max, dtype):
            x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                x, scale, zero_point, quant_min, quant_max, dtype
            )
            max_pool2d_with_indices_default = (
                torch.ops.aten.max_pool2d_with_indices.default(
                    x, [2, 2], [2, 2], [1, 1]
                )[0]
            )
            return max_pool2d_with_indices_default

        assert dtype in [torch.uint8, torch.int8]
        quant_min = 0 if dtype == torch.uint8 else -128
        quant_max = 255 if dtype == torch.uint8 else 127

        use_tensor_overload_list = [False, True]
        for use_tensor_overload in use_tensor_overload_list:
            x = (
                torch.clamp(
                    torch.randn((3, 16, 8, 8), dtype=torch.float32) * 100,
                    quant_min,
                    quant_max,
                )
                .to(dtype)
                .contiguous(memory_format=torch.channels_last)
            )
            zero_point = 100
            scale = 0.01
            if use_tensor_overload:
                zero_point = torch.tensor(zero_point, dtype=torch.int64)
                scale = torch.tensor(scale)
            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x, scale, zero_point, quant_min, quant_max, dtype))
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    def test_dequant_maxpool2d_lowering_uint8(self):
        self._test_dequant_maxpool2d_lowering_helper(torch.uint8)

    @requires_vectorization
    def test_dequant_maxpool2d_lowering_int8(self):
        self._test_dequant_maxpool2d_lowering_helper(torch.int8)

    def _test_tile2d_load_decomposed_dequant_add_relu_quant_helper(self, dtype):
        def fn(
            x,
            scale,
            zero_point,
            x2,
            scale2,
            zero_point2,
            output_scale,
            output_zero_point,
            use_dequant,
            use_dequant2,
            use_quant,
            quant_min,
            quant_max,
            dtype,
        ):
            if use_dequant:
                x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                    x, scale, zero_point, quant_min, quant_max, dtype
                )
            if use_dequant2:
                x2 = torch.ops.quantized_decomposed.dequantize_per_tensor(
                    x2, scale2, zero_point2, quant_min, quant_max, dtype
                )
            temp = x + x2
            y = torch.relu(temp)

            if use_quant:
                y = torch.ops.quantized_decomposed.quantize_per_tensor(
                    y, output_scale, output_zero_point, quant_min, quant_max, dtype
                )
            return y.contiguous()

        assert dtype in [torch.uint8, torch.int8]
        quant_min = 0 if dtype == torch.uint8 else -128
        quant_max = 255 if dtype == torch.uint8 else 127

        use_dequant_list = [False, True]
        use_dequant_list2 = [False, True]
        use_quant_list = [False, True]

        for use_dequant, use_dequant2, use_quant in itertools.product(
            use_dequant_list, use_dequant_list2, use_quant_list
        ):
            x = torch.clamp(
                torch.randn((1, 1024, 14, 14), dtype=torch.float32) * 100,
                quant_min,
                quant_max,
            ).contiguous(memory_format=torch.channels_last)
            x2 = torch.clamp(
                torch.randn((1, 1024, 14, 14), dtype=torch.float32) * 100,
                quant_min,
                quant_max,
            ).contiguous(memory_format=torch.channels_last)
            if use_dequant:
                x = x.to(dtype).contiguous(memory_format=torch.channels_last)
            if use_dequant2:
                x2 = x2.to(dtype).contiguous(memory_format=torch.channels_last)
            zero_point = 1
            scale = 0.01
            zero_point2 = 2
            scale2 = 0.02
            output_zero_point = 3
            output_scale = 0.03
            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(
                    fn,
                    (
                        x,
                        scale,
                        zero_point,
                        x2,
                        scale2,
                        zero_point2,
                        output_scale,
                        output_zero_point,
                        use_dequant,
                        use_dequant2,
                        use_quant,
                        quant_min,
                        quant_max,
                        dtype,
                    ),
                )
                check_metrics_vec_kernel_count(2)

    @requires_vectorization
    def test_tile2d_load_decomposed_dequant_add_relu_quant_uint8(self):
        self._test_tile2d_load_decomposed_dequant_add_relu_quant_helper(torch.uint8)

    @requires_vectorization
    def test_tile2d_load_decomposed_dequant_add_relu_quant_int8(self):
        self._test_tile2d_load_decomposed_dequant_add_relu_quant_helper(torch.int8)

    @requires_vectorization
    def _test_per_tensor_fake_quant_helper(self, dtype):
        def fn(input, scales, zero_points, quant_min, quant_max, dtype):
            input = torch.ops.quantized_decomposed.quantize_per_tensor(
                input, scales, zero_points, quant_min, quant_max, dtype
            )
            input = torch.ops.quantized_decomposed.dequantize_per_tensor(
                input, scales, zero_points, quant_min, quant_max, dtype
            )
            return input

        use_tensor_overload_list = [False, True]
        for use_tensor_overload in use_tensor_overload_list:
            assert dtype in [torch.uint8, torch.int8]
            quant_min = 0 if dtype == torch.uint8 else -128
            quant_max = 255 if dtype == torch.uint8 else 127
            x = torch.clamp(
                torch.randn((1, 7, 7, 9), dtype=torch.float32) * 100,
                quant_min,
                quant_max,
            )
            zero_point = 100
            scale = 0.01
            if use_tensor_overload:
                zero_point = torch.tensor(zero_point, dtype=torch.int64)
                scale = torch.tensor(scale)
            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x, scale, zero_point, quant_min, quant_max, dtype))
                assert metrics.generated_cpp_vec_kernel_count == 1

    @requires_vectorization
    def test_per_tensor_fake_quant_uint8(self):
        self._test_per_tensor_fake_quant_helper(torch.uint8)

    @requires_vectorization
    def test_per_tensor_fake_quant_int8(self):
        self._test_per_tensor_fake_quant_helper(torch.int8)

    def _test_per_channel_fake_quant_helper(
        self, dtype, input_dtype=torch.float32, output_dtype=None
    ):
        def fn(
            input, scales, zero_points, axis, quant_min, quant_max, dtype, output_dtype
        ):
            input = torch.ops.quantized_decomposed.quantize_per_channel(
                input, scales, zero_points, axis, quant_min, quant_max, dtype
            )
            input = torch.ops.quantized_decomposed.dequantize_per_channel(
                input,
                scales,
                zero_points,
                axis,
                quant_min,
                quant_max,
                dtype,
                out_dtype=output_dtype,
            )
            return input

        assert dtype in [torch.uint8, torch.int8]
        quant_min = 0 if dtype == torch.uint8 else -128
        quant_max = 255 if dtype == torch.uint8 else 127
        x = torch.clamp(
            torch.randn((1, 3, 224, 224), dtype=torch.float32) * 100,
            quant_min,
            quant_max,
        )
        if input_dtype != torch.float32:
            x = x.to(dtype=input_dtype)
        scales = torch.ones((3,))
        zero_points = torch.zeros((3,))
        axis = 1
        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(
                fn,
                (
                    x,
                    scales,
                    zero_points,
                    axis,
                    quant_min,
                    quant_max,
                    dtype,
                    output_dtype,
                ),
            )
            check_metrics_vec_kernel_count(1)

    @requires_vectorization
    def test_per_channel_fake_quant_uint8(self):
        self._test_per_channel_fake_quant_helper(torch.uint8)

    @requires_vectorization
    def test_per_channel_fake_quant_module_uint8(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scales = torch.ones((3,)).to(torch.float64)
                self.zero_points = torch.zeros((3,)).to(torch.int64)
                self.axis = 1
                self.quant_min = 0
                self.quant_max = 255
                self.dtype = torch.uint8

            def forward(self, input):
                input = torch.ops.quantized_decomposed.quantize_per_channel(
                    input,
                    self.scales,
                    self.zero_points,
                    self.axis,
                    self.quant_min,
                    self.quant_max,
                    self.dtype,
                )
                input = torch.ops.quantized_decomposed.dequantize_per_channel(
                    input,
                    self.scales,
                    self.zero_points,
                    self.axis,
                    self.quant_min,
                    self.quant_max,
                    self.dtype,
                )
                return input

        m = Mod().eval()
        x = torch.clamp(
            torch.randn((1, 3, 224, 224), dtype=torch.float32) * 100,
            0,
            255,
        )
        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(m, (x,))
            assert metrics.generated_cpp_vec_kernel_count == 1

    @requires_vectorization
    def test_per_channel_fake_quant_int8(self):
        self._test_per_channel_fake_quant_helper(torch.int8)

    @requires_vectorization
    def test_per_channel_fake_quant_uint8_bf16_input(self):
        self._test_per_channel_fake_quant_helper(
            torch.uint8, input_dtype=torch.bfloat16
        )
        self._test_per_channel_fake_quant_helper(
            torch.uint8, input_dtype=torch.bfloat16, output_dtype=torch.bfloat16
        )

    @requires_vectorization
    def test_per_channel_fake_quant_int8_bf16_input(self):
        self._test_per_channel_fake_quant_helper(torch.int8, input_dtype=torch.bfloat16)
        self._test_per_channel_fake_quant_helper(
            torch.int8, input_dtype=torch.bfloat16, output_dtype=torch.bfloat16
        )

    def _test_non_contiguous_load_buf_quant_helper(self, dtype):
        def fn(
            x1,
            x2,
            groups,
            quant_min,
            quant_max,
            dtype,
        ):
            x = torch.cat((x1, x2), dim=1)
            batchsize, num_channels, height, width = x.size()
            channels_per_group = num_channels // groups
            x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                x, 1.0, 0, quant_min, quant_max, dtype
            )
            x = x.view(batchsize, groups, channels_per_group, height, width)
            x = torch.ops.quantized_decomposed.quantize_per_tensor(
                x, 1.0, 0, quant_min, quant_max, dtype
            )
            x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                x, 1.0, 0, quant_min, quant_max, dtype
            )
            x = torch.transpose(x, 1, 2).contiguous()
            x = x.view(batchsize, num_channels, height, width)
            return x

        assert dtype in [torch.uint8, torch.int8]
        quant_min = 0 if dtype == torch.uint8 else -128
        quant_max = 255 if dtype == torch.uint8 else 127

        x = torch.randint(0, 8, (1, 116, 28, 28), dtype=dtype).contiguous(
            memory_format=torch.channels_last
        )
        x2 = torch.randint(0, 8, (1, 116, 28, 28), dtype=dtype).contiguous(
            memory_format=torch.channels_last
        )

        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(
                fn,
                (
                    x,
                    x2,
                    2,
                    quant_min,
                    quant_max,
                    dtype,
                ),
            )
            check_metrics_vec_kernel_count(2)

    @requires_vectorization
    def test_non_contiguous_load_buf_quant_uint8(self):
        self._test_non_contiguous_load_buf_quant_helper(torch.uint8)

    @requires_vectorization
    def test_non_contiguous_load_buf_quant_int8(self):
        self._test_non_contiguous_load_buf_quant_helper(torch.int8)

    def _test_tile2d_store_channel_shuffle_cl_quant_output_helper(self, dtype):
        def channel_shuffle(
            x, groups, output_scale, output_zero_point, quant_min, quant_max, dtype
        ):
            batchsize, num_channels, height, width = x.size()
            channels_per_group = num_channels // groups
            x = x.view(batchsize, groups, channels_per_group, height, width)
            x = torch.transpose(x, 1, 2).contiguous()
            x = x.view(batchsize, -1, height, width)
            x = torch.ops.quantized_decomposed.quantize_per_tensor(
                x, output_scale, output_zero_point, quant_min, quant_max, dtype
            )
            return x.contiguous(memory_format=torch.channels_last)

        assert dtype in [torch.uint8, torch.int8]
        quant_min = 0 if dtype == torch.uint8 else -128
        quant_max = 255 if dtype == torch.uint8 else 127

        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            x = torch.randn(64, 58, 28, 28)
            output_zero_point = 3
            output_scale = 0.03
            self.common(
                channel_shuffle,
                (x, 2, output_scale, output_zero_point, quant_min, quant_max, dtype),
            )
            check_metrics_vec_kernel_count(2)

    @requires_vectorization
    def test_tile2d_store_channel_shuffle_cl_quant_output_uint8(self):
        self._test_tile2d_store_channel_shuffle_cl_quant_output_helper(torch.uint8)

    @requires_vectorization
    def test_tile2d_store_channel_shuffle_cl_quant_output_int8(self):
        self._test_tile2d_store_channel_shuffle_cl_quant_output_helper(torch.int8)

    @requires_vectorization
    def test_to_channels_last_fp8(self):
        def fn(x):
            return x.to(memory_format=torch.channels_last)

        for dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            torch._dynamo.reset()
            metrics.reset()
            self.common(
                fn,
                (torch.randn(20, 16, 48, 48).to(dtype=dtype),),
            )
            check_metrics_vec_kernel_count(2)

    def _test_dequant_relu_quant_dequant_relu_quant_lowering_helper(self, dtype):
        def fn(
            x,
            scale,
            zero_point,
            scale2,
            zero_point2,
            scale3,
            zero_point3,
            quant_min,
            quant_max,
            dtype,
        ):
            x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                x, scale, zero_point, quant_min, quant_max, dtype
            )
            x = torch.relu(x)
            x = torch.ops.quantized_decomposed.quantize_per_tensor(
                x, scale2, zero_point2, quant_min, quant_max, dtype
            )
            x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                x, scale2, zero_point2, quant_min, quant_max, dtype
            )
            x = torch.relu(x)
            x = torch.ops.quantized_decomposed.quantize_per_tensor(
                x, scale3, zero_point3, quant_min, quant_max, dtype
            )
            return x

        assert dtype in [torch.uint8, torch.int8]
        quant_min = 0 if dtype == torch.uint8 else -128
        quant_max = 255 if dtype == torch.uint8 else 127

        for use_tensor_overload in [True, False]:
            x = torch.clamp(
                torch.randn((1, 7, 7, 9), dtype=torch.float32) * 100,
                quant_min,
                quant_max,
            ).to(dtype)
            zero_point_list = [100, 101, 102]
            scale_list = [0.01, 0.02, 0.03]
            if use_tensor_overload:
                for i in range(len(zero_point_list)):
                    zero_point_list[i] = torch.tensor(
                        zero_point_list[i], dtype=torch.int64
                    )
                    scale_list[i] = torch.tensor(scale_list[i])
            zero_point, zero_point2, zero_point3 = zero_point_list
            scale, scale2, scale3 = scale_list
            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(
                    fn,
                    (
                        x,
                        scale,
                        zero_point,
                        scale2,
                        zero_point2,
                        scale3,
                        zero_point3,
                        quant_min,
                        quant_max,
                        dtype,
                    ),
                    rtol=1e-2,
                    atol=1e-2,
                )
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    def test_dequant_relu_quant_dequant_relu_quant_lowering_uint8(self):
        self._test_dequant_relu_quant_dequant_relu_quant_lowering_helper(torch.uint8)

    @requires_vectorization
    def test_dequant_relu_quant_dequant_relu_quant_lowering_int8(self):
        self._test_dequant_relu_quant_dequant_relu_quant_lowering_helper(torch.int8)

    def test_inplace_add_alpha(self):
        def fn(x, y):
            aten.add_.Tensor(x, y, alpha=0.55)
            return (x,)

        x1 = torch.zeros(10)
        x2 = torch.zeros(10)
        x3 = torch.zeros(10)
        y = torch.randn(10)
        fn_fx = make_fx(fn)(x1, y)
        fn_compiled = compile_fx_inner(fn_fx, [x1, y])
        fn(x2, y)
        fn_compiled([x3, y])
        assert same(x2, x3)

    def test_int_div(self):
        def fn(x, y):
            s3 = x.size(1)
            a = torch.ones((1 + s3) // 2)
            a += y
            return a, s3

        p0 = torch.randint(5, (1, 8))
        p1 = torch.randn(1)
        self.common(fn, (p0, p1))

    def test_no_op_squeeze(self):
        @torch.compile(backend="inductor")
        def forward(arg0_1):
            return torch.ops.aten.squeeze.dim(arg0_1, 1)

        x = torch.randn((10, 20))
        self.common(forward, (x,))

    def test_parallel_num_threads(self):
        @torch.compile(backend="inductor")
        def fn(x1, x2):
            return x1 + x2

        x1 = torch.randn((10, 20))
        x2 = torch.randn((10, 20))
        with set_num_threads(1):
            assert same(x1 + x2, fn(x1, x2))
        with set_num_threads(4):
            assert same(x1 + x2, fn(x1, x2))

    @patch("torch.cuda.is_available", lambda: False)
    def test_timed_cpu_only(self):
        timed(lambda: torch.randn(10), ())

    def test_complex_memory_overlap(self):
        dense = torch.zeros(64, 32)
        self.assertFalse(complex_memory_overlap(dense))
        self.assertFalse(complex_memory_overlap(dense.t()))

        strided = dense.split(4, dim=1)
        self.assertFalse(complex_memory_overlap(strided[0]))
        self.assertFalse(complex_memory_overlap(strided[0].t()))

        unsqueezed = dense.unsqueeze(1)
        self.assertFalse(complex_memory_overlap(unsqueezed))
        self.assertFalse(complex_memory_overlap(unsqueezed.permute(1, 2, 0)))

        gathered = dense.index_select(0, torch.IntTensor([1, 0, 1]))
        self.assertFalse(complex_memory_overlap(gathered))
        self.assertFalse(complex_memory_overlap(gathered.t()))

    @requires_vectorization
    def test_vec_dynamic_shapes(self):
        def fn(x):
            return torch.softmax(x, -1)

        value = torch.randn((2, 10))
        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, (value,))

    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    @unittest.skipIf(
        not cpu_vec_isa.valid_vec_isa_list()
        or "avx2" in [str(vec_isa) for vec_isa in cpu_vec_isa.valid_vec_isa_list()]
        or "asimd" in [str(vec_isa) for vec_isa in cpu_vec_isa.valid_vec_isa_list()],
        "Does not support vectorization or not s390x/ppc64le machine",
    )
    @patch("torch.cuda.is_available", lambda: False)
    def test_auto_zvec_vsx_simd(self):
        vec_zvec_vsx = cpu_vec_isa.valid_vec_isa_list()[0]
        self.assertTrue(vec_zvec_vsx.bit_width() == 256)

        with config.patch({"cpp.simdlen": 0}):
            isa = cpu_vec_isa.pick_vec_isa()
            self.assertFalse(isa)

        with config.patch({"cpp.simdlen": 1}):
            isa = cpu_vec_isa.pick_vec_isa()
            self.assertFalse(isa)

        with config.patch({"cpp.simdlen": 257}):
            isa = cpu_vec_isa.pick_vec_isa()
            self.assertFalse(isa)

        with config.patch({"cpp.simdlen": 256}):
            isa = cpu_vec_isa.pick_vec_isa()
            self.assertTrue(isa == vec_zvec_vsx)

        pre_var = os.getenv("ATEN_CPU_CAPABILITY")
        if pre_var:
            os.environ.pop("ATEN_CPU_CAPABILITY")

        try:
            with config.patch({"cpp.simdlen": None}):
                isa = cpu_vec_isa.pick_vec_isa()
                self.assertTrue(isa == vec_zvec_vsx)

            with config.patch({"cpp.simdlen": None}):
                os.environ["ATEN_CPU_CAPABILITY"] = "avx2"
                isa = cpu_vec_isa.pick_vec_isa()
                self.assertTrue(isa == vec_zvec_vsx)

            with config.patch({"cpp.simdlen": None}):
                os.environ["ATEN_CPU_CAPABILITY"] = "avx512"
                isa = cpu_vec_isa.pick_vec_isa()
                self.assertTrue(isa == vec_zvec_vsx)

            with config.patch({"cpp.simdlen": None}):
                os.environ["ATEN_CPU_CAPABILITY"] = "default"
                isa = cpu_vec_isa.pick_vec_isa()
                self.assertFalse(isa)

            with config.patch({"cpp.simdlen": None}):
                os.environ["ATEN_CPU_CAPABILITY"] = "zvector"
                isa = cpu_vec_isa.pick_vec_isa()
                self.assertTrue(isa == vec_zvec_vsx)

            with config.patch({"cpp.simdlen": None}):
                os.environ["ATEN_CPU_CAPABILITY"] = "vsx"
                isa = cpu_vec_isa.pick_vec_isa()
                self.assertTrue(isa == vec_zvec_vsx)

        finally:
            if pre_var:
                os.environ["ATEN_CPU_CAPABILITY"] = pre_var
            elif os.getenv("ATEN_CPU_CAPABILITY"):
                os.environ.pop("ATEN_CPU_CAPABILITY")

    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    @unittest.skipIf(
        platform.machine() != "x86_64" or not cpu_vec_isa.valid_vec_isa_list(),
        "Does not support vectorization or not x86_64 machine",
    )
    @patch("torch.cuda.is_available", lambda: False)
    def test_auto_simd(self):
        vec_amx = cpu_vec_isa.supported_vec_isa_list[0]
        vec_avx512_vnni = cpu_vec_isa.supported_vec_isa_list[1]
        vec_avx512 = cpu_vec_isa.supported_vec_isa_list[2]
        vec_avx2 = cpu_vec_isa.supported_vec_isa_list[3]
        self.assertTrue(vec_amx.bit_width() == 512)
        self.assertTrue(vec_amx.nelements() == 16)
        self.assertTrue(vec_amx.nelements(torch.bfloat16) == 32)
        self.assertTrue(vec_avx512_vnni.bit_width() == 512)
        self.assertTrue(vec_avx512_vnni.nelements(torch.int8) == 64)
        self.assertTrue(vec_avx512_vnni.nelements(torch.uint8) == 64)
        self.assertTrue(vec_avx512.bit_width() == 512)
        self.assertTrue(vec_avx2.bit_width() == 256)
        self.assertTrue(vec_avx512.nelements() == 16)
        self.assertTrue(vec_avx2.nelements() == 8)
        self.assertTrue(vec_avx512.nelements(torch.bfloat16) == 32)
        self.assertTrue(vec_avx2.nelements(torch.bfloat16) == 16)

        with config.patch({"cpp.simdlen": 0}):
            isa = cpu_vec_isa.pick_vec_isa()
            self.assertFalse(isa)

        with config.patch({"cpp.simdlen": 1}):
            isa = cpu_vec_isa.pick_vec_isa()
            self.assertFalse(isa)

        with config.patch({"cpp.simdlen": 257}):
            isa = cpu_vec_isa.pick_vec_isa()
            self.assertFalse(isa)

        with config.patch({"cpp.simdlen": 513}):
            isa_list = cpu_vec_isa.valid_vec_isa_list()
            if vec_avx512 in isa_list:
                self.assertFalse(isa)

        with config.patch({"cpp.simdlen": 512}):
            isa_list = cpu_vec_isa.valid_vec_isa_list()
            isa = cpu_vec_isa.pick_vec_isa()
            if vec_amx in isa_list:
                self.assertTrue(isa == vec_amx)
            elif vec_avx512_vnni in isa_list:
                self.assertTrue(isa == vec_avx512_vnni)
            elif vec_avx512 in isa_list:
                self.assertTrue(isa == vec_avx512)

        with config.patch({"cpp.simdlen": 256}):
            isa_list = cpu_vec_isa.valid_vec_isa_list()
            if vec_avx2 in isa_list:
                isa = cpu_vec_isa.pick_vec_isa()
                self.assertTrue(isa == vec_avx2)

        pre_var = os.getenv("ATEN_CPU_CAPABILITY")
        if pre_var:
            os.environ.pop("ATEN_CPU_CAPABILITY")

        try:
            with config.patch({"cpp.simdlen": None}):
                isa = cpu_vec_isa.pick_vec_isa()
                if vec_amx in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_amx)
                elif vec_avx512_vnni in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_avx512_vnni)
                elif vec_avx512 in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_avx512)
                else:
                    self.assertTrue(isa == vec_avx2)

            with config.patch({"cpp.simdlen": None}):
                os.environ["ATEN_CPU_CAPABILITY"] = "avx2"
                isa = cpu_vec_isa.pick_vec_isa()
                if vec_amx in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_avx2)
                if vec_avx512_vnni in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_avx2)
                elif vec_avx512 in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_avx2)
                elif vec_avx2 in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_avx2)

            with config.patch({"cpp.simdlen": None}):
                os.environ["ATEN_CPU_CAPABILITY"] = "avx512"
                isa = cpu_vec_isa.pick_vec_isa()
                if vec_amx in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_amx)
                elif vec_avx512_vnni in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_avx512_vnni)
                elif vec_avx512 in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_avx512)
                else:
                    self.assertTrue(isa == vec_avx2)

            with config.patch({"cpp.simdlen": None}):
                os.environ["ATEN_CPU_CAPABILITY"] = "default"
                isa = cpu_vec_isa.pick_vec_isa()
                self.assertFalse(isa)

            with config.patch({"cpp.simdlen": None}):
                os.environ["ATEN_CPU_CAPABILITY"] = "zvector"
                isa = cpu_vec_isa.pick_vec_isa()
                if vec_amx in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_amx)
                elif vec_avx512_vnni in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_avx512_vnni)
                elif vec_avx512 in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_avx512)
                else:
                    self.assertTrue(isa == vec_avx2)

            with config.patch({"cpp.simdlen": None}):
                os.environ["ATEN_CPU_CAPABILITY"] = "vsx"
                isa = cpu_vec_isa.pick_vec_isa()
                if vec_amx in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_amx)
                elif vec_avx512_vnni in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_avx512_vnni)
                elif vec_avx512 in cpu_vec_isa.valid_vec_isa_list():
                    self.assertTrue(isa == vec_avx512)
                else:
                    self.assertTrue(isa == vec_avx2)

        finally:
            if pre_var:
                os.environ["ATEN_CPU_CAPABILITY"] = pre_var
            elif os.getenv("ATEN_CPU_CAPABILITY"):
                os.environ.pop("ATEN_CPU_CAPABILITY")

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_masked_fill_softmax(self):
        def fn(value, mask):
            mask = mask.to(torch.bool)
            x = torch.masked_fill(value, mask, -33.0)
            return torch.softmax(x, -1)

        for dtype in vec_dtypes:
            value = torch.randn((2, 17), dtype=dtype)
            mask = torch.randint(0, 1, size=(2, 17), dtype=torch.uint8)
            with config.patch({"cpp.simdlen": None}):
                for cpp_wrapper_flag in [True, False]:
                    with config.patch({"cpp_wrapper": cpp_wrapper_flag}):
                        torch._dynamo.reset()
                        metrics.reset()
                        self.common(fn, (value, mask))
                        assert metrics.generated_cpp_vec_kernel_count >= 1

    def test_channels_last_view_as_complex(self):
        # https://github.com/pytorch/pytorch/issues/122448#issuecomment-2046169554

        def reduce_example(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Applies the rotary embedding to the query and key tensors."""
            x_out = torch.view_as_complex(torch.stack([x.float(), y.float()], dim=-1))
            return x_out

        args = [torch.randn(1, 1, 1, 128), torch.randn(1, 1, 1, 128)]
        expected = reduce_example(*args)
        actual = torch.compile(reduce_example, fullgraph=True)(*args)
        self.assertEqual(expected, actual)

    def test_load_same_bool_tensor_twice(self):
        @torch.compile(backend="inductor")
        def fn(a, b):
            x = torch.masked_fill(a, b, -33.0)
            y = torch.masked_fill(a, b, -33.0)
            return x, y

        value = torch.randn((2, 17))
        mask = torch.randint(0, 1, size=(2, 17), dtype=torch.uint8).to(torch.bool)
        fn(value, mask)

    def test_cpu_vec_cosim(self):
        cpp_vec_op_list = []
        cpp_op_list = []

        for k, v in CppVecOverrides.__dict__.items():
            if isinstance(v, staticmethod):
                cpp_vec_op_list.append(k)
        for k, v in CppOverrides.__dict__.items():
            if isinstance(v, staticmethod):
                cpp_op_list.append(k)

        diff = [
            "airy_ai",
            "bessel_j0",
            "bessel_j1",
            "bessel_y0",
            "bessel_y1",
            "modified_bessel_i0",
            "modified_bessel_i1",
            "modified_bessel_k0",
            "modified_bessel_k1",
            "scaled_modified_bessel_k0",
            "scaled_modified_bessel_k1",
            "spherical_bessel_j0",
            "i1",
            "i1e",
            "ndtr",
            "ndtri",
            "log_ndtr",
            "erfcx",
            "gammainc",
            "gammaincc",
            "igamma",
            "igammac",
            "polygamma",
            "zeta",
            "shifted_chebyshev_polynomial_u",
            "chebyshev_polynomial_u",
            "chebyshev_polynomial_t",
            "shifted_chebyshev_polynomial_w",
            "chebyshev_polynomial_w",
            "shifted_chebyshev_polynomial_t",
            "chebyshev_polynomial_v",
            "shifted_chebyshev_polynomial_v",
            "hermite_polynomial_he",
            "laguerre_polynomial_l",
            "hermite_polynomial_h",
            "legendre_polynomial_p",
            "constant",
            "index_expr",
            "signbit",
            "isinf",
            "frexp",
            "mod",
            "masked",
            "randn",
            "isnan",
            "rand",
            "randint64",
            "logical_and",
            "logical_not",
            "logical_or",
            "logical_xor",
            "bitwise_and",
            "bitwise_left_shift",
            "bitwise_not",
            "bitwise_right_shift",
            "bitwise_or",
            "bitwise_xor",
            "to_dtype_bitcast",
        ]
        union = {*cpp_vec_op_list, *diff}
        self.assertTrue(
            set(cpp_op_list).issubset(union), f"unexpected: {set(cpp_op_list) - union}"
        )

    def test_atomic_add_lowp_fp(self):
        def fn(test_args):
            res = torch.gather(**test_args)
            return res

        for dtype in _lowp_fp_dtypes:
            input_tensor_for_ref = torch.tensor(
                [[3.0, -5.0]], dtype=dtype, requires_grad=True
            )
            input_tensor_for_opt = torch.tensor(
                [[3.0, -5.0]], dtype=dtype, requires_grad=True
            )

            test_args_for_ref = {
                "input": input_tensor_for_ref,
                "dim": 1,
                "index": torch.tensor([[1]]),
            }
            test_args_for_opt = {
                "input": input_tensor_for_opt,
                "dim": 1,
                "index": torch.tensor([[1]]),
            }

            opt_fn = torch.compile(fn)

            ref_fwd = fn(test_args_for_ref)
            res_fwd = opt_fn(test_args_for_opt)
            self.assertEqual(res_fwd, ref_fwd)

            torch.manual_seed(1)
            bwd_tensor_for_ref = torch.randn(ref_fwd.shape, dtype=dtype)
            torch.manual_seed(1)
            bwd_tensor_for_opt = torch.randn(res_fwd.shape, dtype=dtype)
            self.assertEqual(bwd_tensor_for_ref, bwd_tensor_for_opt)

            ref_fwd.backward(bwd_tensor_for_ref)
            res_fwd.backward(bwd_tensor_for_opt)

            ref_grad = test_args_for_ref["input"].grad
            res_grad = test_args_for_opt["input"].grad
            self.assertEqual(ref_grad, res_grad)

    def test_meta_device(self):
        @torch.compile(fullgraph=True)
        def fn():
            x = torch.ops.aten.empty.memory_format(
                [1024, 128, 128],
                dtype=torch.float16,
                device="meta",
                pin_memory=False,
            )
            return x.sin() + 1

        self.assertEqual(fn().shape, [1024, 128, 128])

    def test_decomposed_fake_quant_per_channel(self):
        def fq(input, scales, zero_points, axis, quant_min, quant_max):
            res = torch.fake_quantize_per_channel_affine(
                input, scales, zero_points, axis, quant_min, quant_max
            )
            return res

        def qdq(input, scales, zero_points, axis, quant_min, quant_max):
            res = torch.ops.quantized_decomposed.fake_quant_per_channel(
                input, scales, zero_points, axis, quant_min, quant_max
            )
            return res

        def run_eager_aten_fake_quant(
            input, scales, zero_points, axis, quant_min, quant_max
        ):
            input.grad = None
            res = fq(input, scales, zero_points, axis, quant_min, quant_max)
            res.sum().backward()
            return res, input.grad

        def run_eager_decomposed_fake_quant(
            input, scales, zero_points, axis, quant_min, quant_max
        ):
            input.grad = None
            res = qdq(input, scales, zero_points, axis, quant_min, quant_max)
            res.sum().backward()
            return res, input.grad

        def run_compile_decomposed_fake_quant(
            input, scales, zero_points, axis, quant_min, quant_max
        ):
            input.grad = None
            compiled_qdq = torch.compile(qdq)
            res = compiled_qdq(input, scales, zero_points, axis, quant_min, quant_max)
            res.sum().backward()
            return res, input.grad

        input = torch.randn(2, 3, 224, 224)
        input[1, 2, 3, 4] = 257
        input.requires_grad_()
        scales = torch.ones((3,))
        zero_points = torch.zeros((3,))
        axis = 1
        quant_min = -128
        quant_max = 127

        aten_input = copy.deepcopy(input)
        compiler_input = copy.deepcopy(input)

        res_aten_eager, input_grad_aten_eager = run_eager_aten_fake_quant(
            aten_input, scales, zero_points, axis, quant_min, quant_max
        )
        res_decomp_eager, input_grad_decomp_eager = run_eager_decomposed_fake_quant(
            input, scales, zero_points, axis, quant_min, quant_max
        )
        res, input_grad = run_compile_decomposed_fake_quant(
            compiler_input, scales, zero_points, axis, quant_min, quant_max
        )

        self.assertEqual(res_aten_eager, res)
        self.assertEqual(res_decomp_eager, res)
        self.assertEqual(input_grad_aten_eager, input_grad)
        self.assertEqual(input_grad_decomp_eager, input_grad)
        self.assertEqual(input_grad[1, 2, 3, 4], torch.tensor(0.0))
        # For forward and backward kernel
        check_metrics_vec_kernel_count(2)

    @requires_vectorization
    def test_ops_masked_with_bool_input(self):
        x = torch.zeros(129, dtype=torch.bool)
        size = [2, 3]
        res_aten_eager = torch.constant_pad_nd(x, size)
        cfn = torch.compile(torch.constant_pad_nd)
        res = cfn(x, size)
        self.assertEqual(res_aten_eager, res)
        check_metrics_vec_kernel_count(1)

    @requires_vectorization
    def test_frexp(self):
        def fn(x):
            x_frac, x_exp = torch.frexp(x)  # x_frac: int32, x_exp: float32
            x = x_frac * x_exp
            return x

        x = torch.randn(64, 1)
        torch._dynamo.reset()
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

    def test_bitwise_right_shift(self):
        x = torch.randint(-1, 0, (1, 1, 1), device="cpu", dtype=torch.int64)
        bit_num = 31
        res_aten_eager = torch.bitwise_right_shift(x, bit_num)
        cfn = torch.compile(torch.bitwise_right_shift)
        res = cfn(x, bit_num)
        self.assertEqual(res_aten_eager, res)

    def test_bitwise_shift_corner_inputs(self):
        # Fix https://github.com/pytorch/pytorch/issues/143555
        # and https://github.com/pytorch/pytorch/issues/143566
        bitwise_fns = (
            torch.bitwise_left_shift,
            torch.bitwise_right_shift,
        )
        for bitwise_fn in bitwise_fns:
            torch._dynamo.reset()
            metrics.reset()
            x = torch.tensor(1000, dtype=torch.int64)
            bit_num = torch.tensor(64, dtype=torch.int64)
            res_aten_eager = bitwise_fn(x, bit_num)
            cfn = torch.compile(bitwise_fn)
            res = cfn(x, bit_num)
            self.assertEqual(res_aten_eager, res)

    def test_view_dtype(self):
        def f(x):
            return x.view(torch.int32) >> 2

        input = torch.ones(16, 16)
        res_aten_eager = f(input)
        cfn = torch.compile(f)
        res = cfn(input)
        self.assertEqual(res_aten_eager, res)

    @patch("torch.cuda.is_available", lambda: False)
    def test_scatter_using_atomic_add(self):
        def fn(a, dim, index, b):
            return aten.scatter(a, dim, index, b, reduce="add")

        inps = (
            torch.randn(5, 29, 13),
            2,
            torch.tensor([[[3, 5, 7, 9]]]),
            torch.randn(1, 1, 10),
        )

        def _internal_check(
            _fn,
            _inps,
            _target_code_check=None,
            _target_code_check_not=None,
        ):
            torch._dynamo.reset()
            metrics.reset()
            _fn_opt = torch.compile()(_fn)
            _, code = run_and_get_cpp_code(_fn_opt, *inps)
            if _target_code_check:
                FileCheck().check(_target_code_check).run(code)
            if _target_code_check_not:
                FileCheck().check_not(_target_code_check_not).run(code)
                # Verify that the output isn't empty
                FileCheck().check("Output code:").run(code)

            self.assertEqual(
                _fn(*_inps),
                _fn_opt(*_inps),
            )

        with config.patch({"cpp.fallback_scatter_reduce_sum": False}):
            _internal_check(fn, inps, "atomic_add")

        scatter_reduce_func = (
            "aoti_torch_cpu_scatter_reduce_"
            if config.cpp_wrapper
            else "aten.scatter_reduce_"
        )
        with config.patch({"cpp.fallback_scatter_reduce_sum": True}):
            _internal_check(fn, inps, scatter_reduce_func)

        if "ATen parallel backend: OpenMP" in torch.__config__.parallel_info():
            with set_num_threads(1):
                # When running with a single thread, we expect the aten.scatter will go
                # into the cpp backend codegen instead of a fallback to aten.scatter_reduce_.
                # Avoid the inductor cache so we don't serve an entry compiled above.
                with config.patch(
                    {"fx_graph_cache": False, "fx_graph_remote_cache": False}
                ):
                    _internal_check(
                        fn, inps, _target_code_check_not=scatter_reduce_func
                    )

            with config.patch({"cpp.dynamic_threads": True}), set_num_threads(1):
                _internal_check(fn, inps, scatter_reduce_func)

    @patch("torch.cuda.is_available", lambda: False)
    @requires_vectorization
    @torch._inductor.config.patch({"cpp.fallback_scatter_reduce_sum": False})
    def test_scatter_using_atomic_add_vec(self):
        def fn(a, dim, index, b):
            return aten.scatter(a, dim, index, b, reduce="add")

        inps = (
            torch.zeros(1, 1, 25),
            2,
            torch.tensor([[[3, 5, 7, 9] * 5]]),
            torch.ones(1, 1, 25),
        )
        torch._dynamo.reset()
        metrics.reset()
        self.common(fn, inps)
        assert metrics.generated_cpp_vec_kernel_count == 2

        with (
            set_num_threads(1),
            config.patch({"fx_graph_cache": False, "fx_graph_remote_cache": False}),
        ):
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, inps)
            assert metrics.generated_cpp_vec_kernel_count == 2

    def test_large_mean(self):
        size = (30000, 100000)
        t = torch.rand(size, dtype=torch.float)
        op = torch.mean
        expected = op(t)
        actual = torch.compile(op)(t)
        self.assertEqual(expected, actual)
        with set_num_threads(1):
            expected = op(t)
            actual = torch.compile(op)(t)
            self.assertEqual(expected, actual)

    def test_outer_mean_large_size(self):
        def fn(x):
            x = x.flatten()
            x_one = torch.ones_like(x)
            x = torch.outer(x, x_one)
            return torch.mean(x, dim=1)

        x = torch.randn(2, 2, 64, 64)
        expected = fn(x)
        actual = torch.compile(fn)(x)
        self.assertEqual(expected, actual, atol=1e-4, rtol=1e-4)

    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_new_vec_op_cpu_only(self):
        def fn(x):
            return torch.log1p(torch.expm1(torch.erf(x)))

        for dtype in vec_dtypes:
            torch.manual_seed(0)
            x = torch.randn((2, 9), dtype=dtype)
            x[0, 0] = torch.nan
            x[1, -1] = torch.nan

            with config.patch({"cpp.simdlen": None}):
                for cpp_wrapper_flag in [True, False]:
                    with config.patch({"cpp_wrapper": cpp_wrapper_flag}):
                        torch._dynamo.reset()
                        metrics.reset()
                        self.common(fn, (x,))
                        check_metrics_vec_kernel_count(1)

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_vec_cpu_only_for_all_available_isa(self):
        def fn(x):
            return torch.sin(torch.cos(torch.erf(x)))

        x = torch.randn((2, 9))
        x[0, 0] = torch.nan
        x[1, -1] = torch.nan

        bit_widths = [isa._bit_width for isa in cpu_vec_isa.valid_vec_isa_list()] + [
            None
        ]
        for item in bit_widths:
            with config.patch({"cpp.simdlen": item}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x,))
                check_metrics_vec_kernel_count(1)

    @slowTest
    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    @config.patch("cpp.enable_tiling_heuristics", False)
    def test__adaptive_avg_pool2d(self):
        def wrap_fn(oh, ow):
            def fn(x):
                return torch._adaptive_avg_pool2d(x, (oh, ow))

            return fn

        bit_widths = [isa._bit_width for isa in cpu_vec_isa.valid_vec_isa_list()]
        ih = [16, 65]
        iw = ih
        oh = ih
        ow = ih
        for _ih, _iw, _oh, _ow, _simd_len, dtype in itertools.product(
            ih, iw, oh, ow, bit_widths, vec_dtypes
        ):
            x = torch.randn(2, 3, _ih, _iw, dtype=dtype).to(
                memory_format=torch.channels_last
            )
            _fn = wrap_fn(_oh, _ow)
            with config.patch({"cpp.simdlen": _simd_len}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(_fn, (x,))
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_vec_logical(self):
        def wrap_fn1(op: Callable):
            def fn(x: torch.Tensor):
                return torch.where(op(x), 1.0, 0.0)

            return fn

        def wrap_fn2(op: Callable):
            def fn(x: torch.Tensor, y: torch.Tensor):
                return torch.where(op(x, y), 1.0, 0.0)

            return fn

        for dtype in vec_dtypes:
            x = torch.randn(64, dtype=dtype)
            y = torch.randn(64, dtype=dtype)
            logical_fns = [
                torch.logical_and,
                torch.logical_not,
                torch.logical_or,
                torch.logical_xor,
            ]
            for logical_fn in logical_fns:
                torch._dynamo.reset()
                metrics.reset()
                if logical_fn == torch.logical_not:
                    _fn = wrap_fn1(logical_fn)
                    _args = (x,)
                else:
                    _fn = wrap_fn2(logical_fn)
                    _args = (x, y)
                self.common(_fn, _args)
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    def test_vec_bitwise(self):
        for dtype in [
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int32,
            torch.int64,
        ]:
            x = torch.randn(64, dtype=torch.float32)
            y = torch.randn(64, dtype=torch.float32)
            if dtype == torch.bool:
                x = x > 0
                y = y > 0
            else:
                x = x.to(dtype)
                y = y.to(dtype)
            bitwise_fns = [
                torch.bitwise_and,
                torch.bitwise_not,
                torch.bitwise_or,
                torch.bitwise_xor,
                torch.bitwise_left_shift,
                torch.bitwise_right_shift,
            ]
            for bitwise_fn in bitwise_fns:
                if (
                    bitwise_fn
                    in [
                        torch.bitwise_left_shift,
                        torch.bitwise_right_shift,
                    ]
                    and dtype == torch.bool
                ):
                    # Eager doesn't support bool
                    # https://pytorch.org/docs/stable/generated/torch.bitwise_left_shift.html
                    continue
                torch._dynamo.reset()
                metrics.reset()
                if bitwise_fn == torch.bitwise_not:
                    _args = (x,)
                else:
                    _args = (x, y)
                self.common(bitwise_fn, _args)
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    def test_vec_randn(self):
        funcs = [torch.randn, torch.rand, torch.randint]
        float_dtypes = [
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ]
        int_dtypes = [
            torch.int8,
            torch.uint8,
            torch.int32,
            torch.int64,
        ]
        dtypes = float_dtypes + int_dtypes
        for rand_func, dtype in itertools.product(funcs, dtypes):
            if (rand_func == torch.randint and dtype not in int_dtypes) or (
                rand_func != torch.randint and dtype not in float_dtypes
            ):
                # Skip the invalid combination
                continue
            with config.patch(
                {"fx_graph_cache": False, "fx_graph_remote_cache": False}
            ):
                torch._dynamo.reset()
                metrics.reset()

                def func(seed):
                    torch.manual_seed(seed)
                    if rand_func == torch.randint:
                        return rand_func(0, 64, (64,), dtype=dtype)
                    else:
                        return rand_func(64, dtype=dtype)

                cfn = torch.compile(func)
                # Check the result is deterministic with mauanl seed
                self.assertEqual(cfn(2024), cfn(2024))
                check_metrics_vec_kernel_count(1)
                res_vec = cfn(2024)

                torch._dynamo.reset()
                metrics.reset()
                with config.patch({"cpp.simdlen": 0}):
                    res_scalar = torch.compile(func)(2024)
                    # Check the same result between scalar and vec
                    self.assertEqual(res_vec, res_scalar)

    @requires_vectorization
    def test_bitwise_logical_op_bool(self):
        bitwise_fns = [
            torch.bitwise_and,
            torch.bitwise_or,
            torch.bitwise_xor,
            torch.logical_and,
            torch.logical_or,
            torch.logical_xor,
        ]

        for bitwise_fn in bitwise_fns:

            def fn(a, b):
                c = bitwise_fn((a > 1), (b > 1))
                return c

            a = torch.ones((64), dtype=torch.int64)
            b = torch.ones((64), dtype=torch.uint8)

            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (a, b))

    def test_torch_logit(self):
        # fix https://github.com/pytorch/pytorch/issues/145379
        def fn(*args):
            return torch.logit(args[0], args[1])

        input = torch.tensor(0.3, dtype=torch.float64)
        eps = torch.tensor(0.9, dtype=torch.float64)
        self.common(fn, (input, eps))

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_vec_compare_op_cpu_only(self):
        def fn(x):
            y1 = torch.eq(x, 1.0)
            x = torch.where(y1, x, -x)
            y2 = torch.ne(x, 0.0)
            x = torch.where(y2, x, -x)
            y3 = torch.lt(x, 5.0)
            x = torch.where(y3, x, x - 1.0)
            y4 = torch.gt(x, -2.0)
            x = torch.where(y4, x, x + 1.0)
            y5 = torch.le(x, 8.0)
            x = torch.where(y5, x, x - 1.0)
            y6 = torch.ge(x, -3.0)
            x = torch.where(y6, x, x + 1.0)
            y7 = x == 1.0
            x = torch.where(y7, x, -x)
            y8 = x != 0.0
            x = torch.where(y8, x, -x)
            y9 = x < 5.0
            x = torch.where(y9, x, x - 1.0)
            y10 = x > -2.0
            x = torch.where(y10, x, x + 1.0)
            y11 = x <= 8.0
            x = torch.where(y11, x, x - 1.0)
            y12 = x >= -3.0
            x = torch.where(y12, x, x + 1.0)
            return x

        for dtype in vec_dtypes:
            x = torch.randn((2, 9), dtype=dtype)

            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x,))
                check_metrics_vec_kernel_count(1)
                assert (
                    metrics.generated_kernel_count
                    - metrics.generated_cpp_vec_kernel_count
                ) == 0

    @requires_vectorization
    def test_vec_remainder(self):
        for dtype in [
            torch.int8,
            torch.uint8,
            torch.int32,
            torch.int64,
            torch.bfloat16,
            torch.float16,
            torch.float32,
            torch.float64,
        ]:
            if is_float_dtype(dtype):
                x = torch.randn(64, dtype=dtype)
                y = torch.randn(64, dtype=dtype)
            else:
                lower = 1 if dtype == torch.uint8 else -100
                x = torch.randint(lower, 100, (64,), dtype=dtype)
                y = torch.randint(lower, 100, (64,), dtype=dtype)
                y = torch.where(
                    y == torch.zeros_like(y),
                    torch.ones_like(y),
                    y,
                )

            torch._dynamo.reset()
            metrics.reset()
            _args = (x, y)
            self.common(torch.remainder, _args)
            check_metrics_vec_kernel_count(1)

    def test_skip_cpp_codegen(self):
        with config.patch({"disable_cpp_codegen": True}):
            inps = torch.ones([20]), torch.rand([20])

            def f(x, y):
                return x + y + torch.tensor(1)

            f_opt = torch.compile(f)
            if config.cpp_wrapper:
                # cpp_wrapper on CPU doesn't work without CPP codegen
                with self.assertRaises(InductorError):
                    f_opt(*inps)
                return

            _, code = run_and_get_cpp_code(f_opt, inps[0], inps[1])
            FileCheck().check_not("void kernel").run(code)

            self.assertEqual(
                f(*inps),
                f_opt(*inps),
            )

            # constant needs to be propagated on fallback
            def f(x):
                return x[torch.tensor(1) :] * 2

            f_opt = torch.compile()(f)
            _, code = run_and_get_cpp_code(f_opt, inps[0])
            FileCheck().check_not("void kernel").run(code)
            self.assertEqual(f_opt(inps[0]), f(inps[0]))

            class Model(torch.nn.Module):
                def __init__(
                    self,
                ):
                    super().__init__()

                def forward(self, v1: torch.Tensor):
                    vx = v1.min(dim=1).values
                    v2 = torch.randn_like(vx)
                    return v2

            model = Model()
            x = torch.rand(10, 3, 0)
            model_f = torch.compile()(model)

            self.assertEqual(model(x), model_f(x))

    def test_redundant_to_node_elimination_lowp_fp(self):
        def fn(x, y):
            res = x + y
            res = torch.mean(res)
            return res

        for dtype in _lowp_fp_dtypes:
            x = torch.randn((2, 9), dtype=dtype)
            y = torch.randn((2, 9), dtype=dtype)

            for torch_compile_debug in [True, False]:
                with config.patch(
                    {"trace.enabled": torch_compile_debug, "cpp.simdlen": None}
                ):
                    torch._dynamo.reset()
                    metrics.reset()
                    self.common(fn, (x, y))
                    check_metrics_vec_kernel_count(1)

    def test_do_not_insert_to_dtype_for_memory_copy_only_kernel(self):
        def fn(x):
            res = x.clone()
            return res

        x = torch.randn((100, 100), dtype=torch.bfloat16)

        torch._dynamo.reset()
        metrics.reset()
        self.common(fn, (x,))
        assert metrics.cpp_to_dtype_count == 0
        check_metrics_vec_kernel_count(1)

    def test_insert_to_dtype_count(self):
        def fn(x):
            res = x.relu()
            return res

        x = torch.randn((100, 100), dtype=torch.bfloat16)

        torch._dynamo.reset()
        metrics.reset()
        self.common(fn, (x,))
        assert metrics.cpp_to_dtype_count == 2
        check_metrics_vec_kernel_count(1)

    def test_memory_copy_with_fusion(self):
        def fn(x):
            res = x.relu()
            x.copy_(res)
            return (res,)

        x = torch.randn((100, 100), dtype=torch.bfloat16)

        torch._dynamo.reset()
        metrics.reset()
        self.common(fn, (x,))
        assert metrics.cpp_to_dtype_count == 2
        check_metrics_vec_kernel_count(1)

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_maxpool2d_cpu_only(self):
        for dtype in vec_dtypes:
            input = torch.randn(26, 32, 112, 112, dtype=dtype).to(
                memory_format=torch.channels_last
            )
            maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            def func(x):
                return maxpool(x)

            with patch.object(config.cpp, "simdlen", None):
                torch._dynamo.reset()
                metrics.reset()
                self.common(func, (input,))
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    @config.patch("cpp.enable_tiling_heuristics", False)
    def test_maxpool2d_with_pre_loop_collapse_cpu_only(self):
        x1 = torch.randn(2, 3, 20, 20).to(memory_format=torch.channels_last)
        x2 = torch.randn(2, 3, 20, 20).to(memory_format=torch.channels_last)
        maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        def func(x1, x2):
            y = x1 + x2
            return maxpool(y)

        with patch.object(config.cpp, "simdlen", None):
            torch._dynamo.reset()
            metrics.reset()
            self.common(func, (x1, x2))
            check_metrics_vec_kernel_count(2)

    def test_randint_symint_input(self):
        # https://github.com/pytorch/pytorch/issues/122405
        @torch.compile(fullgraph=True)
        def get_traj_idx(lengths: torch.Tensor, num_slices: int) -> torch.Tensor:
            return torch.randint(lengths.shape[0], (num_slices,), device=lengths.device)

        lengths = torch.zeros(10, dtype=torch.long)
        get_traj_idx(lengths, num_slices=4)
        lengths = torch.zeros(11, dtype=torch.long)
        get_traj_idx(lengths, num_slices=4)

    def test_store_reduction(self):
        # fix https://github.com/pytorch/pytorch/issues/157683
        def fn(x, y):
            r1 = x.amax(dim=0)
            r2 = y.amax(dim=0)
            return r1, r2

        device = "cpu"
        for int_dypte, float_dtype in zip(
            [torch.int64, torch.int32, torch.int16, torch.int8],
            [torch.float64, torch.float32, torch.float16, torch.bfloat16],
        ):
            x = torch.randint(
                low=0, high=100, size=(16, 24, 59), dtype=int_dypte, device=device
            )
            y = torch.randn(16, 24, 59, dtype=float_dtype, device=device)
            self.common(
                fn,
                (
                    x,
                    y,
                ),
            )

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_sign_cpu_only(self):
        def fn(x):
            return torch.sign(x)

        for dtype in vec_dtypes:
            x = torch.randn((2, 9), dtype=dtype)
            x[0, 0] = torch.nan
            x[1, -1] = torch.nan

            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x,))
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_reduction_cpu_only(self):
        def fn(x):
            return torch.argmax(x, -1)

        for dtype in vec_dtypes:
            x = torch.randn((10, 10), dtype=dtype)

            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x,))
                assert metrics.generated_cpp_vec_kernel_count == 1

    @config.patch({"fx_graph_cache": False, "fx_graph_remote_cache": False})
    def test_outer_loop_fusion(self):
        def fn(x):
            max = torch.amax(x, dim=-1, keepdim=True)
            return x - max

        x = torch.randn(4, 12, 1023, 1022)

        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, (x,))
            self.assertEqual(
                len(metrics.cpp_outer_loop_fused_inner_counts),
                1,
            )
            self.assertEqual(
                metrics.cpp_outer_loop_fused_inner_counts[0].inner_kernel_number,
                2,
            )

    def test_outer_loop_fusion_buffer_remove(self):
        # https://github.com/pytorch/pytorch/issues/144186
        def fn(x):
            x = x.sum(dim=-1)
            x = torch.softmax(x, -1)
            return x

        x = torch.randn(8, 8, 2)
        metrics.reset()
        self.common(fn, (x,))

    def test_softmax_with_zero_dim(self):
        def fn(x):
            x = torch.softmax(x, 0)
            return x

        x = torch.rand([], dtype=torch.bfloat16)
        metrics.reset()
        self.common(fn, (x,))

    @config.patch({"fx_graph_cache": False, "fx_graph_remote_cache": False})
    def test_local_buffer_in_outer_loop_fusion(self):
        def fn(x):
            max = torch.nn.functional.softmax(x, dim=-1)
            return x - max

        x = torch.randn(4, 12, 1023, 1022)

        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, (x,))
            self.assertEqual(
                len(metrics.cpp_outer_loop_fused_inner_counts),
                1,
            )
            self.assertEqual(
                metrics.cpp_outer_loop_fused_inner_counts[0].inner_kernel_number,
                3,
            )
            self.assertEqual(
                metrics.cpp_outer_loop_fused_inner_counts[0].local_buffer_number,
                1,
            )
            # Check the number of global buffer allocation
            torch._dynamo.reset()
            metrics.reset()
            _, code = run_and_get_cpp_code(
                torch.compile(fn, backend="inductor"),
                x,
            )
            self.assertEqual(
                code.count(
                    "aoti_torch_empty_strided("
                    if config.cpp_wrapper
                    else "empty_strided_cpu("
                ),
                3,
            )

    @config.patch({"fx_graph_cache": False, "fx_graph_remote_cache": False})
    def test_two_local_buffers_in_outer_loop_fusion(self):
        def fn(x):
            softmax = torch.nn.functional.softmax(x, dim=-1)
            sum = torch.sum(softmax, dim=-1)
            sum_broadcast = torch.broadcast_to(
                sum.unsqueeze(-1), [*(sum.size()[0:3]), 256]
            )
            sum_exp = torch.exp(sum_broadcast)
            sum2 = torch.sum(sum_exp, dim=-1)
            sub = sum_exp - sum2.unsqueeze(-1)
            return x[:, :, :, 0:256] - sub

        x = torch.randn(4, 12, 1023, 1022)

        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            atol = None
            rtol = None
            if (
                not cpu_vec_isa.valid_vec_isa_list()
                or os.getenv("ATEN_CPU_CAPABILITY") == "default"
            ):
                atol = 1e-5
                rtol = 1e-5
            self.common(fn, (x,), atol=atol, rtol=rtol)
            self.assertEqual(
                len(metrics.cpp_outer_loop_fused_inner_counts),
                1,
            )
            self.assertEqual(
                metrics.cpp_outer_loop_fused_inner_counts[0].inner_kernel_number,
                5,
            )
            self.assertEqual(
                metrics.cpp_outer_loop_fused_inner_counts[0].local_buffer_number,
                2,
            )

    @config.patch({"fx_graph_cache": False, "fx_graph_remote_cache": False})
    def test_share_local_buffers_in_outer_loop_fusion(self):
        def fn(x):
            max = torch.nn.functional.softmax(x, dim=-1)
            max = torch.nn.functional.softmax(max, dim=-1)
            return x - max

        x = torch.randn(4, 12, 1023, 1022)

        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, (x,))
            self.assertEqual(
                len(metrics.cpp_outer_loop_fused_inner_counts),
                1,
            )
            self.assertEqual(
                metrics.cpp_outer_loop_fused_inner_counts[0].inner_kernel_number,
                5,
            )
            self.assertEqual(
                metrics.cpp_outer_loop_fused_inner_counts[0].local_buffer_number,
                1,  # 2 global bufs share 1 local buf
            )

    @config.patch({"fx_graph_cache": False, "fx_graph_remote_cache": False})
    def test_two_local_buffers_in_outer_loop_fusion_case2(self):
        # exp and exp2 should be replaced by local buffer
        # since exp will be used after exp2, exp2 can't share the same
        # local buffer as exp
        def fn(x):
            a_max = torch.amax(x, -1, keepdim=True)
            exp = torch.exp(x - a_max)
            sum = torch.sum(exp, -1, keepdim=True)
            exp2 = torch.exp(exp - sum)
            sum2 = torch.sum(exp2, -1, keepdim=True)
            sub = exp2 - sum2
            sub2 = exp - sub
            return sub2

        x = torch.randn(4, 12, 1023, 1022)

        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, (x,))
            self.assertEqual(
                len(metrics.cpp_outer_loop_fused_inner_counts),
                1,
            )
            self.assertEqual(
                metrics.cpp_outer_loop_fused_inner_counts[0].inner_kernel_number,
                4,
            )
            self.assertEqual(
                metrics.cpp_outer_loop_fused_inner_counts[0].local_buffer_number,
                2,
            )

    @config.patch({"fx_graph_cache": False, "fx_graph_remote_cache": False})
    def test_local_buffer_with_line_reuse(self):
        # Test Global buffer which is inplace buffer and replaced by local buffer
        def fn(x, y):
            z = torch.matmul(x, y)
            a_max = torch.amax(x, -1, keepdim=True)
            # Previous is a inplace buffer and now is a local buffer
            exp = torch.exp((z - a_max) / z)
            sum = torch.sum(exp, -1, keepdim=True)
            return exp - sum

        inputs = [torch.rand(4, 32), torch.rand(32, 32)]

        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, inputs)
            self.assertEqual(
                len(metrics.cpp_outer_loop_fused_inner_counts),
                1,
            )
            self.assertEqual(
                metrics.cpp_outer_loop_fused_inner_counts[0].local_buffer_number,
                1,
            )

    @requires_vectorization
    def test_argmin(self):
        def fn(x):
            return torch.argmin(x, -1)

        for dtype in vec_dtypes:
            x = torch.randn((10, 10), dtype=dtype)
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, (x,))
            assert metrics.generated_cpp_vec_kernel_count == 1

    @requires_vectorization
    def test_argmax_argmin_with_nan_value(self):
        def fn(x):
            return torch.argmax(x)

        def fn2(x):
            return torch.argmin(x)

        inputs = [
            torch.Tensor([-755832.1250, 100]),
            torch.Tensor([-755832.1250, 100, 200]),
            torch.Tensor([100, -755832.1250]),
            torch.Tensor([100, 200, -755832.1250]),
        ]

        for x in inputs:
            x = x.repeat(16, 16)
            x = torch.log1p(x)

            # Test argmax
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, (x,))
            assert metrics.generated_cpp_vec_kernel_count == 1

            # Test argmin
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn2, (x,))
            assert metrics.generated_cpp_vec_kernel_count == 1

    # Currently, we enabled AVX2 and AVX512 for vectorization. If the platform is not
    # supported, the vectorization will not work and skip this test case. For ARM or
    # other platforms support, we just need to add the ISA info to the supported_vector_isa
    # and include proper aten vectorization head file.
    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_vec_kernel_cpu_only(self):
        def fn(x1, x2):
            # Current, there are some limitations as follows.
            #   rsqrt:
            #     assert [both a fallback and a decomp for same kernel: aten.rsqrt.default]
            #   round:
            #     couldn't find symbolic meta function/decomposition
            #   fmod/logical_and/logic_or:
            #     vec kernel has not support to_type
            x = torch.abs(x1)
            x = torch.sin(x)
            x = torch.neg(x)
            x = torch.square(x)
            x = torch.sigmoid(x)
            x = torch.relu(x)
            x = torch.cos(x)
            x = torch.exp(x)
            x = torch.sqrt(x)
            x = torch.add(x, x1)
            x = torch.sub(x, x2)
            x = torch.mul(x, x1)
            x = torch.div(x, x1)
            x = torch.pow(x, 10)
            x = torch.log(x)
            x = torch.floor(x)
            x = torch.ceil(x)
            x = torch.trunc(x)
            x = torch.lgamma(x)
            x = torch.fmod(x, x2)
            x = torch.sign(x)
            res = x + x2
            return res

        for dtype in vec_dtypes:
            torch.manual_seed(0)
            x1 = torch.randn((5, 20), dtype=dtype)
            x2 = torch.randn((5, 20), dtype=dtype)

            with config.patch({"cpp.simdlen": 1}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x1, x2))
                assert metrics.generated_cpp_vec_kernel_count == 0

            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x1, x2))
                check_metrics_vec_kernel_count(1)

        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            x1 = torch.randn(10, 20).permute(1, 0)
            x2 = torch.randn((20, 10))
            self.common(fn, (x1, x2))
            check_metrics_vec_kernel_count(2)

            torch._dynamo.reset()
            metrics.reset()
            x1 = torch.randn((10, 7))
            x2 = torch.randn((10, 7))
            self.common(fn, (x1, x2))
            check_metrics_vec_kernel_count(1)

    @unittest.skipIf(IS_FBCODE, "Not yet runnable in fbcode")
    @unittest.skipIf(
        sys.platform not in ["linux", "win32"],
        "cpp kernel profile only support linux now",
    )
    @patch("torch.cuda.is_available", lambda: False)
    @config.patch({"cpp.enable_kernel_profile": True})
    @config.patch({"cpp.descriptive_names": "original_aten"})
    def test_cpp_kernel_profile(self):
        from torch.profiler import profile

        @torch.compile(backend="inductor", fullgraph=True)
        def fn(a, b):
            return a + b

        a = torch.rand((100,))
        b = torch.rand((100,))
        with profile() as prof:
            fn(a, b)

        kernel_profile_events = []
        for e in prof.profiler.function_events:
            if "cpp_fused_add_0" in e.name:
                kernel_profile_events.append(e.name)
        assert len(kernel_profile_events) > 0

    @xfailIfS390X
    @requires_vectorization
    def test_channel_shuffle_cl_output(self):
        """code and shape extracted from shufflenet_v2_x1_0"""

        def channel_shuffle(x, groups):
            batchsize, num_channels, height, width = x.size()
            channels_per_group = num_channels // groups
            x = x.view(batchsize, groups, channels_per_group, height, width)
            x = torch.transpose(x, 1, 2).contiguous()
            x = x.view(batchsize, -1, height, width)
            return x.contiguous(memory_format=torch.channels_last)

        for simdlen in simd_lengths_to_test():
            with config.patch({"cpp.simdlen": simdlen}):
                torch._dynamo.reset()
                metrics.reset()
                x = torch.randn(64, 58, 28, 28)
                self.common(channel_shuffle, (x, 2))
                check_metrics_vec_kernel_count(2)

    @slowTest
    @requires_vectorization
    def test_transpose_with_norm(self):
        """a sub-module from TIMM gmlp_s16_224"""

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(
                    in_features=256, out_features=1536, bias=True
                )
                self.act = torch.nn.GELU()
                self.norm = torch.nn.LayerNorm(768)
                self.proj = torch.nn.Linear(196, 196)
                self.fc = torch.nn.Linear(in_features=768, out_features=256, bias=True)

            def forward(self, x):
                x = self.linear(x)
                x = self.act(x)
                u, v = x.chunk(2, dim=-1)
                v = self.norm(v)
                v = self.proj(v.transpose(-1, -2))
                y = u * v.transpose(-1, -2)
                return self.fc(y)

        x = torch.randn(128, 196, 256)
        for simdlen in simd_lengths_to_test():
            with config.patch({"cpp.simdlen": simdlen}):
                for eval_mode in [True, False]:
                    torch._dynamo.reset()
                    metrics.reset()
                    m = Model().eval() if eval_mode else Model()
                    self.common(m, (x,))
                    check_metrics_vec_kernel_count(6)

    @requires_vectorization
    @config.patch("cpp.enable_tiling_heuristics", False)
    def test_transpose_copy(self):
        def fn(a):
            return a.t().contiguous()

        for simdlen in simd_lengths_to_test():
            with config.patch({"cpp.simdlen": simdlen}):
                for dtype in (torch.float, torch.bfloat16):
                    for shape in (
                        (7, 7),
                        (8, 8),
                        (9, 9),
                        (16, 16),
                        (17, 17),
                        (32, 32),
                        (33, 33),
                    ):
                        torch._dynamo.reset()
                        metrics.reset()
                        x = torch.randn(shape, dtype=dtype)
                        self.common(fn, (x,))
                        check_metrics_vec_kernel_count(2)

    @torch._dynamo.config.patch(specialize_int=False)
    def test_slice_scatter_issue122291(self):
        @torch.compile(fullgraph=True)
        def fn(t, t_src, dim, start, end, step):
            return t.slice_scatter(t_src, dim, start, end, step)

        shape = ((16, 16), (16, 2), 1, 4, 10, 1)
        input_tensor = torch.zeros(shape[0], requires_grad=False, device="cpu")
        src_tensor = torch.ones(shape[1], requires_grad=False, device="cpu")
        with self.assertRaisesRegex(
            torch._inductor.exc.InductorError, r".*shape error in scatter op"
        ):
            fn(input_tensor, src_tensor, shape[2], shape[3], shape[4], shape[5])

    def test_horizontal_fusion(self):
        def fn(a, b, c, idx):
            _a = torch.index_select(a, dim=0, index=idx)
            _b = torch.index_select(b, dim=0, index=idx)
            _c = torch.index_select(c, dim=0, index=idx)
            return _a, _b, _c

        with config.patch({"cpp.max_horizontal_fusion_size": 0}):
            metrics.reset()
            torch._dynamo.reset()
            a = torch.randn(size=(4, 16), dtype=torch.bfloat16)
            b = torch.randn(size=(4, 16), dtype=torch.bfloat16)
            c = torch.randn(size=(4, 16), dtype=torch.bfloat16)
            idx = torch.zeros(size=[4], dtype=torch.int64)
            opt_fn = torch.compile(fn, backend="inductor")
            opt_fn(a, b, c, idx)
            self.assertEqual(metrics.generated_kernel_count, 3)
            self.assertTrue(same(fn(a, b, c, idx), opt_fn(a, b, c, idx)))

        with config.patch({"cpp.max_horizontal_fusion_size": 1}):
            metrics.reset()
            torch._dynamo.reset()
            a = torch.randn(size=(4, 32), dtype=torch.bfloat16)
            b = torch.randn(size=(4, 32), dtype=torch.bfloat16)
            c = torch.randn(size=(4, 32), dtype=torch.bfloat16)
            idx = torch.zeros(size=[4], dtype=torch.int64)
            opt_fn = torch.compile(fn, backend="inductor")
            opt_fn(a, b, c, idx)
            self.assertEqual(metrics.generated_kernel_count, 3)
            self.assertTrue(same(fn(a, b, c, idx), opt_fn(a, b, c, idx)))

        with config.patch({"cpp.max_horizontal_fusion_size": 2}):
            metrics.reset()
            torch._dynamo.reset()
            a = torch.randn(size=(4, 64), dtype=torch.bfloat16)
            b = torch.randn(size=(4, 64), dtype=torch.bfloat16)
            c = torch.randn(size=(4, 64), dtype=torch.bfloat16)
            idx = torch.zeros(size=[4], dtype=torch.int64)
            opt_fn = torch.compile(fn, backend="inductor")
            opt_fn(a, b, c, idx)
            print(metrics.generated_kernel_count)
            self.assertEqual(metrics.generated_kernel_count, 2)
            self.assertTrue(same(fn(a, b, c, idx), opt_fn(a, b, c, idx)))

        with config.patch({"cpp.max_horizontal_fusion_size": 3}):
            metrics.reset()
            torch._dynamo.reset()
            a = torch.randn(size=(4, 128), dtype=torch.bfloat16)
            b = torch.randn(size=(4, 128), dtype=torch.bfloat16)
            c = torch.randn(size=(4, 128), dtype=torch.bfloat16)
            idx = torch.zeros(size=[4], dtype=torch.int64)
            opt_fn = torch.compile(fn, backend="inductor")
            opt_fn(a, b, c, idx)
            self.assertEqual(metrics.generated_kernel_count, 1)
            self.assertTrue(same(fn(a, b, c, idx), opt_fn(a, b, c, idx)))

    def test_lowp_fp_neg_abs(self):
        def fn(x):
            return x.neg().abs()

        for dtype in _lowp_fp_dtypes:
            metrics.reset()
            x = torch.randn(100, 100).to(dtype)
            opt_fn = torch.compile(fn, backend="inductor")
            self.assertTrue(same(fn(x), opt_fn(x)))
            assert metrics.cpp_to_dtype_count == 0
            check_metrics_vec_kernel_count(1)

    @config.patch("cpp.enable_tiling_heuristics", False)
    def test_transpose_non_contiguous(self):
        def fn(a):
            # From part of timm HaloAttn:
            # (https://github.com/rwightman/pytorch-image-models/blob/main/timm/layers/halo_attn.py#L97).
            # Fixed https://github.com/pytorch/pytorch/issues/94269 accuracy issue.
            as_strided = torch.ops.aten.as_strided.default(
                a, [1, 384, 2, 20, 12], [153600, 1, 61440, 384, 7680]
            )
            as_strided_1 = torch.ops.aten.as_strided.default(
                as_strided,
                [1, 384, 2, 2, 12, 12],
                [153600, 1, 61440, 3072, 7680, 384],
            )
            clone_1 = torch.ops.aten.clone.default(
                as_strided_1, memory_format=torch.contiguous_format
            )
            _unsafe_view_1 = torch.ops.aten._unsafe_view.default(
                clone_1, [8, 48, 4, 144]
            )
            permute_2 = torch.ops.aten.permute.default(_unsafe_view_1, [0, 2, 3, 1])
            split_with_sizes = torch.ops.aten.split_with_sizes.default(
                permute_2, [16, 32], -1
            )
            getitem = split_with_sizes[0]
            _getitem_1 = split_with_sizes[1]
            permute_3 = torch.ops.aten.permute.default(getitem, [0, 1, 3, 2])
            expand_1 = torch.ops.aten.expand.default(permute_3, [8, 4, 16, 144])
            clone_3 = torch.ops.aten.clone.default(
                expand_1, memory_format=torch.contiguous_format
            )
            return clone_3

        metrics.reset()
        x = torch.randn(1, 384, 20, 20).to(memory_format=torch.channels_last)
        self.common(fn, (x,))

    def test_issue_148058(self):
        # Fix issue https://github.com/pytorch/pytorch/issues/148058
        def fn(x):
            x = F.gumbel_softmax(x, tau=1.0, hard=True)
            x = torch.where(x > 0.5, x, torch.zeros_like(x))
            x = torch.scatter(
                x,
                dim=1,
                index=torch.ones(1, 2, dtype=torch.long),
                src=torch.ones_like(x),
            )
            return x

        metrics.reset()
        x = torch.randn(1, 2)
        # Only test for functionality since the output of gumbel_softmax has randomness
        torch.compile(fn, backend="inductor")(x)

    def test_non_contiguous_index_with_constant_stride(self):
        def fn(x):
            x1 = x[:, :, :, ::2]
            x2 = x[:, :, :, 1::2]
            x = torch.stack((-x2, x1), dim=-1)
            return x.flatten(-2)

        metrics.reset()
        x = torch.randn(1, 32, 16, 68)
        opt_fn = torch.compile(fn, backend="inductor")
        _, code = run_and_get_cpp_code(opt_fn, x)
        self.assertTrue(same(fn(x), opt_fn(x)))
        # declare, def, and use (declare and def are the same in non-cpp_wrapper mode)
        FileCheck().check_count(
            "cpp_fused", 3 if config.cpp_wrapper else 2, exactly=True
        ).run(code)

    def test_invalid_index_of_empty_tensor(self):
        def fn(a):
            b = a[[0]]
            return b

        a = torch.tensor([])
        with self.assertRaises(RuntimeError):
            torch.compile(fn)(a)

    @torch.no_grad()
    @torch._inductor.config.patch(freezing=True)
    def test_issue122380(self):
        def func(x):
            t1 = torch.unbind(x)
            t2 = torch.stack(t1, dim=1)
            t3 = torch.tanh(t2)
            return t3

        x = torch.randn(2, 3, 4)
        self.assertEqual(torch.compile(func)(x), func(x))

    @config.patch({"fx_graph_cache": False, "fx_graph_remote_cache": False})
    def test_ir_node_str(self):
        @torch.compile
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x.sin(), torch.nn.Softmax(dim=1)(x.cos())

        def run_node_alt(*args, **kwargs):
            rv = run_node(*args, **kwargs)
            strings.append(str(rv))
            return rv

        strings = []
        run_node = GraphLowering.run_node
        with patch.object(GraphLowering, "run_node", run_node_alt):
            fn(torch.randn([8, 128]))
        self.assertGreater(len(strings), 3)

    def test_vertical_sum_cpu_only(self):
        def fn1(a):
            return a.sum(dim=0)

        def fn2(a):
            return a.sum(dim=1)

        metrics.reset()
        x = torch.randn(100, 100)
        self.common(fn1, (x,))
        check_metrics_vec_kernel_count(1)

        metrics.reset()
        x = torch.randn(100, 100, 100)
        self.common(fn2, (x,))
        check_metrics_vec_kernel_count(1)

    def test_transpose_vertical_sum_cpu_only(self):
        def fn(a, b):
            c = a * b
            return c.sum(dim=1)

        metrics.reset()
        x = torch.randn(100, 50, 50)
        y = torch.randn(100, 50, 50).transpose(1, 2)
        self.common(fn, (x, y))
        check_metrics_vec_kernel_count(2)

    def test_transpose_mxn_16_16_bf16_fp16(self):
        def fn(a, b):
            c = a * b
            return c.sum(dim=1)

        for dtype in [torch.bfloat16, torch.float16]:
            metrics.reset()
            x = torch.randn(100, 50, 50).to(dtype)
            y = torch.randn(100, 50, 50).to(dtype).transpose(1, 2)
            self.common(fn, (x, y))
            check_metrics_vec_kernel_count(2)

    def test_transpose_mxn_32_32_bf16_fp16(self):
        def fn(a):
            return a.permute(0, 2, 1).contiguous()

        for dtype in [torch.bfloat16, torch.float16]:
            metrics.reset()
            x = torch.randn(2, 9216, 9216).to(dtype)
            self.common(fn, (x,))
            check_metrics_vec_kernel_count(2)

    def test_transpose_sum2d_cpu_only(self):
        def fn(a, b):
            c = a * b
            return c.sum()

        metrics.reset()
        x = torch.randn(50, 50)
        y = torch.randn(50, 50).transpose(0, 1)
        self.common(fn, (x, y))
        check_metrics_vec_kernel_count(2)

    @config.patch("cpp.enable_tiling_heuristics", False)
    def test_transpose_sum_outer(self):
        # https://github.com/pytorch/pytorch/issues/98573
        def fn(a):
            return a.transpose(2, 3).sum(dim=1).contiguous()

        metrics.reset()
        x = torch.randn(10, 50, 50, 50)
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

    def test_to_dtype_bool_float(self):
        # https://github.com/pytorch/pytorch/issues/100800
        def f(a):
            return torch.where(
                torch.ones_like(a).to(torch.bool),
                torch.zeros_like(a),
                torch.ones_like(a) * 2,
            )

        self.common(f, (torch.ones(16),))

    def test_to_dtype_float_bool(self):
        # https://github.com/pytorch/pytorch/issues/100466
        def f(a):
            a = a * torch.tensor(a >= 0, dtype=torch.float32)
            return a

        x = torch.rand(16)
        self.common(f, (x,))

    def test_constant_store(self):
        # https://github.com/pytorch/pytorch/issues/104515
        def f(a):
            a[0, [3, 3]] = -float("inf")
            return a

        x = torch.rand(4, 5)
        self.common(f, (x,))

    def test_broadcast_scalar_cpp_tile_2d_kernel(self):
        # Based on detectron2_maskrcnn backbone (conv2d -> max_pool2d)
        s0 = 12
        s1 = 21

        data = torch.randn(
            [1, 256, 8 * s0, 8 * s1],
        )
        weight_one = torch.randn([256, 256, 1, 1], requires_grad=True)
        weight_two = torch.randn((256, 256, 3, 3), requires_grad=True)
        bias_one = torch.randn([256], requires_grad=True)
        bias_two = torch.randn([256], requires_grad=True)

        @torch.compile
        def fn(data, weight_one, weight_two, bias_one, bias_two):
            conv_result_one = torch.ops.aten.convolution.default(
                data,
                weight_one,
                bias_one,
                [1, 1],
                [1, 1],
                [1, 1],
                False,
                [0, 0],
                1,
            )

            conv_result_two = torch.ops.aten.convolution.default(
                data,
                weight_two,
                bias_two,
                [1, 1],
                [1, 1],
                [1, 1],
                False,
                [0, 0],
                1,
            )

            max_pool_result = torch.nn.functional.max_pool2d(
                conv_result_one,
                [1, 1],
                [2, 2],
                [0, 0],
                [1, 1],
                False,
            )
            return conv_result_one, conv_result_two, max_pool_result

        torch._dynamo.mark_dynamic(data, 2)
        torch._dynamo.mark_dynamic(data, 3)
        self.common(fn, (data, weight_one, weight_two, bias_one, bias_two))

    def test_to_channels_last_lowp_fp(self):
        def f(a):
            return a.to(memory_format=torch.channels_last)

        for dtype in _lowp_fp_dtypes:
            x = torch.rand(2, 3, 14, 14).to(dtype)
            self.common(f, (x,))

    def test_broadcast_mul_lowp_fp(self):
        def f(a, b):
            return a * b

        for dtype in _lowp_fp_dtypes:
            a = torch.randn(2, 16, 16).to(dtype)
            b = torch.randn(2, 1, 1).to(dtype)
            self.common(f, (a, b))

    def test_linear_buffer_reuse(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = torch.nn.Linear(16, 16)
                self.tanh = torch.nn.Tanh()
                self.linear2 = torch.nn.Linear(16, 16)

            def forward(self, x):
                x = self.linear1(x)
                x = self.tanh(x)
                x = self.linear2(x)
                return x

        mod = M().eval()
        v = torch.randn(1, 16)

        with torch.no_grad():

            def compile_fx_wrapper(model_, example_inputs_):
                return compile_fx(model_, example_inputs_)

            def run(*ex, **kwargs):
                return mod(*ex, **kwargs)

            run = torch.compile(run, backend=compile_fx_wrapper)
            _, code = run_and_get_cpp_code(run, v)
            self.assertFalse("= as_strided(" in code)
            self.assertEqual(run(*v), mod(*v))

    def test_invalid_dropout_args(self):
        class MyModel(torch.nn.Module):
            def forward(self, x):
                x = x * 2
                x = torch.nn.functional.dropout(x, p=0.5)
                x = torch.relu(x)
                return x

        example_inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])

        func = MyModel()
        jit_func = torch.compile(func)
        self.assertRaises(RuntimeError, lambda: func(example_inputs))
        self.assertRaises(RuntimeError, lambda: jit_func(example_inputs))

    def test_nn_param_assign(self):
        # https://github.com/pytorch/pytorch/issues/99569
        class Model2(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
                self.batchnorm = nn.BatchNorm2d(num_features=5)
                self.conv_weight = torch.randn(5, 3, 3, 3)
                self.conv_bias = torch.randn(5)

            def forward(self, x):
                self.conv.weight = nn.Parameter(self.conv_weight)
                self.conv.bias = nn.Parameter(self.conv_bias, requires_grad=False)
                self.conv.eval()
                x = self.conv(x)
                x = self.batchnorm(x)
                x = F.relu(x)
                return x

        input_tensor = torch.randn(1, 3, 10, 10)
        func = Model2().to("cpu")

        with torch.no_grad():
            func.train(False)
            v1 = func(input_tensor)
            jit_func = torch.compile(func, fullgraph=True)
            v2 = jit_func(input_tensor)
            self.assertEqual(v1, v2)

    def test_nn_param_assign_wrapped(self):
        class Model2(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
                self.batchnorm = nn.BatchNorm2d(num_features=5)
                self.conv_weight = torch.randn(5, 3, 3, 3)
                self.conv_bias = torch.randn(5)

            def forward(self, x):
                self.conv.weight = nn.Parameter(self.conv_weight)
                self.conv.bias = nn.Parameter(self.conv_bias, requires_grad=False)
                self.conv.eval()
                x = self.conv(x)
                x = self.batchnorm(x)
                x = F.relu(x)
                return x

        input_tensor = torch.randn(1, 3, 10, 10)
        func = Model2().to("cpu")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        with torch.no_grad():
            func.train(False)
            v1 = func(input_tensor)
            jit_func = torch.compile(wrapper, fullgraph=True)
            v2 = jit_func(input_tensor)
            self.assertEqual(v1, v2)

    @config.patch(inplace_buffers=True)
    def test_in_out_buffer(self):
        def fn(x, y):
            z = torch.matmul(x, y.transpose(-1, -2)) / 8.0
            return z

        inps = [torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4)]
        fn_opt = torch.compile(fn, backend="inductor")
        _, code = run_and_get_cpp_code(fn_opt, *inps)
        self.assertTrue("in_out_ptr" in code)
        self.assertEqual(fn_opt(*inps), fn(*inps))

    def test_eliminate_meaningless_copy(self):
        def fn(x1, x2):
            permute = torch.ops.aten.permute.default(x2, [0, 2, 1, 3])
            clone = torch.ops.aten.clone.default(
                permute, memory_format=torch.contiguous_format
            )
            view = torch.ops.aten.view.default(clone, [1024, -1, 32])
            bmm = torch.ops.aten.bmm.default(view, x1)
            permute = torch.ops.aten.permute.default(view, [0, 2, 1])
            return (bmm, permute)

        metrics.reset()
        self.common(
            fn,
            [
                rand_strided(
                    (1024, 32, 128), (4096, 1, 32), device="cpu", dtype=torch.float32
                ),
                rand_strided(
                    (64, 128, 16, 32),
                    (65536, 512, 32, 1),
                    device="cpu",
                    dtype=torch.float32,
                ),
            ],
        )
        self.assertEqual(metrics.generated_kernel_count, 1)

    def test_relu_permute_reshape_reinterpret_view(self):
        def fn(x):
            n, c, h, w = x.shape
            return torch.relu(x).permute(0, 2, 3, 1).reshape(n, h * w, c)

        x = torch.randn(2, 32, 4, 4).to(memory_format=torch.channels_last)
        torch._dynamo.reset()
        metrics.reset()
        with torch.no_grad():
            expected = fn(x)
            compiled_fn = torch.compile(fn)
            actual, code = run_and_get_cpp_code(compiled_fn, x)
            self.assertEqual(expected, actual)
            # 1 generated kernel
            self.assertEqual(metrics.generated_kernel_count, 1)
            # check that there is no transpose
            FileCheck().check_count("transpose_mxn", 0, exactly=True).run(code)

    def test_attention_size_mismatch(self):
        class Attention(torch.nn.Module):
            def __init__(self, hidden_size, num_heads):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                self.head_size = hidden_size // num_heads
                self.query = torch.nn.Linear(hidden_size, hidden_size)
                self.key = torch.nn.Linear(hidden_size, hidden_size)
                self.value = torch.nn.Linear(hidden_size, hidden_size)
                self.inv_scale = torch.nn.Parameter(
                    torch.Tensor([1 / self.head_size**0.5]), requires_grad=False
                )

            def forward(self, x):
                query = self.query(x)
                key = self.key(x)
                value = self.value(x)
                (batch_size, seq_len, hidden_size) = query.size()
                query = query.view(
                    batch_size, seq_len, self.num_heads, self.head_size
                ).permute(0, 2, 1, 3)
                key = key.view(
                    batch_size, seq_len, self.num_heads, self.head_size
                ).permute(0, 2, 3, 1)
                value = value.view(
                    batch_size, seq_len, self.num_heads, self.head_size
                ).permute(0, 2, 1, 3)
                attention_weights = (
                    torch.matmul(query, key).mul(self.inv_scale).softmax(dim=-1)
                )
                output = torch.matmul(attention_weights, value)
                return output

        torch.manual_seed(123)
        hidden_size = 16
        num_heads = 1
        seq_len = 4
        batch_size = 1
        x = torch.randn(batch_size, seq_len, hidden_size)

        func = Attention(hidden_size, num_heads).to("cpu")

        with torch.no_grad():
            res1 = func(x)
            jit_func = torch.compile(func)
            res2 = jit_func(x)
        self.assertEqual(res1, res2)

    def test_scalar_mul_bfloat16(self):
        def f(x):
            return torch.ops.aten.mul.Tensor(x, 1.7015043497085571)

        metrics.reset()
        x = torch.randn(4, 5, dtype=torch.bfloat16)
        self.common(f, (x,))
        check_metrics_vec_kernel_count(1)

    def test_bf16_zeros(self):
        def fn():
            x = torch.zeros(1, 1, 32, dtype=torch.bfloat16)
            return x

        self.common(fn, ())

    def test_select_tiliing_with_index_expr(self):
        def fn(x, y):
            x = torch.ops.aten.view.default(x, [8, 8, 8, 3136])
            x = torch.ops.aten.permute.default(x, [0, 1, 3, 2])
            y = torch.ops.aten.mul.Tensor(y, x)
            return torch.ops.aten.constant_pad_nd.default(y, [0, 0, 1, 0, 0, 0], 0.0)

        x = torch.randn(8, 64, 56, 56)
        y = torch.randn(8, 8, 3136, 8)
        self.common(fn, (x, y))

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    @config.patch(freezing=True)
    def test_linear_with_no_default_contiguous_input(self):
        dtypes = [
            torch.float32,
        ]
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        mod = torch.nn.Sequential(torch.nn.Linear(16, 16)).eval()
        temp = torch.randn(1, 16, 1, 1)
        v = torch.as_strided(temp, [1, 16], [0, 1], 0)
        self.assertTrue(v.is_contiguous())
        for dtype in dtypes:
            with torch.no_grad():
                self.common(
                    mod.to(dtype),
                    (v.to(dtype),),
                )

    @patch("torch.cuda.is_available", lambda: False)
    @config.patch(freezing=True)
    def test_linear_with_reshape(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(16, 16, bias=False)

            def forward(self, x):
                x = self.linear(x)
                return x.view(4, 4, 4)

        mod = M().eval()
        v = torch.randn(4, 16)
        with torch.no_grad():
            torch._dynamo.reset()
            metrics.reset()
            self.common(
                mod,
                (v,),
            )
            assert metrics.generated_kernel_count == 0

    @config.patch(implicit_fallbacks=True)
    def test_aten_normal_dtype(self):
        for dtype in [torch.float64, torch.float16, None]:

            def fn():
                return torch.normal(2, 3, (10, 10), dtype=dtype, device="cpu")

            self.assertEqual(
                torch.compile(fn, backend="aot_eager_decomp_partition")().dtype,
                dtype if dtype else torch.float32,
            )
            self.assertEqual(
                torch.compile(fn, backend="inductor")().dtype,
                dtype if dtype else torch.float32,
            )

    def test_group_norm_vec(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.group_norm = torch.nn.GroupNorm(3, 90)

            def forward(self, x):
                return self.group_norm(x)

        options = itertools.product(
            vec_dtypes, [torch.contiguous_format, torch.channels_last], [True, False]
        )
        for dtype, fmt, dynamic in options:
            torch._dynamo.reset()
            metrics.reset()
            mod = M().eval()
            x = torch.randn((2, 90, 6, 6), dtype=dtype).to(memory_format=fmt)
            with torch.no_grad():
                expected = mod(x)
                compiled_m = torch.compile(mod, dynamic=dynamic)
                actual, code = run_and_get_cpp_code(compiled_m, x)
                self.assertEqual(expected, actual)
                # 3 generated kernels (first one for var_mean, last two for result)
                check_metrics_vec_kernel_count(3)

                # check loop split optimization
                if fmt == torch.channels_last:
                    # check that there are no non_contiguous loads
                    FileCheck().check_count(
                        "__at_align__ std::array", 0, exactly=True
                    ).run(code)

    @unittest.skipIf(
        os.getenv("ATEN_CPU_CAPABILITY") == "default",
        "Failing in periodic nogpu_NO_AVX2, see #150059 for example",
    )
    def test_group_norm_large_input(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.group_norm = torch.nn.GroupNorm(2, 64)

            def forward(self, x):
                return self.group_norm(x)

        for fmt in [torch.contiguous_format, torch.channels_last]:
            torch._dynamo.reset()
            metrics.reset()
            mod = M().eval()
            x = torch.randn(2, 64, 168, 168).to(memory_format=fmt)
            with torch.no_grad():
                expected = mod(x)
                compiled_m = torch.compile(mod)
                actual = compiled_m(x)
                self.assertEqual(expected, actual)
                # 3 generated kernels (first one for var_mean, last two for result)
                check_metrics_vec_kernel_count(3)
                # check that there is no outer loop fusion.
                self.assertEqual(
                    len(metrics.cpp_outer_loop_fused_inner_counts),
                    0,
                )
                # check for parallel reduction.
                self.assertEqual(metrics.parallel_reduction_count, 1)

    @unittest.skipIf(
        os.getenv("ATEN_CPU_CAPABILITY") == "default",
        "Failing in periodic nogpu_NO_AVX2, see #150059 for example",
    )
    def test_group_norm_large_size(self):
        # https://github.com/pytorch/pytorch/issues/141541
        # We are using the chunk size of 4096 for cascade summation,
        # the reduction size of this test case exceeded it.
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gn = torch.nn.GroupNorm(32, 32)

            def forward(self, x):
                return self.gn(x)

        for simdlen, dynamic in itertools.product([None, 0], [True, False]):
            with config.patch({"cpp.simdlen": simdlen}):
                torch._dynamo.reset()
                metrics.reset()
                mod = M().eval()
                x = torch.randn(1, 32, 128, 128, 128)
                with torch.no_grad():
                    expected = mod(x)
                    compiled_m = torch.compile(mod, dynamic=dynamic)
                    actual = compiled_m(x)
                    self.assertEqual(expected, actual)

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    @config.patch(emulate_precision_casts=True)
    def test_group_norm_backward_symint_divisible_channels(self):
        def fn(x, weight, bias):
            y = torch.nn.functional.group_norm(x, 1, weight=weight, bias=bias)
            return torch.sigmoid(y.max(dim=0).values)

        torch._dynamo.reset()
        metrics.reset()

        shape = (2, 33, 4, 5)
        x_ref = torch.rand(shape, dtype=torch.float32, requires_grad=True)
        weight_ref = torch.rand((33,), dtype=torch.float32, requires_grad=True)
        bias_ref = torch.rand((33,), dtype=torch.float32, requires_grad=True)

        x_cmp = x_ref.clone().detach().requires_grad_(True)
        weight_cmp = weight_ref.clone().detach().requires_grad_(True)
        bias_cmp = bias_ref.clone().detach().requires_grad_(True)

        eager_out = fn(x_ref, weight_ref, bias_ref)
        eager_out.sum().backward()

        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)
        compiled_out = compiled(x_cmp, weight_cmp, bias_cmp)
        compiled_out.sum().backward()

        torch.testing.assert_close(compiled_out, eager_out)
        torch.testing.assert_close(x_cmp.grad, x_ref.grad)
        torch.testing.assert_close(weight_cmp.grad, weight_ref.grad)
        torch.testing.assert_close(bias_cmp.grad, bias_ref.grad)

    @config.patch(emulate_precision_casts=True)
    def test_emulate_precision_casts_cpp_backend_no_error(self):
        """
        See https://github.com/pytorch/pytorch/issues/167205
        emulate_precision_casts threw TypeError on CPP backend.

        Before fix: TypeError: CppVecOverrides.to_dtype() got an unexpected
        keyword argument 'use_compute_types'

        After fix: Should compile and run without error.
        """

        def robust_power(base, exponent, threshold):
            threshold1 = threshold
            broadcasted_base = torch.abs(base)
            threshold_bc = threshold.expand_as(base)
            cond = broadcasted_base < threshold_bc
            return torch.where(cond, base / threshold1, base**exponent)

        device = torch.device("cpu")
        base = torch.randn(10, dtype=torch.float16, device=device)
        exponent = torch.tensor(2.0, dtype=torch.float16, device=device)
        threshold = torch.tensor(0.01, dtype=torch.float16, device=device)
        v = torch.ones_like(base)

        # Main test, this should not raise TypeError (before fix it would)
        compiled_fn = torch.compile(robust_power)
        y, (grad_b, grad_e, grad_t) = vjp(
            lambda b, e, t: compiled_fn(b, e, t), (base, exponent, threshold), v=v
        )

        # Sanity check that gradients were computed
        self.assertIsNotNone(grad_b)
        self.assertEqual(grad_b.dtype, torch.float16)

    def test_int_div_vec(self):
        def fn(x, y, mode):
            return torch.div(x, y, rounding_mode=mode)

        for dtype in [
            torch.int8,
            torch.uint8,
            torch.int32,
            torch.int64,
        ]:
            x = torch.randint(1, 100, (32, 32), dtype=dtype)
            y = torch.randint(1, 100, (32, 32), dtype=dtype)
            for mode in [None, "trunc", "floor"]:
                with torch.no_grad():
                    metrics.reset()
                    self.common(fn, (x, y, mode))
                    check_metrics_vec_kernel_count(1)

    def test_uint8_add(self):
        # https://github.com/pytorch/pytorch/issues/113016
        def fn(x, y):
            return torch.add(x, y).neg().to(torch.int32)

        x = torch.randint(0, 255, (3, 3), dtype=torch.uint8)
        y = torch.randint(0, 255, (3, 3), dtype=torch.uint8)
        self.common(fn, (x, y))

    def test_uint8_sub(self):
        # https://github.com/pytorch/pytorch/issues/113016
        def fn(x, y):
            return torch.sub(x, y).neg().to(torch.int32)

        x = torch.randint(0, 255, (3, 3), dtype=torch.uint8)
        y = torch.randint(0, 255, (3, 3), dtype=torch.uint8)
        self.common(fn, (x, y))

    def test_float32_to_uint8(self):
        # https://github.com/pytorch/pytorch/issues/156788
        @torch.compile
        def fn(x):
            return x.to(torch.uint8)

        x = torch.tensor([-1.0, -2.0, -3.0, -4.0], dtype=torch.float32, device="cpu")
        self.assertEqual(
            x.to(torch.uint8),
            fn(x),
            msg=f"Expected {x.to(torch.uint8)} but got {fn(x)}",
        )

    def test_non_contiguous_reduction_store(self):
        # https://github.com/pytorch/pytorch/issues/113018
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(39, 1, kernel_size=(1, 17), stride=(2, 2))

            def forward(self, x):
                return self.conv(x.max(3).values)

        m = M()
        x = torch.randn(1, 39, 1, 18, 17)
        self.common(m, (x,))

    def test_embedding_vec(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = torch.nn.Embedding(64, 128)

            def forward(self, idx, x):
                return self.emb(idx) + x

        idx = torch.randint(0, 64, (4, 32))
        x = torch.randn(4, 32, 128)
        m = M().eval()
        with torch.no_grad():
            metrics.reset()
            self.common(m, (idx, x))
            check_metrics_vec_kernel_count(1)

    @requires_vectorization
    def test_embedding_vec_bf16(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = torch.nn.Embedding(64, 128)

            def forward(self, idx, x):
                return self.emb(idx)

        idx = torch.randint(0, 64, (4, 32))
        x = torch.randn(4, 32, 128).to(torch.bfloat16)
        m = M().eval()
        with torch.no_grad():
            metrics.reset()
            self.common(m, (idx, x))
            check_metrics_vec_kernel_count(1)

        # we are doing direct load/store, make sure we do not generate
        # redundant type casts
        m_opt = torch.compile(m)
        _, code = run_and_get_cpp_code(m_opt, idx, x)
        self.assertTrue("Vectorized" in code)
        self.assertTrue("cvt_lowp_fp_to_fp32" not in code)
        self.assertTrue("cvt_fp32_to_lowp_fp" not in code)

    @xfailIfS390X
    def test_concat_inner_vec(self):
        def fn(x, y):
            return F.relu(torch.cat([x, y], dim=1))

        x = torch.randn(32, 35)
        y = torch.randn(32, 120)
        metrics.reset()
        self.common(fn, (x, y))
        check_metrics_vec_kernel_count(3)

    @config.patch("cpp.enable_tiling_heuristics", False)
    def test_expr_vec_non_contiguous(self):
        def fn(x):
            # the pattern from sebotnet33ts_256
            y = torch.nn.functional.pad(x, (0, 31)).reshape(-1, 33, 63)
            y = y[:, :32, 31:].reshape(4, 32, 1, 32, 32).expand(-1, -1, 32, -1, -1)
            y = y.permute(0, 3, 1, 4, 2).clone(memory_format=torch.contiguous_format)
            y = y.view(4, 1024, 1024)
            return y.softmax(dim=-1)

        x = torch.randn(128, 2048)
        opt_fn = torch.compile(fn)
        metrics.reset()
        _, code = run_and_get_cpp_code(opt_fn, x)
        self.assertTrue(same(fn(x), opt_fn(x)))
        # 4 kernels for max, exp, sum and div
        check_metrics_vec_kernel_count(4)
        FileCheck().check_count(
            "Vectorized<int>::loadu(tmpbuf.data())", 0, exactly=True
        ).run(code)

    def test_vec_contiguous_ModularIndexing(self):
        # https://github.com/pytorch/pytorch/issues/114488
        class M(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.norm = torch.nn.LayerNorm(dim * 4)

            def forward(self, x):
                # the pattern from swin_base_patch4_window7_224
                B, H, W, C = x.shape
                x = (
                    x.reshape(B, H // 2, 2, W // 2, 2, C)
                    .permute(0, 1, 3, 4, 2, 5)
                    .flatten(3)
                )
                x = self.norm(x)
                return x

        x = torch.randn(1, 56, 56, 128)
        m = M(128)
        opt_m = torch.compile(m)
        with torch.no_grad():
            metrics.reset()
            _, code = run_and_get_cpp_code(opt_m, x)
            self.assertTrue(same(m(x), opt_m(x)))
            # Two kernels: one for reduction, one pointwises
            check_metrics_vec_kernel_count(2)
            FileCheck().check_count(
                "Vectorized<float>::loadu(tmpbuf.data())", 0, exactly=True
            ).run(code)

    @parametrize("dtype", (torch.float16, torch.bfloat16, torch.float))
    @parametrize("shape", ("15,3,13", "4,2048,4096"))
    def test_fp8_cast(self, dtype: torch.dtype, shape: str):
        def fp8_cast(x):
            y0 = x.to(dtype=torch.float8_e4m3fn).to(dtype)
            y1 = x.to(dtype=torch.float8_e5m2).to(dtype)
            return y0, y1

        shape = [int(dim) for dim in shape.split(",")]
        x = torch.rand(*shape, device="cpu", dtype=dtype)
        self.common(fp8_cast, (x,))

    def test_logical_op_store_to_lowp_data_dtype(self):
        # https://github.com/pytorch/pytorch/issues/117624
        # https://github.com/pytorch/pytorch/issues/117627
        def fn(out1, out2, input, other):
            o1 = torch.logical_or(out=out1, input=input, other=other)
            o2 = torch.logical_xor(out=out2, input=input, other=other)
            return o1, o2

        x = torch.rand([3, 3, 2, 8, 9, 2], dtype=torch.float)
        y = torch.rand([3, 3, 2, 8, 9, 2], dtype=torch.float)
        for dtype in _lowp_fp_dtypes:
            o1 = torch.rand([3, 3, 2, 8, 9, 2], dtype=dtype)
            o2 = torch.rand([3, 3, 2, 8, 9, 2], dtype=dtype)
            with torch.no_grad():
                self.common(fn, (o1, o2, x, y))

    def test_constant_bool_vec(self):
        def fn(x):
            mask = torch.zeros(1, dtype=torch.bool)
            return torch.where(mask, x, -1.0)

        x = torch.rand(1000)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

        # Tail vectorization case
        x = torch.rand(37)
        torch._dynamo.reset()
        metrics.reset()
        with torch.no_grad():
            expected = fn(x)
            compiled_fn = torch.compile(fn)
            actual, code = run_and_get_cpp_code(compiled_fn, x)
            self.assertEqual(expected, actual)
            # 1 generated vec kernel
            check_metrics_vec_kernel_count(1)
            # Check that both main and tail loops are vectorized
            if _can_check_vec_metrics():
                FileCheck().check_count(
                    "at::vec::VecMask<float,1>::from", 2, exactly=True
                ).run(code)

    @torch._dynamo.config.patch(dynamic_shapes=True)
    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_symbolic_shape_scalar_value_reduction(self):
        def fn(x, y):
            return y + torch.ones(x).sum()

        with torch.no_grad():
            metrics.reset()
            y = torch.randn(100)
            self.common(fn, (100, y))
            check_metrics_vec_kernel_count(2)

    def test_int32_pointwise_vec(self):
        def fn(x):
            return x * x

        x = torch.randint(0, 100, (32, 32), dtype=torch.int32)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

        # Tail vectorization case
        x = torch.randint(0, 100, (37, 37), dtype=torch.int32)
        torch._dynamo.reset()
        metrics.reset()
        with torch.no_grad():
            expected = fn(x)
            compiled_fn = torch.compile(fn)
            actual, code = run_and_get_cpp_code(compiled_fn, x)
            self.assertEqual(expected, actual)
            # 1 generated vec kernel
            check_metrics_vec_kernel_count(1)
            # Check that both main and tail loops are vectorized
            if _can_check_vec_metrics():
                FileCheck().check_count(
                    "at::vec::Vectorized<int32_t>::loadu", 2, exactly=True
                ).run(code)

    def test_int32_reduction_vec(self):
        def fn(x):
            return x.sum(dim=1)

        x = torch.randint(0, 100, (32, 32), dtype=torch.int32)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

        # Tail vectorization case
        x = torch.randint(0, 100, (37, 37), dtype=torch.int32)
        torch._dynamo.reset()
        metrics.reset()
        with torch.no_grad():
            expected = fn(x)
            compiled_fn = torch.compile(fn)
            actual, code = run_and_get_cpp_code(compiled_fn, x)
            self.assertEqual(expected, actual)
            # 1 generated vec kernel
            check_metrics_vec_kernel_count(1)
            # Check that both main and tail loops are vectorized
            if _can_check_vec_metrics():
                FileCheck().check_count(
                    "at::vec::Vectorized<int32_t>::loadu", 2, exactly=True
                ).run(code)

    def test_uint32_pointwise_vec(self):
        def fn(x):
            return x * x

        x = torch.randint(0, 100, (32, 32), dtype=torch.uint32)
        metrics.reset()
        self.common(fn, (x,))
        # TODO(jgong5): change to 1 with vectorized uint32 load
        assert metrics.generated_cpp_vec_kernel_count == 0

    def test_uint32_reduction_vec(self):
        def fn(x):
            return x.sum(dim=1)

        x = torch.randint(0, 100, (32, 32), dtype=torch.uint32)
        metrics.reset()
        self.common(fn, (x,))
        # TODO(jgong5): change to 1 with vectorized uint32/uint64 load
        assert metrics.generated_cpp_vec_kernel_count == 0

    def test_int64_pointwise_vec(self):
        def fn(x):
            return x * x

        x = torch.randint(0, 100, (32, 32), dtype=torch.int64)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

        # Tail vectorization case
        x = torch.randint(0, 100, (37, 37), dtype=torch.int64)
        torch._dynamo.reset()
        metrics.reset()
        with torch.no_grad():
            expected = fn(x)
            compiled_fn = torch.compile(fn)
            actual, code = run_and_get_cpp_code(compiled_fn, x)
            self.assertEqual(expected, actual)
            # 1 generated vec kernel
            check_metrics_vec_kernel_count(1)
            # Check that both main and tail loops are vectorized
            if _can_check_vec_metrics():
                FileCheck().check_count(
                    "at::vec::VectorizedN<int64_t,2>::loadu", 2, exactly=True
                ).run(code)

    def test_int64_reduction_vec(self):
        def fn(x):
            return x.sum(dim=1)

        x = torch.randint(0, 100, (32, 32), dtype=torch.int64)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

        # Tail vectorization case
        x = torch.randint(0, 100, (37, 37), dtype=torch.int64)
        torch._dynamo.reset()
        metrics.reset()
        with torch.no_grad():
            expected = fn(x)
            compiled_fn = torch.compile(fn)
            actual, code = run_and_get_cpp_code(compiled_fn, x)
            self.assertEqual(expected, actual)
            # 1 generated vec kernel
            check_metrics_vec_kernel_count(1)
            # Check that both main and tail loops are vectorized
            if _can_check_vec_metrics():
                FileCheck().check_count(
                    "at::vec::VectorizedN<int64_t,2>::loadu", 2, exactly=True
                ).run(code)

    def test_uint64_pointwise_vec(self):
        def fn(x):
            return x * x

        x = torch.randint(0, 100, (32, 32), dtype=torch.uint64)
        metrics.reset()
        self.common(fn, (x,))
        # TODO(jgong5): change to 1 with vectorized uint64 load
        assert metrics.generated_cpp_vec_kernel_count == 0

    def test_uint64_reduction_vec(self):
        def fn(x):
            return x.sum(dim=1)

        x = torch.randint(0, 100, (32, 32), dtype=torch.uint64)
        metrics.reset()
        self.common(fn, (x,))
        # TODO(jgong5): change to 1 with vectorized uint64 load
        assert metrics.generated_cpp_vec_kernel_count == 0

    def test_convert_int8_to_half_vec(self):
        src_dtypes = [torch.int8, torch.uint8]
        dst_dtypes = [torch.bfloat16, torch.half]
        _simd_lens = [isa._bit_width for isa in cpu_vec_isa.valid_vec_isa_list()]
        for src_dtype, dst_dtype, _simd_len in itertools.product(
            src_dtypes, dst_dtypes, _simd_lens
        ):

            def fn(x):
                return x.to(dst_dtype)

            low = 0 if src_dtype == torch.uint8 else -100

            x = torch.randint(low, 100, (32, 32), dtype=src_dtype)
            with config.patch({"cpp.simdlen": _simd_len}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x,))
                check_metrics_vec_kernel_count(1)

    def test_convert_int32_to_int64_vec(self):
        def fn(x):
            return x.to(torch.int64)

        x = torch.randint(0, 100, (32, 32), dtype=torch.int32)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

    def test_convert_int64_to_int32_vec(self):
        def fn(x):
            return x.to(torch.int32)

        x = torch.randint(0, 100, (32, 32), dtype=torch.int64)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

    def test_convert_fp32_to_int64_vec(self):
        def fn(x):
            return x.to(torch.int64)

        x = torch.rand(32, 32)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

    def test_convert_int64_to_fp32_vec(self):
        def fn(x):
            return x.to(torch.float32)

        x = torch.randint(0, 100, (32, 32), dtype=torch.int64)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

    def test_double_pointwise_vec(self):
        def fn(x):
            return x * x

        x = torch.randn((32, 32), dtype=torch.double)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

        # Tail vectorization case
        x = torch.randn((37, 37), dtype=torch.double)
        torch._dynamo.reset()
        metrics.reset()
        with torch.no_grad():
            expected = fn(x)
            compiled_fn = torch.compile(fn)
            actual, code = run_and_get_cpp_code(compiled_fn, x)
            self.assertEqual(expected, actual)
            # 1 generated vec kernel
            check_metrics_vec_kernel_count(1)
            # Check that both main and tail loops are vectorized
            if _can_check_vec_metrics():
                FileCheck().check_count(
                    "at::vec::VectorizedN<double,2>::loadu", 2, exactly=True
                ).run(code)

    def test_double_reduction_vec(self):
        def fn(x):
            return x.sum(dim=1)

        x = torch.randn((32, 32), dtype=torch.double)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

        # Tail vectorization case
        x = torch.randn((37, 37), dtype=torch.double)
        torch._dynamo.reset()
        metrics.reset()
        with torch.no_grad():
            expected = fn(x)
            compiled_fn = torch.compile(fn)
            actual, code = run_and_get_cpp_code(compiled_fn, x)
            self.assertEqual(expected, actual)
            # 1 generated vec kernel
            check_metrics_vec_kernel_count(1)
            # Check that both main and tail loops are vectorized
            if _can_check_vec_metrics():
                FileCheck().check_count(
                    "at::vec::VectorizedN<double,2>::loadu", 2, exactly=True
                ).run(code)

    def test_convert_fp32_to_double_vec(self):
        def fn(x):
            return x.to(torch.double)

        x = torch.randn(32, 32)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

        # Tail vectorization case
        x = torch.randn(37, 37)
        torch._dynamo.reset()
        metrics.reset()
        with torch.no_grad():
            expected = fn(x)
            compiled_fn = torch.compile(fn)
            actual, code = run_and_get_cpp_code(compiled_fn, x)
            self.assertEqual(expected, actual)
            # 1 generated vec kernel
            check_metrics_vec_kernel_count(1)
            # Check that both main and tail loops are vectorized
            if _can_check_vec_metrics():
                FileCheck().check_count(
                    "at::vec::convert<double,2,float,1>", 2, exactly=True
                ).run(code)

    def test_convert_double_to_fp32_vec(self):
        def fn(x):
            return x.to(torch.float32)

        x = torch.randn((32, 32), dtype=torch.double)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

        # Tail vectorization case
        x = torch.randn((37, 37), dtype=torch.double)
        torch._dynamo.reset()
        metrics.reset()
        with torch.no_grad():
            expected = fn(x)
            compiled_fn = torch.compile(fn)
            actual, code = run_and_get_cpp_code(compiled_fn, x)
            self.assertEqual(expected, actual)
            # 1 generated vec kernel
            check_metrics_vec_kernel_count(1)
            # Check that both main and tail loops are vectorized
            if _can_check_vec_metrics():
                FileCheck().check_count(
                    "at::vec::convert<float,1,double,2>", 2, exactly=True
                ).run(code)

    def test_no_redundant_to_dtypes_between_fused_scheduler_node(self):
        # https://github.com/pytorch/pytorch/issues/115260
        p0 = torch.tensor([1.0879], dtype=torch.float16)

        class Model1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, *args):
                cat = torch.cat((args[3], args[2], args[1], args[0]), dim=2)
                max_1 = torch.max(args[4], p0)
                mul = torch.mul(cat, max_1)
                tan = torch.tan(mul)
                return (mul, tan)

        metrics.reset()
        m = Model1()
        self.common(
            m,
            (
                torch.randn((17, 5, 1, 7)).half(),
                torch.randn((17, 5, 1, 7)).half(),
                torch.randn((17, 5, 11, 7)).half(),
                torch.randn((17, 5, 1, 7)).half(),
                torch.tensor(4.39, dtype=torch.float16),
            ),
        )

    def test_masked_load_int64_vec(self):
        # https://github.com/pytorch/pytorch/issues/120377
        def fn(x):
            return torch.nn.functional.pad(x, (0, 13))

        x = torch.randint(0, 100, (819,), dtype=torch.int64)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

    def test_highp_to_lowp_cse_var_cache_with_store(self):
        # Fix issue: https://github.com/pytorch/pytorch/issues/128263
        input = torch.randn(5, 128, dtype=torch.float32)
        input2 = torch.randint(0, 10, (5, 128), dtype=torch.int8)
        input3 = torch.randn(128, 128, dtype=torch.float32)

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, x2, x3):
                x2 = x2.to(torch.int32)
                temp = test_operators.realize(x2.to(torch.float16))
                temp2 = temp.to(torch.float32)
                temp2 = temp2 * x
                return torch.mm(temp, x3.to(torch.float16)), temp2

        metrics.reset()
        m = Model()
        self.common(
            m,
            (input, input2, input3),
        )

    def test_reduction_float_to_int64(self):
        # https://github.com/pytorch/pytorch/issues/124821
        def fn(x):
            return x.max(0).values

        x = torch.randint(0, 100, (22, 51), dtype=torch.int64)
        metrics.reset()
        self.common(fn, (x,))
        check_metrics_vec_kernel_count(1)

    @config.patch({"cpp.dynamic_threads": True})
    def test_reduction_with_dynamic_threads(self):
        def fn(a, b):
            return a.sum(), b.sum()

        self.common(
            fn,
            (torch.randn(1000), torch.rand(1000)),
        )

    @patch("torch.cuda.is_available", lambda: False)
    @config.patch(freezing=True)
    def test_linear_float64(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight1 = torch.nn.Parameter(
                    torch.randn(10, 10, dtype=torch.float64)
                )
                self.weight2 = torch.nn.Parameter(
                    torch.randn(10, 10, dtype=torch.float64)
                )
                self.bias = torch.nn.Parameter(torch.randn(10, dtype=torch.float64))

            def forward(self, x1):
                v1 = torch.mm(x1, self.weight1)
                v2 = torch.addmm(self.bias, x1, self.weight2)
                return (v1, v2)

        mod = M().eval()
        v = torch.randn(10, 10, dtype=torch.float64)
        with torch.no_grad():
            self.common(
                mod,
                (v,),
            )

    def test_fused_attention_conv(self):
        # https://github.com/pytorch/pytorch/issues/121174.
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_conv = torch.nn.Conv2d(4, 4, 1)
                self.k_conv = torch.nn.Conv2d(4, 4, 1)
                self.v_conv = torch.nn.Conv2d(4, 4, 1)

            def forward(self, x):
                q = self.q_conv(x)
                k = self.k_conv(x)
                v = self.v_conv(x)
                q = q.permute(0, 2, 1, 3)
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, is_causal=False
                )

        fn = Model()
        x = torch.randn(1, 4, 2, 2)
        self.common(fn, (x,))

    @parametrize("is_inference", (True, False))
    def test_disabled_amp(self, is_inference):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.all_head_size = 12 * 64
                self.dense = nn.Linear(self.all_head_size, self.all_head_size)

            def forward(self, q, k, v):
                context_layer = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=0.2
                )
                context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
                new_context_layer_shape = context_layer.size()[:-2] + (
                    self.all_head_size,
                )
                context_layer = context_layer.view(new_context_layer_shape)
                return self.dense(context_layer)

        mod = M().to(torch.bfloat16).eval()

        q = torch.randn((4, 12, 512, 64), dtype=torch.bfloat16) / 10.0
        k = torch.randn((4, 12, 512, 64), dtype=torch.bfloat16) / 10.0
        v = torch.randn((4, 12, 512, 64), dtype=torch.bfloat16) / 10.0
        inputs = (
            q,
            k,
            v,
        )
        compiler_mode = torch.compile(mod)
        from torch.nn.attention import sdpa_kernel, SDPBackend

        context = contextlib.nullcontext if not is_inference else torch.no_grad
        with (
            config.patch({"fallback_random": True}),
            torch.cpu.amp.autocast(),
            context(),
            sdpa_kernel(SDPBackend.MATH),
        ):
            torch.manual_seed(0)
            eager = mod(*inputs)
            torch.manual_seed(0)
            self.assertEqual(compiler_mode(*inputs), eager)

    def test_fused_node(self):
        # https://github.com/pytorch/pytorch/issues/138550.
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(
                self,
                clone_50,
                gt_scalar,
                div_tensor,
                convert_element_type_default_7,
                convert_element_type_default_13,
                convert_element_type_default_14,
            ):
                convert_element_type_default_4 = (
                    torch.ops.prims.convert_element_type.default(
                        clone_50, torch.float32
                    )
                )
                clone_50 = None
                view_default_6 = torch.ops.aten.view.default(
                    convert_element_type_default_4, [336, 512, 64]
                )
                convert_element_type_default_4 = None
                convert_element_type_default_5 = (
                    torch.ops.prims.convert_element_type.default(
                        view_default_6, torch.bfloat16
                    )
                )
                view_default_6 = None
                mul_tensor = torch.ops.aten.mul.Tensor(gt_scalar, div_tensor)
                mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112)
                mul_tensor = None
                expand_default_2 = torch.ops.aten.expand.default(
                    mul_tensor_1, [28, 12, 512, 512]
                )
                mul_tensor_1 = None
                view_default_3 = torch.ops.aten.view.default(
                    expand_default_2, [336, 512, 512]
                )
                expand_default_2 = None
                permute_default_4 = torch.ops.aten.permute.default(
                    view_default_3, [0, 2, 1]
                )
                view_default_3 = None
                convert_element_type_default_6 = (
                    torch.ops.prims.convert_element_type.default(
                        permute_default_4, torch.bfloat16
                    )
                )
                permute_default_4 = None
                bmm_default_2 = torch.ops.aten.bmm.default(
                    convert_element_type_default_6, convert_element_type_default_5
                )
                convert_element_type_default_6 = None
                convert_element_type_default_10 = (
                    torch.ops.prims.convert_element_type.default(
                        bmm_default_2, torch.float32
                    )
                )
                bmm_default_2 = None
                view_default_7 = torch.ops.aten.view.default(
                    convert_element_type_default_10, [28, 12, 512, 64]
                )
                convert_element_type_default_10 = None
                convert_element_type_default_18 = (
                    torch.ops.prims.convert_element_type.default(
                        view_default_7, torch.bfloat16
                    )
                )
                view_default_7 = None
                permute_default_9 = torch.ops.aten.permute.default(
                    convert_element_type_default_18, [0, 2, 1, 3]
                )
                convert_element_type_default_18 = None
                bmm_default_3 = torch.ops.aten.bmm.default(
                    convert_element_type_default_5, convert_element_type_default_7
                )
                convert_element_type_default_5 = convert_element_type_default_7 = None
                convert_element_type_default_9 = (
                    torch.ops.prims.convert_element_type.default(
                        bmm_default_3, torch.float32
                    )
                )
                bmm_default_3 = None
                view_default_8 = torch.ops.aten.view.default(
                    convert_element_type_default_9, [28, 12, 512, 512]
                )
                convert_element_type_default_9 = None
                convert_element_type_default_11 = (
                    torch.ops.prims.convert_element_type.default(
                        gt_scalar, torch.float32
                    )
                )
                gt_scalar = None
                mul_tensor_2 = torch.ops.aten.mul.Tensor(
                    convert_element_type_default_11, 1.1111111111111112
                )
                convert_element_type_default_11 = None
                mul_tensor_3 = torch.ops.aten.mul.Tensor(view_default_8, mul_tensor_2)
                view_default_8 = mul_tensor_2 = None
                mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, div_tensor)
                mul_tensor_3 = None
                sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(
                    mul_tensor_4, [-1], True
                )
                neg_default = torch.ops.aten.neg.default(div_tensor)
                div_tensor = None
                fma_default = torch.ops.prims.fma.default(
                    neg_default, sum_dim_int_list_1, mul_tensor_4
                )
                neg_default = sum_dim_int_list_1 = mul_tensor_4 = None
                view_default_9 = torch.ops.aten.view.default(
                    fma_default, [336, 512, 512]
                )
                fma_default = None
                convert_element_type_default_12 = (
                    torch.ops.prims.convert_element_type.default(
                        view_default_9, torch.bfloat16
                    )
                )
                view_default_9 = None
                bmm_default_4 = torch.ops.aten.bmm.default(
                    convert_element_type_default_13, convert_element_type_default_12
                )
                convert_element_type_default_13 = None
                convert_element_type_default_17 = (
                    torch.ops.prims.convert_element_type.default(
                        bmm_default_4, torch.float32
                    )
                )
                bmm_default_4 = None
                view_default_10 = torch.ops.aten.view.default(
                    convert_element_type_default_17, [28, 12, 64, 512]
                )
                convert_element_type_default_17 = None
                mul_scalar_2 = torch.ops.aten.mul.Scalar(
                    view_default_10, 0.3535533905932738
                )
                view_default_10 = None
                permute_default_8 = torch.ops.aten.permute.default(
                    mul_scalar_2, [0, 1, 3, 2]
                )
                mul_scalar_2 = None
                convert_element_type_default_19 = (
                    torch.ops.prims.convert_element_type.default(
                        permute_default_8, torch.bfloat16
                    )
                )
                permute_default_8 = None
                _permute_default_10 = torch.ops.aten.permute.default(
                    convert_element_type_default_19, [0, 2, 1, 3]
                )
                convert_element_type_default_19 = None
                bmm_default_5 = torch.ops.aten.bmm.default(
                    convert_element_type_default_12, convert_element_type_default_14
                )
                convert_element_type_default_12 = convert_element_type_default_14 = None
                convert_element_type_default_16 = (
                    torch.ops.prims.convert_element_type.default(
                        bmm_default_5, torch.float32
                    )
                )
                bmm_default_5 = None
                view_default_11 = torch.ops.aten.view.default(
                    convert_element_type_default_16, [28, 12, 512, 64]
                )
                convert_element_type_default_16 = None
                mul_scalar_3 = torch.ops.aten.mul.Scalar(
                    view_default_11, 0.3535533905932738
                )
                view_default_11 = None
                convert_element_type_default_20 = (
                    torch.ops.prims.convert_element_type.default(
                        mul_scalar_3, torch.bfloat16
                    )
                )
                mul_scalar_3 = None
                permute_default_11 = torch.ops.aten.permute.default(
                    convert_element_type_default_20, [0, 2, 1, 3]
                )
                convert_element_type_default_20 = None
                clone_52 = torch.ops.aten.clone.default(
                    permute_default_11, memory_format=torch.contiguous_format
                )
                permute_default_11 = None
                view_283 = torch.ops.aten.view.default(clone_52, [28, 512, 768])
                clone_52 = None
                clone_53 = torch.ops.aten.clone.default(
                    permute_default_9, memory_format=torch.contiguous_format
                )
                permute_default_9 = None
                view_284 = torch.ops.aten.view.default(clone_53, [28, 512, 768])
                clone_53 = None
                view_285 = torch.ops.aten.view.default(view_284, [14336, 768])
                view_284 = None
                return view_283, view_285

        clone_50 = torch.randn((28, 12, 512, 64), dtype=torch.bfloat16) / 10
        gt_scalar = torch.randint(0, 2, (28, 12, 512, 512), dtype=torch.bool)
        div_tensor = torch.randn((28, 12, 512, 512), dtype=torch.float) / 10
        convert_element_type_default_7 = (
            torch.randn((336, 64, 512), dtype=torch.bfloat16) / 10
        )
        convert_element_type_default_13 = (
            torch.randn((336, 64, 512), dtype=torch.bfloat16) / 10
        )
        convert_element_type_default_14 = (
            torch.randn((336, 512, 64), dtype=torch.bfloat16) / 10
        )
        inputs = (
            clone_50,
            gt_scalar,
            div_tensor,
            convert_element_type_default_7,
            convert_element_type_default_13,
            convert_element_type_default_14,
        )

        with torch.cpu.amp.autocast():
            mod = M().to(torch.bfloat16).eval()
            self.common(mod, inputs, atol=1e-3, rtol=1e-3)

    @requires_vectorization
    def test_vec_indirect_load_cse_cache(self):
        # https://github.com/pytorch/pytorch/issues/123502
        from math import inf

        def fn(arg0_1):
            full_default = torch.ops.aten.full.default([209985], 1)
            select = torch.ops.aten.select.int(arg0_1, 0, 0)
            select_1 = torch.ops.aten.select.int(arg0_1, 0, 1)
            view = torch.ops.aten.reshape.default(select_1, [-1])
            expand = torch.ops.aten.expand.default(view, [209985])
            full_default_1 = torch.ops.aten.full.default([10000], 0)
            scatter_add = torch.ops.aten.scatter_add.default(
                full_default_1, 0, expand, full_default
            )
            pow_1 = torch.ops.aten.pow.Tensor_Scalar(scatter_add, -0.5)
            eq = torch.ops.aten.eq.Scalar(pow_1, inf)
            full_default_2 = torch.ops.aten.full.default([], 0.0)
            where = torch.ops.aten.where.self(eq, full_default_2, pow_1)
            index = torch.ops.aten.index.Tensor(where, [select])
            index_1 = torch.ops.aten.index.Tensor(where, [select_1])
            mul_1 = torch.ops.aten.mul.Tensor(index, index_1)
            return (mul_1,)

        x = torch.zeros(2, 209985).to(torch.int64)
        opt_fn = torch.compile(fn, backend="inductor")
        _, code = run_and_get_cpp_code(opt_fn, x)
        FileCheck().check_count(
            "return at::vec::VectorizedN<int64_t,2>::loadu(tmpbuf.data(),",
            8,
            exactly=True,
        ).run(code)

    def test_load_half(self):
        def fn(arg0_1, arg0_2):
            return arg0_1.copy_(arg0_2)

        with config.patch({"cpp.simdlen": 0}):
            x1 = torch.randn(2, 10).to(torch.half)
            x2 = torch.randn(2, 10).to(torch.half)
            opt_fn = torch.compile(fn, backend="inductor")
            _, code = run_and_get_cpp_code(opt_fn, x1, x2)
            FileCheck().check_count(
                "static_cast<float>",
                0,
                exactly=True,
            ).run(code)

    @requires_vectorization
    def test_repeated_exp(self):
        def fn(x):
            y = x.sigmoid()
            return y + 1, y.sum(-1)

        x = torch.randn(1000, 1000)
        opt_fn = torch.compile(fn)
        _, code = run_and_get_cpp_code(opt_fn, x)
        FileCheck().check_count(
            ".exp()",
            1,
            exactly=True,
        ).run(code)

    def test_convert_fp32_int64_oob_vec(self):
        # https://github.com/pytorch/pytorch/issues/129863
        def fn(x):
            float32 = x.to(torch.float32)
            return float32.to(torch.int64)

        x = torch.full((32,), -9223372036854775808, dtype=torch.int64)

        for simdlen in simd_lengths_to_test():
            with config.patch({"cpp.simdlen": simdlen}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x,))
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    def test_consistent_remove_buffers(self):
        def fn(x):
            z = x + x
            z1 = test_operators.realize(z)
            return x + z1

        # The shape makes sure we generate both vec and scalar kernels
        x = torch.randn((65,), dtype=torch.bfloat16)
        with config.patch(inplace_buffers=False):
            metrics.reset()
            self.common(fn, (x,))
            check_metrics_vec_kernel_count(1)
            _, code = run_and_get_cpp_code(torch.compile(fn), x)
            FileCheck().check_count(
                "tmp1 + tmp2",
                2,
                exactly=True,
            ).run(code)

    @requires_vectorization
    def test_bool_reduction_vec(self):
        for op in (
            torch.any,
            torch.min,
            torch.max,
        ):

            def fn(x1, x2, x3):
                return op(x1), op(x2), op(x3)

            c = [False] * 63
            input1 = torch.Tensor(c).to(torch.bool)
            c[10] = True
            input2 = torch.Tensor(c).to(torch.bool)
            input3 = torch.Tensor([True] * 63).to(torch.bool)
            metrics.reset()
            self.common(
                fn,
                (
                    input1,
                    input2,
                    input3,
                ),
            )
            n_veckernel = 6 if op is torch.masked.mean else 3
            check_metrics_vec_kernel_count(n_veckernel)

    @requires_vectorization
    def test_full_bits_lowp(self):
        def check_use_full_bits(func, shapes, dtype, mixed, check_vecn):
            example_inputs = [torch.randn(shape, dtype=dtype) for shape in shapes]
            if mixed:
                example_inputs[0] = example_inputs[0].to(
                    dtype=torch.half if dtype == torch.bfloat16 else torch.bfloat16
                )
            f_opt = torch.compile()(func)
            _, code = run_and_get_cpp_code(f_opt, *example_inputs)
            if check_vecn:
                self.assertTrue(
                    "at::vec::VectorizedN" in code or "at::vec::convert<float,2" in code
                )
            else:
                self.assertFalse(
                    "at::vec::VectorizedN" in code or "at::vec::convert<float,2" in code
                )

        funcs = []

        def func0(arg0, arg1):
            return torch.ops.aten.sum(
                torch.ops.aten.add(torch.ops.aten.atanh(arg0), arg1), (2, 3)
            )

        funcs.append(func0)

        def func1(arg0):
            v = torch.ops.prims.convert_element_type.default(arg0, torch.float)
            v = torch.ops.aten.add(torch.ops.aten.atanh(arg0), v)
            return torch.ops.prims.convert_element_type.default(v, arg0.dtype)

        funcs.append(func1)

        def func2(arg0, arg1):
            v = torch.ops.aten.atanh(arg0)
            v = torch.ops.aten.add(v, arg1)
            return torch.ops.prims.convert_element_type.default(v, arg1.dtype)

        funcs.append(func2)

        # test small shapes
        funcs.append(func2)
        small_size = cpu_vec_isa.pick_vec_isa().nelements(dtype=torch.bfloat16) // 2

        example_shapes = [
            [(10, 32, 20, 20), (10, 32, 20, 20)],
            [(10, 32, 20, 20)],
            [(10, 32, 20, 20), (10, 32, 20, 20)],
            # test small shapes
            [(small_size), (small_size)],
        ]
        mixed_types = [False, False, True, False]
        check_vecns = [True, True, True, False]

        for dtype in [torch.bfloat16, torch.float16]:
            for func, shapes, mixed, check_vecn in zip(
                funcs, example_shapes, mixed_types, check_vecns
            ):
                check_use_full_bits(func, shapes, dtype, mixed, check_vecn)

    @config.patch("cpp.simdlen", 256)
    @requires_vectorization
    def test_avx2_bool_constant_pad_nd(self):
        # NOTE: I tried using (0, 12, 12) and removing the cpp.simdlen=256 override, but
        # that didn't repro the issue.
        result = torch.testing.make_tensor(
            (0, 6, 6), dtype=torch.bool, device=torch.device("cpu")
        )

        def fn(arg):
            return torch.constant_pad_nd(arg, (1, 1, 1, 1, 1, 1))

        self.common(fn, (result,))

    @config.patch(unroll_reductions_threshold=9999)
    @requires_vectorization
    def test_unrolled_bool_prod_vectorized(self):
        result = torch.zeros((37, 37, 37), dtype=torch.bool)
        dim_select = [0, 1]
        result.narrow(dim_select[0], 0, 1).narrow(dim_select[1], 1, 1).zero_()
        result.narrow(dim_select[0], 2, 1).narrow(dim_select[1], 3, 1).zero_()
        result.narrow(dim_select[0], 4, 1).narrow(dim_select[1], 3, 1).zero_()

        def fn(arg):
            return torch.prod(arg, 1, dtype=torch.bool)

        self.common(fn, (result,))

    @requires_vectorization
    @config.patch("cpp.min_chunk_size", 1)
    def test_for_loop_collapsed(self):
        # https://github.com/pytorch/pytorch/issues/122281
        def fn(x):
            return x.transpose(1, 0).contiguous()

        x = torch.randn(199, 2)
        opt_fn = torch.compile(fn, backend="inductor")
        _, code = run_and_get_cpp_code(opt_fn, x)
        self.assertTrue(same(fn(x), opt_fn(x)))
        FileCheck().check_count("#pragma omp for collapse(2)", 1, exactly=True).run(
            code
        )

    def test_dropout(self):
        class Model(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dropout = eval(f"nn.Dropout{dim}d(p=0.5)")

            def forward(self, x):
                torch.manual_seed(0)
                x = self.dropout(x)
                return x

        for dim in [1, 2, 3]:
            model = Model(dim)
            torch.manual_seed(0)
            shape = [1, 3] + [256] * dim
            x = torch.randn(*shape)
            output = model(x)
            c_model = torch.compile(model)
            c_output = c_model(x)
            self.assertTrue(torch.allclose(output, c_output))

    @requires_vectorization
    def test_bool_max(self):
        torch.manual_seed(777)
        x = torch.randn(size=[128, 2501]).ge(0)

        def fn(x):
            return torch.max(x, 1, False)

        self.common(fn, (x,))

    def test_vector_norm_compile(self):
        x = torch.randn([16, 32], dtype=torch.float)
        ref = torch.linalg.vector_norm(x, ord=2, dim=[], keepdim=False, dtype=None)
        compiled_vector_norm = torch.compile(
            torch.linalg.vector_norm, backend="inductor"
        )
        res = compiled_vector_norm(x, ord=2, dim=[], keepdim=False, dtype=None)
        self.assertEqual(ref, res)

    def test_fractional_max_pool2d_3d_input(self):
        """Test for https://github.com/pytorch/pytorch/issues/156682 - 3D input causing assertion error"""

        # Test various 3D input shapes to ensure the compilation crash is fixed
        test_shapes = [
            (1, 8, 8),  # Original failing case
            (3, 16, 16),  # Different channel count
            (2, 12, 10),  # Non-square input
            (5, 20, 20),  # Larger input
        ]

        for shape in test_shapes:
            with self.subTest(shape=shape):
                torch.manual_seed(42)
                x = torch.randn(shape)

                # Generate explicit samples to ensure deterministic, correct results
                n_batch = 1 if x.dim() == 3 else x.size(0)
                torch.manual_seed(42)
                samples = torch.rand(
                    n_batch, x.size(-3), 2, dtype=x.dtype, device=x.device
                )

                def fn(x, samples):
                    return F.fractional_max_pool2d(
                        x, kernel_size=3, output_size=(4, 4), _random_samples=samples
                    )

                # Test that eager mode works
                expected = fn(x, samples)

                # Test that compiled mode works (was failing with AssertionError before fix)
                compiled_fn = torch.compile(fn, backend="inductor")
                result = compiled_fn(x, samples)

                # Verify correctness with explicit samples (should match exactly)
                torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_outer_looop_fusion_with_local_buf(self):
        def fn(
            xs: torch.Tensor,
            Ls: torch.Tensor,
        ):
            arr = -torch.einsum("i...,i->i...", xs, Ls)
            temp = torch.exp(arr)
            Q = torch.einsum("i...->i", temp)
            ans = torch.einsum("i,i...->i...", 1 / Q, temp)
            return ans

        xs = torch.ones((5, 1, 32, 32), requires_grad=False)
        Ls = torch.ones((5), requires_grad=False)
        expected = fn(xs, Ls)
        compiled_func = torch.compile(fn, backend="inductor")
        result = compiled_func(xs, Ls)
        torch.testing.assert_close(result, expected)

    def test_special_float_pow(self):
        def fn(exp: float) -> None:
            val = torch.randn(10)
            torch.testing.assert_close(
                aten.pow(val, exp), torch.compile(aten.pow)(val, exp), equal_nan=True
            )

        fn(-math.inf)
        fn(math.inf)
        fn(math.nan)

    def test_pdist_fallback_continuous(self):
        # https://github.com/pytorch/pytorch/issues/170939
        def fn(x):
            # Creating a non-contiguous tensor via permute
            x = x.permute(1, 0)
            return F.pdist(x)

        torch.compile(fn)(torch.randn(2, 2))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests
    from torch.testing._internal.inductor_utils import HAS_CPU

    if HAS_CPU and not IS_MACOS:
        run_tests(needs="filelock")
