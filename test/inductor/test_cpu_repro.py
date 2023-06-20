# Owner(s): ["module: inductor"]
import contextlib
import itertools
import sys
import unittest
from typing import Callable
from unittest.mock import patch

import numpy as np
import sympy
import torch
import torch._dynamo
from torch._C import FileCheck
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import same
from torch._inductor import codecache, config, metrics
from torch._inductor.codegen.cpp import (
    CppOverrides,
    CppVecKernelChecker,
    CppVecOverrides,
)
from torch._inductor.compile_fx import (
    compile_fx,
    compile_fx_inner,
    complex_memory_overlap,
)
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import InterpreterShim
from torch._inductor.utils import timed
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn import functional as F
from torch.testing._internal.common_utils import IS_MACOS, slowTest
from torch.utils._python_dispatch import TorchDispatchMode

try:
    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


vec_dtypes = test_torchinductor.vec_dtypes
run_and_get_cpp_code = test_torchinductor.run_and_get_cpp_code
TestCase = test_torchinductor.TestCase
aten = torch.ops.aten
check_model = test_torchinductor.check_model


class CPUReproTests(TestCase):
    common = check_model

    def test_conv_stride_constraints(self):
        for fmt in [torch.channels_last, torch.contiguous_format]:
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
                        test_self.assertTrue(args[0].is_contiguous(memory_format=fmt))
                        test_self.assertTrue(args[1].is_contiguous(memory_format=fmt))
                        nonlocal conv_seen
                        conv_seen = True

                    return func(*args, **kwargs)

            with RecordFunctions():
                out = fn_compiled(inps)

            self.assertTrue(conv_seen)

    @patch("torch.cuda.is_available", lambda: False)
    def test_conv2d_bn_mixed_dtype(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
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

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    def test_unsupported_conv_transpose(self):
        class Model(torch.nn.Module):
            def __init__(self):
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
            with self.assertRaisesRegex(
                RuntimeError,
                "output padding must be smaller than either stride or dilation",
            ):
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

        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            with torch.no_grad():
                m = M(224, 224).bfloat16().eval()
                m_opt = torch.compile(m)
                x = torch.randn(224, 224, dtype=torch.bfloat16)
                m_opt(x)
                self.assertEqual(m(x), m_opt(x))

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    def test_linear_packed(self):
        options = itertools.product(
            [[2, 3, 10], [2, 10], [10], [2, 0]], [3, 0], [True, False]
        )
        for input_shape, out_dim, bias in options:
            mod = torch.nn.Sequential(
                torch.nn.Linear(input_shape[-1], out_dim, bias=bias)
            ).eval()

            v = torch.randn(input_shape)
            with torch.no_grad():
                self.common(
                    mod,
                    (v,),
                )
            if torch.ops.mkldnn._is_mkldnn_bf16_supported() and len(input_shape) > 1:
                mod = mod.to(torch.bfloat16)
                v = v.to(torch.bfloat16)
                with torch.no_grad():
                    self.common(
                        mod,
                        (v,),
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
        # Hence, the accurarcy might vary up and down. For short term,
        # we increase the tolerance and will fix it later by using
        # aten parallel.
        self.common(fn, (v,), atol=5e-1, rtol=5e-1)

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

    def test_index_propagation_issue_102065(self):
        def fn(x):
            x = torch.arange(x.numel())
            return (x.unsqueeze(0) - x.unsqueeze(1)) ** 2

        self.common(
            fn,
            (torch.randn(8),),
        )

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
    def test_max_reduction_bfloat16(self):
        def fn(x):
            return torch.ops.aten.max(x, 1, keepdim=True)[0].float()

        self.common(
            fn,
            (torch.randn(1, 32, 4, 4).bfloat16(),),
        )

    @patch("torch.cuda.is_available", lambda: False)
    def test_vec_transpose_bf16(self):
        def fn(x):
            return x.to(memory_format=torch.channels_last).bfloat16()

        self.common(
            fn,
            (torch.randn(2, 3, 4, 4),),
        )

    @patch("torch.cuda.is_available", lambda: False)
    def test_fp32_load_with_to_bf16(self):
        # From llama model.
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cache_k = torch.zeros(8, 4, 2, 2)

            def forward(self, x, xk):
                bsz, seqlen, _ = x.shape
                self.cache_k = self.cache_k.to(x)
                self.cache_k[:bsz, 1 : 1 + seqlen] = xk
                return self.cache_k

        ref_model = Model().eval()
        opt_model = torch.compile()(Model().eval())
        x = torch.randn(4, 2, 2).bfloat16()
        xk = torch.randn(4, 2, 2, 2).bfloat16()
        self.assertEqual(opt_model(x, xk), ref_model(x, xk))

    @unittest.skipIf(
        not codecache.valid_vec_isa_list(), "Does not support vectorization"
    )
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
            assert (
                seq_len % (window_overlap * 2) == 0
            ), f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"

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
            diagonal_attention_scores[
                :, :3, :, window_overlap:
            ] = diagonal_chunked_attention_scores[
                :, :, :window_overlap, : window_overlap + 1
            ]
            return diagonal_attention_scores

        self.common(
            fn,
            (
                torch.randn(1, 1024, 12, 64),
                torch.randn(12, 3, 512, 513),
                256,
            ),
        )

    @unittest.skipIf(
        not codecache.valid_vec_isa_list(), "Does not support vectorization"
    )
    @patch("torch.cuda.is_available", lambda: False)
    def test_decomposed_dequant_relu_quant(self):
        def fn(x, scale, zero_point, use_dequant, use_quant):
            # For quantized_decomposed.dequantize_per_tensor
            # Refer to torch/ao/quantization/fx/_decomposed.py
            if use_dequant:
                x = (x.to(torch.float32) - zero_point) * scale

            x = torch.relu(x)

            # For quantized_decomposed.quantize_per_tensor
            # Refer to torch/ao/quantization/fx/_decomposed.py
            if use_quant:
                inv_scale = 1.0 / scale
                x = torch.clamp(torch.round(x * inv_scale) + zero_point, 0, 255).to(
                    torch.uint8
                )
            return x

        use_dequant_list = [False, True]
        use_quant_list = [False, True]
        for use_dequant, use_quant in itertools.product(
            use_dequant_list, use_quant_list
        ):
            x = torch.clamp(
                torch.randn((1, 7, 7, 9), dtype=torch.float32) * 100, 0, 255
            )
            if use_dequant:
                x = x.to(torch.uint8)
            zero_point = 100
            scale = 0.01
            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x, scale, zero_point, use_dequant, use_quant))
                assert metrics.generated_cpp_vec_kernel_count == 1

    @unittest.skipIf(
        not codecache.valid_vec_isa_list(), "Does not support vectorization"
    )
    @patch("torch.cuda.is_available", lambda: False)
    def test_dequant_quant_lowering(self):
        def fn(x, scale, zero_point, use_dequant, use_quant):
            if use_dequant:
                x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                    x, scale, zero_point, 0, 255, torch.uint8
                )

            x = torch.relu(x)

            if use_quant:
                x = torch.ops.quantized_decomposed.quantize_per_tensor(
                    x, scale, zero_point, 0, 255, torch.uint8
                )
            return x

        use_dequant_list = [False, True]
        use_quant_list = [False, True]
        use_tensor_overload_list = [False, True]
        for use_dequant, use_quant, use_tensor_overload in itertools.product(
            use_dequant_list, use_quant_list, use_tensor_overload_list
        ):
            x = torch.clamp(
                torch.randn((1, 7, 7, 9), dtype=torch.float32) * 100, 0, 255
            )
            if use_dequant:
                x = x.to(torch.uint8)
            zero_point = 100
            scale = 0.01
            if use_tensor_overload:
                zero_point = torch.tensor(zero_point, dtype=torch.int64)
                scale = torch.tensor(scale)
            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x, scale, zero_point, use_dequant, use_quant))
                assert metrics.generated_cpp_vec_kernel_count == 1

    @unittest.skipIf(
        not codecache.valid_vec_isa_list(), "Does not support vectorization"
    )
    @patch("torch.cuda.is_available", lambda: False)
    def test_dequant_maxpool2d_lowering(self):
        def fn(x, scale, zero_point):
            x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                x, scale, zero_point, 0, 255, torch.uint8
            )
            max_pool2d_with_indices_default = (
                torch.ops.aten.max_pool2d_with_indices.default(
                    x, [2, 2], [2, 2], [1, 1]
                )[0]
            )
            return max_pool2d_with_indices_default

        use_tensor_overload_list = [False, True]
        for use_tensor_overload in use_tensor_overload_list:
            x = (
                torch.clamp(
                    torch.randn((3, 16, 8, 8), dtype=torch.float32) * 100, 0, 255
                )
                .to(torch.uint8)
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
                self.common(fn, (x, scale, zero_point))
                assert metrics.generated_cpp_vec_kernel_count == 1

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
            a = torch.zeros((1 + s3) // 2)
            a += y
            return a, s3

        p0 = torch.randint(5, (1, 8))
        p1 = torch.randn(1)
        self.common(fn, (p0, p1))

    def test_no_op_squeeze(self):
        @torch._dynamo.optimize("inductor")
        def forward(arg0_1):
            return torch.ops.aten.squeeze.dim(arg0_1, 1)

        x = torch.randn((10, 20))
        self.common(forward, (x,))

    def test_parallel_num_threads(self):
        @torch._dynamo.optimize("inductor")
        def fn(x1, x2):
            return x1 + x2

        @contextlib.contextmanager
        def set_num_threads(num_threads):
            orig_num_threads = torch.get_num_threads()
            torch.set_num_threads(num_threads)
            yield
            torch.set_num_threads(orig_num_threads)

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

    @unittest.skipIf(
        not codecache.valid_vec_isa_list(), "Does not support vectorization"
    )
    def test_vec_dynamic_shapes(self):
        def fn(x):
            return torch.softmax(x, -1)

        value = torch.randn((2, 10))
        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, (value,))

    @unittest.skipIf(
        not codecache.valid_vec_isa_list(), "Does not support vectorization"
    )
    @patch("torch.cuda.is_available", lambda: False)
    def test_auto_simd(self):
        vec_avx512 = codecache.supported_vec_isa_list[0]
        vec_avx2 = codecache.supported_vec_isa_list[1]
        self.assertTrue(vec_avx512.bit_width() == 512)
        self.assertTrue(vec_avx2.bit_width() == 256)
        self.assertTrue(vec_avx512.nelements() == 16)
        self.assertTrue(vec_avx2.nelements() == 8)
        self.assertTrue(vec_avx512.nelements(torch.bfloat16) == 32)
        self.assertTrue(vec_avx2.nelements(torch.bfloat16) == 16)

        with config.patch({"cpp.simdlen": None}):
            isa = codecache.pick_vec_isa()
            if vec_avx512 in codecache.valid_vec_isa_list():
                self.assertTrue(isa == vec_avx512)
            else:
                self.assertTrue(isa == vec_avx2)

        with config.patch({"cpp.simdlen": 0}):
            isa = codecache.pick_vec_isa()
            self.assertFalse(isa)

        with config.patch({"cpp.simdlen": 1}):
            isa = codecache.pick_vec_isa()
            self.assertFalse(isa)

        with config.patch({"cpp.simdlen": 257}):
            isa = codecache.pick_vec_isa()
            self.assertFalse(isa)

        with config.patch({"cpp.simdlen": 513}):
            isa_list = codecache.valid_vec_isa_list()
            if vec_avx512 in isa_list:
                self.assertFalse(isa)

        with config.patch({"cpp.simdlen": 512}):
            isa_list = codecache.valid_vec_isa_list()
            if vec_avx512 in isa_list:
                isa = codecache.pick_vec_isa()
                self.assertTrue(isa == vec_avx512)

        with config.patch({"cpp.simdlen": 256}):
            isa_list = codecache.valid_vec_isa_list()
            if vec_avx2 in isa_list:
                isa = codecache.pick_vec_isa()
                self.assertTrue(isa == vec_avx2)

    @unittest.skipIf(
        not codecache.valid_vec_isa_list(), "Does not support vectorization"
    )
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

    def test_load_same_bool_tensor_twice(self):
        @torch._dynamo.optimize("inductor")
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
            "index_expr",
            "signbit",
            "isinf",
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

    def test_atomic_add_bf16(self):
        def fn(test_args):
            res = torch.gather(**test_args)
            return res

        input_tensor_for_ref = torch.tensor(
            [[3.0, -5.0]], dtype=torch.bfloat16, requires_grad=True
        )
        input_tensor_for_opt = torch.tensor(
            [[3.0, -5.0]], dtype=torch.bfloat16, requires_grad=True
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
        bwd_tensor_for_ref = torch.randn(ref_fwd.shape, dtype=torch.bfloat16)
        torch.manual_seed(1)
        bwd_tensor_for_opt = torch.randn(res_fwd.shape, dtype=torch.bfloat16)
        self.assertEqual(bwd_tensor_for_ref, bwd_tensor_for_opt)

        ref_fwd.backward(bwd_tensor_for_ref)
        res_fwd.backward(bwd_tensor_for_opt)

        ref_grad = test_args_for_ref["input"].grad
        res_grad = test_args_for_opt["input"].grad
        self.assertEqual(ref_grad, res_grad)

    @unittest.skipIf(
        not codecache.valid_vec_isa_list(), "Does not support vectorization"
    )
    @patch("torch.cuda.is_available", lambda: False)
    def test_new_vec_op_cpu_only(self):
        def fn(x):
            return torch.log1p(torch.expm1(torch.erf(x)))

        for dtype in vec_dtypes:
            torch.manual_seed(0)
            x = torch.randn((2, 9), dtype=dtype)
            x[0, 0] = torch.nan
            x[1, -1] = torch.nan

            tol = 1e-2 if dtype == torch.bfloat16 else 1e-4

            with config.patch({"cpp.simdlen": None}):
                for cpp_wrapper_flag in [True, False]:
                    with config.patch({"cpp_wrapper": cpp_wrapper_flag}):
                        torch._dynamo.reset()
                        metrics.reset()
                        self.common(fn, (x,))
                        assert metrics.generated_cpp_vec_kernel_count == 1

    @unittest.skipIf(
        not codecache.valid_vec_isa_list(), "Does not support vectorization"
    )
    @patch("torch.cuda.is_available", lambda: False)
    def test_vec_cpu_only_for_all_available_isa(self):
        def fn(x):
            return torch.sin(torch.cos(torch.erf(x)))

        x = torch.randn((2, 9))
        x[0, 0] = torch.nan
        x[1, -1] = torch.nan

        bit_widths = [isa._bit_width for isa in codecache.valid_vec_isa_list()] + [None]
        for item in bit_widths:
            with config.patch({"cpp.simdlen": item}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x,))
                assert metrics.generated_cpp_vec_kernel_count == 1

    @slowTest
    @unittest.skipIf(
        not codecache.valid_vec_isa_list(), "Does not support vectorization"
    )
    @patch("torch.cuda.is_available", lambda: False)
    def test__adaptive_avg_pool2d(self):
        def wrap_fn(oh, ow):
            def fn(x):
                return torch._adaptive_avg_pool2d(x, (oh, ow))

            return fn

        bit_widths = [isa._bit_width for isa in codecache.valid_vec_isa_list()]
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
                assert metrics.generated_cpp_vec_kernel_count == 1

    @unittest.skipIf(
        not codecache.valid_vec_isa_list(), "Does not support vectorization"
    )
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
                assert metrics.generated_cpp_vec_kernel_count == 1

    @unittest.skipIf(
        not codecache.valid_vec_isa_list(), "Does not support vectorization"
    )
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
                assert metrics.generated_cpp_vec_kernel_count == 1
                assert (
                    metrics.generated_kernel_count
                    - metrics.generated_cpp_vec_kernel_count
                ) == 0

    def test_skip_cpp_codegen(self):
        with config.patch({"disable_cpp_codegen": True}):
            inps = torch.ones([20]), torch.rand([20])

            def f(x, y):
                return x + y + torch.tensor(1)

            f_opt = torch.compile()(f)

            code = run_and_get_cpp_code(f_opt, inps[0], inps[1])
            FileCheck().check_not("void kernel").run(code)

            self.assertEqual(
                f(*inps),
                f_opt(*inps),
            )

            # constant needs to be propagated on fallback
            def f(x):
                return x[torch.tensor(1) :] * 2

            f_opt = torch.compile()(f)
            code = run_and_get_cpp_code(f_opt, inps[0])
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

    def test_redundant_to_node_elimination_bf16(self):
        def fn(x, y):
            res = x + y
            res = torch.mean(res)
            return res

        x = torch.randn((2, 9), dtype=torch.bfloat16)
        y = torch.randn((2, 9), dtype=torch.bfloat16)

        for torch_compile_debug in [True, False]:
            with config.patch(
                {"trace.enabled": torch_compile_debug, "cpp.simdlen": None}
            ):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x, y))
                if codecache.valid_vec_isa_list():
                    assert metrics.generated_cpp_vec_kernel_count == 1

    def test_do_not_insert_to_dtype_for_memory_copy_only_kernel(self):
        def fn(x):
            res = x.clone()
            return res

        x = torch.randn((100, 100), dtype=torch.bfloat16)

        torch._dynamo.reset()
        metrics.reset()
        self.common(fn, (x,))
        assert metrics.cpp_to_dtype_count == 0
        if codecache.valid_vec_isa_list():
            assert metrics.generated_cpp_vec_kernel_count == 1

    def test_insert_to_dtype_count(self):
        def fn(x):
            res = x.relu()
            return res

        x = torch.randn((100, 100), dtype=torch.bfloat16)

        torch._dynamo.reset()
        metrics.reset()
        self.common(fn, (x,))
        assert metrics.cpp_to_dtype_count == 2
        if codecache.valid_vec_isa_list():
            assert metrics.generated_cpp_vec_kernel_count == 1

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
        if codecache.valid_vec_isa_list():
            assert metrics.generated_cpp_vec_kernel_count == 1

    @unittest.skipIf(
        not codecache.valid_vec_isa_list(), "Does not support vectorization"
    )
    @patch("torch.cuda.is_available", lambda: False)
    def test_cpp_vec_constant_checker(self):
        _graph: torch.fx.Graph = torch.fx.Graph()
        a: torch.fx.Node = _graph.create_node("placeholder", "ops")
        iv: torch.fx.Node = _graph.create_node("placeholder", "iv")
        fv: torch.fx.Node = _graph.create_node("placeholder", "fv")
        b: torch.fx.Node = _graph.create_node(
            "call_method",
            "constant",
            args=(
                a,
                iv,
                torch.int64,
            ),
        )
        c: torch.fx.Node = _graph.create_node(
            "call_method",
            "constant",
            args=(
                a,
                fv,
                torch.double,
            ),
        )
        d: torch.fx.Node = _graph.create_node(
            "call_method",
            "ge",
            args=(
                a,
                b,
                b,
            ),
        )
        _graph.output((d, c))

        def get_index():
            return ""

        submodules = {"get_index": get_index}

        graph_lowering = GraphLowering(
            torch.fx.GraphModule(submodules, _graph),
            shape_env=None,
            num_static_inputs=0,
        )
        with patch.object(graph_lowering, "wrapper_code", ""), V.set_graph_handler(
            graph_lowering
        ):
            # The moset inner loop variable is used in the index_expr
            tiling_factor = codecache.pick_vec_isa().nelements(dtype=torch.float)
            with CppVecKernelChecker(
                args=None, num_threads=1, tiling_factor=tiling_factor
            ) as vec_checker:
                i32_iinfo = np.iinfo(np.int32)
                f32_iinfo = np.finfo(np.float32)
                InterpreterShim(_graph, submodules).run(
                    V.get_ops_handler(), i32_iinfo.max, f32_iinfo.max
                )
                self.assertTrue(vec_checker.simd_vec)

                vec_checker.simd_vec = True
                InterpreterShim(_graph, submodules).run(
                    V.get_ops_handler(), i32_iinfo.min, f32_iinfo.min
                )
                self.assertTrue(vec_checker.simd_vec)

                vec_checker.simd_vec = True
                InterpreterShim(_graph, submodules).run(
                    V.get_ops_handler(), i32_iinfo.min, np.inf
                )
                self.assertTrue(vec_checker.simd_vec)

                vec_checker.simd_vec = True
                InterpreterShim(_graph, submodules).run(
                    V.get_ops_handler(), i32_iinfo.min, -np.inf
                )
                self.assertTrue(vec_checker.simd_vec)

                vec_checker.simd_vec = True
                InterpreterShim(_graph, submodules).run(
                    V.get_ops_handler(), i32_iinfo.min - 1, f32_iinfo.min
                )
                self.assertFalse(vec_checker.simd_vec)

                vec_checker.simd_vec = True
                InterpreterShim(_graph, submodules).run(
                    V.get_ops_handler(), i32_iinfo.max + 1, f32_iinfo.max
                )
                self.assertFalse(vec_checker.simd_vec)

                vec_checker.simd_vec = True
                InterpreterShim(_graph, submodules).run(
                    V.get_ops_handler(), i32_iinfo.min, f32_iinfo.min * (1 + 1e-5)
                )
                self.assertFalse(vec_checker.simd_vec)

                vec_checker.simd_vec = True
                InterpreterShim(_graph, submodules).run(
                    V.get_ops_handler(), i32_iinfo.max, f32_iinfo.max * (1 + 1e-5)
                )
                self.assertFalse(vec_checker.simd_vec)

    @unittest.skipIf(
        not codecache.valid_vec_isa_list(), "Does not support vectorization"
    )
    @patch("torch.cuda.is_available", lambda: False)
    def test_cpp_vec_index_expr_checker(self):
        _graph: torch.fx.Graph = torch.fx.Graph()
        a: torch.fx.Node = _graph.create_node("placeholder", "ops")
        b: torch.fx.Node = _graph.create_node("call_module", "get_index", args=())
        c: torch.fx.Node = _graph.create_node(
            "call_method",
            "index_expr",
            args=(
                a,
                b,
                torch.int64,
            ),
        )
        d: torch.fx.Node = _graph.create_node(
            "call_method",
            "ge",
            args=(
                a,
                c,
                c,
            ),
        )
        _graph.output(d)

        def get_index():
            return ""

        submodules = {"get_index": get_index}
        graph_lowering = GraphLowering(
            torch.fx.GraphModule(submodules, _graph),
            shape_env=None,
            num_static_inputs=0,
        )
        with patch.object(graph_lowering, "wrapper_code", ""), V.set_graph_handler(
            graph_lowering
        ):
            itervars = [sympy.Symbol("i"), sympy.Symbol("j"), sympy.Symbol("k")]

            tiling_factor = codecache.pick_vec_isa().nelements(dtype=torch.float)
            # The moset inner loop variable is used in the index_expr
            with CppVecKernelChecker(
                args=None, num_threads=1, tiling_factor=tiling_factor
            ) as vec_checker:

                def get_index():
                    return -itervars[0] ** 2 + 2 * itervars[0] + itervars[1]

                ranges = [0, 100, 200]
                vec_checker.itervars = itervars[:2]
                vec_checker.ranges = ranges[:2]
                submodules = {"get_index": get_index}
                InterpreterShim(_graph, submodules).run(V.get_ops_handler())
                self.assertFalse(vec_checker.simd_vec)

            # Most inner loop variable irrevalant
            with CppVecKernelChecker(
                args=None, num_threads=1, tiling_factor=tiling_factor
            ) as vec_checker:

                def get_index():
                    return -itervars[0] ** 2 + 2 * itervars[0] + itervars[1]

                ranges = [0, 100, 200]
                vec_checker.itervars = itervars
                vec_checker.ranges = ranges
                submodules = {"get_index": get_index}
                InterpreterShim(_graph, submodules).run(V.get_ops_handler())
                self.assertTrue(vec_checker.simd_vec)

            i32_iinfo = np.iinfo(np.int32)
            _max_value = i32_iinfo.max + 1
            ranges = [_max_value, _max_value, _max_value]
            # Most inner loop variable irrevalant but max value is greater than
            # the max value of INT32
            with CppVecKernelChecker(
                args=None, num_threads=1, tiling_factor=tiling_factor
            ) as vec_checker:

                def get_index():
                    return itervars[0]

                submodules = {"get_index": get_index}
                vec_checker.itervars = itervars
                vec_checker.ranges = ranges
                InterpreterShim(_graph, submodules).run(V.get_ops_handler())
                self.assertFalse(vec_checker.simd_vec)

            # Most inner loop variable irrevalant but min value is greater than
            # the min value of INT32
            with CppVecKernelChecker(
                args=None, num_threads=1, tiling_factor=tiling_factor
            ) as vec_checker:

                def get_index():
                    return -itervars[0] - 2

                submodules = {"get_index": get_index}
                vec_checker.itervars = itervars
                vec_checker.ranges = ranges
                InterpreterShim(_graph, submodules).run(V.get_ops_handler())
                self.assertFalse(vec_checker.simd_vec)

    @unittest.skipIf(
        not codecache.valid_vec_isa_list(), "Does not support vectorization"
    )
    @patch("torch.cuda.is_available", lambda: False)
    def test_maxpool2d_cpu_only(self):
        for dtype in vec_dtypes:
            input = torch.randn(10, 32, 20, 20, dtype=dtype).to(
                memory_format=torch.channels_last
            )
            maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            def func(x):
                return maxpool(x)

            with patch.object(config.cpp, "simdlen", None):
                torch._dynamo.reset()
                metrics.reset()
                self.common(func, (input,))
                assert metrics.generated_cpp_vec_kernel_count == 1

    @unittest.skipIf(
        not codecache.valid_vec_isa_list(), "Does not support vectorization"
    )
    @patch("torch.cuda.is_available", lambda: False)
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
            assert metrics.generated_cpp_vec_kernel_count == 2

    @unittest.skipIf(
        not codecache.valid_vec_isa_list(), "Does not support vectorization"
    )
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
                assert metrics.generated_cpp_vec_kernel_count == 1

    @unittest.skipIf(
        not codecache.valid_vec_isa_list(), "Does not support vectorization"
    )
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
                assert metrics.generated_cpp_vec_kernel_count == 0

    # Currently, we enabled AVX2 and AVX512 for vectorization. If the platform is not
    # supported, the vectorization will not work and skip this test case. For ARM or
    # other platforms support, we just need to add the ISA info to the supported_vector_isa
    # and include proper aten vectorization head file.
    @unittest.skipIf(
        not codecache.valid_vec_isa_list(), "Does not support vectorization"
    )
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

            tol = 1e-2 if dtype == torch.bfloat16 else 1e-4
            with config.patch({"cpp.simdlen": 1}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x1, x2))
                assert metrics.generated_cpp_vec_kernel_count == 0

            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                self.common(fn, (x1, x2))
                assert metrics.generated_cpp_vec_kernel_count == 1

        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            x1 = torch.randn(10, 20).permute(1, 0)
            x2 = torch.randn((20, 10))
            self.common(fn, (x1, x2))
            assert metrics.generated_cpp_vec_kernel_count == 2

            torch._dynamo.reset()
            metrics.reset()
            x1 = torch.randn((10, 7))
            x2 = torch.randn((10, 7))
            self.common(fn, (x1, x2))
            assert metrics.generated_cpp_vec_kernel_count == 1

    @unittest.skipIf(
        sys.platform != "linux", "cpp kernel profile only support linux now"
    )
    @patch("torch.cuda.is_available", lambda: False)
    @config.patch({"cpp.enable_kernel_profile": True})
    @config.patch({"cpp.descriptive_names": "original_aten"})
    def test_cpp_kernel_profile(self):
        from torch.profiler import profile

        @torch._dynamo.optimize("inductor", nopython=True)
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

    @unittest.skipIf(
        not codecache.valid_vec_isa_list(), "Does not support vectorization"
    )
    def test_channel_shuffle_cl_output(self):
        """code and shape extracted from shufflenet_v2_x1_0"""

        def channel_shuffle(x, groups):
            batchsize, num_channels, height, width = x.size()
            channels_per_group = num_channels // groups
            x = x.view(batchsize, groups, channels_per_group, height, width)
            x = torch.transpose(x, 1, 2).contiguous()
            x = x.view(batchsize, -1, height, width)
            return x.contiguous(memory_format=torch.channels_last)

        for simdlen in (None, 256, 1):
            with config.patch({"cpp.simdlen": simdlen}):
                torch._dynamo.reset()
                metrics.reset()
                x = torch.randn(64, 58, 28, 28)
                self.common(channel_shuffle, (x, 2))
                if simdlen != 1:
                    assert metrics.generated_cpp_vec_kernel_count == 2

    @slowTest
    @unittest.skipIf(
        not codecache.valid_vec_isa_list(), "Does not support vectorization"
    )
    def test_transpose_with_norm(self):
        """a sub-module from TIMM gmlp_s16_224"""

        class Model(torch.nn.Module):
            def __init__(self):
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
        for simdlen in (None, 256, 1):
            with config.patch({"cpp.simdlen": simdlen}):
                for eval_mode in [True, False]:
                    torch._dynamo.reset()
                    metrics.reset()
                    m = Model().eval() if eval_mode else Model()
                    self.common(m, (x,))
                    if simdlen != 1:
                        assert metrics.generated_cpp_vec_kernel_count == 6

    @unittest.skipIf(
        not codecache.valid_vec_isa_list(), "Does not support vectorization"
    )
    def test_transpose_copy(self):
        def fn(a):
            return a.t().contiguous()

        for simdlen in (None, 256, 1):
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
                        if simdlen != 1:
                            assert metrics.generated_cpp_vec_kernel_count == 2

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
            opt_fn = torch._dynamo.optimize("inductor")(fn)
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
            opt_fn = torch._dynamo.optimize("inductor")(fn)
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
            opt_fn = torch._dynamo.optimize("inductor")(fn)
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
            opt_fn = torch._dynamo.optimize("inductor")(fn)
            opt_fn(a, b, c, idx)
            self.assertEqual(metrics.generated_kernel_count, 1)
            self.assertTrue(same(fn(a, b, c, idx), opt_fn(a, b, c, idx)))

    def test_bf16_neg_abs(self):
        def fn(x):
            return x.neg().abs()

        metrics.reset()
        x = torch.randn(100, 100).bfloat16()
        opt_fn = torch._dynamo.optimize("inductor")(fn)
        self.assertTrue(same(fn(x), opt_fn(x)))
        assert metrics.cpp_to_dtype_count == 0
        assert metrics.generated_cpp_vec_kernel_count == 1

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
            getitem_1 = split_with_sizes[1]
            permute_3 = torch.ops.aten.permute.default(getitem, [0, 1, 3, 2])
            expand_1 = torch.ops.aten.expand.default(permute_3, [8, 4, 16, 144])
            clone_3 = torch.ops.aten.clone.default(
                expand_1, memory_format=torch.contiguous_format
            )
            return clone_3

        metrics.reset()
        x = torch.randn(1, 384, 20, 20).to(memory_format=torch.channels_last)
        self.common(fn, (x,))
        assert metrics.generated_cpp_vec_kernel_count == 1

    def test_non_contiguous_index_with_constant_stride(self):
        def fn(x):
            x1 = x[:, :, :, ::2]
            x2 = x[:, :, :, 1::2]
            x = torch.stack((-x2, x1), dim=-1)
            return x.flatten(-2)

        metrics.reset()
        x = torch.randn(1, 32, 16, 68)
        opt_fn = torch._dynamo.optimize("inductor")(fn)
        self.assertTrue(same(fn(x), opt_fn(x)))
        assert metrics.generated_cpp_vec_kernel_count == 2

    def test_invalid_index_of_empty_tensor(self):
        def fn(a):
            b = a[[0]]
            return b

        a = torch.tensor([])
        with self.assertRaises(RuntimeError):
            torch.compile(fn)(a)

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
        assert metrics.generated_cpp_vec_kernel_count == 1

        metrics.reset()
        x = torch.randn(100, 100, 100)
        self.common(fn2, (x,))
        assert metrics.generated_cpp_vec_kernel_count == 1

    def test_transpose_vertical_sum_cpu_only(self):
        def fn(a, b):
            c = a * b
            return c.sum(dim=1)

        metrics.reset()
        x = torch.randn(100, 50, 50)
        y = torch.randn(100, 50, 50).transpose(1, 2)
        self.common(fn, (x, y))
        assert metrics.generated_cpp_vec_kernel_count == 2

    def test_transpose_sum2d_cpu_only(self):
        def fn(a, b):
            c = a * b
            return c.sum()

        metrics.reset()
        x = torch.randn(50, 50)
        y = torch.randn(50, 50).transpose(0, 1)
        self.common(fn, (x, y))
        assert metrics.generated_cpp_vec_kernel_count == 2

    def test_transpose_sum_outer(self):
        # https://github.com/pytorch/pytorch/issues/98573
        def fn(a):
            return a.transpose(2, 3).sum(dim=1).contiguous()

        metrics.reset()
        x = torch.randn(10, 50, 50, 50)
        self.common(fn, (x,))
        assert metrics.generated_cpp_vec_kernel_count == 1

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

    def test_to_channels_last_bfloat16(self):
        def f(a):
            return a.to(memory_format=torch.channels_last)

        x = torch.rand(2, 3, 14, 14).bfloat16()
        self.common(f, (x,))

    def test_linear_buffer_reuse(self):
        class M(torch.nn.Module):
            def __init__(self):
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

            run = torch._dynamo.optimize(compile_fx_wrapper)(run)
            code = run_and_get_cpp_code(run, v)
            self.assertFalse("= as_strided(" in code)
            self.assertEqual(run(*v), mod(*v))

    @config.patch(inplace_buffers=True)
    def test_in_out_buffer(self):
        def fn(x, y):
            z = torch.matmul(x, y.transpose(-1, -2)) / 8.0
            return z

        inps = [torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4)]
        fn_opt = torch._dynamo.optimize("inductor")(fn)
        code = run_and_get_cpp_code(fn_opt, *inps)
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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests
    from torch.testing._internal.inductor_utils import HAS_CPU

    if HAS_CPU and not IS_MACOS:
        run_tests(needs="filelock")
