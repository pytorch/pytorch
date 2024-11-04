# Owner(s): ["module: inductor"]
import copy
import functools
import os
import unittest
from typing import Tuple

import torch
from torch import nn, Tensor
from torch._dynamo.convert_frame import maybe_cprofile
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import rand_strided, reduce_to_scalar_loss
from torch._inductor import config, ir, metrics
from torch._inductor.fx_passes import pad_mm as pad_mm_pass
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.utils import ceildiv, run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda,
    serialTest,
)
from torch.testing._internal.inductor_utils import HAS_CUDA


DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"
DO_ACC_TEST = os.environ.get("DO_ACC_TEST", "1") == "1"
WITH_STACK = os.environ.get("WITH_STACK") == "1"
USE_CUDA_GRAPHS = os.environ.get("USE_CUDA_GRAPHS", "1") == "1"

try:
    import transformers  # noqa: F401

    HAS_TRANSFORMER = True
except ImportError:
    HAS_TRANSFORMER = False


def get_optim(m):
    return torch.optim.Adam(m.parameters(), lr=0.01, capturable=True, foreach=True)


def gen_transformer_inputs(vocab_size, bs, seq_length):
    def geninp():
        return torch.randint(
            0, vocab_size, (bs, seq_length), dtype=torch.int64, requires_grad=False
        )

    input_dict = {"input_ids": geninp(), "labels": geninp()}
    return input_dict


class LinearAndSoftmax(nn.Module):
    """
    It's very common that a transformer model will do a matmul and then
    softmax/log_softmax in the end.

    Creating this toy model to capture the pattern and make sure we do
    proper padding.
    """

    def __init__(self, vocab_size=30523, bias=True):
        """
        The default vocab size for BertForMaskedLM is 30522.
        We run a few test cases with good or bad vocab_size around Bert's
        default value.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.linear = nn.Linear(768, vocab_size, bias=bias)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, label):
        x = self.linear(x)
        return self.ce(x.view(-1, self.vocab_size), label.view(-1))

    def get_example_inputs(self, batch_size=16):
        return torch.randn(batch_size, 512, 768), torch.randint(
            0, self.vocab_size, (batch_size, 512)
        )


def forward_and_backward_pass(m, inputs):
    m(*inputs).sum().backward()


@config.patch(
    {
        "benchmark_kernel": True,
        "triton.unique_kernel_names": True,
        "triton.cudagraphs": USE_CUDA_GRAPHS,
    }
)
@requires_cuda
class TestCaseBase(TestCase):
    @classmethod
    def setUpClass(cls):
        if HAS_CUDA:
            cls.prior_float32_matmul_precision = torch.get_float32_matmul_precision()
            cls.prior_default_device = torch.get_default_device()
            torch.set_float32_matmul_precision("high")
            torch.set_default_device("cuda")

    @classmethod
    def tearDownClass(cls):
        if HAS_CUDA:
            torch.set_float32_matmul_precision(cls.prior_float32_matmul_precision)
            torch.set_default_device(cls.prior_default_device)

            cls.prior_float32_matmul_precision = None
            cls.prior_default_device = None

    def check_close(self, ref, act, tol=1e-3):
        if type(ref).__name__ == "LongformerMaskedLMOutput":
            ref = ref.loss
            act = act.loss
        if type(ref).__name__ == "SequenceClassifierOutput":
            ref = ref.logits
            act = act.logits
        if isinstance(ref, dict) and "loss" in ref:
            ref = ref["loss"]
            act = act["loss"]
        self.assertTrue(
            torch.allclose(ref, act, atol=tol, rtol=tol), f"ref:\n{ref}\nact:\n{act}"
        )

    def common_numeric_check(self, f, *args, tol=1e-3, **kwargs):
        ref = f(*args, **kwargs)
        opt_f = torch.compile(f)
        act = opt_f(*args, **kwargs)
        self.check_close(ref, act, tol)

    def do_profiling(
        self,
        f_lhs,
        f_rhs,
        tag_lhs="With padding",
        tag_rhs="Without padding",
        args=(),
        kwargs=None,
    ):
        if kwargs is None:
            kwargs = {}
        torch.cuda.synchronize()
        with torch.profiler.profile(with_stack=WITH_STACK) as p:
            niter = 3
            for _ in range(niter):
                with torch.profiler.record_function(tag_lhs):
                    f_lhs(*args, **kwargs)

                with torch.profiler.record_function(tag_rhs):
                    f_rhs(*args, **kwargs)
            torch.cuda.synchronize()

        profile_path = "/tmp/chrome.json"
        p.export_chrome_trace(profile_path)
        print(f"Chrome trace is written to {profile_path}")


class PerfTestBetweenGoodAndBadShape(TestCaseBase):
    @unittest.skipIf(not DO_PERF_TEST, "Perf test not enabled")
    def test_nobias_LinearAndSoftmax_both_shapes(self):
        self.test_LinearAndSoftmax_both_shapes(bias=False)

    @unittest.skipIf(not DO_PERF_TEST, "Perf test not enabled")
    def test_LinearAndSoftmax_both_shapes(self, bias=True):
        """
        Compare the perf with good and bad shape.
        """
        m_bad_shape = LinearAndSoftmax(vocab_size=30523, bias=bias)
        inptus_bad_shape = m_bad_shape.get_example_inputs()
        m_good_shape = LinearAndSoftmax(vocab_size=30528, bias=bias)
        inputs_good_shape = m_good_shape.get_example_inputs()

        m_bad_shape_opt = torch.compile(m_bad_shape)
        m_good_shape_opt = torch.compile(m_good_shape)

        latency_good_shape = benchmarker.benchmark_gpu(
            lambda: forward_and_backward_pass(m_good_shape_opt, inputs_good_shape)
        )
        latency_bad_shape = benchmarker.benchmark_gpu(
            lambda: forward_and_backward_pass(m_bad_shape_opt, inptus_bad_shape)
        )
        print(
            f"Latency for good shape v.s. bad shape: {latency_good_shape:.3f}ms v.s. {latency_bad_shape:.3f}ms"
        )

    @unittest.skipIf(not DO_PERF_TEST or not HAS_TRANSFORMER, "Perf test not enabled")
    def test_BertForMaskedLM(self, num_layers=1):
        """
        Compare the perf between doing padding and good shape.
        """
        from transformers import BertForMaskedLM

        config_cls = BertForMaskedLM.config_class
        bs = 16
        seq_length = 512

        def create_model(vocab_size):
            config = config_cls()
            config.num_hidden_layers = num_layers
            config.vocab_size = vocab_size
            inputs = gen_transformer_inputs(config.vocab_size, bs, seq_length)
            model = BertForMaskedLM(config)

            optim = get_optim(model)

            def f(**inputs):
                optim.zero_grad(True)
                with torch.cuda.amp.autocast():
                    pred = model(**inputs)
                    loss = pred[0]
                loss.backward()
                optim.step()

            return torch.compile(f), inputs

        f_good_shape, inputs_good_shape = create_model(30528)
        f_bad_shape, inputs_bad_shape = create_model(30522)

        print("benchmark for good shape")
        latency_good_shape = benchmarker.benchmark_gpu(
            lambda: f_good_shape(**inputs_good_shape)
        )
        print("benchmark for bad shape")
        latency_bad_shape = benchmarker.benchmark_gpu(
            lambda: f_bad_shape(**inputs_bad_shape)
        )
        print(
            f"Latency with good and bad shape: {latency_good_shape:.3f} v.s. {latency_bad_shape:.3f}"
        )

        self.do_profiling(
            lambda: f_good_shape(**inputs_good_shape),
            lambda: f_bad_shape(**inputs_bad_shape),
            tag_lhs="With good shape",
            tag_rhs="With bad shape",
        )


class PerfTestWithAndWithoutPadding(TestCaseBase):
    @maybe_cprofile
    def run_acc_and_perf_test(self, model, inputs, perf_inputs=None, tol=1e-3):
        """
        Run accuracy test.

        Also compare the perf with and without the comprehensive padding if
        DO_PERF_TEST is true.
        """
        if perf_inputs is None:
            perf_inputs = inputs

        def _process_inputs(x):
            """
            return args and kwargs
            """
            if isinstance(x, dict):
                return [], x

            if not isinstance(inputs, (tuple, list)):
                x = [x]

            return x, {}

        args, kwargs = _process_inputs(inputs)
        perf_args, perf_kwargs = _process_inputs(perf_inputs)

        if DO_ACC_TEST:
            model.eval()
            self.common_numeric_check(model, *args, **kwargs, tol=tol)
        else:
            print("Accuracy test skipped")

        model.train()

        if DO_PERF_TEST:
            print("Do performance test")

            def get_f(m, optim):
                def f(*args, **kwargs):
                    optim.zero_grad(True)
                    with torch.cuda.amp.autocast():
                        pred = m(*args, **kwargs)
                        loss = reduce_to_scalar_loss(pred)
                    loss.backward()
                    optim.step()

                return f

            latency_with_padding = None
            print("Benchmark with padding")
            with config.patch(comprehensive_padding=True):
                m_copy_with_padding = copy.deepcopy(model)
                optim_with_padding = get_optim(m_copy_with_padding)
                opt_f_with_padding = torch.compile(
                    get_f(m_copy_with_padding, optim_with_padding)
                )
                latency_with_padding = benchmarker.benchmark_gpu(
                    lambda: opt_f_with_padding(*perf_args, **perf_kwargs)
                )
            latency_without_padding = None
            print("bencmark without padding")
            with config.patch(comprehensive_padding=False):
                m_copy_without_padding = copy.deepcopy(model)
                optim_without_padding = get_optim(m_copy_without_padding)
                opt_f_without_padding = torch.compile(
                    get_f(m_copy_without_padding, optim_without_padding)
                )
                latency_without_padding = benchmarker.benchmark_gpu(
                    lambda: opt_f_without_padding(*perf_args, **perf_kwargs)
                )
            print(
                f"Latency with and without padding: {latency_with_padding:.3f} v.s. {latency_without_padding:.3f}"
            )

            # profiling
            self.do_profiling(
                opt_f_with_padding,
                opt_f_without_padding,
                args=perf_args,
                kwargs=perf_kwargs,
            )

    def test_nvidia_deeprecommender(self):
        """
        Compared the perf with and without comprehensive padding.
        """
        layer_sizes = [197951, 512, 512, 1024, 512, 512, 197951]
        x = torch.randn(4, layer_sizes[0])

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                mod_list = []
                for i in range(len(layer_sizes) - 1):
                    mod_list.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                    mod_list.append(nn.SELU())

                    if i == 2:
                        mod_list.append(nn.Dropout(0.8))
                self.seq = nn.Sequential(*mod_list)

            def forward(self, x):
                return self.seq(x)

        m = Model()
        perf_inputs = torch.randn(256, layer_sizes[0])
        self.run_acc_and_perf_test(m, x, perf_inputs)

    @unittest.skipIf(not DO_PERF_TEST or not HAS_TRANSFORMER, "Perf test not enabled")
    def test_longformer(self, bs=4):
        from transformers import AutoConfig, AutoModelForMaskedLM

        config = AutoConfig.from_pretrained("allenai/longformer-base-4096")
        model = AutoModelForMaskedLM.from_config(config)

        vocab_size = model.config.vocab_size
        seq_length = 1024
        input_dict = gen_transformer_inputs(vocab_size, bs, seq_length)

        self.run_acc_and_perf_test(model, input_dict)

    @unittest.skipIf(not DO_PERF_TEST or not HAS_TRANSFORMER, "Perf test not enabled")
    def test_longformer_small_bs(self):
        """
        The model exists in both HF and TB. In TB it uses a samller batch size.
        """
        self.test_longformer(bs=2)


@instantiate_parametrized_tests
class PaddingTest(TestCaseBase):
    @unittest.skipIf(not DO_PERF_TEST, "Perf test not enabled")
    def test_mm_padding_perf(self):
        def naive_mm(a, b):
            return a @ b

        def _compute_padding(s, align):
            return (s + align - 1) // align * align - s

        @torch.compile
        def pad_mm(a, b, align=16):
            """
            NOTE: this function only pad a single dimension which is good
            enough for testing.
            """
            m_padding = _compute_padding(a.size(0), align)
            k_padding = _compute_padding(a.size(1), align)
            n_padding = _compute_padding(b.size(1), align)
            return pad_mm_pass.pad_mm(a, b, m_padding, k_padding, n_padding)

        for M, K, N, f in (
            (8192, 768, 30523, naive_mm),
            (8192, 768, 30523, pad_mm),
            (8192, 768, 30528, naive_mm),
            (30523, 8192, 768, naive_mm),
            (30528, 8192, 768, naive_mm),
        ):
            a = torch.randn(M, K)
            b = torch.randn(K, N)
            ms = benchmarker.benchmark_gpu(lambda: f(a, b))
            print(f"MxKxN {M}x{K}x{N} {f.__name__}: {ms:.3f}ms")

    @unittest.skipIf(not DO_PERF_TEST, "Perf test not enabled")
    def test_padmm(self):
        """
        Latency between origional matmul and padded matmul: 2.717 v.s. 2.356
        """
        mat1_pad = torch.randn(8192, 30522, dtype=torch.float16)
        mat2_pad = torch.randn(30522, 768, dtype=torch.float16)

        def f():
            return mat1_pad @ mat2_pad

        def pad_dim(x: Tensor, padded_length: int, dim: int) -> Tensor:
            pad = x.new_zeros(*x.shape[:dim], padded_length, *x.shape[dim + 1 :])
            return torch.cat([x, pad], dim=dim)

        @torch.compile(fullgraph=True, options={"triton.cudagraphs": False})
        def g():
            mat1 = mat1_pad
            mat2 = mat2_pad
            mat1 = pad_dim(mat1, 6, 1)
            mat2 = pad_dim(mat2, 6, 0)
            return torch.ops.aten.mm(mat1, mat2)

        ori_time = benchmarker.benchmark_gpu(f)
        pad_time = benchmarker.benchmark_gpu(g)

        print(
            f"Latency between origional matmul and padded matmul: {ori_time:.3f} v.s. {pad_time:.3f}"
        )
        self.do_profiling(f, g, "No MM Padding", "With mm padding")

    @unittest.skipIf(not DO_PERF_TEST, "Perf test not enabled")
    def test_matmul(self):
        """
        Latency with good and bad shapes: 1.705 v.s. 2.625
        """
        x_good_shape = torch.randn(8192, 30528, dtype=torch.float16)
        weight_good_shape = torch.randn(30528, 768, dtype=torch.float16)
        out_good_shape = torch.randn(8192, 768, dtype=torch.float16)

        # Using stride (30522, 1) does not make a difference here.
        x_bad_shape = rand_strided(
            (8192, 30522), (30528, 1), device="cuda", dtype=torch.float16
        )
        weight_bad_shape = torch.randn(30522, 768, dtype=torch.float16)
        out_bad_shape = torch.randn(8192, 768, dtype=torch.float16)

        def f(x, weight, out):
            torch.mm(x, weight, out=out)
            return out

        f1 = torch.compile(
            functools.partial(f, x_good_shape, weight_good_shape, out_good_shape)
        )
        f2 = torch.compile(
            functools.partial(f, x_bad_shape, weight_bad_shape, out_bad_shape)
        )
        latency_good_shape = benchmarker.benchmark_gpu(f1)
        latency_bad_shape = benchmarker.benchmark_gpu(f2)
        print(
            f"Latency with good and bad shapes: {latency_good_shape:.3f} v.s. {latency_bad_shape:.3f}"
        )
        self.do_profiling(f1, f2)

    @serialTest()
    def test_nobias_LinearAndSoftmax_codegen(self):
        self.test_LinearAndSoftmax_codegen(bias=False)

    def test_LinearAndSoftmax_codegen(self, bias=True):
        m_bad_shape = LinearAndSoftmax(vocab_size=30523, bias=bias)
        inputs_bad_shape = m_bad_shape.get_example_inputs()
        m_bad_shape_opt = torch.compile(copy.deepcopy(m_bad_shape))

        _, wrapper_codes = run_and_get_code(
            forward_and_backward_pass, m_bad_shape_opt, inputs_bad_shape
        )
        forward_and_backward_pass(m_bad_shape, inputs_bad_shape)
        self.assertEqual(
            m_bad_shape.linear.weight.grad, m_bad_shape_opt.linear.weight.grad
        )
        self.assertTrue(len(wrapper_codes) == 2)  # one for forward and oen for backward
        forward_wrapper = wrapper_codes[0]

        # make sure the load for softmax is aligned
        self.assertTrue(
            "tl.load(in_ptr0 + (r1 + (30528*x0))" in forward_wrapper,
            f"forward_wrapper: {forward_wrapper}",
        )

        if DO_PERF_TEST:
            latency = benchmarker.benchmark_gpu(
                lambda: forward_and_backward_pass(m_bad_shape_opt, inputs_bad_shape)
            )
            print(f"latency: {latency:.3f}ms")

    @config.patch(pattern_matcher=False)
    def test_attention(self):
        batch_size, seq_len, num_heads, hidden_size = 1, 4, 1, 16
        inv_scale = (num_heads / hidden_size) ** 0.5

        class Attention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.query = nn.Linear(hidden_size, hidden_size)
                self.key = nn.Linear(hidden_size, hidden_size)
                self.value = nn.Linear(hidden_size, hidden_size)

            @staticmethod
            def reshape(x):
                return x.view(batch_size, seq_len, num_heads, -1).permute(0, 2, 1, 3)

            @staticmethod
            def cancel_reshape(x):
                return x.permute(0, 2, 1, 3).view(batch_size, seq_len, hidden_size)

            def forward(self, x):
                query, key, value = self.query(x), self.key(x), self.value(x)
                weights = (
                    torch.matmul(
                        self.reshape(query), self.reshape(key).permute(0, 1, 3, 2)
                    )
                    * inv_scale
                ).softmax(dim=-1)
                return self.cancel_reshape(torch.matmul(weights, self.reshape(value)))

        attn = Attention()
        x = torch.randn(batch_size, seq_len, hidden_size)

        self.common_numeric_check(attn, x)

    def test_view(self):
        def f(x):
            return x.view(3, 3, 3)

        x = torch.randn(3, 9)
        self.common_numeric_check(f, x)

    def test_pad_strides(self):
        """
        Note that dim0's stride is also padded even though its previous value
        is already multiple of 16. The reason is we padded dim1's stride.
        We have to correspondingly increase the stride for dim0.
        """
        sizes = [2, 16, 2047]
        in_strides = [2047 * 16, 2047, 1]
        out_strides = list(ir.Layout._pad_strides(in_strides, sizes, torch.float32))
        expected_strides = [2048 * 16, 2048, 1]
        self.assertEqual(
            expected_strides, out_strides, f"{expected_strides} v.s. {out_strides}"
        )

    def test_pad_strides_skip(self):
        """
        The padding is skipped to avoid too much memory overhead.
        """
        sizes = [2, 32, 127]
        in_strides = [4064, 127, 1]
        out_strides = list(ir.Layout._pad_strides(in_strides, sizes, torch.float32))
        expected_strides = [4064, 127, 1]
        self.assertEqual(
            expected_strides, out_strides, f"{expected_strides} v.s. {out_strides}"
        )

    def test_pad_3d_tensor(self):
        """
        Constructing this test case guided by the fact that we don't pad
        placeholder or user visible output's strides.

        Add a matmul in the beginning and end so we can pad strides for
        intermediate tensors.
        """

        def f(x, y):
            x = torch.matmul(x, y)
            x = x + 1
            return torch.matmul(x, y)

        x = torch.randn(2, 16, 2047)
        y = torch.randn(2047, 2047)
        self.common_numeric_check(f, x, y, tol=1e-2)
        self.assertTrue(metrics.num_comprehensive_padding > 0)

    def test_conv(self):
        """
        Padding the input for convolution may cause extra copy kernel being called.
        Check this example trace: https://gist.github.com/shunting314/ce45398f7d51a63ce05fc8d411faddb3
        """
        x_shape = (1, 128, 640, 959)
        x1 = torch.randn(*x_shape)

        padded_stride = ir.Layout._pad_strides(x1.stride(), x1.shape, torch.float32)
        x2 = rand_strided(x_shape, padded_stride, device="cuda")
        x2.copy_(x1)

        weight = torch.randn(64, 128, 3, 3)

        def fun(x, weight):
            return torch.convolution(
                x,
                weight,
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                transposed=False,
                output_padding=(0, 0),
                groups=1,
                bias=None,
            )

        ref = fun(x1, weight)
        act = fun(x2, weight)
        self.check_close(ref, act)
        if DO_PERF_TEST:
            latency_with_padding = benchmarker.benchmark_gpu(lambda: fun(x2, weight))
            latency_without_padding = benchmarker.benchmark_gpu(lambda: fun(x1, weight))
            print(
                f"Latency with and without padding: {latency_with_padding:.3f} v.s. {latency_without_padding:.3f}"
            )

            self.do_profiling(lambda: fun(x2, weight), lambda: fun(x1, weight))

    @unittest.skipIf(not DO_PERF_TEST, "Perf test not enabled")
    def test_cat(self):
        """
        Compare the perf between aten cat and compiled cat.

        Latency between eager and compiled: 1.596 v.s. 0.601

        Eager cat can be 2.66x slower than inductor kernel.
        """
        x = torch.randn(8192, 30522, dtype=torch.float16)

        def f(x):
            pad = x.new_zeros(x.size(0), 6)
            return torch.cat([x, pad], dim=1)

        # disable cudagraphs since cudagraphs need copy the input which
        # distort the latency a lot! (double the latency here for compiled
        # version)
        with config.patch("triton.cudagraphs", False):
            opt_f = torch.compile(f)
            opt_f(x)
        eager_time = benchmarker.benchmark_gpu(lambda: f(x))
        opt_time = benchmarker.benchmark_gpu(lambda: opt_f(x))
        print(
            f"Latency between eager and compiled: {eager_time:.3f} v.s. {opt_time:.3f}"
        )
        self.do_profiling(lambda: f(x), lambda: opt_f(x), "Eager Cat", "Compiled Cat")

    def test_pad_channels_last(self):
        t = torch.randn(2, 3, 5, 1025)
        in_strides = t.stride()
        out_strides = ir.Layout._pad_strides(in_strides, t.shape, torch.float32)
        self.assertTrue(in_strides != out_strides)

        t = t.to(memory_format=torch.channels_last)
        in_strides = t.stride()
        out_strides = ir.Layout._pad_strides(in_strides, t.shape, torch.float32)
        self.assertTrue(in_strides == out_strides)

    @parametrize("alignment_bytes", (32, 128))
    @parametrize("shape", [(21, 19), (3, 5, 71)])
    @parametrize("dtype", (torch.float16, torch.float32))
    def test_pad_outputs(
        self, dtype: torch.dtype, shape: Tuple[int], alignment_bytes: int
    ):
        """
        Tests padding output tensors to a specific alignment.
        This is enabled by a config flag.
        """
        func = torch.add
        inputs = tuple(torch.randn(*shape, dtype=dtype) for input_idx in range(2))

        # Compile and run
        with config.patch(
            {
                "comprehensive_padding": True,
                "padding_alignment_bytes": alignment_bytes,
                "padding_stride_threshold": 0,
                "pad_outputs": True,
            }
        ):
            compiled_func = torch.compile(func)
            compiled_out = compiled_func(*inputs)

        # Check numerics
        eager_out = func(*inputs)
        self.check_close(eager_out, compiled_out)

        # Compute the expected padding
        element_size = torch.tensor([], dtype=dtype).element_size()
        self.assertGreater(alignment_bytes, element_size)
        self.assertEqual(alignment_bytes % element_size, 0)
        alignment_elements = alignment_bytes // element_size
        contiguous_stride = inputs[0].stride()
        expected_stride = [1]
        for dim in reversed(shape[1:]):
            slice_size = dim * expected_stride[0]
            new_stride = alignment_elements * ceildiv(slice_size, alignment_elements)
            expected_stride.insert(0, new_stride)
        expected_stride = tuple(expected_stride)
        self.assertNotEqual(expected_stride, contiguous_stride)

        # Check strides
        self.assertFalse(compiled_out.is_contiguous())
        self.assertEqual(compiled_out.stride(), expected_stride)


if __name__ == "__main__":
    if HAS_CUDA:
        run_tests()
