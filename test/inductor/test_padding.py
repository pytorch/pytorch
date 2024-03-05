# Owner(s): ["module: inductor"]
import copy
import os

import torch
from torch import nn
from torch._dynamo.test_case import run_tests, TestCase
from torch._inductor import config, ir
from torch._inductor.fx_passes import pad_mm as pad_mm_pass
from torch._inductor.utils import do_bench, run_and_get_code
from torch.testing._internal.inductor_utils import HAS_CUDA

DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"


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
    loss = m(*inputs).sum().backward()


@config.patch(
    {
        "benchmark_kernel": True,
        "triton.unique_kernel_names": True,
    }
)
class PaddingTest(TestCase):
    def common_numeric_check(self, f, *args):
        opt_f = torch.compile(f)
        ref = f(*args)
        act = opt_f(*args)
        tol = 1e-3
        self.assertTrue(
            torch.allclose(ref, act, atol=tol, rtol=tol), f"ref:\n{ref}\nact:\n{act}"
        )

    def test_mm_perf(self):
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
            ms = do_bench(lambda: f(a, b))
            print(f"MxKxN {M}x{K}x{N} {f.__name__}: {ms:.3f}ms")

    def test_nobias_single(self):
        self.test_single(bias=False)

    def test_nobias_both(self):
        self.test_both(bias=False)

    def test_single(self, bias=True):
        m_bad_shape = LinearAndSoftmax(vocab_size=30523, bias=bias)
        inputs_bad_shape = m_bad_shape.get_example_inputs()
        m_bad_shape_opt = torch.compile(copy.deepcopy(m_bad_shape))

        _, wrapper_codes = run_and_get_code(
            forward_and_backward_pass, m_bad_shape_opt, inputs_bad_shape
        )
        forward_and_backward_pass(m_bad_shape, inputs_bad_shape)
        self.assertTrue(
            torch.allclose(
                m_bad_shape.linear.weight.grad, m_bad_shape_opt.linear.weight.grad
            )
        )
        self.assertTrue(len(wrapper_codes) == 2)  # one for forward and oen for backward
        forward_wrapper = wrapper_codes[0]

        # make sure the store for softmax is aligned
        self.assertTrue(
            "tl.store(out_ptr2 + (r1 + (30528*x0))" in forward_wrapper,
            f"forward_wrapper: {forward_wrapper}",
        )

        if DO_PERF_TEST:
            latency = do_bench(
                lambda: forward_and_backward_pass(m_bad_shape_opt, inputs_bad_shape)
            )
            print(f"latency: {latency:.3f}ms")

    def test_both(self, bias=True):
        m_bad_shape = LinearAndSoftmax(vocab_size=30523, bias=bias)
        inptus_bad_shape = m_bad_shape.get_example_inputs()
        m_good_shape = LinearAndSoftmax(vocab_size=30528, bias=bias)
        inputs_good_shape = m_good_shape.get_example_inputs()

        m_bad_shape_opt = torch.compile(m_bad_shape)
        m_good_shape_opt = torch.compile(m_good_shape)

        if DO_PERF_TEST:
            latency_good_shape = do_bench(
                lambda: forward_and_backward_pass(m_good_shape_opt, inputs_good_shape)
            )
            latency_bad_shape = do_bench(
                lambda: forward_and_backward_pass(m_bad_shape_opt, inptus_bad_shape)
            )
            print(
                f"Latency for good shape v.s. bad shape: {latency_good_shape:.3f}ms v.s. {latency_bad_shape:.3f}ms"
            )

    @config.patch(pattern_matcher=False)
    def test_attention(self):
        batch_size, seq_len, num_heads, hidden_size = 1, 4, 1, 16
        inv_scale = (num_heads / hidden_size) ** 0.5

        class Attention(nn.Module):
            def __init__(self):
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
        sizes = [2, 16, 127]
        in_strides = [2032, 127, 1]
        out_strides = list(ir.Layout._pad_strides(in_strides, sizes))
        expected_strides = [2048, 128, 1]
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

        x = torch.randn(2, 16, 127)
        y = torch.randn(127, 127)
        self.common_numeric_check(f, x, y)


if __name__ == "__main__":
    if HAS_CUDA:
        torch.set_float32_matmul_precision("high")
        torch.set_default_device("cuda")
        run_tests()
