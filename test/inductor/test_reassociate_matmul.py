# Owner(s): ["module: inductor"]
import functools

import torch
from torch._dynamo.utils import counters
from torch._inductor.test_case import run_tests, TestCase


# float64 keeps the reassociated result comparable at a tight tolerance: the
# reorder is a FLOP win, not a numerics-preserving rewrite.
randn = functools.partial(torch.randn, dtype=torch.float64)


@torch._inductor.config.patch(reassociate_matmul=True)
class ReassociateMatmulTest(TestCase):
    def _check(self, fn, inputs, expected_hits, **compile_kwargs):
        counters.clear()
        expected = fn(*inputs)
        actual = torch.compile(fn, fullgraph=True, **compile_kwargs)(*inputs)
        self.assertEqual(actual, expected, atol=1e-10, rtol=1e-10)
        self.assertEqual(counters["inductor"]["reassociate_matmul"], expected_hits)

    def test_shrinking_output_dim(self):
        # (256 x 256) @ (256 x 256) @ (256 x 4): folding the narrow matrix in
        # first drops the chain from 17M to 0.5M MACs.
        def fn(a, b, c):
            return a @ b @ c

        self._check(fn, (randn(256, 256), randn(256, 256), randn(256, 4)), 1)

    def test_left_to_right_already_optimal(self):
        def fn(a, b, c):
            return a @ b @ c

        self._check(fn, (randn(8, 256), randn(256, 64), randn(64, 64)), 0)

    def test_four_matmul_chain(self):
        def fn(a, b, c, d):
            return a @ b @ c @ d

        inputs = (randn(64, 256), randn(256, 256), randn(256, 256), randn(256, 4))
        self._check(fn, inputs, 1)

    def test_bmm(self):
        def fn(a, b, c):
            return torch.bmm(torch.bmm(a, b), c)

        self._check(fn, (randn(4, 256, 256), randn(4, 256, 256), randn(4, 256, 4)), 1)

    def test_shared_intermediate_is_not_reassociated(self):
        # Reassociating would force `a @ b` to be recomputed for the second user.
        def fn(a, b, c):
            t = a @ b
            return t @ c, t.sum()

        self._check(fn, (randn(256, 256), randn(256, 256), randn(256, 4)), 0)

    def test_dynamic_shapes_are_skipped(self):
        def fn(a, b, c):
            return a @ b @ c

        a = randn(256, 256)
        torch._dynamo.mark_dynamic(a, 0)
        self._check(fn, (a, randn(256, 256), randn(256, 4)), 0, dynamic=True)

    def test_attention_output_projection_is_not_reassociated_yet(self):
        # The motivating case from #194473: attn @ V @ W_o with a narrowing W_o.
        # After decomposition the chain is bmm -> reshape -> mm (single head) or
        # sdpa -> permute/clone/reshape -> mm (multi head); the pass only
        # matches directly adjacent mm/bmm nodes, so it does not fire here.
        # This pins the gap rather than the wish: flip to 1 once the matcher
        # can see through the head-merge views.
        B, T, D, H, OUT = 2, 16, 64, 4, 8

        def single_head(attn, x, w_v, w_o):
            return (attn @ (x @ w_v)) @ w_o

        def multi_head(x, w_qkv, w_o):
            q, k, v = (x @ w_qkv).chunk(3, -1)
            q, k, v = (t.view(B, T, H, D // H).transpose(1, 2) for t in (q, k, v))
            a = torch.softmax(q @ k.transpose(-1, -2) / (D // H) ** 0.5, -1) @ v
            return a.transpose(1, 2).reshape(B, T, D) @ w_o

        attn = torch.softmax(randn(B, T, T), -1)
        self._check(single_head, (attn, randn(B, T, D), randn(D, D), randn(D, OUT)), 0)
        self._check(multi_head, (randn(B, T, D), randn(D, 3 * D), randn(D, OUT)), 0)


if __name__ == "__main__":
    run_tests()
