# Owner(s): ["module: inductor"]

import functools

import torch
from torch.nn.attention.templated_attention import sdpa

from torch.testing._internal.common_utils import run_tests, TestCase


def create_attention(score_mod):
    return functools.partial(sdpa, score_mod=score_mod)


class TestTemplatedSDPA(TestCase):
    def run_test(self, score_mod):
        sdpa_partial = create_attention(score_mod)
        compiled_sdpa = torch.compile(sdpa_partial)
        q = torch.randn((4, 8, 2048, 64), dtype=torch.float16, device="cuda")
        k = torch.randn((4, 8, 2048, 64), dtype=torch.float16, device="cuda")
        v = torch.randn((4, 8, 2048, 64), dtype=torch.float16, device="cuda")
        ref_out = sdpa_partial(
            q.to(torch.float64), k.to(torch.float64), v.to(torch.float64)
        )
        compiled_out = compiled_sdpa(q, k, v)
        torch.testing.assert_close(ref_out.to(dtype=torch.float32), compiled_out)

    def test_identity(self):
        def score_mod(score, b, h, m, n):
            return score

        self.run_test(score_mod)

    def test_causal_mask(self):
        def score_mod(score, b, h, m, n):
            return torch.where(m <= n, score, float("-inf"))

        self.run_test(score_mod)

    def test_rel_bias(self):
        def score_mod(score, b, h, m, n):
            return score + (m - n)

        self.run_test(score_mod)

    def test_alibi_bias(self):
        def score_mod(score, b, h, m, n):
            return score + (m - n) * h

        self.run_test(score_mod)

    def test_rel_causal(self):
        def score_mod(score, b, h, m, n):
            return (score + (m - n)) * torch.where(m <= n, score, float("-inf"))

        self.run_test(score_mod)

    def test_alibi_causal(self):
        def score_mod(score, b, h, m, n):
            return (score + (m - n) * h) * torch.where(m <= n, score, float("-inf"))

        self.run_test(score_mod)


if __name__ == "__main__":
    run_tests()
