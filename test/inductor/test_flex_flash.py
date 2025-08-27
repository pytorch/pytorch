# Owner(s): ["module: inductor"]

import torch
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.nn.attention.flex_attention import flex_attention
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize


def _times_two(score, _b, _h, _m, _n):
    return score * 2


def _causal(score, _b, _h, token_q, token_kv):
    return torch.where(token_q >= token_kv, score, float("-inf"))


def _rel_bias(score, _b, _h, token_q, token_kv):
    return score + (token_q - token_kv)


def create_test_tensors(
    batch_size=2, num_heads=4, seq_len=512, dim=64, dtype=torch.float16, device="cuda"
):
    shape = (batch_size, num_heads, seq_len, dim)
    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=False)
    k = torch.randn(shape, device=device, dtype=dtype, requires_grad=False)
    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=False)
    return q, k, v


def flash_vs_triton(q, k, v, score_mod=None, rtol=5e-3, atol=5e-3):
    compiled_fn = torch.compile(flex_attention)
    out_flash = compiled_fn(
        q, k, v, score_mod=score_mod, kernel_options={"disable_flash": False}
    )
    out_no_flash = compiled_fn(
        q, k, v, score_mod=score_mod, kernel_options={"disable_flash": True}
    )
    torch.testing.assert_close(out_flash, out_no_flash, rtol=rtol, atol=atol)
    return out_flash, out_no_flash


def name_fn(score_mod):
    return score_mod.__name__.lstrip("_")


class TestFlexFlash(InductorTestCase):
    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_basic(self, device, dtype):
        q, k, v = create_test_tensors(dtype=dtype, device=device)
        flash_vs_triton(q, k, v)

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("score_mod", [_times_two, _causal, _rel_bias], name_fn=name_fn)
    def test_flash_attention_with_score_mod(self, device, dtype, score_mod):
        q, k, v = create_test_tensors(dtype=dtype, device=device)
        flash_vs_triton(q, k, v, score_mod=score_mod)

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_different_seq_lens(self, device, dtype):
        for seq_len in [128, 256, 1024, 2048]:
            q, k, v = create_test_tensors(seq_len=seq_len, dtype=dtype, device=device)
            compiled_fn = torch.compile(flex_attention)
            out = compiled_fn(q, k, v, kernel_options={"disable_flash": False})
            self.assertEqual(out.shape, q.shape)


instantiate_device_type_tests(TestFlexFlash, globals(), only_for="cuda")


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
