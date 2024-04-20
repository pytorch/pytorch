# Owner(s): ["module: inductor"]

import functools
from collections import namedtuple
from typing import Callable

from unittest import expectedFailure, skipUnless
from unittest.mock import patch

import torch
from torch._higher_order_ops.templated_attention import (
    templated_attention as templated_attention_hop,
)
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.nn.attention._templated_attention import _compose, _templated_attention
from torch.testing import FileCheck
from torch.testing._internal import common_utils
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_BF16
from torch.utils._triton import has_triton

# Skip tests if Triton is not available
supported_platform = skipUnless(
    torch.cuda.is_available() and has_triton() and torch.version.hip is None,
    "Requires CUDA and Triton",
)

Tolerances = namedtuple("Tolerances", ["atol", "rtol"])
torch.set_float32_matmul_precision("high")


def create_attention(score_mod):
    return functools.partial(_templated_attention, score_mod=score_mod)


test_dtypes = (
    [torch.float16, torch.bfloat16, torch.float32]
    if PLATFORM_SUPPORTS_BF16
    else [torch.float16, torch.float32]
)

# TODO float16 was causing ERRORs for tests on ROCm
# See https://github.com/pytorch/pytorch/issues/123531
if common_utils.TEST_WITH_ROCM:
    test_dtypes = [torch.float32]


def _identity_mod(score, b, h, m, n):
    return score


def _causal_mod(score, b, h, token_q, token_kv):
    return torch.where(token_q >= token_kv, score, float("-inf"))


class TestTemplatedSDPA(InductorTestCase):
    def run_test(self, score_mod: Callable, dtype: torch.dtype = torch.float16):
        sdpa_partial = create_attention(score_mod)
        compiled_sdpa = torch.compile(sdpa_partial)
        q = torch.randn((4, 8, 2048, 64), dtype=dtype, device="cuda")
        k = torch.randn((4, 8, 2048, 64), dtype=dtype, device="cuda")
        v = torch.randn((4, 8, 2048, 64), dtype=dtype, device="cuda")
        golden_out = sdpa_partial(
            q.to(torch.float64), k.to(torch.float64), v.to(torch.float64)
        )
        ref_out = sdpa_partial(q, k, v)
        compiled_out = compiled_sdpa(q, k, v)

        compiled_error = (golden_out - compiled_out).abs().mean()
        ref_error = (golden_out - ref_out).abs().mean()
        # Note, it seems like we really are less accurate than the float32
        # computation, likely due to the online softmax
        if dtype == torch.float32:
            fudge_factor = 4.0
        else:
            fudge_factor = 1.1
        if compiled_error > ref_error * fudge_factor:
            msg = f"Compiled error {compiled_error} is greater than ref error {ref_error} by more than {fudge_factor}X."
            self.assertTrue(False, msg)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    def test_identity(self, dtype: torch.dtype):
        def score_mod(score, b, h, m, n):
            return score

        self.run_test(score_mod, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    def test_causal_mask(self, dtype: torch.dtype):
        def score_mod(score, b, h, token_q, token_kv):
            return torch.where(token_q >= token_kv, score, float("-inf"))

        self.run_test(score_mod, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    def test_rel_bias(self, dtype: torch.dtype):
        def score_mod(score, b, h, m, n):
            return score + (m - n)

        self.run_test(score_mod, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    def test_alibi_bias(self, dtype: torch.dtype):
        def score_mod(score, b, h, m, n):
            return score + (m - n) * h

        self.run_test(score_mod, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    def test_rel_causal(self, dtype: torch.dtype):
        def score_mod(score, b, h, m, n):
            return torch.where(m <= n, score + (m - n), float("-inf"))

        self.run_test(score_mod, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    def test_skip_odd_keys(self, dtype: torch.dtype):
        def score_mod(score, b, h, q, kv):
            return torch.where(kv % 2 == 0, score, float("-inf"))

        self.run_test(score_mod, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    def test_alibi_causal(self, dtype: torch.dtype):
        def score_mod(score, b, h, m, n):
            return torch.where(m <= n, score + (m - n) * h, float("-inf"))

        self.run_test(score_mod, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    def test_function_composition(self, dtype: torch.dtype):
        def score_mod_1(score, b, h, m, n):
            return score + (m - n)

        def score_mod_2(score, b, h, m, n):
            return torch.where(m <= n, score, float("-inf"))

        composed_score_mod = _compose(score_mod_1, score_mod_2)

        self.run_test(composed_score_mod, dtype)

    # TODO We are currently not capturing free variables in the closure correctly
    @expectedFailure
    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    def test_captured_buffers(self, dtype: torch.dtype):
        head_offset = torch.rand(8, device="cuda", dtype=dtype)

        def score_mod(score, b, h, m, n):
            return score + head_offset[h]

        self.run_test(score_mod, dtype)

    @supported_platform
    def test_backwards_fails(self):
        make_tensor = functools.partial(
            torch.randn,
            (4, 8, 2048, 64),
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        out = _templated_attention(q, k, v, _identity_mod)
        with self.assertRaisesRegex(
            RuntimeError, "Autograd not implemented for templated_attention"
        ):
            out.backward(torch.ones_like(out))

    @supported_platform
    def test_mixed_dtypes_fails(self):
        query = torch.randn((1, 1, 2048, 64), dtype=torch.float32, device="cuda")
        key = torch.randn((1, 1, 2048, 64), dtype=torch.float16, device="cuda")
        value = torch.randn((1, 1, 2048, 64), dtype=torch.float16, device="cuda")
        with self.assertRaisesRegex(
            ValueError, "Expected query, key, and value to have the same dtype"
        ):
            _templated_attention(query, key, value, _identity_mod)

    @supported_platform
    def test_different_sequence_length_fails(self):
        query = torch.randn((1, 1, 2048, 64), dtype=torch.float32, device="cuda")
        key = torch.randn((1, 1, 1024, 64), dtype=torch.float32, device="cuda")
        value = torch.randn((1, 1, 1024, 64), dtype=torch.float32, device="cuda")
        with self.assertRaisesRegex(ValueError, "NYI: The target sequence length"):
            _templated_attention(query, key, value, _identity_mod)

    @supported_platform
    @patch.object(torch._inductor.config, "max_autotune", True)
    def test_max_autotune(self):
        def score_mod(score, b, h, m, n):
            return score * 2

        self.run_test(score_mod)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    @common_utils.parametrize("score_mod", [_identity_mod, _causal_mod])
    def test_logsumexp_correctness(self, dtype, score_mod):
        @torch.compile
        def sdpa_hop(q, k, v, score_mod):
            return templated_attention_hop(q, k, v, score_mod)

        make_tensor = functools.partial(
            torch.randn,
            (4, 8, 2048, 64),
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        ref_out, ref_lse = templated_attention_hop(
            q.to(torch.float64), k.to(torch.float64), v.to(torch.float64), score_mod
        )
        compiled_out, compiled_lse = sdpa_hop(q, k, v, score_mod)

        # Comparing LSE for the ref and the compiled version
        # The compiled uses a change of base trick to more efficiently compute the LSE
        # this means that the base for the LSE computed by ref is e while for the compiled
        # version it is 2. To compare we use the change of base formula
        # log_2(x_compiled) = log_e(x_ref) * log_2(e) where
        # x_ref      = ∑_i e^(scores[i])
        # x_compiled = ∑_i 2^(log2(e) * scores[i])

        self.assertTrue(ref_lse.dtype == torch.float32)
        self.assertTrue(compiled_lse.dtype == torch.float32)
        ref_lse = ref_lse * torch.log2(torch.tensor(torch.e))

        tolerance = Tolerances(atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(
            ref_out.to(dtype=torch.float32),
            compiled_out.to(dtype=torch.float32),
            atol=tolerance.atol,
            rtol=tolerance.rtol,
        )
        torch.testing.assert_close(
            ref_lse.to(dtype=torch.float32),
            compiled_lse.to(dtype=torch.float32),
            atol=tolerance.atol,
            rtol=tolerance.rtol,
        )

    @supported_platform
    def test_logsumexp_only_return(self):
        make_tensor = functools.partial(
            torch.randn,
            (4, 8, 2048, 64),
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        @torch.compile
        def func(q, k, v, score_mod):
            _, lse = templated_attention_hop(q, k, v, score_mod)
            lse_2 = lse * 2
            return lse_2

        _, code = run_and_get_code(func, q, k, v, _identity_mod)
        # Ensure that two kernels are generated
        FileCheck().check_count(".run(", 2, True).run(code[0])

    @supported_platform
    def test_logsumexp_is_not_fused(self):
        make_tensor = functools.partial(
            torch.randn,
            (4, 8, 2048, 64),
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        @torch.compile
        def func(q, k, v, score_mod):
            out, lse = templated_attention_hop(q, k, v, score_mod)
            lse_2 = lse * 2
            return out, lse_2

        _, code = run_and_get_code(func, q, k, v, _identity_mod)
        # Ensure that two kernels are generated
        FileCheck().check_count(".run(", 2, True).run(code[0])


common_utils.instantiate_parametrized_tests(TestTemplatedSDPA)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
