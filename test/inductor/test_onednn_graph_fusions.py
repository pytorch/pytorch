# Owner(s): ["module: inductor"]
import itertools
import math

import torch
import torch._inductor.config
import torch._inductor.config as inductor_config
import torch.utils.checkpoint
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import IS_AVX512_VNNI_SUPPORTED, IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CPU


def checkpoint_wrapper(fn):
    def inner(*args):
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=True)

    return inner


@inductor_config.patch(onednn_graph=True)
class TestSDPAPatternRewriterTemplate(TestCase):
    use_static_shapes = True

    def _clone_inputs(self, inputs):
        def clone(x):
            if not isinstance(x, torch.Tensor):
                return x
            return x.clone()

        return [clone(x) for x in inputs]

    def _check_common(
        self,
        dot_prod_attention,
        args1=None,
        atol=1e-5,
        has_fuse_pattern=True,
        has_dropout=False,
        override_check_equal=False,
        dtype=torch.float,
        uses_causal_mask=False,
        rtol=1.3e-6,
        batch_size=4,
    ):
        if args1 is None:
            tensor_shape = (batch_size, 2, 16, 32)
            args1 = [
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
            ]
            if uses_causal_mask:
                args1.append(
                    torch.full(
                        (),
                        math.sqrt(args1[0].size(1)),
                        dtype=dtype,
                        device="cpu",
                    )
                )
                args1.append(
                    torch.full((), -3.4028234663852886e38, dtype=dtype, device="cpu")
                )
                if batch_size == 1:
                    args1.append(torch.ones(1, 1, 16, 16).to(torch.bool))
                else:
                    args1.append(torch.ones(1, 1, 2, 2).to(torch.bool))
        else:
            args1 = list(args1)
        args2 = self._clone_inputs(args1)

        for x in itertools.chain(args1[:], args2[:]):
            if isinstance(x, torch.Tensor) and x.is_floating_point():
                x.requires_grad = False

        if not self.use_static_shapes:
            torch._dynamo.mark_dynamic(args2[0], 0)
            torch._dynamo.mark_dynamic(args2[1], 0)
            torch._dynamo.mark_dynamic(args2[2], 0)

        dropout_arg = [False] if has_dropout else []
        torch.manual_seed(1234)
        result1 = dot_prod_attention(*(args1 + dropout_arg))

        counters.clear()
        torch.manual_seed(1234)
        result2, source_code = run_and_get_code(
            torch.compile(dot_prod_attention, fullgraph=True),
            *(args2 + dropout_arg),
        )
        source_code = "\n".join(source_code)
        self.assertGreaterEqual(counters["inductor"]["fuse_attention"], 1)
        # some tests configured with very low dropout where we still want to check equality
        # if not has_dropout or override_check_equal:
        #    self.assertEqual(result1, result2, atol=atol, rtol=1.3e-6)

    def _test_sdpa_rewriter_18(self):
        def sfdp_pattern_18(
            query, key, value, inv_scale, causal_mask_value, causal_mask
        ):
            # for hf_GPT2 with dropout
            query = query.permute([0, 2, 1, 3])
            key = key.permute([0, 2, 1, 3])
            value = value.permute([0, 2, 1, 3])
            attn_weights = torch.matmul(query, key.permute(0, 1, 3, 2))
            attn_weights = attn_weights.div(inv_scale)
            attn_weights = torch.where(causal_mask, attn_weights, causal_mask_value)
            return (
                (
                    torch.nn.functional.dropout(attn_weights.softmax(dim=-1), 0.0)
                    .matmul(value)
                    .permute([0, 2, 1, 3])
                    .contiguous()
                ),
                key,
                value,
            )

        self._check_common(sfdp_pattern_18, uses_causal_mask=True)
        self._check_common(checkpoint_wrapper(sfdp_pattern_18), uses_causal_mask=True)

    def _test_sdpa_rewriter_19(self):
        def sfdp_pattern_19(
            query, key, value, inv_scale, causal_mask_value, causal_mask
        ):
            # for hf_GPT2 with dropout (batch size 1)
            attn_weights = torch.matmul(query, key.permute(0, 1, 3, 2))
            attn_weights = attn_weights.div(inv_scale)
            attn_weights = torch.where(causal_mask, attn_weights, causal_mask_value)
            return (
                torch.nn.functional.dropout(attn_weights.softmax(dim=-1), 0.0)
                .matmul(value)
                .permute([0, 2, 1, 3])
                .contiguous()
            )

        self._check_common(sfdp_pattern_19, uses_causal_mask=True, batch_size=1)
        self._check_common(
            checkpoint_wrapper(sfdp_pattern_19), uses_causal_mask=True, batch_size=1
        )


if HAS_CPU:

    class SDPAPatternRewriterCpuTests(TestSDPAPatternRewriterTemplate):
        device = "cpu"
        test_sdpa_rewriter_18_cpu = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_18
        )
        test_sdpa_rewriter_19_cpu = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_19
        )


if __name__ == "__main__":
    if IS_LINUX and torch._C._has_onednn_graph and IS_AVX512_VNNI_SUPPORTED:
        run_tests()
