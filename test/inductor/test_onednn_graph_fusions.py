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
        rtol=1.3e-6,
    ):
        if args1 is None:
            tensor_shape = (4, 2, 16, 32)
            args1 = [
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
            ]
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
        if not has_dropout or override_check_equal:
            self.assertEqual(result1, result2, atol=atol, rtol=1.3e-6)

    def _test_sdpa_rewriter_5(self):
        def sfdp_pattern_5_v1(query, key, value):
            attn_mask = torch.ones(
                query.size(-2), key.size(-2), dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            attn_mask = attn_mask.masked_fill(
                torch.logical_not(attn_mask), -float("inf")
            )
            attn_weight = torch.softmax(
                (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask,
                dim=-1,
            )
            return attn_weight @ value

        def sfdp_pattern_5_v2(query, key, value):
            # https://github.com/pytorch/pytorch/issues/100318.
            attn_mask = torch.zeros(
                query.size(-2), key.size(-2), dtype=torch.bool, device=query.device
            ).bool()
            attn_weight = torch.softmax(
                (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask,
                dim=-1,
            )
            return attn_weight @ value

        self._check_common(sfdp_pattern_5_v1)
        self._check_common(checkpoint_wrapper(sfdp_pattern_5_v1))
        self._check_common(sfdp_pattern_5_v2)
        self._check_common(checkpoint_wrapper(sfdp_pattern_5_v2))


if HAS_CPU:

    class SDPAPatternRewriterCpuTests(TestSDPAPatternRewriterTemplate):
        device = "cpu"
        test_sdpa_rewriter_5_cpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_5

    class SDPAPatternRewriterCpuDynamicTests(SDPAPatternRewriterCpuTests):
        use_static_shapes = False


if __name__ == "__main__":
    if IS_LINUX and torch._C._has_onednn_graph and IS_AVX512_VNNI_SUPPORTED:
        run_tests()
