# Owner(s): ["module: dynamo"]

import torch
import torch._inductor.test_case
import torch.fx.traceback as fx_traceback
import torch.utils.checkpoint
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.test_case import run_tests
from torch._inductor.utils import run_fw_bw_and_get_code
from torch.fx.passes.regional_inductor import compile_fx_annotated_nodes_with_inductor
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
    flex_attention_hop,
)

# from torch._inductor.utils import run_and_get_code
from torch.testing._internal import common_utils
from torch.testing._internal.triton_utils import requires_cuda_and_triton


# Some issues raised in the HOP meeting
# 1) CSE will not differentiate different meta custom nodes and do wrong thing.
# 2) SAC - The recomputed forward will be smaller than the forward. Will we
# compile a smaller region than?
# 3) What happens if you have a op in the middle whcih does not disturb
# topology, is it still 1 subgraph?
# 4) What happens with the nesting of fx_traceback.annotate? Are there ordering
# requirements?
# 5) What are we going to use the annotations for?
#   a) compile flex
#   b) streams
#   c) nn.MOdule info to organize MoE runtime
#   d) PP stages
#   e) rename graph nodes for more debugging.
#   f) No nested regional compile


def checkpoint_wrapper(fn):
    def inner(*args):
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=True)

    return inner


def aot_eager_regional_inductor():
    return aot_autograd(
        fw_compiler=compile_fx_annotated_nodes_with_inductor,
        bw_compiler=compile_fx_annotated_nodes_with_inductor,
    )


class RegionalInductorTests(torch._inductor.test_case.TestCase):
    def test_simple(self):
        def fn(x, y):
            sin = torch.sin(x)

            with fx_traceback.annotate({"compile_with_inductor": 0}):
                mul = sin * y
                add = mul + 1

            return torch.sin(add)

        opt_fn = torch.compile(
            fn, backend=aot_eager_regional_inductor(), fullgraph=True
        )
        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10, requires_grad=True)

        # Check that inductor compilation is called twicw
        _, codes = run_fw_bw_and_get_code(lambda: opt_fn(x, y))
        self.assertEqual(len(codes), 2)

    def test_repeated_blocks(self):
        def fn(x, y):
            sin = torch.sin(x)

            with fx_traceback.annotate({"compile_with_inductor": 0}):
                mul = sin * y
                add = mul + 1

            return torch.sin(add)

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                a = fn(x, y)
                return fn(a, y)

        mod = Mod()

        opt_mod = torch.compile(
            mod, backend=aot_eager_regional_inductor(), fullgraph=True
        )
        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10, requires_grad=True)

        # Check that inductor compilation is called 4 times
        # there will be 2 partitions in the fwd and 2 in the bwd, totalling 4
        _, codes = run_fw_bw_and_get_code(lambda: opt_mod(x, y))
        self.assertEqual(len(codes), 4)

    def test_invoke_subgraph(self):
        # Checks that get_attr nodes custom metadata is propagated
        @torch.compiler.nested_compile_region
        def gn(x):
            return torch.sin(x)

        def fn(x):
            x = x + 1
            with fx_traceback.annotate({"compile_with_inductor": 0}):
                z = gn(x)
            return torch.sigmoid(z)

        opt_fn = torch.compile(
            fn, backend=aot_eager_regional_inductor(), fullgraph=True
        )
        x = torch.randn(10, requires_grad=True)

        _, codes = run_fw_bw_and_get_code(lambda: opt_fn(x))
        self.assertEqual(len(codes), 2)

    def test_invoke_subgraph_inner(self):
        # Checks that the inductor regions are searched recursively.
        @torch.compiler.nested_compile_region
        def gn(x):
            with fx_traceback.annotate({"compile_with_inductor": 0}):
                return torch.sin(x)

        def fn(x):
            x = x + 1
            x = gn(x)
            x = x + 1
            x = gn(x)
            return torch.sigmoid(x)

        opt_fn = torch.compile(
            fn, backend=aot_eager_regional_inductor(), fullgraph=True
        )
        x = torch.randn(10, requires_grad=True)

        _, codes = run_fw_bw_and_get_code(lambda: opt_fn(x))
        # the invoke_subgraph is called twice - but the inside code is compiled
        # once - so in total 2 (1 fwd + 1 bwd)
        self.assertEqual(len(codes), 2)

    @requires_cuda_and_triton
    def test_flex_attention(self):
        def _squared(score, b, h, m, n):
            return score * score

        def mask_mod(b, h, q, k):
            return q >= 0

        a = 12
        b = 64
        block_mask = create_block_mask(mask_mod, None, None, a * b, a * b)

        def fn(x):
            x = torch.sin(x)
            with fx_traceback.annotate({"compile_with_inductor": 0}):
                x = flex_attention(x, x, x, block_mask=block_mask, score_mod=_squared)
            return torch.cos(x)

        x = torch.randn(
            1,
            1,
            a * b,
            b,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )

        opt_fn = torch.compile(
            fn,
            backend=aot_eager_regional_inductor(),
            fullgraph=True,
        )

        _, codes = run_fw_bw_and_get_code(lambda: opt_fn(x))
        # flex in forward and flex_backward in backward
        self.assertEqual(len(codes), 2)

    @requires_cuda_and_triton
    @common_utils.parametrize(
        "ops_to_save",
        [
            [
                torch.ops.aten.mm.default,
            ],
            [
                flex_attention_hop,
            ],
            [torch.ops.aten.mm.default, flex_attention_hop],
        ],
    )
    def test_selective_ac_flex(self, device, ops_to_save):
        class FlexAttentionModule(torch.nn.Module):
            def __init__(self, hidden_size, num_heads):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                self.head_dim = hidden_size // num_heads

                # In-projections (query, key, value)
                self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
                self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
                self.v_proj = torch.nn.Linear(hidden_size, hidden_size)

                # Out-projection
                self.out_proj = torch.nn.Linear(hidden_size, hidden_size)

            def forward(self, x):
                batch_size, seq_len, _ = x.size()

                # Project queries, keys, and values
                q = (
                    self.q_proj(x)
                    .view(batch_size, seq_len, self.num_heads, self.head_dim)
                    .transpose(1, 2)
                )
                k = (
                    self.k_proj(x)
                    .view(batch_size, seq_len, self.num_heads, self.head_dim)
                    .transpose(1, 2)
                )
                v = (
                    self.v_proj(x)
                    .view(batch_size, seq_len, self.num_heads, self.head_dim)
                    .transpose(1, 2)
                )

                # Apply flex attention
                attn_output = flex_attention(
                    q,
                    k,
                    v,
                )

                # Reshape output
                attn_output = (
                    attn_output.transpose(1, 2)
                    .contiguous()
                    .view(batch_size, seq_len, self.hidden_size)
                )

                # Out projection
                output = self.out_proj(attn_output)

                return output

        from torch.utils.checkpoint import (
            checkpoint,
            create_selective_checkpoint_contexts,
        )

        context_fn = functools.partial(
            create_selective_checkpoint_contexts, ops_to_save
        )

        # Define a model that uses FlexAttention with selective activation checkpointing
        class SacModule(torch.nn.Module):
            def __init__(self, hidden_size, num_heads, context_fn):
                super().__init__()
                self.flex_attn = FlexAttentionModule(hidden_size, num_heads)
                self.context_fn = context_fn

            def forward(self, x):
                def flex_attn_fn(x):
                    return self.flex_attn(x)

                output = checkpoint(
                    flex_attn_fn,
                    x,
                    use_reentrant=False,
                    context_fn=self.context_fn,
                )

                return output

        flex_module = SacModule(hidden_size=512, num_heads=8, context_fn=context_fn).to(
            "cuda", dtype=torch.bfloat16
        )
        x = torch.ones(8, 1024, 512, device="cuda", dtype=torch.bfloat16)

        # Run without compilation
        output_module = flex_module(x)
        compiled_module = torch.compile(flex_module)
        output_compiled = compiled_module(x)


if __name__ == "__main__":
    run_tests()
