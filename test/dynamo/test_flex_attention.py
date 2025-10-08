# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.config
import torch._dynamo.test_case
from torch import nn
from torch._dynamo.test_case import TestCase
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch._dynamo.testing import CompileCounterWithBackend
from torch._inductor.exc import InductorError
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


class TestFlexAttention(TestCase):
    @parametrize("backend", ["eager", "inductor"])
    def test_graph_break_fix(self, backend: str):
        # https://github.com/pytorch/pytorch/issues/164247

        class MixedFakeModeModel(nn.Module):
            def __init__(self, dim=64):
                super().__init__()
                self.dim = dim
                self.lin = torch.nn.Linear(64, 64)

            def forward(self, x):
                batch_size, seq_len, _ = x.shape

                # Process input first - this creates fake tensors in export's fake mode
                processed = self.lin(x)

                # Create some computation that depends on processed tensor
                intermediate = processed.sum(dim=-1).detach()  # Shape: (batch, seq_len)

                def dynamic_mask_function(batch_idx, head_idx, q_idx, kv_idx):
                    threshold = intermediate[
                        batch_idx, q_idx % seq_len
                    ]  # Access the captured tensor
                    return (kv_idx <= q_idx) & (threshold > 0)

                block_mask = create_block_mask(
                    mask_mod=dynamic_mask_function,
                    B=batch_size,
                    H=None,
                    Q_LEN=seq_len,
                    KV_LEN=seq_len,
                    device=x.device,
                    _compile=False,
                )
                q = processed.view(batch_size, 1, seq_len, self.dim)
                k = processed.view(batch_size, 1, seq_len, self.dim)
                v = processed.view(batch_size, 1, seq_len, self.dim)

                out = torch.compile(flex_attention)(q, k, v, block_mask=block_mask)
                out = flex_attention(q, k, v, block_mask=block_mask)

                return out

        backend_counter = CompileCounterWithBackend(backend)
        model = MixedFakeModeModel()
        compiled = torch.compile(model, backend=backend_counter, fullgraph=True)

        if backend == "inductor":
            # A known LoweringException Issue https://github.com/pytorch/pytorch/issues/157612
            with self.assertRaisesRegex(
                InductorError,
                r"Unsupported for now if query, key, value are the same buffer",
            ):
                compiled(torch.randn(2, 128, 64))
        else:
            compiled(torch.randn(2, 128, 64))

        # One graph, so no graph breaks
        self.assertEqual(backend_counter.frame_count, 1)
        self.assertEqual(len(backend_counter.graphs), 1)


instantiate_parametrized_tests(
    TestFlexAttention,
)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
