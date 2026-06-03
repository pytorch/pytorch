# Owner(s): ["oncall: distributed"]


import torch
import torch.distributed as dist
from torch.distributed.token_switch import TokenSwitch, TokenSwitchNCCL
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
)


def requires_nccl_ep():
    return skip_but_pass_in_sandcastle_if(
        not torch.cuda.is_available()
        or not hasattr(torch._C._distributed_c10d, "_NcclEpGroup"),
        "Test requires USE_NCCL_EP build",
    )


NUM_TOKENS = 16
TOP_K = 1
HIDDEN = 64
TOKEN_SIZE_BYTES = HIDDEN * 2
NUM_MULTI_ROUND_DISPATCH_COMBINE = 3


def _generate_topk(rank, world_size, num_tokens, top_k, device):
    remote_expert = (rank + 1) % world_size
    topk_idx = torch.full(
        (num_tokens, top_k), remote_expert, dtype=torch.int64, device=device
    )
    topk_weights = torch.full(
        (num_tokens, top_k), 1.0 / top_k, dtype=torch.float32, device=device
    )
    return topk_idx, topk_weights


@requires_nccl_ep()
class TokenSwitchNCCLTest(MultiProcContinuousTest):
    _cached_token_switch: TokenSwitchNCCL | None = None

    @classmethod
    def backend_str(cls):
        return "nccl"

    @property
    def device(self):
        return torch.device("cuda", self.rank)

    @classmethod
    def get_token_switch(cls) -> TokenSwitchNCCL:
        if cls._cached_token_switch is None:
            pg = dist.distributed_c10d._get_default_group()
            rank = dist.get_rank(pg)
            world_size = dist.get_world_size(pg)
            print(f"rank {rank} creating token switch")
            dist.barrier(pg)
            cls._cached_token_switch = TokenSwitchNCCL(
                pg, world_size, NUM_TOKENS, world_size * NUM_TOKENS, TOKEN_SIZE_BYTES
            )
        return cls._cached_token_switch

    def _init(self):
        torch.cuda.set_device(self.device)
        dist.barrier()

    @skip_if_lt_x_gpu(2)
    def test_create_routing(self):
        self._init()
        ts = self.get_token_switch()
        num_experts = self.world_size
        num_local_experts = num_experts // self.world_size
        topk_idx, _topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        per_expert_counts = torch.zeros(
            num_local_experts, dtype=torch.int32, device=self.device
        )
        ts.create_routing(topk_idx, per_expert_counts)
        self.assertEqual(per_expert_counts.dtype, torch.int32)
        self.assertEqual(per_expert_counts.numel(), num_local_experts)
        self.assertEqual(per_expert_counts.item(), NUM_TOKENS)
        torch.cuda.synchronize()

    @skip_if_lt_x_gpu(2)
    def test_dispatch(self):
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        routing = ts.create_routing(topk_idx)

        token_val = float(self.rank + 1)
        tokens = torch.full(
            (NUM_TOKENS, HIDDEN), token_val, dtype=torch.bfloat16, device=self.device
        )
        out_tokens = torch.zeros(
            (num_recv_tokens, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        out_topk_weights = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.float32, device=self.device
        )
        out_topk_idx = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.int64, device=self.device
        )

        ts.dispatch(
            routing, tokens, topk_weights, out=(out_tokens, out_topk_weights, out_topk_idx)
        )
        torch.cuda.synchronize()

        src_rank = (self.rank - 1) % self.world_size
        expected_val = float(src_rank + 1)
        received = out_tokens[:NUM_TOKENS].float()
        self.assertTrue(
            received.eq(expected_val).all(),
            f"rank {self.rank}: expected {expected_val}, got {received[0, 0].item()}",
        )

    @skip_if_lt_x_gpu(2)
    def test_dispatch_combine_roundtrip(self):
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        routing = ts.create_routing(topk_idx)

        token_val = float(self.rank + 1)
        tokens = torch.full(
            (NUM_TOKENS, HIDDEN), token_val, dtype=torch.bfloat16, device=self.device
        )
        out_tokens = torch.zeros(
            (num_recv_tokens, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        out_topk_weights = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.float32, device=self.device
        )
        out_topk_idx = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.int64, device=self.device
        )

        ts.dispatch(
            routing, tokens, topk_weights, out=(out_tokens, out_topk_weights, out_topk_idx)
        )
        torch.cuda.synchronize()

        expert_tokens = out_tokens[:NUM_TOKENS].contiguous()
        combined = torch.zeros(
            (NUM_TOKENS, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        ts.combine(routing, expert_tokens, out=combined)
        torch.cuda.synchronize()

        expected = torch.full((NUM_TOKENS, HIDDEN), token_val, dtype=torch.bfloat16)
        self.assertEqual(combined.cpu(), expected)

    @skip_if_lt_x_gpu(2)
    def test_dispatch_autograd_backward(self):
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        routing = ts.create_routing(topk_idx)

        tokens = torch.full(
            (NUM_TOKENS, HIDDEN), float(self.rank + 1), dtype=torch.bfloat16, device=self.device,
            requires_grad=True,
        )
        out_tokens, _out_weights, _out_idx = ts.dispatch(routing, tokens, topk_weights, num_recv_tokens)
        out_tokens.sum().backward()
        torch.cuda.synchronize()

        # grad_out_tokens is all-ones; combine routes them back: each token gets 1.0 per top-k slot
        self.assertIsNotNone(tokens.grad)
        self.assertEqual(tokens.grad, torch.ones_like(tokens))

    @skip_if_lt_x_gpu(2)
    def test_combine_autograd_backward(self):
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        routing = ts.create_routing(topk_idx)

        # Pre-dispatch so the routing handle is primed, then test combine autograd
        tokens = torch.full(
            (NUM_TOKENS, HIDDEN), float(self.rank + 1), dtype=torch.bfloat16, device=self.device
        )
        out_tokens_buf = torch.zeros(num_recv_tokens, HIDDEN, dtype=torch.bfloat16, device=self.device)
        out_weights_buf = torch.zeros(num_recv_tokens, TOP_K, dtype=torch.float32, device=self.device)
        out_idx_buf = torch.zeros(num_recv_tokens, TOP_K, dtype=torch.int64, device=self.device)
        ts.dispatch(routing, tokens, topk_weights, out=(out_tokens_buf, out_weights_buf, out_idx_buf))
        torch.cuda.synchronize()

        expert_tokens = out_tokens_buf[:NUM_TOKENS].contiguous().detach().requires_grad_(True)
        combined = ts.combine(routing, expert_tokens)
        combined.sum().backward()
        torch.cuda.synchronize()

        # grad_out_tokens is all-ones; dispatch routes them back to expert ranks
        self.assertIsNotNone(expert_tokens.grad)
        self.assertEqual(expert_tokens.grad, torch.ones_like(expert_tokens))

    @skip_if_lt_x_gpu(2)
    def test_dispatch_combine_autograd_roundtrip(self):
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        routing = ts.create_routing(topk_idx)

        token_val = float(self.rank + 1)
        tokens = torch.full(
            (NUM_TOKENS, HIDDEN), token_val, dtype=torch.bfloat16, device=self.device,
            requires_grad=True,
        )
        dispatched, _weights, _idx = ts.dispatch(routing, tokens, topk_weights, num_recv_tokens)
        expert_out = dispatched[:NUM_TOKENS].contiguous()
        combined = ts.combine(routing, expert_out)
        combined.sum().backward()
        torch.cuda.synchronize()

        # With identity expert (no-op), gradient should round-trip back as all-ones
        self.assertIsNotNone(tokens.grad)
        self.assertEqual(tokens.grad, torch.ones_like(tokens))

    @skip_if_lt_x_gpu(2)
    def test_dispatch_combine_multiple_rounds(self):
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        routing = ts.create_routing(topk_idx)

        out_tokens = torch.zeros(
            (num_recv_tokens, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        out_topk_weights = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.float32, device=self.device
        )
        out_topk_idx = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.int64, device=self.device
        )
        combined = torch.zeros(
            (NUM_TOKENS, HIDDEN), dtype=torch.bfloat16, device=self.device
        )

        for r in range(NUM_MULTI_ROUND_DISPATCH_COMBINE):
            token_val = float(self.rank + 1 + r)
            tokens = torch.full(
                (NUM_TOKENS, HIDDEN),
                token_val,
                dtype=torch.bfloat16,
                device=self.device,
            )
            ts.dispatch(
                routing, tokens, topk_weights, out=(out_tokens, out_topk_weights, out_topk_idx)
            )
            torch.cuda.synchronize()

            expert_tokens = out_tokens[:NUM_TOKENS].contiguous()
            ts.combine(routing, expert_tokens, out=combined)
            torch.cuda.synchronize()

        expected = torch.full((NUM_TOKENS, HIDDEN), token_val, dtype=torch.bfloat16, device=self.device)
        self.assertEqual(combined, expected)


if __name__ == "__main__":
    run_tests()
