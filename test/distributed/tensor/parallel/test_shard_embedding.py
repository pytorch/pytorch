# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, Replicate, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    NUM_DEVICES,
    skip_unless_torch_gpu,
    with_comms,
)


class DistShardEmbeddingTest(DTensorTestBase):
    # Test that sharding behavior is consistent regardless of the order of the sharded
    # dimension. This would be helpful when we shard multiple embedding modules in sequence
    # parallelism, we we might need to make element-wise addition (after broadcasting).
    @with_comms
    @skip_unless_torch_gpu
    def test_sharding_consistency_under_reordered_dimension(self):
        vocab_size, batch_size, seq_len, dim = 25, 5, 13, 13
        # The assertions are commented out to demonstrate they are unnecessary.
        # assert dim % self.world_size == 0  # embedding is sharded on dim
        # assert seq_len % self.world_size == 0  # output is sharded on seq_len

        torch.manual_seed(5)
        emb_output_shard_0 = nn.Embedding(vocab_size, dim)
        emb_output_shard_1 = deepcopy(emb_output_shard_0)

        device_mesh = DeviceMesh(self.device_type, torch.arange(0, NUM_DEVICES))
        parallelize_module(
            emb_output_shard_0, device_mesh, ColwiseParallel(output_layouts=Shard(0))
        )
        parallelize_module(
            emb_output_shard_1, device_mesh, ColwiseParallel(output_layouts=Shard(1))
        )

        torch.manual_seed(0)
        inp_1 = torch.randint(
            vocab_size, [batch_size, seq_len], device=self.device_type
        )
        inp_0 = inp_1.t().contiguous()  # [seq_len, batch_size]
        # outputs of shape [batch_size, seq_len, dim]
        output_0 = emb_output_shard_0(inp_0).transpose(0, 1)
        output_1 = emb_output_shard_1(inp_1)
        self.assertEqual(output_0, output_1)

    @with_comms
    @skip_unless_torch_gpu
    @parametrize("is_input_sharded", [True, False])
    def test_e2e_seq_parallel_shard_embedding_consistency(self, is_input_sharded=False):
        class EmbeddingModule(nn.Module):
            def __init__(self, vocab_size, max_seq_len, dim):
                super().__init__()
                self.max_seq_len = max_seq_len
                self.tok_embeddings = nn.Embedding(vocab_size, dim)
                self.pos_embeddings = nn.Embedding(max_seq_len, dim)
                self.output = nn.Linear(dim, vocab_size, bias=False)

            def forward(self, tokens):
                _bsz, seq_len = tokens.size()
                assert seq_len <= self.max_seq_len
                h = self.tok_embeddings(tokens)
                pos = torch.arange(0, seq_len, device=tokens.device)
                # positional embeddings of shape (seq_len, dim)
                p = self.pos_embeddings(pos)
                output = self.output(h + p)
                return output

        vocab_size, max_seq_len, dim = 11, 25, 13
        # The assertions are commented out to demonstrate they are unnecessary.
        # assert dim % self.world_size == 0  # embeddings are sharded on dim
        # assert vocab_size % self.world_size == 0  # last layer is sharded on vocab_size
        torch.manual_seed(5)
        model = EmbeddingModule(vocab_size, max_seq_len, dim).to(self.device_type)
        model_sp = deepcopy(model)

        device_mesh = DeviceMesh(self.device_type, torch.arange(0, NUM_DEVICES))
        parallelize_plan = {
            "tok_embeddings": ColwiseParallel(
                input_layouts=Shard(0), output_layouts=Shard(1)
            )
            if is_input_sharded
            else ColwiseParallel(output_layouts=Shard(1)),
            "pos_embeddings": ColwiseParallel(output_layouts=Shard(0)),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(0) if is_input_sharded else Replicate(),
            ),
        }

        input_size = [7, 13]  # [batch_size, seq_len]
        # assert input_size[1] % self.world_size == 0  # output is sharded on seq_len
        rng_seed = self.rank if is_input_sharded else 0
        torch.manual_seed(rng_seed)
        inp = torch.randint(vocab_size, input_size, device=self.device_type)
        self.assertEqual(model(inp), model_sp(inp))


instantiate_parametrized_tests(DistShardEmbeddingTest)

if __name__ == "__main__":
    run_tests()
