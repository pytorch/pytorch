# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
import sys
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, with_comms
from torch.testing._internal.common_utils import IS_FBCODE, run_tests, skipIfHpu, parametrize, instantiate_parametrized_tests
from unittest.mock import patch


class TestDynamic(DTensorTestBase):
    @with_comms
    # FIXME: Currently broken for fake tensor cache
    @parametrize("fake_tensor_cache_enabled", [False])
    def test_embedding(self, fake_tensor_cache_enabled):
        with patch.object(torch._dynamo.config, "fake_tensor_cache_enabled", fake_tensor_cache_enabled):
            device_mesh = self.build_device_mesh()

            placements = (Replicate(),)

            num_embeddings = 202048
            embedding_dim = 256
            weight = distribute_tensor(
                torch.rand(
                    [num_embeddings, embedding_dim],
                    dtype=torch.float32,
                    device="cuda",
                    requires_grad=True,
                ),
                device_mesh,
                placements, # [Replicate()],
            )

            def forward(input_batch_inputs_):
                to = weight.to(torch.float32)
                emb = torch.nn.functional.embedding(input_batch_inputs_, to)
                return emb

            arg0 = torch.randint(low=0, high=100, size=(2, 512), dtype=torch.int64, device="cuda")
            arg0 = DTensor.from_local(arg0, device_mesh, placements)

            compiled_forward = torch.compile(forward, fullgraph=True, dynamic=True)
            out = compiled_forward(arg0)


instantiate_parametrized_tests(TestDynamic)


if __name__ == "__main__":
    run_tests()
