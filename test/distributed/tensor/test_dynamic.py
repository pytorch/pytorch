# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from unittest.mock import patch

import torch
from torch.distributed.tensor import distribute_tensor, DTensor
from torch.distributed.tensor.placement_types import Replicate
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import requires_gpu


class TestDynamic(DTensorTestBase):
    @requires_gpu
    @with_comms
    @parametrize("fake_tensor_cache_enabled", [False, True])
    def test_embedding(self, fake_tensor_cache_enabled):
        with patch.object(
            torch._dynamo.config, "fake_tensor_cache_enabled", fake_tensor_cache_enabled
        ):
            device_mesh = self.build_device_mesh()

            placements = (Replicate(),)

            num_embeddings = 202048
            embedding_dim = 256
            weight = distribute_tensor(
                torch.rand(
                    [num_embeddings, embedding_dim],
                    dtype=torch.float32,
                    device=GPU_TYPE,
                    requires_grad=True,
                ),
                device_mesh,
                placements,  # [Replicate()],
            )

            def forward(input_batch_inputs_):
                to = weight.to(torch.float32)
                emb = torch.nn.functional.embedding(input_batch_inputs_, to)
                return emb

            arg0 = torch.randint(
                low=0, high=100, size=(2, 512), dtype=torch.int64, device=GPU_TYPE
            )
            arg0 = DTensor.from_local(arg0, device_mesh, placements)

            compiled_forward = torch.compile(forward, fullgraph=True, dynamic=True)
            _out = compiled_forward(arg0)


instantiate_parametrized_tests(TestDynamic)


if __name__ == "__main__":
    run_tests()
