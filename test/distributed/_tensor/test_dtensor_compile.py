# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
import torch._dynamo
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh
from torch.distributed.tensor.parallel import PairwiseParallel, parallelize_module
from torch.distributed.tensor.parallel.fsdp import enable_2d_with_fsdp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class ToyMLP(nn.Module):
    def __init__(self):
        super(ToyMLP, self).__init__()
        self.net1 = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(32, 10)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_0 = ToyMLP()
        self.mlp_1 = ToyMLP()
    def forward(self, input):
        return self.mlp_1(self.mlp_0(input))


class TestDTensorCompile(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_2d_fsdp_tp_compile(self):
        data_parallel_size = 2
        model = SimpleModel().cuda(self.rank)
        import copy
        model_copy = copy.deepcopy(model)
        enable_2d_with_fsdp()

        # 2-D mesh is [dp, tp]
        twod_mesh = DeviceMesh(
            device_type="cuda",
            mesh=torch.arange(0, self.world_size).view(data_parallel_size, -1),
        )

        fsdp_pg = twod_mesh.get_dim_groups()[0]

        inp = torch.rand(20, 10).cuda(self.rank)
        tp_model = parallelize_module(model, twod_mesh, PairwiseParallel(), tp_mesh_dim=1)
        eager_2d = FSDP(tp_model, process_group=fsdp_pg, device_id=self.rank, use_orig_params=True)
        out = eager_2d(inp)
        # TODO: once aot autograd support is ready we can just use default backend
        tp_model2 = parallelize_module(model_copy, twod_mesh, PairwiseParallel(), tp_mesh_dim=1)
        compiled_tp = torch.compile(tp_model2, backend="eager", fullgraph=True)

        # TODO: now we first apply torch compile on tp model then use fsdp to wrap it, ideally
        # we should apply torch.compile after fsdp wrap, but the current graph break approach
        # have some issues with the tensor subclass compilation, need to dig into this later
        compiled_2d = FSDP(compiled_tp, process_group=fsdp_pg, device_id=self.rank, use_orig_params=True)

        compiled_output = compiled_2d(inp)

        self.assertEqual(out, compiled_output)


if __name__ == "__main__":
    run_tests()
