# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import copy

import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.parallel import PairwiseParallel, parallelize_module
from torch.distributed.tensor.parallel.fsdp import enable_2d_with_fsdp
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    with_comms,
)
from torch.testing._internal.distributed.fake_pg import FakeStore


class SimpleModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mlp_0 = MLPModule(device)
        self.mlp_1 = MLPModule(device)

    def forward(self, input):
        return self.mlp_1(self.mlp_0(input))


class TestDTensorCompile(torch._dynamo.test_case.TestCase):
    def setUp(self):
        super().setUp()
        fake_store = FakeStore()
        dist.init_process_group(
            "fake", store=fake_store, rank=0, world_size=self.world_size
        )

    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()

    @property
    def device_type(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def world_size(self) -> int:
        return 2

    def test_fakify_dtensor(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # pass in DTensor as inputs/outputs to the function
        def fn(x):
            return x

        x = DTensor.from_local(torch.rand(1), mesh, [Shard(0)], run_check=False)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    def test_dynamo_dtensor(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # test passing in DTensor as inputs/outputs and run some tensor computation
        def fn(x):
            return x * x + 2

        x = DTensor.from_local(torch.rand(1), mesh, [Shard(0)], run_check=False)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    def test_dynamo_dtensor_from_local(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # create DTensor inside fn and run some compute
        def fn(x):
            dt = DTensor.from_local(x, mesh, [Replicate()], run_check=False)
            return dt.to_local() + 2

        # below is the op approach for reference
        # from torch.distributed._tensor.api import _FromTorchTensor
        # def from_local_tensor(x):
        #     return _FromTorchTensor.apply(x, mesh, [Replicate()], False)

        # _dt_lib_def = torch.library.Library("dtensor", "DEF")
        # _dt_lib_def.define("from_local(Tensor self) -> Tensor")

        # _dt_lib_impl = torch.library.Library("dtensor", "IMPL")
        # _dt_lib_impl.impl("from_local", from_local_tensor, "Autograd")

        x = torch.ones(1)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    def test_dynamo_dtensor_from_local_redistribute(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # pass in tensor as inputs/outputs, create DTensor and run redistribute
        # (allgather collective) inside the fn
        def fn(x):
            dt = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
            return dt.redistribute(mesh, [Replicate()]).to_local() + 2

        x = torch.ones(1)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)


class TestDTensorCompileE2E(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_tp_compile_fullgraph(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        model = MLPModule(self.device_type)
        model = parallelize_module(model, mesh, PairwiseParallel())
        inp = torch.rand(20, 10, device=self.device_type)
        out = model(inp)
        compiled_mod = torch.compile(model, backend="eager", fullgraph=True)
        compiled_out = compiled_mod(inp)
        self.assertEqual(compiled_out, out)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_fsdp_tp_compile(self):
        data_parallel_size = 2
        model = SimpleModel(self.device_type)
        model_copy = copy.deepcopy(model)
        enable_2d_with_fsdp()

        # 2-D mesh is [dp, tp]
        twod_mesh = DeviceMesh(
            device_type="cuda",
            mesh=torch.arange(0, self.world_size).view(data_parallel_size, -1),
        )

        fsdp_pg = twod_mesh.get_dim_groups()[0]

        inp = torch.rand(20, 10, device=self.device_type)
        tp_model = parallelize_module(
            model, twod_mesh, PairwiseParallel(), tp_mesh_dim=1
        )
        eager_2d = FSDP(
            tp_model, process_group=fsdp_pg, device_id=self.rank, use_orig_params=True
        )
        out = eager_2d(inp)
        # TODO: once aot autograd support is ready we can just use default backend
        tp_model2 = parallelize_module(
            model_copy, twod_mesh, PairwiseParallel(), tp_mesh_dim=1
        )
        compiled_tp = torch.compile(tp_model2, backend="eager", fullgraph=True)

        # TODO: now we first apply torch compile on tp model then use fsdp to wrap it, ideally
        # we should apply torch.compile after fsdp wrap, but the current graph break approach
        # have some issues with the tensor subclass compilation, need to dig into this later
        compiled_2d = FSDP(
            compiled_tp,
            process_group=fsdp_pg,
            device_id=self.rank,
            use_orig_params=True,
        )

        compiled_output = compiled_2d(inp)

        self.assertEqual(out, compiled_output)


if __name__ == "__main__":
    run_tests()
