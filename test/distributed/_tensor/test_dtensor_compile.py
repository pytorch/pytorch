# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import copy
import functools

import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import (
    DeviceMesh,
    DTensor,
    init_device_mesh,
    Replicate,
    Shard,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
)
from torch.distributed.tensor.parallel.fsdp import enable_2d_with_fsdp
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
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


def extract_graph(fx_g, _, graph_cell):
    graph_cell[0] = fx_g
    return fx_g


# Make a custom compiler that runs aot autograd but extracts the fw graph
fw_graph_cell = [None]
bw_graph_cell = [None]
fw_compiler = functools.partial(extract_graph, graph_cell=fw_graph_cell)
bw_compiler = functools.partial(extract_graph, graph_cell=bw_graph_cell)

from functorch.compile import min_cut_rematerialization_partition
from torch._dynamo.backends.common import aot_autograd

aot_eager_graph = aot_autograd(
    fw_compiler=fw_compiler,
    bw_compiler=bw_compiler,
    partition_fn=min_cut_rematerialization_partition,
)


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

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    def test_dynamo_dtensor(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # test passing in DTensor as inputs/outputs and run some tensor computation
        def fn(x):
            return x * x + 2

        x = DTensor.from_local(torch.rand(1), mesh, [Shard(0)], run_check=False)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
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
        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

        # test if user calls from_local with mesh/placements as kwargs and that should still work
        def from_local_kwargs_fn(x):
            dt = DTensor.from_local(
                x, device_mesh=mesh, placements=[Replicate()], run_check=False
            )
            return dt.to_local() + 2

        ref = from_local_kwargs_fn(x)
        opt_kwargs_fn = torch.compile(
            from_local_kwargs_fn, backend="aot_eager", fullgraph=True
        )
        res = opt_kwargs_fn(x)
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
        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

        def redistribute_kwargs_fn(x):
            dt = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
            return (
                dt.redistribute(device_mesh=mesh, placements=[Replicate()]).to_local()
                + 2
            )

        x = torch.ones(1)
        ref = redistribute_kwargs_fn(x)
        opt_kwargs_fn = torch.compile(
            redistribute_kwargs_fn, backend="aot_eager", fullgraph=True
        )
        res = opt_kwargs_fn(x)
        self.assertEqual(res, ref)


class TestDTensorCompileE2E(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    @parametrize("is_seq_parallel", [True, False])
    def test_tp_compile_fullgraph(self, is_seq_parallel):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        model = SimpleModel(self.device_type)
        module_prepare_input = (
            PrepareModuleInput()
            if is_seq_parallel
            else PrepareModuleInput(input_layouts=Replicate())
        )
        no_input_prepare_colwise_style = ColwiseParallel(input_layouts=None)
        colwise_style = (
            ColwiseParallel(input_layouts=Shard(0))
            if is_seq_parallel
            else ColwiseParallel()
        )
        rowwise_style = (
            RowwiseParallel(output_layouts=Shard(0))
            if is_seq_parallel
            else RowwiseParallel()
        )
        model = parallelize_module(
            model,
            mesh,
            parallelize_plan={
                "mlp_0": module_prepare_input,
                "mlp_0.net1": no_input_prepare_colwise_style,
                "mlp_0.net2": rowwise_style,
                "mlp_1.net1": colwise_style,
                "mlp_1.net2": rowwise_style,
            },
        )
        rng_seed = self.rank if is_seq_parallel else 0
        torch.manual_seed(rng_seed)
        inp = torch.rand(20, 10, device=self.device_type)
        out = model(inp)
        compiled_mod = torch.compile(model, backend="aot_eager", fullgraph=True)
        compiled_out = compiled_mod(inp)
        self.assertEqual(compiled_out, out)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_fsdp_tp_compile(self):
        data_parallel_size = 2
        model = SimpleModel(self.device_type)
        model_copy = copy.deepcopy(model)

        # 2-D mesh is [dp, tp]
        twod_mesh = init_device_mesh(
            "cuda",
            (data_parallel_size, self.world_size // data_parallel_size),
            mesh_dim_names=["dp", "tp"],
        )

        fsdp_pg = twod_mesh.get_dim_groups()[0]

        inp = torch.rand(20, 10, device=self.device_type)
        parallelize_plan = {
            "mlp_0.net1": ColwiseParallel(),
            "mlp_0.net2": RowwiseParallel(),
            "mlp_1.net1": ColwiseParallel(),
            "mlp_1.net2": RowwiseParallel(),
        }
        tp_model = parallelize_module(model, twod_mesh["tp"], parallelize_plan)
        eager_2d = FSDP(
            tp_model,
            device_id=self.rank,
            use_orig_params=True,
            device_mesh=twod_mesh["dp"],
        )
        out = eager_2d(inp)
        tp_model2 = parallelize_module(
            model_copy,
            twod_mesh["tp"],
            parallelize_plan,
        )
        fsdp_2d = FSDP(
            tp_model2,
            device_id=self.rank,
            use_orig_params=True,
            device_mesh=twod_mesh["dp"],
        )

        # TODO: once aot autograd support is ready we can just use default backend
        compiled_2d = torch.compile(fsdp_2d, backend="aot_eager")
        compiled_output = compiled_2d(inp)

        self.assertEqual(out, compiled_output)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_compile_dtensor_redistribute_backward(self):
        mesh = DeviceMesh(device_type="cuda", mesh=torch.arange(self.world_size))
        #            device_type="cuda",
        #            mesh=torch.arange(0, self.world_size).view(data_parallel_size, -1),

        def fn(x, y):
            dt = DTensor.from_local(x.reshape(2, 4), mesh, [Shard(0)], run_check=False)
            dt2 = DTensor.from_local(y.reshape(4, 2), mesh, [Shard(1)], run_check=False)
            dt_out = torch.matmul(dt, dt2)
            dt_out_redistribute = dt_out.redistribute(mesh, [Replicate()])
            return dt_out.to_local()

        opt_fn = torch.compile(fn, backend=aot_eager_graph, fullgraph=True)

        x_ref = torch.arange(8, requires_grad=True, dtype=torch.float32)
        y_ref = torch.arange(8, requires_grad=True, dtype=torch.float32)
        ref = fn(x_ref, y_ref)

        x = torch.arange(8, requires_grad=True, dtype=torch.float32)
        y = torch.arange(8, requires_grad=True, dtype=torch.float32)
        res = opt_fn(x, y)

        self.assertEqual(res, ref)

        # Now run and assert the backward + gradients
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(x_ref.grad, x.grad)
        self.assertEqual(y_ref.grad, y.grad)


instantiate_parametrized_tests(TestDTensorCompileE2E)

if __name__ == "__main__":
    run_tests()
