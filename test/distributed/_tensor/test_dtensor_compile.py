# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import copy
import functools

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

    # This is a test where we do the conversions to-from DTensors *outside* of the compiled region,
    # and the compiled graph has DTensor inputs and outputs.
    def test_compile_dtensor_simple(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def fn(x, y):
            return torch.matmul(x, y)

        opt_fn = torch.compile(fn, backend=aot_eager_graph, fullgraph=True)
        dt = DTensor.from_local(torch.ones(2, 4), mesh, [Shard(0)], run_check=False)
        dt2 = DTensor.from_local(torch.ones(4, 2), mesh, [Shard(1)], run_check=False)

        ref = fn(dt, dt2)
        ref_local = ref.redistribute(mesh, [Replicate()]).to_local()

        res = opt_fn(dt, dt2)
        res_local = res.redistribute(mesh, [Replicate()]).to_local()

        self.assertEqual(res_local, ref_local)
        # The fw graph here is pretty messy (there are unnecessary cat and split calls?)
        # But the important ting to note is:
        # (1) we have a c10d_functional op in the graph, thanks to DTensor
        # (2) we have a wait() call in the graph that inductor can reorder
        # (3) We have a mm() in the graph (from the user code)
        self.assertExpectedInline(
            fw_graph_cell[0].code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    all_gather_into_tensor = torch.ops.c10d_functional.all_gather_into_tensor.default(arg1_1, 'ptd:0', [0, 1], 2);  arg1_1 = None
    wait_tensor = torch.ops.c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
    split = torch.ops.aten.split.Tensor(wait_tensor, 4);  wait_tensor = None
    getitem = split[0]
    getitem_1 = split[1];  split = None
    cat = torch.ops.aten.cat.default([getitem, getitem_1], 1);  getitem = getitem_1 = None
    mm = torch.ops.aten.mm.default(arg0_1, cat);  arg0_1 = cat = None
    return [mm]""",
        )

    # This is a test where we compile *everything*: local-to-dtensor conversion, compute, and to_local() conversion.
    def test_compile_dtensor_redistribute(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def fn(x, y):
            dt = DTensor.from_local(x.reshape(2, 4), mesh, [Shard(0)], run_check=False)
            dt2 = DTensor.from_local(y.reshape(4, 2), mesh, [Shard(1)], run_check=False)
            dt_out = torch.matmul(dt, dt2)
            dt_out_redistribute = dt_out.redistribute(mesh, [Replicate()])
            return dt_out.to_local()

        opt_fn = torch.compile(fn, backend=aot_eager_graph, fullgraph=True)

        x = torch.arange(8, requires_grad=True, dtype=torch.float32)
        y = torch.arange(8, requires_grad=True, dtype=torch.float32)
        ref = fn(x, y)

        res = opt_fn(x, y)

        self.assertEqual(res, ref)
        # The fw graph here is pretty messy (there are unnecessary cat and split calls?)
        # But the important thing to note is:
        # (1) we have a c10d_functional op in the graph, thanks to DTensor
        # (2) we have a wait() call in the graph that inductor can reorder
        # (3) We have a mm() in the graph (from the user code)
        self.assertExpectedInline(
            fw_graph_cell[0].code.strip(),
            """\
def forward(self, primals_1, primals_2):
    view = torch.ops.aten.view.default(primals_1, [2, 4]);  primals_1 = None
    _to_copy = torch.ops.aten._to_copy.default(view, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0));  view = None
    view_1 = torch.ops.aten.view.default(_to_copy, [2, 4]);  _to_copy = None
    view_2 = torch.ops.aten.view.default(primals_2, [4, 2]);  primals_2 = None
    _to_copy_1 = torch.ops.aten._to_copy.default(view_2, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0));  view_2 = None
    view_3 = torch.ops.aten.view.default(_to_copy_1, [4, 2]);  _to_copy_1 = None
    all_gather_into_tensor = torch.ops.c10d_functional.all_gather_into_tensor.default(view_3, 'ptd:0', [0, 1], 2)
    wait_tensor = torch.ops.c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
    split = torch.ops.aten.split.Tensor(wait_tensor, 4);  wait_tensor = None
    getitem = split[0]
    getitem_1 = split[1];  split = None
    cat = torch.ops.aten.cat.default([getitem, getitem_1], 1);  getitem = getitem_1 = None
    mm = torch.ops.aten.mm.default(view_1, cat);  cat = None
    view_4 = torch.ops.aten.view.default(mm, [2, 4]);  mm = None
    t_1 = torch.ops.aten.t.default(view_1);  view_1 = None
    t_3 = torch.ops.aten.t.default(view_3);  view_3 = None
    clone = torch.ops.aten.clone.default(t_3, memory_format = torch.contiguous_format);  t_3 = None
    return [view_4, t_1, clone]""",  # noqa: B950
        )

        # And assert the backward graph
        # Run the bw to populate the backward graph cells
        ref.sum().backward()
        res.sum().backward()

        self.assertExpectedInline(
            bw_graph_cell[0].code.strip(),
            """\
def forward(self, t_1, clone, tangents_1):
    mm_2 = torch.ops.aten.mm.default(t_1, tangents_1);  t_1 = None
    all_gather_into_tensor_2 = torch.ops.c10d_functional.all_gather_into_tensor.default(clone, 'ptd:0', [0, 1], 2);  clone = None
    wait_tensor_2 = torch.ops.c10d_functional.wait_tensor.default(all_gather_into_tensor_2);  all_gather_into_tensor_2 = None
    mm_4 = torch.ops.aten.mm.default(tangents_1, wait_tensor_2);  tangents_1 = wait_tensor_2 = None
    split_1 = torch.ops.aten.split.Tensor(mm_2, 2, 1);  mm_2 = None
    getitem_2 = split_1[0]
    getitem_3 = split_1[1];  split_1 = None
    cat_1 = torch.ops.aten.cat.default([getitem_2, getitem_3]);  getitem_2 = getitem_3 = None
    reduce_scatter_tensor = torch.ops.c10d_functional.reduce_scatter_tensor.default(cat_1, 'SUM', 'ptd:0', [0, 1], 2);  cat_1 = None
    wait_tensor_3 = torch.ops.c10d_functional.wait_tensor.default(reduce_scatter_tensor);  reduce_scatter_tensor = None
    view_5 = torch.ops.aten.view.default(wait_tensor_3, [4, 2]);  wait_tensor_3 = None
    _to_copy_2 = torch.ops.aten._to_copy.default(view_5, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'));  view_5 = None
    view_6 = torch.ops.aten.view.default(_to_copy_2, [8]);  _to_copy_2 = None
    view_7 = torch.ops.aten.view.default(mm_4, [2, 4]);  mm_4 = None
    _to_copy_3 = torch.ops.aten._to_copy.default(view_7, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'));  view_7 = None
    view_8 = torch.ops.aten.view.default(_to_copy_3, [8]);  _to_copy_3 = None
    return [view_8, view_6]""",  # noqa: B950
        )


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
        compiled_mod = torch.compile(model, backend="aot_eager", fullgraph=True)
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
        tp_model2 = parallelize_module(
            model_copy, twod_mesh, PairwiseParallel(), tp_mesh_dim=1
        )
        fsdp_2d = FSDP(
            tp_model2,
            process_group=fsdp_pg,
            device_id=self.rank,
            use_orig_params=True,
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


if __name__ == "__main__":
    run_tests()
