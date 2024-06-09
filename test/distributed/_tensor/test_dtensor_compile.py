# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import copy
import functools
import unittest
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.testing
import torch.distributed as dist
import torch.nn as nn
from torch._C import FileCheck
from torch._inductor.utils import run_and_get_triton_code
from torch.distributed._tensor import (
    DeviceMesh,
    DTensor,
    init_device_mesh,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
)
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
from torch.utils._triton import has_triton
from torch.utils.checkpoint import checkpoint


class SimpleModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mlp_0 = MLPModule(device)
        self.mlp_1 = MLPModule(device)

    def forward(self, input):
        return self.mlp_1(self.mlp_0(input))


def extract_graph(fx_g, _, graph_cell):
    graph_cell[0] = fx_g.code
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

    def test_placement_compile(self):
        def fn(x):
            a = 0
            if x.is_replicate():
                a += 1
            if x.is_shard():
                a += 2
                if x.dim < 0:
                    raise RuntimeError("dim < 0")
            if x.is_shard(0):
                a += 2
            if x.is_shard(dim=0):
                a += 2
            if x.is_shard(dim=None):
                a += 2
            if x.is_partial():
                a += 3
            return a

        compiled_fn = torch.compile(backend="aot_eager", fullgraph=True)(fn)

        for x in [Shard(0), Replicate(), Partial()]:
            opt_fn = fn(x)
            compiled_out = compiled_fn(x)
            self.assertEqual(opt_fn, compiled_out)

    def test_device_mesh_compile(self):
        def fn(x):
            # test size()
            a = x.size()
            b = x.size(0)
            c = x.size(mesh_dim=0)
            size = a + b + c
            # test get_coordinate()
            coord = x.get_coordinate()
            # test get_group()
            group = x.get_group()
            return size, coord, group

        compiled_fn = torch.compile(backend="aot_eager", fullgraph=True)(fn)

        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        opt_fn = fn(mesh)
        compiled_out = compiled_fn(mesh)
        self.assertEqual(opt_fn, compiled_out)

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

    def test_dtensor_attribute_access_on_intermediate(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def fn(x):
            tmp = x * 2
            if tmp.placements[0].is_shard():
                return tmp._local_tensor + 2
            else:
                return tmp._local_tensor + 3

        x = DTensor.from_local(torch.ones(4), mesh, [Shard(0)], run_check=False)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    def test_dtensor_constructor_w_graph_break(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        x = torch.randn(64, 32, requires_grad=True)
        spec = DTensorSpec(
            mesh,
            (Replicate(), Shard(0)),
            tensor_meta=TensorMeta(
                shape=torch.Size([128, 32]), stride=(32, 1), dtype=x.dtype
            ),
        )

        # test passing in DTensor as inputs/outputs and run some tensor computation
        def fn(x):
            print("graph break!")
            return DTensor(
                x,
                spec,
                requires_grad=x.requires_grad,
            )

        out = fn(x)
        out2 = torch.compile(fn, backend="eager")(x)

    def test_dtensor_constructor_w_dynamo_disable(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        x = torch.randn(32, requires_grad=True)
        spec = DTensorSpec(
            mesh,
            (Replicate(),),
            tensor_meta=TensorMeta(shape=torch.Size([32]), stride=(1,), dtype=x.dtype),
        )

        @torch._dynamo.disable(recursive=False)
        def fn(x):
            print("foo")
            return DTensor(
                x,
                spec,
                requires_grad=x.requires_grad,
            )

        out = fn(x)
        out2 = torch.compile(fn, backend="eager")(x)
        self.assertEqual(out, out2)

    def test_dtensor_noncontiguous_output(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # test passing in DTensor as inputs/outputs and run some tensor computation
        def fn(x, y, z):
            x_transposed = x.permute(0, 2, 1).contiguous()
            tmp = torch._C._nn.linear(x_transposed, y, z)
            return tmp.permute(0, 2, 1)

        x_inner = torch.randn(4, 16, 4, requires_grad=True)
        y_inner = torch.randn(4, 16, requires_grad=True)
        z_inner = torch.randn(4, requires_grad=True)
        x = DTensor.from_local(x_inner, mesh, [Shard(1)], run_check=False)
        y = DTensor.from_local(y_inner, mesh, [Shard(1)], run_check=False)
        z = DTensor.from_local(z_inner, mesh, [Replicate()], run_check=False)
        out = torch.compile(fn, backend="aot_eager", fullgraph=True)(x, y, z)
        out.contiguous().sum().backward()

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

        x = torch.ones(1, requires_grad=True)
        ref = fn(x)
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        res = opt_fn(x)
        # backward should work as well
        res.sum().backward()

        self.assertEqual(res, ref)
        self.assertEqual(cnt.frame_count, 1)

        # test if user calls from_local with mesh/placements as kwargs and that should still work
        def from_local_kwargs_fn(x):
            dt = DTensor.from_local(
                x, device_mesh=mesh, placements=[Replicate()], run_check=False
            )
            return dt.to_local() + 2

        ref = from_local_kwargs_fn(x)
        opt_kwargs_fn = torch.compile(from_local_kwargs_fn, backend=cnt, fullgraph=True)
        res = opt_kwargs_fn(x)
        self.assertEqual(res, ref)
        self.assertEqual(cnt.frame_count, 2)

    def test_dtensor_partial_placement_redistribute_unbalanced_correct_strides(self):
        # Partial -> Shard on an unbalanced tensor results in:
        # - A contiguous DTensor
        # - where the inner _local_tensor is noncontiguous
        placement = Shard(1)

        def fn(x):
            out = x.redistribute(mesh, [placement])
            return out

        # Temporarily ignore setUp(), and use rank3 graphs during tracing
        dist.destroy_process_group()
        fake_store = FakeStore()
        dist.init_process_group("fake", store=fake_store, rank=3, world_size=2)
        mesh = DeviceMesh(self.device_type, [1, 3])

        x = torch.randn(10, 257, 160, requires_grad=True)
        x_dt = DTensor.from_local(
            x,
            mesh,
            [Partial()],
            run_check=False,
            shape=(10, 257, 160),
            stride=(41120, 160, 1),
        )

        # tmp_dt has an inner, non-contiguous tensor, and is an autograd non-leaf
        tmp_dt = fn(x_dt)
        fake_mode = torch._subclasses.FakeTensorMode()
        tmp_dt_fake = fake_mode.from_tensor(tmp_dt)
        self.assertEqual(tmp_dt.shape, tmp_dt_fake.shape)
        self.assertEqual(tmp_dt.stride(), tmp_dt_fake.stride())
        self.assertEqual(tmp_dt._local_tensor.shape, tmp_dt_fake._local_tensor.shape)
        self.assertEqual(
            tmp_dt._local_tensor.stride(), tmp_dt_fake._local_tensor.stride()
        )

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_dtensor_contiguous_dtensor_noncontiguous_local_as_tangent(self):
        # Partial -> Shard on an unbalanced tensor results in:
        # - A contiguous DTensor
        # - where the inner _local_tensor is noncontiguous
        # When this tensor is a fwd graph output,
        # AOTAutograd needs to make sure we trace the backward
        # with a contiguous tangent
        placement = Shard(1)

        def fn(x):
            out = x.redistribute(mesh, [placement])
            return out

        # Temporarily ignore setUp(), and use rank3 graphs during tracing
        dist.destroy_process_group()
        fake_store = FakeStore()
        dist.init_process_group("fake", store=fake_store, rank=3, world_size=2)
        mesh = DeviceMesh(self.device_type, [1, 3])

        x = torch.randn(10, 257, 160, requires_grad=True)
        x_dt = DTensor.from_local(
            x,
            mesh,
            [Partial()],
            run_check=False,
            shape=(10, 257, 160),
            stride=(41120, 160, 1),
        )

        out_dt = torch.compile(fn)(x_dt)
        # If we don't properly contiguify our traced tangents,
        # this fails with an inductor stride assert
        out_dt.to_local().sum().backward()

    def test_dynamo_to_local_kwargs(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def fn(x):
            return dt.to_local(grad_placements=[Shard(0)]) + 2

        fn_opt = torch.compile(fn, backend="aot_eager", fullgraph=True)
        x = torch.ones(4)
        dt = DTensor.from_local(x, mesh, [Replicate()], run_check=False)

        out_ref = fn(dt)
        out_test = fn_opt(dt)
        self.assertEqual(out_ref, out_test)

    def test_dynamo_to_local_kwargs_forward_hook(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def fw_hook(module, inp, out):
            tmp = out.to_local(grad_placements=out.placements) + 2
            return DTensor.from_local(tmp, mesh, out.placements, run_check=False)

        mod = torch.nn.Linear(4, 4)
        mod.register_forward_hook(fw_hook)

        mod = torch.nn.Linear(4, 4)
        mod.register_forward_hook(fw_hook)
        mod.weight = torch.nn.Parameter(
            DTensor.from_local(mod.weight, mesh, [Replicate()], run_check=False)
        )
        mod.bias = torch.nn.Parameter(
            DTensor.from_local(mod.bias, mesh, [Replicate()], run_check=False)
        )
        opt_mod = torch.compile(mod, backend="aot_eager", fullgraph=True)

        x = torch.ones(4, 4)
        dt = DTensor.from_local(x, mesh, [Replicate()], run_check=False)

        out_ref = mod(dt)
        out_test = opt_mod(dt)
        self.assertEqual(out_ref, out_test)

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_dtensor_different_gradient_placement(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def fn(x, y, z):
            permute = x.permute(0, 2, 1)
            permute2 = permute.contiguous()
            layer_norm = torch.nn.functional.layer_norm(permute2, (4,), y, z, 1e-05)
            out = layer_norm.permute(0, 2, 1)
            return out

        x = torch.randn(4, 2, 4, requires_grad=True, device="cuda")
        x_dt = DTensor.from_local(x, mesh, [Shard(1)], run_check=False)

        y = torch.randn(4, requires_grad=True, device="cuda")
        y_dt = DTensor.from_local(y, mesh, [Replicate()], run_check=False)

        z = torch.randn(4, requires_grad=True, device="cuda")
        z_dt = DTensor.from_local(z, mesh, [Replicate()], run_check=False)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        tmp_dt = opt_fn(x_dt, y_dt, z_dt)
        out_dt = torch.matmul(tmp_dt, x_dt).permute(0, 2, 1)
        out_dt.sum().backward()

    def test_dynamo_dtensor_from_local_redistribute(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # pass in tensor as inputs/outputs, create DTensor and run redistribute
        # (allgather collective) inside the fn
        def fn(x):
            dt = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
            return dt.redistribute(mesh, [Replicate()]).to_local() + 2

        x = torch.ones(1)
        ref = fn(x)
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
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
            redistribute_kwargs_fn, backend=cnt, fullgraph=True
        )
        res = opt_kwargs_fn(x)
        self.assertEqual(res, ref)

    def test_dtensor_dynamo_device_mesh_attrs(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # pass in tensor as inputs/outputs, create DTensor and run redistribute
        # (allgather collective) inside the fn
        def fn(x_dt):
            if x_dt.device_mesh.device_type == "cuda":
                return x_dt + 1
            else:
                return x_dt + 2

        x = torch.ones(4, 4)
        x_dt = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
        ref = fn(x_dt)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x_dt)
        self.assertEqual(ref, res)

    def test_graph_input_is_async(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def fn(x):
            return x.sin().sin()

        opt_fn = torch.compile(fn, backend=aot_eager_graph, fullgraph=True)

        x = torch.randn(4, 4, requires_grad=True)
        x_dt = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
        x2 = x_dt.redistribute(mesh, [Replicate()], async_op=True)
        x2 = x2.to_local()
        out = opt_fn(x2)
        # The important part: we get a wait_tensor() in the graph.
        # At runtime, the input to the graph is an AsyncCollectiveTensor,
        # and inside the graph we need to issue a wait() to synchronize.
        self.assertExpectedInline(
            str(fw_graph_cell[0]).strip(),
            """\
def forward(self, primals_1):
    wait_tensor = torch.ops._c10d_functional.wait_tensor.default(primals_1)
    sin = torch.ops.aten.sin.default(wait_tensor)
    sin_1 = torch.ops.aten.sin.default(sin);  sin = None
    return [sin_1, primals_1, wait_tensor]""",
        )

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_dtensor_partial_placement_graph_output(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        def fn(x):
            return x + x

        x = torch.randn(4, 4, requires_grad=True)
        x_dt = DTensor.from_local(x, mesh, [Partial()], run_check=False)

        y = torch.randn(4, 4, requires_grad=True)
        y_dt = DTensor.from_local(y, mesh, [Replicate()], run_check=False)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        tmp_dt = opt_fn(x_dt)
        out_dt = torch.matmul(tmp_dt, y_dt)
        out_dt.sum().backward()

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(1)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", True)
    def test_tp_compile_comm_reordering(self):
        class FakeAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.wq = nn.Linear(16, 16)
                self.wk = nn.Linear(16, 16)
                self.wv = nn.Linear(16, 16)
                self.wo = nn.Linear(16, 16)

            def forward(self, x):
                xq = self.wq(x)
                xk = self.wk(x)
                xv = self.wv(x)
                # fake attention:
                xo = xq + xk + xv
                return self.wo(xo)

        class FakeTransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = FakeAttention()

            def forward(self, x):
                return self.attn(x)

        class FakeTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = FakeTransformerBlock()

            def forward(self, input):
                return self.block(input)

        model = FakeTransformer().to(self.device_type)

        tp_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("tp",))

        # apply sequence parallel
        parallel_plan = {
            "attn": PrepareModuleInput(
                input_layouts=Shard(0), desired_input_layouts=Replicate()
            ),
            "attn.wq": ColwiseParallel(),
            "attn.wk": ColwiseParallel(),
            "attn.wv": ColwiseParallel(),
            "attn.wo": RowwiseParallel(output_layouts=Shard(0)),
        }

        parallelize_module(
            module=model.block,
            device_mesh=tp_mesh,
            parallelize_plan=parallel_plan,
        )

        cnt = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        compiled_model = torch.compile(model, backend=cnt, fullgraph=True)
        inp = torch.rand(20, 16).to(self.device_type)
        out = compiled_model(inp)
        out.sum().backward()
        self.assertEqual(cnt.frame_count, 1)

        code = run_and_get_triton_code(compiled_model, inp)
        FileCheck().check(
            "buf0 = torch.ops._c10d_functional.all_gather_into_tensor.default(primal"
        ).check("buf1 = torch.ops._c10d_functional.wait_tensor.default(buf0").check(
            "extern_kernels.mm(buf0,"
        ).run(
            code
        )


@instantiate_parametrized_tests
class TestDTensorCompileE2E(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    @parametrize("is_seq_parallel", [True, False])
    def test_tp_compile_fullgraph(self, is_seq_parallel):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        model = SimpleModel(self.device_type)

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

        if is_seq_parallel:
            # use input preparation to test out the compile of it
            prepare_module_input = PrepareModuleInput(
                input_layouts=Shard(0),
                desired_input_layouts=Replicate(),
            )
            prepare_module_out = PrepareModuleOutput(
                output_layouts=Replicate(),
                desired_output_layouts=Shard(0),
            )
            plan = {
                "mlp_0": prepare_module_input,
                "mlp_0.net1": ColwiseParallel(),
                "mlp_0.net2": rowwise_style,
                "mlp_1.net1": colwise_style,
                "mlp_1.net2": RowwiseParallel(),
                "mlp_1": prepare_module_out,
            }
        else:
            plan = {
                "mlp_0.net1": colwise_style,
                "mlp_0.net2": rowwise_style,
                "mlp_1.net1": colwise_style,
                "mlp_1.net2": rowwise_style,
            }

        model = parallelize_module(
            model,
            mesh,
            parallelize_plan=plan,
        )
        rng_seed = self.rank if is_seq_parallel else 0
        torch.manual_seed(rng_seed)
        inp = torch.rand(20, 10, device=self.device_type)
        out = model(inp)
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        compiled_mod = torch.compile(model, backend=cnt, fullgraph=True)
        compiled_out = compiled_mod(inp)
        compiled_out.sum().backward()
        self.assertEqual(compiled_out, out)
        self.assertEqual(cnt.frame_count, 1)

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

        fsdp_pg = twod_mesh.get_group(mesh_dim=0)

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
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        compiled_2d = torch.compile(fsdp_2d, backend=cnt)
        compiled_output = compiled_2d(inp)

        self.assertEqual(out, compiled_output)
        self.assertEqual(cnt.frame_count, 1)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_fsdp_tp_ac_compile(self):
        dp_degree = 2
        tp_degree = self.world_size // dp_degree
        model = SimpleModel(self.device_type)
        model_copy = copy.deepcopy(model)

        # 2-D mesh is [dp, tp]
        mesh_2d = init_device_mesh(
            "cuda", mesh_shape=(dp_degree, tp_degree), mesh_dim_names=("dp", "tp")
        )

        inp = torch.rand(20, 10, device=self.device_type)
        parallelize_plan = {
            "mlp_0.net1": ColwiseParallel(),
            "mlp_0.net2": RowwiseParallel(),
            "mlp_1.net1": ColwiseParallel(),
            "mlp_1.net2": RowwiseParallel(),
        }
        tp_model = parallelize_module(model, mesh_2d["tp"], parallelize_plan)
        tp_model = checkpoint_wrapper(
            tp_model,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            checkpoint_fn=checkpoint,
            use_reentrant=False,
        )
        eager_2d = FSDP(tp_model, device_mesh=mesh_2d["dp"], use_orig_params=True)

        tp_model2 = parallelize_module(model_copy, mesh_2d["tp"], parallelize_plan)
        fsdp_2d = FSDP(
            tp_model2,
            device_mesh=mesh_2d["dp"],
            use_orig_params=True,
        )
        # TODO: once aot autograd support is ready we can just use default backend
        compiled_2d = torch.compile(fsdp_2d, backend="aot_eager")

        # forward pass
        out = eager_2d(inp)
        compiled_output = compiled_2d(inp)
        self.assertEqual(out, compiled_output)

        # backward pass
        out.sum().backward()
        compiled_output.sum().backward()

        # compare the gradients:
        for n, p in zip(fsdp_2d.parameters(), compiled_2d.parameters()):
            self.assertEqual(n.grad, p.grad)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_compile_dtensor_redistribute_backward(self):
        mesh = DeviceMesh(device_type="cuda", mesh=torch.arange(self.world_size))

        def fn(x, y):
            dt = DTensor.from_local(x.reshape(2, 4), mesh, [Shard(0)], run_check=False)
            dt2 = DTensor.from_local(y.reshape(4, 2), mesh, [Shard(1)], run_check=False)
            dt_out = torch.matmul(dt, dt2)
            dt_out_redistribute = dt_out.redistribute(mesh, [Replicate()])
            return dt_out_redistribute.to_local()

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
