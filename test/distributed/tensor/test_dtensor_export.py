# Owner(s): ["oncall: distributed"]

import contextlib
import unittest

import torch
import torch.distributed as dist
from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
from torch._functorch.aot_autograd import aot_export_joint_with_descriptors
from torch._functorch.partitioners import min_cut_rematerialization_partition
from torch._guards import detect_fake_mode, TracingContext, tracing
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda,
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed._tensor.common_dtensor import MLPModule
from torch.testing._internal.distributed.fake_pg import FakeStore


class SimpleModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mlp_0 = MLPModule(device)
        self.mlp_1 = MLPModule(device)

    def forward(self, input):
        return self.mlp_1(self.mlp_0(input))

class SimpleModelDynamicShapes(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mlp_0 = MLPModule(device)
        self.mlp_1 = MLPModule(device)

    def forward(self, input):
        if input.shape[0] > 4:
            return self.mlp_0(input.sin())
        return self.mlp_1(input.cos())


def strict_export_and_aot_export_joint_with_descriptors(model, inputs):
    # needed for stric export
    torch.utils._pytree.register_constant(DTensorSpec)

    # install_free_tensors is required for dynamo to work
    with torch._dynamo.config.patch(
        install_free_tensors=True, inline_inbuilt_nn_modules=True
    ):
        with torch._export.utils._disable_aten_to_metadata_assertions():
            ep = torch.export.export(model, (inputs,), strict=True)

    # joint_gm produced here is missing the backward region, due to incompatiblility
    # between ep.module() and aot_export_joint_with_descriptors.
    # Keeping this here to show the issue.
    return aot_export_joint_with_descriptors_alone(ep.module(), inputs)


def graph_capture_and_aot_export_joint_with_descriptors(model, inputs):
    with torch._dynamo.config.patch(install_free_tensors=True):
        # TODO: switch to use the official graph_capture API once it is ready
        gm = _dynamo_graph_capture_for_export(model)(inputs)
    
    # TODO we actually need to get this from graph capture output, but this is fine.
    fake_vals: list[torch.Tensor] = [] 
    for node in gm.graph.nodes:
        if (
            node.op == "placeholder" and 
            "val" in node.meta and 
            isinstance(node.meta["val"], torch.Tensor)
        ):
            fake_vals.append(node.meta["val"])
    
    # TODO need to attach export shape guards util here. 
    fake_mode = detect_fake_mode(fake_vals)
    with tracing(TracingContext(fake_mode)):
        return aot_export_joint_with_descriptors_alone(gm, inputs)


def aot_export_joint_with_descriptors_alone(model, inputs):
    with contextlib.ExitStack() as stack:
        joint_with_descriptors = aot_export_joint_with_descriptors(
            stack,
            model,
            (inputs,),
        )
        return joint_with_descriptors.graph_module


def _count_op(gm, target):
    return sum(1 for node in gm.graph.nodes if node.target == target)


@requires_cuda
class DTensorExportTest(TestCase):
    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()

    def setUp(self):
        super().setUp()
        self.world_size = 8
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=self.world_size, store=store
        )
        self.device_type = "cuda"

    def _run_test(self, export_fn):
        dp_degree = 2
        tp_degree = self.world_size // dp_degree

        # 2-D mesh is [dp, tp]
        mesh_2d = init_device_mesh(
            self.device_type,
            mesh_shape=(dp_degree, tp_degree),
            mesh_dim_names=["dp", "tp"],
        )

        model = SimpleModel(self.device_type)
        parallelize_plan = {
            "mlp_0.net1": ColwiseParallel(),
            "mlp_0.net2": RowwiseParallel(),
            "mlp_1.net1": ColwiseParallel(),
            "mlp_1.net2": RowwiseParallel(),
        }
        tp_model = parallelize_module(model, mesh_2d["tp"], parallelize_plan)

        inputs = torch.rand(20, 10, device=self.device_type)
        inputs = distribute_tensor(inputs, mesh_2d["tp"], placements=[Replicate()])

        joint_gm = export_fn(tp_model, inputs)
        fw_gm, bw_gm = min_cut_rematerialization_partition(
            joint_gm, None, num_fwd_outputs=1
        )

        self.assertTrue(
            _count_op(joint_gm, torch.ops._c10d_functional.all_reduce.default),
            3,
        )
        self.assertTrue(
            _count_op(fw_gm, torch.ops._c10d_functional.all_reduce.default),
            2,
        )
        self.assertTrue(
            _count_op(bw_gm, torch.ops._c10d_functional.all_reduce.default),
            1,
        )

    @parametrize(
        "export_fn",
        [
            graph_capture_and_aot_export_joint_with_descriptors,
            aot_export_joint_with_descriptors_alone,
        ],
    )
    def test_export_parallelize_module_with_dtensor_input(
        self,
        export_fn,
    ):
        self._run_test(export_fn)

    # aot_export_joint_with_descriptors on strict-exported exported_program.module()
    # is producing a joint graph with backward region missing
    @unittest.expectedFailure
    def test_strict_export_parallelize_module_with_dtensor_input(self):
        self._run_test(strict_export_and_aot_export_joint_with_descriptors)
    
    def test_dynamic_shapes(self):
        dp_degree = 2
        tp_degree = self.world_size // dp_degree

        # 2-D mesh is [dp, tp]
        mesh_2d = init_device_mesh(
            self.device_type,
            mesh_shape=(dp_degree, tp_degree),
            mesh_dim_names=["dp", "tp"],
        )

        model = SimpleModelDynamicShapes(self.device_type)
        parallelize_plan = {
            "mlp_0.net1": ColwiseParallel(),
            "mlp_0.net2": RowwiseParallel(),
            "mlp_1.net1": ColwiseParallel(),
            "mlp_1.net2": RowwiseParallel(),
        }
        tp_model = parallelize_module(model, mesh_2d["tp"], parallelize_plan)

        inputs = torch.rand(20, 10, device=self.device_type)
        inputs = distribute_tensor(inputs, mesh_2d["tp"], placements=[Replicate()])
        torch._dynamo.mark_dynamic(inputs, 0, min=5, max=100)

        joint_gm = graph_capture_and_aot_export_joint_with_descriptors(tp_model, inputs)
        # TODO (better test would be here to actually check guards)
        self.assertExpectedInline(
            joint_gm.print_readable().strip(),
            """\
class inner_f(torch.nn.Module):
    def forward(self, primals, tangents):
        primals_1: "f32[4, 10]"; primals_2: "f32[4]"; primals_3: "f32[10, 4]"; primals_4: "f32[10]"; primals_5: "f32[s22, 10]"; primals_6: "Sym(s6)"; tangents_1: "f32[s22, 10]"; 
    
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
         # File: /data/users/tmanlaibaatar/pytorch/test/distributed/tensor/test_dtensor_export.py:48 in forward, code: return self.mlp_0(input.sin())
        sin: "f32[s22, 10]" = torch.ops.aten.sin.default(primals_5)
        sym_size_int: "Sym(s22)" = torch.ops.aten.sym_size.int(primals_5, 0);  primals_5 = None
        
         # File: /data/users/tmanlaibaatar/pytorch/torch/testing/_internal/distributed/_tensor/common_dtensor.py:93 in forward, code: return self.net2(self.relu(self.net1(x)))
        t: "f32[10, 4]" = torch.ops.aten.t.default(primals_1);  primals_1 = None
        addmm: "f32[s22, 4]" = torch.ops.aten.addmm.default(primals_2, sin, t);  primals_2 = t = None
        
         # File: /data/users/tmanlaibaatar/pytorch/torch/distributed/tensor/parallel/style.py:146 in _prepare_output_fn, code: return outputs.to_local() if use_local_output else outputs
        view: "f32[s22, 4]" = torch.ops.aten.view.default(addmm, [sym_size_int, 4]);  addmm = None
        
         # File: /data/users/tmanlaibaatar/pytorch/torch/testing/_internal/distributed/_tensor/common_dtensor.py:93 in forward, code: return self.net2(self.relu(self.net1(x)))
        relu: "f32[s22, 4]" = torch.ops.aten.relu.default(view);  view = None
        detach: "f32[s22, 4]" = torch.ops.aten.detach.default(relu)
        detach_1: "f32[s22, 4]" = torch.ops.aten.detach.default(detach);  detach = None
        
         # File: /data/users/tmanlaibaatar/pytorch/torch/distributed/tensor/parallel/style.py:230 in _prepare_input_fn, code: input_tensor = DTensor.from_local(
        view_1: "f32[s22, 4]" = torch.ops.aten.view.default(relu, [sym_size_int, 4]);  relu = None
        
         # File: /data/users/tmanlaibaatar/pytorch/torch/testing/_internal/distributed/_tensor/common_dtensor.py:93 in forward, code: return self.net2(self.relu(self.net1(x)))
        t_1: "f32[4, 10]" = torch.ops.aten.t.default(primals_3);  primals_3 = None
        div: "f32[10]" = torch.ops.aten.div.Tensor(primals_4, 4);  primals_4 = None
        addmm_1: "f32[s22, 10]" = torch.ops.aten.addmm.default(div, view_1, t_1);  div = None
        
         # File: /data/users/tmanlaibaatar/pytorch/torch/distributed/tensor/parallel/style.py:285 in _prepare_output_fn, code: outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        all_reduce: "f32[s22, 10]" = torch.ops._c10d_functional.all_reduce.default(addmm_1, 'sum', '5');  addmm_1 = None
        wait_tensor: "f32[s22, 10]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce);  all_reduce = None
        
         # File: /data/users/tmanlaibaatar/pytorch/torch/distributed/tensor/parallel/style.py:287 in _prepare_output_fn, code: return outputs.to_local() if use_local_output else outputs
        view_2: "f32[s22, 10]" = torch.ops.aten.view.default(wait_tensor, [sym_size_int, 10]);  wait_tensor = sym_size_int = None
        
         # File: /data/users/tmanlaibaatar/pytorch/torch/testing/_internal/distributed/_tensor/common_dtensor.py:93 in forward, code: return self.net2(self.relu(self.net1(x)))
        t_2: "f32[10, 4]" = torch.ops.aten.t.default(t_1);  t_1 = None
        mm: "f32[s22, 4]" = torch.ops.aten.mm.default(tangents_1, t_2);  t_2 = None
        t_3: "f32[10, s22]" = torch.ops.aten.t.default(tangents_1)
        mm_1: "f32[10, 4]" = torch.ops.aten.mm.default(t_3, view_1);  t_3 = view_1 = None
        t_4: "f32[4, 10]" = torch.ops.aten.t.default(mm_1);  mm_1 = None
        sum_1: "f32[1, 10]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_3: "f32[10]" = torch.ops.aten.view.default(sum_1, [10]);  sum_1 = None
        t_5: "f32[10, 4]" = torch.ops.aten.t.default(t_4);  t_4 = None
        detach_2: "f32[s22, 4]" = torch.ops.aten.detach.default(detach_1);  detach_1 = None
        detach_3: "f32[s22, 4]" = torch.ops.aten.detach.default(detach_2);  detach_2 = None
        threshold_backward: "f32[s22, 4]" = torch.ops.aten.threshold_backward.default(mm, detach_3, 0);  mm = detach_3 = None
        t_6: "f32[4, s22]" = torch.ops.aten.t.default(threshold_backward)
        mm_2: "f32[4, 10]" = torch.ops.aten.mm.default(t_6, sin);  t_6 = sin = None
        t_7: "f32[10, 4]" = torch.ops.aten.t.default(mm_2);  mm_2 = None
        sum_2: "f32[1, 4]" = torch.ops.aten.sum.dim_IntList(threshold_backward, [0], True);  threshold_backward = None
        view_4: "f32[4]" = torch.ops.aten.view.default(sum_2, [4]);  sum_2 = None
        t_8: "f32[4, 10]" = torch.ops.aten.t.default(t_7);  t_7 = None
        return pytree.tree_unflatten([view_2, t_8, view_4, t_5, view_3, None], self._out_spec)"""
        )


instantiate_parametrized_tests(DTensorExportTest)


if __name__ == "__main__":
    run_tests()
