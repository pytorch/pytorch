# Owner(s): ["oncall: distributed"]

import contextlib
import unittest

import torch
import torch.distributed as dist
import torch.fx.traceback as fx_traceback
from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
from torch._functorch.aot_autograd import aot_export_joint_with_descriptors
from torch._functorch.partitioners import min_cut_rematerialization_partition
from torch._guards import tracing, TracingContext
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


class SimpleModelAnnotated(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mlp_0 = MLPModule(device)
        self.mlp_1 = MLPModule(device)

    def forward(self, input):
        with fx_traceback.annotate({"pp_stage": 0}):
            x = self.mlp_0(input)
        return self.mlp_1(x)


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
        fake_mode = gm.meta.get("fake_mode", None)
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

    def _run_test(self, export_fn, test_annotation=False):
        dp_degree = 2
        tp_degree = self.world_size // dp_degree

        # 2-D mesh is [dp, tp]
        mesh_2d = init_device_mesh(
            self.device_type,
            mesh_shape=(dp_degree, tp_degree),
            mesh_dim_names=["dp", "tp"],
        )

        model = None
        if test_annotation:
            model = SimpleModelAnnotated(self.device_type)
        else:
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

        if test_annotation:

            def has_tag(node):
                return "custom" in node.meta and node.meta["custom"] == {"pp_stage": 0}

            def marked_nodes(gm):
                return [
                    node.name
                    for node in gm.graph.nodes
                    if has_tag(node) and node.op == "call_function"
                ]

            def unmarked_nodes(gm):
                return [
                    node.name
                    for node in gm.graph.nodes
                    if not has_tag(node) and node.op == "call_function"
                ]

            marked_nodes_fw = [
                "t",
                "addmm",
                "view",
                "relu",
                "view_1",
                "t_1",
                "div",
                "addmm_1",
                "all_reduce",
                "wait_tensor",
                "view_2",
                "t_12",
            ]
            unmarked_nodes_fw = [
                "view_3",
                "t_2",
                "addmm_2",
                "view_4",
                "relu_1",
                "view_5",
                "t_3",
                "div_1",
                "addmm_3",
                "all_reduce_1",
                "wait_tensor_1",
                "view_6",
                "t_4",
                "t_8",
            ]

            marked_nodes_bw = [
                "mm_4",
                "t_13",
                "view_1",
                "mm_5",
                "t_14",
                "sum_3",
                "view_9",
                "t_15",
                "detach",
                "detach_1",
                "detach_6",
                "detach_7",
                "threshold_backward_1",
                "t_16",
                "mm_6",
                "t_17",
                "sum_4",
                "view_10",
                "t_18",
            ]
            unmarked_nodes_bw = [
                "mm",
                "t_5",
                "view_5",
                "mm_1",
                "t_6",
                "sum_1",
                "view_7",
                "t_7",
                "detach_2",
                "detach_3",
                "detach_4",
                "detach_5",
                "threshold_backward",
                "mm_2",
                "t_9",
                "mm_3",
                "t_10",
                "sum_2",
                "view_8",
                "t_11",
                "all_reduce_2",
                "wait_tensor_2",
            ]

            self.assertEqual(marked_nodes(fw_gm), marked_nodes_fw)
            self.assertEqual(unmarked_nodes(fw_gm), unmarked_nodes_fw)

            self.assertEqual(marked_nodes(bw_gm), marked_nodes_bw)
            self.assertEqual(unmarked_nodes(bw_gm), unmarked_nodes_bw)

            self.assertEqual(
                set(marked_nodes(joint_gm)), set(marked_nodes_fw + marked_nodes_bw)
            )
            self.assertEqual(
                set(unmarked_nodes(joint_gm)),
                set(unmarked_nodes_fw + unmarked_nodes_bw),
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

    def test_annotate_aot_export_joint_with_descriptors_alone(self):
        self._run_test(aot_export_joint_with_descriptors_alone, True)

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

        res = []
        for node in joint_gm.graph.nodes:
            if node.op == "placeholder":
                assert "val" in node.meta
                fake_val = node.meta["val"]
                if isinstance(fake_val, torch._subclasses.fake_tensor.FakeTensor):
                    res.append(list(fake_val.shape))

        self.assertExpectedInline(
            str(res),
            """[[4, 10], [4], [10, 4], [10], [s22, 10], [s22, 10]]""",
        )


instantiate_parametrized_tests(DTensorExportTest)


if __name__ == "__main__":
    run_tests()
