# Owner(s): ["oncall: distributed"]

import contextlib
import unittest

import torch
import torch.distributed as dist
from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
from torch._functorch.aot_autograd import aot_export_joint_with_descriptors
from torch._functorch.partitioners import min_cut_rematerialization_partition
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.utils.fuser_utils import fuse_by_partitions
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


def partition_mod(gm, supported_ops):
    import functools

    from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner

    partitioner = CapabilityBasedPartitioner(
        gm, supported_ops, allows_single_node_partition=True
    )

    candidate_partitions = partitioner.propose_partitions()
    partitioned_gm = fuse_by_partitions(
        partitioner.graph_module,
        [candidate_partitions[0].nodes],
        prefix="submod_",
        always_return_tuple=True,
    )

    def listify_inputs(fn):
        # Handles boxed arguments expectation from compile_fx_inner
        @functools.wraps(fn)
        def inner(*args):
            return fn(list(args))

        return inner

    # partitioned_gm.print_readable()
    for node in partitioned_gm.graph.nodes:
        if node.op == "call_module" and node.target.startswith("submod_"):
            fake_inputs = []
            for inp_node in node.all_input_nodes:
                if hasattr(inp_node, "meta") and "val" in inp_node.meta:
                    fake_inputs.append(inp_node.meta["val"])

            submod = getattr(partitioned_gm, node.target)
            # Ensure that it runs in eager
            submod(*fake_inputs)

            from torch._inductor.compile_fx import compile_fx_inner

            compiled_submod = listify_inputs(compile_fx_inner(submod, fake_inputs))

            with partitioned_gm.graph.inserting_after(node):
                new_node = partitioned_gm.graph.call_function(
                    compiled_submod, args=node.args, kwargs=node.kwargs
                )
                new_node.meta = node.meta
                node.replace_all_uses_with(new_node)
                partitioned_gm.graph.erase_node(node)
                del partitioned_gm._modules[node.target]

    partitioned_gm.recompile()
    return partitioned_gm


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

    def test_aot_eager_regional_inductor(self):
        def fn(x: torch.Tensor):
            a = torch.cos(x)
            b = torch.sin(a)
            c = torch.sin(x)
            d = torch.cos(b)
            return d + c

        x = torch.randn(4)
        joint_gm = graph_capture_and_aot_export_joint_with_descriptors(fn, x)
        fw_gm, bw_gm = min_cut_rematerialization_partition(
            joint_gm, None, num_fwd_outputs=1
        )

        class SupportSin(OperatorSupport):
            def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
                return (
                    node.op == "call_function"
                    and node.target is torch.ops.aten.sin.default
                )

        fw_gm = partition_mod(fw_gm, SupportSin())
        self.assertEqual(fw_gm(x)[0], fn(x))

    def test_aot_eager_compiled_flex(self):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        def _squared(score, b, h, m, n):
            """Joint graph needed for correctness"""
            return score * score

        def mask_mod(b, h, q, k):
            return q >= 0

        a = 12
        b = 64
        block_mask = create_block_mask(mask_mod, None, None, a * b, a * b)

        def fn(x: torch.Tensor):
            a = torch.cos(x)
            b = flex_attention(a, a, a, block_mask=block_mask, score_mod=_squared)
            c = torch.cos(b)
            return c

        v = torch.zeros(
            1,
            1,
            a * b,
            b,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        with torch._dynamo.config.patch(install_free_tensors=True):
            # TODO: switch to use the official graph_capture API once it is ready
            dynamo_gm = _dynamo_graph_capture_for_export(fn)(v)

        with contextlib.ExitStack() as stack:
            joint_with_descriptors = aot_export_joint_with_descriptors(
                stack,
                dynamo_gm,
                (v,),
            )

        joint_gm = joint_with_descriptors.graph_module
        fw_gm, bw_gm = min_cut_rematerialization_partition(
            joint_gm, None, num_fwd_outputs=1
        )

        flex_nodes = set(
            fw_gm.graph.find_nodes(
                op="call_function", target=torch.ops.higher_order.flex_attention
            )
        )
        flex_subgraph_nodes = set()
        for flex_node in flex_nodes:
            for arg in flex_node.all_input_nodes:
                if arg.op == "get_attr":
                    flex_subgraph_nodes.add(arg)

        supported_nodes = flex_nodes.union(flex_subgraph_nodes)

        class FlexOperatorSupport(OperatorSupport):
            def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
                return node in supported_nodes

        fw_gm = partition_mod(fw_gm, FlexOperatorSupport())

        fw_inputs = []
        for name in (
            joint_with_descriptors.params_spec + joint_with_descriptors.buffers_spec
        ):
            fw_inputs.append(getattr(dynamo_gm, name))
        fw_inputs.append(v)
        self.assertEqual(fw_gm(*fw_inputs)[0], fn(v))


instantiate_parametrized_tests(DTensorExportTest)


if __name__ == "__main__":
    run_tests()
