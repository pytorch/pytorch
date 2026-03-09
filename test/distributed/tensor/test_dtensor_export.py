# Owner(s): ["oncall: distributed"]

import contextlib

import torch
import torch.distributed as dist
import torch.fx.traceback as fx_traceback
from torch._dynamo.functional_export import dynamo_graph_capture_for_export
from torch._functorch.aot_autograd import aot_export_joint_with_descriptors
from torch._functorch.partitioners import min_cut_rematerialization_partition
from torch._guards import tracing, TracingContext
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Partial, Replicate, Shard
from torch.distributed.tensor._api import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
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
from torch.utils._pytree import register_pytree_node


class SimpleModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mlp_0 = MLPModule(device)
        self.mlp_1 = MLPModule(device)

    def forward(self, input):
        return self.mlp_1(self.mlp_0(input))


class EinsumModel(torch.nn.Module):
    """Simple model that uses einsum with DTensor inputs and returns DTensor."""

    def __init__(self):
        super().__init__()
        self.placement = None

    def forward(self, x, y, z):
        result = torch.einsum("bsh,hd->bsd", x, y)
        self.placement = result.placements[0]
        self.placement_2 = y.placements[0]
        self.placement_3 = z.placements[0]
        return result


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


class FlexAttentionModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.proj_q = torch.nn.Linear(16, 128, device=device)
        self.proj_k = torch.nn.Linear(16, 128, device=device)
        self.proj_v = torch.nn.Linear(16, 128, device=device)
        self.proj_out = torch.nn.Linear(128, 16, device=device)
        self.num_heads = 8
        self.head_dim = 16

    def forward(self, x, *, block_mask=None):
        batch_size, seq_len, embed_dim = x.shape
        # Project to Q, K, V
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)
        # After colwise parallel, q/k/v are sharded on the last dimension
        # Get the actual size after sharding
        hidden_size = q.shape[-1]
        num_heads_local = hidden_size // self.head_dim
        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, num_heads_local, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads_local, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads_local, self.head_dim).transpose(1, 2)
        # Apply flex_attention
        attn_output_raw = flex_attention(q, k, v, block_mask=block_mask)
        # Reshape back to (batch, seq_len, hidden_size)
        attn_output = (
            attn_output_raw.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, hidden_size)
        )
        # Output projection
        output = self.proj_out(attn_output)
        return output


def strict_export_and_aot_export_joint_with_descriptors(model, args, kwargs=None):
    if kwargs is None:
        kwargs = {}
    # needed for stric export
    torch.utils._pytree.register_constant(DTensorSpec)

    # install_free_tensors is required for dynamo to work
    with torch._dynamo.config.patch(
        install_free_tensors=True, inline_inbuilt_nn_modules=True
    ):
        with torch._export.utils._disable_aten_to_metadata_assertions():
            ep = torch.export.export(model, args, kwargs, strict=True)

    # joint_gm produced here is missing the backward region, due to incompatiblility
    # between ep.module() and aot_export_joint_with_descriptors.
    # Keeping this here to show the issue.
    return aot_export_joint_with_descriptors_alone(ep.module(), args, kwargs)


def graph_capture_and_aot_export_joint_with_descriptors_v2(model, args, kwargs=None):
    if kwargs is None:
        kwargs = {}
    gm = dynamo_graph_capture_for_export(model)(*args, **kwargs)
    fake_mode = gm.meta.get("fake_mode", None)
    with tracing(TracingContext(fake_mode)):
        return aot_export_joint_with_descriptors_alone(gm, args, kwargs)


def aot_export_joint_with_descriptors_alone(model, args, kwargs=None):
    if kwargs is None:
        kwargs = {}
    with contextlib.ExitStack() as stack:
        joint_with_descriptors = aot_export_joint_with_descriptors(
            stack,
            model,
            args,
            kwargs,
        )
        return joint_with_descriptors.graph_module


def _count_op(gm, target):
    return sum(1 for node in gm.graph.nodes if node.target == target)


register_pytree_node(
    BlockMask,
    BlockMask._flatten,
    BlockMask._unflatten,
    flatten_with_keys_fn=BlockMask._flatten_with_keys,
    serialized_type_name="torch.nn.attention.flex_attention.BlockMask",
)


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

        inp = torch.rand(20, 10, device=self.device_type)
        inputs = (distribute_tensor(inp, mesh_2d["tp"], placements=[Replicate()]),)

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
                "t_16",
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
                "t_5",
                "t_11",
            ]

            marked_nodes_bw = [
                "mm_8",
                "t_17",
                "view_1",
                "mm_9",
                "t_18",
                "sum_5",
                "view_11",
                "t_19",
                "detach",
                "detach_3",
                "threshold_backward_1",
                "t_20",
                "mm_10",
                "t_21",
                "sum_6",
                "view_12",
                "t_22",
            ]
            unmarked_nodes_bw = [
                "mm_1",
                "t_7",
                "view_5",
                "mm_3",
                "t_8",
                "sum_2",
                "view_8",
                "t_9",
                "detach_1",
                "detach_2",
                "threshold_backward",
                "mm_5",
                "t_13",
                "mm_7",
                "t_14",
                "sum_4",
                "view_10",
                "t_15",
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
            graph_capture_and_aot_export_joint_with_descriptors_v2,
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
    def test_strict_export_parallelize_module_with_dtensor_input(self):
        self._run_test(strict_export_and_aot_export_joint_with_descriptors)

    def test_annotate_aot_export_joint_with_descriptors_alone(self):
        self._run_test(aot_export_joint_with_descriptors_alone, True)

    @parametrize(
        "export_fn_with_answer",
        [
            (
                graph_capture_and_aot_export_joint_with_descriptors_v2,
                "[[4, 10], [4], [10, 4], [10], [4, 10], [4], [10, 4], [10], [s64, 10], [s64, 10]]",
            ),
        ],
    )
    def test_dynamic_shapes(self, export_fn_with_answer):
        export_fn, answer = export_fn_with_answer
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

        inp = torch.rand(20, 10, device=self.device_type)
        inp_dtensor = distribute_tensor(inp, mesh_2d["tp"], placements=[Replicate()])
        torch._dynamo.mark_dynamic(inp_dtensor, 0, min=5, max=100)
        inputs = (inp_dtensor,)

        joint_gm = export_fn(tp_model, inputs)

        res = []
        for node in joint_gm.graph.nodes:
            if node.op == "placeholder":
                if "val" not in node.meta:
                    raise AssertionError(f"Expected 'val' in node.meta for {node}")
                fake_val = node.meta["val"]
                if isinstance(fake_val, torch._subclasses.fake_tensor.FakeTensor):
                    res.append(list(fake_val.shape))

        self.assertEqual(str(res), answer)

    @parametrize(
        "export_fn",
        [
            dynamo_graph_capture_for_export,
        ],
    )
    def test_einsum_dtensor_export(self, export_fn):
        """Test exporting a model with einsum that has DTensor inputs/outputs with side effects"""
        world_size = 4
        # Create device mesh
        device_mesh = init_device_mesh(self.device_type, mesh_shape=(world_size,))
        model = EinsumModel()

        x = torch.randn(4, 8, 16)
        x_dtensor = distribute_tensor(x, device_mesh, placements=[Shard(0)])

        # y: [16, 16] replicated
        y = torch.randn(16, 16)
        z = torch.randn(16, 16)
        y_dtensor = distribute_tensor(y, device_mesh, placements=[Replicate()])
        z_dtensor = DTensor.from_local(z, device_mesh, placements=[Partial()])
        inputs = (x_dtensor, y_dtensor, z_dtensor)

        # Run model to verify it works
        output = model(*inputs)
        gm = export_fn(model)(*inputs)
        output_gm = gm(*inputs)
        self.assertEqual(output, output_gm)

    @parametrize(
        "export_fn",
        [
            graph_capture_and_aot_export_joint_with_descriptors_v2,
        ],
    )
    def test_flex_attention_dtensor_export(self, export_fn):
        device_mesh = init_device_mesh(self.device_type, mesh_shape=(self.world_size,))
        model = FlexAttentionModel(self.device_type)

        # Parallelize the model: shard on head dimension
        # proj_q, proj_k, proj_v are colwise parallel (output is sharded on head dimension)
        # proj_out is rowwise parallel (input is sharded, output needs reduction)
        parallelize_plan = {
            "proj_q": ColwiseParallel(),
            "proj_k": ColwiseParallel(),
            "proj_v": ColwiseParallel(),
            "proj_out": RowwiseParallel(),
        }
        tp_model = parallelize_module(model, device_mesh, parallelize_plan)
        batch_size = 4
        seq_len = 64
        embed_dim = 16
        num_heads = 8

        # Input tensor replicated across all devices
        inp = torch.randn(batch_size, seq_len, embed_dim, device=self.device_type)
        inputs = (distribute_tensor(inp, device_mesh, placements=[Replicate()]),)

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal_mask,
            batch_size,
            num_heads,
            seq_len,
            seq_len,
            device=self.device_type,
        )

        flex_kwargs = {"block_mask": block_mask}

        joint_gm = export_fn(tp_model, inputs, flex_kwargs)

        self.assertTrue(
            _count_op(joint_gm, torch.ops.higher_order.flex_attention),
            1,
        )

        self.assertTrue(
            _count_op(joint_gm, torch.ops.higher_order.flex_attention_backward),
            2,
        )

    def test_union_typed_annotation(self):
        def fn(leaf: torch.Tensor | DTensor):
            def nest_fn(leaf: torch.Tensor | DTensor):
                # def nest_fn(leaf: Union[torch.Tensor, DTensor]):  # this works
                if isinstance(leaf, DTensor):
                    leaf = leaf.to_local()
                return leaf

            return nest_fn(leaf) + 1

        z = torch.randn(16, 16)
        gm = graph_capture_and_aot_export_joint_with_descriptors_v2(fn, (z,))

        self.assertEqual(fn(z), gm(z)[0])

    def test_dtensor_data_dependent_index_and_slice(self):
        device_mesh = init_device_mesh(self.device_type, mesh_shape=(self.world_size,))

        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x[y]

        x = torch.randn(10)
        y = torch.randint(1, (10,)).bool()
        x_dt = distribute_tensor(x, device_mesh, placements=[Replicate()])
        y_dt = distribute_tensor(y, device_mesh, placements=[Replicate()])
        dynamo_graph_capture_for_export(Foo())(x_dt, y_dt)

        class Bar(torch.nn.Module):
            def forward(self, x):
                val = torch.clamp(x.max(), min=1).item()
                torch._check(val >= 1)
                return x[:val]

        x = torch.randint(1000, (4, 64, 16))
        x_dt = distribute_tensor(x, device_mesh, placements=[Replicate()])
        gm = dynamo_graph_capture_for_export(Bar())(x_dt)
        self.assertExpectedInline(
            str(gm.graph).strip(),
            """\
graph():
    %l_x_ : torch.distributed.tensor.DTensor [num_users=2] = placeholder[target=L_x_]
    %max_1 : [num_users=1] = call_method[target=max](args = (%l_x_,), kwargs = {})
    %clamp : [num_users=1] = call_function[target=torch.clamp](args = (%max_1,), kwargs = {min: 1})
    %item : [num_users=2] = call_method[target=item](args = (%clamp,), kwargs = {})
    %ge_1 : [num_users=1] = call_function[target=operator.ge](args = (%item, 1), kwargs = {})
    %_assert_scalar_default : [num_users=0] = call_function[target=torch.ops.aten._assert_scalar.default](args = (%ge_1, Runtime assertion failed for expression u0 >= 1 on node 'ge_1'), kwargs = {})
    %getitem : [num_users=3] = call_function[target=operator.getitem](args = (%l_x_, slice(None, item, None)), kwargs = {})
    %getattr_1 : [num_users=1] = call_function[target=builtins.getattr](args = (%getitem, _local_tensor), kwargs = {})
    %sym_size_int : [num_users=2] = call_function[target=torch.ops.aten.sym_size.int](args = (%getattr_1, 0), kwargs = {})
    %sym_size_int_1 : [num_users=2] = call_function[target=torch.ops.aten.sym_size.int](args = (%getitem, 0), kwargs = {})
    %ge_2 : [num_users=1] = call_function[target=operator.ge](args = (%sym_size_int, 0), kwargs = {})
    %_assert_scalar_default_1 : [num_users=0] = call_function[target=torch.ops.aten._assert_scalar.default](args = (%ge_2, Runtime assertion failed for expression u2 >= 0 on node 'ge_2'), kwargs = {})
    %le : [num_users=1] = call_function[target=operator.le](args = (%sym_size_int, 4), kwargs = {})
    %_assert_scalar_default_2 : [num_users=0] = call_function[target=torch.ops.aten._assert_scalar.default](args = (%le, Runtime assertion failed for expression u2 <= 4 on node 'le'), kwargs = {})
    %ge_3 : [num_users=1] = call_function[target=operator.ge](args = (%sym_size_int_1, 0), kwargs = {})
    %_assert_scalar_default_3 : [num_users=0] = call_function[target=torch.ops.aten._assert_scalar.default](args = (%ge_3, Runtime assertion failed for expression u1 >= 0 on node 'ge_3'), kwargs = {})
    %le_1 : [num_users=1] = call_function[target=operator.le](args = (%sym_size_int_1, 4), kwargs = {})
    %_assert_scalar_default_4 : [num_users=0] = call_function[target=torch.ops.aten._assert_scalar.default](args = (%le_1, Runtime assertion failed for expression u1 <= 4 on node 'le_1'), kwargs = {})
    return (getitem,)""",  # noqa: B950
        )

    def test_dtensor_mark_unbacked(self):
        device_mesh = init_device_mesh(
            self.device_type, mesh_shape=(self.world_size // 2, 2)
        )

        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x @ y

        x_dt = distribute_tensor(
            torch.randn(64, 64), device_mesh, placements=[Replicate(), Replicate()]
        )
        y_dt = x_dt.clone()
        for i in range(2):
            torch._dynamo.decorators.mark_unbacked(x_dt, i)
            torch._dynamo.decorators.mark_unbacked(y_dt, i)

        gm = dynamo_graph_capture_for_export(Foo())(x_dt, y_dt)
        n = 0
        for node in gm.graph.nodes:
            if bindings := node.meta.get("unbacked_bindings", {}):
                # 2 outer sizes, 2 inner sizes
                self.assertEqual(len(bindings), 4)
                n += 1
        self.assertEqual(n, 2)  # 2 nodes with bindings (x, y)

        # test size-0 tensor
        z_dt = distribute_tensor(
            torch.randn(0, 0), device_mesh, placements=[Replicate(), Replicate()]
        )
        self.assertEqual(gm(z_dt, z_dt).shape, (0, 0))


instantiate_parametrized_tests(DTensorExportTest)


if __name__ == "__main__":
    run_tests()
