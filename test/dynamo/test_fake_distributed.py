# Owner(s): ["module: dynamo"]
from unittest import skipIf

import torch
import torch.distributed as dist
from torch._dynamo.test_case import TestCase as DynamoTestCase
from torch._dynamo.testing import AotEagerAndRecordGraphs, normalize_gm
from torch.testing._internal.common_utils import instantiate_parametrized_tests


if dist.is_available():
    from torch.distributed._functional_collectives import (
        all_to_all_single_autograd,
        wait_tensor,
    )
    from torch.distributed.device_mesh import init_device_mesh


def normalize_graph(gm):
    return normalize_gm(gm.print_readable(print_output=False))


@skipIf(not dist.is_available(), "requires distributed")
class TestFakeDistributed(DynamoTestCase):
    def setUp(self):
        # Use FakeProcessGroup to run tests on a single process
        dist.init_process_group(backend="fake", rank=0, world_size=2)
        self.local_rank = 0
        self.world_size = 2

    def tearDown(self):
        dist.destroy_process_group()

    def test_all_to_all_single_autograd(self):
        backend = AotEagerAndRecordGraphs()

        @torch.compile(fullgraph=True, backend=backend)
        def fn(x):
            return all_to_all_single_autograd(
                x,
                None,  # Will use equal splits
                None,  # Will use equal splits
                group=dist.group.WORLD,
            )

        # Test backed shapes
        x = torch.randn(8, 8, requires_grad=True)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(x, 1)
        wait_tensor(fn(x))
        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)
        self.assertExpectedInline(
            normalize_graph(backend.fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(s77)", primals_2: "Sym(s27)", primals_3: "f32[s77, s27]"):
        floordiv: "Sym((s77//2))" = primals_1 // 2

        all_to_all_single: "f32[2*((s77//2)), s27]" = torch.ops._c10d_functional.all_to_all_single.default(primals_3, [floordiv, floordiv], [floordiv, floordiv], '0');  primals_3 = None

        wait_tensor: "f32[2*((s77//2)), s27]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single);  all_to_all_single = None
        return (wait_tensor, primals_1, primals_2, floordiv)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            normalize_graph(backend.bw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(s77)", primals_2: "Sym(s27)", floordiv: "Sym((s77//2))", tangents_1: "f32[2*((s77//2)), s27]"):
        all_to_all_single_1: "f32[2*((s77//2)), s27]" = torch.ops._c10d_functional.all_to_all_single.default(tangents_1, [floordiv, floordiv], [floordiv, floordiv], '0');  tangents_1 = floordiv = None
        wait_tensor_1: "f32[2*((s77//2)), s27]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_1);  all_to_all_single_1 = None
        return (None, None, wait_tensor_1)
""",  # noqa: B950
        )

        backend.fw_graphs.clear()
        backend.bw_graphs.clear()

        # Test unbacked shapes
        x = torch.randn(8, 8, 8, requires_grad=True)
        torch._dynamo.decorators.mark_unbacked(x, 0)
        torch._dynamo.decorators.mark_unbacked(x, 1)
        torch._dynamo.decorators.mark_unbacked(x, 2)
        wait_tensor(fn(x))
        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)
        self.assertExpectedInline(
            normalize_graph(backend.fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(u0)", primals_2: "Sym(u1)", primals_3: "Sym(u2)", primals_4: "f32[u0, u1, u2]"):
        ge: "Sym(u0 >= 0)" = primals_1 >= 0
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar = None
        ge_1: "Sym(u1 >= 0)" = primals_2 >= 0
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_1 = None
        ge_2: "Sym(u2 >= 0)" = primals_3 >= 0
        _assert_scalar_2 = torch.ops.aten._assert_scalar.default(ge_2, "Runtime assertion failed for expression u2 >= 0 on node 'ge_2'");  ge_2 = _assert_scalar_2 = None

        floordiv: "Sym((u0//2))" = primals_1 // 2

        all_to_all_single: "f32[2*((u0//2)), u1, u2]" = torch.ops._c10d_functional.all_to_all_single.default(primals_4, [floordiv, floordiv], [floordiv, floordiv], '0');  primals_4 = None

        wait_tensor: "f32[2*((u0//2)), u1, u2]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single);  all_to_all_single = None
        return (wait_tensor, primals_1, primals_2, primals_3, floordiv)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            normalize_graph(backend.bw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(u0)", primals_2: "Sym(u1)", primals_3: "Sym(u2)", floordiv: "Sym((u0//2))", tangents_1: "f32[2*((u0//2)), u1, u2]"):
        all_to_all_single_1: "f32[2*((u0//2)), u1, u2]" = torch.ops._c10d_functional.all_to_all_single.default(tangents_1, [floordiv, floordiv], [floordiv, floordiv], '0');  tangents_1 = floordiv = None
        wait_tensor_1: "f32[2*((u0//2)), u1, u2]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_1);  all_to_all_single_1 = None
        return (None, None, None, wait_tensor_1)
""",  # noqa: B950
        )

    def test_device_mesh_get_local_rank(self):
        device_mesh = init_device_mesh(
            device_type="cpu",
            mesh_shape=(self.world_size,),
            mesh_dim_names=("dp",),  # data parallel dimension
        )

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            local_rank = device_mesh.get_local_rank()
            global_rank = device_mesh.get_rank()
            if "dp" not in device_mesh.mesh_dim_names:
                x = x * 2
            return x + local_rank + global_rank

        x = torch.ones(10)
        res = fn(x)
        self.assertEqual(res, x)

    def test_device_mesh_flatten(self):
        device_mesh = init_device_mesh(
            device_type="cpu",
            mesh_shape=(
                1,
                self.world_size,
            ),
            mesh_dim_names=("dp", "tp"),
        )
        self.assertEqual(device_mesh.get_coordinate(), [0, 0])

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            dm = device_mesh._flatten()
            return x + 1, dm.get_coordinate()

        x = torch.ones(10)
        res = fn(x)
        self.assertEqual(res, (x + 1, [0]))

    def test_device_mesh_as_fake_script_object(self):
        """
        Test that DeviceMeshVariable correctly handles DeviceMesh wrapped in
        FakeScriptObject during fake distributed tracing.
        """
        from torch._library.fake_class_registry import FakeScriptObject

        device_mesh = init_device_mesh(
            device_type="cpu",
            mesh_shape=(self.world_size,),
            mesh_dim_names=("dp",),
        )

        fake_mesh = FakeScriptObject(
            wrapped_obj=device_mesh,
            script_class_name="DeviceMesh",
            x=device_mesh,
        )

        from torch._dynamo.variables.distributed import DeviceMeshVariable
        from torch._dynamo.source import ConstantSource

        self.assertTrue(DeviceMeshVariable.is_device_mesh(fake_mesh))

        var = DeviceMeshVariable(
            fake_mesh, source=ConstantSource("fake_mesh")
        )
        self.assertEqual(var.value, device_mesh)
        self.assertEqual(var.value.ndim, device_mesh.ndim)
        self.assertEqual(var.value.device_type, device_mesh.device_type)

    def test_process_group_as_graph_input(self):
        """
        Test that ProcessGroupVariable correctly handles proxy tracking.
        ProcessGroup objects should be lifted as graph inputs rather than
        being embedded as constants (since repr() produces invalid Python syntax).
        """
        from torch._dynamo.variables.distributed import ProcessGroupVariable
        from torch._dynamo.source import ConstantSource
        import torch.fx as fx

        pg = dist.group.WORLD
        graph = fx.Graph()
        proxy = graph.placeholder("process_group")

        pg_var = ProcessGroupVariable(
            pg, proxy=proxy, source=ConstantSource("pg")
        )

        self.assertEqual(pg_var.value, pg)
        self.assertEqual(pg_var.as_python_constant(), pg)
        self.assertEqual(pg_var.as_proxy(), proxy)

    def test_is_process_group_detection(self):
        """
        Test _is_process_group correctly identifies ProcessGroup and FakeProcessGroup.
        This helper is used by both the partitioner and graph compiler to properly
        handle ProcessGroup objects in AOT Autograd.
        """
        from torch._functorch.partitioners import _is_process_group

        pg = dist.group.WORLD
        self.assertTrue(_is_process_group(pg))

        self.assertFalse(_is_process_group(torch.tensor([1, 2, 3])))
        self.assertFalse(_is_process_group("not_a_pg"))
        self.assertFalse(_is_process_group(None))

    def test_extract_runtime_device_meshes(self):
        """
        Test extract_runtime_device_meshes correctly extracts DeviceMesh from DTensor.
        This is used in the 'device mesh in, device mesh out' pattern for
        precompilation: when loading a precompiled artifact, the output DTensors
        should use the live DeviceMesh from the runtime inputs.
        """
        from torch._functorch._aot_autograd.subclass_utils import (
            extract_runtime_device_meshes,
        )
        from torch.distributed.tensor import DTensor

        device_mesh = init_device_mesh(
            device_type="cpu",
            mesh_shape=(self.world_size,),
            mesh_dim_names=("dp",),
        )

        local_tensor = torch.randn(4, 4)
        dtensor = DTensor.from_local(local_tensor, device_mesh, [])

        extracted_mesh = extract_runtime_device_meshes([dtensor])
        self.assertEqual(extracted_mesh, device_mesh)

        no_mesh = extract_runtime_device_meshes([torch.randn(4, 4)])
        self.assertIsNone(no_mesh)

        no_mesh_empty = extract_runtime_device_meshes([])
        self.assertIsNone(no_mesh_empty)


instantiate_parametrized_tests(TestFakeDistributed)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
