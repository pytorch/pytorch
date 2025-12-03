# Owner(s): ["module: inductor"]

import torch
from torch._inductor import config
from torch._inductor.dependencies import WeakDep
from torch._inductor.test_case import run_tests, TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CUDA_AND_TRITON,
    requires_gpu,
)


class TestControlDeps(InductorTestCase):
    @config.patch(reorder_for_locality=False)
    @requires_gpu()
    def test_control_deps_prevents_fusion(self):
        def fn(a, b):
            c = a + 1
            d = (b @ b) * 4.5
            e = c * 2
            return d, e

        # Custom pass to add control dependency from d -> c
        def add_control_deps(graph):
            nodes = list(graph.nodes)

            nodes = [n for n in graph.nodes if n.op == "call_function"]
            assert len(nodes) == 4
            c_node = nodes[0]
            d_node = nodes[1]
            e_node = nodes[3]

            assert d_node.target == torch.ops.aten.mm.default

            from torch.utils._ordered_set import OrderedSet

            deps_map = {d_node: OrderedSet([c_node]), e_node: OrderedSet([d_node])}
            torch._inductor.fx_passes.control_dependencies.preserve_node_ordering(
                graph, deps_map
            )
            sub_g = graph.find_nodes(
                op="call_function", target=torch.ops.higher_order.control_deps
            )
            assert len(sub_g) == 2

            assert list(sub_g[0].meta["val"].shape) == [256, 256]
            assert list(sub_g[1].meta["val"].shape) == [256, 256]

            for attr in graph.find_nodes(op="get_attr"):
                for n in getattr(graph.owning_module, attr.target).graph.nodes:
                    assert list(n.meta["val"].shape) == [256, 256]

            return graph

        with torch._inductor.config.patch(
            post_grad_custom_post_pass=add_control_deps,
        ):
            compiled_fn = torch.compile(fn)
            a = torch.rand([256, 256], device=GPU_TYPE)
            b = torch.rand([256, 256], device=GPU_TYPE)

            _, code = run_and_get_code(torch.compile(fn), a, b)
            result = compiled_fn(a, b)

            FileCheck().check(".run(").check("extern_kernels.mm(").check(".run(").run(
                code[0]
            )

            expected = fn(a, b)
            torch.testing.assert_close(result, expected)

    @config.patch(reorder_for_locality=False)
    @requires_gpu()
    def test_control_deps_do_not_extend_buffer_lifetime(self):
        """
        Test that control_deps (which use is_fake WeakDeps) do not extend buffer lifetimes.

        Control dependencies are for ordering only - they should not prevent
        a buffer from being freed/reused when it's no longer actually used.
        This test verifies that buffers can be reused even when there's a
        control dependency on them.
        """

        def fn(a, b):
            c = a + 1
            d = (b @ b) * 4.5
            e = c * 2
            return d, e

        def add_control_deps(graph):
            nodes = [n for n in graph.nodes if n.op == "call_function"]
            c_node = None
            mm_node = None
            e_node = None
            for n in nodes:
                if n.target == torch.ops.aten.add.Tensor:
                    c_node = n
                elif n.target == torch.ops.aten.mm.default:
                    mm_node = n
                elif n.target == torch.ops.aten.mul.Tensor and c_node is not None:
                    if c_node in n.args:
                        e_node = n

            if e_node is None:
                mul_nodes = [n for n in nodes if n.target == torch.ops.aten.mul.Tensor]
                if len(mul_nodes) >= 2:
                    e_node = mul_nodes[1]

            if mm_node and e_node:
                from torch.utils._ordered_set import OrderedSet

                # Add control dep: e depends on mm result (d)
                # This should NOT extend the lifetime of mm's output buffer
                deps_map = {e_node: OrderedSet([mm_node])}
                torch._inductor.fx_passes.control_dependencies.preserve_node_ordering(
                    graph, deps_map
                )
            return graph

        with torch._inductor.config.patch(
            post_grad_custom_post_pass=add_control_deps,
        ):
            a = torch.rand([256, 256], device=GPU_TYPE)
            b = torch.rand([256, 256], device=GPU_TYPE)

            _, code = run_and_get_code(torch.compile(fn), a, b)

            # The key check: buffer reuse should still happen
            # Without the fix, the mm output buffer would not be reused
            # because the control dep would incorrectly extend its lifetime
            # With the fix, we should see "buf2 = buf0; del buf0  # reuse"
            self.assertIn(
                "# reuse",
                code[0],
                "Buffer reuse should occur - control deps should not extend lifetime",
            )


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA_AND_TRITON:
        run_tests(needs="filelock")
