# Owner(s): ["module: inductor"]

import torch
from torch._inductor import config
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
            d = b @ b
            e = c * 2
            return d, e

        def add_control_deps(graph):
            nodes = [n for n in graph.nodes if n.op == "call_function"]
            assert len(nodes) == 3
            c_node = nodes[0]
            d_node = nodes[1]
            e_node = nodes[2]

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

    @config.patch(allow_buffer_reuse=False)
    @requires_gpu()
    def test_control_deps_do_not_extend_buffer_lifetime(self):
        """
        Control deps should not extend buffer lifetimes - buf0/buf1 should be
        deleted before the 4th matmul, not kept alive by the control dependency.
        """

        def fn(a, b):
            # Chain of 4 matmuls: mm0 -> mm1 -> mm2 -> mm3
            mm0 = a @ b
            mm1 = mm0 @ b
            mm2 = mm1 @ b
            mm3 = mm2 @ b
            return mm3

        def add_control_deps(graph):
            from torch.utils._ordered_set import OrderedSet

            mm_nodes = graph.find_nodes(
                op="call_function", target=torch.ops.aten.mm.default
            )
            assert len(mm_nodes) == 4, f"Expected 4 mm nodes, got {len(mm_nodes)}"

            # Add control dep: mm3 depends on mm0's output
            # This should NOT extend mm0's buffer lifetime
            deps_map = {mm_nodes[3]: OrderedSet([mm_nodes[0]])}
            torch._inductor.fx_passes.control_dependencies.preserve_node_ordering(
                graph, deps_map
            )
            return graph

        with torch._inductor.config.patch(
            post_grad_custom_post_pass=add_control_deps,
        ):
            a = torch.rand([256, 256], device=GPU_TYPE)
            b = torch.rand([256, 256], device=GPU_TYPE)

            result, code = run_and_get_code(torch.compile(fn), a, b)
            torch.testing.assert_close(result, fn(a, b))

            # buf0 should be allocated, passed in out=, used once, then del
            FileCheck().check("buf0 = ").check_count(
                "extern_kernels.mm", 2, exactly=True
            ).check("del buf0").run(code[0])


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA_AND_TRITON:
        run_tests(needs="filelock")
