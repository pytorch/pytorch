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

        # Custom pass to add control dependency from d -> c
        def add_control_deps(graph):
            nodes = list(graph.nodes)

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


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA_AND_TRITON:
        run_tests(needs="filelock")
