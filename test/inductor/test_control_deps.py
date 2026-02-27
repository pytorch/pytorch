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
            if len(nodes) != 3:
                raise AssertionError(f"Expected 3 nodes, got {len(nodes)}")
            c_node = nodes[0]
            d_node = nodes[1]
            e_node = nodes[2]

            if d_node.target != torch.ops.aten.mm.default:
                raise AssertionError(f"Expected mm.default, got {d_node.target}")

            from torch.utils._ordered_set import OrderedSet

            deps_map = {d_node: OrderedSet([c_node]), e_node: OrderedSet([d_node])}
            torch._inductor.fx_passes.control_dependencies.preserve_node_ordering(
                graph, deps_map
            )
            sub_g = graph.find_nodes(
                op="call_function", target=torch.ops.higher_order.control_deps
            )
            if len(sub_g) != 2:
                raise AssertionError(f"Expected 2 control_deps nodes, got {len(sub_g)}")

            if list(sub_g[0].meta["val"].shape) != [256, 256]:
                raise AssertionError(
                    f"Expected shape [256, 256], got {list(sub_g[0].meta['val'].shape)}"
                )
            if list(sub_g[1].meta["val"].shape) != [256, 256]:
                raise AssertionError(
                    f"Expected shape [256, 256], got {list(sub_g[1].meta['val'].shape)}"
                )

            for attr in graph.find_nodes(op="get_attr"):
                for n in getattr(graph.owning_module, attr.target).graph.nodes:
                    if list(n.meta["val"].shape) != [256, 256]:
                        raise AssertionError(
                            f"Expected shape [256, 256], got {list(n.meta['val'].shape)}"
                        )

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
            if len(mm_nodes) != 4:
                raise AssertionError(f"Expected 4 mm nodes, got {len(mm_nodes)}")

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

    @config.patch(reorder_for_locality=False)
    @requires_gpu()
    def test_control_deps_with_nested_args(self):
        """Test control_deps with operations that have nested args (e.g., torch.cat)."""

        def fn(a, b, c):
            x = a + 1
            y = b * 2
            # torch.cat has nested args: (List[Tensor], dim)
            cat_result = torch.cat([x, y], dim=0)
            z = cat_result + c
            return z

        def add_control_deps(graph):
            from torch.utils._ordered_set import OrderedSet

            # Find the cat node which has nested args
            cat_nodes = graph.find_nodes(
                op="call_function", target=torch.ops.aten.cat.default
            )
            if len(cat_nodes) != 1:
                raise AssertionError(f"Expected 1 cat node, got {len(cat_nodes)}")
            cat_node = cat_nodes[0]

            # Verify it has nested args (list of tensors)
            if not isinstance(cat_node.args[0], (list, tuple)):
                raise AssertionError(
                    f"Expected nested args, got {type(cat_node.args[0])}"
                )

            # Find a node that comes before cat to use as dependency
            add_nodes = graph.find_nodes(
                op="call_function", target=torch.ops.aten.add.Tensor
            )
            # Use the first add node (x = a + 1)
            dep_node = add_nodes[0]

            deps_map = {cat_node: OrderedSet([dep_node])}
            torch._inductor.fx_passes.control_dependencies.preserve_node_ordering(
                graph, deps_map
            )

            # Verify control_deps was created
            control_deps_nodes = graph.find_nodes(
                op="call_function", target=torch.ops.higher_order.control_deps
            )
            if len(control_deps_nodes) != 1:
                raise AssertionError(
                    f"Expected 1 control_deps node, got {len(control_deps_nodes)}"
                )
            return graph

        with torch._inductor.config.patch(
            post_grad_custom_post_pass=add_control_deps,
        ):
            a = torch.rand([128, 64], device=GPU_TYPE)
            b = torch.rand([128, 64], device=GPU_TYPE)
            c = torch.rand([256, 64], device=GPU_TYPE)

            compiled_fn = torch.compile(fn)
            result = compiled_fn(a, b, c)

            expected = fn(a, b, c)
            torch.testing.assert_close(result, expected)

    @requires_gpu()
    def test_control_deps_with_triton_kernel(self):
        """Test control_deps with triton_kernel_wrapper_mutation."""
        import triton
        import triton.language as tl

        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            tl.store(out_ptr + offsets, x + y, mask=mask)

        def fn(x, y):
            z = x * 2
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            return output + z

        def add_control_deps(graph):
            from torch.utils._ordered_set import OrderedSet

            # Find triton_kernel_wrapper_mutation nodes
            triton_nodes = graph.find_nodes(
                op="call_function",
                target=torch.ops.higher_order.triton_kernel_wrapper_functional,
            )
            if not triton_nodes:
                raise AssertionError("Expected triton_kernel_wrapper_functional nodes")
            # Find mul node (z = x * 2) to use as dependency
            mul_nodes = graph.find_nodes(
                op="call_function", target=torch.ops.aten.mul.Tensor
            )
            if not mul_nodes:
                raise AssertionError("Expected mul.Tensor nodes")
            deps_map = {triton_nodes[0]: OrderedSet([mul_nodes[0]])}
            torch._inductor.fx_passes.control_dependencies.preserve_node_ordering(
                graph, deps_map
            )
            return graph

        with torch._inductor.config.patch(
            post_grad_custom_post_pass=add_control_deps,
        ):
            x = torch.rand([256], device=GPU_TYPE)
            y = torch.rand([256], device=GPU_TYPE)

            compiled_fn = torch.compile(fn)
            result = compiled_fn(x, y)

            expected = fn(x, y)
            torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA_AND_TRITON:
        run_tests(needs="filelock")
