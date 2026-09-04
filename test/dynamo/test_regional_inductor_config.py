# Owner(s): ["module: dynamo"]

import torch
import torch._inductor.test_case
from torch._higher_order_ops.invoke_subgraph import (
    get_invoke_subgraph_compile_options,
    NestedCompileRegionOptions,
)
from torch._inductor.test_case import run_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfTorchDynamo,
)


@skipIfTorchDynamo("Not a suitable dynamo wrapped test")
@torch._dynamo.config.patch("enable_invoke_subgraph_regional_compile", True)
@instantiate_parametrized_tests
class NestedRegionInductorConfigTests(torch._inductor.test_case.TestCase):
    @staticmethod
    def _empty_graph_module():
        graph = torch.fx.Graph()
        graph.output(())
        return torch.fx.GraphModule({}, graph)

    @classmethod
    def _configured_region_graph_module(cls, nested_config, body=None):
        if body is None:
            body = cls._empty_graph_module()
        root = torch.nn.Module()
        root.add_module("body", body)
        graph = torch.fx.Graph()
        body_node = graph.get_attr("body")
        region = graph.call_function(
            torch.ops.higher_order.invoke_subgraph, (body_node,)
        )
        region.meta["custom"] = {"nested_region_config": nested_config}
        graph.output(())
        return torch.fx.GraphModule(root, graph)

    @staticmethod
    def _reference_submodule(gm, target):
        output = next(iter(gm.graph.find_nodes(op="output")))
        with gm.graph.inserting_before(output):
            gm.graph.get_attr(target)
        gm.recompile()

    def test_invalid_inductor_config(self):
        """Test that invalid inductor config keys are caught with a clear error."""

        with self.assertRaisesRegex(
            ValueError,
            "Invalid inductor config key 'invalid_config_key'",
        ):
            get_invoke_subgraph_compile_options(
                fw_inductor_config_patches={
                    "invalid_config_key": True,
                }
            )

    @parametrize("direction", ("forward", "backward"))
    def test_unsupported_nested_region_inductor_config(self, direction):
        config_arg = (
            "fw_inductor_config_patches"
            if direction == "forward"
            else "bw_inductor_config_patches"
        )
        with self.assertRaisesRegex(
            ValueError,
            "Inductor config key 'triton.persistent_reductions' "
            f"is not supported in {direction}",
        ):
            get_invoke_subgraph_compile_options(
                **{config_arg: {"triton.persistent_reductions": False}}
            )

    def test_nested_region_options_validate_direct_construction(self):
        with self.assertRaisesRegex(
            ValueError,
            "Inductor config key 'graph_partition' is not supported in forward",
        ):
            NestedCompileRegionOptions(
                inductor_config_patches={"graph_partition": True}
            )

    @parametrize("direction", ("forward", "backward"))
    def test_nested_region_options_revalidate_mutated_config(self, direction):
        patches = {}
        config_arg = (
            "fw_inductor_config_patches"
            if direction == "forward"
            else "bw_inductor_config_patches"
        )
        nested_config = get_invoke_subgraph_compile_options(**{config_arg: patches})
        patches["cudagraph_unsafe_unbacked_ops"] = []

        graph = torch.fx.Graph()
        node = graph.call_function(torch.ops.higher_order.invoke_subgraph)
        node.meta["custom"] = {"nested_region_config": nested_config}
        graph.output(())
        gm = torch.fx.GraphModule({}, graph)

        from torch._inductor.compile_fx import create_compiler_config_extra

        with self.assertRaisesRegex(
            ValueError,
            "Inductor config key 'cudagraph_unsafe_unbacked_ops' "
            f"is not supported in {direction}",
        ):
            create_compiler_config_extra(gm)

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(fx_graph_cache=False, fx_graph_remote_cache=False)
    def test_nested_region_options_revalidate_lazy_backward(self):
        backward_patches = {}
        nested_config = get_invoke_subgraph_compile_options(
            bw_inductor_config_patches=backward_patches
        )
        pass_calls = []

        def forbidden_pass(graph):
            pass_calls.append(graph)

        @torch.compiler.nested_compile_region(options=nested_config)
        def region(x):
            return torch.sin(x)

        x = torch.randn(10, requires_grad=True)
        result = torch.compile(region, backend="inductor", fullgraph=True)(x)

        backward_patches["post_grad_custom_post_pass"] = forbidden_pass
        with self.assertRaisesRegex(
            ValueError,
            "Inductor config key 'post_grad_custom_post_pass' "
            "is not supported in backward",
        ):
            result.sum().backward()
        self.assertEqual(pass_calls, [])

    @torch._inductor.config.patch(
        freezing=True,
        fx_graph_cache=False,
        fx_graph_remote_cache=False,
        pre_grad_pass_timing="early",
    )
    def test_nested_region_options_revalidate_freezing_after_pre_grad(self):
        from torch._dynamo.exc import BackendCompilerFailed

        patches = {}
        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches=patches
        )
        pass_calls = []

        def forbidden_pass(graph):
            pass_calls.append(graph)

        def mutate_config(_graph):
            patches["post_grad_custom_post_pass"] = forbidden_pass

        @torch.compiler.nested_compile_region(options=nested_config)
        def region(x):
            return torch.sin(x)

        def fn(x):
            return region(torch.cos(x)) + 1

        with (
            torch.no_grad(),
            torch._inductor.config.patch(pre_grad_custom_pass=mutate_config),
            self.assertRaisesRegex(
                BackendCompilerFailed,
                "Inductor config key 'post_grad_custom_post_pass' is not supported",
            ),
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(torch.randn(10))
        self.assertEqual(pass_calls, [])

    def test_nested_region_options_ignore_unreferenced_graph_module(self):
        patches = {}
        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches=patches
        )
        patches["cudagraph_unsafe_unbacked_ops"] = []
        unused = self._configured_region_graph_module(nested_config)
        gm = self._empty_graph_module()
        gm.add_module("unused", unused)

        from torch._inductor.compile_fx import create_compiler_config_extra

        create_compiler_config_extra(gm)
        self._reference_submodule(gm, "unused")

        with self.assertRaisesRegex(
            ValueError,
            "Inductor config key 'cudagraph_unsafe_unbacked_ops' "
            "is not supported in forward",
        ):
            create_compiler_config_extra(gm)

    @parametrize("target", ("live_alias", "container.region"))
    def test_nested_region_options_validate_referenced_graph_module(self, target):
        patches = {}
        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches=patches
        )
        patches["cudagraph_unsafe_unbacked_ops"] = []
        region = self._configured_region_graph_module(nested_config)
        gm = self._empty_graph_module()
        if target == "live_alias":
            gm.add_module("unused_alias", region)
            gm.add_module(target, region)
        else:
            container = torch.nn.Module()
            container.add_module("region", region)
            gm.add_module("container", container)
        self._reference_submodule(gm, target)

        from torch._inductor.compile_fx import create_compiler_config_extra

        with self.assertRaisesRegex(
            ValueError,
            "Inductor config key 'cudagraph_unsafe_unbacked_ops' "
            "is not supported in forward",
        ):
            create_compiler_config_extra(gm)


if __name__ == "__main__":
    run_tests()
