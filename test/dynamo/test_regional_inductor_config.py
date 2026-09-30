# Owner(s): ["module: dynamo"]

import gc
import importlib
import unittest
from dataclasses import FrozenInstanceError
from unittest import mock

import torch
import torch._inductor.test_case
from torch._higher_order_ops.invoke_subgraph import (
    get_invoke_subgraph_compile_options,
    NestedCompileRegionOptions,
)
from torch._inductor.test_case import run_tests
from torch._inductor.utils import run_and_get_code, run_fw_bw_and_get_code
from torch._inductor.virtualized import V
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfTorchDynamo,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, IS_BIG_GPU
from torch.testing._internal.triton_utils import (
    requires_cuda_and_triton,
    requires_gpu_and_triton,
)


# TORCHINDUCTOR_FORCE_DISABLE_CACHES is env_name_force, so a test that needs a
# warm cache cannot patch its way out of it.
skip_if_caches_force_disabled = unittest.skipIf(
    torch._inductor.config.force_disable_caches,
    "needs caching; caches are force-disabled for this process",
)


@skipIfTorchDynamo("Not a suitable dynamo wrapped test")
@torch._dynamo.config.patch("enable_invoke_subgraph_regional_compile", True)
@instantiate_parametrized_tests
class NestedRegionInductorConfigTests(torch._inductor.test_case.TestCase):
    @staticmethod
    def _generated_fn_body(code, signature):
        start = code.index(signature)
        indent = start - (code.rfind("\n", 0, start) + 1)
        lines = code[start:].split("\n")
        body = [lines[0]]
        for line in lines[1:]:
            if line.strip() and len(line) - len(line.lstrip()) <= indent:
                break
            body.append(line)
        return "\n".join(body)

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

    def test_check_multiple_devices_or_any_cpu_nodes(self):
        """A device-less mapping has no GPU work, so capture is refused.

        This helper is shared with whole-graph compiles and the dynamo cudagraphs
        backend, so the empty case has to be right for them too: a CPU-only graph
        under partitioning lands here once the cpu entry is popped, and reporting
        no reason would claim an all-CPU graph is capturable.
        """
        from torch._inductor.cudagraph_utils import (
            check_multiple_devices_or_any_cpu_nodes,
        )

        graph = torch.fx.Graph()
        meta = torch.device("meta")
        cuda0 = torch.device("cuda", 0)
        cuda1 = torch.device("cuda", 1)
        cpu = torch.device("cpu")

        for mapping, partition, expected in (
            ({}, False, "no GPU ops"),
            ({meta: graph.placeholder("m")}, False, "no GPU ops"),
            # A CPU-only graph under partitioning: the cpu entry is tolerated and
            # popped, and what is left is not capturable.
            ({cpu: graph.placeholder("c1")}, True, "no GPU ops"),
            (
                {cuda0: graph.placeholder("a"), meta: graph.placeholder("m2")},
                False,
                None,
            ),
            ({cuda0: graph.placeholder("a2")}, False, None),
            (
                {cuda0: graph.placeholder("a4"), cpu: graph.placeholder("c2")},
                True,
                None,
            ),
            (
                {cuda0: graph.placeholder("a3"), cuda1: graph.placeholder("b")},
                False,
                "multiple devices",
            ),
            ({cpu: graph.placeholder("c")}, False, "cpu device"),
        ):
            with self.subTest(devices=sorted(map(str, mapping)), partition=partition):
                reason = check_multiple_devices_or_any_cpu_nodes(
                    dict(mapping), use_cudagraph_partition=partition
                )
                if expected is None:
                    self.assertIsNone(reason)
                else:
                    self.assertIn(expected, reason)

    @parametrize("cpp_wrapper,aot_mode", ((True, False), (False, True), (False, False)))
    @torch._inductor.config.patch("graph_partition", True)
    def test_maybe_disable_graph_partition(self, cpp_wrapper, aot_mode):
        from torch._inductor.compile_fx import maybe_disable_graph_partition

        with maybe_disable_graph_partition(cpp_wrapper, aot_mode):
            self.assertEqual(
                torch._inductor.config.graph_partition, not (cpp_wrapper or aot_mode)
            )
        self.assertTrue(torch._inductor.config.graph_partition)

    @parametrize("cpp_wrapper,aot_mode", ((True, False), (False, True), (False, False)))
    @parametrize("region_opts_in", (False, True))
    def test_unsupported_graph_partition_and_regional_cudagraphs(
        self, cpp_wrapper, aot_mode, region_opts_in
    ):
        """cpp_wrapper/aot_mode cannot partition, so a region cannot be isolated.

        The decision has to land on the box the caller holds, in this process,
        and it depends on which way the region's override points.
        """
        from torch._inductor.compile_fx import compile_fx_inner
        from torch._inductor.utils import BoxedBool

        graph = torch.fx.Graph()
        relu = graph.call_function(
            torch.ops.aten.relu.default, (graph.placeholder("x"),)
        )
        graph.output((relu,))
        gm = torch.fx.GraphModule({}, graph)
        cudagraphs = BoxedBool(True)

        with (
            mock.patch(
                "torch._inductor.compile_fx.fx_codegen_and_compile"
            ) as codegen_and_compile,
            V.set_aot_compilation(aot_mode),
        ):
            compile_fx_inner(
                gm,
                [torch.empty(0)],
                cudagraphs=cudagraphs,
                cpp_wrapper=cpp_wrapper,
                cudagraphs_region_aware=True,
                cudagraphs_top_level=not region_opts_in,
                cudagraph_region_forced_partition=True,
            )

        unsupported = cpp_wrapper or aot_mode
        # An opt-in only asked for capture inside the region, so without
        # partitioning it must be dropped rather than widened to the whole graph.
        # An opt-out keeps the enclosing capture: losing every CUDA graph in the
        # model is further from the request than failing to exclude one region.
        self.assertEqual(bool(cudagraphs), not (unsupported and region_opts_in))
        # Either way the forced partitioning must not reach lowering.
        self.assertEqual(
            codegen_and_compile.call_args.kwargs["cudagraph_region_forced_partition"],
            not unsupported,
        )

    @parametrize(
        "is_backward,is_inference",
        ((False, False), (True, False), (False, True)),
    )
    def test_invoke_subgraph_compile_passes_cudagraph_state(
        self, is_backward, is_inference
    ):
        from torch._higher_order_ops.invoke_subgraph import (
            invoke_subgraph_inductor_compile,
        )
        from torch._inductor.utils import BoxedBool

        def compiled_fn(args):
            return args

        compiled_fn._boxed_call = True
        with mock.patch(
            "torch._inductor.compile_fx.compile_fx_inner",
            return_value=compiled_fn,
        ) as compile_fx_inner:
            invoke_subgraph_inductor_compile(
                self._empty_graph_module(),
                [],
                {"triton.cudagraphs": True},
                is_backward=is_backward,
                is_inference=is_inference,
            )

        compile_kwargs = compile_fx_inner.call_args.kwargs
        self.assertIsInstance(compile_kwargs["cudagraphs"], BoxedBool)
        self.assertTrue(compile_kwargs["cudagraphs"])
        self.assertEqual(compile_kwargs["is_backward"], is_backward)
        self.assertEqual(compile_kwargs["is_inference"], is_inference)

    def test_invoke_subgraph_compile_shares_paired_cudagraph_state(self):
        def compiled_fn(args):
            return args

        compiled_fn._boxed_call = True
        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True},
            bw_inductor_config_patches={"triton.cudagraphs": False},
        )
        if nested_config.fw_compiler is None or nested_config.bw_compiler is None:
            raise AssertionError("expected forward and backward compilers")

        with mock.patch(
            "torch._inductor.compile_fx.compile_fx_inner",
            return_value=compiled_fn,
        ) as compile_fx_inner:
            forward_artifact = nested_config.fw_compiler(
                self._empty_graph_module(), [], cudagraph_state_key=(0, 0)
            )
            forward_kwargs = compile_fx_inner.call_args.kwargs
            forward_kwargs["boxed_forward_device_index"].set(0)
            nested_config.bw_compiler(
                self._empty_graph_module(), [], cudagraph_state_key=(0, 0)
            )

        backward_kwargs = compile_fx_inner.call_args.kwargs
        self.assertIs(
            backward_kwargs["boxed_forward_device_index"],
            forward_kwargs["boxed_forward_device_index"],
        )
        self.assertTrue(backward_kwargs["cudagraphs_forward_enabled"])
        self.assertFalse(backward_kwargs["cudagraphs"])
        self.assertIsNotNone(forward_artifact)

    def test_invoke_subgraph_compile_keeps_cudagraph_state_per_call(self):
        from torch._inductor.utils import BoxedBool

        def compiled_fn(args):
            return args

        compiled_fn._boxed_call = True
        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True},
            bw_inductor_config_patches={"triton.cudagraphs": True},
        )
        if nested_config.fw_compiler is None or nested_config.bw_compiler is None:
            raise AssertionError("expected forward and backward compilers")

        with mock.patch(
            "torch._inductor.compile_fx.compile_fx_inner",
            return_value=compiled_fn,
        ) as compile_fx_inner:
            forward_artifacts = [
                nested_config.fw_compiler(
                    self._empty_graph_module(), [], cudagraph_state_key=(0, 0)
                )
            ]
            first_forward = compile_fx_inner.call_args.kwargs
            BoxedBool.disable(first_forward["cudagraphs"])
            forward_artifacts.append(
                nested_config.fw_compiler(
                    self._empty_graph_module(), [], cudagraph_state_key=(0, 1)
                )
            )
            nested_config.bw_compiler(
                self._empty_graph_module(), [], cudagraph_state_key=(0, 0)
            )
            first_backward = compile_fx_inner.call_args.kwargs
            nested_config.bw_compiler(
                self._empty_graph_module(), [], cudagraph_state_key=(0, 1)
            )
            second_backward = compile_fx_inner.call_args.kwargs

        self.assertFalse(first_backward["cudagraphs"])
        self.assertTrue(second_backward["cudagraphs"])
        self.assertEqual(len(forward_artifacts), 2)

    def test_invoke_subgraph_cudagraph_state_does_not_leak(self):
        from torch._higher_order_ops.invoke_subgraph import (
            _invoke_subgraph_cudagraph_states,
        )

        def compiled_fn(args):
            return args

        compiled_fn._boxed_call = True
        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True}
        )
        if nested_config.fw_compiler is None:
            raise AssertionError("expected a forward compiler")
        state_key = object()

        with mock.patch(
            "torch._inductor.compile_fx.compile_fx_inner",
            return_value=compiled_fn,
        ):
            artifact = nested_config.fw_compiler(
                self._empty_graph_module(), [], cudagraph_state_key=state_key
            )
            self.assertIn(state_key, _invoke_subgraph_cudagraph_states)
            del artifact
            gc.collect()

        self.assertNotIn(state_key, _invoke_subgraph_cudagraph_states)

    def test_backward_config_specialization_reaches_child_graph_modules(self):
        from torch._higher_order_ops.invoke_subgraph import (
            _specialize_nested_region_configs_for_backward,
        )

        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True},
            bw_inductor_config_patches={"triton.cudagraphs": False},
        )
        child = self._configured_region_graph_module(nested_config)
        root = torch.nn.Module()
        root.add_module("child", child)
        graph = torch.fx.Graph()
        graph.call_module("child")
        graph.output(())
        gm = torch.fx.GraphModule(root, graph)

        _specialize_nested_region_configs_for_backward(gm)

        region = child.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.invoke_subgraph
        )[0]
        backward_config = region.meta["custom"]["nested_region_config"]
        self.assertFalse(backward_config.inductor_config_patches["triton.cudagraphs"])
        self.assertIsNone(backward_config.bw_inductor_config_patches)

    def test_backward_config_specialization_mirrors_without_clobbering(self):
        """Seed a fresh module mirror, but leave one shared with the forward alone."""
        from torch._higher_order_ops.invoke_subgraph import (
            _specialize_nested_region_configs_for_backward,
        )

        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True},
            bw_inductor_config_patches={"triton.cudagraphs": False},
        )
        shared = self._empty_graph_module()
        shared.meta["nested_region_config"] = nested_config
        fresh = self._empty_graph_module()
        root = torch.nn.Module()
        root.add_module("shared", shared)
        root.add_module("submod", torch.nn.Module())
        root.submod.add_module("fresh", fresh)
        graph = torch.fx.Graph()
        for target in ("shared", "submod.fresh"):
            region = graph.call_function(
                torch.ops.higher_order.invoke_subgraph, (graph.get_attr(target),)
            )
            region.meta["custom"] = {"nested_region_config": nested_config}
        graph.output(())
        gm = torch.fx.GraphModule(root, graph)

        _specialize_nested_region_configs_for_backward(gm)

        for region in gm.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.invoke_subgraph
        ):
            node_config = region.meta["custom"]["nested_region_config"]
            self.assertFalse(node_config.inductor_config_patches["triton.cudagraphs"])
        # The dotted target resolves, so the fresh mirror is seeded...
        self.assertFalse(
            fresh.meta["nested_region_config"].inductor_config_patches[
                "triton.cudagraphs"
            ]
        )
        # ...while a mirror the forward may still compile under is left alone.
        self.assertIs(shared.meta["nested_region_config"], nested_config)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "triton.cudagraphs": False,
        }
    )
    @parametrize(
        "forward_cudagraphs,backward_cudagraphs",
        ((False, False), (False, True), (True, False), (True, True)),
    )
    def test_regional_inductor_backend_runs_cudagraphs(
        self, forward_cudagraphs, backward_cudagraphs
    ):
        from torch._dynamo.backends.common import aot_autograd
        from torch._inductor import cudagraph_trees
        from torch._inductor.compile_fx import cudagraphify
        from torch.fx.passes.regional_inductor_invoke_subgraph import (
            regional_inductor_invoke_subgraph,
        )

        cudagraph_trees.reset_cudagraph_trees()
        self.addCleanup(cudagraph_trees.reset_cudagraph_trees)

        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": forward_cudagraphs},
            bw_inductor_config_patches={"triton.cudagraphs": backward_cudagraphs},
        )

        @torch.compiler.nested_compile_region(options=nested_config)
        def region(x):
            return torch.sin(x)

        def fn(x):
            detached = region(x).detach()
            return torch.cos(region(x)).sum() + detached.sum()

        def backend(gm, example_inputs):
            return regional_inductor_invoke_subgraph(gm, *example_inputs)

        with mock.patch(
            "torch._inductor.compile_fx.cudagraphify", wraps=cudagraphify
        ) as cudagraphify_mock:
            compiled_fn = torch.compile(
                fn,
                backend=aot_autograd(fw_compiler=backend, bw_compiler=backend),
                fullgraph=True,
            )
            for _ in range(3):
                x = torch.randn(16, 16, device=GPU_TYPE, requires_grad=True)
                torch.compiler.cudagraph_mark_step_begin()
                result = compiled_fn(x)
                result.backward()

        expected_x = x.detach().clone().requires_grad_()
        expected = fn(expected_x)
        expected.backward()
        self.assertEqual(result, expected)
        self.assertEqual(x.grad, expected_x.grad)
        expected_directions = set()
        if forward_cudagraphs:
            expected_directions.add(False)
        if backward_cudagraphs:
            expected_directions.add(True)
        self.assertEqual(
            {call.kwargs["is_backward"] for call in cudagraphify_mock.call_args_list},
            expected_directions,
        )
        if x.device.index is None:
            raise AssertionError("expected a CUDA device index")
        manager = cudagraph_trees.get_container(x.device.index).tree_manager
        if forward_cudagraphs or backward_cudagraphs:
            if manager is None:
                self.fail("expected CUDA Graph recording for the regional backend")
            self.assertGreaterEqual(
                manager.new_graph_id().id,
                int(forward_cudagraphs) + int(backward_cudagraphs),
            )
            self.assertFalse(manager.running_forwards_with_pending_backwards)
        else:
            self.assertIsNone(manager)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "triton.cudagraphs": False,
        }
    )
    @parametrize("backward_cudagraphs", (False, True))
    def test_regional_inductor_backend_backward_generation_transition(
        self, backward_cudagraphs
    ):
        """A captured backward region transitions the tree from its own node.

        Forcing the manager into BACKWARD mode before it runs would let that
        node start a new generation and free the forward pool it reads from.
        """
        from torch._dynamo.backends.common import aot_autograd
        from torch._inductor import cudagraph_trees
        from torch._inductor.cudagraph_trees import CUDAGraphTreeManager
        from torch.fx.passes.regional_inductor_invoke_subgraph import (
            regional_inductor_invoke_subgraph,
        )

        cudagraph_trees.reset_cudagraph_trees()
        self.addCleanup(cudagraph_trees.reset_cudagraph_trees)

        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True},
            bw_inductor_config_patches={"triton.cudagraphs": backward_cudagraphs},
        )

        @torch.compiler.nested_compile_region(options=nested_config)
        def region(x):
            # exp saves its output, so the backward reads from the forward pool.
            return torch.exp(x) * 2.0

        def fn(x):
            return region(x).sum()

        def backend(gm, example_inputs):
            return regional_inductor_invoke_subgraph(gm, *example_inputs)

        compiled_fn = torch.compile(
            fn,
            backend=aot_autograd(fw_compiler=backend, bw_compiler=backend),
            fullgraph=True,
        )
        with mock.patch.object(
            CUDAGraphTreeManager,
            "set_to_running_backward",
            side_effect=CUDAGraphTreeManager.set_to_running_backward,
            autospec=True,
        ) as set_to_running_backward:
            for _ in range(3):
                x = torch.randn(16, 16, device="cuda", requires_grad=True)
                torch.compiler.cudagraph_mark_step_begin()
                compiled_fn(x).backward()
                expected_x = x.detach().clone().requires_grad_()
                fn(expected_x).backward()
                self.assertEqual(x.grad, expected_x.grad)

        # With the backward region captured nobody may get ahead of its own
        # transition. Otherwise exactly two callers transition per backward: the
        # enclosing eager backward here, and the uncaptured backward region's own
        # post_compile wrapper.
        self.assertEqual(
            set_to_running_backward.call_count, 0 if backward_cudagraphs else 2 * 3
        )

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
            "graph_partition": False,
            "triton.cudagraphs": False,
        }
    )
    @skip_if_caches_force_disabled
    def test_nested_region_cudagraph_config_reuses_fx_graph_cache(self):
        from torch._dynamo.utils import counters
        from torch._inductor import cudagraph_trees
        from torch._inductor.codecache import FxGraphCache

        FxGraphCache.clear()
        self.addCleanup(FxGraphCache.clear)
        cudagraph_trees.reset_cudagraph_trees()
        self.addCleanup(cudagraph_trees.reset_cudagraph_trees)
        counters.clear()

        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True}
        )

        @torch.compiler.nested_compile_region(options=options)
        def region(x):
            return torch.sin(x)

        def fn(x):
            return torch.cos(region(x))

        x = torch.randn(16, 16, device=GPU_TYPE)
        for _ in range(2):
            torch._dynamo.reset()
            result = torch.compile(fn, backend="inductor", fullgraph=True)(x).clone()

        self.assertEqual(result, fn(x))
        self.assertGreaterEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
            "graph_partition": False,
            "triton.cudagraphs": False,
        }
    )
    @skip_if_caches_force_disabled
    def test_backward_opt_out_reuses_aot_autograd_cache(self):
        """A warm start reuses the forward's box; the backward opted out of it."""
        from torch._dynamo.utils import counters
        from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
        from torch._inductor import cudagraph_trees
        from torch._inductor.compile_fx import cudagraphify

        AOTAutogradCache.clear()
        self.addCleanup(AOTAutogradCache.clear)
        cudagraph_trees.reset_cudagraph_trees()
        self.addCleanup(cudagraph_trees.reset_cudagraph_trees)

        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True},
            bw_inductor_config_patches={"triton.cudagraphs": False},
        )

        @torch.compiler.nested_compile_region(options=options)
        def region(x):
            return torch.sin(x)

        def fn(x):
            return (torch.cos(x) + region(x)).sum()

        with mock.patch(
            "torch._inductor.compile_fx.cudagraphify", wraps=cudagraphify
        ) as cudagraphify_mock:
            for _ in range(2):
                counters.clear()
                cudagraphify_mock.reset_mock()
                torch._dynamo.reset()
                x = torch.randn(16, 16, device="cuda", requires_grad=True)
                torch.compile(fn, backend="inductor", fullgraph=True)(x).backward()

        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 0)
        self.assertEqual(
            {call.kwargs["is_backward"] for call in cudagraphify_mock.call_args_list},
            {False},
        )

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
        }
    )
    @parametrize(
        "global_cudagraphs,regional_cudagraphs",
        ((False, False), (False, True), (True, False), (True, True)),
    )
    def test_regional_inductor_backend_inference_uses_forward_cudagraphs(
        self, global_cudagraphs, regional_cudagraphs
    ):
        from torch._dynamo.backends.common import aot_autograd
        from torch._inductor import cudagraph_trees
        from torch._inductor.compile_fx import cudagraphify
        from torch.fx.passes.regional_inductor_invoke_subgraph import (
            regional_inductor_invoke_subgraph,
        )

        cudagraph_trees.reset_cudagraph_trees()
        self.addCleanup(cudagraph_trees.reset_cudagraph_trees)

        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": regional_cudagraphs},
            bw_inductor_config_patches={"triton.cudagraphs": False},
        )

        @torch.compiler.nested_compile_region(options=nested_config)
        def region(x):
            return torch.sin(x)

        def fn(x):
            return torch.cos(region(x))

        def backend(gm, example_inputs):
            return regional_inductor_invoke_subgraph(gm, *example_inputs)

        with (
            torch._inductor.config.patch("triton.cudagraphs", global_cudagraphs),
            mock.patch(
                "torch._inductor.compile_fx.cudagraphify", wraps=cudagraphify
            ) as cudagraphify_mock,
        ):
            compiled_fn = torch.compile(
                fn,
                backend=aot_autograd(fw_compiler=backend, inference_compiler=backend),
                fullgraph=True,
            )
            for _ in range(3):
                x = torch.randn(16, 16, device=GPU_TYPE)
                torch.compiler.cudagraph_mark_step_begin()
                result = compiled_fn(x)

        self.assertEqual(result, fn(x))
        self.assertEqual(len(cudagraphify_mock.call_args_list) > 0, regional_cudagraphs)
        for call in cudagraphify_mock.call_args_list:
            self.assertFalse(call.kwargs["is_backward"])
            self.assertTrue(call.kwargs["is_inference"])
        if x.device.index is None:
            raise AssertionError("expected a CUDA device index")
        manager = cudagraph_trees.get_container(x.device.index).tree_manager
        if regional_cudagraphs:
            if manager is None:
                self.fail("expected CUDA Graph recording for regional inference")
            self.assertGreater(manager.new_graph_id().id, 0)
        else:
            self.assertIsNone(manager)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "triton.cudagraphs": False,
        }
    )
    def test_regional_inductor_backend_unpaired_forward_cudagraph(self):
        from torch._dynamo.backends.common import aot_autograd
        from torch._inductor import cudagraph_trees
        from torch.fx.passes.regional_inductor_invoke_subgraph import (
            regional_inductor_invoke_subgraph,
        )

        cudagraph_trees.reset_cudagraph_trees()
        self.addCleanup(cudagraph_trees.reset_cudagraph_trees)

        forward_options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True},
            bw_inductor_config_patches={"triton.cudagraphs": False},
        )

        @torch.compiler.nested_compile_region(options=forward_options)
        def forward_only_region(x):
            return torch.sin(x)

        def fn(x):
            forward_only = forward_only_region(x).detach()
            return torch.cos(x).sum() + forward_only.sum()

        def backend(gm, example_inputs):
            return regional_inductor_invoke_subgraph(gm, *example_inputs)

        compiled_fn = torch.compile(
            fn,
            backend=aot_autograd(fw_compiler=backend, bw_compiler=backend),
            fullgraph=True,
        )
        for _ in range(3):
            x = torch.randn(16, 16, device=GPU_TYPE, requires_grad=True)
            torch.compiler.cudagraph_mark_step_begin()
            result = compiled_fn(x)
            result.backward()

        expected_x = x.detach().clone().requires_grad_()
        expected = fn(expected_x)
        expected.backward()
        self.assertEqual(result, expected)
        self.assertEqual(x.grad, expected_x.grad)
        if x.device.index is None:
            raise AssertionError("expected a CUDA device index")
        manager = cudagraph_trees.get_container(x.device.index).tree_manager
        if manager is None:
            self.fail("expected CUDA Graph recording for the forward-only region")
        self.assertFalse(manager.running_forwards_with_pending_backwards)

    @requires_cuda_and_triton
    @unittest.skipUnless(TEST_MULTIGPU, "requires multiple cuda devices")
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "triton.cudagraphs": False,
        }
    )
    def test_regional_inductor_backend_unpaired_forward_on_another_device(self):
        from torch._dynamo.backends.common import aot_autograd
        from torch._inductor import cudagraph_trees
        from torch.fx.passes.regional_inductor_invoke_subgraph import (
            regional_inductor_invoke_subgraph,
        )

        cudagraph_trees.reset_cudagraph_trees()
        self.addCleanup(cudagraph_trees.reset_cudagraph_trees)

        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True},
            bw_inductor_config_patches={"triton.cudagraphs": False},
        )

        @torch.compiler.nested_compile_region(options=options)
        def forward_only_region(x):
            return torch.sin(x)

        def fn(x0, x1):
            return forward_only_region(x0).detach(), torch.cos(x1)

        def backend(gm, example_inputs):
            return regional_inductor_invoke_subgraph(gm, *example_inputs)

        compiled_fn = torch.compile(
            fn,
            backend=aot_autograd(fw_compiler=backend, bw_compiler=backend),
            fullgraph=True,
        )
        for _ in range(3):
            x0 = torch.randn(16, 16, device="cuda:0", requires_grad=True)
            x1 = torch.randn(16, 16, device="cuda:1", requires_grad=True)
            torch.compiler.cudagraph_mark_step_begin()
            _, differentiable = compiled_fn(x0, x1)
            differentiable.sum().backward()

        manager = cudagraph_trees.get_container(0).tree_manager
        if manager is None:
            self.fail("expected CUDA Graph recording on cuda:0")
        self.assertFalse(manager.running_forwards_with_pending_backwards)

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch("triton.cudagraphs", False)
    def test_regional_inductor_backend_rejects_nested_cudagraph_conflict(self):
        from torch._dynamo.backends.common import aot_autograd
        from torch.fx.passes.regional_inductor_invoke_subgraph import (
            regional_inductor_invoke_subgraph,
        )

        inner_options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": False}
        )
        outer_options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True}
        )

        @torch.compiler.nested_compile_region(options=inner_options)
        def inner_region(x):
            return torch.sin(x)

        @torch.compiler.nested_compile_region(options=outer_options)
        def outer_region(x):
            return torch.cos(inner_region(x))

        def backend(gm, example_inputs):
            return regional_inductor_invoke_subgraph(gm, *example_inputs)

        with self.assertRaisesRegex(
            RuntimeError,
            "nested compile regions cannot have conflicting cudagraph configs",
        ):
            torch.compile(
                outer_region,
                backend=aot_autograd(fw_compiler=backend, inference_compiler=backend),
                fullgraph=True,
            )(torch.randn(4))

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "triton.cudagraphs": False,
        }
    )
    @parametrize(
        "forward_cudagraphs,backward_cudagraphs",
        ((False, True), (True, False)),
    )
    def test_regional_inductor_backend_accepts_nested_directional_config(
        self, forward_cudagraphs, backward_cudagraphs
    ):
        from torch._dynamo.backends.common import aot_autograd
        from torch.fx.passes.regional_inductor_invoke_subgraph import (
            regional_inductor_invoke_subgraph,
        )

        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": forward_cudagraphs},
            bw_inductor_config_patches={"triton.cudagraphs": backward_cudagraphs},
        )

        @torch.compiler.nested_compile_region(options=options)
        def inner_region(x):
            return torch.sin(x)

        @torch.compiler.nested_compile_region(options=options)
        def outer_region(x):
            return torch.cos(inner_region(x))

        def backend(gm, example_inputs):
            return regional_inductor_invoke_subgraph(gm, *example_inputs)

        compiled_fn = torch.compile(
            outer_region,
            backend=aot_autograd(fw_compiler=backend, bw_compiler=backend),
            fullgraph=True,
        )
        x = torch.randn(16, 16, device=GPU_TYPE, requires_grad=True)
        result = compiled_fn(x)
        result.sum().backward()

        expected_x = x.detach().clone().requires_grad_()
        expected = outer_region(expected_x)
        expected.sum().backward()
        self.assertEqual(result, expected)
        self.assertEqual(x.grad, expected_x.grad)

    def test_post_compile_uses_current_cudagraph_state(self):
        from torch._inductor.output_code import CompiledFxGraph
        from torch._inductor.utils import BoxedBool

        compiled_graph = mock.Mock(spec=CompiledFxGraph)
        compiled_graph.partition_maps = None
        compiled_graph.fx_kwargs = {
            "cudagraphs": BoxedBool(True),
            "cudagraphs_region_aware": True,
            "is_backward": False,
            "is_inference": True,
        }
        compiled_graph.disabled_cudagraphs_reason = None
        compiled_graph.device_types = {"cuda"}
        compiled_graph.inputs_to_check = ()
        compiled_graph.mutated_input_idxs = set()
        compiled_graph._original_gm = object()
        compiled_graph._serialized_original_gm = None
        compiled_graph._wrap_compiled_regions = False

        graph_kwargs = {
            "cudagraphs": BoxedBool(False),
            "is_backward": False,
        }
        with (
            mock.patch(
                "torch._inductor.output_code.set_tracing_context_output_strides"
            ),
            mock.patch("torch._inductor.output_code.maybe_realign_inputs"),
            mock.patch(
                "torch._inductor.output_code.cudagraph_post_compile"
            ) as post_compile,
        ):
            CompiledFxGraph.post_compile(compiled_graph, [], mock.Mock(), graph_kwargs)

        post_compile.assert_not_called()

    def test_disabled_backward_transitions_forward_cudagraph_generation(self):
        from torch._inductor.cudagraph_utils import BoxedDeviceIndex
        from torch._inductor.output_code import CompiledFxGraph
        from torch._inductor.utils import BoxedBool

        compiled_graph = mock.Mock(spec=CompiledFxGraph)
        compiled_graph.partition_maps = None
        compiled_graph.fx_kwargs = {
            "is_backward": True,
            "is_inference": False,
        }
        compiled_graph.inputs_to_check = ()
        compiled_graph.mutated_input_idxs = set()
        compiled_graph._original_gm = object()
        compiled_graph._serialized_original_gm = None
        compiled_graph._wrap_compiled_regions = False
        forward_device_index = BoxedDeviceIndex(0)
        graph_kwargs = {
            "cudagraphs": BoxedBool(False),
            "is_backward": True,
            "boxed_forward_device_index": forward_device_index,
            "cudagraphs_forward_enabled": True,
        }

        with (
            mock.patch(
                "torch._inductor.output_code.set_tracing_context_output_strides"
            ),
            mock.patch("torch._inductor.output_code.maybe_realign_inputs"),
            mock.patch(
                "torch._inductor.output_code.maybe_handle_backward_generation"
            ) as handle_backward,
        ):
            CompiledFxGraph.post_compile(compiled_graph, [], mock.Mock(), graph_kwargs)

        handle_backward.assert_called_once_with(
            compiled_graph,
            forward_device_index,
            forward_cudagraphs_enabled=True,
        )

    @torch._inductor.config.patch("triton.cudagraph_trees", True)
    def test_backward_generation_keeps_forward_state_invariants(self):
        from torch._inductor.output_code import maybe_handle_backward_generation

        compiled_graph = mock.Mock()
        compiled_graph.current_callable = lambda args: args
        compiled_graph.fx_kwargs = {
            "is_backward": True,
            "cudagraphs_region_aware": True,
        }

        with self.assertRaisesRegex(
            AssertionError, "boxed_forward_device_index must not be None"
        ):
            maybe_handle_backward_generation(compiled_graph, None)

    @torch._inductor.config.patch(
        {"graph_partition": False, "triton.cudagraphs": False}
    )
    def test_backward_does_not_override_forward_cudagraph_disable(self):
        from torch._higher_order_ops.invoke_subgraph import (
            get_backward_nested_region_config,
        )
        from torch._inductor.compile_fx import (
            compile_fx_backward,
            create_compiler_config_extra,
        )
        from torch._inductor.utils import BoxedBool

        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True},
            bw_inductor_config_patches={"triton.cudagraphs": True},
        )
        compiler_config = create_compiler_config_extra(
            self._configured_region_graph_module(nested_config)
        )
        BoxedBool.disable(compiler_config.cudagraphs)

        backward_config = get_backward_nested_region_config(nested_config)
        if backward_config is None:
            raise AssertionError("expected a backward region config")
        captured_kwargs = {}

        def inner_compile(gm, example_inputs, **kwargs):
            captured_kwargs.update(kwargs)
            return mock.sentinel.compiled

        compile_fx_backward(
            self._configured_region_graph_module(backward_config),
            [],
            compiler_config,
            inner_compile,
        )

        self.assertFalse(captured_kwargs["cudagraphs"])

    @torch._inductor.config.patch(
        {"graph_partition": False, "triton.cudagraphs": False}
    )
    def test_backward_only_regional_cudagraph_can_enable(self):
        from torch._higher_order_ops.invoke_subgraph import (
            get_backward_nested_region_config,
        )
        from torch._inductor.compile_fx import (
            compile_fx_backward,
            create_compiler_config_extra,
        )

        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": False},
            bw_inductor_config_patches={"triton.cudagraphs": True},
        )
        compiler_config = create_compiler_config_extra(
            self._configured_region_graph_module(nested_config)
        )
        backward_config = get_backward_nested_region_config(nested_config)
        if backward_config is None:
            raise AssertionError("expected a backward region config")
        captured_kwargs = {}

        def inner_compile(gm, example_inputs, **kwargs):
            captured_kwargs.update(kwargs)
            return mock.sentinel.compiled

        compile_fx_backward(
            self._configured_region_graph_module(backward_config),
            [],
            compiler_config,
            inner_compile,
        )

        self.assertTrue(captured_kwargs["cudagraphs"])
        self.assertFalse(captured_kwargs["cudagraphs_forward_enabled"])

    @torch._inductor.config.patch("triton.cudagraphs", False)
    def test_redundant_cudagraph_override_is_not_regional(self):
        from torch._inductor import _CudagraphAnnotation
        from torch._inductor.compile_fx import create_compiler_config_extra

        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True},
            bw_inductor_config_patches={"triton.cudagraphs": True},
        )
        gm = self._configured_region_graph_module(nested_config)
        gm.meta["cudagraph_annotation"] = _CudagraphAnnotation(fwd=True, bwd=True)

        compiler_config = create_compiler_config_extra(gm)

        self.assertFalse(compiler_config.has_regional_cudagraphs)
        self.assertTrue(compiler_config.top_level_cudagraphs)

    @requires_gpu_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": True,
            "max_autotune_gemm_backends": "TRITON",
            "test_configs.max_mm_configs": 1,
            "triton.cudagraphs": False,
        }
    )
    @parametrize("parent_max_autotune", (False, True))
    @parametrize("nested_max_autotune", (False, True))
    def test_nested_region_inductor_config_max_autotune(
        self, parent_max_autotune, nested_max_autotune
    ):
        """Check GEMM backend selection in both forward and backward code."""
        if not IS_BIG_GPU:
            self.skipTest("requires a GPU with Triton GEMM template support")

        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"max_autotune": nested_max_autotune},
            bw_inductor_config_patches={"max_autotune": nested_max_autotune},
        )

        @torch.compiler.nested_compile_region(options=nested_config)
        def region(x, y):
            return x @ y

        def fn(x, y, a, b):
            # Keep the otherwise identical regional and parent GEMMs separate.
            return torch.cat((region(x, y), a @ b))

        inputs = [
            torch.randn(128, 128, device=GPU_TYPE, requires_grad=True) for _ in range(4)
        ]
        with torch.no_grad():
            expected = fn(*inputs)

        with torch._inductor.config.patch(max_autotune=parent_max_autotune):
            result, codes = run_fw_bw_and_get_code(
                lambda: torch.compile(fn, backend="inductor", fullgraph=True)(*inputs)
            )

        # Triton GEMM and hipBLAS sum fp32 products in different orders; both are
        # equally close to fp64, but differ by up to ~4e-4 relative on ROCm.
        if TEST_WITH_ROCM:
            self.assertEqual(result, expected, atol=1e-5, rtol=1e-3)
        else:
            self.assertEqual(result, expected)
        self.assertEqual(len(codes), 2)
        fw_code, bw_code = codes
        regions_and_settings = (
            (
                self._generated_fn_body(fw_code, "def partitioned_fw_subgraph_0_0("),
                nested_max_autotune,
            ),
            (
                self._generated_fn_body(fw_code, "    def call(self, args):"),
                parent_max_autotune,
            ),
            (
                self._generated_fn_body(bw_code, "def partitioned_bw_subgraph_0_0("),
                nested_max_autotune,
            ),
            (
                self._generated_fn_body(bw_code, "    def call(self, args):"),
                parent_max_autotune,
            ),
        )
        for code, max_autotune in regions_and_settings:
            expected_backend = "triton_tem_" if max_autotune else "extern_kernels.mm"
            unexpected_backend = "extern_kernels.mm" if max_autotune else "triton_tem_"
            self.assertIn(expected_backend, code)
            self.assertNotIn(unexpected_backend, code)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": False,
            "triton.cudagraphs": False,
        }
    )
    @parametrize(
        "forward_cudagraphs,backward_cudagraphs",
        ((False, False), (False, True), (True, False), (True, True)),
    )
    def test_nested_region_inductor_config_cudagraphs_forces_partition(
        self, forward_cudagraphs, backward_cudagraphs
    ):
        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": forward_cudagraphs},
            bw_inductor_config_patches={"triton.cudagraphs": backward_cudagraphs},
        )

        @torch.compiler.nested_compile_region(options=nested_config)
        def region(x):
            return torch.sin(x)

        def fn(x):
            return (torch.cos(x) + region(x)).sum()

        compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        x = torch.randn(16, 16, device=GPU_TYPE, requires_grad=True)
        with torch.no_grad():
            expected = fn(x)

        from torch._inductor.output_code import cudagraph_partition_post_compile

        with mock.patch(
            "torch._inductor.output_code.cudagraph_partition_post_compile",
            wraps=cudagraph_partition_post_compile,
        ) as partition_post_compile:
            result, codes = run_fw_bw_and_get_code(lambda: compiled_fn(x))

        self.assertEqual(result, expected)
        self.assertEqual(len(codes), 2)
        self.assertEqual(
            partition_post_compile.call_count,
            int(forward_cudagraphs) + int(backward_cudagraphs),
        )
        for code, subgraph_name, use_cudagraphs in zip(
            codes,
            ("partitioned_fw_subgraph_0_0(", "partitioned_bw_subgraph_0_0("),
            (forward_cudagraphs, backward_cudagraphs),
        ):
            if use_cudagraphs:
                partition = self._generated_fn_body(code, "def partition_0(args):")
                self.assertIn(subgraph_name, partition)
            else:
                self.assertNotIn("def partition_0(args):", code)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": False,
        }
    )
    @parametrize(
        "global_cudagraphs,regional_cudagraphs",
        ((False, False), (False, True), (True, False), (True, True)),
    )
    def test_nested_region_cudagraphs_independent_from_global_config(
        self, global_cudagraphs, regional_cudagraphs
    ):
        from torch._inductor import cudagraph_trees

        cudagraph_trees.reset_cudagraph_trees()
        self.addCleanup(cudagraph_trees.reset_cudagraph_trees)

        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": regional_cudagraphs}
        )

        @torch.compiler.nested_compile_region(options=nested_config)
        def region(x):
            return torch.sin(x)

        def fn(x):
            return (torch.cos(x) + region(x)).sum()

        x = torch.randn(16, 16, device=GPU_TYPE)
        expected = fn(x)

        from torch._inductor.output_code import (
            cudagraph_partition_post_compile,
            cudagraph_post_compile,
        )

        with (
            torch._inductor.config.patch("triton.cudagraphs", global_cudagraphs),
            mock.patch(
                "torch._inductor.output_code.cudagraph_partition_post_compile",
                wraps=cudagraph_partition_post_compile,
            ) as partition_post_compile,
            mock.patch(
                "torch._inductor.output_code.cudagraph_post_compile",
                wraps=cudagraph_post_compile,
            ) as whole_graph_post_compile,
        ):
            compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
            result, codes = run_and_get_code(compiled_fn, x)
            result = result.clone()
            torch.compiler.cudagraph_mark_step_begin()
            replayed_result = compiled_fn(x).clone()

        self.assertEqual(result, expected)
        self.assertEqual(replayed_result, expected)
        self.assertEqual(len(codes), 1)
        settings_differ = global_cudagraphs != regional_cudagraphs
        self.assertEqual(partition_post_compile.call_count, int(settings_differ))
        self.assertEqual(
            whole_graph_post_compile.call_count,
            int(global_cudagraphs and regional_cudagraphs),
        )
        if settings_differ:
            partition_bodies = []
            partition_index = 0
            while (signature := f"def partition_{partition_index}(args):") in codes[0]:
                partition_bodies.append(self._generated_fn_body(codes[0], signature))
                partition_index += 1
            self.assertGreater(len(partition_bodies), 0)
            self.assertEqual(
                any("repeated_subgraph0(" in body for body in partition_bodies),
                regional_cudagraphs,
            )
        else:
            self.assertNotIn("def partition_0(args):", codes[0])

        if x.device.index is None:
            raise AssertionError("expected a CUDA device index")
        manager = cudagraph_trees.get_container(x.device.index).tree_manager
        if global_cudagraphs or regional_cudagraphs:
            if manager is None:
                self.fail("expected CUDA Graph recording for enabled work")
            self.assertGreater(manager.new_graph_id().id, 0)
        else:
            self.assertIsNone(manager)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": False,
            "triton.cudagraphs": True,
        }
    )
    def test_region_opt_out_keeps_enclosing_cpu_op_disable(self):
        """An opt-out must not widen capture: graph_partition is still off here.

        Forcing partitioning for the region would otherwise let the enclosing
        graph split around its CPU op and capture what graph_partition=False
        says must stay uncaptured -- including the backward, which shares the
        forward's box.
        """
        from torch._inductor import cudagraph_trees
        from torch._inductor.compile_fx import cudagraphify

        cudagraph_trees.reset_cudagraph_trees()
        self.addCleanup(cudagraph_trees.reset_cudagraph_trees)

        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": False}
        )

        @torch.compiler.nested_compile_region(options=options)
        def region(x):
            return torch.sin(x)

        def fn(x, weight):
            y = (x @ weight).cpu().to("cuda")
            return (region(y) + torch.cos(y)).sum()

        x = torch.randn(16, 16, device="cuda", requires_grad=True)
        weight = torch.randn(16, 16, device="cuda", requires_grad=True)
        compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        with mock.patch(
            "torch._inductor.compile_fx.cudagraphify", wraps=cudagraphify
        ) as cudagraphify_mock:
            result, codes = run_fw_bw_and_get_code(lambda: compiled_fn(x, weight))

        cudagraphify_mock.assert_not_called()
        # Positive controls: the region really is still a region (so the opt-out
        # had something to act on), and the model still computes the right thing.
        self.assertEqual(len(codes), 2)
        for code, subgraph_name in zip(
            codes, ("partitioned_fw_subgraph_0_0(", "partitioned_bw_subgraph_0_0(")
        ):
            self.assertIn(subgraph_name, code)
        expected_x = x.detach().clone().requires_grad_()
        expected_weight = weight.detach().clone().requires_grad_()
        expected = fn(expected_x, expected_weight)
        expected.backward()
        self.assertEqual(result, expected)
        self.assertEqual(x.grad, expected_x.grad)
        self.assertEqual(weight.grad, expected_weight.grad)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": True,
            "triton.cudagraphs": True,
        }
    )
    def test_backward_lowers_under_its_own_patches_with_stale_mirror(self):
        """The HOP node meta wins over the subgraph module's config mirror.

        The partitioner can hand the forward and the backward the same nested
        module object, so when the backward is lowered that mirror may still
        hold the forward's config. Simulate that by restoring the forward
        config on every region body after the backward is specialized.
        """
        from torch._higher_order_ops.invoke_subgraph import (
            _specialize_nested_region_configs_for_backward as specialize,
        )
        from torch._inductor.compile_fx import _get_invoke_subgraph_graph_module

        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True},
            bw_inductor_config_patches={"triton.cudagraphs": False},
        )

        def specialize_then_stale_mirror(gm):
            specialize(gm)
            # gm is the partitioned backward body; leave the forward config on
            # its mirror, which is what a module shared with the forward looks
            # like by the time the backward is lowered.
            gm.meta["nested_region_config"] = options
            for module in gm.modules():
                if not isinstance(module, torch.fx.GraphModule):
                    continue
                for node in module.graph.find_nodes(
                    op="call_function",
                    target=torch.ops.higher_order.invoke_subgraph,
                ):
                    body = _get_invoke_subgraph_graph_module(module, node)
                    if body is not None:
                        body.meta["nested_region_config"] = options

        @torch.compiler.nested_compile_region(options=options)
        def region(x):
            return torch.sin(x)

        def fn(x):
            return (torch.cos(x) + region(x)).sum()

        x = torch.randn(16, 16, device="cuda", requires_grad=True)
        compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        # Patch the module object, not the dotted string: the package re-exports
        # the HOP under the submodule's own name, so on Python < 3.12 mock's
        # getattr-first resolution of "torch._higher_order_ops.invoke_subgraph"
        # lands on InvokeSubgraphHOP and the patch fails with AttributeError.
        invoke_subgraph_module = importlib.import_module(
            "torch._higher_order_ops.invoke_subgraph"
        )
        with mock.patch.object(
            invoke_subgraph_module,
            "_specialize_nested_region_configs_for_backward",
            specialize_then_stale_mirror,
        ):
            result, codes = run_fw_bw_and_get_code(lambda: compiled_fn(x))

        expected_x = x.detach().clone().requires_grad_()
        expected = fn(expected_x)
        expected.backward()
        self.assertEqual(result, expected)
        self.assertEqual(x.grad, expected_x.grad)
        self.assertEqual(len(codes), 2)
        # The forward agrees with the top-level setting, so its region is
        # captured inside a partition; the backward opts out, so its region has
        # to stay in call() no matter what the stale mirror says.
        forward_partition = self._generated_fn_body(codes[0], "def partition_0(args):")
        self.assertIn("partitioned_fw_subgraph_0_0(", forward_partition)
        partition_index = 0
        while (signature := f"def partition_{partition_index}(args):") in codes[1]:
            self.assertNotIn(
                "partitioned_bw_subgraph_0_0(",
                self._generated_fn_body(codes[1], signature),
            )
            partition_index += 1
        self.assertGreater(partition_index, 0)
        self.assertIn("partitioned_bw_subgraph_0_0(", codes[1])

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": False,
            "size_asserts": True,
            "triton.cudagraphs": False,
        }
    )
    def test_region_partition_does_not_duplicate_input_asserts(self):
        """Partition functions carry the asserts, so call() must not repeat them.

        The gate cannot read the ambient triton.cudagraphs: a regional request
        leaves it off while partitions are still emitted.
        """
        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True}
        )

        @torch.compiler.nested_compile_region(options=options)
        def region(x):
            return torch.sin(x)

        def fn(x):
            return torch.cos(region(x + 1))

        x = torch.randn(16, 16, device="cuda")
        result, codes = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), x
        )

        self.assertEqual(result, fn(x))
        self.assertIn("def partition_0(args):", codes[0])
        self.assertEqual(codes[0].count("assert_size_stride("), 1)

    @requires_cuda_and_triton
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": True,
            "triton.cudagraphs": False,
        }
    )
    def test_customized_partition_wrapper_lowers_as_partitioned(self):
        """Parity with is_using_cudagraph_partition().

        A customized partition wrapper emits partition functions with
        triton.cudagraphs off, so lowering still has to treat the graph as
        partitioned even though the cudagraphs box is False.
        """
        from torch._inductor.graph import GraphLowering
        from torch._inductor.utils import (
            _unstable_customized_partition_wrapper,
            set_customized_partition_wrappers,
        )

        def wrapper(fn, metadata):
            return fn

        previous = _unstable_customized_partition_wrapper.wrapper
        set_customized_partition_wrappers(wrapper)
        self.addCleanup(set_customized_partition_wrappers, previous)

        use_cudagraph_partition = []

        class RecordingGraphLowering(GraphLowering):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                use_cudagraph_partition.append(self.use_cudagraph_partition)

        def fn(x):
            return torch.cos(torch.sin(x) + 1)

        x = torch.randn(16, 16, device="cuda")
        with mock.patch(
            "torch._inductor.compile_fx.GraphLowering", RecordingGraphLowering
        ):
            result = torch.compile(fn, backend="inductor", fullgraph=True)(x)

        self.assertEqual(result, fn(x))
        self.assertEqual(use_cudagraph_partition, [True])

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": False,
            "triton.cudagraphs": True,
        }
    )
    def test_region_partition_is_forced_on_the_graph_not_the_config(self):
        """Forcing partitioning for a region must not move the ambient config.

        Passes that merely assume "partitioning will split this off" -- such as
        ConstructorMoverPass moving CPU inputs/outputs to GPU -- read
        config.graph_partition, and must keep seeing what the user asked for.
        """
        import torch._inductor.fx_passes.post_grad as post_grad
        from torch._inductor import cudagraph_trees

        cudagraph_trees.reset_cudagraph_trees()
        self.addCleanup(cudagraph_trees.reset_cudagraph_trees)

        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": False}
        )

        @torch.compiler.nested_compile_region(options=options)
        def region(x):
            return torch.sin(x)

        def fn(x):
            return torch.cos(region(x + 1))

        seen_allow_inputs_outputs = []
        constructor_mover_pass = post_grad.ConstructorMoverPass

        class RecordingConstructorMoverPass(constructor_mover_pass):
            def __init__(
                self, target, *, allow_inputs=False, allow_outputs=False, **kwargs
            ):
                seen_allow_inputs_outputs.append((allow_inputs, allow_outputs))
                super().__init__(
                    target,
                    allow_inputs=allow_inputs,
                    allow_outputs=allow_outputs,
                    **kwargs,
                )

        x = torch.randn(16, 16, device="cuda")
        with mock.patch.object(
            post_grad, "ConstructorMoverPass", RecordingConstructorMoverPass
        ):
            result, codes = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), x
            )

        self.assertEqual(result, fn(x))
        # The region really was carved out...
        self.assertIn("def partition_0(args):", codes[0])
        # ...without telling the rest of the compile that partitioning is on.
        self.assertTrue(seen_allow_inputs_outputs)
        self.assertEqual(set(seen_allow_inputs_outputs), {(False, False)})

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": True,
            "triton.cudagraphs": True,
        }
    )
    def test_redundant_cudagraph_config_does_not_split_partition(self):
        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True}
        )

        @torch.compiler.nested_compile_region(options=nested_config)
        def region(x):
            return torch.sin(x)

        def fn(x):
            return torch.cos(region(x + 1))

        x = torch.randn(16, 16, device=GPU_TYPE)
        result, codes = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), x
        )

        self.assertEqual(result, fn(x))
        self.assertEqual(codes[0].count("def partition_"), 1)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": True,
            "triton.cudagraphs": True,
        }
    )
    def test_redundant_disabled_cudagraph_config_does_not_partition(self):
        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": False}
        )

        @torch.compiler.nested_compile_region(options=nested_config)
        def region(x):
            return torch.sin(x)

        @torch._dynamo.override_cudagraphs(fwd=False)
        def fn(x):
            return torch.cos(region(x + 1))

        x = torch.randn(16, 16, device=GPU_TYPE)
        result, codes = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), x
        )

        self.assertEqual(result, fn(x))
        self.assertNotIn("def partition_0(args):", codes[0])

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": False,
            "triton.cudagraphs": False,
        }
    )
    def test_reused_region_codegen_is_not_duplicated(self):
        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True}
        )

        @torch.compiler.nested_compile_region(options=nested_config)
        def region(x):
            return torch.sin(x)

        def fn(x):
            for _ in range(8):
                x = region(x)
            return x

        x = torch.randn(16, 16, device=GPU_TYPE)
        result, codes = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), x
        )

        self.assertEqual(result, fn(x))
        self.assertEqual(codes[0].count(" = async_compile.triton("), 1)
        # Adjacent calls share one capture policy, so they share one partition
        # instead of paying a capture/replay boundary per call.
        self.assertEqual(codes[0].count("def partition_"), 1)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "expand_dimension_for_pointwise_nodes": True,
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": False,
            "triton.cudagraph_min_partition_size": 1,
            "triton.cudagraphs": False,
        }
    )
    def test_nested_region_partition_count_initializes_codegen_state(self):
        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True}
        )

        @torch.compiler.nested_compile_region(options=nested_config)
        def region(x):
            first = torch.sin(x)
            return first, torch.cos(first[:2])

        x = torch.randn(16, 16, device=GPU_TYPE)
        result, codes = run_and_get_code(
            torch.compile(region, backend="inductor", fullgraph=True), x
        )

        self.assertEqual(result, region(x))
        self.assertIn("def partition_0(args):", codes[0])

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": False,
            "triton.cudagraph_min_partition_size": 2,
            "triton.cudagraphs": False,
        }
    )
    def test_nested_region_cudagraph_min_partition_size_counts_body(self):
        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True}
        )

        @torch.compiler.nested_compile_region(options=nested_config)
        def region(x, weight0, weight1):
            return (x @ weight0) @ weight1

        def fn(x, weight0, weight1):
            return torch.sin(region(x, weight0, weight1))

        inputs = [torch.randn(16, 16, device=GPU_TYPE) for _ in range(3)]
        result, codes = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), *inputs
        )

        self.assertEqual(result, fn(*inputs))
        partition = self._generated_fn_body(codes[0], "def partition_0(args):")
        self.assertIn("repeated_subgraph0(", partition)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": False,
            "triton.cudagraphs": False,
        }
    )
    def test_nested_region_cudagraph_unsafe_body_is_not_partitioned(self):
        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True}
        )

        @torch.compiler.nested_compile_region(options=nested_config)
        def region(x):
            return torch.sin(x).cpu().cuda()

        compiled_region = torch.compile(region, backend="inductor", fullgraph=True)
        x = torch.randn(16, 16, device=GPU_TYPE)
        result, codes = run_and_get_code(compiled_region, x)

        self.assertEqual(result, region(x))
        self.assertIn("def repeated_subgraph0(", codes[0])
        self.assertNotIn("def partition_0(args):", codes[0])

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
    @parametrize(
        "config_key,config_value",
        (
            ("triton.cudagraph_min_partition_size", 1),
            ("triton.cudagraph_skip_dynamic_graphs", True),
            ("triton.cudagraph_trees", True),
            ("triton.persistent_reductions", False),
        ),
    )
    def test_unsupported_nested_region_inductor_config(
        self, direction, config_key, config_value
    ):
        config_arg = (
            "fw_inductor_config_patches"
            if direction == "forward"
            else "bw_inductor_config_patches"
        )
        with self.assertRaisesRegex(
            ValueError,
            f"Inductor config key '{config_key}' is not supported in {direction}",
        ):
            get_invoke_subgraph_compile_options(
                **{config_arg: {config_key: config_value}}
            )

    @torch._inductor.config.patch("triton.cudagraphs", False)
    def test_nested_region_cudagraph_rejects_dynamo_override(self):
        from torch._inductor import _CudagraphAnnotation
        from torch._inductor.compile_fx import create_compiler_config_extra

        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": False}
        )
        gm = self._configured_region_graph_module(nested_config)
        gm.meta["cudagraph_annotation"] = _CudagraphAnnotation(fwd=True, bwd=None)

        with self.assertRaisesRegex(
            RuntimeError,
            "override_cudagraphs.*cannot be combined.*nested compile-region",
        ):
            create_compiler_config_extra(gm)

    @parametrize(
        "top_level,annotation_fwd,annotation_bwd,region_cudagraphs",
        ((True, True, None, False), (False, None, False, True)),
    )
    def test_no_op_dynamo_override_allows_regional_cudagraphs(
        self, top_level, annotation_fwd, annotation_bwd, region_cudagraphs
    ):
        """An annotation that restates the ambient config is not a conflict."""
        from torch._inductor import _CudagraphAnnotation
        from torch._inductor.compile_fx import create_compiler_config_extra

        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": region_cudagraphs}
        )
        gm = self._configured_region_graph_module(nested_config)
        gm.meta["cudagraph_annotation"] = _CudagraphAnnotation(
            fwd=annotation_fwd, bwd=annotation_bwd
        )

        with torch._inductor.config.patch("triton.cudagraphs", top_level):
            compiler_config = create_compiler_config_extra(gm)

        self.assertTrue(compiler_config.has_regional_cudagraphs)
        self.assertEqual(compiler_config.top_level_cudagraphs, top_level)
        self.assertIsNone(compiler_config.cudagraphs_bwd_override)

    @torch._inductor.config.patch("triton.cudagraphs", False)
    def test_nested_region_rejects_conflicting_cudagraph_configs(self):
        from torch._inductor.compile_fx import create_compiler_config_extra

        inner_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": False}
        )
        outer_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True}
        )
        inner = self._configured_region_graph_module(inner_config)
        gm = self._configured_region_graph_module(outer_config, inner)

        with self.assertRaisesRegex(
            RuntimeError,
            "nested compile regions cannot have conflicting cudagraph configs",
        ):
            create_compiler_config_extra(gm)

    def test_nested_region_options_validate_direct_construction(self):
        with self.assertRaisesRegex(
            ValueError,
            "Inductor config key 'graph_partition' is not supported in forward",
        ):
            NestedCompileRegionOptions(
                inductor_config_patches={"graph_partition": True}
            )

    @parametrize("direction", ("forward", "backward"))
    def test_nested_region_options_freeze_config(self, direction):
        patches = {"fallback_by_default": True}
        field = (
            "inductor_config_patches"
            if direction == "forward"
            else "bw_inductor_config_patches"
        )
        nested_config = NestedCompileRegionOptions(**{field: patches})
        frozen_patches = getattr(nested_config, field)

        patches["fallback_by_default"] = False
        self.assertEqual(frozen_patches, {"fallback_by_default": True})
        with self.assertRaisesRegex(TypeError, "does not support mutation"):
            frozen_patches["fallback_by_default"] = False
        with self.assertRaisesRegex(
            FrozenInstanceError, f"cannot assign to field '{field}'"
        ):
            setattr(nested_config, field, {})

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(fx_graph_cache=False, fx_graph_remote_cache=False)
    def test_nested_region_options_snapshot_lazy_backward(self):
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
        result.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(pass_calls, [])

    @torch._inductor.config.patch(
        freezing=True,
        fx_graph_cache=False,
        fx_graph_remote_cache=False,
        pre_grad_pass_timing="early",
    )
    def test_nested_region_options_snapshot_freezing_after_pre_grad(self):
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

        x = torch.randn(10)
        expected = fn(x)
        with (
            torch.no_grad(),
            torch._inductor.config.patch(pre_grad_custom_pass=mutate_config),
        ):
            result = torch.compile(fn, backend="inductor", fullgraph=True)(x)
        self.assertEqual(result, expected)
        self.assertEqual(pass_calls, [])

    @requires_gpu_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(graph_partition=True)
    @torch._inductor.config.patch("triton.cudagraphs", True)
    def test_region_body_cpu_op_is_not_cudagraph_partitioned(self):
        @torch.compiler.nested_compile_region
        def region(x):
            return torch.sin(x).cpu().to(GPU_TYPE)

        def fn(x):
            return torch.cos(region(x))

        x = torch.randn(16, 16, device=GPU_TYPE)
        result, codes = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), x
        )

        self.assertEqual(result, fn(x))
        self.assertIn("repeated_subgraph0(", codes[0])
        partition = self._generated_fn_body(codes[0], "def partition_0(args):")
        self.assertNotIn("repeated_subgraph0(", partition)

    @requires_gpu_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(graph_partition=True)
    @torch._inductor.config.patch("triton.cudagraphs", True)
    @torch._inductor.config.patch("triton.cudagraph_min_partition_size", 3)
    def test_region_body_kernels_count_toward_min_partition_size(self):
        # The region call site is a couple of scheduler nodes, but its body
        # holds four kernels, which is what the partition size should weigh.
        @torch.compiler.nested_compile_region
        def region(x, weight0, weight1, weight2, weight3):
            return ((x @ weight0) @ weight1 @ weight2) @ weight3

        inputs = [torch.randn(16, 16, device=GPU_TYPE) for _ in range(5)]
        result, codes = run_and_get_code(
            torch.compile(region, backend="inductor", fullgraph=True), *inputs
        )

        self.assertEqual(result, region(*inputs))
        partition = self._generated_fn_body(codes[0], "def partition_0(args):")
        self.assertIn("repeated_subgraph0(", partition)


if __name__ == "__main__":
    run_tests()
