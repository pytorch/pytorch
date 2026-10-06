# Owner(s): ["module: dynamo"]

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
        compiled_graph.fx_kwargs = {"is_backward": True}

        with self.assertRaisesRegex(
            AssertionError, "boxed_forward_device_index must not be None"
        ):
            maybe_handle_backward_generation(compiled_graph, None)

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
        ((False, False), (True, False), (True, True)),
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
    def test_unsupported_graph_partition_and_regional_cudagraphs(
        self, cpp_wrapper, aot_mode
    ):
        """cpp_wrapper/aot_mode cannot partition, so a region cannot be isolated."""
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
                cudagraphs_top_level=True,
                cudagraph_region_forced_partition=True,
            )

        unsupported = cpp_wrapper or aot_mode
        # An opt-out keeps the enclosing capture: losing every CUDA graph in the
        # model is further from the request than failing to exclude one region.
        self.assertTrue(cudagraphs)
        # The forced partitioning must not reach lowering.
        self.assertEqual(
            codegen_and_compile.call_args.kwargs["cudagraph_region_forced_partition"],
            not unsupported,
        )


if __name__ == "__main__":
    run_tests()
