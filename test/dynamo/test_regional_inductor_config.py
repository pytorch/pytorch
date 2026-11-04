# Owner(s): ["module: dynamo"]

import contextlib
import copy
import importlib
import re
import unittest
from dataclasses import FrozenInstanceError
from unittest import mock

import torch
import torch._inductor.test_case
from torch._dynamo.exc import BackendCompilerFailed
from torch._higher_order_ops.invoke_subgraph import (
    get_invoke_subgraph_compile_options,
    NestedCompileRegionOptions,
)
from torch._inductor.test_case import run_tests
from torch._inductor.utils import (
    _unstable_customized_partition_wrapper,
    run_and_get_code,
    run_fw_bw_and_get_code,
    set_customized_partition_wrappers,
)
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


class _CpuRoundTripBackward(torch.autograd.Function):
    """A backward that lowering cannot capture."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sin()

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return (grad.cpu() * 2).to(GPU_TYPE) * x.cos()


@torch.library.custom_op(
    "regional_cudagraph_test::unsafe_clone",
    mutates_args=(),
    tags=(torch._C.Tag.cudagraph_unsafe,),
)
def _cudagraph_unsafe_clone(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_cudagraph_unsafe_clone.register_fake
def _(x):
    return torch.empty_like(x)


@torch.library.custom_op("regional_cudagraph_test::count_positive", mutates_args=())
def _count_positive(x: torch.Tensor) -> int:
    return int((x[:, 0] > 0).sum().item())


@_count_positive.register_fake
def _(x):
    return torch.library.get_ctx().new_dynamic_size()


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

    @classmethod
    def _partition_bodies(cls, code):
        bodies = []
        while (signature := f"def partition_{len(bodies)}(args):") in code:
            bodies.append(cls._generated_fn_body(code, signature))
        return bodies

    @staticmethod
    @contextlib.contextmanager
    def _record_captures():
        """Collect the kwargs of every cudagraphify call, i.e. of every capture."""
        from torch._inductor.compile_fx import cudagraphify

        captures = []

        def record(*args, **kwargs):
            captures.append(kwargs)
            return cudagraphify(*args, **kwargs)

        with mock.patch("torch._inductor.compile_fx.cudagraphify", record):
            yield captures

    def _run_fw_bw_and_compare(self, fn, *inputs, backend="inductor", **kwargs):
        """Compile fn, run it forward and backward on copies of inputs, and
        compare the output and the input grads with eager. Returns the code."""

        def copy_inputs():
            return [x.detach().clone().requires_grad_(x.requires_grad) for x in inputs]

        compiled_inputs, eager_inputs = copy_inputs(), copy_inputs()
        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True, **kwargs)
        result, codes = run_fw_bw_and_get_code(lambda: compiled_fn(*compiled_inputs))
        expected = fn(*eager_inputs)
        expected.sum().backward()
        self.assertEqual(result, expected)
        self.assertEqual(
            [x.grad for x in compiled_inputs], [x.grad for x in eager_inputs]
        )
        return codes

    def _assert_opt_out_captures(self, codes, captures):
        """Both directions capture their partitions, none of which calls the
        opted-out region."""
        self.assertEqual(len(codes), 2)
        partitions = [self._partition_bodies(code) for code in codes]
        self.assertEqual(
            sorted(c["is_backward"] for c in captures),
            [False] * len(partitions[0]) + [True] * len(partitions[1]),
        )
        for code, code_partitions, subgraph_name in zip(
            codes,
            partitions,
            ("partitioned_fw_subgraph_0_0(", "partitioned_bw_subgraph_0_0("),
        ):
            self.assertIn(subgraph_name, code)
            self.assertGreater(len(code_partitions), 0)
            for partition in code_partitions:
                self.assertNotIn(subgraph_name, partition)

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

    @requires_gpu_and_triton
    @torch._inductor.config.patch({"graph_partition": True, "triton.cudagraphs": True})
    def test_dynamo_cudagraphs_backend_ignores_inductor_graph_partition(self):
        """The cudagraphs backend never partitions, so a CPU op still skips capture."""
        if GPU_TYPE != "cuda":
            self.skipTest("the cudagraphs backend is CUDA-only")

        def fn(x, y):
            return x + 1, y + 2

        compiled = torch.compile(fn, backend="cudagraphs")
        for i in range(3):
            x = torch.full((4,), float(i), device=GPU_TYPE)
            y = torch.full((4,), float(10 * i))
            self.assertEqual(compiled(x, y), fn(x, y))
        skips = torch._dynamo.utils.counters["inductor"]["cudagraph_skips"]
        self.assertEqual(skips, 1)

    def test_check_multiple_devices_or_any_cpu_nodes(self):
        """Nothing left to capture once cpu/meta are popped is refused as "no GPU ops"."""
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
                    dict(mapping), cudagraphs_per_partition=partition
                )
                if expected is None:
                    self.assertIsNone(reason)
                else:
                    self.assertIn(expected, reason)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(
        inline_single_use_invoke_subgraph=False, capture_scalar_outputs=True
    )
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": True,
            "triton.cudagraphs": True,
            "cudagraph_unsafe_unbacked_ops": [
                "regional_cudagraph_test::count_positive"
            ],
        }
    )
    def test_region_consuming_unsafe_unbacked_size_stays_out_of_partitions(self):
        """The size comes from a cudagraph_unsafe_unbacked_ops op outside the
        region, so only the call site, not the body, can see it is unsafe."""

        @torch.compiler.nested_compile_region
        def region(y):
            return torch.sin(y) * 2

        def fn(x):
            n = _count_positive(x)
            torch._check_is_size(n)
            torch._check(n <= x.shape[0])
            return region(x[:n] + 1)

        x = torch.randn(64, 16, device=GPU_TYPE)
        with self._record_captures() as captures:
            result, (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), x
            )

        self.assertEqual(result, fn(x))
        self.assertIn("repeated_subgraph0(", code)
        partitions = self._partition_bodies(code)
        self.assertEqual(len(captures), len(partitions))
        for partition in partitions:
            self.assertNotIn("repeated_subgraph0(", partition)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(
        inline_single_use_invoke_subgraph=False,
        capture_dynamic_output_shape_ops=True,
    )
    @torch._inductor.config.patch(fx_graph_cache=False, fx_graph_remote_cache=False)
    @parametrize("requested_by", ("top_level", "region", "partition_wrapper"))
    @parametrize(
        "body",
        (
            "cpu_round_trip",
            "cudagraph_unsafe_op",
            "unbacked",
            "nested_region",
            "multi_output",
        ),
    )
    def test_uncapturable_region_body_stays_out_of_partitions(self, requested_by, body):
        """A region is judged by its body, not as one opaque call node."""
        from torch._dynamo.utils import counters

        def cpu_round_trip(x):
            return torch.sin(x).cpu().to(GPU_TYPE)

        def cudagraph_unsafe_op(x):
            return _cudagraph_unsafe_clone(torch.sin(x))

        inner_region = torch.compiler.nested_compile_region(cudagraph_unsafe_op)
        bodies = {
            "cpu_round_trip": cpu_round_trip,
            "cudagraph_unsafe_op": cudagraph_unsafe_op,
            "unbacked": lambda x: x + torch.nonzero(x > 0).sum(),
            "nested_region": lambda x: torch.cos(inner_region(x)),
            "multi_output": lambda x: (cpu_round_trip(x), torch.cos(x)),
        }
        options = None
        config_patches = {
            "graph_partition": requested_by != "region",
            "triton.cudagraphs": requested_by == "top_level",
        }
        if requested_by == "region":
            options = get_invoke_subgraph_compile_options(
                fw_inductor_config_patches={"triton.cudagraphs": True}
            )
        elif requested_by == "partition_wrapper":
            previous = _unstable_customized_partition_wrapper.wrapper
            set_customized_partition_wrappers(lambda fn, metadata: fn)
            self.addCleanup(set_customized_partition_wrappers, previous)
        region = torch.compiler.nested_compile_region(bodies[body], options=options)

        def fn(x):
            out = region(x)
            if body == "multi_output":
                return torch.cos(out[0]) * out[1]
            return torch.cos(out)

        x = torch.randn(16, 16, device=GPU_TYPE)
        with (
            torch._inductor.config.patch(config_patches),
            self._record_captures() as captures,
        ):
            result, (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), x
            )

        self.assertEqual(result, fn(x))
        # The MultiOutputs that unpack the region's results must stay with it.
        (region_buf,) = re.findall(r"(\w+) = repeated_subgraph0\(", code)
        partitions = self._partition_bodies(code)
        for partition in partitions:
            self.assertNotIn("repeated_subgraph0(", partition)
            self.assertNotIn(f"{region_buf}[", partition)
        if requested_by == "region":
            self.assertEqual(captures, [])
            # The only capture asked for was refused, which is a skip.
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)
        else:
            # The rest of the graph is still partitioned, also when only a
            # customized partition wrapper asks for partitions.
            self.assertGreater(len(partitions), 0)
            captured = requested_by == "top_level"
            self.assertEqual(len(captures), len(partitions) if captured else 0)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": True,
            "triton.cudagraphs": False,
        }
    )
    def test_partition_wrapper_partitions_capturable_region(self):
        """With cudagraphs off, a customized partition wrapper still gets a
        capturable region partitioned like any other op."""
        previous = _unstable_customized_partition_wrapper.wrapper
        set_customized_partition_wrappers(lambda fn, metadata: fn)
        self.addCleanup(set_customized_partition_wrappers, previous)

        @torch.compiler.nested_compile_region
        def region(x):
            return torch.sin(x) * 2

        def fn(x):
            return torch.cos(region(x + 1))

        x = torch.randn(16, 16, device=GPU_TYPE)
        result, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), x
        )

        self.assertEqual(result, fn(x))
        (partition,) = self._partition_bodies(code)
        self.assertIn("repeated_subgraph0(", partition)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "triton.cudagraph_min_partition_size": 3,
        }
    )
    @parametrize("region_opts_in", (False, True))
    def test_region_body_kernels_count_toward_min_partition_size(self, region_opts_in):
        # The region call site is a couple of scheduler nodes, but its body
        # holds four kernels, which is what the partition size should weigh.
        options = None
        if region_opts_in:
            options = get_invoke_subgraph_compile_options(
                fw_inductor_config_patches={"triton.cudagraphs": True}
            )

        @torch.compiler.nested_compile_region(options=options)
        def region(x, weight0, weight1, weight2, weight3):
            return ((x @ weight0) @ weight1 @ weight2) @ weight3

        def fn(*inputs):
            # Without an outer kernel the partition holds only the region call.
            out = region(*inputs)
            return torch.sin(out) if region_opts_in else out

        inputs = [torch.randn(16, 16, device=GPU_TYPE) for _ in range(5)]
        top_level = not region_opts_in
        with (
            torch._inductor.config.patch(
                {"graph_partition": top_level, "triton.cudagraphs": top_level}
            ),
            self._record_captures() as captures,
        ):
            result, (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), *inputs
            )

        self.assertEqual(result, fn(*inputs))
        (partition,) = self._partition_bodies(code)
        self.assertIn("repeated_subgraph0(", partition)
        self.assertEqual(len(captures), 1)
        if region_opts_in:
            self.assertNotIn(".run(", partition)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": True,
            "triton.cudagraphs": True,
            "triton.cudagraph_min_partition_size": 5,
        }
    )
    @parametrize("num_calls", (1, 2))
    def test_repeated_region_calls_count_toward_min_partition_size(self, num_calls):
        """Each call runs the body's three kernels again, so two calls of one
        body reach the threshold of five and a single call does not."""

        @torch.compiler.nested_compile_region
        def region(x, weight0, weight1, weight2):
            return ((x @ weight0) @ weight1) @ weight2

        def fn(weight0, weight1, weight2, *xs):
            return tuple(region(x, weight0, weight1, weight2) for x in xs)

        inputs = [torch.randn(16, 16, device=GPU_TYPE) for _ in range(3 + num_calls)]
        with self._record_captures() as captures:
            result, (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), *inputs
            )

        self.assertEqual(result, fn(*inputs))
        self.assertEqual(code.count("= repeated_subgraph0("), num_calls)
        captured = num_calls == 2
        self.assertEqual(len(self._partition_bodies(code)), int(captured))
        self.assertEqual(len(captures), int(captured))

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

        with (
            torch._inductor.config.patch("triton.cudagraphs", global_cudagraphs),
            self._record_captures() as captures,
        ):
            compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
            result, (code,) = run_and_get_code(compiled_fn, x)
            result = result.clone()
            torch.compiler.cudagraph_mark_step_begin()
            replayed_result = compiled_fn(x).clone()

        self.assertEqual(result, expected)
        self.assertEqual(replayed_result, expected)
        partitions = self._partition_bodies(code)
        if global_cudagraphs == regional_cudagraphs:
            self.assertEqual(partitions, [])
            self.assertEqual(len(captures), int(global_cudagraphs))
            return
        self.assertGreater(len(partitions), 0)
        self.assertEqual(len(captures), len(partitions))
        for partition in partitions:
            # An opt-in captures the region and nothing else; an opt-out
            # captures everything else.
            self.assertEqual("repeated_subgraph0(" in partition, regional_cudagraphs)
            if regional_cudagraphs:
                self.assertNotIn(".run(", partition)

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
    @parametrize("forward_loses_capture", (False, True))
    def test_region_opt_in_training(self, forward_loses_capture):
        """Both directions capture only the opted-in region. A backward cannot
        bring back capture that its forward requested and lost, and then is not
        partitioned either."""
        from torch._dynamo.utils import counters

        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True}
        )

        @torch.compiler.nested_compile_region(options=options)
        def region(x, y):
            return x @ y

        def fn(x, y):
            return torch.sin(region(x, y))

        # 2 is too small for the one-matmul forward, not for the two-matmul
        # backward.
        min_partition_size = 2 if forward_loses_capture else 1
        compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        with (
            torch._inductor.config.patch(
                "triton.cudagraph_min_partition_size", min_partition_size
            ),
            self._record_captures() as captures,
        ):
            for step in range(3):
                inputs = [
                    torch.randn(16, 16, device=GPU_TYPE, requires_grad=True)
                    for _ in range(2)
                ]
                expected_inputs = [x.detach().clone().requires_grad_() for x in inputs]
                torch.compiler.cudagraph_mark_step_begin()
                if step == 0:
                    result, codes = run_fw_bw_and_get_code(lambda: compiled_fn(*inputs))
                else:
                    result = compiled_fn(*inputs)
                    result.sum().backward()
                expected = fn(*expected_inputs)
                expected.sum().backward()
                self.assertEqual(result, expected)
                self.assertEqual(
                    [x.grad for x in inputs], [x.grad for x in expected_inputs]
                )

        fw_partitions, bw_partitions = map(self._partition_bodies, codes)
        if forward_loses_capture:
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)
            self.assertEqual(captures, [])
            self.assertEqual(bw_partitions, [])
            return
        self.assertEqual(
            sorted(c["is_backward"] for c in captures),
            [False] * len(fw_partitions) + [True] * len(bw_partitions),
        )
        for partitions, subgraph_name in (
            (fw_partitions, "partitioned_fw_subgraph_0_0("),
            (bw_partitions, "partitioned_bw_subgraph_0_0("),
        ):
            self.assertGreater(len(partitions), 0)
            for partition in partitions:
                self.assertIn(subgraph_name, partition)
                self.assertNotIn(".run(", partition)

    @requires_cuda_and_triton
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": False,
            "triton.cudagraphs": True,
        }
    )
    def test_uncaptured_backward_runs_after_cudagraph_trees_reset(self):
        def fn(x, weight):
            return _CpuRoundTripBackward.apply(x * weight)

        compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        inputs = [
            torch.randn(64, 64, device=GPU_TYPE, requires_grad=True) for _ in range(2)
        ]
        expected_inputs = [x.detach().clone().requires_grad_() for x in inputs]
        result = compiled_fn(*inputs)
        torch._inductor.cudagraph_trees.reset_cudagraph_trees()
        result.sum().backward()
        fn(*expected_inputs).sum().backward()
        self.assertEqual([x.grad for x in inputs], [x.grad for x in expected_inputs])

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(
        enable_aot_compile=True, inline_single_use_invoke_subgraph=False
    )
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": False,
            "triton.cudagraphs": False,
        }
    )
    def test_precompile_keeps_region_opt_in(self):
        import tempfile

        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True}
        )

        @torch.compiler.nested_compile_region(options=options)
        def region(x):
            return torch.sin(x) * 2

        def fn(x):
            return torch.cos(region(x + 1))

        x = torch.randn(16, 16, device=GPU_TYPE)
        compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled_fn = compiled_fn.aot_compile(((x,), {}))
        with (
            tempfile.TemporaryDirectory() as tmp,
            self._record_captures() as captures,
        ):
            path = f"{tmp}/model.pt"
            compiled_fn.save_compiled_function(path)
            torch._dynamo.reset()
            with open(path, "rb") as f:
                loaded_fn = torch.compiler.load_compiled_function(f)
            for _ in range(2):
                torch.compiler.cudagraph_mark_step_begin()
                self.assertEqual(loaded_fn(x), fn(x))
        self.assertEqual(len(captures), 1)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(
        enable_aot_compile=True, inline_single_use_invoke_subgraph=False
    )
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": False,
            "triton.cudagraphs": False,
        }
    )
    @parametrize("loader_cudagraphs", (False, True))
    @parametrize("mode", ("inference", "training", "backward_only"))
    def test_precompile_reports_dropped_region_opt_in(self, mode, loader_cudagraphs):
        """A region-only opt-in that cpp_wrapper dropped is reported as a cudagraph
        skip on a precompile load too, whatever the loading process's config."""
        import tempfile

        from torch._dynamo.utils import counters

        backward_only = mode == "backward_only"
        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": not backward_only},
            bw_inductor_config_patches=(
                {"triton.cudagraphs": True} if backward_only else None
            ),
        )

        @torch.compiler.nested_compile_region(options=options)
        def region(x):
            return torch.sin(x) * 2

        def fn(x):
            return torch.cos(region(x))

        x = torch.randn(16, 16, device=GPU_TYPE, requires_grad=mode != "inference")

        def run(f):
            inputs = x.detach().clone().requires_grad_(x.requires_grad)
            out = f(inputs)
            if inputs.requires_grad:
                out.sum().backward()
            return out, inputs.grad

        counters.clear()
        with (
            tempfile.TemporaryDirectory() as tmp,
            self._record_captures() as captures,
        ):
            with torch._inductor.config.patch(cpp_wrapper=True):
                compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
                compiled_fn = compiled_fn.aot_compile(((x,), {}))
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)
            path = f"{tmp}/model.pt"
            compiled_fn.save_compiled_function(path)
            torch._dynamo.reset()
            counters.clear()
            with torch._inductor.config.patch("triton.cudagraphs", loader_cudagraphs):
                with open(path, "rb") as f:
                    loaded_fn = torch.compiler.load_compiled_function(f)
                self.assertEqual(run(loaded_fn), run(fn))
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)
        self.assertEqual(captures, [])

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(
        enable_aot_compile=True, inline_single_use_invoke_subgraph=False
    )
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": False,
            "triton.cudagraphs": False,
        }
    )
    @parametrize("artifact", ("region_backward_only", "top_level"))
    def test_precompile_keeps_region_backward_only_opt_in(self, artifact):
        """A backward compiled for capture by a region is captured when loaded
        with cudagraphs off; one compiled for capture by the top level is not."""
        import tempfile

        top_level = artifact == "top_level"

        def body(x):
            return torch.sin(x) * 2

        if not top_level:
            options = get_invoke_subgraph_compile_options(
                fw_inductor_config_patches={"triton.cudagraphs": False},
                bw_inductor_config_patches={"triton.cudagraphs": True},
            )
            body = torch.compiler.nested_compile_region(options=options)(body)

        def fn(x):
            return torch.cos(body(x))

        x = torch.randn(16, 16, device=GPU_TYPE, requires_grad=True)
        with torch._inductor.config.patch("triton.cudagraphs", top_level):
            compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
            compiled_fn = compiled_fn.aot_compile(((x,), {}))
        with (
            tempfile.TemporaryDirectory() as tmp,
            self._record_captures() as captures,
        ):
            path = f"{tmp}/model.pt"
            compiled_fn.save_compiled_function(path)
            torch._dynamo.reset()
            with open(path, "rb") as f:
                loaded_fn = torch.compiler.load_compiled_function(f)
            for _ in range(2):
                inputs = [x.detach().clone().requires_grad_() for _ in range(2)]
                torch.compiler.cudagraph_mark_step_begin()
                loaded_fn(inputs[0]).sum().backward()
                fn(inputs[1]).sum().backward()
                self.assertEqual(inputs[0].grad, inputs[1].grad)
        self.assertEqual(
            [c["is_backward"] for c in captures], [] if top_level else [True]
        )

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

        x = torch.randn(16, 16, device=GPU_TYPE)
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
        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": False}
        )

        @torch.compiler.nested_compile_region(options=options)
        def region(x):
            return torch.sin(x)

        def fn(x, weight):
            y = (x @ weight).cpu().to(GPU_TYPE)
            return (region(y) + torch.cos(y)).sum()

        x = torch.randn(16, 16, device=GPU_TYPE, requires_grad=True)
        weight = torch.randn(16, 16, device=GPU_TYPE, requires_grad=True)
        with self._record_captures() as captures:
            codes = self._run_fw_bw_and_compare(fn, x, weight)

        self.assertEqual(captures, [])
        # Positive control: the region really is still a region, so the opt-out
        # had something to act on.
        self.assertEqual(len(codes), 2)
        for code, subgraph_name in zip(
            codes, ("partitioned_fw_subgraph_0_0(", "partitioned_bw_subgraph_0_0(")
        ):
            self.assertIn(subgraph_name, code)
        # The forward lost capture before the backward was compiled, so there is
        # nothing left to carve the region out of.
        self.assertEqual(self._partition_bodies(codes[1]), [])

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
    def test_region_opt_out_in_module_training(self):
        """Training stays correct across steps, and saved activations are not
        static inputs of the captured partitions."""
        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": False}
        )

        @torch.compiler.nested_compile_region(options=options)
        def region(x, weight, bias):
            return torch.relu(torch.nn.functional.linear(x, weight, bias))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = torch.nn.Linear(16, 16)
                self.block = torch.nn.Linear(16, 16)
                self.head = torch.nn.Linear(16, 16)

            def forward(self, x):
                h = region(self.inner(x), self.block.weight, self.block.bias)
                return self.head(h)

        model = Model().to(GPU_TYPE)
        compiled_model = torch.compile(
            copy.deepcopy(model), backend="inductor", fullgraph=True
        )
        with self._record_captures() as captures:
            for step in range(3):
                # Churn the allocator, so a stale or uninitialized buffer shows up
                # as a wrong value.
                garbage = [torch.randn(n, device=GPU_TYPE) for n in (64, 4096, 1 << 18)]
                del garbage
                x = torch.randn(8, 16, device=GPU_TYPE)
                torch.compiler.cudagraph_mark_step_begin()
                if step == 0:
                    result, codes = run_fw_bw_and_get_code(lambda: compiled_model(x))
                else:
                    result = compiled_model(x)
                    result.sum().backward()
                expected = model(x)
                expected.sum().backward()
                self.assertEqual(result, expected)
                self.assertEqual(
                    [p.grad for p in compiled_model.parameters()],
                    [p.grad for p in model.parameters()],
                )
                model.zero_grad()
                compiled_model.zero_grad()

        self._assert_opt_out_captures(codes, captures)
        for is_backward in (False, True):
            static_names = [
                {c["placeholders"][i].name for i in c["static_input_idxs"]}
                for c in captures
                if c["is_backward"] == is_backward
            ]
            self.assertTrue(any(static_names))
            for names in static_names:
                self.assertTrue(all(n.startswith("primals_") for n in names), names)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "triton.cudagraphs": True,
        }
    )
    @parametrize("graph_partition", (False, True))
    def test_region_opt_out_follows_compile_options_graph_partition(
        self, graph_partition
    ):
        """The lazily compiled backward runs outside torch.compile(options=...)
        until inner_compile re-applies them, so its partitioning decision has to
        be made under them, like the forward's."""
        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": False}
        )

        @torch.compiler.nested_compile_region(options=options)
        def region(x):
            return torch.sin(x) * 2

        def fn(x):
            return torch.cos(region(x + 1)) * 3

        x = torch.randn(16, 16, device=GPU_TYPE, requires_grad=True)
        with (
            torch._inductor.config.patch(graph_partition=graph_partition),
            self._record_captures() as captures,
        ):
            codes = self._run_fw_bw_and_compare(
                fn, x, options={"graph_partition": not graph_partition}
            )

        self._assert_opt_out_captures(codes, captures)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
            "graph_partition": False,
            "triton.cudagraphs": True,
        }
    )
    @skip_if_caches_force_disabled
    def test_region_opt_out_warm_start(self):
        """A cache hit still leaves the opted-out region out of every capture."""
        from torch._dynamo.utils import counters

        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": False}
        )

        @torch.compiler.nested_compile_region(options=options)
        def region(x):
            return torch.sin(x) * 2

        def fn(x):
            return torch.cos(region(x + 1)) * 3

        x = torch.randn(16, 16, device=GPU_TYPE, requires_grad=True)
        for _ in range(2):
            torch._dynamo.reset()
            counters.clear()
            with self._record_captures() as captures:
                codes = self._run_fw_bw_and_compare(fn, x)
            self._assert_opt_out_captures(codes, captures)

        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 2)

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
        }
    )
    @parametrize("region_opts_in", (False, True))
    def test_cpp_wrapper_cannot_isolate_regional_cudagraphs(self, region_opts_in):
        """cpp_wrapper cannot partition, so a region cannot be isolated.

        An opt-in only asked for capture inside the region, so it is dropped
        rather than widened to the whole graph. An opt-out keeps the enclosing
        capture: losing every CUDA graph in the model is further from the
        request than failing to exclude one region.
        """
        from torch._dynamo.utils import counters

        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": region_opts_in}
        )

        @torch.compiler.nested_compile_region(options=options)
        def region(x):
            return torch.sin(x)

        def fn(x):
            return torch.cos(region(x + 1))

        x = torch.randn(16, 16, device=GPU_TYPE)
        with (
            torch._inductor.config.patch("triton.cudagraphs", not region_opts_in),
            self._record_captures() as captures,
        ):
            compiled_fn = torch.compile(
                fn, backend="inductor", fullgraph=True, options={"cpp_wrapper": True}
            )
            result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, fn(x))
        self.assertEqual(len(captures), 0 if region_opts_in else 1)
        # Only a dropped opt-in is a skip; an opt-out keeps the whole-graph capture.
        self.assertEqual(counters["inductor"]["cudagraph_skips"], int(region_opts_in))
        self.assertNotIn("partition_0", code)

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
    def test_cpp_wrapper_warm_start_realigns_inputs(self):
        """A cached graph whose regional capture was dropped (cpp_wrapper cannot
        isolate the region) still copies misaligned inputs on a warm start."""
        from torch._dynamo.utils import counters

        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True}
        )

        @torch.compiler.nested_compile_region(options=options)
        def region(x):
            return torch.sin(x) * 2

        def fn(x, weight):
            return torch.cos(region(x)) * weight + x

        # Large enough for vectorized loads, which fault on a misaligned input.
        weight = torch.randn(1 << 16, device=GPU_TYPE)
        misaligned = torch.randn((1 << 16) + 1, device=GPU_TYPE)[1:]
        for warm in (False, True):
            torch._dynamo.reset()
            counters.clear()
            compiled_fn = torch.compile(
                fn, backend="inductor", fullgraph=True, options={"cpp_wrapper": True}
            )
            for x in (torch.randn(1 << 16, device=GPU_TYPE), misaligned):
                self.assertEqual(compiled_fn(x, weight), fn(x, weight))
            # The dropped capture is reported on a cache hit too.
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], int(warm))

    @requires_gpu_and_triton
    @torch._inductor.config.patch(fx_graph_cache=False, fx_graph_remote_cache=False)
    @parametrize("region_cudagraphs", (False, True))
    def test_aot_inductor_regional_cudagraphs_smoke(self, region_cudagraphs):
        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": region_cudagraphs}
        )

        @torch.compiler.nested_compile_region(options=options)
        def region(x):
            return torch.sin(x)

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.cos(region(x + 1))

        x = torch.randn(16, 16, device=GPU_TYPE)
        exported = torch.export.export(Model(), (x,))
        (node,) = exported.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.invoke_subgraph
        )
        nested_config = node.meta["custom"]["nested_region_config"]
        self.assertEqual(
            nested_config.inductor_config_patches["triton.cudagraphs"],
            region_cudagraphs,
        )
        with torch._inductor.config.patch("triton.cudagraphs", not region_cudagraphs):
            package_path = torch._inductor.aoti_compile_and_package(exported)
        result = torch._inductor.aoti_load_package(package_path)(x)
        self.assertEqual(result, Model()(x))

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

        x = torch.randn(16, 16, device=GPU_TYPE)
        result, codes = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), x
        )

        self.assertEqual(result, fn(x))
        self.assertIn("def partition_0(args):", codes[0])
        self.assertEqual(codes[0].count("assert_size_stride("), 1)

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
        self.assertEqual(codes[0].count("def triton_"), 1)
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
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
            "graph_partition": False,
            "triton.cudagraphs": False,
        }
    )
    @skip_if_caches_force_disabled
    def test_nested_region_cudagraph_config_reuses_fx_graph_cache(self):
        from torch._dynamo.utils import counters

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
            counters.clear()
            with self._record_captures() as captures:
                result = torch.compile(fn, backend="inductor", fullgraph=True)(x)
            self.assertEqual(result, fn(x))
            # The cache hit still captures the region on its own.
            self.assertEqual(len(captures), 1)

        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
            "graph_partition": False,
            "triton.cudagraphs": False,
            # Too small for the one-matmul forward, not for the two-matmul
            # backward.
            "triton.cudagraph_min_partition_size": 2,
        }
    )
    @skip_if_caches_force_disabled
    @parametrize("forward_cudagraphs", (False, True))
    @parametrize("warm", (False, True))
    def test_backward_cannot_bring_back_lost_forward_capture(
        self, forward_cudagraphs, warm
    ):
        """A region opting in both ways captures its backward only if its
        forward was captured; a backward-only opt-in stands on its own."""
        from torch._dynamo.utils import counters

        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": forward_cudagraphs},
            bw_inductor_config_patches={"triton.cudagraphs": True},
        )

        @torch.compiler.nested_compile_region(options=options)
        def region(x, y):
            return x @ y

        def fn(x, y):
            return torch.sin(region(x, y))

        inputs = [
            torch.randn(16, 16, device=GPU_TYPE, requires_grad=True) for _ in range(2)
        ]
        if warm:
            self._run_fw_bw_and_compare(fn, *inputs)
            torch._dynamo.reset()
            counters.clear()
        with self._record_captures() as captures:
            self._run_fw_bw_and_compare(fn, *inputs)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], int(warm))
        self.assertEqual(
            [c["is_backward"] for c in captures], [] if forward_cudagraphs else [True]
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
    @parametrize(
        "forward_cudagraphs,backward_cudagraphs,region_needs_grad,checkpoint,index_put",
        (
            (True, False, True, False, False),
            (False, True, True, False, False),
            # The region has no backward, though its backward setting asks for
            # capture.
            (True, True, False, False, False),
            # As above, and lowering a boolean index_put records a cudagraph
            # disabled reason in the backward compiled without capture.
            (False, True, False, False, True),
            # Recomputed in the backward, the region runs under its backward
            # setting.
            (True, False, True, True, False),
        ),
    )
    def test_aot_autograd_warm_start_captures_like_cold_compile(
        self,
        forward_cudagraphs,
        backward_cudagraphs,
        region_needs_grad,
        checkpoint,
        index_put,
    ):
        from torch._dynamo.utils import counters

        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": forward_cudagraphs},
            bw_inductor_config_patches={"triton.cudagraphs": backward_cudagraphs},
        )

        @torch.compiler.nested_compile_region(options=options)
        def region(x):
            return torch.sin(x) * 2

        mask = torch.arange(16, device=GPU_TYPE) % 2 == 0
        values = torch.randn(16, device=GPU_TYPE)

        def block(x, weight):
            if index_put:
                weight = weight.index_put((mask,), values)
            return torch.cos(region(x)) * weight

        def fn(x, weight):
            if checkpoint:
                return torch.utils.checkpoint.checkpoint(
                    block, x, weight, use_reentrant=False
                )
            return block(x, weight)

        x = torch.randn(16, 16, device=GPU_TYPE, requires_grad=region_needs_grad)
        weight = torch.randn(16, 16, device=GPU_TYPE, requires_grad=True)
        runs = []
        for _ in range(2):
            torch._dynamo.reset()
            counters.clear()
            with self._record_captures() as captures:
                self._run_fw_bw_and_compare(fn, x, weight)
            runs.append(
                (
                    sorted(c["is_backward"] for c in captures),
                    counters["inductor"]["cudagraph_skips"],
                )
            )

        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 0)
        cold, warm = runs
        expected_captures = [False] * forward_cudagraphs + [True] * (
            backward_cudagraphs and region_needs_grad
        )
        self.assertEqual(cold[0], expected_captures)
        self.assertEqual(warm, cold)

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
    @parametrize("forward_cudagraphs", (False, True))
    def test_aot_autograd_warm_start_backward_follows_lost_forward_capture(
        self, forward_cudagraphs
    ):
        """On a cache hit whose backward was compiled with capture, a forward
        that loses capture in post_compile also stops a backward that shares its
        request; a backward-only opt-in stands on its own."""
        from torch._dynamo.utils import counters
        from torch._inductor.cudagraph_utils import CUDAGraphPolicy

        class DeclineForward(CUDAGraphPolicy):
            def should_wrap(self, compiled_graph):
                return compiled_graph.fx_kwargs["is_backward"]

        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": forward_cudagraphs},
            bw_inductor_config_patches={"triton.cudagraphs": True},
        )

        @torch.compiler.nested_compile_region(options=options)
        def region(x):
            return torch.sin(x) * 2

        def fn(x):
            return torch.cos(region(x))

        x = torch.randn(16, 16, device=GPU_TYPE, requires_grad=True)
        with self._record_captures() as captures:
            self._run_fw_bw_and_compare(fn, x)
        self.assertEqual(
            sorted(c["is_backward"] for c in captures),
            [False] * forward_cudagraphs + [True],
        )

        torch._dynamo.reset()
        counters.clear()
        # The policy is not part of the cache key.
        with (
            torch._inductor.config.patch(cudagraph_policy=DeclineForward()),
            self._record_captures() as captures,
        ):
            self._run_fw_bw_and_compare(fn, x)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(
            [c["is_backward"] for c in captures], [] if forward_cudagraphs else [True]
        )

    @requires_cuda_and_triton
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
            "triton.cudagraphs": True,
        }
    )
    @skip_if_caches_force_disabled
    def test_aot_autograd_cache_keys_on_backward_cudagraphs_override(self):
        """The override_cudagraphs annotation is stripped from the graph before
        hashing, so only the backward's box keeps bwd=False from reusing a
        backward that was compiled with capture."""
        from torch._dynamo.utils import counters

        def fn(x):
            return torch.cos(torch.sin(x) * 2)

        # fwd=True restates the config; it only gives both runs the same graph
        # inputs.
        restated = torch._dynamo.override_cudagraphs(fwd=True)(fn)
        no_backward = torch._dynamo.override_cudagraphs(fwd=True, bwd=False)(fn)
        x = torch.randn(16, 16, device=GPU_TYPE, requires_grad=True)
        with self._record_captures() as captures:
            self._run_fw_bw_and_compare(restated, x)
        self.assertEqual(sorted(c["is_backward"] for c in captures), [False, True])

        torch._dynamo.reset()
        counters.clear()
        with self._record_captures() as captures:
            self._run_fw_bw_and_compare(no_backward, x)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual([c["is_backward"] for c in captures], [False])

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

        x = torch.randn(16, 16, device=GPU_TYPE, requires_grad=True)
        with self._record_captures() as captures:
            codes = self._run_fw_bw_and_compare(fn, x)

        self.assertEqual(
            [c["is_backward"] for c in captures],
            [False] * forward_cudagraphs + [True] * backward_cudagraphs,
        )
        self.assertEqual(len(codes), 2)
        for code, subgraph_name, use_cudagraphs in zip(
            codes,
            ("partitioned_fw_subgraph_0_0(", "partitioned_bw_subgraph_0_0("),
            (forward_cudagraphs, backward_cudagraphs),
        ):
            partitions = self._partition_bodies(code)
            self.assertEqual(len(partitions), int(use_cudagraphs))
            # Only the region is captured; the code around it stays in call().
            for partition in partitions:
                self.assertIn(subgraph_name, partition)
                self.assertNotIn(".run(", partition)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "freezing": True,
            "graph_partition": False,
        }
    )
    @parametrize("region_opts_in", (False, True))
    def test_freezing_honors_regional_cudagraphs(self, region_opts_in):
        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": region_opts_in}
        )

        # The region takes the weight as an argument: Dynamo cannot trace a
        # region body that reads a parameter under freezing.
        @torch.compiler.nested_compile_region(options=options)
        def region(x, weight):
            return torch.relu(x @ weight)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(16, 16))
                self.head = torch.nn.Linear(16, 16)

            def forward(self, x):
                return self.head(region(x + 1, self.weight))

        model = Model().to(GPU_TYPE)
        x = torch.randn(8, 16, device=GPU_TYPE)
        with (
            torch.no_grad(),
            torch._inductor.config.patch("triton.cudagraphs", not region_opts_in),
            self._record_captures() as captures,
        ):
            expected = model(x)
            result, (code,) = run_and_get_code(
                torch.compile(model, backend="inductor", fullgraph=True), x
            )

        self.assertEqual(result, expected)
        partitions = self._partition_bodies(code)
        self.assertGreater(len(partitions), 0)
        self.assertEqual(len(captures), len(partitions))
        for partition in partitions:
            self.assertEqual("repeated_subgraph0(" in partition, region_opts_in)
            if region_opts_in:
                self.assertNotIn("extern_kernels.", partition)
                self.assertNotIn(".run(", partition)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": False,
            "triton.cudagraphs": False,
            # Too small to capture, so post-compile drops the backward's capture.
            "triton.cudagraph_min_partition_size": 1000,
        }
    )
    def test_backward_only_regional_cudagraph_survives_dropped_capture(self):
        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": False},
            bw_inductor_config_patches={"triton.cudagraphs": True},
        )

        @torch.compiler.nested_compile_region(options=nested_config)
        def region(x):
            return torch.sin(x) * 2.0

        def fn(x):
            return (torch.cos(x) + region(x)).sum()

        x = torch.randn(16, 16, device=GPU_TYPE, requires_grad=True)
        with self._record_captures() as captures:
            self._run_fw_bw_and_compare(fn, x)
        self.assertEqual(captures, [])

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
        "top_level,region_bw",
        (
            # A backward-only opt-in: only the region's backward is captured.
            (False, True),
            # An opt-out: everything but the region is captured, both ways.
            (True, False),
        ),
    )
    def test_captured_backward_partitions_record_once(self, top_level, region_bw):
        """Saved activations are static inputs of a captured backward partition
        only if they really are, so fresh inputs replay without re-recording."""
        from torch._inductor.cudagraph_trees import CUDAGraphTreeManager

        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": False},
            bw_inductor_config_patches={"triton.cudagraphs": region_bw},
        )

        @torch.compiler.nested_compile_region(options=options)
        def region(x):
            return torch.sin(x) * 2

        def fn(x, weight):
            return region(x * weight)

        with torch._inductor.config.patch("triton.cudagraphs", top_level):
            compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
            with (
                self._record_captures() as captures,
                mock.patch.object(
                    CUDAGraphTreeManager,
                    "record_function",
                    autospec=True,
                    side_effect=CUDAGraphTreeManager.record_function,
                ) as record_function,
            ):
                for _ in range(5):
                    garbage = [torch.randn(n, device=GPU_TYPE) for n in (64, 1 << 16)]
                    del garbage
                    inputs = [
                        torch.randn(32, 32, device=GPU_TYPE, requires_grad=True)
                        for _ in range(2)
                    ]
                    expected_inputs = [
                        x.detach().clone().requires_grad_() for x in inputs
                    ]
                    result = compiled_fn(*inputs)
                    result.sum().backward()
                    expected = fn(*expected_inputs)
                    expected.sum().backward()
                    self.assertEqual(result, expected)
                    self.assertEqual(
                        [x.grad for x in inputs], [x.grad for x in expected_inputs]
                    )

        self.assertIn(True, [c["is_backward"] for c in captures])
        if not top_level:
            self.assertEqual([c["is_backward"] for c in captures], [True])
        self.assertEqual(record_function.call_count, len(captures))

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
            "graph_partition": False,
            "triton.cudagraphs": True,
        }
    )
    @skip_if_caches_force_disabled
    @parametrize("uncaptured_by", ("cpu_op", "region_opt_out", "override_cudagraphs"))
    def test_warm_start_uncaptured_backward_lets_forward_replay(self, uncaptured_by):
        """A backward compiled without capture still tells the tree its
        forward's backward ran, also when it is loaded from the cache."""
        from torch._dynamo.utils import counters
        from torch._inductor.cudagraph_trees import CUDAGraphTreeManager, get_manager

        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": False}
        )

        @torch.compiler.nested_compile_region(options=options)
        def region(x):
            return torch.cos(x) * 3

        @torch._dynamo.override_cudagraphs(fwd=True, bwd=False)
        def forward_only_cudagraphs(h):
            return torch.sin(h)

        def fn(x, weight):
            h = (region(x) if uncaptured_by == "region_opt_out" else x) * weight
            if uncaptured_by == "override_cudagraphs":
                return forward_only_cudagraphs(h)
            return _CpuRoundTripBackward.apply(h)

        for _ in range(2):
            torch._dynamo.reset()
            counters.clear()
            compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
            with mock.patch.object(
                CUDAGraphTreeManager,
                "record_function",
                autospec=True,
                side_effect=CUDAGraphTreeManager.record_function,
            ) as record_function:
                for _ in range(4):
                    inputs = [
                        torch.randn(64, 64, device=GPU_TYPE, requires_grad=True)
                        for _ in range(2)
                    ]
                    expected_inputs = [
                        x.detach().clone().requires_grad_() for x in inputs
                    ]
                    compiled_fn(*inputs).sum().backward()
                    manager = get_manager(
                        torch.cuda.current_device(), create_if_none_exists=False
                    )
                    self.assertFalse(manager.running_forwards_with_pending_backwards)
                    fn(*expected_inputs).sum().backward()
                    self.assertEqual(
                        [x.grad for x in inputs], [x.grad for x in expected_inputs]
                    )
            self.assertEqual(record_function.call_count, 1)

        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 2)

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

        region = gm.child.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.invoke_subgraph
        )[0]
        backward_config = region.meta["custom"]["nested_region_config"]
        self.assertFalse(backward_config.inductor_config_patches["triton.cudagraphs"])
        self.assertIsNone(backward_config.bw_inductor_config_patches)

    @parametrize("get_container", (False, True))
    def test_backward_config_specialization_copies_modules_it_changes(
        self, get_container
    ):
        """A nested module the forward also reaches is copied, not edited."""
        from torch._higher_order_ops.invoke_subgraph import (
            _specialize_nested_region_configs_for_backward,
        )
        from torch._inductor.compile_fx import _get_invoke_subgraph_graph_module

        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True},
            bw_inductor_config_patches={"triton.cudagraphs": False},
        )
        body = self._empty_graph_module()
        body.meta["nested_region_config"] = nested_config
        shared = self._configured_region_graph_module(nested_config, body)
        root = torch.nn.Module()
        root.add_module("submod", torch.nn.Module())
        root.submod.add_module("shared", shared)

        def calls_shared():
            graph = torch.fx.Graph()
            if get_container:
                # Getting the container whole makes FX install it by reference,
                # so the graphs share the container too, not just the module.
                graph.get_attr("submod")
            graph.call_function(
                torch.ops.higher_order.invoke_subgraph,
                (graph.get_attr("submod.shared"),),
            )
            graph.output(())
            return torch.fx.GraphModule(root, graph)

        def region(gm):
            return gm.graph.find_nodes(
                op="call_function", target=torch.ops.higher_order.invoke_subgraph
            )[0]

        def region_body(gm):
            return _get_invoke_subgraph_graph_module(gm, region(gm))

        # Both graphs reach the same nested module, like a region body the proxy
        # dispatch cache handed to the forward and the backward.
        forward, backward = calls_shared(), calls_shared()
        self.assertIs(region_body(forward), region_body(backward))
        self.assertEqual(forward.submod is backward.submod, get_container)

        _specialize_nested_region_configs_for_backward(backward)

        self.assertIs(forward.submod.shared, shared)
        self.assertIs(region_body(forward), shared)
        forward_config = region(shared).meta["custom"]["nested_region_config"]
        self.assertIs(forward_config, nested_config)
        self.assertIs(body.meta["nested_region_config"], nested_config)
        backward_shared = region_body(backward)
        self.assertIsNot(backward_shared, shared)
        for config in (
            region(backward_shared).meta["custom"]["nested_region_config"],
            region_body(backward_shared).meta["nested_region_config"],
        ):
            self.assertFalse(config.inductor_config_patches["triton.cudagraphs"])

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

        A mirror is seeded with setdefault, first writer wins, so a module the
        forward and the backward both reach can still hold the forward's
        config when the backward is lowered. Simulate that by restoring the
        forward config on every region body after the backward is specialized.
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

        x = torch.randn(16, 16, device=GPU_TYPE, requires_grad=True)
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
        backward_partitions = self._partition_bodies(codes[1])
        self.assertGreater(len(backward_partitions), 0)
        for partition in backward_partitions:
            self.assertNotIn("partitioned_bw_subgraph_0_0(", partition)
        self.assertIn("partitioned_bw_subgraph_0_0(", codes[1])

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._functorch.config.patch(enable_autograd_cache=False)
    @torch._inductor.config.patch(
        {"fx_graph_cache": False, "fx_graph_remote_cache": False}
    )
    def test_recomputed_region_backward_passes_run_under_backward_patches(self):
        from torch._inductor.compile_fx import post_grad_passes

        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"max_autotune": True},
            bw_inductor_config_patches={"max_autotune": False},
        )

        @torch.compiler.nested_compile_region(options=nested_config)
        def region(x):
            return torch.sin(x) * 2

        def fn(x):
            return torch.utils.checkpoint.checkpoint(region, x, use_reentrant=False)

        region_max_autotune = []

        def record_region_config(gm, *args, **kwargs):
            if "nested_region_config" in gm.meta:
                region_max_autotune.append(torch._inductor.config.max_autotune)
            return post_grad_passes(gm, *args, **kwargs)

        x = torch.randn(8, requires_grad=True)
        with mock.patch(
            "torch._inductor.compile_fx.post_grad_passes", record_region_config
        ):
            codes = self._run_fw_bw_and_compare(fn, x)

        self.assertIn("partitioned_fw_subgraph_0_0(", codes[1])
        # The forward body, then the backward body and the forward body
        # recomputed in the backward.
        self.assertEqual(region_max_autotune, [True, False, False])

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._functorch.config.patch(enable_autograd_cache=False)
    @torch._inductor.config.patch(
        {"fx_graph_cache": False, "fx_graph_remote_cache": False}
    )
    def test_nested_region_backward_passes_run_under_backward_patches(self):
        from torch._inductor.compile_fx import post_grad_passes

        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"max_autotune": True},
            bw_inductor_config_patches={"max_autotune": False},
        )

        @torch.compiler.nested_compile_region(options=nested_config)
        def inner(x):
            return x.sin()

        @torch.compiler.nested_compile_region
        def outer(x):
            return inner(x) * x

        def fn(x):
            return outer(x).sum()

        region_max_autotune = []

        def record_region_config(gm, *args, **kwargs):
            if "nested_region_config" in gm.meta:
                region_max_autotune.append(torch._inductor.config.max_autotune)
            return post_grad_passes(gm, *args, **kwargs)

        x = torch.randn(4, requires_grad=True)
        with mock.patch(
            "torch._inductor.compile_fx.post_grad_passes", record_region_config
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(x).backward()

        # inner's forward body, then its backward body inside outer's backward.
        self.assertEqual(region_max_autotune, [True, False])

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "triton.cudagraphs": False,
        }
    )
    @parametrize(
        "outer_fw,outer_bw,inner_fw,inner_bw,direction",
        (
            # The forwards conflict, with each backward reusing its forward config.
            (True, None, False, None, "forward"),
            # The forwards agree and only the backwards conflict.
            (True, True, True, False, "backward"),
            # As above, with the inner backward inheriting its forward config.
            (True, False, True, None, "backward"),
            # A plain region encloses its regions with the top-level setting.
            (None, None, True, None, "forward"),
        ),
    )
    # Activation checkpointing doesn't open a scope of its own.
    @parametrize("checkpoint", (False, True))
    def test_nested_region_rejects_conflicting_cudagraph_configs(
        self, outer_fw, outer_bw, inner_fw, inner_bw, direction, checkpoint
    ):
        def options(fw, bw):
            if fw is None:
                return None
            bw_patches = None if bw is None else {"triton.cudagraphs": bw}
            return get_invoke_subgraph_compile_options(
                fw_inductor_config_patches={"triton.cudagraphs": fw},
                bw_inductor_config_patches=bw_patches,
            )

        @torch.compiler.nested_compile_region(options=options(inner_fw, inner_bw))
        def inner_region(x):
            return torch.sin(x)

        @torch.compiler.nested_compile_region(options=options(outer_fw, outer_bw))
        def outer_region(x):
            return torch.cos(inner_region(x))

        def fn(x):
            if checkpoint:
                return torch.utils.checkpoint.checkpoint(
                    outer_region, x, use_reentrant=False
                )
            return outer_region(x) + 1

        inherited = direction == "backward" and inner_bw is None
        inner_value = inner_fw if direction == "forward" or inherited else inner_bw
        if inherited:
            direction += re.escape(
                " (inherited from its forward; set triton.cudagraphs in "
                "bw_inductor_config_patches to override)"
            )
        with self.assertRaisesRegex(
            BackendCompilerFailed,
            f"(?s)ValueError: nested compile region '.*' sets "
            f"triton.cudagraphs={inner_value} for its {direction}, but the region "
            f"enclosing it resolves to {not inner_value}.*Called from:.*"
            r"return torch\.cos\(inner_region\(x\)\)",
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(
                torch.randn(4, requires_grad=True)
            )

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": False,
        }
    )
    @parametrize("conflict", ("nested_region", "dynamo_override"))
    @parametrize("requires_grad", (False, True))
    def test_backward_only_cudagraph_conflict_needs_a_backward(
        self, conflict, requires_grad
    ):
        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True},
            bw_inductor_config_patches={"triton.cudagraphs": True},
        )
        inner_options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True},
            bw_inductor_config_patches={"triton.cudagraphs": False},
        )

        @torch.compiler.nested_compile_region(options=inner_options)
        def inner_region(x):
            return torch.sin(x)

        @torch.compiler.nested_compile_region(options=options)
        def outer_region(x):
            return torch.cos(inner_region(x))

        @torch.compiler.nested_compile_region(options=options)
        def region(x):
            return torch.sin(x)

        def nested_fn(x):
            return outer_region(x) + 1

        @torch._dynamo.override_cudagraphs(bwd=False)
        def override_fn(x):
            return torch.cos(region(x))

        fn, top_level = {
            "nested_region": (nested_fn, False),
            "dynamo_override": (override_fn, True),
        }[conflict]
        x = torch.randn(16, 16, device=GPU_TYPE, requires_grad=requires_grad)
        compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        with torch._inductor.config.patch("triton.cudagraphs", top_level):
            if requires_grad:
                with self.assertRaisesRegex(BackendCompilerFailed, "ValueError"):
                    compiled_fn(x)
            else:
                self.assertEqual(compiled_fn(x), fn(x))

    @requires_cuda_and_triton
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": False,
            "triton.cudagraphs": False,
        }
    )
    @parametrize("calls", (1, 2))
    def test_single_use_plain_region_does_not_enclose_configured_region(self, calls):
        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True}
        )

        @torch.compiler.nested_compile_region(options=options)
        def inner_region(x):
            return torch.sin(x)

        @torch.compiler.nested_compile_region
        def plain_region(x):
            return torch.cos(inner_region(x))

        def fn(x):
            for _ in range(calls):
                x = plain_region(x)
            return x + 1

        # Dynamo inlines a plain region called once, which leaves the configured
        # region first-level. A plain region called more than once stays a region
        # and encloses it.
        x = torch.randn(16, 16, device=GPU_TYPE)
        compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        if calls == 1:
            with self._record_captures() as captures:
                self.assertEqual(compiled_fn(x), fn(x))
            self.assertEqual(len(captures), 1)
        else:
            with self.assertRaisesRegex(
                BackendCompilerFailed,
                "only first-level regions can set their own cudagraphs",
            ):
                compiled_fn(x)

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
    def test_nested_region_restating_enclosing_cudagraphs_is_allowed(self):
        options = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": True}
        )

        @torch.compiler.nested_compile_region(options=options)
        def inner_region(x):
            return torch.sin(x)

        @torch.compiler.nested_compile_region(options=options)
        def outer_region(x):
            return torch.cos(inner_region(x))

        def fn(x):
            return outer_region(x) + 1

        x = torch.randn(16, 16, device=GPU_TYPE)
        with self._record_captures() as captures:
            result, (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), x
            )

        self.assertEqual(result, fn(x))
        self.assertEqual(len(captures), 1)
        (partition,) = self._partition_bodies(code)
        self.assertIn("repeated_subgraph", partition)
        self.assertNotIn(".run(", partition)

    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
        }
    )
    @parametrize(
        "top_level,fwd,bwd,region_fw,region_bw",
        (
            # The annotation moves the forward, where the region differs.
            (False, True, None, False, None),
            # The annotation moves the backward, where the region differs.
            (True, None, False, True, True),
        ),
    )
    def test_nested_region_cudagraph_rejects_dynamo_override(
        self, top_level, fwd, bwd, region_fw, region_bw
    ):
        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": region_fw},
            bw_inductor_config_patches=(
                None if region_bw is None else {"triton.cudagraphs": region_bw}
            ),
        )

        @torch.compiler.nested_compile_region(options=nested_config)
        def region(x):
            return torch.sin(x)

        @torch._dynamo.override_cudagraphs(fwd=fwd, bwd=bwd)
        def fn(x):
            return torch.cos(region(x))

        with (
            torch._inductor.config.patch("triton.cudagraphs", top_level),
            self.assertRaisesRegex(
                BackendCompilerFailed,
                "ValueError: torch._dynamo.override_cudagraphs cannot be combined "
                "with a nested compile region whose triton.cudagraphs differs",
            ),
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(
                torch.randn(4, requires_grad=True)
            )

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
        "top_level,fwd,bwd,region_cudagraphs,region_in_partitions,captures",
        (
            # The annotation moves only the backward, where the region agrees.
            (True, None, False, False, ([False], []), [False]),
            # The annotation restates the ambient setting.
            (True, True, None, False, ([False], [False]), [False, True]),
            (False, None, False, True, ([True], [True]), [False, True]),
            # The region restates the annotation, so it is not regional and
            # each graph is captured whole.
            (False, True, True, True, ([], []), [False, True]),
        ),
    )
    def test_dynamo_override_allows_unambiguous_regional_cudagraphs(
        self, top_level, fwd, bwd, region_cudagraphs, region_in_partitions, captures
    ):
        nested_config = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches={"triton.cudagraphs": region_cudagraphs}
        )

        @torch.compiler.nested_compile_region(options=nested_config)
        def region(x):
            return torch.sin(x)

        @torch._dynamo.override_cudagraphs(fwd=fwd, bwd=bwd)
        def fn(x):
            return torch.cos(region(x)) * 2

        x = torch.randn(16, 16, device=GPU_TYPE, requires_grad=True)
        with (
            torch._inductor.config.patch("triton.cudagraphs", top_level),
            self._record_captures() as recorded,
        ):
            codes = self._run_fw_bw_and_compare(fn, x)

        self.assertEqual(len(codes), 2)
        self.assertEqual(
            tuple(
                [
                    "_subgraph_0_0(" in partition
                    for partition in self._partition_bodies(code)
                ]
                for code in codes
            ),
            region_in_partitions,
        )
        self.assertEqual(sorted(c["is_backward"] for c in recorded), captures)


if __name__ == "__main__":
    run_tests()
