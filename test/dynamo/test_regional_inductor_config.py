# Owner(s): ["module: dynamo"]

import contextlib
import copy
import re
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
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": True,
            "triton.cudagraphs": True,
        }
    )
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
    def test_uncapturable_region_body_stays_out_of_partitions(self, body):
        """A region is judged by its body, not as one opaque call node."""

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
        region = torch.compiler.nested_compile_region(bodies[body])

        def fn(x):
            out = region(x)
            if body == "multi_output":
                return torch.cos(out[0]) * out[1]
            return torch.cos(out)

        x = torch.randn(16, 16, device=GPU_TYPE)
        with self._record_captures() as captures:
            result, (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), x
            )

        self.assertEqual(result, fn(x))
        # The MultiOutputs that unpack the region's results must stay with it.
        (region_buf,) = re.findall(r"(\w+) = repeated_subgraph0\(", code)
        partitions = self._partition_bodies(code)
        self.assertGreater(len(partitions), 0)
        self.assertEqual(len(captures), len(partitions))
        for partition in partitions:
            self.assertNotIn("repeated_subgraph0(", partition)
            self.assertNotIn(f"{region_buf}[", partition)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": True,
            "triton.cudagraphs": True,
            "triton.cudagraph_min_partition_size": 3,
        }
    )
    def test_region_body_kernels_count_toward_min_partition_size(self):
        # The region call site is a couple of scheduler nodes, but its body
        # holds four kernels, which is what the partition size should weigh.
        @torch.compiler.nested_compile_region
        def region(x, weight0, weight1, weight2, weight3):
            return ((x @ weight0) @ weight1 @ weight2) @ weight3

        inputs = [torch.randn(16, 16, device=GPU_TYPE) for _ in range(5)]
        with self._record_captures() as captures:
            result, (code,) = run_and_get_code(
                torch.compile(region, backend="inductor", fullgraph=True), *inputs
            )

        self.assertEqual(result, region(*inputs))
        (partition,) = self._partition_bodies(code)
        self.assertIn("repeated_subgraph0(", partition)
        self.assertEqual(len(captures), 1)

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
    @parametrize("region_opts_in", (False,))
    def test_cpp_wrapper_cannot_isolate_regional_cudagraphs(self, region_opts_in):
        """cpp_wrapper cannot partition, so a region cannot be isolated.

        An opt-out keeps the enclosing capture: losing every CUDA graph in the
        model is further from the request than failing to exclude one region.
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


if __name__ == "__main__":
    run_tests()
