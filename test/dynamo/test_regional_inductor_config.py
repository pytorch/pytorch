# Owner(s): ["module: dynamo"]

import contextlib
import re
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


@torch.library.custom_op("regional_cudagraph_test::plain_clone", mutates_args=())
def _plain_clone(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_plain_clone.register_fake
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
        """Collect the kwargs of every cudagraphify wrapper built at compile time,
        one per graph or partition set up for capture. Recording happens later,
        when the compiled function runs: count that with _num_recorded_graphs."""
        from torch._inductor.compile_fx import cudagraphify

        captures = []

        def record(*args, **kwargs):
            captures.append(kwargs)
            return cudagraphify(*args, **kwargs)

        with mock.patch("torch._inductor.compile_fx.cudagraphify", record):
            yield captures

    @staticmethod
    def _num_recorded_graphs():
        """CUDA graphs recorded by the tree manager so far. Consumes a graph id,
        so call it once, after the last run."""
        from torch._inductor.cudagraph_trees import get_manager

        manager = get_manager(torch.cuda.current_device(), create_if_none_exists=False)
        return 0 if manager is None else manager.new_graph_id().id

    def _run_and_get_code_and_recordings(self, fn, *inputs):
        """Run torch.compile(fn) three times, so that the CUDA graph tree warms
        up, records and replays, comparing every result with eager. Returns the
        generated code and the number of CUDA graphs recorded."""
        compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        expected = fn(*inputs)
        result, (code,) = run_and_get_code(compiled_fn, *inputs)
        self.assertEqual(result, expected)
        for _ in range(2):
            self.assertEqual(compiled_fn(*inputs), expected)
        return code, self._num_recorded_graphs()

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
            return region(x[:n] + 1), torch.cos(x)

        x = torch.randn(64, 16, device=GPU_TYPE)
        code, recorded = self._run_and_get_code_and_recordings(fn, x)

        self.assertIn("= repeated_subgraph0(", code)
        partitions = self._partition_bodies(code)
        self.assertGreater(len(partitions), 0)
        self.assertEqual(recorded, len(partitions))
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
            "custom_should_partition_op",
            "unbacked",
            "cond",
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
            "custom_should_partition_op": lambda x: _plain_clone(torch.sin(x)),
            "unbacked": lambda x: x + torch.nonzero(x > 0).sum(),
            "cond": lambda x: torch.cond(x.sum() > 0, torch.sin, torch.cos, (x,)),
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
        custom_should_partition_ops = (
            ["regional_cudagraph_test::plain_clone"]
            if body == "custom_should_partition_op"
            else []
        )
        with torch._inductor.config.patch(
            custom_should_partition_ops=custom_should_partition_ops
        ):
            code, recorded = self._run_and_get_code_and_recordings(fn, x)

        # The MultiOutputs that unpack the region's results must stay with it.
        (region_buf,) = re.findall(r"(\w+) = repeated_subgraph0\(", code)
        partitions = self._partition_bodies(code)
        self.assertGreater(len(partitions), 0)
        self.assertEqual(recorded, len(partitions))
        for partition in partitions:
            self.assertNotIn("repeated_subgraph0(", partition)
            self.assertNotIn(f"{region_buf}[", partition)
        if body == "nested_region":
            # The inner region must survive as its own call for the body walk
            # to recurse into it.
            body_code = self._generated_fn_body(code, "def repeated_subgraph0(")
            self.assertRegex(body_code, r"= \w*repeated_subgraph\w*\(")

    @requires_cuda_and_triton
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._inductor.config.patch(
        {
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "graph_partition": True,
            "triton.cudagraphs": True,
            "triton.cudagraph_skip_dynamic_graphs": True,
        }
    )
    def test_dynamic_region_body_stays_out_of_partitions(self):
        @torch.compiler.nested_compile_region
        def region(x):
            return torch.sin(x).sum(0)

        def fn(x, weight):
            return torch.cos(region(x) @ weight)

        x = torch.randn(8, 16, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(x, 0)
        weight = torch.randn(16, 16, device=GPU_TYPE)
        code, recorded = self._run_and_get_code_and_recordings(fn, x, weight)

        self.assertIn("= repeated_subgraph0(", code)
        partitions = self._partition_bodies(code)
        self.assertGreater(len(partitions), 0)
        self.assertEqual(recorded, len(partitions))
        for partition in partitions:
            self.assertNotIn("repeated_subgraph0(", partition)

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
    def test_uncapturable_region_body_stays_out_of_partitions_training(self):
        @torch.compiler.nested_compile_region
        def region(x):
            return torch.sin(x).cpu().to(GPU_TYPE) * 2

        def fn(x, weight):
            return torch.cos(region(x @ weight))

        inputs = [
            torch.randn(16, 16, device=GPU_TYPE, requires_grad=True) for _ in range(2)
        ]
        compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        with self._record_captures() as captures:
            for i in range(3):
                compiled_inputs, eager_inputs = (
                    [x.detach().clone().requires_grad_() for x in inputs]
                    for _ in range(2)
                )
                if i == 0:
                    result, codes = run_fw_bw_and_get_code(
                        lambda: compiled_fn(*compiled_inputs)
                    )
                else:
                    result = compiled_fn(*compiled_inputs)
                    result.sum().backward()
                expected = fn(*eager_inputs)
                expected.sum().backward()
                self.assertEqual(result, expected)
                self.assertEqual(
                    [x.grad for x in compiled_inputs], [x.grad for x in eager_inputs]
                )

        self.assertEqual(len(codes), 2)
        partitions = [self._partition_bodies(code) for code in codes]
        self.assertEqual(
            sorted(c["is_backward"] for c in captures),
            [False] * len(partitions[0]) + [True] * len(partitions[1]),
        )
        self.assertEqual(self._num_recorded_graphs(), len(captures))
        for code, code_partitions, subgraph_name in zip(
            codes,
            partitions,
            ("partitioned_fw_subgraph_0_0(", "partitioned_bw_subgraph_0_0("),
        ):
            self.assertIn(f"= {subgraph_name}", code)
            self.assertGreater(len(code_partitions), 0)
            for partition in code_partitions:
                self.assertNotIn(subgraph_name, partition)

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
    @parametrize("nested", (False, True))
    def test_region_body_kernels_count_toward_min_partition_size(self, nested):
        """The region call site is a couple of scheduler nodes, but its body
        holds three matmuls (two of them in a nested region when nested=True),
        which is what the partition size should weigh."""

        @torch.compiler.nested_compile_region
        def inner(x, weight0, weight1):
            return (x @ weight0) @ weight1

        @torch.compiler.nested_compile_region
        def region(x, weight0, weight1, weight2):
            if nested:
                return inner(x, weight0, weight1) @ weight2
            return ((x @ weight0) @ weight1) @ weight2

        inputs = [torch.randn(16, 16, device=GPU_TYPE) for _ in range(4)]
        code, recorded = self._run_and_get_code_and_recordings(region, *inputs)

        (partition,) = self._partition_bodies(code)
        self.assertIn("= repeated_subgraph0(", partition)
        self.assertEqual(recorded, 1)
        if nested:
            body_code = self._generated_fn_body(code, "def repeated_subgraph0(")
            self.assertRegex(body_code, r"= \w*repeated_subgraph\w*\(")

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
        code, recorded = self._run_and_get_code_and_recordings(fn, *inputs)

        self.assertEqual(code.count("= repeated_subgraph0("), num_calls)
        captured = num_calls == 2
        self.assertEqual(len(self._partition_bodies(code)), int(captured))
        self.assertEqual(recorded, int(captured))


if __name__ == "__main__":
    run_tests()
