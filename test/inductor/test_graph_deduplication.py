# Owner(s): ["module: inductor"]
from unittest import mock

import torch
import torch._dynamo
from torch._inductor.test_case import run_tests, TestCase


class GraphDeduplicationInductorWrapperTests(TestCase):
    def tearDown(self):
        torch._dynamo.reset()
        super().tearDown()

    def _compile_and_count_invoke_subgraphs(
        self,
        *,
        graph_deduplication=True,
        cpp_wrapper=False,
        fallback_by_default=False,
        dynamic=None,
        triton_cudagraphs=None,
    ):
        class RecordingInductorWrapper(torch._TorchCompileInductorWrapper):
            def __init__(self):
                options = {
                    "cpp_wrapper": cpp_wrapper,
                    "fallback_by_default": fallback_by_default,
                    "graph_deduplication": graph_deduplication,
                }
                if triton_cudagraphs is not None:
                    options["triton.cudagraphs"] = triton_cudagraphs
                super().__init__(
                    mode=None,
                    options=options,
                    dynamic=dynamic,
                )
                self.graphs = []

            def __call__(self, gm, example_inputs, *, config_patches=None):
                self.graphs.append(gm)
                return gm.forward

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return x + self.linear(x).relu()

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Block() for _ in range(3)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        backend = RecordingInductorWrapper()
        opt_model = torch._dynamo.optimize(backend)(Model())
        opt_model(torch.randn(2, 4))
        self.assertEqual(len(backend.graphs), 1)
        return sum(
            node.op == "call_function"
            and node.target is torch.ops.higher_order.invoke_subgraph
            for node in backend.graphs[0].graph.nodes
        )

    def _compile_string_inductor_and_count_invoke_subgraphs(self, *, dynamic=None):
        import torch._inductor.compile_fx as compile_fx

        graphs = []

        def fake_compile_fx(gm, example_inputs, **kwargs):
            graphs.append(gm)
            return gm.forward

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return x + self.linear(x).relu()

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Block() for _ in range(3)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        with mock.patch.object(compile_fx, "compile_fx", side_effect=fake_compile_fx):
            opt_model = torch._dynamo.optimize("inductor", dynamic=dynamic)(Model())
            opt_model(torch.randn(2, 4))

        self.assertEqual(len(graphs), 1)
        return sum(
            node.op == "call_function"
            and node.target is torch.ops.higher_order.invoke_subgraph
            for node in graphs[0].graph.nodes
        )

    def test_inductor_wrapper_enables_graph_deduplication(self):
        self.assertGreater(
            self._compile_and_count_invoke_subgraphs(graph_deduplication=True),
            0,
        )

    def test_optimize_string_inductor_enables_graph_deduplication(self):
        self.assertGreater(
            self._compile_string_inductor_and_count_invoke_subgraphs(),
            0,
        )

    def test_optimize_string_inductor_disables_graph_deduplication_for_dynamic_shapes(
        self,
    ):
        self.assertEqual(
            self._compile_string_inductor_and_count_invoke_subgraphs(dynamic=True),
            0,
        )

    def test_optimize_string_inductor_disables_graph_deduplication_for_cudagraphs(self):
        with torch._inductor.config.patch("triton.cudagraphs", True):
            self.assertEqual(
                self._compile_string_inductor_and_count_invoke_subgraphs(),
                0,
            )

    def test_optimize_string_inductor_disables_graph_deduplication_for_complex_wrapper(
        self,
    ):
        with torch._functorch.config.patch(enable_complex_wrapper=True):
            self.assertEqual(
                self._compile_string_inductor_and_count_invoke_subgraphs(),
                0,
            )

    def test_optimize_string_inductor_disables_graph_deduplication_for_custom_passes(
        self,
    ):
        def custom_pass(gm):
            return None

        for config_name in (
            "post_grad_custom_pre_pass",
            "post_grad_custom_post_pass",
            "joint_custom_pre_pass",
            "joint_custom_post_pass",
            "pre_grad_custom_pass",
        ):
            with self.subTest(config_name=config_name):
                with torch._inductor.config.patch({config_name: custom_pass}):
                    self.assertEqual(
                        self._compile_string_inductor_and_count_invoke_subgraphs(),
                        0,
                    )

    def test_optimize_string_inductor_disables_graph_deduplication_for_overlap_scheduling(
        self,
    ):
        for config_name in (
            "reorder_for_compute_comm_overlap",
            "aten_distributed_optimizations.enable_overlap_scheduling",
        ):
            with self.subTest(config_name=config_name):
                with torch._inductor.config.patch({config_name: True}):
                    self.assertEqual(
                        self._compile_string_inductor_and_count_invoke_subgraphs(),
                        0,
                    )

    def test_optimize_string_inductor_disables_graph_deduplication_for_collective_bucketing(
        self,
    ):
        with torch._inductor.config.patch(
            {"aten_distributed_optimizations.collective_bucketing": True}
        ):
            self.assertEqual(
                self._compile_string_inductor_and_count_invoke_subgraphs(),
                0,
            )

    def test_inductor_wrapper_can_disable_graph_deduplication(self):
        self.assertEqual(
            self._compile_and_count_invoke_subgraphs(graph_deduplication=False),
            0,
        )

    def test_inductor_wrapper_keeps_different_targets_separate(self):
        class RecordingInductorWrapper(torch._TorchCompileInductorWrapper):
            def __init__(self):
                super().__init__(
                    mode=None,
                    options={"graph_deduplication": True},
                    dynamic=None,
                )
                self.graphs = []

            def __call__(self, gm, example_inputs, *, config_patches=None):
                self.graphs.append(gm)
                return gm.forward

        def non_zero_rand(size, dtype):
            a = torch.rand(size=size, dtype=dtype)
            return a + (a == 0).to(dtype)

        def fn():
            dtype = torch.complex128
            a = non_zero_rand((2, 2), dtype=dtype)
            b = non_zero_rand((2, 2), dtype=dtype)
            c = non_zero_rand((2, 2), dtype=dtype)
            alpha = 0.5 * (1 + 1j)

            expected = a + (alpha * b) / c
            actual = torch.addcdiv(a, b, c, value=alpha)
            return expected, actual

        backend = RecordingInductorWrapper()
        opt_fn = torch._dynamo.optimize(backend, nopython=True)(fn)
        expected, actual = opt_fn()
        self.assertEqual(len(backend.graphs), 1)
        self.assertEqual(expected, actual)

    def test_inductor_wrapper_keeps_distinct_python_targets_separate(self):
        class RecordingInductorWrapper(torch._TorchCompileInductorWrapper):
            def __init__(self):
                super().__init__(
                    mode=None,
                    options={"graph_deduplication": True},
                    dynamic=None,
                )
                self.graphs = []

            def __call__(self, gm, example_inputs, *, config_patches=None):
                self.graphs.append(gm)
                return gm.forward

        def make(scale):
            @torch._dynamo.allow_in_graph
            def inner(x):
                return x * scale

            return inner

        f2 = make(2)
        f3 = make(3)

        def fn(x):
            return f2(x) + 1, f3(x) + 1

        backend = RecordingInductorWrapper()
        opt_fn = torch._dynamo.optimize(backend, nopython=True)(fn)
        x = torch.randn(4)
        self.assertEqual(opt_fn(x), fn(x))
        self.assertEqual(len(backend.graphs), 1)

    def test_inductor_wrapper_disables_graph_deduplication_for_cpp_wrapper(self):
        self.assertEqual(
            self._compile_and_count_invoke_subgraphs(
                graph_deduplication=True, cpp_wrapper=True
            ),
            0,
        )

    def test_inductor_wrapper_disables_graph_deduplication_when_grad_disabled(self):
        with torch.no_grad():
            self.assertEqual(
                self._compile_and_count_invoke_subgraphs(graph_deduplication=True),
                0,
            )

    def test_inductor_wrapper_disables_graph_deduplication_for_dynamic_shapes(self):
        self.assertEqual(
            self._compile_and_count_invoke_subgraphs(
                graph_deduplication=True, dynamic=True
            ),
            0,
        )

    def test_inductor_wrapper_disables_graph_deduplication_for_lite_mode(self):
        self.assertEqual(
            self._compile_and_count_invoke_subgraphs(
                graph_deduplication=True, fallback_by_default=True
            ),
            0,
        )

    def test_inductor_wrapper_disables_graph_deduplication_for_cudagraphs(self):
        self.assertEqual(
            self._compile_and_count_invoke_subgraphs(
                graph_deduplication=True, triton_cudagraphs=True
            ),
            0,
        )

    def test_inductor_wrapper_disables_graph_deduplication_for_regional_compile(self):
        with torch._dynamo.config.patch(enable_invoke_subgraph_regional_compile=True):
            self.assertEqual(
                self._compile_and_count_invoke_subgraphs(graph_deduplication=True),
                0,
            )

    def test_inductor_wrapper_disables_graph_deduplication_for_trace_autograd_ops(self):
        with torch._dynamo.config.patch(trace_autograd_ops=True):
            self.assertEqual(
                self._compile_and_count_invoke_subgraphs(graph_deduplication=True),
                0,
            )

    def test_inductor_wrapper_disables_graph_deduplication_for_compiled_autograd(self):
        import torch._dynamo.compiled_autograd as compiled_autograd

        backend = torch._TorchCompileInductorWrapper(
            mode=None,
            options={"graph_deduplication": True},
            dynamic=None,
        )
        prior = compiled_autograd.in_compiled_autograd_region
        try:
            compiled_autograd.in_compiled_autograd_region = True
            with (
                torch._dynamo.config.patch(use_graph_deduplication=False),
                backend.backend_ctx_ctor(),
            ):
                self.assertFalse(torch._dynamo.config.use_graph_deduplication)
        finally:
            compiled_autograd.in_compiled_autograd_region = prior

    def test_inductor_wrapper_disables_graph_deduplication_for_compile_subprocess(self):
        import torch._inductor.compile_fx as compile_fx

        with mock.patch.object(
            compile_fx, "fx_compile_mode", compile_fx.FxCompileMode.SUBPROCESS
        ):
            self.assertEqual(
                self._compile_and_count_invoke_subgraphs(graph_deduplication=True),
                0,
            )

    def test_inductor_wrapper_skips_mutating_external_inputs(self):
        class RecordingInductorWrapper(torch._TorchCompileInductorWrapper):
            def __init__(self):
                super().__init__(
                    mode=None,
                    options={"graph_deduplication": True},
                    dynamic=None,
                )
                self.graphs = []

            def __call__(self, gm, example_inputs, *, config_patches=None):
                self.graphs.append(gm)
                return gm.forward

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn1 = torch.nn.BatchNorm2d(3)
                self.bn2 = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                return self.bn2(self.bn1(x))

        backend = RecordingInductorWrapper()
        opt_model = torch._dynamo.optimize(backend, nopython=True)(Model().train())
        opt_model(torch.randn(2, 3, 4, 4))
        self.assertEqual(len(backend.graphs), 1)
        self.assertEqual(
            sum(
                node.op == "call_function"
                and node.target is torch.ops.higher_order.invoke_subgraph
                for node in backend.graphs[0].graph.nodes
            ),
            0,
        )

    def test_inductor_wrapper_skips_symint_outputs(self):
        def inner(x):
            return x.shape[0] + 1

        def fn(x):
            y = inner(x)
            z = inner(x)
            return x[: y - 1] + x[: z - 1]

        x = torch.randn(4, 3)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_inductor_wrapper_handles_existing_invoke_subgraph(self):
        from torch._higher_order_ops.invoke_subgraph import mark_compile_region

        @mark_compile_region
        def gn(x, y):
            return x + y

        @torch.compile(backend="inductor")
        def fn(x, y):
            return gn(x, y) + gn(x, y)

        a = torch.randn(4)
        b = torch.randn(4)
        self.assertEqual(fn(a, b), (a + b) * 2)

    def test_inductor_wrapper_handles_vmap_batched_tensor_metadata(self):
        @torch.compile(backend="inductor", fullgraph=True)
        def fn(x):
            return torch.vmap(torch.mul, in_dims=(0, None))(x, 3.14)

        x = torch.randn(3)
        self.assertEqual(fn(x), x * 3.14)


if __name__ == "__main__":
    run_tests()
