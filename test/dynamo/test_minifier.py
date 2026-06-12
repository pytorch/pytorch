# Owner(s): ["module: dynamo"]

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch._dynamo
from torch._dynamo.test_minifier_common import MinifierTestBase
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import skipIfNNModuleInlined


class MinifierTests(MinifierTestBase):
    # Test that compile, runtime, and accuracy errors after dynamo can be repro'd
    def _test_after_dynamo(self, device, backend, expected_error):
        run_code = f"""\
@torch.compile(backend={backend!r})
def inner(x):
    for _ in range(10):
        x = torch.sin(x)
    x = torch.relu(x)
    for _ in range(10):
        x = torch.cos(x)
    return x

inner(torch.randn(20, 20, device="{device}"))
"""
        self._run_full_test(run_code, "dynamo", expected_error, isolate=False)

    def test_after_dynamo_compile_error(self, device):
        self._test_after_dynamo(
            device, "relu_compile_error_TESTING_ONLY", "ReluCompileError"
        )

    def test_after_dynamo_runtime_error(self, device):
        self._test_after_dynamo(
            device, "relu_runtime_error_TESTING_ONLY", "ReluRuntimeError"
        )

    def test_after_dynamo_accuracy_error(self, device):
        self._test_after_dynamo(
            device, "relu_accuracy_error_TESTING_ONLY", "AccuracyError"
        )

    def test_after_dynamo_non_leaf_compile_error(self, device):
        run_code = f"""\
@torch.compile(backend="non_leaf_compile_error_TESTING_ONLY")
def inner(x):
    return x + 1

inner(torch.randn(20, 20, requires_grad=True, device="{device}") + 1)
"""
        self._run_full_test(
            run_code, "dynamo", "TestingOnlyCompileError", isolate=False
        )

    # Ensure that the testing backends pass when relu is not present.
    def _test_after_dynamo_backend_passes(self, device, backend):
        @torch.compile(backend=backend)
        def inner(x):
            for _ in range(10):
                x = torch.sin(x)
            for _ in range(10):
                x = torch.cos(x)
            return x

        inner(torch.randn(20, 20, device=device))

    def test_after_dynamo_compile_backend_passes(self, device):
        self._test_after_dynamo_backend_passes(
            device, "relu_compile_error_TESTING_ONLY"
        )

    def test_after_dynamo_runtime_backend_passes(self, device):
        self._test_after_dynamo_backend_passes(
            device, "relu_runtime_error_TESTING_ONLY"
        )

    def test_after_dynamo_accuracy_backend_passes(self, device):
        self._test_after_dynamo_backend_passes(
            device, "relu_accuracy_error_TESTING_ONLY"
        )

    # Test that a module with mixed cpu/device parts  with an error after dynamo can be repro'd
    @skipIfNNModuleInlined()
    def test_cpu_device_module_after_dynamo(self, device):
        backend_name = "relu_compile_error_TESTING_ONLY"
        run_code = f"""\
device = "{device}"

class CpuDeviceModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.m_x = torch.nn.Linear(20, 20).to(device)
        self.m_y = torch.nn.Linear(20, 20)
        self.p_x = torch.nn.Parameter(torch.randn(20, 20).to(device))
        self.p_y = torch.nn.Parameter(torch.randn(20, 20))
        self.b_x = torch.nn.Buffer(torch.ones(20, 20).to(device))
        self.b_y = torch.nn.Buffer(torch.ones(20, 20))

    def forward(self, x, y):
        return self.m_x(x) + self.p_x + self.b_x, self.m_y(y) + self.p_y + self.b_y

mod = CpuDeviceModule()

@torch.compile(backend={backend_name!r})
def inner(x1, y1):
    x2 = torch.randn(20, 20).to(device)
    y2 = torch.randn(20, 20)
    x3, y3 = mod(x1 + x2, y1 + y2)
    return torch.relu(x3.cpu() + y3)

inner(torch.randn(20, 20).to(device), torch.randn(20, 20))
"""

        res = self._run_full_test(run_code, "dynamo", "ReluCompileError", isolate=False)

        self.assertExpectedInline(
            res.minifier_module(),
            """\
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.G__mod___m_x = Linear(in_features=20, out_features=20, bias=True).to(device)
        self.G__mod___m_y = Linear(in_features=20, out_features=20, bias=True)
        self.register_buffer('G__mod___b_x', torch.randn([20, 20], dtype=torch.float32).to(device))
        self.register_buffer('G__mod___b_y', torch.randn([20, 20], dtype=torch.float32))
        self.G__mod___p_x = torch.nn.Parameter(torch.randn([20, 20], dtype=torch.float32, device=device))
        self.G__mod___p_y = torch.nn.Parameter(torch.randn([20, 20], dtype=torch.float32))

    def forward(self, L_x1_ : torch.Tensor, L_y1_ : torch.Tensor):
        l_x1_ = L_x1_
        l_y1_ = L_y1_
        randn = torch.randn(20, 20)
        x2 = randn.to(device);  randn = None
        y2 = torch.randn(20, 20)
        add = l_x1_ + x2;  l_x1_ = x2 = None
        add_1 = l_y1_ + y2;  l_y1_ = y2 = None
        g__mod___m_x = self.G__mod___m_x(add);  add = None
        g__mod___p_x = self.G__mod___p_x
        add_2 = g__mod___m_x + g__mod___p_x;  g__mod___m_x = g__mod___p_x = None
        g__mod___b_x = self.G__mod___b_x
        x3 = add_2 + g__mod___b_x;  add_2 = g__mod___b_x = None
        g__mod___m_y = self.G__mod___m_y(add_1);  add_1 = None
        g__mod___p_y = self.G__mod___p_y
        add_4 = g__mod___m_y + g__mod___p_y;  g__mod___m_y = g__mod___p_y = None
        g__mod___b_y = self.G__mod___b_y
        y3 = add_4 + g__mod___b_y;  add_4 = g__mod___b_y = None
        cpu = x3.cpu();  x3 = None
        add_6 = cpu + y3;  cpu = y3 = None
        relu = torch.relu(add_6);  add_6 = None
        return (relu,)""",
        )

    # Test if we can actually get a minified graph
    def test_if_graph_minified(self, device):
        backend_name = "relu_compile_error_TESTING_ONLY"
        run_code = f"""\
@torch.compile(backend={backend_name!r})
def inner(x):
    for _ in range(20):
        x = torch.sin(x)
    x = torch.relu(x)
    for _ in range(20):
        x = torch.cos(x)
    return x

inner(torch.randn(20, 20, device="{device}"))
"""

        res = self._run_full_test(run_code, "dynamo", "ReluCompileError", isolate=False)

        self.assertExpectedInline(
            res.repro_module(),
            """\
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, sin_19):
        relu = torch.relu(sin_19);  sin_19 = None
        return (relu,)""",
        )


class TestAutocastDeviceDetection(torch._dynamo.test_case.TestCase):
    def _make_options(
        self, accuracy="", autocast=False, backend="eager", only_fwd=True
    ):
        import argparse

        return argparse.Namespace(
            accuracy=accuracy,
            autocast=autocast,
            backend=backend,
            only_fwd=only_fwd,
        )

    def test_repro_minify_autocast_uses_tensor_device(self, device):
        if torch.device(device).type == "cpu":
            self.skipTest("device detection only meaningful for non-CPU devices")

        from torch._dynamo.repro.after_dynamo import repro_minify

        gm = torch.fx.symbolic_trace(torch.nn.Identity())
        args = [torch.randn(4, device=device)]
        options = self._make_options()

        def fake_compiler(gm, example_inputs, compiler_name=None):
            return gm.forward

        mock_autocast = MagicMock()

        with (
            patch("torch._dynamo.repro.after_dynamo.run_load_args", return_value=args),
            patch(
                "torch._dynamo.repro.after_dynamo.lookup_backend",
                return_value=fake_compiler,
            ),
            patch("torch._dynamo.optimize", new=lambda backend: lambda m: m),
            patch("torch.amp.autocast", mock_autocast),
        ):
            repro_minify(options, gm, None)

        mock_autocast.assert_called_once_with(torch.device(device).type, enabled=False)

    def test_repro_run_accuracy_branch_autocast_uses_tensor_device(self, device):
        if torch.device(device).type == "cpu":
            self.skipTest("device detection only meaningful for non-CPU devices")

        from torch._dynamo.repro.after_dynamo import repro_run

        gm = torch.fx.symbolic_trace(torch.nn.Identity())
        args = [torch.randn(4, device=device)]
        options = self._make_options(accuracy="strict")
        mock_autocast = MagicMock()

        with (
            patch("torch._dynamo.repro.after_dynamo.run_load_args", return_value=args),
            patch("torch._dynamo.optimize", new=lambda backend: lambda m: m),
            patch(
                "torch._dynamo.repro.after_dynamo.same_two_models", return_value=True
            ),
            patch("torch.amp.autocast", mock_autocast),
        ):
            repro_run(options, gm, None)

        mock_autocast.assert_called_once_with(torch.device(device).type, enabled=False)


instantiate_device_type_tests(TestAutocastDeviceDetection, globals(), allow_xpu=True)


class ReproGenerationTests(torch._dynamo.test_case.TestCase):
    def test_after_dynamo_repro_uses_constructor_for_fake_quant_with_child_repr(self):
        from torch._dynamo.repro.after_dynamo import generate_dynamo_fx_repro_string
        from torch.ao.quantization import FusedMovingAvgObsFakeQuantize

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.call_module("fake_quant", (x,))
        graph.output((y,))
        fake_quant = torch.ao.quantization.get_default_qat_qconfig(
            "fbgemm"
        ).activation()
        fake_quant.register_parameter(
            "secret_weight", torch.nn.Parameter(torch.full((8,), 123456.0))
        )
        fake_quant.register_buffer("secret_buffer", torch.full((8,), 654321.0))
        gm = torch.fx.GraphModule({"fake_quant": fake_quant}, graph)

        with tempfile.TemporaryDirectory() as save_dir:
            for module_save_dir in (save_dir, None):
                with self.subTest(save_dir=module_save_dir):
                    code = generate_dynamo_fx_repro_string(
                        gm, [torch.randn(2)], "eager", save_dir=module_save_dir
                    )

                    self.assertIn(
                        "self.fake_quant = "
                        "torch.ao.quantization.fake_quantize."
                        "FusedMovingAvgObsFakeQuantize(",
                        code,
                    )
                    self.assertNotIn("(activation_post_process):", code)
                    self.assertNotIn("base64", code)
                    self.assertNotIn("weights_only=False", code)
                    self.assertNotIn("nn_module_", code)
                    if module_save_dir is not None:
                        self.assertEqual(
                            list(Path(module_save_dir).glob("nn_module_*.pt")), []
                        )
                    compile(code, "<generated minifier repro>", "exec")

                    namespace = {"__name__": "not_main"}
                    exec(code, namespace)
                    mod = namespace["mod"]

                    self.assertIsInstance(mod.fake_quant, FusedMovingAvgObsFakeQuantize)
                    self.assertEqual(mod.fake_quant.quant_min, fake_quant.quant_min)
                    self.assertEqual(mod.fake_quant.quant_max, fake_quant.quant_max)
                    self.assertEqual(
                        mod.fake_quant.activation_post_process.reduce_range,
                        fake_quant.activation_post_process.reduce_range,
                    )
                    self.assertFalse(hasattr(mod.fake_quant, "secret_weight"))
                    self.assertFalse(hasattr(mod.fake_quant, "secret_buffer"))
                    self.assertEqual(mod(torch.randn(2))[0].shape, (2,))

    def test_after_dynamo_repro_preserves_fake_quant_buffer_device(self):
        from torch._dynamo.debug_utils import NNModuleToString

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.call_module("fake_quant", (x,))
        graph.output((y,))
        fake_quant = torch.ao.quantization.get_default_qat_qconfig(
            "fbgemm"
        ).activation()
        fake_quant.to("meta")
        gm = torch.fx.GraphModule({"fake_quant": fake_quant}, graph)

        code = NNModuleToString.convert(gm)

        self.assertIn('.to("meta")', code)

    def test_after_dynamo_repro_uses_constructor_for_qat_fused_module(self):
        from torch._dynamo.repro.after_dynamo import generate_dynamo_fx_repro_string

        for backend, expected_backend in (
            ("fbgemm", "fbgemm"),
            ("x86", "fbgemm"),
            ("qnnpack", "qnnpack"),
        ):
            with self.subTest(backend=backend):
                qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
                conv = torch.ao.nn.intrinsic.qat.ConvBnReLU2d(
                    3,
                    4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                    qconfig=qconfig,
                )

                graph = torch.fx.Graph()
                x = graph.placeholder("x")
                y = graph.call_module("conv", (x,))
                graph.output((y,))
                gm = torch.fx.GraphModule({"conv": conv}, graph)

                code = generate_dynamo_fx_repro_string(
                    gm, [torch.randn(2, 3, 8, 8)], "eager"
                )

                self.assertIn(
                    "self.conv = "
                    "torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d(",
                    code,
                )
                self.assertIn(
                    "qconfig=torch.ao.quantization.get_default_qat_qconfig"
                    f"('{expected_backend}')",
                    code,
                )
                self.assertNotIn("(weight_fake_quant):", code)
                self.assertNotIn("(activation_post_process):", code)
                self.assertNotIn("base64", code)
                self.assertNotIn("weights_only=False", code)
                compile(code, "<generated minifier repro>", "exec")

                namespace = {"__name__": "not_main"}
                exec(code, namespace)
                mod = namespace["mod"]

                self.assertIsInstance(mod.conv, torch.ao.nn.intrinsic.qat.ConvBnReLU2d)
                self.assertEqual(mod(torch.randn(2, 3, 8, 8))[0].shape, (2, 4, 4, 4))

    def test_after_dynamo_repro_rejects_custom_qat_qconfig(self):
        from torch._dynamo.repro.after_dynamo import generate_dynamo_fx_repro_string

        qconfig = torch.ao.quantization.QConfig(
            activation=torch.ao.quantization.default_fake_quant,
            weight=torch.ao.quantization.default_weight_fake_quant,
        )
        conv = torch.ao.nn.intrinsic.qat.ConvBnReLU2d(
            3,
            4,
            kernel_size=3,
            qconfig=qconfig,
        )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.call_module("conv", (x,))
        graph.output((y,))
        gm = torch.fx.GraphModule({"conv": conv}, graph)

        with self.assertRaisesRegex(AssertionError, "Cannot convert module"):
            generate_dynamo_fx_repro_string(gm, [torch.randn(2, 3, 8, 8)], "eager")

    def test_after_aot_repro_falls_back_for_unconvertible_module_repr(self):
        from torch._dynamo.repro.after_aot import generate_compiler_repro_string

        class UnsupportedModule(torch.nn.Module):
            def forward(self, x):
                return x

            def __repr__(self):
                return "<lambda>()"

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.call_module("submod", (x,))
        graph.output((y,))
        gm = torch.fx.GraphModule({"submod": UnsupportedModule()}, graph)

        code = generate_compiler_repro_string(gm, [torch.randn(2)], stable_output=True)

        self.assertIn("self.submod = <lambda>()", code)


instantiate_device_type_tests(MinifierTests, globals(), allow_xpu=True)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
