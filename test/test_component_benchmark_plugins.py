# Owner(s): ["module: benchmark"]

import json
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils.component_benchmark import (
    AccuracyCheckPlugin,
    BenchmarkResult,
    Component,
    component_loss_and_bwd,
    ComponentCapturePlugin,
    ComponentResult,
    GPUBoundSimulationPlugin,
    MemoryPlugin,
    MFUPlugin,
    ProfilerPlugin,
    ReportPlugin,
    TorchDispatchModesPlugin,
)
from torch.utils.component_benchmark.execution import Iteration


def make_optimizer(model, device):
    return torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)


class ComponentBenchmarkPluginTest(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_tensor_io_estimate(self, device):
        component = Component(nn.Sigmoid(), [torch.randn(4, device=device)])
        metrics = (
            TorchDispatchModesPlugin(min_run_time=0.01).run(component).module_result
        )
        self.assertEqual(metrics["estimated_tensor_io_bytes"], 2 * 4 * 4)
        self.assertEqual(sum(metrics["operator_tensor_io_bytes"].values()), 2 * 4 * 4)
        self.assertGreater(metrics["estimated_bandwidth_bytes_per_second"], 0)

    def test_accuracy_includes_optimizer_and_gradients(self, device):
        model = nn.Linear(3, 4, device=device)
        component = Component(model, [torch.randn(5, 3, device=device)])
        plugin = AccuracyCheckPlugin(
            compile_options={"backend": "aot_eager"},
            loss_and_bwd=component_loss_and_bwd,
            optimizer_creator_fn=make_optimizer,
        )
        result = plugin.run(component, copy_input=True).module_result
        self.assertEqual(
            result,
            {
                "output": True,
                "parameters": True,
                "gradients": True,
                "buffers": True,
                "optimizer": True,
            },
        )

    def test_profiler_and_local_replay(self, device):
        component = Component(
            nn.Linear(3, 4, device=device), [torch.randn(5, 3, device=device)]
        )
        with tempfile.TemporaryDirectory() as directory:
            profile = ProfilerPlugin(directory, iterations=1).run(
                component, copy_input=True
            )
            trace = json.loads(Path(profile.artifacts["trace"]).read_text())
            self.assertIn("traceEvents", trace)
            self.assertTrue(
                json.loads(Path(profile.artifacts["operators"]).read_text())
            )
            snapshot = ComponentCapturePlugin(directory).run(component)
            restored = Component.load(snapshot.artifacts["component"])
            self.assertEqual(Iteration(restored)()[0], Iteration(component)()[0])

    def test_memory(self, device):
        component = Component(
            nn.Linear(3, 4, device=device), [torch.randn(5, 3, device=device)]
        )
        result = MemoryPlugin().run(component, copy_input=True).module_result
        if torch.device(device).type == "cuda":
            self.assertGreaterEqual(
                result["peak_allocated_bytes"], result["baseline_allocated_bytes"]
            )
            self.assertGreaterEqual(result["incremental_peak_bytes"], 0)
        else:
            self.assertIsNone(result["peak_allocated_bytes"])

    def test_cuda_graph(self, device):
        component = Component(
            nn.Linear(3, 4, device=device), [torch.randn(5, 3, device=device)]
        )
        if torch.device(device).type == "cuda":
            result = GPUBoundSimulationPlugin(iterations=5).run(component).module_result
            self.assertGreater(result["mean_ms"], 0)
        else:
            with self.assertRaisesRegex(ValueError, "CUDA component"):
                GPUBoundSimulationPlugin().run(component)

    def test_utilization(self, device):
        model = nn.Linear(3, 4, device=device)
        component = Component(model, [torch.randn(5, 3, device=device)])
        mfu = (
            MFUPlugin(1e12, min_run_time=0.01)
            .run(component, copy_input=True)
            .module_result
        )
        self.assertEqual(
            mfu["percent"], 100 * mfu["flops"] / (mfu["median_ms"] / 1000) / 1e12
        )


class ComponentBenchmarkPluginHostTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_result_export_and_collision(self):
        result = BenchmarkResult()
        result.add_component_metrics(
            "model", ComponentResult({"value": 2}, plugin_name="custom")
        )
        result.add_error("model", "<script>bad</script>", "failure")
        result.add_component_metrics(
            "model.<script>", ComponentResult({"value": "</script>"})
        )
        with self.assertRaisesRegex(ValueError, "Duplicate metrics"):
            result.add_component_metrics(
                "model", ComponentResult({"value": 3}, plugin_name="custom")
            )
        with tempfile.TemporaryDirectory() as directory:
            ReportPlugin(directory).run(result)
            self.assertEqual(
                json.loads((Path(directory) / "results.json").read_text()),
                result.to_dict(),
            )
            report = (Path(directory) / "report.html").read_text()
            self.assertIn("&lt;script&gt;bad&lt;/script&gt;", report)
            self.assertIn('id="tree"', report)
            self.assertTrue((Path(directory) / "results.csv").is_file())

    @parametrize("value", [0, -1, float("nan")])
    def test_invalid_peak(self, value):
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            MFUPlugin(value)


instantiate_device_type_tests(
    ComponentBenchmarkPluginTest,
    globals(),
    except_for=("mps", "xpu", "hpu", "privateuse1"),
)
instantiate_parametrized_tests(ComponentBenchmarkPluginHostTest)

if __name__ == "__main__":
    run_tests()
