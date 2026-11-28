# Owner(s): ["module: benchmark"]

from test_component_benchmark import SharedModel

import torch
from torch import nn
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)
from torch.utils.component_benchmark import (
    capture_components,
    Component,
    component_loss_and_bwd,
    ComponentBenchmark,
    ComponentResult,
    CoreForwardBenchmarkPlugin,
    FlopsCounterPlugin,
    IPlugin,
    PyTorchProvider,
    run_cb,
)


class FailurePlugin(IPlugin):
    def run(self, component, copy_input=False):
        component.model.requires_grad_(False)
        raise RuntimeError("intentional plugin failure")


class HealthyPlugin(IPlugin):
    def run(self, component, copy_input=False):
        return ComponentResult(
            {
                "requires_grad": all(
                    p.requires_grad for p in component.model.parameters()
                )
            },
            plugin_name="healthy",
        )


class ComponentBenchmarkRunnerTest(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_flops_count_covers_call_sequence_and_backward(self, device):
        model = SharedModel(device)
        x = torch.randn(5, 3, device=device, requires_grad=True)
        captured = capture_components(Component(model, [x]), ["shared"])["shared"]
        forward = FlopsCounterPlugin().run(captured, copy_input=True).module_result
        backward = (
            FlopsCounterPlugin(component_loss_and_bwd)
            .run(captured, copy_input=True)
            .module_result
        )
        self.assertEqual(forward["total"], 2 * 5 * 3 * 4)
        self.assertEqual(backward["total"], 3 * forward["total"])

    def test_iteration_timing(self, device):
        model = nn.Linear(3, 4, device=device)
        component = Component(model, [torch.randn(5, 3, device=device)])
        timing = (
            CoreForwardBenchmarkPlugin(min_run_time=0.01)
            .run(component, copy_input=True)
            .module_result
        )
        self.assertGreater(timing["median_ms"], 0)
        self.assertEqual(timing["sample_count"], len(timing["samples_ms"]))


class ComponentBenchmarkRunnerHostTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_provider_is_loaded_once(self):
        class Provider(PyTorchProvider):
            def __init__(self):
                super().__init__(nn.Linear(3, 4), (torch.randn(5, 3),))
                self.loads = 0

            def get_component(self):
                self.loads += 1
                return super().get_component()

        provider = Provider()
        result = ComponentBenchmark(provider, custom_plugins=[]).run(max_depth=0)
        self.assertEqual(provider.loads, 1)
        self.assertEqual(list(result.result), ["model"])

    def test_failure_isolation_and_discovery(self):
        model = SharedModel("cpu")
        result = run_cb(
            model,
            torch.randn(5, 3),
            max_depth=1,
            custom_plugins=[FailurePlugin(), HealthyPlugin()],
        )
        self.assertIn(
            "intentional plugin failure", result.errors["model.shared"]["FailurePlugin"]
        )
        self.assertTrue(
            result.result["model.shared"].module_result["healthy.requires_grad"]
        )
        self.assertEqual(
            result.result["model.shared"].module_result["invocation_count"], 3
        )
        self.assertEqual(
            result.result["model.unused"].module_result["status"], "inactive"
        )
        self.assertEqual(
            result.result["model.containers"].module_result["status"], "container"
        )

    def test_depth_filter_and_provider(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
        provider = PyTorchProvider(model, (torch.randn(5, 3),))
        benchmark = ComponentBenchmark(provider, custom_plugins=[])
        self.assertEqual(
            benchmark.list_subcomponents(max_depth=0, submodule_path="0"), ["0"]
        )
        result = benchmark.run(submodule_filter=lambda path: path == "0")
        self.assertEqual(list(result.result), ["model.0"])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            benchmark.list_subcomponents(max_depth=-1)

    def test_spawned_worker(self):
        result = run_cb(
            nn.Linear(3, 4),
            torch.randn(5, 3),
            max_depth=0,
            custom_plugins=[FlopsCounterPlugin()],
            use_multiprocess=True,
        )
        self.assertEqual(result.errors, {})
        self.assertEqual(
            result.result["model"].module_result["flops.total"], 2 * 5 * 3 * 4
        )


instantiate_device_type_tests(
    ComponentBenchmarkRunnerTest,
    globals(),
    except_for=("mps", "xpu", "hpu", "privateuse1"),
)


if __name__ == "__main__":
    run_tests()
