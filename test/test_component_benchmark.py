# Owner(s): ["module: benchmark"]

import json
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)
from torch.utils.component_benchmark import (
    BenchmarkResult,
    CallInput,
    capture_components,
    Component,
    ComponentResult,
    copy_data,
    get_submodules_with_depths,
    repeat_batch,
    to_dtype,
)
from torch.utils.component_benchmark.execution import Iteration


class SharedModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.shared = nn.Linear(3, 4, bias=False, device=device)
        self.unused = nn.Linear(3, 4, device=device)
        self.containers = nn.ModuleList([nn.Identity()])

    def forward(self, x, *, multiplier=1):
        return torch.cat([self.shared(part) for part in x.split(2)]) * multiplier


class ComponentBenchmarkCoreTest(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_capture_respects_no_grad(self, device):
        model = nn.Sequential(
            nn.Linear(3, 4, device=device), nn.Linear(4, 2, device=device)
        )
        with torch.no_grad():
            captured = capture_components(
                Component(model, [torch.randn(5, 3, device=device)]), ["1"]
            )["1"]
        self.assertFalse(captured.get_iteration_call_inputs()[0].args[0].requires_grad)

    def test_bootstrap_batches_remain_separate(self, device):
        model = SharedModel(device)
        component = Component(
            model, [torch.randn(5, 3, device=device), torch.randn(3, 3, device=device)]
        )
        captured = capture_components(component, ["shared"])["shared"]
        self.assertEqual(captured.get_iteration_call_count(), 3)
        self.assertEqual(len(captured.warmup_data), 2)
        iteration = Iteration(captured)
        iteration.warmup()
        self.assertEqual(len(iteration()), 3)
        converted = to_dtype(torch.bfloat16)(captured)
        self.assertEqual(converted.warmup_data[0].args[0].dtype, torch.bfloat16)
        converted_iteration = Iteration(converted)
        converted_iteration.warmup()
        self.assertEqual(len(converted_iteration()), 3)

    def test_capture_and_kwargs(self, device):
        model = SharedModel(device)
        data = torch.randn(5, 3, device=device)
        rng_before = torch.get_rng_state()
        captured = capture_components(
            Component(model, [CallInput((data,), {"multiplier": 2})]),
            ["", "shared", "unused"],
        )
        calls = captured["shared"].get_iteration_call_inputs()
        self.assertEqual([call.args[0].shape[0] for call in calls], [2, 2, 1])
        self.assertEqual(torch.cat([call.args[0] for call in calls]), data)
        self.assertEqual(
            captured[""].get_iteration_call_inputs()[0].kwargs, {"multiplier": 2}
        )
        self.assertEqual(captured["unused"].data, [])
        self.assertEqual(torch.get_rng_state(), rng_before)
        self.assertEqual(model.shared._forward_pre_hooks, {})

    def test_snapshots_preserve_aliases_and_strides(self, device):
        original = torch.randn(4, 6, device=device, requires_grad=True)
        view = original[:, ::2]
        copied = copy_data(
            {"view": view, "alias": view, "other_view": original[:, 1::2]}
        )
        self.assertIs(copied["view"], copied["alias"])
        self.assertEqual(copied["view"].stride(), view.stride())
        self.assertEqual(copied["view"], view)
        self.assertEqual(
            copied["view"].untyped_storage().data_ptr(),
            copied["other_view"].untyped_storage().data_ptr(),
        )
        self.assertNotEqual(
            copied["view"].untyped_storage().data_ptr(),
            original.untyped_storage().data_ptr(),
        )
        self.assertTrue(copied["view"].requires_grad)
        self.assertTrue(copied["view"].is_leaf)

    def test_replay_boundary_gradients(self, device):
        model = SharedModel(device)
        x = torch.randn(5, 3, device=device, requires_grad=True)
        expected = model(x, multiplier=3)
        expected.square().sum().backward()
        captured = capture_components(
            Component(model, [CallInput((x,), {"multiplier": 3})]),
            ["shared"],
            capture_output_grad=True,
            loss_and_bwd=lambda output: output.square().sum().backward(),
        )
        iteration = Iteration(captured["shared"], copy_input=True)
        iteration()
        self.assertEqual(iteration.model.weight.grad, model.shared.weight.grad)

    def test_capture_does_not_modify_buffers_or_inputs(self, device):
        model = nn.Sequential(nn.BatchNorm1d(3, device=device), nn.ReLU(inplace=True))
        x = torch.randn(5, 3, device=device)
        original = x.clone()
        captured = capture_components(Component(model, [x]), ["0", "1"])
        self.assertEqual(x, original)
        self.assertEqual(
            model[0].num_batches_tracked,
            torch.zeros((), dtype=torch.long, device=device),
        )
        self.assertEqual(
            captured["0"].model.num_batches_tracked, model[0].num_batches_tracked
        )
        self.assertEqual(model[0]._forward_pre_hooks, {})

    def test_autocast_is_per_invocation(self, device):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 4, device=device)

            def forward(self, x):
                with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                    a = self.linear(x)
                b = self.linear(x)
                return a.float() + b

        model = Model()
        x = torch.randn(5, 3, device=device)
        captured = capture_components(Component(model, [x]), ["linear"])["linear"]
        calls = captured.get_iteration_call_inputs()
        self.assertEqual([call.autocast_enabled for call in calls], [True, False])
        outputs = Iteration(captured, copy_input=True)()
        self.assertEqual(
            [output.dtype for output in outputs], [torch.bfloat16, torch.float32]
        )

    def test_precision_and_explicit_batch_scaling(self, device):
        model = nn.Embedding(10, 4, device=device)
        indices = torch.ones(3, 2, dtype=torch.long, device=device)
        mask = torch.zeros(2, 2, device=device)
        component = Component(model, [CallInput((indices,), {"mask": mask})])
        converted = to_dtype(torch.bfloat16)(component)
        call = converted.get_iteration_call_inputs()[0]
        self.assertEqual(call.args[0].dtype, torch.long)
        self.assertEqual(call.kwargs["mask"].dtype, torch.bfloat16)
        scaled = repeat_batch(
            2, lambda tensor: 0 if tensor.dtype == torch.long else None
        )(converted)
        call = scaled.get_iteration_call_inputs()[0]
        self.assertEqual(call.args[0].shape, (6, 2))
        self.assertEqual(call.kwargs["mask"].shape, (2, 2))
        self.assertEqual(indices.shape, (3, 2))

    def test_tuple_inputs_and_nonleaf_call_input(self, device):
        model = nn.Linear(3, 4, device=device)
        x = torch.randn(5, 3, device=device, requires_grad=True) * 2
        component = Component(model, [((x,), {})], data_is_args_kwargs_tuple=True)
        outputs = Iteration(component, copy_input=True)()
        self.assertEqual(outputs[0], model(x))
        component = Component(model, [CallInput((x,))])
        self.assertEqual(Iteration(component, copy_input=True)()[0], model(x))


class ComponentBenchmarkCoreHostTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_depth(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
        self.assertEqual(get_submodules_with_depths(model, 1), ["", "0", "1"])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            get_submodules_with_depths(model, -1)

    def test_result_json_csv_and_collision(self):
        result = BenchmarkResult()
        result.add_component_metrics(
            "model", ComponentResult({"value": 2}, plugin_name="custom")
        )
        with self.assertRaisesRegex(ValueError, "Duplicate metrics"):
            result.add_component_metrics(
                "model", ComponentResult({"value": 3}, plugin_name="custom")
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            result.save_json(path / "result.json")
            result.save_csv(path / "result.csv")
            self.assertEqual(
                json.loads((path / "result.json").read_text()), result.to_dict()
            )
            self.assertTrue((path / "result.csv").is_file())


instantiate_device_type_tests(
    ComponentBenchmarkCoreTest,
    globals(),
    except_for=("mps", "xpu", "hpu", "privateuse1"),
)


if __name__ == "__main__":
    run_tests()
