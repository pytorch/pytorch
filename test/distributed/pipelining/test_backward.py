# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import copy
import weakref

from model_registry import MLPModule, MultiInterMediateModel

import torch
from torch.distributed.pipelining._backward import (
    stage_backward,
    stage_backward_input,
    stage_backward_weight,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    skipXPUIf,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


d_hid = 512
batch_size = 256


class StageBackwardTests(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @skipXPUIf(True, "https://github.com/intel/torch-xpu-ops/issues/1682")
    def test_stage_backward(self, device):
        # MLP as a stage module
        mod = MLPModule(d_hid).to(device)
        x = torch.randn(batch_size, d_hid, device=device)
        # As in a pipeline stage, the inputs to this stage requires gradients
        x.requires_grad_(True)
        target = torch.randn(batch_size, d_hid, device=device)
        loss_fn = torch.nn.MSELoss(reduction="sum")

        # Make a copy
        ref_mod = copy.deepcopy(mod).to(device)
        ref_x = x.detach().requires_grad_(x.requires_grad).to(device)
        ref_target = target.detach().to(device)

        # Forward and backward in stage manner
        out = mod(x)
        loss = loss_fn(out, target)
        grad_inputs = stage_backward(
            stage_output=loss,
            output_grads=None,
            input_values=(x,),
        )

        # Run reference
        ref_out = ref_mod(ref_x)
        ref_loss = loss_fn(ref_out, ref_target)
        ref_loss.backward()

        torch.testing.assert_close(grad_inputs[0], ref_x.grad)

        # Every rank checks gradients
        for name, p in mod.named_parameters():
            ref_p = ref_mod.get_parameter(name)
            try:
                torch.testing.assert_close(p.grad, ref_p.grad)
            except AssertionError:
                print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
                raise

    def test_stage_backward_input(self, device):
        # MLP as a stage module
        mod = MLPModule(d_hid).to(device)
        x = torch.randn(batch_size, d_hid, device=device)
        # As in a pipeline stage, the inputs to this stage requires gradients
        x.requires_grad_(True)
        target = torch.randn(batch_size, d_hid, device=device)
        loss_fn = torch.nn.MSELoss(reduction="sum")

        # Make a copy
        ref_mod = copy.deepcopy(mod).to(device)
        ref_x = x.detach().requires_grad_(x.requires_grad).to(device)
        ref_target = target.detach().to(device)

        # Forward, then backward of loss with respect to inputs
        out = mod(x)
        loss = loss_fn(out, target)
        dinputs, _param_groups = stage_backward_input(
            stage_outputs_or_loss=(loss,),
            output_grads=None,
            input_values=[x],
            weights=mod.parameters(),
        )

        # Run reference
        ref_out = ref_mod(ref_x)
        ref_loss = loss_fn(ref_out, ref_target)
        ref_loss.backward()

        torch.testing.assert_close(x.grad, ref_x.grad)
        torch.testing.assert_close(dinputs[0], ref_x.grad)
        for _, p in mod.named_parameters():
            # Check that the weight gradients were not updated
            self.assertEqual(p.grad, None)

    def test_stage_backward_input_ignores_non_tensor_inputs(self, device):
        mod = MLPModule(d_hid).to(device)
        x = torch.randn(batch_size, d_hid, device=device, requires_grad=True)
        non_tensor_input = object()

        ref_mod = copy.deepcopy(mod).to(device)
        ref_x = x.detach().requires_grad_(True).to(device)

        loss = mod(x).sum()
        dinputs, _param_groups = stage_backward_input(
            stage_outputs_or_loss=(loss,),
            output_grads=None,
            input_values=[non_tensor_input, x],
            weights=mod.parameters(),
        )

        ref_mod(ref_x).sum().backward()
        self.assertEqual(dinputs[0], None)
        torch.testing.assert_close(x.grad, ref_x.grad)
        torch.testing.assert_close(dinputs[1], ref_x.grad)

    @skipXPUIf(True, "https://github.com/intel/torch-xpu-ops/issues/1682")
    def test_stage_backward_weight(self, device):
        # MLP as a stage module
        mod = MLPModule(d_hid).to(device)
        x = torch.randn(batch_size, d_hid, device=device)
        # As in a pipeline stage, the inputs to this stage requires gradients
        x.requires_grad_(True)
        target = torch.randn(batch_size, d_hid, device=device)
        loss_fn = torch.nn.MSELoss(reduction="sum")

        # Make a copy
        ref_mod = copy.deepcopy(mod).to(device)
        ref_x = x.detach().requires_grad_(x.requires_grad).to(device)
        ref_target = target.detach().to(device)
        # Forward, then backward of loss with respect to inputs
        out = mod(x)
        loss = loss_fn(out, target)
        _dinputs, param_groups = stage_backward_input(
            stage_outputs_or_loss=(loss,),
            output_grads=None,
            input_values=[x],
            weights=mod.parameters(),
        )

        # backward of loss with respect to weights
        stage_backward_weight(mod.parameters(), param_groups, retain_graph=True)

        # Run reference
        ref_out = ref_mod(ref_x)
        ref_loss = loss_fn(ref_out, ref_target)
        ref_loss.backward()

        # Every rank checks gradients
        for name, p in mod.named_parameters():
            ref_p = ref_mod.get_parameter(name)
            try:
                torch.testing.assert_close(p.grad, ref_p.grad)
            except AssertionError:
                print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
                raise

    @skipXPUIf(True, "https://github.com/intel/torch-xpu-ops/issues/1682")
    def test_stage_backward_weight_multiple_iters(self, device):
        # MLP as a stage module
        mod = MLPModule(d_hid).to(device)
        inputs = []
        for _ in range(10):
            x = torch.randn(batch_size, d_hid, device=device)
            inputs.append(x)
            # As in a pipeline stage, the inputs to this stage requires gradients
            x.requires_grad_(True)

        target = torch.randn(batch_size, d_hid, device=device)
        loss_fn = torch.nn.MSELoss(reduction="sum")

        # Make a copy
        ref_mod = copy.deepcopy(mod).to(device)
        ref_inputs = []
        for x in inputs:
            ref_x = x.detach().requires_grad_(x.requires_grad).to(device)
            ref_inputs.append(ref_x)
        ref_target = target.detach().to(device)

        # Forward, then backward of loss with respect to inputs
        for x in inputs:
            out = mod(x)
            loss = loss_fn(out, target)
            _dinputs, param_groups = stage_backward_input(
                stage_outputs_or_loss=(loss,),
                output_grads=None,
                input_values=[x],
                weights=mod.parameters(),
            )

            # backward of loss with respect to weights
            stage_backward_weight(mod.parameters(), param_groups)

        # Run reference
        for ref_x in ref_inputs:
            ref_out = ref_mod(ref_x)
            ref_loss = loss_fn(ref_out, ref_target)
            ref_loss.backward()

        # Every rank checks gradients
        for name, p in mod.named_parameters():
            ref_p = ref_mod.get_parameter(name)
            try:
                torch.testing.assert_close(p.grad, ref_p.grad)
            except AssertionError:
                print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
                raise

    def test_stage_backward_weight_grad_validation(self, device):
        test_cases = [
            (
                "size == 2",
                lambda: MultiInterMediateModel([d_hid // 2, d_hid // 2]).to(device),
                lambda: [
                    (
                        torch.randn(batch_size, d_hid // 2, device=device),
                        torch.randn(d_hid // 2, d_hid // 2, device=device),
                    )
                ],
            ),
            (
                "size = 1",
                lambda: MLPModule(d_hid).to(device),
                lambda: [(torch.randn(batch_size, d_hid, device=device),)],
            ),
            (
                "1 grad, 1 None",
                lambda: MultiInterMediateModel([d_hid // 2, d_hid // 2]).to(device),
                lambda: [(torch.randn(batch_size, d_hid // 2, device=device), None)],
            ),
            (
                "1 None, 1 grad",
                lambda: MultiInterMediateModel([d_hid // 2, d_hid // 2]).to(device),
                lambda: [(None, torch.randn(d_hid // 2, d_hid // 2, device=device))],
            ),
        ]

        for description, module_factory, mock_grads_factory in test_cases:
            with self.subTest(description=description):
                mod = module_factory()
                x = torch.randn(batch_size, d_hid, device=device)
                x.requires_grad_(True)
                out = mod(x)
                loss = torch.sum(out)
                dinputs, param_groups = stage_backward_input(
                    stage_outputs_or_loss=[loss],
                    output_grads=None,
                    input_values=[x],
                    weights=mod.parameters(),
                )

                # Set up mock grads
                for param_group in param_groups:
                    param_group["grads"] = mock_grads_factory()

                stage_backward_weight(mod.parameters(), param_groups)

    def test_stage_backward_multi_output_intermediate(self, device):
        mod = MultiInterMediateModel([d_hid // 2, d_hid // 2]).to(device)
        x = torch.randn(batch_size, d_hid, device=device, requires_grad=True)

        out = mod(x)
        loss = out.sum()

        dinputs, param_groups = stage_backward_input(
            stage_outputs_or_loss=[loss],
            output_grads=None,
            input_values=[x],
            weights=mod.parameters(),
        )

        stage_backward_weight(mod.parameters(), param_groups)

        ref_mod = copy.deepcopy(mod)
        ref_x = x.detach().clone().requires_grad_(True)
        ref_out = ref_mod(ref_x)
        ref_loss = ref_out.sum()
        ref_loss.backward()

        torch.testing.assert_close(dinputs[0], ref_x.grad)
        for name, p in mod.named_parameters():
            ref_p = ref_mod.get_parameter(name)
            torch.testing.assert_close(p.grad, ref_p.grad)

    def test_stage_backward_weight_shared_weights(self, device):
        class SharedWeightModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(d_hid, d_hid))

            def forward(self, x):
                x = torch.matmul(x, self.w)
                x = torch.relu(x)
                return torch.matmul(x, self.w)

        mod = SharedWeightModule().to(device)
        x = torch.randn(batch_size, d_hid, device=device, requires_grad=True)

        ref_mod = copy.deepcopy(mod)
        ref_x = x.detach().clone().requires_grad_(True)

        out = mod(x)
        loss = out.sum()

        dinputs, param_groups = stage_backward_input(
            stage_outputs_or_loss=[loss],
            output_grads=None,
            input_values=[x],
            weights=mod.parameters(),
        )
        stage_backward_weight(mod.parameters(), param_groups)

        ref_out = ref_mod(ref_x)
        ref_loss = ref_out.sum()
        ref_loss.backward()

        torch.testing.assert_close(dinputs[0], ref_x.grad)
        for name, p in mod.named_parameters():
            ref_p = ref_mod.get_parameter(name)
            torch.testing.assert_close(p.grad, ref_p.grad)

    def test_stage_backward_from_gradient_edge(self, device):
        """Match tensor-rooted backward after releasing the output."""
        mod = MLPModule(d_hid).to(device)
        x = torch.randn(batch_size, d_hid, device=device, requires_grad=True)
        output_grad = torch.randn(batch_size, d_hid, device=device)

        ref_mod = copy.deepcopy(mod).to(device)
        ref_x = x.detach().clone().requires_grad_(True)

        out = mod(x)
        edge = torch.autograd.graph.get_gradient_edge(out)
        # The edge must not keep the activation alive.
        released = weakref.ref(out)
        del out
        self.assertIsNone(released())

        grad_inputs = stage_backward(
            stage_output=(edge,),
            output_grads=(output_grad,),
            input_values=(x,),
        )

        ref_out = ref_mod(ref_x)
        ref_out.backward(output_grad)

        torch.testing.assert_close(grad_inputs[0], ref_x.grad)
        for name, p in mod.named_parameters():
            torch.testing.assert_close(p.grad, ref_mod.get_parameter(name).grad)

    def test_stage_backward_mixed_edge_and_tensor_outputs(self, device):
        """Handle edge, tensor, and non-grad outputs together."""

        class TwoOutputModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net1 = torch.nn.Linear(d_hid, d_hid)
                self.net2 = torch.nn.Linear(d_hid, d_hid)

            def forward(self, x):
                return self.net1(x), self.net2(x), torch.ones(1, device=x.device)

        mod = TwoOutputModule().to(device)
        x = torch.randn(batch_size, d_hid, device=device, requires_grad=True)
        grads = (
            torch.randn(batch_size, d_hid, device=device),
            torch.randn(batch_size, d_hid, device=device),
            None,
        )

        ref_mod = copy.deepcopy(mod).to(device)
        ref_x = x.detach().clone().requires_grad_(True)

        first, second, no_grad_out = mod(x)
        edge = torch.autograd.graph.get_gradient_edge(first)
        del first

        self.assertFalse(no_grad_out.requires_grad)
        grad_inputs = stage_backward(
            stage_output=(edge, second, None),
            output_grads=grads,
            input_values=(x,),
        )

        ref_first, ref_second, _ = ref_mod(ref_x)
        torch.autograd.backward((ref_first, ref_second), (grads[0], grads[1]))

        torch.testing.assert_close(grad_inputs[0], ref_x.grad)
        for name, p in mod.named_parameters():
            torch.testing.assert_close(p.grad, ref_mod.get_parameter(name).grad)

    def test_stage_backward_edge_keeps_python_autograd_function_alive(self, device):
        """Keep a Python autograd graph alive through its edge."""

        class ScaleBy(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, factor):
                ctx.factor = factor
                return x * factor

            @staticmethod
            def backward(ctx, grad_out):
                return grad_out * ctx.factor, None

        x = torch.randn(batch_size, d_hid, device=device, requires_grad=True)
        output_grad = torch.randn(batch_size, d_hid, device=device)

        out = ScaleBy.apply(x, 3.0)
        edge = torch.autograd.graph.get_gradient_edge(out)
        del out

        grad_inputs = stage_backward(
            stage_output=(edge,),
            output_grads=(output_grad,),
            input_values=(x,),
        )
        torch.testing.assert_close(grad_inputs[0], output_grad * 3.0)

    def test_stage_backward_input_from_gradient_edge(self, device):
        """Match tensor-rooted split backward."""
        mod = MLPModule(d_hid).to(device)
        x = torch.randn(batch_size, d_hid, device=device, requires_grad=True)
        output_grad = torch.randn(batch_size, d_hid, device=device)

        ref_mod = copy.deepcopy(mod).to(device)
        ref_x = x.detach().clone().requires_grad_(True)

        out = mod(x)
        edge = torch.autograd.graph.get_gradient_edge(out)
        del out

        dinputs, param_groups = stage_backward_input(
            stage_outputs_or_loss=[edge],
            output_grads=[output_grad],
            input_values=[x],
            weights=mod.parameters(),
        )
        stage_backward_weight(mod.parameters(), param_groups)

        ref_mod(ref_x).backward(output_grad)

        torch.testing.assert_close(dinputs[0], ref_x.grad)
        for name, p in mod.named_parameters():
            torch.testing.assert_close(p.grad, ref_mod.get_parameter(name).grad)

    def test_stage_backward_input_edge_requires_output_grads(self, device):
        mod = MLPModule(d_hid).to(device)
        x = torch.randn(batch_size, d_hid, device=device, requires_grad=True)
        edge = torch.autograd.graph.get_gradient_edge(mod(x))

        with self.assertRaisesRegex(AssertionError, "requires output gradients"):
            stage_backward_input(
                stage_outputs_or_loss=[edge],
                output_grads=None,
                input_values=[x],
                weights=mod.parameters(),
            )


instantiate_device_type_tests(StageBackwardTests, globals(), allow_xpu=True)

if __name__ == "__main__":
    run_tests()
