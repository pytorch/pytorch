# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import copy

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
from torch.testing._internal.common_utils import run_tests, TestCase


d_hid = 512
batch_size = 256


class StageBackwardTests(TestCase):
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


devices = ["cpu", "cuda", "hpu", "xpu"]
instantiate_device_type_tests(
    StageBackwardTests, globals(), only_for=devices, allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
