# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import copy

from model_registry import MLPModule

import torch
from torch.distributed.pipelining._backward import (
    stage_backward,
    stage_backward_input,
    stage_backward_weight,
)
from torch.testing._internal.common_utils import run_tests, TestCase


d_hid = 512
batch_size = 256


class StageBackwardTests(TestCase):
    def test_stage_backward(self):
        # MLP as a stage module
        mod = MLPModule(d_hid)
        x = torch.randn(batch_size, d_hid)
        # As in a pipeline stage, the inputs to this stage requires gradients
        x.requires_grad_(True)
        target = torch.randn(batch_size, d_hid)
        loss_fn = torch.nn.MSELoss(reduction="sum")

        # Make a copy
        ref_mod = copy.deepcopy(mod)
        ref_x = x.detach().requires_grad_(x.requires_grad)
        ref_target = target.detach()

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

    def test_stage_backward_input(self):
        # MLP as a stage module
        mod = MLPModule(d_hid)
        x = torch.randn(batch_size, d_hid)
        # As in a pipeline stage, the inputs to this stage requires gradients
        x.requires_grad_(True)
        target = torch.randn(batch_size, d_hid)
        loss_fn = torch.nn.MSELoss(reduction="sum")

        # Make a copy
        ref_mod = copy.deepcopy(mod)
        ref_x = x.detach().requires_grad_(x.requires_grad)
        ref_target = target.detach()

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

    def test_stage_backward_weight(self):
        # MLP as a stage module
        mod = MLPModule(d_hid)
        x = torch.randn(batch_size, d_hid)
        # As in a pipeline stage, the inputs to this stage requires gradients
        x.requires_grad_(True)
        target = torch.randn(batch_size, d_hid)
        loss_fn = torch.nn.MSELoss(reduction="sum")

        # Make a copy
        ref_mod = copy.deepcopy(mod)
        ref_x = x.detach().requires_grad_(x.requires_grad)
        ref_target = target.detach()

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

    def test_stage_backward_weight_multiple_iters(self):
        # MLP as a stage module
        mod = MLPModule(d_hid)
        inputs = []
        for _ in range(10):
            x = torch.randn(batch_size, d_hid)
            inputs.append(x)
            # As in a pipeline stage, the inputs to this stage requires gradients
            x.requires_grad_(True)

        target = torch.randn(batch_size, d_hid)
        loss_fn = torch.nn.MSELoss(reduction="sum")

        # Make a copy
        ref_mod = copy.deepcopy(mod)
        ref_inputs = []
        for x in inputs:
            ref_inputs.append(x.detach().requires_grad_(x.requires_grad))
        ref_target = target.detach()

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


if __name__ == "__main__":
    run_tests()
