import math
import unittest

import torch
import torch.nn as nn
from torch.testing._internal.common_device_type import instantiate_device_type_tests, onlyCPU
from torch.testing._internal.common_utils import TestCase, run_tests


class TestStateSpaceModel(TestCase):
    @onlyCPU
    def test_shapes_and_state_returned(self, device):
        batch = 2
        time = 5
        input_size = 3
        state_size = 4
        output_size = 6

        model = nn.StateSpaceModel(input_size, state_size, output_size).to(device)
        x = torch.randn(batch, time, input_size, device=device)

        y, final_state = model(x)

        self.assertEqual(y.shape, (batch, time, output_size))
        self.assertEqual(final_state.shape, (batch, state_size))

    @onlyCPU
    def test_grad_flow(self, device):
        batch = 1
        time = 3
        input_size = 2
        state_size = 2
        output_size = 1

        model = nn.StateSpaceModel(input_size, state_size, output_size).to(device)
        x = torch.randn(batch, time, input_size, device=device, requires_grad=True)

        y, _ = model(x)
        loss = y.pow(2).mean()
        loss.backward()

        # We just want to be sure parameters get gradients; the exact values
        # are not important for this smoke test.
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
        self.assertIsNotNone(x.grad)

    @onlyCPU
    def test_zero_input_is_stable(self, device):
        batch = 2
        time = 8
        input_size = 3
        state_size = 4
        output_size = 2

        model = nn.StateSpaceModel(input_size, state_size, output_size).to(device)

        zeros = torch.zeros(batch, time, input_size, device=device)
        y, final_state = model(zeros)

        # With small A and zero inputs, the magnitude should not explode.
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(torch.isfinite(final_state).all())
        self.assertLess(y.abs().max().item(), 1.0 + 1e-3)


instantiate_device_type_tests(TestStateSpaceModel, globals(), only_for="cpu")


if __name__ == "__main__":
    run_tests()


