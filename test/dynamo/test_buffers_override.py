# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
import torch.nn as nn


class TestBuffersOverride(torch._dynamo.test_case.TestCase):
    def test_buffers_override(self):
        class SomeModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Override buffers; should not cause breakage
                # this is because we use `named_buffers` for
                # static marking
                self.register_buffer("A", torch.ones(3, 3))
                self.buffers = []

            def forward(self):
                return self.A * torch.zeros(1, 1)

        model = SomeModel().to(torch.device("cpu"))
        compiled_model = torch.compile(model, backend="eager")
        self.assertEqual(compiled_model.A, torch.ones(3, 3))
        compiled_model()

    def test_named_buffers_override(self):
        class SomeModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Override buffers; should not cause breakage
                # but skip the marking static here since
                # named_buffers is overridden
                self.register_buffer("B", torch.ones(3, 3))
                self.named_buffers = []

            def forward(self):
                return self.B * torch.zeros(1, 1)

        model = SomeModel().to(torch.device("cpu"))
        compiled_model = torch.compile(model, backend="eager")
        self.assertEqual(compiled_model.B, torch.ones(3, 3))
        compiled_model()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
