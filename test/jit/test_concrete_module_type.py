# Owner(s): ["oncall: jit"]

import unittest

import torch
from torch.testing._internal.common_utils import raise_on_run_directly


class TestConcreteModuleTypeFindSubmodule(unittest.TestCase):
    def test_error_message_includes_submodule_name(self):
        class ChildModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 3)

            def forward(self, x):
                return self.linear(x)

        class ParentModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.existing_child = ChildModule()

            def forward(self, x):
                return self.existing_child(x)

        module = ParentModule()
        scripted_module = torch.jit.script(module)

        self.assertIsNotNone(scripted_module.existing_child)

        # Now try to trigger the error by accessing a non-existent submodule
        # through the internal ConcreteModuleType mechanism. This happens
        # when the TorchScript compiler tries to resolve submodule references.
        class BrokenModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parent = ParentModule()

            def forward(self, x):
                return self.parent.missing_submodule(x)

        broken_module = BrokenModule()

        with self.assertRaises(RuntimeError) as context:
            torch.jit.script(broken_module)

        error_msg = str(context.exception)
        self.assertIn("missing_submodule", error_msg.lower())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
