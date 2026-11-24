# Owner(s): ["oncall: jit"]

import io

import torch
from torch.jit._script import interface, script
from torch.jit._serialization import load, save
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


class TestInterfaceSchema(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_schema_outlives_loaded_module(self):
        @interface
        class ForwardInterface(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                pass

        class Child(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        class Parent(torch.nn.Module):
            impl: ForwardInterface

            def __init__(self):
                super().__init__()
                self.impl = Child()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.impl.forward(x)

        archive = io.BytesIO()
        save(script(Parent()), archive)
        archive.seek(0)
        loaded = load(archive)
        call = loaded.forward.graph.findNode("prim::CallMethod")
        iface = call.inputsAt(0).type()
        self.assertIsInstance(iface, torch._C.InterfaceType)
        schema = iface.getMethod("forward")
        self.assertIsNotNone(schema)
        del iface, call, loaded

        receiver = schema.arguments[0]
        self.assertEqual(receiver.name, "self")
        self.assertIsInstance(receiver.type, torch._C.InterfaceType)
        self.assertEqual(receiver.type.getMethodNames(), ["forward"])


if __name__ == "__main__":
    run_tests()
