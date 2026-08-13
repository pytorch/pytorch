# Owner(s): ["module: fx"]

import torch
from torch._library.opaque_object import MemberType, register_custom_class
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import raise_on_run_directly, TestCase


# Define a simple opaque type for testing
class OpaqueCounter(torch._custom_class_base.CustomClassBase):
    """A simple opaque object that holds a counter."""

    def __init__(self, value: int):
        self.value = value

    def get_value(self) -> int:
        """Get the current counter value."""
        return self.value


# Register it as an opaque type (reference semantics for identity/mutation tracking)
register_custom_class(OpaqueCounter, typ="symbolic")


class OpaqueMemberBox(torch._custom_class_base.CustomClassBase):
    def __init__(self, value: int):
        self.value = value
        self.property_reads = 0
        self.dynamic_reads = 0

    @property
    def bumping_property(self):
        self.property_reads += 1
        return self.value

    def __getattr__(self, name):
        if name == "dynamic":
            self.dynamic_reads += 1
            return self.value
        raise AttributeError(name)

    def mutates_and_returns_tensor(self, x):
        self.value += 1
        return x + self.value


register_custom_class(
    OpaqueMemberBox,
    typ="symbolic",
    members={
        "value": MemberType.USE_REAL,
        "bumping_property": MemberType.USE_REAL,
        "dynamic": MemberType.USE_REAL,
        "mutates_and_returns_tensor": MemberType.INLINED,
    },
)


# Define a wrapper class that holds an opaque object as an attribute
class WrapperWithOpaque:
    """A wrapper class that contains an opaque object."""

    def __init__(self, counter: OpaqueCounter):
        self.counter = counter
        self.data = torch.tensor([1.0, 2.0, 3.0])


class TestOpaqueInfrastructure(TestCase):
    """
    Test opaque object descriptor and tracking infrastructure.

    This test validates infrastructure for tracking and passing opaque objects
    (like ProcessGroups) through AOTAutograd compilation without unwrapping/wrapping them.
    """

    def test_opaque_object_in_traced_graph(self):
        """Test that opaque objects can be traced into FX graphs."""

        counter = OpaqueCounter(42)

        def fn(x, opaque_obj):
            # Just pass the opaque object through and return the tensor
            # The opaque object should appear in the graph
            return x + 1, opaque_obj

        # Trace the function with make_fx
        x = torch.randn(3, 3)
        traced = make_fx(fn)(x, counter)

        # Verify the graph was created
        self.assertIsNotNone(traced.graph)

        # Run the traced function and verify opaque object is passed through
        result_tensor, result_opaque = traced(x, counter)
        self.assertTrue(torch.allclose(result_tensor, x + 1))
        self.assertIs(result_opaque, counter)

        # Verify the opaque object appears as an input placeholder in the graph
        placeholders = [node for node in traced.graph.nodes if node.op == "placeholder"]
        self.assertEqual(len(placeholders), 2)  # x and opaque_obj

        # The second placeholder should be for the opaque object
        opaque_placeholder = placeholders[1]
        self.assertTrue(opaque_placeholder.name.startswith("opaque_obj"))

    def test_registered_members_are_resolved_lazily(self):
        for tracing_mode in ("fake", "symbolic"):
            with self.subTest(tracing_mode=tracing_mode):
                box = OpaqueMemberBox(2)
                x = torch.ones(1)

                make_fx(lambda x, box: x + 1, tracing_mode=tracing_mode)(x, box)

                self.assertEqual(box.property_reads, 0)
                self.assertEqual(box.dynamic_reads, 0)

                make_fx(
                    lambda x, box: x + box.bumping_property + box.dynamic,
                    tracing_mode=tracing_mode,
                )(x, box)

                self.assertEqual(box.property_reads, 1)
                self.assertEqual(box.dynamic_reads, 1)

    def test_inlined_method_uses_fake_receiver(self):
        for tracing_mode in ("fake", "symbolic"):
            with self.subTest(tracing_mode=tracing_mode):
                box = OpaqueMemberBox(2)

                with self.assertRaisesRegex(AttributeError, "__setattr__"):
                    make_fx(
                        lambda x, box: box.mutates_and_returns_tensor(x),
                        tracing_mode=tracing_mode,
                    )(torch.ones(1), box)

                self.assertEqual(box.value, 2)


if __name__ == "__main__":
    raise_on_run_directly("test/test_fx.py")
