# Owner(s): ["module: fx"]

import torch
from torch._library.opaque_object import register_opaque_type
from torch.testing._internal.common_utils import raise_on_run_directly, TestCase


# Define a simple opaque type for testing
class OpaqueCounter:
    """A simple opaque object that holds a counter."""

    def __init__(self, value: int):
        self.value = value

    def get_value(self) -> int:
        """Get the current counter value."""
        return self.value


# Register it as an opaque type (reference semantics for identity/mutation tracking)
register_opaque_type(OpaqueCounter, typ="reference")


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

    def test_plain_aot_input_descriptor_evaluation(self):
        """Test PlainAOTInput descriptor evaluation."""
        from torch._functorch._aot_autograd.descriptors import PlainAOTInput
        from torch._functorch._aot_autograd.runtime_wrappers import (
            _evaluate_opaque_descriptor,
        )

        counter = OpaqueCounter(42)
        wrapper = WrapperWithOpaque(counter)

        # Test PlainAOTInput descriptor - simple indexing
        plain_desc = PlainAOTInput(idx=0)
        args = [wrapper]
        result = _evaluate_opaque_descriptor(plain_desc, args)
        self.assertIs(result, wrapper)

        # Test with different index
        plain_desc2 = PlainAOTInput(idx=1)
        args2 = [wrapper, counter]
        result2 = _evaluate_opaque_descriptor(plain_desc2, args2)
        self.assertIs(result2, counter)

    def test_subclass_get_attr_descriptor_evaluation(self):
        """Test SubclassGetAttrAOTInput descriptor evaluation."""
        from torch._functorch._aot_autograd.descriptors import (
            PlainAOTInput,
            SubclassGetAttrAOTInput,
        )
        from torch._functorch._aot_autograd.runtime_wrappers import (
            _evaluate_opaque_descriptor,
        )

        counter = OpaqueCounter(42)
        wrapper = WrapperWithOpaque(counter)

        # Test SubclassGetAttrAOTInput descriptor (nested attribute access)
        # This represents: args[0].counter
        plain_desc = PlainAOTInput(idx=0)
        attr_desc = SubclassGetAttrAOTInput(plain_desc, "counter")

        args = [wrapper]
        result = _evaluate_opaque_descriptor(attr_desc, args)

        # Verify we got the correct counter object
        self.assertIs(result, counter)
        self.assertEqual(result.get_value(), 42)

    def test_nested_subclass_descriptors(self):
        """Test deeply nested descriptor evaluation."""
        from torch._functorch._aot_autograd.descriptors import (
            PlainAOTInput,
            SubclassGetAttrAOTInput,
        )
        from torch._functorch._aot_autograd.runtime_wrappers import (
            _evaluate_opaque_descriptor,
        )

        # Create nested structure: wrapper1.wrapper2.counter
        counter = OpaqueCounter(100)
        inner_wrapper = WrapperWithOpaque(counter)

        class OuterWrapper:
            def __init__(self, inner):
                self.inner = inner

        outer_wrapper = OuterWrapper(inner_wrapper)

        # Build descriptor: args[0].inner.counter
        plain_desc = PlainAOTInput(idx=0)
        inner_desc = SubclassGetAttrAOTInput(plain_desc, "inner")
        counter_desc = SubclassGetAttrAOTInput(inner_desc, "counter")

        args = [outer_wrapper]
        result = _evaluate_opaque_descriptor(counter_desc, args)

        # Verify we got the correct counter
        self.assertIs(result, counter)
        self.assertEqual(result.get_value(), 100)

    def test_opaque_object_in_traced_graph(self):
        """Test that opaque objects can be traced into FX graphs."""
        from torch.fx.experimental.proxy_tensor import make_fx

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


if __name__ == "__main__":
    raise_on_run_directly("test/test_fx.py")
