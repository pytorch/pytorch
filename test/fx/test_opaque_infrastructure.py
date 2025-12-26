# Owner(s): ["module: fx"]

import torch
from torch._functorch._aot_autograd.descriptors import (
    PlainAOTInput,
    SubclassGetAttrAOTInput,
)
from torch._functorch._aot_autograd.runtime_wrappers import (
    _evaluate_opaque_descriptor,
    AOTDispatchSubclassWrapper,
)
from torch._functorch._aot_autograd.schemas import (
    AOTConfig,
    SubclassMeta,
    ViewAndMutationMeta,
)
from torch._library.opaque_object import register_opaque_type
from torch.fx.experimental.proxy_tensor import make_fx
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

    def test_opaque_inp_descs_in_aot_autograd(self):
        """Test that opaque_inp_descs work correctly in AOTDispatchSubclassWrapper."""
        # Create a wrapper object that contains an opaque object
        # This simulates a subclass input (like DTensor with device_mesh)
        counter = OpaqueCounter(999)
        wrapper = WrapperWithOpaque(counter)

        # Create descriptor that describes how to extract opaque from runtime args
        # Runtime args: [wrapper]
        # Descriptor: args[0].counter
        wrapper_desc = PlainAOTInput(idx=0)
        opaque_desc = SubclassGetAttrAOTInput(wrapper_desc, "counter")

        # Create runtime metadata with opaque_inp_descs
        runtime_metadata = ViewAndMutationMeta(
            input_info=[],
            output_info=[],
            num_intermediate_bases=0,
            keep_input_mutations=False,
            is_train=False,
            traced_tangents=[],
            traced_tangents_descs=[],
            subclass_inp_meta=[],  # No tensor subclasses in this simple test
            subclass_fw_graph_out_meta=[],
            subclass_tangent_meta=[],
            opaque_inp_descs=[opaque_desc],  # This is what we're testing!
        )

        # Track what the compiled function receives
        compiled_fn_args = {"called": False, "args": None}

        # Create a mock compiled function
        # It should receive the opaque_object appended after any unwrapped args
        def compiled_fn(args):
            compiled_fn_args["called"] = True
            compiled_fn_args["args"] = args
            # The wrapper first unwraps args, then appends opaques
            # With subclass_inp_meta=[], unwrapping returns the original args
            # So we expect: [wrapper, opaque_object]
            self.assertEqual(len(args), 2)
            # First arg is the wrapper (not unwrapped since no subclass_inp_meta)
            self.assertIs(args[0], wrapper)
            # Second arg is the extracted opaque object
            self.assertIsInstance(args[1], OpaqueCounter)
            self.assertEqual(args[1].get_value(), 999)
            self.assertIs(args[1], counter)
            return []  # Return empty list (no outputs)

        # Create SubclassMeta (needed for wrapper)
        subclass_meta = SubclassMeta()

        # Create wrapper
        wrapper_obj = AOTDispatchSubclassWrapper(
            trace_joint=False,
            fw_only=None,
            maybe_subclass_meta=subclass_meta,
            num_fw_outs_saved_for_bw=None,
        )

        # Create a minimal AOTConfig (many fields can be None for this test)
        aot_config = AOTConfig(
            fw_compiler=None,
            bw_compiler=None,
            partition_fn=None,
            decompositions={},
            num_params_buffers=0,
            aot_id=0,
            keep_inference_input_mutations=False,
        )

        # Call post_compile to get the wrapped function
        wrapped_fn = wrapper_obj.post_compile(
            compiled_fn, aot_config, runtime_metadata=runtime_metadata
        )

        # Call the wrapped function with runtime args
        # The wrapper should extract the opaque object and pass it to compiled_fn
        runtime_args = [wrapper]
        wrapped_fn(runtime_args)

        # Verify the compiled function was called with the extracted opaque object
        self.assertTrue(compiled_fn_args["called"], "Compiled function was not called")
        self.assertIsNotNone(compiled_fn_args["args"])
        # All verifications of the opaque object happened inside compiled_fn


if __name__ == "__main__":
    raise_on_run_directly("test/test_fx.py")
