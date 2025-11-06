# Owner(s): ["oncall: pt2"]
"""
Tests for the fix to issue #132651: Avoid double metadata collection in subclass codepath.

This test verifies that subclass_fw_graph_out_meta is now a computed property that
dynamically generates the correct metadata based on is_train and keep_input_mutations,
eliminating the need to re-run run_functionalized_fw_and_collect_metadata.
"""

from unittest.mock import patch

import torch
from torch._functorch._aot_autograd.collect_metadata_analysis import (
    run_functionalized_fw_and_collect_metadata,
)
from torch._functorch._aot_autograd.schemas import PlainTensorMeta, ViewAndMutationMeta
from torch._functorch.aot_autograd import aot_function
from torch.testing._internal.common_subclass import WrapperTensor
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._pytree import tree_map


# Test tensor subclass for testing
class SimpleTensorSubclass(WrapperTensor):
    """A simple tensor subclass that wraps a single tensor."""

    @classmethod
    def get_wrapper_properties(cls, t, requires_grad=False):
        return t, {"requires_grad": requires_grad}

    def __init__(self, t, requires_grad=False):
        self.t = t

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        if kwargs is None:
            kwargs = {}

        def unwrap(e):
            return e.t if isinstance(e, SimpleTensorSubclass) else e

        def wrap(e):
            return SimpleTensorSubclass(e) if isinstance(e, torch.Tensor) else e

        rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
        return rs

    def __repr__(self):
        return super().__repr__(tensor_contents=f"t={self.t}")

    def __tensor_flatten__(self):
        return ["t"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        return SimpleTensorSubclass(inner_tensors["t"])


class TestAOTMetadataProperty(TestCase):
    """Test that subclass_fw_graph_out_meta is correctly computed as a property."""

    def test_property_computes_correct_metadata_training(self):
        """Test that the property correctly computes metadata in training mode."""
        # Create a simple ViewAndMutationMeta with mocked data
        from torch._functorch._aot_autograd.schemas import (
            InputAliasInfo,
            OutputAliasInfo,
        )

        # Mock input info: 3 inputs, first one is mutated
        input_info = [
            InputAliasInfo(
                is_leaf=True,
                mutates_data=True,
                mutates_metadata=False,
                mutations_hidden_from_autograd=False,
                mutations_under_no_grad_or_inference_mode=False,
                mutation_inductor_storage_resize=False,
                mutates_storage_metadata=False,
                requires_grad=False,
                keep_input_mutations=False,
            ),
            InputAliasInfo(
                is_leaf=True,
                mutates_data=False,
                mutates_metadata=False,
                mutations_hidden_from_autograd=False,
                mutations_under_no_grad_or_inference_mode=False,
                mutation_inductor_storage_resize=False,
                mutates_storage_metadata=False,
                requires_grad=False,
                keep_input_mutations=False,
            ),
            InputAliasInfo(
                is_leaf=True,
                mutates_data=False,
                mutates_metadata=False,
                mutations_hidden_from_autograd=False,
                mutations_under_no_grad_or_inference_mode=False,
                mutation_inductor_storage_resize=False,
                mutates_storage_metadata=False,
                requires_grad=False,
                keep_input_mutations=False,
            ),
        ]

        # Mock output info: 2 outputs
        from torch._functorch._aot_autograd.schemas import OutputType

        output_info = [
            OutputAliasInfo(
                output_type=OutputType.non_alias,
                raw_type=torch.Tensor,
                base_idx=None,
                dynamic_dims=None,
                requires_grad=False,
            ),
            OutputAliasInfo(
                output_type=OutputType.non_alias,
                raw_type=torch.Tensor,
                base_idx=None,
                dynamic_dims=None,
                requires_grad=False,
            ),
        ]

        # Create component metadata (now as tuple)
        subclass_inp_meta = [PlainTensorMeta(0), PlainTensorMeta(1), PlainTensorMeta(2)]
        subclass_meta_components = (
            [PlainTensorMeta(0), PlainTensorMeta(1)],  # user outputs
            [PlainTensorMeta(0)],  # intermediate bases
        )

        # Create metadata in training mode
        metadata = ViewAndMutationMeta(
            input_info=input_info,
            output_info=output_info,
            num_intermediate_bases=1,
            keep_input_mutations=False,
            traced_tangents=[],
            traced_tangents_descs=[],
            subclass_inp_meta=subclass_inp_meta,
            _subclass_meta_components=subclass_meta_components,
            subclass_tangent_meta=[],
            is_train=True,
        )

        # In training mode with is_train=True:
        # - mutated inputs should be included (1 input)
        # - user outputs should be included (2 outputs)
        # - intermediate bases should be included (1 base)
        # Total: 4 items
        fw_graph_out = metadata.subclass_fw_graph_out_meta
        self.assertEqual(len(fw_graph_out), 4)

        # Check order: mutated_inputs, user_outputs, intermediate_bases
        user_outs_meta, intermediate_bases_meta = subclass_meta_components
        self.assertEqual(fw_graph_out[0], subclass_inp_meta[0])  # mutated input
        self.assertEqual(fw_graph_out[1], user_outs_meta[0])  # user output 1
        self.assertEqual(fw_graph_out[2], user_outs_meta[1])  # user output 2
        self.assertEqual(
            fw_graph_out[3], intermediate_bases_meta[0]
        )  # intermediate base

    def test_property_computes_correct_metadata_inference(self):
        """Test that the property correctly computes metadata in inference mode."""
        from torch._functorch._aot_autograd.schemas import (
            InputAliasInfo,
            OutputAliasInfo,
            OutputType,
        )

        # Mock input info: 3 inputs, first one is mutated
        input_info = [
            InputAliasInfo(
                is_leaf=True,
                mutates_data=True,
                mutates_metadata=False,
                mutations_hidden_from_autograd=False,
                mutations_under_no_grad_or_inference_mode=False,
                mutation_inductor_storage_resize=False,
                mutates_storage_metadata=False,
                requires_grad=False,
                keep_input_mutations=False,
            ),
            InputAliasInfo(
                is_leaf=True,
                mutates_data=False,
                mutates_metadata=False,
                mutations_hidden_from_autograd=False,
                mutations_under_no_grad_or_inference_mode=False,
                mutation_inductor_storage_resize=False,
                mutates_storage_metadata=False,
                requires_grad=False,
                keep_input_mutations=False,
            ),
            InputAliasInfo(
                is_leaf=True,
                mutates_data=False,
                mutates_metadata=False,
                mutations_hidden_from_autograd=False,
                mutations_under_no_grad_or_inference_mode=False,
                mutation_inductor_storage_resize=False,
                mutates_storage_metadata=False,
                requires_grad=False,
                keep_input_mutations=False,
            ),
        ]

        # Mock output info: 2 outputs
        output_info = [
            OutputAliasInfo(
                output_type=OutputType.non_alias,
                raw_type=torch.Tensor,
                base_idx=None,
                dynamic_dims=None,
                requires_grad=False,
            ),
            OutputAliasInfo(
                output_type=OutputType.non_alias,
                raw_type=torch.Tensor,
                base_idx=None,
                dynamic_dims=None,
                requires_grad=False,
            ),
        ]

        # Create component metadata
        subclass_inp_meta = [PlainTensorMeta(0), PlainTensorMeta(1), PlainTensorMeta(2)]
        subclass_meta_components = (
            [PlainTensorMeta(0), PlainTensorMeta(1)],  # user outputs
            [PlainTensorMeta(0)],  # intermediate bases
        )

        # Create metadata in inference mode
        metadata = ViewAndMutationMeta(
            input_info=input_info,
            output_info=output_info,
            num_intermediate_bases=1,
            keep_input_mutations=False,
            traced_tangents=[],
            traced_tangents_descs=[],
            subclass_inp_meta=subclass_inp_meta,
            _subclass_meta_components=subclass_meta_components,
            subclass_tangent_meta=[],
            is_train=False,
        )

        # In inference mode with is_train=False, keep_input_mutations=False:
        # - mutated inputs should be included (1 input)
        # - user outputs should be included (2 outputs)
        # - intermediate bases should NOT be included (0 bases)
        # Total: 3 items
        fw_graph_out = metadata.subclass_fw_graph_out_meta
        self.assertEqual(len(fw_graph_out), 3)

        # Check order: mutated_inputs, user_outputs
        user_outs_meta, _ = subclass_meta_components
        self.assertEqual(fw_graph_out[0], subclass_inp_meta[0])  # mutated input
        self.assertEqual(fw_graph_out[1], user_outs_meta[0])  # user output 1
        self.assertEqual(fw_graph_out[2], user_outs_meta[1])  # user output 2

    def test_property_with_keep_input_mutations(self):
        """Test that the property correctly handles keep_input_mutations=True."""
        from torch._functorch._aot_autograd.schemas import (
            InputAliasInfo,
            OutputAliasInfo,
            OutputType,
        )

        # Mock input info: 3 inputs, first is data-mutated, second is metadata-mutated
        input_info = [
            InputAliasInfo(
                is_leaf=True,
                mutates_data=True,
                mutates_metadata=False,
                mutations_hidden_from_autograd=False,
                mutations_under_no_grad_or_inference_mode=False,
                mutation_inductor_storage_resize=False,
                mutates_storage_metadata=False,
                requires_grad=False,
                keep_input_mutations=True,
            ),
            InputAliasInfo(
                is_leaf=True,
                mutates_data=False,
                mutates_metadata=True,
                mutations_hidden_from_autograd=False,
                mutations_under_no_grad_or_inference_mode=False,
                mutation_inductor_storage_resize=False,
                mutates_storage_metadata=False,
                requires_grad=False,
                keep_input_mutations=True,
            ),
            InputAliasInfo(
                is_leaf=True,
                mutates_data=False,
                mutates_metadata=False,
                mutations_hidden_from_autograd=False,
                mutations_under_no_grad_or_inference_mode=False,
                mutation_inductor_storage_resize=False,
                mutates_storage_metadata=False,
                requires_grad=False,
                keep_input_mutations=True,
            ),
        ]

        # Mock output info: 2 outputs
        output_info = [
            OutputAliasInfo(
                output_type=OutputType.non_alias,
                raw_type=torch.Tensor,
                base_idx=None,
                dynamic_dims=None,
                requires_grad=False,
            ),
            OutputAliasInfo(
                output_type=OutputType.non_alias,
                raw_type=torch.Tensor,
                base_idx=None,
                dynamic_dims=None,
                requires_grad=False,
            ),
        ]

        # Create component metadata
        subclass_inp_meta = [PlainTensorMeta(0), PlainTensorMeta(1), PlainTensorMeta(2)]
        subclass_meta_components = (
            [PlainTensorMeta(0), PlainTensorMeta(1)],  # user outputs
            [],  # no intermediate bases
        )

        # Create metadata in inference mode with keep_input_mutations=True
        metadata = ViewAndMutationMeta(
            input_info=input_info,
            output_info=output_info,
            num_intermediate_bases=0,
            keep_input_mutations=True,
            traced_tangents=[],
            traced_tangents_descs=[],
            subclass_inp_meta=subclass_inp_meta,
            _subclass_meta_components=subclass_meta_components,
            subclass_tangent_meta=[],
            is_train=False,
        )

        # In inference mode with keep_input_mutations=True:
        # - only metadata-mutated inputs should be included (1 input)
        # - user outputs should be included (2 outputs)
        # Total: 3 items
        fw_graph_out = metadata.subclass_fw_graph_out_meta
        self.assertEqual(len(fw_graph_out), 3)

        # Check order: metadata_mutated_inputs, user_outputs
        user_outs_meta, _ = subclass_meta_components
        self.assertEqual(
            fw_graph_out[0], subclass_inp_meta[1]
        )  # metadata-mutated input
        self.assertEqual(fw_graph_out[1], user_outs_meta[0])  # user output 1
        self.assertEqual(fw_graph_out[2], user_outs_meta[1])  # user output 2

    def test_no_double_tracing_with_subclasses(self):
        """
        Test that we don't double-trace when switching from training to inference mode
        with tensor subclasses. This is the key fix for issue #132651.
        """

        def fn(x, y):
            # Simple function: x and y require grad initially (training guess),
            # but the output doesn't require grad (inference reality)
            return x + y

        # Use tensor subclasses
        x = SimpleTensorSubclass(torch.randn(3, 3, requires_grad=True))
        y = SimpleTensorSubclass(torch.randn(3, 3, requires_grad=True))

        # Count how many times run_functionalized_fw_and_collect_metadata is called
        call_count = 0
        original_func = run_functionalized_fw_and_collect_metadata

        def counting_wrapper(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_func(*args, **kwargs)

        # Patch and compile
        with patch(
            "torch._functorch.aot_autograd.run_functionalized_fw_and_collect_metadata",
            counting_wrapper,
        ):
            # Use a simple compiler that does nothing
            def nop_compiler(fx_module, _):
                return fx_module

            compiled_fn = aot_function(
                fn, fw_compiler=nop_compiler, bw_compiler=nop_compiler
            )

            # Run the compiled function
            with torch.no_grad():  # Force inference mode
                compiled_fn(x, y)

        # We should only call run_functionalized_fw_and_collect_metadata ONCE
        # Before the fix, it would be called TWICE in the subclass path
        self.assertEqual(
            call_count,
            1,
            f"Expected run_functionalized_fw_and_collect_metadata to be called once, "
            f"but it was called {call_count} times. This indicates the double-tracing bug still exists.",
        )

    def test_metadata_consistency_across_modes(self):
        """
        Test that changing is_train flag on the same metadata object produces
        consistent results with the property.
        """
        from torch._functorch._aot_autograd.schemas import (
            InputAliasInfo,
            OutputAliasInfo,
            OutputType,
        )

        # Mock input info: 1 mutated input
        input_info = [
            InputAliasInfo(
                is_leaf=True,
                mutates_data=True,
                mutates_metadata=False,
                mutations_hidden_from_autograd=False,
                mutations_under_no_grad_or_inference_mode=False,
                mutation_inductor_storage_resize=False,
                mutates_storage_metadata=False,
                requires_grad=False,
                keep_input_mutations=False,
            ),
        ]

        # Mock output info: 1 output
        output_info = [
            OutputAliasInfo(
                output_type=OutputType.non_alias,
                raw_type=torch.Tensor,
                base_idx=None,
                dynamic_dims=None,
                requires_grad=False,
            )
        ]

        # Create component metadata
        subclass_inp_meta = [PlainTensorMeta(0)]
        subclass_meta_components = (
            [PlainTensorMeta(0)],  # user outputs
            [PlainTensorMeta(0)],  # intermediate bases
        )

        # Create metadata in training mode
        metadata_train = ViewAndMutationMeta(
            input_info=input_info,
            output_info=output_info,
            num_intermediate_bases=1,
            keep_input_mutations=False,
            traced_tangents=[],
            traced_tangents_descs=[],
            subclass_inp_meta=subclass_inp_meta,
            _subclass_meta_components=subclass_meta_components,
            subclass_tangent_meta=[],
            is_train=True,
        )

        # Get the property value in training mode
        train_output = metadata_train.subclass_fw_graph_out_meta
        # Should have: mutated_input + user_output + intermediate_base = 3
        self.assertEqual(len(train_output), 3)

        # Now create the same metadata but in inference mode
        metadata_inference = ViewAndMutationMeta(
            input_info=input_info,
            output_info=output_info,
            num_intermediate_bases=1,
            keep_input_mutations=False,
            traced_tangents=[],
            traced_tangents_descs=[],
            subclass_inp_meta=subclass_inp_meta,
            _subclass_meta_components=subclass_meta_components,
            subclass_tangent_meta=[],
            is_train=False,
        )

        # Get the property value in inference mode
        inference_output = metadata_inference.subclass_fw_graph_out_meta
        # Should have: mutated_input + user_output = 2 (no intermediate bases)
        self.assertEqual(len(inference_output), 2)

        # The common parts should be identical
        self.assertEqual(inference_output[0], train_output[0])  # mutated input
        self.assertEqual(inference_output[1], train_output[1])  # user output


if __name__ == "__main__":
    run_tests()
