# Owner(s): ["module: inductor"]

import torch
from torch._inductor.kernel_template_choice import KernelTemplateChoice
from torch._inductor.template_heuristics.params import DictKernelTemplateParams
from torch._inductor.test_case import run_tests, TestCase


class TestKernelTemplateChoice(TestCase):
    def test_serialization_deserialization(self):
        """Test that KernelTemplateChoice can be serialized and deserialized."""
        # Create TensorBox inputs for a matrix multiplication
        M, N, K = 128, 256, 512
        device = torch.device("cpu")  # Use CPU for simplicity

        from torch._inductor import ir

        # Create input tensors A (M x K) and B (K x N)
        box_a = ir.TensorBox.create(
            ir.Buffer(
                name="a",
                layout=ir.FixedLayout(
                    device=device,
                    dtype=torch.float32,
                    size=[M, K],
                    stride=ir.FlexibleLayout.contiguous_strides([M, K]),
                ),
            )
        )

        box_b = ir.TensorBox.create(
            ir.Buffer(
                name="b",
                layout=ir.FixedLayout(
                    device=device,
                    dtype=torch.float32,
                    size=[K, N],
                    stride=ir.FlexibleLayout.contiguous_strides([K, N]),
                ),
            )
        )

        # Create kernel inputs and extract output layout from it
        kernel_inputs = torch._inductor.kernel_inputs.MMKernelInputs([box_a, box_b])
        output_layout = kernel_inputs.output_layout()

        # Create template params (using simple dict params)
        params = DictKernelTemplateParams(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "num_stages": 2,
                "num_warps": 4,
            }
        )

        # Create extra kwargs
        extra_kwargs = {"use_fp16_acc": False}

        # Create original KernelTemplateChoice
        original_ktc = KernelTemplateChoice(
            template=torch._inductor.kernel.mm.mm_template,
            params=params,
            extra_kwargs=extra_kwargs,
            layout=output_layout,
            inputs=kernel_inputs,
        )

        # Test serialization
        bundle_dict = original_ktc.to_bundle_dict()

        # Verify bundle structure
        self.assertIn("template_id", bundle_dict)
        self.assertIn("params", bundle_dict)
        self.assertIn("extra_kwargs", bundle_dict)
        self.assertEqual(
            bundle_dict["template_id"], torch._inductor.kernel.mm.mm_template.uid
        )
        self.assertEqual(bundle_dict["extra_kwargs"], extra_kwargs)

        # Test deserialization
        deserialized_ktc = KernelTemplateChoice.from_bundle_dict(
            bundle_dict=bundle_dict,
            layout=output_layout,
            inputs=kernel_inputs,
        )

        # Verify the deserialized object has the same properties
        self.assertEqual(deserialized_ktc.template.uid, original_ktc.template.uid)
        self.assertEqual(deserialized_ktc.extra_kwargs, original_ktc.extra_kwargs)
        self.assertEqual(
            deserialized_ktc.params.to_kwargs(), original_ktc.params.to_kwargs()
        )

        # Verify the layout and inputs are the same references
        self.assertIs(deserialized_ktc.layout, output_layout)
        self.assertIs(deserialized_ktc.inputs, kernel_inputs)

        # Test round-trip serialization
        second_bundle = deserialized_ktc.to_bundle_dict()
        self.assertEqual(bundle_dict, second_bundle)

    def test_missing_template_error(self):
        """Test that deserialization fails gracefully with missing template."""
        from torch._inductor import ir

        bundle_dict = {
            "template_id": "nonexistent_template_uid",
            "params": {"BLOCK_M": 64},
            "extra_kwargs": {},
        }

        # Create dummy layout and inputs
        device = torch.device("cpu")
        dummy_box = ir.TensorBox.create(
            ir.Buffer(
                name="dummy",
                layout=ir.FixedLayout(
                    device=device,
                    dtype=torch.float32,
                    size=[32, 32],
                    stride=ir.FlexibleLayout.contiguous_strides([32, 32]),
                ),
            )
        )
        inputs = torch._inductor.kernel_inputs.MMKernelInputs([dummy_box, dummy_box])
        layout = inputs.output_layout()

        # Should raise KeyError for missing template
        with self.assertRaises(KeyError) as cm:
            KernelTemplateChoice.from_bundle_dict(bundle_dict, layout, inputs)

        self.assertIn("nonexistent_template_uid", str(cm.exception))

    def test_missing_bundle_keys_error(self):
        """Test that deserialization fails gracefully with missing bundle keys."""
        # Missing required keys
        incomplete_bundle = {
            "template_id": torch._inductor.kernel.mm.mm_template.uid,
            # Missing 'params' and 'extra_kwargs'
        }

        from torch._inductor import ir

        # Create dummy layout and inputs
        device = torch.device("cpu")
        dummy_box = ir.TensorBox.create(
            ir.Buffer(
                name="dummy",
                layout=ir.FixedLayout(
                    device=device,
                    dtype=torch.float32,
                    size=[32, 32],
                    stride=ir.FlexibleLayout.contiguous_strides([32, 32]),
                ),
            )
        )
        inputs = torch._inductor.kernel_inputs.MMKernelInputs([dummy_box, dummy_box])
        layout = inputs.output_layout()

        # Should raise KeyError for missing keys
        with self.assertRaises(KeyError) as cm:
            KernelTemplateChoice.from_bundle_dict(incomplete_bundle, layout, inputs)

        self.assertIn("Missing required keys", str(cm.exception))


if __name__ == "__main__":
    run_tests()
