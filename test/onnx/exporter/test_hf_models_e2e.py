# Owner(s): ["module: onnx"]
"""Unit LLM tests for the onnx dynamo exporter."""

from __future__ import annotations

from typing import Any

import transformers

import torch
from torch.onnx._internal.exporter import _testing as onnx_testing
from torch.testing._internal import common_utils


class DynamoExporterHfModelsTest(common_utils.TestCase):
    def export(self, model, args=(), kwargs=None, **options) -> torch.onnx.ONNXProgram:
        onnx_program = torch.onnx.export(
            model,
            args,
            kwargs=kwargs,
            dynamo=True,
            verbose=False,
            **options,
        )
        assert onnx_program is not None
        return onnx_program

    def test_onnx_export_huggingface_llm_models_with_kv_cache(self):
        model, kwargs, dynamic_axes, input_names, output_names = (
            _prepare_llm_model_gptj_to_test()
        )
        onnx_program = self.export(
            model,
            kwargs=kwargs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        onnx_testing.assert_onnx_program(onnx_program)

    def test_onnx_export_with_custom_axis_names_in_dynamic_shapes(self):
        model, kwargs, _, input_names, output_names = _prepare_llm_model_gptj_to_test()

        dynamic_shapes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "past_key_values": [
                (
                    {0: "batch_size", 2: "past_sequence_length"},
                    {0: "batch_size", 2: "past_sequence_length"},
                ),
                (
                    {0: "batch_size", 2: "past_sequence_length"},
                    {0: "batch_size", 2: "past_sequence_length"},
                ),
                (
                    {0: "batch_size", 2: "past_sequence_length"},
                    {0: "batch_size", 2: "past_sequence_length"},
                ),
                (
                    {0: "batch_size", 2: "past_sequence_length"},
                    {0: "batch_size", 2: "past_sequence_length"},
                ),
                (
                    {0: "batch_size", 2: "past_sequence_length"},
                    {0: "batch_size", 2: "past_sequence_length"},
                ),
            ],
            "attention_mask": {0: "batch_size", 1: "masked_sequence_length"},
            "position_ids": {0: "batch_size", 1: "sequence_length"},
        }

        onnx_program = self.export(
            model,
            kwargs=kwargs,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes,
            optimize=False,
        )
        onnx_testing.assert_onnx_program(onnx_program)

        # Check that the dynamic axes are correctly set in the ONNX model
        for dim, custom_name in zip(
            onnx_program.model.graph.inputs[0].shape,
            dynamic_shapes["input_ids"].values(),
        ):
            self.assertEqual(dim.value, custom_name)
        for idx in range(1, 11):
            shape_value = [
                dim if isinstance(dim, int) else dim.value
                for dim in onnx_program.model.graph.inputs[idx].shape
            ]
            self.assertEqual(shape_value, ["batch_size", 4, "past_sequence_length", 8])
        for dim, custom_name in zip(
            onnx_program.model.graph.inputs[11].shape,
            dynamic_shapes["attention_mask"].values(),
        ):
            self.assertEqual(dim.value, custom_name)
        for dim, custom_name in zip(
            onnx_program.model.graph.inputs[12].shape,
            dynamic_shapes["position_ids"].values(),
        ):
            self.assertEqual(dim.value, custom_name)


def _prepare_llm_model_gptj_to_test() -> tuple[
    torch.nn.Module,
    dict[str, Any],
    dict[str, dict[int, str]],
    list[str],
    list[str],
]:
    model = transformers.GPTJForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-gptj"
    )

    batch_size = 2
    input_seq_len = 16
    mask_seq_len = 32
    active_prob = 0.5
    vocab_size = 1000

    # Generate random input_ids with values between 0 and vocab_size-1
    input_ids = torch.randint(100, vocab_size, (batch_size, input_seq_len))
    # Generate random attention_mask with values 0 or 1, where 1 indicates an active token
    attention_mask = torch.bernoulli(
        torch.full((batch_size, mask_seq_len), active_prob)
    ).int()
    position_ids = torch.tensor(
        [
            [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        ]
    )
    past_key_values = [
        (torch.randn(2, 4, 16, 8), torch.randn(2, 4, 16, 8)),
        (torch.randn(2, 4, 16, 8), torch.randn(2, 4, 16, 8)),
        (torch.randn(2, 4, 16, 8), torch.randn(2, 4, 16, 8)),
        (torch.randn(2, 4, 16, 8), torch.randn(2, 4, 16, 8)),
        (torch.randn(2, 4, 16, 8), torch.randn(2, 4, 16, 8)),
    ]
    kwargs = {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "past_key_values.0.key": {0: "batch_size", 2: "past_sequence_length"},
        "past_key_values.0.value": {0: "batch_size", 2: "past_sequence_length"},
        "past_key_values.1.key": {0: "batch_size", 2: "past_sequence_length"},
        "past_key_values.1.value": {0: "batch_size", 2: "past_sequence_length"},
        "past_key_values.2.key": {0: "batch_size", 2: "past_sequence_length"},
        "past_key_values.2.value": {0: "batch_size", 2: "past_sequence_length"},
        "past_key_values.3.key": {0: "batch_size", 2: "past_sequence_length"},
        "past_key_values.3.value": {0: "batch_size", 2: "past_sequence_length"},
        "past_key_values.4.key": {0: "batch_size", 2: "past_sequence_length"},
        "past_key_values.4.value": {0: "batch_size", 2: "past_sequence_length"},
        "attention_mask": {
            0: "batch_size",
            1: "past_sequence_length + sequence_length",
        },
        "position_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
        "present.0.key": {0: "batch_size", 2: "past_sequence_length + sequence_length"},
        "present.0.value": {
            0: "batch_size",
            2: "past_sequence_length + sequence_length",
        },
        "present.1.key": {0: "batch_size", 2: "past_sequence_length + sequence_length"},
        "present.1.value": {
            0: "batch_size",
            2: "past_sequence_length + sequence_length",
        },
        "present.2.key": {0: "batch_size", 2: "past_sequence_length + sequence_length"},
        "present.2.value": {
            0: "batch_size",
            2: "past_sequence_length + sequence_length",
        },
        "present.3.key": {0: "batch_size", 2: "past_sequence_length + sequence_length"},
        "present.3.value": {
            0: "batch_size",
            2: "past_sequence_length + sequence_length",
        },
        "present.4.key": {0: "batch_size", 2: "past_sequence_length + sequence_length"},
        "present.4.value": {
            0: "batch_size",
            2: "past_sequence_length + sequence_length",
        },
    }
    input_names = [
        "input_ids",
        "past_key_values.0.key",
        "past_key_values.0.value",
        "past_key_values.1.key",
        "past_key_values.1.value",
        "past_key_values.2.key",
        "past_key_values.2.value",
        "past_key_values.3.key",
        "past_key_values.3.value",
        "past_key_values.4.key",
        "past_key_values.4.value",
        "attention_mask",
        "position_ids",
    ]
    output_names = [
        "logits",
        "present.0.key",
        "present.0.value",
        "present.1.key",
        "present.1.value",
        "present.2.key",
        "present.2.value",
        "present.3.key",
        "present.3.value",
        "present.4.key",
        "present.4.value",
    ]

    return model, kwargs, dynamic_axes, input_names, output_names


if __name__ == "__main__":
    common_utils.run_tests()
