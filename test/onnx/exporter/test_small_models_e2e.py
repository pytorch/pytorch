# Owner(s): ["module: onnx"]
"""Unit tests for the onnx dynamo exporter."""

from __future__ import annotations

from typing import Any

import torchvision
from transformers import GPTJForCausalLM

import torch
from torch.onnx._internal.exporter import _testing as onnx_testing
from torch.testing._internal import common_utils


@common_utils.instantiate_parametrized_tests
class DynamoExporterTest(common_utils.TestCase):
    def test_insert_contiguous_between_transpose_and_view(self):
        class Model(torch.nn.Module):
            def forward(self, query, key, value):
                res = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value
                )
                rest = res.transpose(0, 1)
                return rest.view(8, 32, 128 * 64)

        model = Model()

        query = torch.rand(32, 8, 128, 64, dtype=torch.float16)
        key = torch.rand(32, 8, 128, 64, dtype=torch.float16)
        value = torch.rand(32, 8, 128, 64, dtype=torch.float16)

        ep = torch.export.export(model, (query, key, value), strict=False)
        self.assertNotIn("call_method", str(ep.graph))

        onnx_program = torch.onnx.export(
            model, (query, key, value), dynamo=True, fallback=False
        )
        onnx_testing.assert_onnx_program(onnx_program, atol=1e-3, rtol=1)

    def test_constant_complex(self):
        class MulModule(torch.nn.Module):
            def forward(self, x):
                y = 2 + 3j
                return torch.ops.aten.mul(x, y)

        # Example usage with complex inputs
        x = torch.tensor(
            [[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]], dtype=torch.complex64
        )

        onnx_program = torch.onnx.export(MulModule(), (x,), dynamo=True)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_pow_does_not_trigger_type_promotion(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x**2.0

        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)

        onnx_program = torch.onnx.export(Model(), (x,), dynamo=True)
        onnx_testing.assert_onnx_program(onnx_program)
        self.assertNotIn("Cast", [node.op_type for node in onnx_program.model.graph])

    def test_onnx_export_control_flow(self):
        class CondModel(torch.nn.Module):
            def forward(self, x):
                def true_fn(x):
                    return x + 1.0

                def false_fn(x):
                    return x - 42.0

                y = torch.cond(x.sum() > 0, true_fn, false_fn, [x])
                return y

        onnx_program = torch.onnx.export(
            CondModel(),
            (torch.tensor([1, 2]),),
            dynamo=True,
            fallback=False,
        )
        onnx_model = onnx_program.model
        self.assertIn("If", [node.op_type for node in onnx_model.graph])
        onnx_testing.assert_onnx_program(onnx_program)
        # Test different branches
        onnx_testing.assert_onnx_program(onnx_program, args=(torch.tensor([-1, -2]),))

    def test_onnx_export_nested_control_flow_and_nested_weights(self):
        class Submodule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Nested weight
                self.weight = torch.nn.Parameter(torch.tensor([100.0]))

            def forward(self, x):
                def true_fn(x):
                    return x * self.weight

                def false_fn(x):
                    return x / self.weight

                y = torch.cond(x.sum() <= 0, true_fn, false_fn, [x])
                return y

        class CondModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.submodule = Submodule()
                self.weight = torch.nn.Parameter(torch.tensor([42.0]))

            def forward(self, x):
                def true_fn(x):
                    return self.submodule(x - self.weight)

                def false_fn(x):
                    return x - self.weight

                y = torch.cond(x.sum() > 0, true_fn, false_fn, [x])
                return y

        onnx_program = torch.onnx.export(
            CondModel(),
            (torch.tensor([1, 2]),),
            dynamo=True,
            fallback=False,
        )
        onnx_testing.assert_onnx_program(onnx_program)
        onnx_testing.assert_onnx_program(onnx_program, args=(torch.tensor([0, 0]),))
        onnx_testing.assert_onnx_program(onnx_program, args=(torch.tensor([43, 43]),))

    def test_onnx_export_torchvision_ops(self):
        class VisionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, *x):
                out = torchvision.ops.nms(x[0], x[1], x[2])
                return out

        args = (
            torch.tensor([[0, 0, 1, 1], [0.5, 0.5, 1, 1]], dtype=torch.float),
            torch.tensor([0.1, 0.2]),
            0,
        )
        onnx_program = torch.onnx.export(VisionModel(), args, dynamo=True)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_onnx_export_huggingface_llm_models_with_kv_cache(self):
        model, kwargs, dynamic_axes, input_names, output_names = (
            _prepare_llm_model_gptj_to_test()
        )
        onnx_program = torch.onnx.export(
            model,
            kwargs=kwargs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=True,
        )
        # TODO(titaiwang): Investigate why ORT fails without optimization
        onnx_program.optimize()
        onnx_testing.assert_onnx_program(onnx_program)

    # TODO(justinchuby): Test multi-output HOPs


def _prepare_llm_model_gptj_to_test() -> (
    tuple[
        torch.nn.Module,
        dict[str, Any],
        dict[str, dict[int, str]],
        list[str],
        list[str],
    ]
):
    model = GPTJForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gptj")

    input_ids = torch.tensor(
        [
            [
                156,
                600,
                759,
                397,
                725,
                535,
                387,
                110,
                22,
                634,
                581,
                757,
                122,
                986,
                710,
                643,
            ],
            [
                926,
                305,
                867,
                280,
                682,
                649,
                50,
                186,
                171,
                311,
                523,
                113,
                288,
                141,
                516,
                571,
            ],
        ]
    )
    attention_mask = torch.tensor(
        [
            [
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            [
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
        ]
    )
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
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": past_key_values,
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
