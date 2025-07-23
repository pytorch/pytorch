# Owner(s): ["oncall: quantization"]
import copy
from typing import Any, Optional

import torch
from torch._higher_order_ops.out_dtype import out_dtype  # noqa: F401
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer import Quantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.export import export_for_training
from torch.testing._internal.common_quantization import (
    NodeSpec as ns,
    QuantizationTestCase,
    skipIfNoQNNPACK,
    TestHelperModules,
)
from torch.testing._internal.common_utils import raise_on_run_directly


@skipIfNoQNNPACK
class TestPT2ERepresentation(QuantizationTestCase):
    def _test_representation(
        self,
        model: torch.nn.Module,
        example_inputs: tuple[Any, ...],
        quantizer: Quantizer,
        ref_node_occurrence: dict[ns, int],
        non_ref_node_occurrence: dict[ns, int],
        fixed_output_tol: Optional[float] = None,
        output_scale_idx: int = 2,
    ) -> torch.nn.Module:
        # resetting dynamo cache
        torch._dynamo.reset()
        model = export_for_training(model, example_inputs, strict=True).module()
        model_copy = copy.deepcopy(model)

        model = prepare_pt2e(model, quantizer)
        # Calibrate
        model(*example_inputs)
        model = convert_pt2e(model, use_reference_representation=True)
        self.checkGraphModuleNodes(model, expected_node_occurrence=ref_node_occurrence)
        # make sure it runs
        pt2e_quant_output = model(*example_inputs)

        # TODO: torchdynamo times out when we do this, we can enable numerical checking
        # after that is fixed
        model_copy = prepare_pt2e(model_copy, quantizer)
        # Calibrate
        model_copy(*example_inputs)
        model_copy = convert_pt2e(model_copy, use_reference_representation=False)
        self.checkGraphModuleNodes(
            model_copy, expected_node_occurrence=non_ref_node_occurrence
        )
        pt2e_quant_output_copy = model_copy(*example_inputs)

        output_tol = None
        if fixed_output_tol is not None:
            output_tol = fixed_output_tol
        else:
            idx = 0
            for n in model_copy.graph.nodes:
                if (
                    n.target
                    == torch.ops.quantized_decomposed.quantize_per_tensor.default
                ):
                    idx += 1
                    if idx == output_scale_idx:
                        output_tol = n.args[1]
            assert output_tol is not None

        # make sure the result is off by one at most in the quantized integer representation
        self.assertTrue(
            torch.max(torch.abs(pt2e_quant_output_copy - pt2e_quant_output))
            <= (2 * output_tol + 1e-5)
        )

    def test_static_linear(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x)

        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=False)
        quantizer.set_global(operator_config)
        example_inputs = (torch.randn(2, 5),)

        self._test_representation(
            M().eval(),
            example_inputs,
            quantizer,
            ref_node_occurrence={},
            non_ref_node_occurrence={},
        )

    def test_dynamic_linear(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x)

        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(
            is_per_channel=False, is_dynamic=True
        )
        quantizer.set_global(operator_config)
        example_inputs = (torch.randn(2, 5),)

        self._test_representation(
            M().eval(),
            example_inputs,
            quantizer,
            ref_node_occurrence={},
            non_ref_node_occurrence={},
            fixed_output_tol=1e-4,
        )

    def test_conv2d(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv2d = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                return self.conv2d(x)

        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=False)
        quantizer.set_global(operator_config)
        example_inputs = (torch.randn(1, 3, 3, 3),)

        self._test_representation(
            M().eval(),
            example_inputs,
            quantizer,
            ref_node_occurrence={},
            non_ref_node_occurrence={},
        )

    def test_add(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                return x + y

        quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(quantization_config)
        M().eval()

        example_inputs = (
            torch.randn(1, 3, 3, 3),
            torch.randn(1, 3, 3, 3),
        )

        self._test_representation(
            M().eval(),
            example_inputs,
            quantizer,
            ref_node_occurrence={},
            non_ref_node_occurrence={},
        )

    def test_add_relu(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                out = x + y
                out = torch.nn.functional.relu(out)
                return out

        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(operator_config)

        example_inputs = (
            torch.randn(1, 3, 3, 3),
            torch.randn(1, 3, 3, 3),
        )
        ref_node_occurrence = {
            ns.call_function(out_dtype): 2,
        }

        self._test_representation(
            M().eval(),
            example_inputs,
            quantizer,
            ref_node_occurrence=ref_node_occurrence,
            non_ref_node_occurrence={},
        )

    def test_maxpool2d(self):
        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(operator_config)
        m_eager = TestHelperModules.ConvMaxPool2d().eval()

        example_inputs = (torch.randn(1, 2, 2, 2),)

        self._test_representation(
            m_eager,
            example_inputs,
            quantizer,
            ref_node_occurrence={},
            non_ref_node_occurrence={},
        )

    def test_qdq_per_channel(self):
        """Test representation for quantize_per_channel and dequantize_per_channel op"""

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x)

        quantizer = XNNPACKQuantizer()
        # use per channel quantization for weight
        operator_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(operator_config)
        M().eval()

        inputs = [
            (torch.randn(1, 5),),
            (torch.randn(1, 3, 5),),
            (torch.randn(1, 3, 3, 5),),
            (torch.randn(1, 3, 3, 3, 5),),
        ]
        for example_inputs in inputs:
            ref_node_occurrence = {
                ns.call_function(
                    torch.ops.quantized_decomposed.quantize_per_channel.default
                ): 0,
                ns.call_function(
                    torch.ops.quantized_decomposed.dequantize_per_channel.default
                ): 0,
            }
            non_ref_node_occurrence = {
                # quantize_per_channel is folded
                ns.call_function(
                    torch.ops.quantized_decomposed.quantize_per_channel.default
                ): 0,
                ns.call_function(
                    torch.ops.quantized_decomposed.dequantize_per_channel.default
                ): 1,
            }

            self._test_representation(
                M().eval(),
                example_inputs,
                quantizer,
                ref_node_occurrence,
                non_ref_node_occurrence,
                output_scale_idx=2,
            )

    def test_qdq(self):
        """Test representation for quantize and dequantize op"""

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                return x + y

        quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(quantization_config)
        M().eval()

        example_inputs = (
            torch.randn(1, 3, 3, 3),
            torch.randn(1, 3, 3, 3),
        )
        ref_node_occurrence = {
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 0,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 0,
        }
        non_ref_node_occurrence = {
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 3,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 3,
        }
        self._test_representation(
            M().eval(),
            example_inputs,
            quantizer,
            ref_node_occurrence,
            non_ref_node_occurrence,
        )


if __name__ == "__main__":
    raise_on_run_directly("test/test_quantization.py")
