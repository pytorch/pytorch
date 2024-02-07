# Owner(s): ["oncall: quantization"]
import copy
import unittest
from typing import Any, Dict

import torch
import torch._export as export

from torch.ao.quantization.observer import (
    HistogramObserver,
    MinMaxObserver,
    PlaceholderObserver,
)
from torch.ao.quantization.pt2e.util_passes import DuplicateDynamicQuantChainPass
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
    SharedQuantizationSpec,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    OP_TO_ANNOTATOR,
    QuantizationConfig,
)

from torch.testing._internal.common_quantization import (
    NodeSpec as ns,
    QuantizationTestCase,
    TestHelperModules,
)
from torch.testing._internal.common_utils import IS_WINDOWS


class MyTestHelperModules:
    class TwoFanOutLinears(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(8, 16, bias=False)
            self.linear2 = torch.nn.Linear(8, 16)

        def forward(self, x):
            x1 = self.linear1(x)
            x2 = self.linear2(x)
            return x1 + x2


_DEQUANTIZE_OPS = [
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.dequantize_per_channel.default,
]


@unittest.skipIf(IS_WINDOWS, "Windows not yet supported for torch.compile")
class TestDuplicateDynamicQuantChainPass(QuantizationTestCase):
    def _test_duplicate_chain(
        self,
        model,
        example_inputs,
        quantizer,
        before_node_occurrences,
        after_node_occurrences,
    ):
        m_eager = model.eval()

        # program capture
        m = copy.deepcopy(m_eager)
        m = export.capture_pre_autograd_graph(
            m,
            example_inputs,
        )

        m = prepare_pt2e(m, quantizer)
        # Calibrate
        m(*example_inputs)
        m = convert_pt2e(m, fold_quantize=True)
        print(m)
        node_occurrence = {
            ns.call_function(k): v for k, v in before_node_occurrences.items()
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)
        DuplicateDynamicQuantChainPass()(m)
        node_occurrence = {
            ns.call_function(k): v for k, v in after_node_occurrences.items()
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)
        return m

    def test_no_need_for_duplicate(self):
        """
        Model under test
        linear -> linear
        Check two chose qparams, q, dq before and after the pass
        """

        quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(
            is_per_channel=True, is_dynamic=True
        )
        quantizer.set_global(quantization_config)
        example_inputs = (torch.randn(9, 8),)
        before_node_occurrence = {
            torch.ops.quantized_decomposed.choose_qparams.tensor: 2,
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 2,
            # note: quantize op for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
        }
        self._test_duplicate_chain(
            TestHelperModules.TwoLinearModule().eval(),
            example_inputs,
            quantizer,
            before_node_occurrences=before_node_occurrence,
            after_node_occurrences=before_node_occurrence,
        )

    def test_simple_duplicate_chain(self):
        """
        Model under test
        x -> linear  -> add
         |           |
          -> linear -
        Before duplication there should be only 1 dynamic q chain
        After duplication there should be 2 dynamic q chains
        """

        quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(
            is_per_channel=True, is_dynamic=True
        )
        quantizer.set_global(quantization_config)
        example_inputs = (torch.randn(9, 8),)
        before_node_occurrence = {
            torch.ops.quantized_decomposed.choose_qparams.tensor: 1,
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 1,
            # note: quantize op for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
        }
        after_node_occurrence = {
            torch.ops.quantized_decomposed.choose_qparams.tensor: 2,
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 2,
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
        }
        self._test_duplicate_chain(
            MyTestHelperModules.TwoFanOutLinears().eval(),
            example_inputs,
            quantizer,
            before_node_occurrences=before_node_occurrence,
            after_node_occurrences=after_node_occurrence,
        )

    @unittest.skip("Set module name API does not work correctly when used as here.")
    def test_no_duplicate_chain_different_qscheme(self):
        """
        Model under test
        x -> linear1  -> linear 2
        """

        quantizer = XNNPACKQuantizer()
        dynamic_qconfig = get_symmetric_quantization_config(
            is_per_channel=True, is_dynamic=True
        )
        static_qconfig = get_symmetric_quantization_config(is_per_channel=False)
        quantizer.set_module_name("linear1", dynamic_qconfig)
        quantizer.set_module_name("linear2", static_qconfig)
        example_inputs = (torch.randn(9, 8),)
        before_node_occurrence = {
            torch.ops.quantized_decomposed.choose_qparams.tensor: 1,
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 1,
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
        }
        after_node_occurrence = {
            torch.ops.quantized_decomposed.choose_qparams.tensor: 1,
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 1,
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
        }
        self._test_duplicate_chain(
            TestHelperModules.TwoLinearModule().eval(),
            example_inputs,
            quantizer,
            before_node_occurrences=before_node_occurrence,
            after_node_occurrences=after_node_occurrence,
        )
