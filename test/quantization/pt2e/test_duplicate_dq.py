# Owner(s): ["oncall: quantization"]
import copy
import unittest
from typing import Any, Dict

import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.observer import (
    HistogramObserver,
    MinMaxObserver,
    PlaceholderObserver,
)
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
    SharedQuantizationSpec,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    OP_TO_ANNOTATOR,
    QuantizationConfig,
)
from torch.testing._internal.common_quantization import QuantizationTestCase
from torch.testing._internal.common_utils import IS_WINDOWS


class TestHelperModules:
    class Conv2dWithObsSharingOps(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.hardtanh = torch.nn.Hardtanh()
            self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, x):
            x = self.conv(x)
            x = self.adaptive_avg_pool2d(x)
            x = self.hardtanh(x)
            x = x.view(-1, 3)
            x = self.linear(x)
            return x

    class Conv2dWithSharedDQ(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 3, 3)
            self.conv2 = torch.nn.Conv2d(3, 3, 1)
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, x):
            x = self.conv1(x)
            z = x.view(-1, 3)
            w = self.linear(z)

            y = self.conv2(x)
            add_output = x + y

            extra_output = x * 2
            return w, add_output, extra_output

    class ModuleForDifferentQconfig(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 3, 3)
            self.conv2 = torch.nn.Conv2d(3, 3, 1)
            self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.conv1(x)
            w = self.adaptive_avg_pool2d(x)

            y = self.conv2(x)
            add_output = x + y

            extra_output = x + 2
            return w, add_output, extra_output


_DEQUANTIZE_OPS = [
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.dequantize_per_channel.default,
]


@unittest.skipIf(IS_WINDOWS, "Windows not yet supported for torch.compile")
class TestDuplicateDQPass(QuantizationTestCase):
    def _test_duplicate_dq(
        self,
        model,
        example_inputs,
        quantizer,
    ):
        m_eager = model.eval()

        # program capture
        m = copy.deepcopy(m_eager)
        m = capture_pre_autograd_graph(
            m,
            example_inputs,
        )

        m = prepare_pt2e(m, quantizer)
        # Calibrate
        m(*example_inputs)
        m = convert_pt2e(m)

        pt2_quant_output = m(*example_inputs)
        for n in m.graph.nodes:
            annotation = n.meta.get("quantization_annotation", None)
            if annotation is not None:
                for arg in n.args:
                    if isinstance(arg, torch.fx.Node) and arg.target in _DEQUANTIZE_OPS:
                        self.assertEqual(len(arg.users.keys()), 1)

    def test_no_need_for_duplicate_dq(self):
        """
        Model under test
        conv2d -> avgpool -> hardtanh -> linear
        Check quantization tags on conv2d, avgpool and linear are correctly set
        """

        class BackendAQuantizer(Quantizer):
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                backend_string = "BackendA"
                quantization_config = get_symmetric_quantization_config(
                    is_per_channel=True
                )
                OP_TO_ANNOTATOR["linear"](gm, quantization_config)
                OP_TO_ANNOTATOR["conv"](gm, quantization_config)
                OP_TO_ANNOTATOR["adaptive_avg_pool2d"](gm, quantization_config)

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        example_inputs = (torch.randn(1, 3, 5, 7),)
        self._test_duplicate_dq(
            TestHelperModules.Conv2dWithObsSharingOps(),
            example_inputs,
            BackendAQuantizer(),
        )

    def test_simple_duplicate_dq(self):
        """
        Model under test
        conv2d -> conv2d -> add
             |          |
              --------->
             |
              -----> view_copy --> linear
             |
              -----> mul
        There should be three dq nodes because output for the
        first conv2d is fed to next conv2d, add, and view_copy + linear.
        All three are quantized.
        Thus DQ node is not duplicated for those three uses
        """

        class BackendAQuantizer(Quantizer):
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                backend_string = "BackendA"
                quantization_config = get_symmetric_quantization_config(
                    is_per_channel=True
                )
                OP_TO_ANNOTATOR["linear"](gm, quantization_config)
                OP_TO_ANNOTATOR["conv"](gm, quantization_config)
                OP_TO_ANNOTATOR["add"](gm, quantization_config)

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        example_inputs = (torch.randn(1, 3, 5, 7),)
        self._test_duplicate_dq(
            TestHelperModules.Conv2dWithSharedDQ(),
            example_inputs,
            BackendAQuantizer(),
        )

    def test_no_add_quant_duplicate_dq(self):
        """
        Model under test
        conv2d -> conv2d -> add
             |          |
              --------->
             |
              -----> view_copy --> linear
             |
              -----> mul
        There should be three dq nodes because output for the
        first conv2d is fed to next conv2d, and view_copy + linear.
        Both are quantized.
        However the skip connection to add and mul are not quantized.
        Thus DQ node is not duplicated for those two uses
        """

        class BackendAQuantizer(Quantizer):
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                backend_string = "BackendA"
                quantization_config = get_symmetric_quantization_config(
                    is_per_channel=True
                )
                OP_TO_ANNOTATOR["linear"](gm, quantization_config)
                OP_TO_ANNOTATOR["conv"](gm, quantization_config)

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        example_inputs = (torch.randn(1, 3, 5, 7),)
        self._test_duplicate_dq(
            TestHelperModules.Conv2dWithSharedDQ(),
            example_inputs,
            BackendAQuantizer(),
        )

    def test_avgpool_use_different_qconfig(self):
        """
        Model under test
        conv2d -> conv2d -> add
             |          |
              --------->
             |
              -----> adaptive_avgpool2d (different qconfig)
             |
              -----> add
        output
        conv2d -> dq -> conv2d -> add
             |                  |
              -------> dq ----->
             |
              -> dq -> q -> dq -----> adaptive_avgpool2d (different qconfig)
             |
              -> dq -----> add
        """

        def _get_uint8_quantization_config():
            act_observer_or_fake_quant_ctr = HistogramObserver  # type: ignore[assignment]
            act_quantization_spec = QuantizationSpec(
                dtype=torch.uint8,
                quant_min=0,
                quant_max=255,
                qscheme=torch.per_tensor_affine,
                observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(
                    eps=2**-12
                ),
            )
            weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = (  # noqa: F821
                MinMaxObserver
            )

            extra_args: Dict[str, Any] = {"eps": 2**-12}
            weight_quantization_spec = QuantizationSpec(
                dtype=torch.uint8,
                quant_min=0,
                quant_max=255,
                qscheme=torch.per_tensor_affine,
                ch_axis=0,
                is_dynamic=False,
                observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr.with_args(
                    **extra_args
                ),
            )

            bias_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = (  # noqa: F821
                PlaceholderObserver
            )
            bias_quantization_spec = QuantizationSpec(
                dtype=torch.float,
                observer_or_fake_quant_ctr=bias_observer_or_fake_quant_ctr,
            )
            quantization_config = QuantizationConfig(
                act_quantization_spec,
                act_quantization_spec,
                weight_quantization_spec,
                bias_quantization_spec,
            )
            return quantization_config

        class BackendAQuantizer(Quantizer):
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                backend_string = "BackendA"
                quantization_config = get_symmetric_quantization_config(
                    is_per_channel=True
                )
                avgpool_qconfig = _get_uint8_quantization_config()
                OP_TO_ANNOTATOR["conv"](gm, quantization_config)
                OP_TO_ANNOTATOR["add"](gm, quantization_config)
                for n in gm.graph.nodes:
                    if n.op == "call_function" and n.target == torch.ops.aten.mean.dim:
                        qspec = avgpool_qconfig.input_activation
                        input_act = n.args[0]
                        output_qspec = SharedQuantizationSpec((input_act, n))
                        n.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={input_act: qspec},
                            output_qspec=output_qspec,
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        example_inputs = (torch.randn(1, 3, 5, 7),)
        self._test_duplicate_dq(
            TestHelperModules.ModuleForDifferentQconfig(),
            example_inputs,
            BackendAQuantizer(),
        )
