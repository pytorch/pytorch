# Owner(s): ["oncall: quantization"]
import copy

import unittest
from typing import List

import torch
import torch._export
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer import QuantizationAnnotation, Quantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import OP_TO_ANNOTATOR

from torch.fx import Node

from torch.testing._internal.common_quantization import QuantizationTestCase
from torch.testing._internal.common_utils import IS_WINDOWS


class TestHelperModules:
    class Conv2dWithObsSharingOps(torch.nn.Module):
        def __init__(self):
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


def _tag_partitions(
    backend_name: str, op_name: str, annotated_partitions: List[List[Node]]
):
    for index, partition_nodes in enumerate(annotated_partitions):
        tag_name = backend_name + "_" + op_name + "_" + str(index)
        for node in partition_nodes:
            assert "quantization_tag" not in node.meta, f"{node} is already tagged"
            node.meta["quantization_tag"] = tag_name


_QUANT_OPS = {
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.quantize_per_channel.default,
    torch.ops.quantized_decomposed.dequantize_per_channel.default,
    torch.ops.quantized_decomposed.choose_qparams.tensor,
}


# TODO: rename to TestPortMetadataPass to align with the util name?
@unittest.skipIf(IS_WINDOWS, "Windows not yet supported for torch.compile")
class TestMetaDataPorting(QuantizationTestCase):
    def _test_quant_tag_preservation_through_decomp(
        self, model, example_inputs, from_node_to_tags
    ):
        ep = torch.export.export(model, example_inputs)
        found_tags = True
        not_found_nodes = ""
        for from_node, tag in from_node_to_tags.items():
            for n in ep.graph_module.graph.nodes:
                from_node_meta = n.meta.get("from_node", None)
                if from_node_meta is None:
                    continue
                if not isinstance(from_node_meta, list):
                    raise ValueError(
                        f"from_node metadata is of type {type(from_node_meta)}, but expected list"
                    )
                for meta in from_node_meta:
                    node_target = meta[1]
                    if node_target == from_node:
                        node_tag = n.meta.get("quantization_tag", None)
                        if node_tag is None or tag != node_tag:
                            not_found_nodes += str(n.target) + ", "
                            found_tags = False
                            break
                if not found_tags:
                    break
        self.assertTrue(
            found_tags,
            f"Decomposition did not preserve quantization tag for {not_found_nodes}",
        )

    def _test_metadata_porting(
        self,
        model,
        example_inputs,
        quantizer,
        node_tags=None,
    ) -> torch.fx.GraphModule:
        m_eager = model.eval()

        # program capture
        m = copy.deepcopy(m_eager)
        m = torch._export.capture_pre_autograd_graph(
            m,
            example_inputs,
        )

        m = prepare_pt2e(m, quantizer)
        # Calibrate
        m(*example_inputs)
        m = convert_pt2e(m)

        pt2_quant_output = m(*example_inputs)
        recorded_node_tags = {}
        for n in m.graph.nodes:
            if "quantization_tag" not in n.meta:
                continue
            if n.op == "call_function" and n.target in _QUANT_OPS:
                key = n.target
            elif n.op == "get_attr":
                key = "get_attr"
            else:
                continue

            if key not in recorded_node_tags:
                recorded_node_tags[key] = set()

            if (
                n.op == "call_function"
                and n.meta["quantization_tag"] in recorded_node_tags[key]
            ):
                raise ValueError(
                    f"{key} {n.format_node()} has tag {n.meta['quantization_tag']} that "
                    "is associated with another node of the same type"
                )
            recorded_node_tags[key].add(n.meta["quantization_tag"])

        self.assertEqual(set(recorded_node_tags.keys()), set(node_tags.keys()))
        for k, v in recorded_node_tags.items():
            self.assertEqual(v, node_tags[k])
        return m

    def test_simple_metadata_porting(self):
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
                annotated_partitions = OP_TO_ANNOTATOR["linear"](
                    gm, quantization_config
                )
                _tag_partitions(backend_string, "linear", annotated_partitions)
                annotated_partitions = OP_TO_ANNOTATOR["conv"](gm, quantization_config)
                _tag_partitions(backend_string, "conv2d", annotated_partitions)
                annotated_partitions = OP_TO_ANNOTATOR["adaptive_avg_pool2d"](
                    gm, quantization_config
                )
                _tag_partitions(
                    backend_string, "adaptive_avg_pool2d", annotated_partitions
                )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        example_inputs = (torch.randn(1, 3, 5, 5),)
        get_attr_tags = {
            "BackendA_conv2d_0",
            "BackendA_linear_0",
        }
        quantize_per_tensor_tags = {
            "BackendA_conv2d_0",
            "BackendA_adaptive_avg_pool2d_0",
            "BackendA_linear_0",
        }
        dequantize_per_tensor_tags = {
            "BackendA_adaptive_avg_pool2d_0",
            "BackendA_conv2d_0",
            "BackendA_linear_0",
        }
        dequantize_per_channel_tags = {"BackendA_conv2d_0", "BackendA_linear_0"}
        node_tags = {
            "get_attr": get_attr_tags,
            torch.ops.quantized_decomposed.quantize_per_tensor.default: quantize_per_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: dequantize_per_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: dequantize_per_channel_tags,
        }
        m = self._test_metadata_porting(
            TestHelperModules.Conv2dWithObsSharingOps(),
            example_inputs,
            BackendAQuantizer(),
            node_tags,
        )

        from_node_to_tags = {
            torch.ops.aten.adaptive_avg_pool2d.default: "BackendA_adaptive_avg_pool2d_0",
            torch.ops.aten.linear.default: "BackendA_linear_0",
        }
        self._test_quant_tag_preservation_through_decomp(
            m, example_inputs, from_node_to_tags
        )

    def test_metadata_porting_with_no_quant_inbetween(self):
        """
        Model under test
        conv2d -> avgpool -> hardtanh -> linear
        Dont quantize avgpool
        Check quantization tags on conv2d and linear are correctly set
        """

        class BackendAQuantizer(Quantizer):
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                backend_string = "BackendA"
                quantization_config = get_symmetric_quantization_config(
                    is_per_channel=True
                )
                annotated_partitions = OP_TO_ANNOTATOR["linear"](
                    gm, quantization_config
                )
                _tag_partitions(backend_string, "linear", annotated_partitions)
                annotated_partitions = OP_TO_ANNOTATOR["conv"](gm, quantization_config)
                _tag_partitions(backend_string, "conv2d", annotated_partitions)

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        example_inputs = (torch.randn(1, 3, 5, 5),)
        get_attr_tags = {"BackendA_conv2d_0", "BackendA_linear_0"}
        quantize_per_tensor_tags = {"BackendA_conv2d_0", "BackendA_linear_0"}
        dequantize_per_tensor_tags = {"BackendA_conv2d_0", "BackendA_linear_0"}
        dequantize_per_channel_tags = {"BackendA_conv2d_0", "BackendA_linear_0"}
        node_tags = {
            "get_attr": get_attr_tags,
            torch.ops.quantized_decomposed.quantize_per_tensor.default: quantize_per_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: dequantize_per_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: dequantize_per_channel_tags,
        }
        self._test_metadata_porting(
            TestHelperModules.Conv2dWithObsSharingOps(),
            example_inputs,
            BackendAQuantizer(),
            node_tags,
        )

    @unittest.skip("Temporarily disabled")
    def test_metadata_porting_for_dq(self):
        """
        Model under test
        conv2d -> avgpool -> hardtanh -> linear
        Quantize all except linear.
        Quantize linear with dynamic quantization
        Check quantization tags on conv2d, avgpool and linear are correctly set
        """

        class BackendAQuantizer(Quantizer):
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                backend_string = "BackendA"
                # static quantiazation
                quantization_config = get_symmetric_quantization_config(
                    is_per_channel=True
                )
                annotated_partitions = OP_TO_ANNOTATOR["conv"](gm, quantization_config)
                _tag_partitions(backend_string, "conv2d", annotated_partitions)
                annotated_partitions = OP_TO_ANNOTATOR["adaptive_avg_pool2d"](
                    gm, quantization_config
                )
                _tag_partitions(
                    backend_string, "adaptive_avg_pool2d", annotated_partitions
                )

                # dynamic quantization
                quantization_config_dynamic = get_symmetric_quantization_config(
                    is_per_channel=True, is_dynamic=True
                )
                annotated_partitions = OP_TO_ANNOTATOR["linear"](
                    gm, quantization_config_dynamic
                )
                _tag_partitions(backend_string, "linear_dynamic", annotated_partitions)

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        example_inputs = (torch.randn(1, 3, 5, 5),)
        # TODO: add get_attr_tags when the test is re-enabled
        get_attr_tags = {}
        quantize_per_tensor_tags = {
            "BackendA_conv2d_0",
            "BackendA_adaptive_avg_pool2d_0",
        }
        quantize_per_tensor_tensor_tags = {"BackendA_linear_dynamic_0"}
        choose_qparams_tensor_tensor_tags = {"BackendA_linear_dynamic_0"}
        dequantize_per_tensor_tags = {
            "BackendA_adaptive_avg_pool2d_0",
            "BackendA_conv2d_0",
        }
        dequantize_per_tensor_tensor_tags = {"BackendA_linear_dynamic_0"}
        dequantize_per_channel_tags = {
            "BackendA_conv2d_0",
            "BackendA_linear_dynamic_0",
        }
        node_tags = {
            "get_attr": get_attr_tags,
            torch.ops.quantized_decomposed.quantize_per_tensor.default: quantize_per_tensor_tags,
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: quantize_per_tensor_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: dequantize_per_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: dequantize_per_tensor_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: dequantize_per_channel_tags,
            torch.ops.quantized_decomposed.choose_qparams.tensor: choose_qparams_tensor_tensor_tags,
        }
        self._test_metadata_porting(
            TestHelperModules.Conv2dWithObsSharingOps(),
            example_inputs,
            BackendAQuantizer(),
            node_tags,
        )

    def test_metadata_porting_for_two_dq(self):
        """
        Model under test
        conv2d -> avgpool -> hardtanh -> linear
        Quantize linear and conv with dynamic quantization
        Check quantization tags on conv2d, avgpool and linear are correctly set
        """

        class BackendAQuantizer(Quantizer):
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                backend_string = "BackendA"

                # dynamic quantization
                quantization_config_dynamic = get_symmetric_quantization_config(
                    is_per_channel=True, is_dynamic=True
                )
                annotated_partitions = OP_TO_ANNOTATOR["conv"](
                    gm, quantization_config_dynamic
                )
                _tag_partitions(backend_string, "conv2d_dynamic", annotated_partitions)
                annotated_partitions = OP_TO_ANNOTATOR["linear"](
                    gm, quantization_config_dynamic
                )
                _tag_partitions(backend_string, "linear_dynamic", annotated_partitions)

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        example_inputs = (torch.randn(1, 3, 5, 5),)
        get_attr_tags = {
            "BackendA_conv2d_dynamic_0",
            "BackendA_linear_dynamic_0",
        }
        choose_qparams_tensor_tags = {
            "BackendA_conv2d_dynamic_0",
            "BackendA_linear_dynamic_0",
        }
        quantize_per_tensor_tensor_tags = {
            "BackendA_conv2d_dynamic_0",
            "BackendA_linear_dynamic_0",
        }
        dequantize_per_tensor_tensor_tags = {
            "BackendA_conv2d_dynamic_0",
            "BackendA_linear_dynamic_0",
        }
        dequantize_per_channel_tags = {
            "BackendA_conv2d_dynamic_0",
            "BackendA_linear_dynamic_0",
        }
        node_tags = {
            "get_attr": get_attr_tags,
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: quantize_per_tensor_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: dequantize_per_tensor_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: dequantize_per_channel_tags,
            torch.ops.quantized_decomposed.choose_qparams.tensor: choose_qparams_tensor_tags,
        }
        self._test_metadata_porting(
            TestHelperModules.Conv2dWithObsSharingOps(),
            example_inputs,
            BackendAQuantizer(),
            node_tags,
        )

    def test_metadata_porting_for_dq_no_static_q(self):
        """
        Model under test
        conv2d -> avgpool -> hardtanh -> linear
        Dont quantize anything except linear.
        Quantize linear with dynamic quantization
        Check quantization tags on conv2d, avgpool and linear are correctly set
        """

        class BackendAQuantizer(Quantizer):
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                backend_string = "BackendA"
                # dynamic quantization
                quantization_config_dynamic = get_symmetric_quantization_config(
                    is_per_channel=True, is_dynamic=True
                )
                annotated_partitions = OP_TO_ANNOTATOR["linear"](
                    gm, quantization_config_dynamic
                )
                _tag_partitions(backend_string, "linear_dynamic", annotated_partitions)

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        example_inputs = (torch.randn(1, 3, 5, 5),)
        get_attr_tags = {"BackendA_linear_dynamic_0"}
        choose_qparams_tensor_tags = {"BackendA_linear_dynamic_0"}
        quantize_per_tensor_tensor_tags = {"BackendA_linear_dynamic_0"}
        dequantize_per_tensor_tensor_tags = {"BackendA_linear_dynamic_0"}
        dequantize_per_channel_tags = {"BackendA_linear_dynamic_0"}
        node_tags = {
            "get_attr": get_attr_tags,
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: quantize_per_tensor_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: dequantize_per_tensor_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: dequantize_per_channel_tags,
            torch.ops.quantized_decomposed.choose_qparams.tensor: choose_qparams_tensor_tags,
        }
        self._test_metadata_porting(
            TestHelperModules.Conv2dWithObsSharingOps(),
            example_inputs,
            BackendAQuantizer(),
            node_tags,
        )

    def test_no_metadata_porting(self):
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

        example_inputs = (torch.randn(1, 3, 5, 5),)
        node_tags = {}
        m = self._test_metadata_porting(
            TestHelperModules.Conv2dWithObsSharingOps(),
            example_inputs,
            BackendAQuantizer(),
            node_tags,
        )

        from_node_to_tags = {}
        self._test_quant_tag_preservation_through_decomp(
            m, example_inputs, from_node_to_tags
        )

    def test_no_metadata_porting_through_unknown_ops(self):
        """
        Model under test
        matmul -> add -> relu
        matmul has get_attr as first input, but the quantization_tag should not be
        propagated to add even if it's part of a chain that ends at get_attr
        """

        class MatmulWithConstInput(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.rand(8, 16)))

            def forward(self, x, y):
                x = torch.matmul(self.w, x)
                z = x + y
                return torch.nn.functional.relu(z)

        class BackendAQuantizer(Quantizer):
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                backend_string = "BackendA"
                qconfig = get_symmetric_quantization_config()
                for n in gm.graph.nodes:
                    if n.op != "call_function":
                        continue

                    n.meta["quantization_annotation"] = QuantizationAnnotation(
                        input_qspec_map={n.args[0]: qconfig.input_activation},
                        output_qspec=qconfig.output_activation,
                    )

                    tag = str(n.target)
                    n.meta["quantization_tag"] = tag
                    for arg in n.args:
                        if arg.op == "get_attr":
                            arg.meta["quantization_tag"] = tag

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        example_inputs = (torch.randn(16, 24), torch.randn(8, 24))
        get_attr_tags = {"aten.matmul.default"}
        quantize_per_tensor_tensor_tags = {
            "aten.matmul.default",
            "aten.add.Tensor",
            "aten.relu.default",
        }
        dequantize_per_tensor_tensor_tags = {
            "aten.matmul.default",
            "aten.add.Tensor",
            "aten.relu.default",
        }
        node_tags = {
            "get_attr": get_attr_tags,
            torch.ops.quantized_decomposed.quantize_per_tensor.default: quantize_per_tensor_tensor_tags,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: dequantize_per_tensor_tensor_tags,
        }
        m = self._test_metadata_porting(
            MatmulWithConstInput(),
            example_inputs,
            BackendAQuantizer(),
            node_tags,
        )
