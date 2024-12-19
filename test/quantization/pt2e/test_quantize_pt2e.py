# Owner(s): ["oncall: quantization"]
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.ao.quantization import observer, ObserverOrFakeQuantize, QConfigMapping
from torch.ao.quantization.qconfig import (
    default_per_channel_symmetric_qnnpack_qconfig,
    float_qparams_weight_only_qconfig,
    per_channel_weight_observer_range_neg_127_to_127,
    QConfig,
    weight_observer_range_neg_127_to_127,
)
from torch.ao.quantization.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torch.ao.quantization.quantizer import (
    DerivedQuantizationSpec,
    EdgeOrNode,
    FixedQParamsQuantizationSpec,
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
    SharedQuantizationSpec,
)
from torch.ao.quantization.quantizer.composable_quantizer import (  # noqa: F811
    ComposableQuantizer,
)
from torch.ao.quantization.quantizer.embedding_quantizer import (  # noqa: F811
    EmbeddingQuantizer,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    OP_TO_ANNOTATOR,
    QuantizationConfig,
)
from torch.export import export_for_training
from torch.fx import Node
from torch.testing._internal.common_quantization import (
    NodeSpec as ns,
    PT2EQuantizationTestCase,
    skipIfNoQNNPACK,
    TestHelperModules,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfHpu,
    TemporaryFileName,
    TEST_CUDA,
    TEST_HPU,
)


@skipIfNoQNNPACK
class TestQuantizePT2E(PT2EQuantizationTestCase):
    def test_simple_quantizer(self):
        # TODO: use OP_TO_ANNOTATOR
        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.conv2d.default
                    ):
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_observer,
                        )
                        weight_qspec = QuantizationSpec(
                            dtype=torch.int8,
                            quant_min=-128,
                            quant_max=127,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_weight_observer,
                        )
                        bias_qspec = QuantizationSpec(
                            dtype=torch.float32,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.PlaceholderObserver,
                        )
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                                weight: weight_qspec,
                                bias: bias_qspec,
                            },
                            output_qspec=act_qspec,
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        example_inputs = (torch.randn(1, 3, 5, 5),)
        node_occurrence = {
            # two for input of the first conv, one for output for the first conv
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
        }
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        ]
        self._test_quantizer(
            TestHelperModules.ConvWithBNRelu(relu=False, bn=False),
            example_inputs,
            BackendAQuantizer(),
            node_occurrence,
            node_list,
        )

    def test_wo_annotate_conv_output_quantizer(self):
        # TODO: use OP_TO_ANNOTATOR
        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                act_qspec = QuantizationSpec(
                    dtype=torch.uint8,
                    quant_min=0,
                    quant_max=255,
                    qscheme=torch.per_tensor_affine,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=observer.default_observer,
                )
                weight_qspec = QuantizationSpec(
                    dtype=torch.int8,
                    quant_min=-128,
                    quant_max=127,
                    qscheme=torch.per_tensor_affine,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=observer.default_weight_observer,
                )
                bias_qspec = QuantizationSpec(
                    dtype=torch.float32,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=observer.PlaceholderObserver,
                )
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.conv2d.default
                    ):
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                                weight: weight_qspec,
                                bias: bias_qspec,
                            },
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        m = torch.nn.Conv2d(2, 2, 1)
        x = torch.rand(1, 2, 14, 14)
        example_inputs = (x,)
        m = self._quantize(m, BackendAQuantizer(), example_inputs)
        # Ensure the conv has no observer inserted at output
        node_occurrence = {
            # two for input of conv
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 1,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 2,
        }
        node_list = [
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.conv2d.default),
        ]
        self.checkGraphModuleNodes(
            m, expected_node_list=node_list, expected_node_occurrence=node_occurrence
        )

    def test_max_pool2d_quantizer(self):
        # TODO: use OP_TO_ANNOTATOR
        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                act_qspec = QuantizationSpec(
                    dtype=torch.uint8,
                    quant_min=0,
                    quant_max=255,
                    qscheme=torch.per_tensor_affine,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=observer.default_observer,
                )
                weight_qspec = QuantizationSpec(
                    dtype=torch.int8,
                    quant_min=-128,
                    quant_max=127,
                    qscheme=torch.per_tensor_affine,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=observer.default_weight_observer,
                )
                bias_qspec = QuantizationSpec(
                    dtype=torch.float32,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=observer.PlaceholderObserver,
                )
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.conv2d.default
                    ):
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                                weight: weight_qspec,
                                bias: bias_qspec,
                            },
                            _annotated=True,
                        )
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.max_pool2d.default
                    ):
                        maxpool_node = node
                        input_act = maxpool_node.args[0]
                        assert isinstance(input_act, Node)
                        maxpool_node.meta[
                            "quantization_annotation"
                        ] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                            },
                            output_qspec=SharedQuantizationSpec(
                                (input_act, maxpool_node)
                            ),
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        m = TestHelperModules.ConvMaxPool2d()
        x = torch.rand(1, 2, 14, 14)
        example_inputs = (x,)
        m = self._quantize(m, BackendAQuantizer(), example_inputs)
        node_occurrence = {
            # two for input of conv
            # one for input of maxpool
            # one for output of maxpool
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 3,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 4,
        }
        node_list = [
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.conv2d.default),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.max_pool2d.default),
        ]
        self.checkGraphModuleNodes(
            m, expected_node_list=node_list, expected_node_occurrence=node_occurrence
        )

    def test_derived_qspec(self):
        # TODO: use OP_TO_ANNOTATOR
        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.conv2d.default
                    ):
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_observer,
                        )
                        weight_qspec = QuantizationSpec(
                            dtype=torch.int8,
                            quant_min=-128,
                            quant_max=127,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_weight_observer,
                        )

                        def derive_qparams_fn(
                            obs_or_fqs: List[ObserverOrFakeQuantize],
                        ) -> Tuple[Tensor, Tensor]:
                            assert (
                                len(obs_or_fqs) == 2
                            ), f"Expecting two obs/fqs, one for activation and one for weight, got: {len(obs_or_fqs)}"
                            act_obs_or_fq = obs_or_fqs[0]
                            weight_obs_or_fq = obs_or_fqs[1]
                            act_scale, act_zp = act_obs_or_fq.calculate_qparams()
                            (
                                weight_scale,
                                weight_zp,
                            ) = weight_obs_or_fq.calculate_qparams()
                            return torch.tensor([act_scale * weight_scale]).to(
                                torch.float32
                            ), torch.tensor([0]).to(torch.int32)

                        bias_qspec = DerivedQuantizationSpec(
                            derived_from=[(input_act, node), (weight, node)],
                            derive_qparams_fn=derive_qparams_fn,
                            dtype=torch.int32,
                            quant_min=-(2**31),
                            quant_max=2**31 - 1,
                            qscheme=torch.per_tensor_symmetric,
                        )
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                                weight: weight_qspec,
                                bias: bias_qspec,
                            },
                            output_qspec=act_qspec,
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        m = TestHelperModules.ConvWithBNRelu(relu=False, bn=False).eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)

        m = self._quantize(m, BackendAQuantizer(), example_inputs)
        node_occurrence = {
            # input, weight, bias, output for the conv
            # note: quantize op for weight and bias are const propagated
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 2,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 4,
        }
        node_list = [
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.conv2d.default),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
        ]
        self.checkGraphModuleNodes(
            m, expected_node_list=node_list, expected_node_occurrence=node_occurrence
        )

    def test_derived_qspec_per_channel(self):
        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.conv2d.default
                    ):
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_observer,
                        )
                        weight_qspec = QuantizationSpec(
                            dtype=torch.int8,
                            quant_min=-128,
                            quant_max=127,
                            qscheme=torch.per_channel_affine,
                            is_dynamic=False,
                            ch_axis=0,
                            observer_or_fake_quant_ctr=observer.default_per_channel_weight_observer,
                        )

                        def derive_qparams_fn(
                            obs_or_fqs: List[ObserverOrFakeQuantize],
                        ) -> Tuple[Tensor, Tensor]:
                            assert (
                                len(obs_or_fqs) == 1
                            ), f"Expecting one weight obs/fq, got: {len(obs_or_fqs)}"
                            weight_obs_or_fq = obs_or_fqs[0]
                            (
                                weight_scale,
                                weight_zp,
                            ) = weight_obs_or_fq.calculate_qparams()
                            return weight_scale, torch.zeros_like(weight_scale)

                        bias_qspec = DerivedQuantizationSpec(
                            derived_from=[(weight, node)],
                            derive_qparams_fn=derive_qparams_fn,
                            dtype=torch.int32,
                            quant_min=-(2**31),
                            quant_max=2**31 - 1,
                            qscheme=torch.per_channel_symmetric,
                            ch_axis=0,
                        )
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                                weight: weight_qspec,
                                bias: bias_qspec,
                            },
                            output_qspec=act_qspec,
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        m = TestHelperModules.ConvWithBNRelu(relu=False, bn=False).eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)

        m = self._quantize(m, BackendAQuantizer(), example_inputs)

        node_occurrence = {
            # input, output for the conv
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 2,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 2,
            # weight and bias for conv
            # note: quantize op for weight and bias are const propagated
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_channel.default
            ): 0,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_channel.default
            ): 2,
        }
        node_list = [
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_channel.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_channel.default
            ),
            ns.call_function(torch.ops.aten.conv2d.default),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
        ]
        self.checkGraphModuleNodes(
            m, expected_node_list=node_list, expected_node_occurrence=node_occurrence
        )

    def test_fixed_qparams_qspec_ptq(self):
        self._test_fixed_qparams_qspec(is_qat=False)

    # TODO: refactor and move this to test_quantize_pt2_qat.py
    def test_fixed_qparams_qspec_qat(self):
        self._test_fixed_qparams_qspec(is_qat=True)

    def _test_fixed_qparams_qspec(self, is_qat: bool):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.sigmoid(x)

        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.sigmoid.default
                    ):
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        act_qspec = FixedQParamsQuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            scale=1.0 / 256.0,
                            zero_point=0,
                        )
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                            },
                            output_qspec=act_qspec,
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        m = M().eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)

        m = self._quantize(m, BackendAQuantizer(), example_inputs, is_qat)
        fixed_scale = 1.0 / 256.0
        fixed_zero_point = 0
        for n in m.graph.nodes:
            if n.op == "call_function":
                if (
                    n.target
                    == torch.ops.quantized_decomposed.quantize_per_tensor.default
                ):
                    scale_0 = n.args[1]
                    zero_point_0 = n.args[2]
                if (
                    n.target
                    == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                ):
                    scale_1 = n.args[1]
                    zero_point_1 = n.args[2]
        self.assertEqual(scale_0, fixed_scale)
        self.assertEqual(zero_point_0, fixed_zero_point)
        self.assertEqual(scale_1, fixed_scale)
        self.assertEqual(zero_point_1, fixed_zero_point)
        node_occurrence = {
            # two for input of the first conv, one for output for the first conv
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 2,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 2,
        }
        node_list = [
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.sigmoid.default),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
        ]
        self.checkGraphModuleNodes(
            m, expected_node_list=node_list, expected_node_occurrence=node_occurrence
        )

    def test_fixed_qparams_qspec_observer_dedup(self):
        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                act_qspec = FixedQParamsQuantizationSpec(
                    dtype=torch.uint8,
                    quant_min=0,
                    quant_max=255,
                    qscheme=torch.per_tensor_affine,
                    scale=1.0 / 256.0,
                    zero_point=0,
                )
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.sigmoid.default
                    ):
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                            },
                            output_qspec=act_qspec,
                            _annotated=True,
                        )
                    elif (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.add.Tensor
                    ):
                        input_act0 = node.args[0]
                        assert isinstance(input_act, Node)
                        input_act1 = node.args[1]
                        assert isinstance(input_act, Node)
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act0: act_qspec,
                                input_act1: act_qspec,
                            },
                            output_qspec=act_qspec,
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.sigmoid(x) + y

            def example_inputs(self):
                return (
                    torch.randn(1, 3, 5, 5),
                    torch.randn(1, 3, 5, 5),
                )

        m = M().eval()
        example_inputs = m.example_inputs()
        m = self._quantize(m, BackendAQuantizer(), example_inputs, is_qat=False)

        node_occurrence = {
            # two for input of the first conv, one for output for the first conv
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 4,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 4,
        }
        node_list = [
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.sigmoid.default),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.add.Tensor),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
        ]
        self.checkGraphModuleNodes(
            m, expected_node_list=node_list, expected_node_occurrence=node_occurrence
        )

    def test_shared_qspec(self):
        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.conv2d.default
                    ):
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_observer,
                        )
                        weight_qspec = QuantizationSpec(
                            dtype=torch.int8,
                            quant_min=-128,
                            quant_max=127,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_weight_observer,
                        )
                        bias_qspec = QuantizationSpec(
                            dtype=torch.float32,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.PlaceholderObserver,
                        )
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                                weight: weight_qspec,
                                bias: bias_qspec,
                            },
                            output_qspec=act_qspec,
                            _annotated=True,
                        )
                    elif node.target is torch.ops.aten.cat.default:
                        cat_node = node
                        input_nodes = cat_node.args[0]
                        first_input_node = input_nodes[0]
                        input_qspec_map = {}
                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_observer,
                        )
                        input_qspec_map[first_input_node] = act_qspec
                        share_qparams_with_input_act0_qspec = SharedQuantizationSpec(
                            (first_input_node, cat_node)
                        )
                        for input_node in input_nodes[1:]:
                            input_qspec_map[
                                input_node
                            ] = share_qparams_with_input_act0_qspec

                        cat_node.meta[
                            "quantization_annotation"
                        ] = QuantizationAnnotation(
                            input_qspec_map=input_qspec_map,
                            output_qspec=share_qparams_with_input_act0_qspec,
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        m = TestHelperModules.Conv2dWithCat().eval()
        example_inputs = (torch.randn(1, 3, 5, 5), torch.randn(1, 3, 5, 5))

        # program capture
        m = export_for_training(
            m,
            example_inputs,
        ).module()
        m = prepare_pt2e(m, BackendAQuantizer())
        # make sure the two observers for input are shared
        conv_output_obs = []
        for n in m.graph.nodes:
            if n.op == "call_function" and n.target == torch.ops.aten.conv2d.default:
                conv_output_obs.append(getattr(m, next(iter(n.users)).target))
            if n.op == "call_function" and n.target == torch.ops.aten.cat.default:
                inputs = n.args[0]
                input0 = inputs[0]
                input1 = inputs[1]
                assert input0.op == "call_module"
                assert input1.op == "call_module"
                obs_ins0 = getattr(m, input0.target)
                obs_ins1 = getattr(m, input1.target)
                assert obs_ins0 == obs_ins1
        assert (
            len(conv_output_obs) == 2
        ), "expecting two observer that follows conv2d ops"
        # checking that the output observers for the two convs are shared as well
        assert conv_output_obs[0] == conv_output_obs[1]

        m(*example_inputs)
        m = convert_pt2e(m)

        node_occurrence = {
            # two for input of the first conv, one for output for the first conv
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 5,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 7,
        }
        node_list = [
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.cat.default),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
        ]
        self.checkGraphModuleNodes(
            m, expected_node_list=node_list, expected_node_occurrence=node_occurrence
        )

    def _test_transitive_sharing_with_cat_helper(self, quantizer):
        m = TestHelperModules.Conv2dWithTwoCat().eval()
        example_inputs = (
            torch.randn(1, 3, 5, 5),
            torch.randn(1, 3, 5, 5),
            torch.randn(1, 6, 3, 3),
            torch.randn(1, 6, 3, 3),
        )

        # program capture
        m = export_for_training(
            m,
            example_inputs,
        ).module()
        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        # make sure the two input observers and output are shared
        conv_output_obs = []
        for n in m.graph.nodes:
            if n.op == "call_function" and n.target == torch.ops.aten.conv2d.default:
                conv_output_obs.append(getattr(m, next(iter(n.users)).target))
            if n.op == "call_function" and n.target == torch.ops.aten.cat.default:
                inputs = n.args[0]
                input0 = inputs[0]
                input1 = inputs[1]
                assert input0.op == "call_module"
                assert input1.op == "call_module"
                obs_ins0 = getattr(m, input0.target)
                obs_ins1 = getattr(m, input1.target)
                assert obs_ins0 == obs_ins1

                output_obs = next(iter(n.users))
                assert output_obs.op == "call_module"
                obs_ins2 = getattr(m, output_obs.target)
                assert obs_ins0 == obs_ins2, "input observer does not match output"

        assert (
            len(conv_output_obs) == 2
        ), "expecting two observer that follows conv2d ops"
        # checking that the output observers for the two convs are shared as well
        assert conv_output_obs[0] == conv_output_obs[1]

        m(*example_inputs)
        m = convert_pt2e(m)

        node_occurrence = {
            # two for input of the first conv, one for output for the first conv
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 7,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 9,
        }
        node_list = [
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.cat.default),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.cat.default),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
        ]
        self.checkGraphModuleNodes(
            m, expected_node_list=node_list, expected_node_occurrence=node_occurrence
        )

    def test_shared_qspec_transitivity(self):
        """This tests the transitivity of SharedQuantizationSpec, that is
        if A is shared with B, B is shared with C, then C should be shared with A as well

        x1 -> conv1 -> cat1 -----> cat2
        x2 -> conv2 -/            /
                       x3 -> add /
                       x4  /

        both cat has shared input and output, and because of cat and (cat1 -> cat2) is the same Tensor
        so there is an implicit sharing here, all tensors connect to cat1 and cat2 are in the same
        sharing group after transitive sharing
        """

        # TODO: refactor this to a common util
        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.conv2d.default
                    ):
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_observer,
                        )
                        weight_qspec = QuantizationSpec(
                            dtype=torch.int8,
                            quant_min=-128,
                            quant_max=127,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_weight_observer,
                        )
                        bias_qspec = QuantizationSpec(
                            dtype=torch.float32,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.PlaceholderObserver,
                        )
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                                weight: weight_qspec,
                                bias: bias_qspec,
                            },
                            output_qspec=act_qspec,
                            _annotated=True,
                        )
                    elif node.target is torch.ops.aten.cat.default:
                        cat_node = node
                        input_nodes = cat_node.args[0]
                        first_input_node = input_nodes[0]
                        input_qspec_map = {}
                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_observer,
                        )
                        input_qspec_map[first_input_node] = act_qspec
                        share_qparams_with_input_act0_qspec = SharedQuantizationSpec(
                            (first_input_node, cat_node)
                        )
                        for input_node in input_nodes[1:]:
                            input_qspec_map[
                                input_node
                            ] = share_qparams_with_input_act0_qspec

                        cat_node.meta[
                            "quantization_annotation"
                        ] = QuantizationAnnotation(
                            input_qspec_map=input_qspec_map,
                            output_qspec=share_qparams_with_input_act0_qspec,
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        self._test_transitive_sharing_with_cat_helper(BackendAQuantizer())

    def test_shared_qspec_transitivity_case_2(self):
        """This tests the transitivity of SharedQuantizationSpec, that is
        if A is shared with B, B is shared with C, then C should be shared with A as well

        x1 -> conv1 -> cat1 -----> cat2
        x2 -> conv2 -/            /
                       x3 -> add /
                       x4  /

        both cat has shared input and output, and because of cat and (cat1 -> cat2) is the same Tensor
        so there is an implicit sharing here, all tensors connect to cat1 and cat2 are in the same
        sharing group after transitive sharing

        the difference is that for this one, all edges and nodes are shared with the second input edge of cat
        instead of the first input edge of cat as in previous example
        """

        # TODO: refactor this to a common util
        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.conv2d.default
                    ):
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_observer,
                        )
                        weight_qspec = QuantizationSpec(
                            dtype=torch.int8,
                            quant_min=-128,
                            quant_max=127,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_weight_observer,
                        )
                        bias_qspec = QuantizationSpec(
                            dtype=torch.float32,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.PlaceholderObserver,
                        )
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                                weight: weight_qspec,
                                bias: bias_qspec,
                            },
                            output_qspec=act_qspec,
                            _annotated=True,
                        )
                    elif node.target is torch.ops.aten.cat.default:
                        cat_node = node
                        input_nodes = cat_node.args[0]
                        first_input_node = input_nodes[0]
                        second_input_node = input_nodes[1]
                        input_qspec_map = {}
                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_observer,
                        )
                        input_qspec_map[second_input_node] = act_qspec
                        share_qparams_with_input_act1_qspec = SharedQuantizationSpec(
                            (second_input_node, cat_node)
                        )
                        input_qspec_map[
                            first_input_node
                        ] = share_qparams_with_input_act1_qspec

                        cat_node.meta[
                            "quantization_annotation"
                        ] = QuantizationAnnotation(
                            input_qspec_map=input_qspec_map,
                            output_qspec=share_qparams_with_input_act1_qspec,
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        self._test_transitive_sharing_with_cat_helper(BackendAQuantizer())

    def test_allow_implicit_sharing(self):
        """This tests the allow_transitive_sharing flag of QuantizationAnnotation, that is
        if a node is configured with allow_implicit_sharing=False, we will not have implicit sharing
        for node and (node, consumer) even they refer to the same Tensor

        x1 -> add1 -----> add3
        x2 -/              /
               x3 -> add2 /
               x4 -/

        all add has shared input and output, and second input is using shared quantization spec pointing
        to first input, but we set allow_implicit_sharing to False for all add nodes so input and output of add1,
        add2 and add3 will each belong to one sharing group, so we'll have:

        x1 -> obs1 -> add1 -> obs1 -> obs3--> add3 -> obs3
        x2 -> obs1 -/                         /
               x3 -> obs2 -> add2 -> obs2 -> obs3
               x4 -> obs2 -/
        """

        # TODO: refactor this to a common util
        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for node in model.graph.nodes:
                    if node.target is torch.ops.aten.add.Tensor:
                        add_node = node
                        first_input_node = add_node.args[0]
                        second_input_node = add_node.args[1]
                        input_qspec_map = {}
                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_observer,
                        )
                        input_qspec_map[second_input_node] = act_qspec
                        share_qparams_with_input_act1_qspec = SharedQuantizationSpec(
                            (second_input_node, add_node)
                        )
                        input_qspec_map[
                            first_input_node
                        ] = share_qparams_with_input_act1_qspec

                        add_node.meta[
                            "quantization_annotation"
                        ] = QuantizationAnnotation(
                            input_qspec_map=input_qspec_map,
                            output_qspec=share_qparams_with_input_act1_qspec,
                            allow_implicit_sharing=False,
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        m = TestHelperModules.ThreeAdd().eval()
        example_inputs = (
            torch.randn(1, 3, 5, 5),
            torch.randn(1, 3, 5, 5),
            torch.randn(1, 3, 5, 5),
            torch.randn(1, 3, 5, 5),
        )

        # program capture
        m = export_for_training(
            m,
            example_inputs,
        ).module()
        quantizer = BackendAQuantizer()
        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        observers = []
        for n in m.graph.nodes:
            if n.target == torch.ops.aten.add.Tensor:
                input_obs1 = getattr(m, n.args[0].target)
                input_obs2 = getattr(m, n.args[1].target)
                output_obs = getattr(m, next(iter(n.users)).target)
                self.assertIs(input_obs1, input_obs2)
                self.assertIs(input_obs1, output_obs)
                observers.append(input_obs1)
        assert len(observers) == 3
        self.assertIsNot(observers[0], observers[1])
        self.assertIsNot(observers[0], observers[2])
        self.assertIsNot(observers[1], observers[2])

    @skipIfHpu
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    @parametrize("quant_dtype", (torch.int16, torch.float8_e5m2, torch.float8_e4m3fn))
    def test_quantization_dtype(self, dtype, quant_dtype):
        class DtypeActQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                info_fun = torch.iinfo if quant_dtype == torch.int16 else torch.finfo
                activate_qspec = QuantizationSpec(
                    dtype=quant_dtype,
                    quant_min=int(info_fun(quant_dtype).min),
                    quant_max=int(info_fun(quant_dtype).max),
                    qscheme=torch.per_tensor_affine,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=observer.default_observer,
                )
                int8_qspec = QuantizationSpec(
                    dtype=torch.int8,
                    quant_min=-128,
                    quant_max=127,
                    qscheme=torch.per_tensor_symmetric,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=observer.default_weight_observer,
                )
                quantization_config = QuantizationConfig(
                    input_activation=activate_qspec,
                    weight=int8_qspec,
                    bias=None,
                    output_activation=activate_qspec,
                )
                OP_TO_ANNOTATOR["conv"](model, quantization_config)

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        class M(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3, dtype=dtype)

            def forward(self, x):
                return self.conv(x)

        quantizer = DtypeActQuantizer()
        node_occurrence = {
            # one for input of the first conv, one for output for the first conv
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
        }
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        ]
        example_inputs = (torch.randn(1, 3, 3, 3, dtype=dtype),)
        m = self._test_quantizer(
            M(dtype).eval(),
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )

        def verify_quant_dequant_iotypes(m):
            for node in m.graph.nodes:
                if (
                    node.op == "call_function"
                    and node.target.__name__ == "dequantize_per_tensor.default"
                ):
                    # Check dequantize node
                    dequant_node = node
                    dequant_in_dtype = dequant_node.args[5]
                    dequant_out_dtype = torch.float32
                    if "out_dtype" in dequant_node.kwargs:
                        dequant_out_dtype = dequant_node.kwargs["out_dtype"]

                    # Check preceding quantize node
                    # Depending on fold_quantize flag, quantize node may be absent
                    quant_node = node.args[0]
                    if (
                        quant_node.op == "call_function"
                        and quant_node.target.__name__ == "quantize_per_tensor.default"
                    ):
                        quant_in_dtype = torch.float32
                        if "val" in quant_node.args[0].meta:
                            quant_in_dtype = quant_node.args[0].meta["val"].dtype
                        quant_out_dtype = quant_node.args[5]
                        assert (
                            quant_in_dtype == dequant_out_dtype
                            and quant_out_dtype == dequant_in_dtype
                        ), "quant dequant io dtype check failed!"

        verify_quant_dequant_iotypes(m)

    def test_input_edge_sanity_check(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x + 6

        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.add.Tensor
                    ):
                        input_act1 = node.args[0]
                        # this is a constant, so not valid for annotation
                        input_act2 = node.args[1]
                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_observer,
                        )
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act1: act_qspec,
                                # this is supposed to error out
                                input_act2: act_qspec,
                            },
                            output_qspec=act_qspec,
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        m = M().eval()
        example_inputs = torch.randn(1, 2, 3, 3)
        m = export_for_training(m, (example_inputs,)).module()
        with self.assertRaises(Exception):
            m = prepare_pt2e(m, BackendAQuantizer())

    def test_fold_quantize(self):
        """Test to make sure the quantized model gets quantized weight (quantize_per_tensor op is folded)"""
        m = self._get_pt2e_quantized_linear()
        node_occurrence = {
            # quantize op for weight node is folded
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 2,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 3,
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_fold_quantize_per_channel(self):
        """Test to make sure the quantized model gets quantized weight (quantize_per_channel op is folded)"""
        m = self._get_pt2e_quantized_linear(is_per_channel=True)
        node_occurrence = {
            # quantize op for weight node is folded
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 2,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_channel.default
            ): 1,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 2,
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_dont_fold_other_constant(self):
        """Make sure the constant propagation does not apply to things unrelated to
        quantization
        """

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)
                self.dont_fold_me = torch.nn.Parameter(torch.randn(2, 2))

            def forward(self, x):
                t = self.dont_fold_me.t()
                return self.linear(x) + t

        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=False)
        # only quantize linear, so add is not quantized and the constant Tensor
        # should not be folded
        quantizer.set_module_type(torch.nn.Linear, operator_config)
        example_inputs = (torch.randn(2, 2),)
        m = M().eval()
        m = self._quantize(m, quantizer, example_inputs)
        node_occurrence = {
            # quantize op for weight node is folded
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 2,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 3,
            # transpose op not folded
            ns.call_function(torch.ops.aten.t.default): 1,
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_fold_all_ops_before_quantize(self):
        """Test folding all ops that's before quantized operator:
        Before:
            get_attr(weight) -> transpose -> quantize -> dequantize
        After:
            get_attr(folded_weight) -> dequantize
        """

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.randn(2, 2)

            def forward(self, x):
                t = self.weight.t()
                return torch.nn.functional.linear(x, t)

        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=False)
        quantizer.set_global(operator_config)
        example_inputs = (torch.randn(2, 2),)
        m = M().eval()
        m = self._quantize(m, quantizer, example_inputs)
        node_occurrence = {
            # quantize op for weight node is folded
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 2,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 3,
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_constant_prop_preserve_metadata(self):
        """Test to make sure the get_attr node for const propagated weight Tensor gets the correct
        metadata (from original get_attr node from weight)
        """

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config()
        quantizer.set_global(operator_config)
        example_inputs = (torch.randn(2, 2),)
        m = M().eval()
        m = export_for_training(
            m,
            example_inputs,
        ).module()
        weight_meta = None
        for n in m.graph.nodes:
            if (
                n.op == "get_attr"
                and next(iter(n.users)).target == torch.ops.aten.linear.default
            ):
                weight_meta = n.meta
                break
        assert weight_meta is not None, "Expect to find metadata for weight node"

        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        m = convert_pt2e(m)

        for n in m.graph.nodes:
            if n.op == "get_attr" and "frozen_param" in n.target:
                for key in n.meta:
                    self.assertEqual(n.meta[key], weight_meta[key])

    def test_save_load(self):
        """Test save/load a quantized model"""
        m = self._get_pt2e_quantized_linear()
        example_inputs = (torch.randn(2, 2),)
        ref_res = m(*example_inputs)

        with TemporaryFileName() as fname:
            # serialization
            quantized_ep = torch.export.export(m, example_inputs)
            torch.export.save(quantized_ep, fname)
            # deserialization
            loaded_ep = torch.export.load(fname)
            loaded_quantized_model = loaded_ep.module()
            res = loaded_quantized_model(*example_inputs)
            self.assertEqual(ref_res, res)

    def test_composable_quantizer_throw(self):
        class BadQuantizer(Quantizer):
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for n in gm.graph.nodes:
                    n.meta["quantization_annotation"] = None

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(quantization_config)
        bad_quantizer = BadQuantizer()
        composable_quantizer = ComposableQuantizer([quantizer, bad_quantizer])
        m_eager = TestHelperModules.ConvLinearWPermute().eval()
        example_inputs = (torch.randn(2, 3, 4, 4),)
        self.assertRaises(
            RuntimeError,
            lambda: self._test_quantizer(
                m_eager, example_inputs, composable_quantizer, {}
            ),
        )

    def test_transform_for_annotation(self):
        class TestQuantizer(Quantizer):
            def transform_for_annotation(
                self, model: torch.fx.GraphModule
            ) -> torch.fx.GraphModule:
                # Make a copy of the graph to ensure that we are using the
                # return value of this function.
                graph = torch.fx.Graph()
                graph.graph_copy(model.graph, {})
                for n in graph.nodes:
                    if n.target == torch.ops.aten.add.Tensor:
                        n.target = torch.ops.aten.mul.Tensor
                model = torch.fx.GraphModule(model, graph)
                return model

            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                return model

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        class M(torch.nn.Module):
            def forward(self, x):
                return x + 3

        m = M().eval()
        quantizer = TestQuantizer()
        example_inputs = (torch.randn(1, 2, 3, 3),)
        m = export_for_training(m, example_inputs).module()
        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        node_occurrence = {
            ns.call_function(torch.ops.aten.add.Tensor): 0,
            ns.call_function(torch.ops.aten.mul.Tensor): 1,
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_composable_quantizer_transform_for_annotation(self):
        class TestQuantizer1(Quantizer):
            def transform_for_annotation(
                self, model: torch.fx.GraphModule
            ) -> torch.fx.GraphModule:
                for n in model.graph.nodes:
                    if n.target == torch.ops.aten.add.Tensor:
                        n.target = torch.ops.aten.mul.Tensor
                return model

            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                return model

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        class TestQuantizer2(Quantizer):
            def transform_for_annotation(
                self, model: torch.fx.GraphModule
            ) -> torch.fx.GraphModule:
                for n in model.graph.nodes:
                    if n.target == torch.ops.aten.sub.Tensor:
                        n.target = torch.ops.aten.div.Tensor
                return model

            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                return model

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        class M(torch.nn.Module):
            def forward(self, x, y, z):
                return x + y - z

        m = M().eval()
        quantizer = ComposableQuantizer([TestQuantizer1(), TestQuantizer2()])
        example_inputs = (
            torch.randn(1, 2, 3, 3),
            torch.randn(1, 2, 3, 3),
            torch.randn(1, 2, 3, 3),
        )
        m = export_for_training(m, example_inputs).module()
        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        node_occurrence = {
            ns.call_function(torch.ops.aten.add.Tensor): 0,
            ns.call_function(torch.ops.aten.sub.Tensor): 0,
            ns.call_function(torch.ops.aten.mul.Tensor): 1,
            ns.call_function(torch.ops.aten.div.Tensor): 1,
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_embedding_quantizer(self):
        m_eager = TestHelperModules.EmbeddingModule().eval()
        indices = torch.tensor(
            [
                9,
                6,
                5,
                7,
                8,
                8,
                9,
                2,
                8,
                6,
                6,
                9,
                1,
                6,
                8,
                8,
                3,
                2,
                3,
                6,
                3,
                6,
                5,
                7,
                0,
                8,
                4,
                6,
                5,
                8,
                2,
                3,
            ]
        )
        example_inputs = (indices,)

        quantizer = EmbeddingQuantizer()
        node_occurrence = {
            # note: quantize op for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
            torch.ops.aten.embedding.default,
        ]
        # Compare against short term workflow
        # cannot compare against fx quant because of the numerical differences coming
        # from quantize and dequantize ops
        qconfig = default_per_channel_symmetric_qnnpack_qconfig
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        qconfig_mapping = qconfig_mapping.set_object_type(
            torch.nn.Embedding, float_qparams_weight_only_qconfig
        )
        self._test_quantizer(
            m_eager,
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
            True,
            qconfig_mapping,
        )

    def test_composable_quantizer_linear_conv(self):
        dynamic_quantizer = XNNPACKQuantizer()
        quantization_config_dynamic = get_symmetric_quantization_config(
            is_per_channel=False, is_dynamic=True
        )
        dynamic_quantizer.set_global(quantization_config_dynamic)
        static_quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        static_quantizer.set_global(quantization_config)
        # Note that dynamic quantization must be applied first here.
        # this is because static quantizer also quantizes linear with static qspec
        # and if we apply static_quantizer first then dynamic_quantizer cannot be applied
        composable_quantizer = ComposableQuantizer(
            [dynamic_quantizer, static_quantizer]
        )
        m_eager = TestHelperModules.ConvLinearWPermute().eval()

        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 1,
            # note: quantize op for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 4,
            # note: quantize op for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        act_affine_quant_obs = observer.PlaceholderObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine,
            quant_min=-128,
            quant_max=127,
            eps=2**-12,
            is_dynamic=True,
        )
        dynamic_qconfig = QConfig(
            activation=act_affine_quant_obs,
            weight=weight_observer_range_neg_127_to_127,
        )
        # Test with 2d inputs
        example_inputs = (torch.randn(2, 3, 4, 4),)
        qconfig = default_per_channel_symmetric_qnnpack_qconfig
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        qconfig_mapping.set_object_type(torch.nn.Linear, dynamic_qconfig)
        # Had to turn off check against fx because fx quant workflow does not seem
        # to propagate observers for permute node for this model.
        # Suprisingly it does propagate it for EmbeddingConvLinearModule
        # TODO: Figure out the right behavior for propagation
        self._test_quantizer(
            m_eager,
            example_inputs,
            composable_quantizer,
            node_occurrence,
            [],
            False,
            qconfig_mapping,
        )

    def test_embedding_conv_linear_quantization(self):
        m_eager = TestHelperModules.EmbeddingConvLinearModule().eval()
        indices = torch.tensor(
            [
                9,
                6,
                5,
                7,
                8,
                8,
                9,
                2,
                8,
                6,
                6,
                9,
                1,
                6,
                8,
                8,
                3,
                2,
                3,
                6,
                3,
                6,
                5,
                7,
                0,
                8,
                4,
                6,
                5,
                8,
                2,
                3,
            ]
        )
        indices = torch.unsqueeze(indices, 0)
        example_inputs = (indices,)

        embedding_quantizer = EmbeddingQuantizer()
        dynamic_quantizer = XNNPACKQuantizer()
        quantization_config_dynamic = get_symmetric_quantization_config(
            is_per_channel=True, is_dynamic=True
        )
        dynamic_quantizer.set_global(quantization_config_dynamic)
        static_quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        static_quantizer.set_global(quantization_config)
        composed_quantizer = ComposableQuantizer(
            [embedding_quantizer, dynamic_quantizer, static_quantizer]
        )

        act_affine_quant_obs = observer.PlaceholderObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine,
            quant_min=-128,
            quant_max=127,
            eps=2**-12,
            is_dynamic=True,
        )
        dynamic_qconfig = QConfig(
            activation=act_affine_quant_obs,
            weight=per_channel_weight_observer_range_neg_127_to_127,
        )
        qconfig = default_per_channel_symmetric_qnnpack_qconfig
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        qconfig_mapping.set_object_type(torch.nn.Linear, dynamic_qconfig)
        qconfig_mapping = qconfig_mapping.set_object_type(
            torch.nn.Embedding, float_qparams_weight_only_qconfig
        )

        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 4,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 4,
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 1,
            # note: quantize op for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 3,
        }
        self._test_quantizer(
            m_eager,
            example_inputs,
            composed_quantizer,
            node_occurrence,
            [],
            True,
            qconfig_mapping,
        )

    def _get_node(self, m: torch.fx.GraphModule, target: torch._ops.OpOverload):
        """
        Return the first node matching the specified target, throwing an exception
        if no such batch norm node is found.
        """
        for n in m.graph.nodes:
            if n.target == target:
                return n
        raise ValueError("Did not find node with target ", target)

    def _test_move_exported_model_dropout(self, inplace: bool):
        """
        Test switching dropout behavior between train and eval modes using
        `move_exported_model_to_eval` and `move_exported_model_to_train` APIs.
        """

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dropout = torch.nn.Dropout(0.5, inplace=inplace)

            def forward(self, x):
                return self.dropout(x)

        example_inputs = (torch.randn(1),)
        m = M().train()
        m = export_for_training(m, example_inputs).module()
        if inplace:
            target = torch.ops.aten.dropout_.default
        else:
            target = torch.ops.aten.dropout.default

        # Assert that dropout op exists and is in train mode
        dropout_node = self._get_node(m, target)
        self.assertTrue(dropout_node is not None)
        self.assertTrue(dropout_node.args[2])

        # Move to eval
        torch.ao.quantization.move_exported_model_to_eval(m)

        # Assert that dropout op is now in eval mode
        dropout_node = self._get_node(m, target)
        self.assertTrue(dropout_node is not None)
        self.assertTrue(not dropout_node.args[2])

        # Move back to train
        torch.ao.quantization.move_exported_model_to_train(m)

        # Assert that dropout op is now in train mode again
        dropout_node = self._get_node(m, target)
        self.assertTrue(dropout_node is not None)
        self.assertTrue(dropout_node.args[2])

    def test_move_exported_model_dropout(self):
        self._test_move_exported_model_dropout(inplace=False)

    def test_move_exported_model_dropout_inplace(self):
        self._test_move_exported_model_dropout(inplace=True)

    def _get_bn_train_eval_ops(self):
        return (
            torch.ops.aten.batch_norm.default,
            torch.ops.aten.batch_norm.default,
        )

    @parametrize(
        "device",
        ["cpu"] + (["cuda"] if TEST_CUDA else []) + (["hpu"] if TEST_HPU else []),
    )
    def test_move_exported_model_bn(self, device):
        """
        Test switching batch_norm behavior between train and eval modes using
        `move_exported_model_to_eval` and `move_exported_model_to_train` APIs.
        """

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                return self.bn(x)

        if TEST_CUDA or TEST_HPU:
            m = M().train().to(device)
            example_inputs = (torch.randn((1, 3, 3, 3), device=device),)

        else:
            m = M().train()
            example_inputs = (torch.randn(1, 3, 3, 3),)
        bn_train_op, bn_eval_op = self._get_bn_train_eval_ops()
        m = export_for_training(m, example_inputs).module()

        # Assert that batch norm op exists and is in train mode
        bn_node = self._get_node(m, bn_train_op)
        self.assertTrue(bn_node is not None)
        self.assertTrue(bn_node.args[5])

        # Move to eval
        torch.ao.quantization.move_exported_model_to_eval(m)

        # Assert that batch norm op is now in eval mode
        bn_node = self._get_node(m, bn_eval_op)
        self.assertTrue(bn_node is not None)

        # Move to train
        torch.ao.quantization.move_exported_model_to_train(m)

        # Assert that batch norm op is now in train mode again
        bn_node = self._get_node(m, bn_train_op)
        self.assertTrue(bn_node is not None)
        self.assertTrue(bn_node.args[5])

    def test_disallow_eval_train(self):
        m = TestHelperModules.ConvWithBNRelu(relu=True)
        example_inputs = (torch.rand(3, 3, 5, 5),)

        # Before export: this is OK
        m.eval()
        m.train()

        # After export: this is not OK
        m = export_for_training(m, example_inputs).module()
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()

        # After prepare: still not OK
        quantizer = XNNPACKQuantizer()
        m = prepare_qat_pt2e(m, quantizer)
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()

        # After convert: still not OK
        m = convert_pt2e(m)
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()

    @skipIfHpu
    def test_allow_exported_model_train_eval(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(3)
                self.dropout = torch.nn.Dropout(0.5)

            def forward(self, x):
                x = self.bn(x)
                x = self.dropout(x)
                return x

        if TEST_CUDA:
            m = M().train().cuda()
            example_inputs = (torch.randn(1, 3, 3, 3).cuda(),)
        else:
            m = M().train()
            example_inputs = (torch.randn(1, 3, 3, 3),)
        bn_train_op, bn_eval_op = self._get_bn_train_eval_ops()
        m = export_for_training(m, example_inputs).module()

        def _assert_ops_are_correct(m: torch.fx.GraphModule, train: bool):
            targets = [n.target for n in m.graph.nodes]
            bn_op = bn_train_op if train else bn_eval_op
            bn_node = self._get_node(m, bn_op)
            self.assertTrue(bn_node is not None)
            if TEST_CUDA:
                self.assertEqual(bn_node.args[5], train)
            dropout_node = self._get_node(m, torch.ops.aten.dropout.default)
            self.assertEqual(dropout_node.args[2], train)

        # Before wrapping: this is not OK
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()

        # After wrapping: does not error and swaps the ops accordingly
        torch.ao.quantization.allow_exported_model_train_eval(m)
        m.eval()
        _assert_ops_are_correct(m, train=False)
        m.train()
        _assert_ops_are_correct(m, train=True)

        # After prepare but before wrapping: this is not OK
        quantizer = XNNPACKQuantizer()
        m = prepare_qat_pt2e(m, quantizer)
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()

        # After prepare and after wrapping: does not error and swaps the ops accordingly
        torch.ao.quantization.allow_exported_model_train_eval(m)
        m.eval()
        _assert_ops_are_correct(m, train=False)
        m.train()
        _assert_ops_are_correct(m, train=True)

        # After convert but before wrapping: this is not OK
        m = convert_pt2e(m, fold_quantize=True)
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()

        # After convert and after wrapping: does not error and swaps the ops accordingly
        torch.ao.quantization.allow_exported_model_train_eval(m)
        m.eval()
        _assert_ops_are_correct(m, train=False)
        m.train()
        _assert_ops_are_correct(m, train=True)

    def test_allow_exported_model_train_eval_idempotent(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.bn(x)
                return x

        m = M().train()
        example_inputs = (torch.randn(1, 3, 3, 3),)
        m = export_for_training(m, example_inputs).module()
        torch.ao.quantization.allow_exported_model_train_eval(m)

        # Mock m.recompile() to count how many times it's been called
        m._recompile_count = 0

        def _fake_recompile():
            m._recompile_count += 1

        m.recompile = _fake_recompile

        # First train after export should always recompile
        m.train()
        self.assertNotEqual(m._recompile_count, 0)
        count1 = m._recompile_count

        # Train -> train should not recompile
        m.train()
        self.assertEqual(m._recompile_count, count1)

        # Train -> eval should recompile
        m.eval()
        self.assertNotEqual(m._recompile_count, count1)
        count2 = m._recompile_count

        # Eval -> eval should not recompile
        m.eval()
        self.assertEqual(m._recompile_count, count2)

    def test_model_is_exported(self):
        m = TestHelperModules.ConvWithBNRelu(relu=True)
        example_inputs = (torch.rand(3, 3, 5, 5),)
        exported_gm = export_for_training(m, example_inputs).module()
        fx_traced_gm = torch.fx.symbolic_trace(m, example_inputs)
        self.assertTrue(
            torch.ao.quantization.pt2e.export_utils.model_is_exported(exported_gm)
        )
        self.assertFalse(
            torch.ao.quantization.pt2e.export_utils.model_is_exported(fx_traced_gm)
        )
        self.assertFalse(torch.ao.quantization.pt2e.export_utils.model_is_exported(m))

    def test_reentrant(self):
        """Test we can safely call quantization apis multiple times"""
        m = TestHelperModules.ConvBnReLU2dAndLinearReLU()
        example_inputs = (torch.randn(3, 3, 10, 10),)

        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=True, is_qat=True)
        )
        m.conv_bn_relu = export_for_training(m.conv_bn_relu, example_inputs).module()
        m.conv_bn_relu = prepare_qat_pt2e(m.conv_bn_relu, quantizer)
        m(*example_inputs)
        m.conv_bn_relu = convert_pt2e(m.conv_bn_relu)

        quantizer = XNNPACKQuantizer().set_module_type(
            torch.nn.Linear, get_symmetric_quantization_config(is_per_channel=False)
        )
        m = export_for_training(m, example_inputs).module()
        m = prepare_pt2e(m, quantizer)
        m = convert_pt2e(m)

        node_occurrence = {
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 4,
            # one for weight
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 5,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_channel.default
            ): 1,
        }
        node_list = [
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.conv2d.default),
            ns.call_function(torch.ops.aten.relu.default),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.linear.default),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
        ]
        self.checkGraphModuleNodes(
            m, expected_node_occurrence=node_occurrence, expected_node_list=node_list
        )

    def test_groupwise_per_channel_quant(self):
        m = TestHelperModules.GroupwiseConv2d()
        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(operator_config)
        example_inputs = m.example_inputs()
        m = self._quantize(m, quantizer, example_inputs)
        # make sure it runs
        m(*example_inputs)

    def test_observer_callback(self):
        from torch.library import impl, Library

        test_lib = Library("test_int4", "DEF")  # noqa: TOR901
        test_lib.define(
            "quantize_per_tensor_int4(Tensor input, float scale, int zero_point) -> Tensor"
        )

        @impl(test_lib, "quantize_per_tensor_int4", "CompositeExplicitAutograd")
        def quantize_per_tensor_int4(
            input: torch.Tensor,
            scale: float,
            zero_point: int,
        ) -> torch.Tensor:
            inv_scale = 1.0 / scale
            return (
                torch.clamp(torch.round(input * inv_scale) + zero_point, 0, 15)
                .to(torch.uint8)
                .view(torch.bits8)
            )

        test_lib.define(
            "dequantize_per_tensor_int4(Tensor input, float scale, int zero_point) -> Tensor"
        )

        @impl(test_lib, "dequantize_per_tensor_int4", "CompositeExplicitAutograd")
        def dequantize_per_tensor_int4(
            input: torch.Tensor,
            scale: float,
            zero_point: int,
        ) -> torch.Tensor:
            return (input.view(torch.uint8).to(torch.float32) - zero_point) * scale

        from torch.ao.quantization.observer import ObserverBase

        class Int4Observer(ObserverBase):
            def __init__(self, *args, **kwargs):
                # just faking a dtype here
                super().__init__(dtype=torch.int8)

            def forward(self, x):
                return x

            def calculate_qparams(self, **kwargs):
                pass

            def convert(self, model: torch.fx.GraphModule, observer_node: Node):
                with model.graph.inserting_before(observer_node):
                    q_node = model.graph.call_function(
                        torch.ops.test_int4.quantize_per_tensor_int4,
                        (observer_node.args[0], 1.0, 0),
                        {},
                    )
                    dq_node = model.graph.call_function(
                        torch.ops.test_int4.dequantize_per_tensor_int4,
                        (q_node, 1.0, 0),
                        {},
                    )
                    observer_node.replace_all_uses_with(dq_node)
                    model.graph.erase_node(observer_node)

        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.add.Tensor
                    ):
                        input_act0 = node.args[0]
                        assert isinstance(input_act0, Node)
                        input_act1 = node.args[1]
                        assert isinstance(input_act1, Node)

                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=Int4Observer,
                        )
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act0: act_qspec,
                                input_act1: act_qspec,
                            },
                            output_qspec=act_qspec,
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        class M(torch.nn.Module):
            def forward(self, x1, x2):
                return x1 + x2

        example_inputs = (
            torch.randn(1, 3, 5, 5),
            torch.randn(1, 3, 5, 5),
        )
        node_occurrence = {
            # two for input of the first conv, one for output for the first conv
            torch.ops.test_int4.quantize_per_tensor_int4: 3,
            torch.ops.test_int4.dequantize_per_tensor_int4: 3,
        }
        node_list = [
            torch.ops.test_int4.dequantize_per_tensor_int4,
            torch.ops.test_int4.dequantize_per_tensor_int4,
            torch.ops.aten.add.Tensor,
            torch.ops.test_int4.quantize_per_tensor_int4,
        ]
        self._test_quantizer(
            M().eval(),
            example_inputs,
            BackendAQuantizer(),
            node_occurrence,
            node_list,
        )

    def test_speed(self):
        import time

        def dynamic_quantize_pt2e(model, example_inputs):
            torch._dynamo.reset()
            model = export_for_training(model, example_inputs).module()
            # Per channel quantization for weight
            # Dynamic quantization for activation
            # Please read a detail: https://fburl.com/code/30zds51q
            embedding_quantizer = EmbeddingQuantizer()
            dynamic_quantizer = XNNPACKQuantizer()
            operator_config_dynamic = get_symmetric_quantization_config(
                is_per_channel=True, is_dynamic=True
            )
            dynamic_quantizer.set_global(operator_config_dynamic)
            composed_quantizer = ComposableQuantizer(
                [embedding_quantizer, dynamic_quantizer]
            )
            prev = time.time()
            model = prepare_qat_pt2e(model, composed_quantizer)
            cur = time.time()
            # print("prepare time:", cur - prev)
            # Without Calibraiton, scale/zero value will have an initialized value of 1.0
            # Per channel quantization needs a proper scale/zero shape/value to work properly.
            # So we need to run calibration before converting to quantized model.
            model(*example_inputs)
            prev = time.time()
            model = convert_pt2e(model)
            cur = time.time()
            # uncomment to see the time
            # print("convert time:", cur - prev)
            return model

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x)

        m = M().eval()
        example_inputs = (torch.randn(5, 5),)
        _ = dynamic_quantize_pt2e(m, example_inputs)

    def test_conv_transpose_bn_relu(self):
        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                int8_qspec = QuantizationSpec(
                    dtype=torch.int8,
                    quant_min=-128,
                    quant_max=127,
                    qscheme=torch.per_tensor_symmetric,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=observer.default_weight_observer,
                )
                quantization_config = QuantizationConfig(
                    input_activation=int8_qspec,
                    weight=int8_qspec,
                    bias=None,
                    output_activation=int8_qspec,
                )
                # conv_transpose + bn is fused automatically in PTQ (not configurable)
                # so we just need to annotate conv_transpose + relu for conv_transpose + bn + relu
                # pattern
                OP_TO_ANNOTATOR["conv_transpose_relu"](model, quantization_config)

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        example_inputs = (torch.randn(1, 3, 5, 5),)
        node_occurrence = {
            # two for input of the first conv, one for output for the first conv
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
        }
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv_transpose2d.input,
            torch.ops.aten.relu.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        ]
        self._test_quantizer(
            TestHelperModules.ConvTWithBNRelu(relu=True, bn=True),
            example_inputs,
            BackendAQuantizer(),
            node_occurrence,
            node_list,
        )

    def test_multi_users_without_output_observer(self):
        """
        Test the case in which a node is used by multiple users,
        and had its output observer removed.
        """

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                x = self.conv(x)
                return x, x + 1

        example_inputs = (torch.randn(1, 3, 5, 5),)
        m = M()
        m = export_for_training(m, example_inputs).module()
        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(),
        )
        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)

        # Remove output observer
        observer_to_remove = None
        for n in m.graph.nodes:
            if n.op == "output":
                observer_to_remove = n.args[0][0]
                assert observer_to_remove.op == "call_module"
                assert observer_to_remove.target.startswith("activation_post_process_")
                break
        assert observer_to_remove is not None
        observer_to_remove.replace_all_uses_with(observer_to_remove.args[0])
        m.graph.erase_node(observer_to_remove)
        m.recompile()

        # Convert should succeed
        m = convert_pt2e(m)
        m(*example_inputs)

    def test_prepare_obs_or_fq_callback(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                x = torch.nn.functional.max_pool2d(x, 2, 2)
                x = torch.nn.functional.pixel_shuffle(x, 2)
                return x.permute(0, 2, 3, 1)

        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                act_qspec = QuantizationSpec(
                    dtype=torch.uint8,
                    quant_min=0,
                    quant_max=255,
                    qscheme=torch.per_tensor_affine,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=observer.default_observer,
                )
                for node in model.graph.nodes:
                    if node.op == "call_function" and node.target in (
                        torch.ops.aten.max_pool2d.default,
                        torch.ops.aten.permute.default,
                        torch.ops.aten.pixel_shuffle.default,
                    ):
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                node.args[0]: act_qspec,
                            },
                            output_qspec=SharedQuantizationSpec((node.args[0], node)),
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

            def prepare_obs_or_fq_callback(
                self,
                model: torch.fx.GraphModule,
                edge_or_node_to_obs_or_fq: Dict[EdgeOrNode, ObserverOrFakeQuantize],
            ) -> None:
                # hard code output quant by updating entire sharing group
                output_node = next(n for n in model.graph.nodes if n.op == "output")
                output_value = output_node.args[0][0]
                old_observer = edge_or_node_to_obs_or_fq[output_value]
                sharing_group = [
                    k for k, v in edge_or_node_to_obs_or_fq.items() if v is old_observer
                ]
                new_observer = observer.FixedQParamsObserver(
                    scale=0.125,
                    zero_point=42,
                    dtype=torch.uint8,
                    quant_min=0,
                    quant_max=255,
                    qscheme=torch.per_tensor_affine,
                )
                for x in sharing_group:
                    edge_or_node_to_obs_or_fq[x] = new_observer

        example_inputs = (torch.rand(1, 32, 16, 16),)
        gm = export_for_training(Model().eval(), example_inputs).module()
        gm = prepare_pt2e(gm, BackendAQuantizer())
        gm = convert_pt2e(gm)
        for n in gm.graph.nodes:
            if n.op == "call_function" and n.target in (
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            ):
                # Entire graph share the same qspec which was overriden by FixedQParamsObserver
                self.assertEqual(n.args[1], 0.125)
                self.assertEqual(n.args[2], 42)

    def test_preserve_nn_module_stack(self):
        """Test we can preserve nn_module_stack on replaced pattern's nodes"""
        m = TestHelperModules.ConvBnReLU2dAndLinearReLU()
        example_inputs = (torch.randn(3, 3, 10, 10),)

        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=True, is_qat=True)
        )

        def check_nn_module(node):
            self.assertTrue("nn_module_stack" in node.meta)
            self.assertTrue(
                "ConvWithBNRelu" in node.meta["nn_module_stack"]["L__self__"][1]
            )

        m.conv_bn_relu = export_for_training(m.conv_bn_relu, example_inputs).module()
        for node in m.conv_bn_relu.graph.nodes:
            if node.op not in ["placeholder", "output", "get_attr"]:
                check_nn_module(node)
        m.conv_bn_relu = prepare_qat_pt2e(m.conv_bn_relu, quantizer)
        for node in m.conv_bn_relu.graph.nodes:
            if node.name == "mul":
                check_nn_module(node)


@skipIfNoQNNPACK
class TestQuantizePT2EAffineQuantization(PT2EQuantizationTestCase):
    def test_channel_group_quantization(self):
        from torch.ao.quantization.observer import MappingType, PerGroup, PerToken
        from torch.ao.quantization.pt2e._affine_quantization import (
            AffineQuantizedMinMaxObserver,
        )

        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.linear.default
                    ):
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)

                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=None,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=AffineQuantizedMinMaxObserver.with_args(
                                # TODO: maybe align the arg name here
                                target_dtype=torch.uint8,
                                mapping_type=MappingType.SYMMETRIC,
                                granularity=PerToken(),
                            ),
                        )

                        weight_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=None,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=AffineQuantizedMinMaxObserver.with_args(
                                target_dtype=torch.uint8,
                                mapping_type=MappingType.SYMMETRIC,
                                granularity=PerGroup(group_size=128),
                            ),
                        )
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                                weight: weight_qspec,
                            },
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(128, 20)

            def forward(self, x):
                return self.linear(x)

        node_occurrence = {
            torch.ops.quant.quantize_affine: 2,
            torch.ops.quant.dequantize_affine: 2,
        }
        node_list = [
            torch.ops.quant.quantize_affine,
            torch.ops.quant.dequantize_affine,
            torch.ops.quant.quantize_affine,
            torch.ops.quant.dequantize_affine,
        ]
        example_inputs = (torch.randn(5, 128),)
        self._test_quantizer(
            M().eval(),
            example_inputs,
            BackendAQuantizer(),
            node_occurrence,
            node_list,
            is_debug_mode=True,
        )


instantiate_parametrized_tests(TestQuantizePT2E)
