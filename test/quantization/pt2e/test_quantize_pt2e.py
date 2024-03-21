# Owner(s): ["oncall: quantization"]
from typing import List, Tuple
import sys
import unittest

import torch
from torch._export import (
    capture_pre_autograd_graph,
)
from torch import Tensor
from torch.ao.quantization import (
    observer,
    ObserverOrFakeQuantize,
    QConfigMapping,
)
from torch.ao.quantization.quantizer import (
    DerivedQuantizationSpec,
    FixedQParamsQuantizationSpec,
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
    SharedQuantizationSpec,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    OP_TO_ANNOTATOR,
    QuantizationConfig,
)
from torch.ao.quantization.quantizer.composable_quantizer import (  # noqa: F811
    ComposableQuantizer,
)
from torch.ao.quantization.quantizer.embedding_quantizer import (  # noqa: F811
    EmbeddingQuantizer,
)

from torch.ao.quantization.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)

from torch.ao.quantization.qconfig import (
    default_per_channel_symmetric_qnnpack_qconfig,
    float_qparams_weight_only_qconfig,
    per_channel_weight_observer_range_neg_127_to_127,
    QConfig,
    weight_observer_range_neg_127_to_127,
)
from torch.fx import Node

from torch.testing._internal.common_quantization import (
    NodeSpec as ns,
    PT2EQuantizationTestCase,
    skipIfNoQNNPACK,
    TestHelperModules,
)
from torch.testing._internal.common_utils import (
    TemporaryFileName,
)


@skipIfNoQNNPACK
@unittest.skipIf(sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+")
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
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 1,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default): 2,
        }
        node_list = [
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default),
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default),
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
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 3,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default): 4,
        }
        node_list = [
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default),
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default),
            ns.call_function(torch.ops.aten.conv2d.default),
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default),
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default),
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

    def test_fixed_qparams_qspec(self):
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

        m = self._quantize(m, BackendAQuantizer(), example_inputs)
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
                        share_qparams_with_input_act0_qspec = SharedQuantizationSpec((first_input_node, cat_node))
                        for input_node in input_nodes[1:]:
                            input_qspec_map[input_node] = share_qparams_with_input_act0_qspec

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
        m = capture_pre_autograd_graph(
            m,
            example_inputs,
        )
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
        assert len(conv_output_obs) == 2, "expecting two observer that follows conv2d ops"
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
        example_inputs = (torch.randn(1, 3, 5, 5), torch.randn(1, 3, 5, 5), torch.randn(1, 6, 3, 3), torch.randn(1, 6, 3, 3))

        # program capture
        m = capture_pre_autograd_graph(
            m,
            example_inputs,
        )
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

        assert len(conv_output_obs) == 2, "expecting two observer that follows conv2d ops"
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
                        share_qparams_with_input_act0_qspec = SharedQuantizationSpec((first_input_node, cat_node))
                        for input_node in input_nodes[1:]:
                            input_qspec_map[input_node] = share_qparams_with_input_act0_qspec

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
                        share_qparams_with_input_act1_qspec = SharedQuantizationSpec((second_input_node, cat_node))
                        input_qspec_map[first_input_node] = share_qparams_with_input_act1_qspec

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
                        share_qparams_with_input_act1_qspec = SharedQuantizationSpec((second_input_node, add_node))
                        input_qspec_map[first_input_node] = share_qparams_with_input_act1_qspec

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
        example_inputs = (torch.randn(1, 3, 5, 5), torch.randn(1, 3, 5, 5), torch.randn(1, 3, 5, 5), torch.randn(1, 3, 5, 5))

        # program capture
        m = capture_pre_autograd_graph(
            m,
            example_inputs,
        )
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

    def test_int16(self):
        class Int16ActQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                # using int32 to simulate int16
                int16_qspec = QuantizationSpec(
                    dtype=torch.int16,
                    quant_min=-2**15,
                    quant_max=2**15 - 1,
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
                    input_activation=int16_qspec,
                    weight=int8_qspec,
                    bias=None,
                    output_activation=int16_qspec,
                )
                OP_TO_ANNOTATOR["conv"](model, quantization_config)

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                return self.conv(x)

        quantizer = Int16ActQuantizer()
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
        example_inputs = (torch.randn(1, 3, 3, 3),)
        self._test_quantizer(
            M().eval(),
            example_inputs,
            Int16ActQuantizer(),
            node_occurrence,
            node_list,
        )

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
        m = capture_pre_autograd_graph(m, example_inputs)
        with self.assertRaises(Exception):
            m = prepare_pt2e(m, BackendAQuantizer())

    def test_fold_quantize(self):
        """Test to make sure the quantized model gets quantized weight (quantize_per_tensor op is folded)
        """
        m = self._get_pt2e_quantized_linear()
        node_occurrence = {
            # quantize op for weight node is folded
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 2,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default): 3,
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_fold_quantize_per_channel(self):
        """Test to make sure the quantized model gets quantized weight (quantize_per_channel op is folded)
        """
        m = self._get_pt2e_quantized_linear(is_per_channel=True)
        node_occurrence = {
            # quantize op for weight node is folded
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 2,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel.default): 1,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default): 2,
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_dont_fold_other_constant(self):
        """Make sure the constant propagation does not apply to things unrelated to
        quantization
        """
        class M(torch.nn.Module):
            def __init__(self):
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
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 2,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default): 3,
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
            def __init__(self):
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
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 2,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default): 3,
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_constant_prop_preserve_metadata(self):
        """Test to make sure the get_attr node for const propagated weight Tensor gets the correct
        metadata (from original get_attr node from weight)
        """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config()
        quantizer.set_global(operator_config)
        example_inputs = (torch.randn(2, 2),)
        m = M().eval()
        m = capture_pre_autograd_graph(
            m,
            example_inputs,
        )
        weight_meta = None
        for n in m.graph.nodes:
            if n.op == "get_attr" and next(iter(n.users)).target == torch.ops.aten.linear.default:
                weight_meta = n.meta
                break
        assert weight_meta is not None, "Expect to find metadata for weight node"

        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        m = convert_pt2e(m)

        for n in m.graph.nodes:
            if n.op == "get_attr" and "frozen_param" in n.target:
                self.assertIn("stack_trace", n.meta)
                for key in n.meta:
                    self.assertEqual(n.meta[key], weight_meta[key])

    def test_save_load(self):
        """Test save/load a quantized model
        """
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
            def transform_for_annotation(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for n in model.graph.nodes:
                    if n.target == torch.ops.aten.add.Tensor:
                        n.target = torch.ops.aten.mul.Tensor
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
        m = capture_pre_autograd_graph(m, example_inputs)
        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        node_occurrence = {
            ns.call_function(torch.ops.aten.add.Tensor): 0,
            ns.call_function(torch.ops.aten.mul.Tensor): 1,
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_composable_quantizer_transform_for_annotation(self):
        class TestQuantizer1(Quantizer):
            def transform_for_annotation(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for n in model.graph.nodes:
                    if n.target == torch.ops.aten.add.Tensor:
                        n.target = torch.ops.aten.mul.Tensor
                return model

            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                return model

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        class TestQuantizer2(Quantizer):
            def transform_for_annotation(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
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
        quantizer = ComposableQuantizer(
            [TestQuantizer1(), TestQuantizer2()]
        )
        example_inputs = (torch.randn(1, 2, 3, 3), torch.randn(1, 2, 3, 3), torch.randn(1, 2, 3, 3))
        m = capture_pre_autograd_graph(m, example_inputs)
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
            def __init__(self):
                super().__init__()
                self.dropout = torch.nn.Dropout(0.5, inplace=inplace)

            def forward(self, x):
                return self.dropout(x)

        example_inputs = (torch.randn(1),)
        m = M().train()
        m = capture_pre_autograd_graph(m, example_inputs)
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

    def test_move_exported_model_bn(self):
        """
        Test switching batch_norm behavior between train and eval modes using
        `move_exported_model_to_eval` and `move_exported_model_to_train` APIs.
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                return self.bn(x)

        example_inputs = (torch.randn(1, 3, 3, 3),)
        m = M().train()
        m = capture_pre_autograd_graph(m, example_inputs)

        # Assert that batch norm op exists and is in train mode
        bn_node = self._get_node(m, torch.ops.aten._native_batch_norm_legit.default)
        self.assertTrue(bn_node is not None)
        self.assertTrue(bn_node.args[5])

        # Move to eval
        torch.ao.quantization.move_exported_model_to_eval(m)

        # Assert that batch norm op is now in eval mode
        bn_node = self._get_node(m, torch.ops.aten._native_batch_norm_legit_no_training.default)
        self.assertTrue(bn_node is not None)

        # Move to train
        torch.ao.quantization.move_exported_model_to_train(m)

        # Assert that batch norm op is now in train mode again
        bn_node = self._get_node(m, torch.ops.aten._native_batch_norm_legit.default)
        self.assertTrue(bn_node is not None)
        self.assertTrue(bn_node.args[5])

    def test_disallow_eval_train(self):
        m = TestHelperModules.ConvWithBNRelu(relu=True)
        example_inputs = (torch.rand(3, 3, 5, 5),)

        # Before export: this is OK
        m.eval()
        m.train()

        # After export: this is not OK
        m = capture_pre_autograd_graph(m, example_inputs)
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

    def test_allow_exported_model_train_eval(self):

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(3)
                self.dropout = torch.nn.Dropout(0.5)

            def forward(self, x):
                x = self.bn(x)
                x = self.dropout(x)
                return x

        example_inputs = (torch.randn(1, 3, 3, 3),)
        m = M().train()
        m = capture_pre_autograd_graph(m, example_inputs)

        def _assert_ops_are_correct(m: torch.fx.GraphModule, train: bool):
            targets = [n.target for n in m.graph.nodes]
            bn_train_target = torch.ops.aten._native_batch_norm_legit.default
            bn_eval_target = torch.ops.aten._native_batch_norm_legit_no_training.default
            if train:
                self.assertTrue(bn_train_target in targets)
                self.assertTrue(bn_eval_target not in targets)
            else:
                self.assertTrue(bn_eval_target in targets)
                self.assertTrue(bn_train_target not in targets)
            dropout_node = self._get_node(m, torch.ops.aten.dropout.default)
            self.assertTrue(dropout_node.args[2] == train)

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

    def test_model_is_exported(self):
        m = TestHelperModules.ConvWithBNRelu(relu=True)
        example_inputs = (torch.rand(3, 3, 5, 5),)
        exported_gm = capture_pre_autograd_graph(m, example_inputs)
        fx_traced_gm = torch.fx.symbolic_trace(m, example_inputs)
        self.assertTrue(torch.ao.quantization.pt2e.export_utils.model_is_exported(exported_gm))
        self.assertFalse(torch.ao.quantization.pt2e.export_utils.model_is_exported(fx_traced_gm))
        self.assertFalse(torch.ao.quantization.pt2e.export_utils.model_is_exported(m))

    def test_reentrant(self):
        """Test we can safely call quantization apis multiple times"""
        m = TestHelperModules.ConvBnReLU2dAndLinearReLU()
        example_inputs = (torch.randn(3, 3, 10, 10),)

        quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True, is_qat=True))
        m.conv_bn_relu = capture_pre_autograd_graph(m.conv_bn_relu, example_inputs)
        m.conv_bn_relu = prepare_qat_pt2e(m.conv_bn_relu, quantizer)
        m(*example_inputs)
        m.conv_bn_relu = convert_pt2e(m.conv_bn_relu)

        quantizer = XNNPACKQuantizer().set_module_type(torch.nn.Linear, get_symmetric_quantization_config(is_per_channel=False))
        m = capture_pre_autograd_graph(m, example_inputs)
        m = prepare_pt2e(m, quantizer)
        m = convert_pt2e(m)

        node_occurrence = {
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 4,
            # one for weight
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default): 5,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel.default): 1,
        }
        node_list = [
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default),
            ns.call_function(torch.ops.aten.conv2d.default),
            ns.call_function(torch.ops.aten.relu.default),
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default),
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default),
            ns.call_function(torch.ops.aten.linear.default),
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default),
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
        from torch.library import Library, impl
        test_lib = Library("test_int4", "DEF")  # noqa: TOR901
        test_lib.define("quantize_per_tensor_int4(Tensor input, float scale, int zero_point) -> Tensor")

        @impl(test_lib, "quantize_per_tensor_int4", "CompositeExplicitAutograd")
        def quantize_per_tensor_int4(
            input: torch.Tensor,
            scale: float,
            zero_point: int,
        ) -> torch.Tensor:
            inv_scale = 1.0 / scale
            return torch.clamp(torch.round(input * inv_scale) + zero_point, 0, 15).to(torch.uint8).view(torch.bits8)

        test_lib.define("dequantize_per_tensor_int4(Tensor input, float scale, int zero_point) -> Tensor")

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
                        torch.ops.test_int4.quantize_per_tensor_int4, (observer_node.args[0], 1.0, 0), {})
                    dq_node = model.graph.call_function(
                        torch.ops.test_int4.dequantize_per_tensor_int4, (q_node, 1.0, 0), {})
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

        example_inputs = (torch.randn(1, 3, 5, 5), torch.randn(1, 3, 5, 5),)
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
            model = capture_pre_autograd_graph(model, example_inputs)
            # Per channel quantization for weight
            # Dynamic quantization for activation
            # Please read a detail: https://fburl.com/code/30zds51q
            embedding_quantizer = EmbeddingQuantizer()
            dynamic_quantizer = XNNPACKQuantizer()
            operator_config_dynamic = get_symmetric_quantization_config(
                is_per_channel=True, is_dynamic=True
            )
            dynamic_quantizer.set_global(operator_config_dynamic)
            composed_quantizer = ComposableQuantizer([embedding_quantizer, dynamic_quantizer])
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
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x)

        m = M().eval()
        example_inputs = (torch.randn(5, 5),)
        _ = dynamic_quantize_pt2e(m, example_inputs)
