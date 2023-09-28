# Owner(s): ["oncall: quantization"]
import copy
import operator
from typing import List, Tuple

import torch
import torch._dynamo as torchdynamo
from torch._export import capture_pre_autograd_graph
from torch import Tensor
from torch.ao.ns.fx.utils import compute_sqnr
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
    _convert_to_reference_decomposed_fx,
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torch.ao.quantization.backend_config import (
    get_executorch_backend_config,
    get_qnnpack_backend_config,
)

from torch.ao.quantization.qconfig import (
    default_per_channel_symmetric_qnnpack_qconfig,
    default_symmetric_qnnpack_qconfig,
    float_qparams_weight_only_qconfig,
    per_channel_weight_observer_range_neg_127_to_127,
    QConfig,
    weight_observer_range_neg_127_to_127,
)
from torch.ao.quantization.quantize_fx import (
    convert_to_reference_fx,
    prepare_fx,
)
from torch.fx import Node

from torch.testing._internal.common_quantization import (
    NodeSpec as ns,
    QuantizationTestCase,
    skip_if_no_torchvision,
    skipIfNoQNNPACK,
    TestHelperModules,
)
from torch.testing._internal.common_utils import (
    TemporaryFileName,
)
from torch.ao.quantization import (
    default_dynamic_qconfig,
)
from torch.testing._internal.common_quantized import override_quantized_engine
from torch._export import dynamic_dim


class PT2EQuantizationTestCase(QuantizationTestCase):
    """
    Base QuantizationTestCase for PT2 with some helper methods.
    """
    _MAP_TO_FX_TRACED_OPS = {
        torch.ops.quantized_decomposed.quantize_per_tensor: torch.ops.quantized_decomposed.quantize_per_tensor.default,
        torch.ops.quantized_decomposed.dequantize_per_tensor: torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        torch.ops.quantized_decomposed.quantize_per_channel: torch.ops.quantized_decomposed.quantize_per_channel.default,
        torch.ops.quantized_decomposed.dequantize_per_channel: torch.ops.quantized_decomposed.dequantize_per_channel.default,
        torch.ops.quantized_decomposed.quantize_per_tensor.tensor: torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
        torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
    }

    def _quantize(self, m, quantizer, example_inputs):
        m = capture_pre_autograd_graph(
            m,
            example_inputs,
        )
        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        m = convert_pt2e(m, fold_quantize=True)
        return m

    def _get_pt2e_quantized_linear(self, is_per_channel=False) -> torch.fx.GraphModule:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=is_per_channel)
        quantizer.set_global(operator_config)
        example_inputs = (torch.randn(2, 2),)
        m = M().eval()
        return self._quantize(m, quantizer, example_inputs)

    def _test_quantizer(
        self,
        model,
        example_inputs,
        quantizer,
        expected_node_occurrence,
        expected_node_list=None,
        check_against_fx_quant=False,
        fx_qconfig_mapping=None,
        export_with_dynamic_shape=False,
    ):
        # resetting dynamo cache
        torch._dynamo.reset()
        m_eager = model.eval()

        # program capture
        m = copy.deepcopy(m_eager)
        m = capture_pre_autograd_graph(
            m,
            example_inputs,
            constraints=[dynamic_dim(example_inputs[0], 0)] if export_with_dynamic_shape else [],
        )

        m = prepare_pt2e(m, quantizer)
        # Calibrate
        m(*example_inputs)
        m = convert_pt2e(m, fold_quantize=True)

        pt2_quant_output = m(*example_inputs)
        node_occurrence = {
            ns.call_function(k): v for k, v in expected_node_occurrence.items()
        }
        if expected_node_list is None:
            expected_node_list = []
        node_list = [ns.call_function(n) for n in expected_node_list]
        self.checkGraphModuleNodes(
            m, expected_node_occurrence=node_occurrence, expected_node_list=node_list
        )
        if check_against_fx_quant:
            qconfig_mapping = fx_qconfig_mapping
            backend_config = get_executorch_backend_config()
            m_copy = copy.deepcopy(m_eager)
            m_fx = prepare_fx(
                m_copy, qconfig_mapping, example_inputs, backend_config=backend_config
            )
            m_fx(*example_inputs)
            m_fx = _convert_to_reference_decomposed_fx(
                m_fx, backend_config=backend_config
            )
            m_fx = capture_pre_autograd_graph(
                m_fx,
                example_inputs,
                constraints=[dynamic_dim(example_inputs[0], 0)] if export_with_dynamic_shape else [],
            )
            node_occurrence = {}
            for k, v in PT2EQuantizationTestCase._MAP_TO_FX_TRACED_OPS.items():
                if k in expected_node_occurrence:
                    node_occurrence[ns.call_function(v)] = expected_node_occurrence[k]
            self.checkGraphModuleNodes(m_fx, expected_node_occurrence=node_occurrence)
            fx_quant_output = m_fx(*example_inputs)
            self.assertEqual(fx_quant_output, pt2_quant_output)


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
                            ), f"Expecting two obs/fqs, one for activation and one for weight, got: {len(obs_or_fq)}"
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
                            ), f"Expecting one weight obs/fq, got: {len(obs_or_fq)}"
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
                conv_output_obs.append(getattr(m, list(n.users)[0].target))
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
        m = convert_pt2e(m, fold_quantize=True)

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
                OP_TO_ANNOTATOR["conv2d"](model, quantization_config)

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
            if n.op == "get_attr" and list(n.users)[0].target == torch.ops.aten.linear.default:
                weight_meta = n.meta
                break
        assert weight_meta is not None, "Expect to find metadata for weight node"

        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        m = convert_pt2e(m, fold_quantize=True)

        for n in m.graph.nodes:
            if n.op == "get_attr" and "frozen_param" in n.target:
                self.assertIn("stack_trace", n.meta)
                for key in n.meta:
                    self.assertEqual(n.meta[key], weight_meta[key])

    def test_add_and_inplace_add(self):
        quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(quantization_config)
        example_inputs = (torch.randn(1, 3, 5, 5), torch.randn(1, 3, 5, 5),)
        node_occurrence = {
            # two input and one output for first add, and output for second add
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 4,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 5,
        }
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.add.Tensor,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.add_.Tensor,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        ]
        self._test_quantizer(
            TestHelperModules.AddInplaceAdd(),
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )

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

    def test_mul_and_inplace_mul(self):
        quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(quantization_config)
        example_inputs = (torch.randn(1, 3, 5, 5), torch.randn(1, 3, 5, 5),)
        node_occurrence = {
            # two input and one output for first add, and output for second add
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 4,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 5,
        }
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.mul.Tensor,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.mul_.Tensor,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        ]
        self._test_quantizer(
            TestHelperModules.MulInplaceMul(),
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )

    def test_xnnpack_quantizer_conv(self):
        quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(quantization_config)
        example_inputs = (torch.randn(1, 3, 5, 5),)
        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
            # quantize_per_channel for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        ]
        self._test_quantizer(
            TestHelperModules.ConvWithBNRelu(relu=False, bn=False),
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )

    def test_xnnpack_quantizer_linear(self):
        quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(quantization_config)
        m_eager = TestHelperModules.TwoLinearModule().eval()

        # Test with 2d inputs
        example_inputs_2d = (torch.randn(9, 8),)
        example_inputs_3d = (torch.randn(9, 10, 8),)
        example_inputs_4d = (torch.randn(9, 10, 11, 8),)
        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
            # quantize_per_channel for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
        }
        qconfig = default_per_channel_symmetric_qnnpack_qconfig
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        for example_inputs in [example_inputs_2d, example_inputs_3d, example_inputs_4d]:
            self._test_quantizer(
                m_eager,
                example_inputs,
                quantizer,
                node_occurrence,
                [],
                True,
                qconfig_mapping,
            )

    def test_xnnpack_quantizer_conv_linear_no_permute(self):
        quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(quantization_config)
        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 5,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 5,
            # quantize_per_channel for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 3,
        }
        qconfig = default_per_channel_symmetric_qnnpack_qconfig
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        # Test with 2d inputs
        example_inputs = (torch.randn(2, 3, 4, 4),)
        self._test_quantizer(
            TestHelperModules.Conv2dWithTwoLinear(),
            example_inputs,
            quantizer,
            node_occurrence,
            [],
            True,
            qconfig_mapping,
        )

    def test_xnnpack_quantizer_conv_linear(self):
        quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(quantization_config)

        # Test with 2d inputs
        example_inputs = (torch.randn(2, 3, 4, 4),)
        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 5,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 5,
            # quantize_per_channel for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 3,
        }
        qconfig = default_per_channel_symmetric_qnnpack_qconfig
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        self._test_quantizer(
            TestHelperModules.Conv2dWithTwoLinearPermute(),
            example_inputs,
            quantizer,
            node_occurrence,
            [],
            True,
            qconfig_mapping,
        )

    def test_xnnpack_quantizer_linear_with_dynamic_shape(self):
        quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(quantization_config)
        m_eager = TestHelperModules.TwoLinearModule().eval()

        # Test with 2d inputs
        example_inputs_3d = (torch.randn(9, 10, 8),)
        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
            # quantize_per_channel for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
        }
        qconfig = default_per_channel_symmetric_qnnpack_qconfig
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        self._test_quantizer(
            m_eager,
            example_inputs_3d,
            quantizer,
            node_occurrence,
            [],
            True,
            qconfig_mapping,
            export_with_dynamic_shape=True,
        )

    def test_xnnpack_quantizer_obs_sharing_ops(self):
        quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(quantization_config)
        m = TestHelperModules.Conv2dWithObsSharingOps().eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)
        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 5,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 5,
            # quantize_per_channel for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.adaptive_avg_pool2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.hardtanh.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.mean.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        ]
        self._test_quantizer(m, example_inputs, quantizer, node_occurrence, node_list)

    def test_xnnpack_quantizer_set_module_name(self):
        class Sub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
                self.sub = Sub()

            def forward(self, x):
                x = self.linear(x)
                x = self.sub(x)
                return x

        m = M().eval()
        example_inputs = (torch.randn(3, 5),)
        quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_module_name("sub", quantization_config)
        node_occurrence = {
            torch.ops.aten.linear.default: 2,
            # input and output for the second linear
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
        }
        node_list = [
            # first linear is not quantized
            torch.ops.aten.linear.default,
            # second linear is quantized
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.linear.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        ]
        self._test_quantizer(m, example_inputs, quantizer, node_occurrence, node_list)

    def test_xnnpack_quantizer_set_module_type(self):
        class Sub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
                self.sub = Sub()

            def forward(self, x):
                x = self.linear(x)
                x = self.sub(x)
                return x

        m = M().eval()
        example_inputs = (torch.randn(3, 5),)
        quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_module_type(Sub, quantization_config)
        node_occurrence = {
            torch.ops.aten.linear.default: 2,
            # input and output for the second linear
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
        }
        node_list = [
            # first linear is not quantized
            torch.ops.aten.linear.default,
            # second linear is quantized
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.linear.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        ]
        self._test_quantizer(m, example_inputs, quantizer, node_occurrence, node_list)

    def test_propagate_annotation(self):
        quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(quantization_config)
        m = TestHelperModules.Conv2dPropAnnotaton().eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)

        # program capture
        m = capture_pre_autograd_graph(
            m,
            example_inputs,
        )

        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        self.assertEqual(
            id(m.activation_post_process_2), id(m.activation_post_process_3)
        )
        self.assertEqual(
            id(m.activation_post_process_3), id(m.activation_post_process_4)
        )
        m = convert_pt2e(m, fold_quantize=True)
        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 5,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 5,
            # note: quantize op for weights are const propagated
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_channel.default
            ): 0,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_channel.default
            ): 2,
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_xnnpack_quantizer_dynamic_linear(self):
        quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(
            is_per_channel=True, is_dynamic=True
        )
        quantizer.set_global(quantization_config)
        m_eager = TestHelperModules.TwoLinearModule().eval()

        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 2,
            # note: quantize op for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
        }
        act_affine_quant_obs = observer.PlaceholderObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine,
            quant_min=-128,
            quant_max=127,
            eps=2**-12,
            is_dynamic=True,
        )
        qconfig = QConfig(
            activation=act_affine_quant_obs,
            weight=per_channel_weight_observer_range_neg_127_to_127,
        )
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        # Test with 2d inputs
        example_inputs_2d = (torch.randn(9, 8),)
        example_inputs_4d = (torch.randn(9, 10, 11, 8),)
        for example_inputs in [example_inputs_2d, example_inputs_4d]:
            self._test_quantizer(
                m_eager,
                example_inputs,
                quantizer,
                node_occurrence,
                [],
                True,
                qconfig_mapping,
            )

    def test_xnnpack_quantizer_dynamic_linear_with_conv(self):
        quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(
            is_per_channel=False, is_dynamic=True
        )
        quantizer.set_global(quantization_config)
        m_eager = TestHelperModules.ConvLinearWPermute().eval()

        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 1,
            # note: quantize op for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
        }
        act_affine_quant_obs = observer.PlaceholderObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine,
            quant_min=-128,
            quant_max=127,
            eps=2**-12,
            is_dynamic=True,
        )
        qconfig = QConfig(
            activation=act_affine_quant_obs,
            weight=weight_observer_range_neg_127_to_127,
        )
        # Test with 2d inputs
        example_inputs = (torch.randn(2, 3, 4, 4),)
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        self._test_quantizer(
            m_eager,
            example_inputs,
            quantizer,
            node_occurrence,
            [],
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

    def test_move_exported_model_to_eval(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = torch.nn.Dropout(0.5)

            def forward(self, x):
                return self.dropout(x)

        example_inputs = (torch.randn(1),)
        m = M().train()
        m = capture_pre_autograd_graph(m, example_inputs)

        # Assert that dropout op exists and is in train mode
        dropout_node = None
        for n in m.graph.nodes:
            if n.target == torch.ops.aten.native_dropout.default:
                dropout_node = n
                break
        self.assertTrue(dropout_node is not None)
        self.assertTrue(dropout_node.args[2])

        # Do the subgraph rewriting
        torch.ao.quantization.move_exported_model_to_eval(m)

        # Assert that dropout op is now replaced with a clone op
        targets = [n.target for n in m.graph.nodes]
        self.assertTrue(torch.ops.aten.clone.default in targets)
        self.assertTrue(torch.ops.aten.native_dropout.default not in targets)

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
        m = convert_pt2e(m, fold_quantize=True)
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()


@skipIfNoQNNPACK
class TestQuantizePT2EOps(QuantizationTestCase):
    def test_gru(self):
        """ this is a test for annotating fp32 GRU so that it produces
        q -> dq -> fp32_gru -> q -> dq, this is currently enough for our use cases,
        but we may change the annotation to be more precise in the future
        """
        class RNNDynamicModel(torch.nn.Module):
            def __init__(self, mod_type):
                super().__init__()
                self.qconfig = default_dynamic_qconfig
                if mod_type == 'GRU':
                    self.mod = torch.nn.GRU(2, 2).to(dtype=torch.float)
                if mod_type == 'LSTM':
                    self.mod = torch.nn.LSTM(2, 2).to(dtype=torch.float)

            def forward(self, input_tensor, hidden_tensor):
                input_tensor = 1 * input_tensor
                hidden_tensor = 1 * hidden_tensor
                output_tensor, hidden_out = self.mod(input_tensor, hidden_tensor)
                return 1 * output_tensor, 1 * hidden_out

        with override_quantized_engine("qnnpack"):
            model_fx = RNNDynamicModel("GRU")
            module_types = [torch.nn.GRU]
            niter = 10
            example_inputs = (
                # input_tensor
                torch.tensor([[100, -155],
                              [-155, 100],
                              [100, -155]], dtype=torch.float).unsqueeze(0).repeat(niter, 1, 1),
                # hidden_tensor
                # (D * num_layers, N, H_out)
                torch.tensor([[[100, -155]]], dtype=torch.float).repeat(1, 3, 1),
            )
            model_graph = copy.deepcopy(model_fx)

            qconfig_mapping = QConfigMapping().set_object_type(operator.mul, default_symmetric_qnnpack_qconfig)
            model_fx = prepare_fx(model_fx, qconfig_mapping, example_inputs, backend_config=get_qnnpack_backend_config())
            model_fx(*example_inputs)
            model_fx = _convert_to_reference_decomposed_fx(model_fx)

            torchdynamo.config.allow_rnn = True
            model_graph = capture_pre_autograd_graph(
                model_graph,
                example_inputs,
            )
            quantizer = XNNPACKQuantizer()
            quantization_config = get_symmetric_quantization_config(
                is_per_channel=False, is_dynamic=False
            )
            quantizer.set_global(quantization_config)
            model_graph = prepare_pt2e(model_graph, quantizer)
            model_graph(*example_inputs)
            model_graph = convert_pt2e(model_graph, fold_quantize=True)
            self.assertEqual(model_fx(*example_inputs), model_graph(*example_inputs))


    def test_linear_gru(self):
        """ this test is to make sure GRU annotation does not interfere with linear annotation
        """
        class RNNDynamicModel(torch.nn.Module):
            def __init__(self, mod_type):
                super().__init__()
                self.qconfig = default_dynamic_qconfig
                self.linear = torch.nn.Linear(2, 2)
                if mod_type == 'GRU':
                    self.mod = torch.nn.GRU(2, 2).to(dtype=torch.float)
                if mod_type == 'LSTM':
                    self.mod = torch.nn.LSTM(2, 2).to(dtype=torch.float)

            def forward(self, input_tensor, hidden_tensor):
                input_tensor = self.linear(input_tensor)
                input_tensor = 1 * input_tensor
                hidden_tensor = 1 * hidden_tensor
                output_tensor, hidden_out = self.mod(input_tensor, hidden_tensor)
                return 1 * output_tensor, 1 * hidden_out

        with override_quantized_engine("qnnpack"):
            model_fx = RNNDynamicModel("GRU")
            module_types = [torch.nn.GRU]
            niter = 10
            example_inputs = (
                # input_tensor
                torch.tensor([[100, -155],
                              [-155, 100],
                              [100, -155]], dtype=torch.float).unsqueeze(0).repeat(niter, 1, 1),
                # hidden_tensor
                # (D * num_layers, N, H_out)
                torch.tensor([[[100, -155]]], dtype=torch.float).repeat(1, 3, 1),
            )
            model_graph = copy.deepcopy(model_fx)

            qconfig_mapping = (
                QConfigMapping().set_object_type(
                    operator.mul, default_symmetric_qnnpack_qconfig
                ).set_object_type(
                    torch.nn.Linear, default_symmetric_qnnpack_qconfig
                )
            )
            model_fx = prepare_fx(model_fx, qconfig_mapping, example_inputs, backend_config=get_qnnpack_backend_config())
            model_fx(*example_inputs)
            model_fx = _convert_to_reference_decomposed_fx(model_fx)

            torchdynamo.config.allow_rnn = True
            model_graph = capture_pre_autograd_graph(
                model_graph,
                example_inputs,
            )
            quantizer = XNNPACKQuantizer()
            quantization_config = get_symmetric_quantization_config(
                is_per_channel=False, is_dynamic=False
            )
            quantizer.set_global(quantization_config)
            model_graph = prepare_pt2e(model_graph, quantizer)
            model_graph(*example_inputs)
            model_graph = convert_pt2e(model_graph, fold_quantize=True)
            self.assertEqual(model_fx(*example_inputs), model_graph(*example_inputs))


# TODO: express this using self._test_quantizer, add test for inception_v4
class TestQuantizePT2EModels(PT2EQuantizationTestCase):
    @skip_if_no_torchvision
    @skipIfNoQNNPACK
    def test_resnet18(self):
        import torchvision

        with override_quantized_engine("qnnpack"):
            example_inputs = (torch.randn(1, 3, 224, 224),)
            m = torchvision.models.resnet18().eval()
            m_copy = copy.deepcopy(m)
            # program capture
            m = capture_pre_autograd_graph(
                m,
                example_inputs,
            )

            quantizer = XNNPACKQuantizer()
            quantization_config = get_symmetric_quantization_config(is_per_channel=True)
            quantizer.set_global(quantization_config)
            m = prepare_pt2e(m, quantizer)
            # checking that we inserted observers correctly for maxpool operator (input and
            # output share observer instance)
            self.assertEqual(
                id(m.activation_post_process_3), id(m.activation_post_process_2)
            )
            after_prepare_result = m(*example_inputs)
            m = convert_pt2e(m, fold_quantize=True)

            after_quant_result = m(*example_inputs)

            # comparing with existing fx graph mode quantization reference flow
            qconfig = default_per_channel_symmetric_qnnpack_qconfig
            qconfig_mapping = QConfigMapping().set_global(qconfig)
            backend_config = get_qnnpack_backend_config()
            m_fx = prepare_fx(
                m_copy, qconfig_mapping, example_inputs, backend_config=backend_config
            )
            after_prepare_result_fx = m_fx(*example_inputs)
            m_fx = convert_to_reference_fx(m_fx, backend_config=backend_config)

            after_quant_result_fx = m_fx(*example_inputs)

            # the result matches exactly after prepare
            # Note: this currently will always be true since we are inserting observers
            # the check becomes useful when we add qat examples
            # but we can still manully inspect the printed observers to make sure
            # it matches
            self.assertEqual(after_prepare_result, after_prepare_result_fx)
            self.assertEqual(
                compute_sqnr(after_prepare_result, after_prepare_result_fx),
                torch.tensor(float("inf")),
            )
            # there are slight differences after convert due to different implementations
            # of quant/dequant
            self.assertTrue(
                torch.max(after_quant_result - after_quant_result_fx) < 1e-1
            )
            self.assertTrue(
                compute_sqnr(after_quant_result, after_quant_result_fx) > 35
            )
