# Owner(s): ["oncall: quantization"]
import copy
import operator
from typing import Any, List, Optional, Tuple

import torch
import torch._dynamo as torchdynamo
from torch import Tensor
from torch.ao.ns.fx.utils import compute_sqnr
from torch.ao.quantization import (
    FusedMovingAvgObsFakeQuantize,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    observer,
    ObserverOrFakeQuantize,
    QConfigMapping,
)
from torch.ao.quantization._pt2e.quantizer import (
    ComposableQuantizer,
    DerivedQuantizationSpec,
    EmbeddingQuantizer,
    OperatorConfig,
    QNNPackQuantizer,
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
    FixedQParamsQuantizationSpec,
    SharedQuantizationSpec,
)
from torch.ao.quantization._pt2e.quantizer.composable_quantizer import (  # noqa: F811
    ComposableQuantizer,
)
from torch.ao.quantization._pt2e.quantizer.qnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from torch.ao.quantization._quantize_pt2e import (
    _convert_to_reference_decomposed_fx,
    convert_pt2e,
    prepare_pt2e_quantizer,
    prepare_qat_pt2e_quantizer,
)
from torch.ao.quantization.backend_config import (
    get_executorch_backend_config,
    get_qnnpack_backend_config,
)

from torch.ao.quantization.qconfig import (
    default_per_channel_symmetric_qnnpack_qat_qconfig,
    default_per_channel_symmetric_qnnpack_qconfig,
    default_symmetric_qnnpack_qconfig,
    default_symmetric_qnnpack_qat_qconfig,
    float_qparams_weight_only_qconfig,
    per_channel_weight_observer_range_neg_127_to_127,
    QConfig,
    weight_observer_range_neg_127_to_127,
)
from torch.ao.quantization.quantize_fx import (
    convert_to_reference_fx,
    prepare_fx,
    prepare_qat_fx,
)
from torch.fx import Node

from torch.testing._internal.common_quantization import (
    NodeSpec as ns,
    QuantizationTestCase,
    skip_if_no_torchvision,
    skipIfNoQNNPACK,
)
from torch.ao.quantization import (
    default_dynamic_qconfig,
)
from torch.testing._internal.common_quantized import override_quantized_engine

# TODO: Move to common utils or use existing quant utils to fetch model instances
class TestHelperModules:
    class Conv2dPropAnnotaton(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, x):
            x = self.conv(x)
            x = x.view(-1, 3)
            x = torch.nn.functional.hardtanh(x, -0.5, 0.5)
            x = self.linear(x)
            return x

    class Conv2dWithObsSharingOps(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.hardtanh = torch.nn.Hardtanh()
            self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.conv(x)
            x = self.adaptive_avg_pool2d(x)
            x = self.hardtanh(x)
            x = torch.mean(x)
            return x

    class Conv2dWithTwoLinearPermute(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3)
            self.linear1 = torch.nn.Linear(16, 8, bias=False)
            self.linear2 = torch.nn.Linear(8, 8)

        def forward(self, x):
            conv_out = self.conv(x)
            permute_out = torch.permute(conv_out, (0, 2, 3, 1))
            return self.linear2(self.linear1(permute_out))

    class Conv2dWithTwoLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3)
            self.linear1 = torch.nn.Linear(64, 8, bias=False)
            self.linear2 = torch.nn.Linear(8, 8)

        def forward(self, x):
            conv_out = self.conv(x)
            reshape_out = torch.reshape(conv_out, (2, 64))
            return self.linear2(self.linear1(reshape_out))

    class ConvLinearWPermute(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, 3)
            self.linear1 = torch.nn.Linear(8, 8)

        def forward(self, x):
            conv_out = self.conv(x)
            permute_out = torch.permute(conv_out, (0, 2, 3, 1))
            return self.linear1(permute_out)

    class TwoLinearModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(8, 16, bias=False)
            self.linear2 = torch.nn.Linear(16, 8)

        def forward(self, x):
            return self.linear2(self.linear1(x))

    class ConvMaxPool2d(torch.nn.Module):
        def __init__(self):
            super(TestHelperModules.ConvMaxPool2d, self).__init__()
            self.conv = torch.nn.Conv2d(2, 2, 1)
            self.pool = torch.nn.MaxPool2d(1, 1)

        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            return x

    class ConvWithBNRelu(torch.nn.Module):
        def __init__(self, relu, bn=True, bias=True):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3, bias=bias)
            if bn:
                self.bn = torch.nn.BatchNorm2d(3)
            else:
                self.bn = torch.nn.Identity()
            if relu:
                self.relu = torch.nn.ReLU()
            else:
                self.relu = torch.nn.Identity()

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            return self.relu(x)

    class EmbeddingModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=12)

        def forward(self, indices):
            return self.emb(indices)

    class EmbeddingConvLinearModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=8)
            self.conv = torch.nn.Conv2d(8, 16, (1, 3))
            self.linear = torch.nn.Linear(16, 8)

        def forward(self, indices):
            embeddings = self.emb(indices)
            embeddings = torch.unsqueeze(embeddings, dim=0)
            embeddings = torch.permute(embeddings, (0, 3, 1, 2))
            conv_out = self.conv(embeddings)
            conv_out = torch.permute(conv_out, (0, 2, 3, 1))
            conv_out = torch.squeeze(conv_out, dim=0)
            return self.linear(conv_out)


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

    def _test_quantizer(
        self,
        model,
        example_inputs,
        quantizer,
        expected_node_occurrence,
        expected_node_list=None,
        check_against_fx_quant=False,
        fx_qconfig_mapping=None,
    ):
        m_eager = model.eval()

        # program capture
        m = copy.deepcopy(m_eager)
        m, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
            tracing_mode="real",
        )

        m = prepare_pt2e_quantizer(m, quantizer)
        # Calibrate
        m(*example_inputs)
        m = convert_pt2e(m)
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
            m_fx, guards = torchdynamo.export(
                m_fx,
                *copy.deepcopy(example_inputs),
                aten_graph=True,
                tracing_mode="real",
            )
            node_occurrence = {}
            for k, v in PT2EQuantizationTestCase._MAP_TO_FX_TRACED_OPS.items():
                if k in expected_node_occurrence:
                    node_occurrence[ns.call_function(v)] = expected_node_occurrence[k]
            self.checkGraphModuleNodes(m_fx, expected_node_occurrence=node_occurrence)
            fx_quant_output = m_fx(*example_inputs)
            self.assertTrue(torch.allclose(fx_quant_output, pt2_quant_output))

    def _verify_symmetric_qnnpack_qat_numerics(
        self,
        model: torch.nn.Module,
        example_inputs: Tuple[Any, ...],
        is_per_channel: bool,
        verify_convert: bool = False,
    ):
        """
        Helper method to verify that the QAT numerics for PT2E quantization match those of
        FX graph mode quantization for symmetric qnnpack.
        """
        # PT2 export

        model_pt2e = copy.deepcopy(model)
        quantizer = QNNPackQuantizer()
        quantizer.set_global(
            get_symmetric_quantization_config(
                is_per_channel=is_per_channel, is_qat=True
            )
        )
        model_pt2e, guards = torchdynamo.export(
            model_pt2e,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        model_pt2e = prepare_qat_pt2e_quantizer(model_pt2e, quantizer)
        after_prepare_result_pt2e = model_pt2e(*example_inputs)

        # FX
        # Note: In order to match the PT2E numerics exactly, we need to feed the
        # example inputs to the model once before calling prepare, since this is
        # what torchdynamo.export does. Otherwise, the BN running mean and variance
        # would diverge in the two flows and this test would fail. For more detail,
        # see https://github.com/pytorch/pytorch/issues/95900.
        model_fx = copy.deepcopy(model)
        model_fx(*example_inputs)
        if is_per_channel:
            default_qconfig = default_per_channel_symmetric_qnnpack_qat_qconfig
        else:
            default_qconfig = default_symmetric_qnnpack_qat_qconfig
        qconfig_mapping = QConfigMapping().set_global(default_qconfig)
        backend_config = get_qnnpack_backend_config()
        model_fx = prepare_qat_fx(
            model_fx, qconfig_mapping, example_inputs, backend_config=backend_config
        )
        after_prepare_result_fx = model_fx(*example_inputs)

        # Verify that numerics match
        self.assertEqual(after_prepare_result_pt2e, after_prepare_result_fx)

        if verify_convert:
            model_pt2e.eval()
            model_pt2e = convert_pt2e(model_pt2e)
            quant_result_pt2e = model_pt2e(*example_inputs)
            model_fx.eval()
            model_fx = _convert_to_reference_decomposed_fx(
                model_fx, backend_config=backend_config,
            )
            quant_result_fx = model_fx(*example_inputs)
            self.assertEqual(quant_result_pt2e, quant_result_fx)

    def _verify_symmetric_qnnpack_qat_graph(
        self,
        m: torch.fx.GraphModule,
        example_inputs: Tuple[Any, ...],
        is_per_channel: bool,
        has_relu: bool,
        has_bias: bool = True,
        expected_conv_constant_args: Optional[Tuple[Any, ...]] = None,
    ):
        """
        Verify that the graph module matches the fused QAT [conv - bn (- relu)] pattern
        with fake quantizes inserted into the correct places.
        # TODO: also verify that metadata is copied over to the new nodes.
        """
        quantizer = QNNPackQuantizer()
        quantizer.set_global(
            get_symmetric_quantization_config(is_per_channel, is_qat=True)
        )
        m, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
            tracing_mode="real",
        )
        m = prepare_qat_pt2e_quantizer(m, quantizer)
        m(*example_inputs)

        # Verify: getitem output activation fake quantize
        output_node = list(m.graph.nodes)[-1]
        output_fq_node = output_node.args[0][0]
        self.assertTrue(output_fq_node.target.startswith("activation_post_process_"))
        output_fq_mod = getattr(m, output_fq_node.target)
        self.assertEqual(type(output_fq_mod), FusedMovingAvgObsFakeQuantize)
        self.assertEqual(
            type(output_fq_mod.activation_post_process), MovingAverageMinMaxObserver
        )
        self.assertEqual(output_fq_mod.dtype, torch.qint8)
        self.assertEqual(output_fq_mod.quant_min, -128)
        self.assertEqual(output_fq_mod.quant_max, 127)

        # Verify: getitem(bn, 0) or relu(getitem(bn, 0))
        if has_relu:
            relu_node = output_fq_node.args[0]
            getitem_node = relu_node.args[0]
            self.assertEqual(relu_node.target, torch.ops.aten.relu.default)
        else:
            relu_node = None
            getitem_node = output_fq_node.args[0]
        bn_node = getitem_node.args[0]
        self.assertEqual(getitem_node.target, operator.getitem)
        self.assertEqual(
            bn_node.target, torch.ops.aten._native_batch_norm_legit.default
        )

        # Verify: conv / scale_factor.reshape [+ bias.reshape]
        if has_bias:
            add_bias_node = bn_node.args[0]
            (div_scale_factor_node, bias_reshape_node) = add_bias_node.args
            self.assertEqual(add_bias_node.target, torch.ops.aten.add.Tensor)
            self.assertEqual(bias_reshape_node.target, torch.ops.aten.view.default)
        else:
            div_scale_factor_node = bn_node.args[0]
        (conv_node, scale_factor_reshape_node) = div_scale_factor_node.args
        self.assertEqual(div_scale_factor_node.target, torch.ops.aten.div.Tensor)
        self.assertEqual(conv_node.target, torch.ops.aten.convolution.default)
        self.assertEqual(scale_factor_reshape_node.target, torch.ops.aten.view.default)

        # Verify: conv constant args
        if expected_conv_constant_args is not None:
            assert (
                len(expected_conv_constant_args) == 6
            ), "wrong num conv args, bad test setup"
            for i in range(6):
                self.assertEqual(conv_node.args[i + 3], expected_conv_constant_args[i])

        # Verify: conv input activation fake quantize
        conv_input_fq_node = conv_node.args[0]
        conv_input_node = conv_input_fq_node.args[0]
        self.assertTrue(
            conv_input_fq_node.target.startswith("activation_post_process_")
        )
        conv_input_fq_mod = getattr(m, conv_input_fq_node.target)
        self.assertEqual(type(conv_input_fq_mod), FusedMovingAvgObsFakeQuantize)
        self.assertEqual(
            type(conv_input_fq_mod.activation_post_process), MovingAverageMinMaxObserver
        )
        self.assertEqual(conv_input_fq_mod.dtype, torch.qint8)
        self.assertEqual(conv_input_fq_mod.quant_min, -128)
        self.assertEqual(conv_input_fq_mod.quant_max, 127)
        self.assertTrue(conv_input_node.op, "placeholder")

        # Verify: conv weight fake quantize
        conv_weight_fq_node = conv_node.args[1]
        self.assertTrue(
            conv_weight_fq_node.target.startswith("activation_post_process_")
        )
        conv_weight_fq_mod = getattr(m, conv_weight_fq_node.target)
        if is_per_channel:
            expected_weight_observer_type = MovingAveragePerChannelMinMaxObserver
        else:
            expected_weight_observer_type = MovingAverageMinMaxObserver
        self.assertEqual(type(conv_weight_fq_mod), FusedMovingAvgObsFakeQuantize)
        self.assertEqual(
            type(conv_weight_fq_mod.activation_post_process),
            expected_weight_observer_type,
        )
        self.assertEqual(conv_weight_fq_mod.dtype, torch.qint8)
        self.assertEqual(conv_weight_fq_mod.quant_min, -127)
        self.assertEqual(conv_weight_fq_mod.quant_max, 127)

        # Verify: conv(fq(input), fq(weight * scale_factor.reshape), zero_bias)
        zero_bias_node = conv_node.args[2]
        mul_weight_scale_factor_node = conv_weight_fq_node.args[0]
        (
            conv_weight_fq_node,
            scale_factor_reshape_node,
        ) = mul_weight_scale_factor_node.args
        if has_bias:
            self.assertEqual(zero_bias_node.target, torch.ops.aten.zeros_like.default)
        else:
            self.assertTrue(zero_bias_node is None)
        self.assertEqual(mul_weight_scale_factor_node.target, torch.ops.aten.mul.Tensor)
        self.assertEqual(scale_factor_reshape_node.target, torch.ops.aten.view.default)

        # Verify: scale_factor = bn_weight / sqrt(bn_running_var + eps)
        scale_factor_node = scale_factor_reshape_node.args[0]
        (bn_weight_node, sqrt_node) = scale_factor_node.args
        bn_running_var_add_node = sqrt_node.args[0]
        (bn_running_var_node, eps) = bn_running_var_add_node.args
        self.assertEqual(scale_factor_node.target, torch.ops.aten.div.Tensor)
        self.assertTrue("param_constant" in bn_weight_node.target)
        self.assertEqual(sqrt_node.target, torch.ops.aten.sqrt.default)
        self.assertEqual(bn_running_var_add_node.target, torch.ops.aten.add.Tensor)
        self.assertTrue("tensor_constant" in bn_running_var_node.target)
        self.assertEqual(eps, 1e-5)


@skipIfNoQNNPACK
class TestQuantizePT2E(PT2EQuantizationTestCase):
    def test_simple_quantizer(self):
        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.convolution.default
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

            @classmethod
            def get_supported_operators(cls) -> List[OperatorConfig]:
                pass

        example_inputs = (torch.randn(1, 3, 5, 5),)
        node_occurrence = {
            # two for input of the first conv, one for output for the first conv
            torch.ops.quantized_decomposed.quantize_per_tensor: 3,
            torch.ops.quantized_decomposed.dequantize_per_tensor: 3,
        }
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor,
            torch.ops.quantized_decomposed.dequantize_per_tensor,
            torch.ops.aten.convolution.default,
            torch.ops.quantized_decomposed.quantize_per_tensor,
        ]
        self._test_quantizer(
            TestHelperModules.ConvWithBNRelu(relu=False, bn=False),
            example_inputs,
            BackendAQuantizer(),
            node_occurrence,
            node_list,
        )

    def test_max_pool2d_quantizer(self):
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
                        and node.target == torch.ops.aten.convolution.default
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
                            output_qspec=act_qspec,
                            _annotated=True,
                        )
                    if (
                        node.op == "call_function"
                        and node.target == operator.getitem
                        and node.args[1] == 0
                    ):
                        getitem_node = node
                        maxpool_node = getitem_node.args[0]
                        input_act = maxpool_node.args[0]
                        assert isinstance(input_act, Node)
                        maxpool_node.meta[
                            "quantization_annotation"
                        ] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                            },
                            _annotated=True,
                        )
                        getitem_node.meta[
                            "quantization_annotation"
                        ] = QuantizationAnnotation(
                            output_qspec=SharedQuantizationSpec(
                                (input_act, maxpool_node)
                            ),
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

            @classmethod
            def get_supported_operators(cls) -> List[OperatorConfig]:
                pass

        m = TestHelperModules.ConvMaxPool2d()
        x = torch.rand(1, 2, 14, 14)
        example_inputs = (x,)
        node_occurrence = {
            # two for input of maxpool
            # one for input for maxpool
            # one for output of maxpool
            torch.ops.quantized_decomposed.quantize_per_tensor: 4,
            torch.ops.quantized_decomposed.dequantize_per_tensor: 4,
        }
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor,
            torch.ops.quantized_decomposed.dequantize_per_tensor,
            torch.ops.aten.convolution.default,
            torch.ops.quantized_decomposed.quantize_per_tensor,
            torch.ops.quantized_decomposed.dequantize_per_tensor,
            torch.ops.aten.max_pool2d_with_indices.default,
            torch.ops.quantized_decomposed.quantize_per_tensor,
            torch.ops.quantized_decomposed.dequantize_per_tensor,
        ]
        self._test_quantizer(
            m, example_inputs, BackendAQuantizer(), node_occurrence, node_list
        )

    def test_derived_qspec(self):
        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.convolution.default
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
                            ), "Expecting two obs/fqs, one for activation and one for weight, got: {}".format(
                                len(obs_or_fq)
                            )
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

            @classmethod
            def get_supported_operators(cls) -> List[OperatorConfig]:
                pass

        m = TestHelperModules.ConvWithBNRelu(relu=False, bn=False).eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)

        # program capture
        m, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        m = prepare_pt2e_quantizer(m, BackendAQuantizer())
        m(*example_inputs)
        m = convert_pt2e(m)
        node_occurrence = {
            # input, weight, bias, output for the conv
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 4,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 4,
        }
        node_list = [
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
            ns.call_function(torch.ops.aten.convolution.default),
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
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

            @classmethod
            def get_supported_operators(cls) -> List[OperatorConfig]:
                pass

        m = M().eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)

        # program capture
        m, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        m = prepare_pt2e_quantizer(m, BackendAQuantizer())
        m(*example_inputs)
        m = convert_pt2e(m)
        fixed_scale = 1.0 / 256.0
        fixed_zero_point = 0
        self.assertEqual(m._scale_0, fixed_scale)
        self.assertEqual(m._zero_point_0, fixed_zero_point)
        self.assertEqual(m._scale_1, fixed_scale)
        self.assertEqual(m._zero_point_1, fixed_zero_point)
        node_occurrence = {
            # two for input of the first conv, one for output for the first conv
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 2,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 2,
        }
        node_list = [
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor),
            ns.call_function(torch.ops.aten.sigmoid.default),
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor),
        ]
        self.checkGraphModuleNodes(
            m, expected_node_list=node_list, expected_node_occurrence=node_occurrence
        )

    def test_qnnpack_quantizer_conv(self):
        quantizer = QNNPackQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(operator_config)
        example_inputs = (torch.randn(1, 3, 5, 5),)
        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor: 2,
            torch.ops.quantized_decomposed.quantize_per_channel: 1,
            torch.ops.quantized_decomposed.dequantize_per_channel: 1,
        }
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor,
            torch.ops.quantized_decomposed.dequantize_per_channel,
            torch.ops.aten.convolution.default,
            torch.ops.quantized_decomposed.quantize_per_tensor,
        ]
        self._test_quantizer(
            TestHelperModules.ConvWithBNRelu(relu=False, bn=False),
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )

    def test_qnnpack_quantizer_linear(self):
        quantizer = QNNPackQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(operator_config)
        m_eager = TestHelperModules.TwoLinearModule().eval()

        # Test with 2d inputs
        example_inputs_2d = (torch.randn(9, 8),)
        example_inputs_3d = (torch.randn(9, 10, 8),)
        example_inputs_4d = (torch.randn(9, 10, 11, 8),)
        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor: 3,
            torch.ops.quantized_decomposed.dequantize_per_tensor: 3,
            torch.ops.quantized_decomposed.quantize_per_channel: 2,
            torch.ops.quantized_decomposed.dequantize_per_channel: 2,
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

    def test_qnnpack_quantizer_conv_linear_no_permute(self):
        quantizer = QNNPackQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(operator_config)
        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor: 5,
            torch.ops.quantized_decomposed.dequantize_per_tensor: 5,
            torch.ops.quantized_decomposed.quantize_per_channel: 3,
            torch.ops.quantized_decomposed.dequantize_per_channel: 3,
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

    def test_qnnpack_quantizer_conv_linear(self):
        quantizer = QNNPackQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(operator_config)

        # Test with 2d inputs
        example_inputs = (torch.randn(2, 3, 4, 4),)
        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor: 5,
            torch.ops.quantized_decomposed.dequantize_per_tensor: 5,
            torch.ops.quantized_decomposed.quantize_per_channel: 3,
            torch.ops.quantized_decomposed.dequantize_per_channel: 3,
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

    def test_qnnpack_quantizer_obs_sharing_ops(self):
        quantizer = QNNPackQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(operator_config)
        m = TestHelperModules.Conv2dWithObsSharingOps().eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)
        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor: 5,
            torch.ops.quantized_decomposed.dequantize_per_tensor: 5,
            torch.ops.quantized_decomposed.quantize_per_channel: 1,
            torch.ops.quantized_decomposed.dequantize_per_channel: 1,
        }
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor,
            torch.ops.quantized_decomposed.dequantize_per_channel,
            torch.ops.aten.convolution.default,
            torch.ops.quantized_decomposed.quantize_per_tensor,
            torch.ops.quantized_decomposed.dequantize_per_tensor,
            torch.ops.aten.mean.dim,
            torch.ops.quantized_decomposed.quantize_per_tensor,
            torch.ops.quantized_decomposed.dequantize_per_tensor,
            torch.ops.aten.hardtanh.default,
            torch.ops.quantized_decomposed.quantize_per_tensor,
            torch.ops.quantized_decomposed.dequantize_per_tensor,
            torch.ops.aten.mean.default,
            torch.ops.quantized_decomposed.quantize_per_tensor,
            torch.ops.quantized_decomposed.dequantize_per_tensor,
        ]
        self._test_quantizer(m, example_inputs, quantizer, node_occurrence, node_list)

    def test_propagate_annotation(self):
        quantizer = QNNPackQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(operator_config)
        m = TestHelperModules.Conv2dPropAnnotaton().eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)

        # program capture
        m, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )

        m = prepare_pt2e_quantizer(m, quantizer)
        m(*example_inputs)
        self.assertEqual(
            id(m.activation_post_process_2), id(m.activation_post_process_3)
        )
        self.assertEqual(
            id(m.activation_post_process_3), id(m.activation_post_process_4)
        )
        m = convert_pt2e(m)
        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 5,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 5,
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_channel): 2,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel): 2,
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_qnnpack_quantizer_dynamic_linear(self):
        quantizer = QNNPackQuantizer()
        operator_config = get_symmetric_quantization_config(
            is_per_channel=True, is_dynamic=True
        )
        quantizer.set_global(operator_config)
        m_eager = TestHelperModules.TwoLinearModule().eval()

        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 2,
            torch.ops.quantized_decomposed.quantize_per_channel: 2,
            torch.ops.quantized_decomposed.dequantize_per_channel: 2,
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
            # program capture
            self._test_quantizer(
                m_eager,
                example_inputs,
                quantizer,
                node_occurrence,
                [],
                True,
                qconfig_mapping,
            )

    def test_qnnpack_quantizer_dynamic_linear_with_conv(self):
        quantizer = QNNPackQuantizer()
        operator_config = get_symmetric_quantization_config(
            is_per_channel=False, is_dynamic=True
        )
        quantizer.set_global(operator_config)
        m_eager = TestHelperModules.ConvLinearWPermute().eval()

        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 1,
            torch.ops.quantized_decomposed.quantize_per_tensor: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor: 1,
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
        dynamic_quantizer = QNNPackQuantizer()
        operator_config_dynamic = get_symmetric_quantization_config(
            is_per_channel=False, is_dynamic=True
        )
        dynamic_quantizer.set_global(operator_config_dynamic)
        static_quantizer = QNNPackQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=True)
        static_quantizer.set_global(operator_config)
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
            torch.ops.quantized_decomposed.quantize_per_tensor: 4,
            torch.ops.quantized_decomposed.dequantize_per_tensor: 4,
            torch.ops.quantized_decomposed.quantize_per_channel: 1,
            torch.ops.quantized_decomposed.dequantize_per_channel: 1,
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

            @classmethod
            def get_supported_operators(cls) -> List[OperatorConfig]:
                pass

        quantizer = QNNPackQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(operator_config)
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
            torch.ops.quantized_decomposed.quantize_per_channel: 1,
            torch.ops.quantized_decomposed.dequantize_per_channel: 1,
        }
        node_list = [
            torch.ops.quantized_decomposed.quantize_per_channel,
            torch.ops.quantized_decomposed.dequantize_per_channel,
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
        dynamic_quantizer = QNNPackQuantizer()
        operator_config_dynamic = get_symmetric_quantization_config(
            is_per_channel=True, is_dynamic=True
        )
        dynamic_quantizer.set_global(operator_config_dynamic)
        static_quantizer = QNNPackQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=True)
        static_quantizer.set_global(operator_config)
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
            torch.ops.quantized_decomposed.quantize_per_tensor: 4,
            torch.ops.quantized_decomposed.dequantize_per_tensor: 4,
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 1,
            torch.ops.quantized_decomposed.quantize_per_channel: 3,
            torch.ops.quantized_decomposed.dequantize_per_channel: 3,
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

    def test_prepare_qat_conv_bn_fusion(self):
        example_inputs = (torch.randn(1, 3, 5, 5),)
        m = TestHelperModules.ConvWithBNRelu(relu=False)
        self._verify_symmetric_qnnpack_qat_graph(
            m, example_inputs, is_per_channel=False, has_relu=False
        )
        m = TestHelperModules.ConvWithBNRelu(relu=False)
        self._verify_symmetric_qnnpack_qat_graph(
            m, example_inputs, is_per_channel=True, has_relu=False
        )

    def test_prepare_qat_conv_bn_fusion_constant_args(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3, stride=(2, 2), padding=(4, 4))
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        example_inputs = (torch.randn(1, 3, 5, 5),)
        # stride, padding, dilation, transposed, output_padding, groups
        conv_args = ((2, 2), (4, 4), (1, 1), False, (0, 0), 1)
        self._verify_symmetric_qnnpack_qat_graph(
            M(),
            example_inputs,
            is_per_channel=False,
            has_relu=False,
            expected_conv_constant_args=conv_args,
        )
        self._verify_symmetric_qnnpack_qat_graph(
            M(),
            example_inputs,
            is_per_channel=True,
            has_relu=False,
            expected_conv_constant_args=conv_args,
        )
        self._verify_symmetric_qnnpack_qat_numerics(
            M(), example_inputs, is_per_channel=False
        )
        self._verify_symmetric_qnnpack_qat_numerics(
            M(), example_inputs, is_per_channel=True
        )

    def test_prepare_qat_conv_bn_fusion_no_conv_bias(self):
        class M2(torch.nn.Module):
            """
            Mixed conv + BN with and without conv bias.
            """

            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3, bias=False)
                self.bn1 = torch.nn.BatchNorm2d(3)
                self.conv2 = torch.nn.Conv2d(3, 3, 3, bias=True)
                self.bn2 = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.conv2(x)
                x = self.bn2(x)
                return x

        m1 = TestHelperModules.ConvWithBNRelu(relu=False, bias=False)
        example_inputs = (torch.randn(3, 3, 5, 5),)
        self._verify_symmetric_qnnpack_qat_graph(
            m1, example_inputs, is_per_channel=False, has_relu=False, has_bias=False
        )
        m1 = TestHelperModules.ConvWithBNRelu(relu=False, bias=False)
        self._verify_symmetric_qnnpack_qat_graph(
            m1, example_inputs, is_per_channel=True, has_relu=False, has_bias=False
        )
        m1 = TestHelperModules.ConvWithBNRelu(relu=False, bias=False)
        self._verify_symmetric_qnnpack_qat_numerics(
            m1, example_inputs, is_per_channel=False
        )
        m1 = TestHelperModules.ConvWithBNRelu(relu=False, bias=False)
        self._verify_symmetric_qnnpack_qat_numerics(
            m1, example_inputs, is_per_channel=True
        )
        self._verify_symmetric_qnnpack_qat_numerics(
            M2(), example_inputs, is_per_channel=False
        )
        self._verify_symmetric_qnnpack_qat_numerics(
            M2(), example_inputs, is_per_channel=True
        )

    def test_prepare_qat_conv_bn_relu_fusion(self):
        m1 = TestHelperModules.ConvWithBNRelu(relu=True)
        example_inputs = (torch.randn(1, 3, 5, 5),)
        self._verify_symmetric_qnnpack_qat_graph(
            m1, example_inputs, is_per_channel=False, has_relu=True
        )
        m1 = TestHelperModules.ConvWithBNRelu(relu=True)
        self._verify_symmetric_qnnpack_qat_graph(
            m1, example_inputs, is_per_channel=True, has_relu=True
        )

    def test_qat_inplace_add_relu(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.relu = torch.nn.ReLU(inplace=True)

            def forward(self, x):
                x0 = x
                x = self.conv(x)
                x += x0
                x = self.relu(x)
                return x

        example_inputs = (torch.randn(1, 1, 3, 3),)
        self._verify_symmetric_qnnpack_qat_numerics(
            M(), example_inputs, is_per_channel=False, verify_convert=True,
        )
        self._verify_symmetric_qnnpack_qat_numerics(
            M(), example_inputs, is_per_channel=True, verify_convert=True,
        )

    def test_prepare_qat_conv_bn_fusion_getitem_placeholder(self):
        """
        Test this special case seen in resnet18:

          maxpool -> maxpool_getitem -> conv -> bn -> conv_bn_getitem

        We want the metadata to be copied from the `conv_bn_getitem` node, not `maxpool_getitem`.
        """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.maxpool = torch.nn.MaxPool2d(kernel_size=1)
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.maxpool(x)
                x = self.conv(x)
                x = self.bn(x)
                return x

        def _get_getitem_nodes(m: torch.fx.GraphModule):
            """
            Return a 2-tuple of (maxpool_getitem_node, conv_bn_getitem_node) from the graph.
            """
            maxpool_getitem_node, conv_bn_getitem_node = None, None
            for node in m.graph.nodes:
                if node.target != operator.getitem:
                    continue
                if (
                    node.args[0].target
                    == torch.ops.aten.max_pool2d_with_indices.default
                ):
                    maxpool_getitem_node = node
                elif (
                    node.args[0].target
                    == torch.ops.aten._native_batch_norm_legit.default
                ):
                    conv_bn_getitem_node = node
                else:
                    raise ValueError("Unexpected getitem node ", node, node.args)
            assert (
                maxpool_getitem_node is not None
            ), "did not find maxpool getitem node, bad test setup"
            assert (
                conv_bn_getitem_node is not None
            ), "did not find conv bn getitem node, bad test setup"
            return (maxpool_getitem_node, conv_bn_getitem_node)

        # Program capture
        example_inputs = (torch.randn(1, 3, 5, 5),)
        m, guards = torchdynamo.export(
            M(),
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        m.graph.eliminate_dead_code()
        m.recompile()
        (_, original_conv_bn_getitem_node) = _get_getitem_nodes(m)

        # Prepare QAT
        quantizer = QNNPackQuantizer()
        quantizer.set_global(
            get_symmetric_quantization_config(is_per_channel=False, is_qat=True)
        )
        m = prepare_qat_pt2e_quantizer(m, quantizer)
        (maxpool_getitem_node, conv_bn_getitem_node) = _get_getitem_nodes(m)

        # Verify that the metadata was copied from `conv_bn_getitem`, not `maxpool_getitem`
        original_conv_bn_getitem_meta = original_conv_bn_getitem_node.meta[
            "quantization_annotation"
        ]
        conv_bn_getitem_meta = conv_bn_getitem_node.meta["quantization_annotation"]
        self.assertEqual(conv_bn_getitem_meta, original_conv_bn_getitem_meta)

    # TODO: merge these numerics tests with the graph tests above
    def test_qat_conv_bn_numerics(self):
        m = TestHelperModules.ConvWithBNRelu(relu=False)
        example_inputs = (torch.randn(1, 3, 5, 5),)
        self._verify_symmetric_qnnpack_qat_numerics(
            m, example_inputs, is_per_channel=False, verify_convert=True,
        )
        self._verify_symmetric_qnnpack_qat_numerics(
            m, example_inputs, is_per_channel=True, verify_convert=True,
        )

    def test_qat_conv_bn_relu_numerics(self):
        m = TestHelperModules.ConvWithBNRelu(relu=True)
        example_inputs = (torch.randn(1, 3, 5, 5),)
        self._verify_symmetric_qnnpack_qat_numerics(
            m, example_inputs, is_per_channel=False, verify_convert=True,
        )
        self._verify_symmetric_qnnpack_qat_numerics(
            m, example_inputs, is_per_channel=True, verify_convert=True,
        )

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
            model_graph, guards = torchdynamo.export(
                model_graph,
                *copy.deepcopy(example_inputs),
                aten_graph=True,
                tracing_mode="real",
            )
            quantizer = QNNPackQuantizer()
            operator_config = get_symmetric_quantization_config(
                is_per_channel=False, is_dynamic=False
            )
            quantizer.set_global(operator_config)
            model_graph = prepare_pt2e_quantizer(model_graph, quantizer)
            model_graph(*example_inputs)
            model_graph = convert_pt2e(model_graph)
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
            model_graph, guards = torchdynamo.export(
                model_graph,
                *copy.deepcopy(example_inputs),
                aten_graph=True,
                tracing_mode="real",
            )
            quantizer = QNNPackQuantizer()
            operator_config = get_symmetric_quantization_config(
                is_per_channel=False, is_dynamic=False
            )
            quantizer.set_global(operator_config)
            model_graph = prepare_pt2e_quantizer(model_graph, quantizer)
            model_graph(*example_inputs)
            model_graph = convert_pt2e(model_graph)
            self.assertEqual(model_fx(*example_inputs), model_graph(*example_inputs))


# TODO: express this using self._test_quantizer
class TestQuantizePT2EModels(PT2EQuantizationTestCase):
    @skip_if_no_torchvision
    @skipIfNoQNNPACK
    def test_resnet18_with_quantizer_api(self):
        import torchvision

        with override_quantized_engine("qnnpack"):
            example_inputs = (torch.randn(1, 3, 224, 224),)
            m = torchvision.models.resnet18().eval()
            m_copy = copy.deepcopy(m)
            # program capture
            m, guards = torchdynamo.export(
                m,
                *copy.deepcopy(example_inputs),
                aten_graph=True,
            )

            quantizer = QNNPackQuantizer()
            operator_config = get_symmetric_quantization_config(is_per_channel=True)
            quantizer.set_global(operator_config)
            m = prepare_pt2e_quantizer(m, quantizer)
            # checking that we inserted observers correctly for maxpool operator (input and
            # output share observer instance)
            self.assertEqual(
                id(m.activation_post_process_3), id(m.activation_post_process_2)
            )
            after_prepare_result = m(*example_inputs)
            m = convert_pt2e(m)

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

    @skip_if_no_torchvision
    @skipIfNoQNNPACK
    def test_qat_resnet18(self):
        import torchvision
        with override_quantized_engine("qnnpack"):
            example_inputs = (torch.randn(1, 3, 224, 224),)
            m = torchvision.models.resnet18()
            # TODO: verify convert numerics in a future PR
            self._verify_symmetric_qnnpack_qat_numerics(
                m, example_inputs, is_per_channel=False,
            )
            self._verify_symmetric_qnnpack_qat_numerics(
                m, example_inputs, is_per_channel=True,
            )
