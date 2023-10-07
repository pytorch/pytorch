# Owner(s): ["oncall: quantization"]
import copy
import operator
import unittest
from typing import Any, Optional, Tuple

import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization import (
    FusedMovingAvgObsFakeQuantize,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    QConfigMapping,
)
from torch.ao.quantization.backend_config import get_qnnpack_backend_config
from torch.ao.quantization.qconfig import (
    default_per_channel_symmetric_qnnpack_qat_qconfig,
    default_symmetric_qnnpack_qat_qconfig,
)
from torch.ao.quantization.quantize_fx import prepare_qat_fx
from torch.ao.quantization.quantize_pt2e import (
    _convert_to_reference_decomposed_fx,
    convert_pt2e,
    prepare_qat_pt2e,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    skip_if_no_torchvision,
    skipIfNoQNNPACK,
    TestHelperModules,
)
from torch.testing._internal.common_quantized import override_quantized_engine


class PT2EQATTestCase(QuantizationTestCase):
    """
    Base QuantizationTestCase for PT2E QAT with some helper methods.
    """

    def _verify_symmetric_xnnpack_qat_numerics(
        self,
        model: torch.nn.Module,
        example_inputs: Tuple[Any, ...],
    ):
        self._verify_symmetric_xnnpack_qat_numerics_helper(
            model,
            example_inputs,
            is_per_channel=True,
        )
        self._verify_symmetric_xnnpack_qat_numerics_helper(
            model,
            example_inputs,
            is_per_channel=False,
        )

    def _verify_symmetric_xnnpack_qat_numerics_helper(
        self,
        model: torch.nn.Module,
        example_inputs: Tuple[Any, ...],
        is_per_channel: bool,
        verify_convert: bool = True,
    ):
        """
        Helper method to verify that the QAT numerics for PT2E quantization match those of
        FX graph mode quantization for symmetric qnnpack.
        """
        # resetting dynamo cache
        torch._dynamo.reset()
        MANUAL_SEED = 100

        # PT2 export

        model_pt2e = copy.deepcopy(model)
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(
            get_symmetric_quantization_config(
                is_per_channel=is_per_channel, is_qat=True
            )
        )
        model_pt2e = capture_pre_autograd_graph(
            model_pt2e,
            example_inputs,
        )
        model_pt2e = prepare_qat_pt2e(model_pt2e, quantizer)
        torch.manual_seed(MANUAL_SEED)
        after_prepare_result_pt2e = model_pt2e(*example_inputs)

        model_fx = copy.deepcopy(model)
        if is_per_channel:
            default_qconfig = default_per_channel_symmetric_qnnpack_qat_qconfig
        else:
            default_qconfig = default_symmetric_qnnpack_qat_qconfig
        qconfig_mapping = QConfigMapping().set_global(default_qconfig)
        backend_config = get_qnnpack_backend_config()
        model_fx = prepare_qat_fx(
            model_fx, qconfig_mapping, example_inputs, backend_config=backend_config
        )
        torch.manual_seed(MANUAL_SEED)
        after_prepare_result_fx = model_fx(*example_inputs)

        # Verify that numerics match
        self.assertEqual(after_prepare_result_pt2e, after_prepare_result_fx)

        if verify_convert:
            torch.ao.quantization.move_exported_model_to_eval(model_pt2e)
            model_pt2e = convert_pt2e(model_pt2e)
            quant_result_pt2e = model_pt2e(*example_inputs)
            model_fx.eval()
            model_fx = _convert_to_reference_decomposed_fx(
                model_fx,
                backend_config=backend_config,
            )
            quant_result_fx = model_fx(*example_inputs)
            self.assertEqual(quant_result_pt2e, quant_result_fx)

    def _verify_symmetric_xnnpack_qat_graph(
        self,
        m: torch.fx.GraphModule,
        example_inputs: Tuple[Any, ...],
        has_relu: bool,
        has_bias: bool = True,
        is_cuda: bool = False,
        expected_conv_literal_args: Optional[Tuple[Any, ...]] = None,
    ):
        self._verify_symmetric_xnnpack_qat_graph_helper(
            m,
            example_inputs,
            is_per_channel=True,
            has_relu=has_relu,
            has_bias=has_bias,
            is_cuda=is_cuda,
            expected_conv_literal_args=expected_conv_literal_args,
        )
        self._verify_symmetric_xnnpack_qat_graph_helper(
            m,
            example_inputs,
            is_per_channel=False,
            has_relu=has_relu,
            has_bias=has_bias,
            is_cuda=is_cuda,
            expected_conv_literal_args=expected_conv_literal_args,
        )

    def _verify_symmetric_xnnpack_qat_graph_helper(
        self,
        m: torch.fx.GraphModule,
        example_inputs: Tuple[Any, ...],
        is_per_channel: bool,
        has_relu: bool,
        has_bias: bool = True,
        is_cuda: bool = False,
        expected_conv_literal_args: Optional[Tuple[Any, ...]] = None,
    ):
        """
        Verify that the graph module matches the fused QAT [conv - bn (- relu)] pattern
        with fake quantizes inserted into the correct places.
        # TODO: also verify that metadata is copied over to the new nodes.
        """
        m = copy.deepcopy(m)
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(
            get_symmetric_quantization_config(is_per_channel, is_qat=True)
        )
        m = capture_pre_autograd_graph(
            m,
            example_inputs,
        )
        m = prepare_qat_pt2e(m, quantizer)
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
        self.assertEqual(output_fq_mod.dtype, torch.int8)
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
        if is_cuda:
            if torch.version.cuda is not None:
                expected_bn_op = torch.ops.aten.cudnn_batch_norm.default
            elif torch.version.hip is not None:
                expected_bn_op = torch.ops.aten.miopen_batch_norm.default
        else:
            expected_bn_op = torch.ops.aten._native_batch_norm_legit.default
        self.assertEqual(getitem_node.target, operator.getitem)
        self.assertEqual(bn_node.target, expected_bn_op)

        # Verify: conv / scale_factor.reshape [+ bias.reshape]
        if has_bias:
            add_bias_node = bn_node.args[0]
            (div_scale_factor_node, bias_reshape_node) = add_bias_node.args
            self.assertEqual(add_bias_node.target, torch.ops.aten.add.Tensor)
            self.assertEqual(bias_reshape_node.target, torch.ops.aten.reshape.default)
        else:
            div_scale_factor_node = bn_node.args[0]
        (conv_node, scale_factor_reshape_node) = div_scale_factor_node.args
        self.assertEqual(div_scale_factor_node.target, torch.ops.aten.div.Tensor)
        self.assertEqual(conv_node.target, torch.ops.aten.conv2d.default)
        self.assertEqual(
            scale_factor_reshape_node.target, torch.ops.aten.reshape.default
        )

        # Verify: conv literal args
        if expected_conv_literal_args is not None:
            assert (
                len(expected_conv_literal_args) == 6
            ), "wrong num conv args, bad test setup"
            for i in range(6):
                if i + 3 < len(conv_node.args):
                    self.assertEqual(
                        conv_node.args[i + 3], expected_conv_literal_args[i]
                    )

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
        self.assertEqual(conv_input_fq_mod.dtype, torch.int8)
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
        self.assertEqual(conv_weight_fq_mod.dtype, torch.int8)
        self.assertEqual(conv_weight_fq_mod.quant_min, -127)
        self.assertEqual(conv_weight_fq_mod.quant_max, 127)

        # Verify: conv(fq(input), fq(weight * scale_factor.reshape), zero_bias)
        zero_bias_node = conv_node.args[2] if len(conv_node.args) > 2 else None
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
        self.assertEqual(
            scale_factor_reshape_node.target, torch.ops.aten.reshape.default
        )

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
class TestQuantizePT2EQAT(PT2EQATTestCase):
    def test_qat_conv_no_bias(self):
        class M(torch.nn.Module):
            def __init__(self, has_relu: bool):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3, bias=False)
                self.relu = torch.nn.ReLU() if has_relu else torch.nn.Identity()

            def forward(self, x):
                x = self.conv(x)
                x = self.relu(x)
                return x

        example_inputs = (torch.randn(1, 3, 5, 5),)
        self._verify_symmetric_xnnpack_qat_numerics(M(has_relu=False), example_inputs)
        self._verify_symmetric_xnnpack_qat_numerics(M(has_relu=True), example_inputs)

    def test_qat_conv_bn_fusion(self):
        m = TestHelperModules.ConvWithBNRelu(relu=False)
        example_inputs = (torch.randn(1, 3, 5, 5),)
        self._verify_symmetric_xnnpack_qat_graph(m, example_inputs, has_relu=False)
        self._verify_symmetric_xnnpack_qat_numerics(m, example_inputs)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_qat_conv_bn_fusion_cuda(self):
        m = TestHelperModules.ConvWithBNRelu(relu=False).cuda()
        example_inputs = (torch.randn(1, 3, 5, 5).cuda(),)
        self._verify_symmetric_xnnpack_qat_graph(
            m,
            example_inputs,
            has_relu=False,
            is_cuda=True,
        )
        self._verify_symmetric_xnnpack_qat_numerics(m, example_inputs)

    def test_qat_conv_bn_fusion_literal_args(self):
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
        self._verify_symmetric_xnnpack_qat_graph(
            M(),
            example_inputs,
            has_relu=False,
            expected_conv_literal_args=conv_args,
        )
        self._verify_symmetric_xnnpack_qat_numerics(M(), example_inputs)

    def test_qat_conv_bn_fusion_no_conv_bias(self):
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
        self._verify_symmetric_xnnpack_qat_graph(
            m1,
            example_inputs,
            has_relu=False,
            has_bias=False,
        )
        self._verify_symmetric_xnnpack_qat_numerics(m1, example_inputs)
        self._verify_symmetric_xnnpack_qat_numerics(M2(), example_inputs)

    def test_qat_conv_bn_relu_fusion(self):
        m = TestHelperModules.ConvWithBNRelu(relu=True)
        example_inputs = (torch.randn(1, 3, 5, 5),)
        self._verify_symmetric_xnnpack_qat_graph(m, example_inputs, has_relu=True)
        self._verify_symmetric_xnnpack_qat_numerics(m, example_inputs)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_qat_conv_bn_relu_fusion_cuda(self):
        m = TestHelperModules.ConvWithBNRelu(relu=True).cuda()
        example_inputs = (torch.randn(1, 3, 5, 5).cuda(),)
        self._verify_symmetric_xnnpack_qat_graph(
            m,
            example_inputs,
            has_relu=True,
            is_cuda=True,
        )
        self._verify_symmetric_xnnpack_qat_numerics(m, example_inputs)

    def test_qat_conv_bn_relu_fusion_no_conv_bias(self):
        m = TestHelperModules.ConvWithBNRelu(relu=True, bias=False)
        example_inputs = (torch.randn(3, 3, 5, 5),)
        self._verify_symmetric_xnnpack_qat_graph(
            m,
            example_inputs,
            has_relu=True,
            has_bias=False,
        )
        self._verify_symmetric_xnnpack_qat_numerics(m, example_inputs)

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
        self._verify_symmetric_xnnpack_qat_numerics(M(), example_inputs)

    def test_prepare_qat_conv_bn_fusion_getitem_placeholder(self):
        """
        Test the case where the placeholder node for the [conv - bn - getitem] pattern
        is also a getitem node:

          some_op -> unrelated_getitem -> conv -> bn -> conv_bn_getitem

        We want the metadata to be copied from the `conv_bn_getitem` node, not from
        the `unrelated_getitem` node, which is not part of the conv-bn pattern but
        is returned as part of the match anyway (as a placeholder).
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn1 = torch.nn.BatchNorm2d(3)
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.bn2 = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.bn1(x)
                x = self.conv(x)
                x = self.bn2(x)
                return x

        def _get_getitem_nodes(m: torch.fx.GraphModule):
            """
            Return a 2-tuple of (unrelated_getitem_node, conv_bn_getitem_node) from the graph.
            """
            unrelated_getitem_node, conv_bn_getitem_node = None, None
            for node in m.graph.nodes:
                if (
                    node.target != operator.getitem
                    or node.args[0].target
                    != torch.ops.aten._native_batch_norm_legit.default
                ):
                    continue
                if node.args[0].args[0].op == "placeholder":
                    unrelated_getitem_node = node
                else:
                    conv_bn_getitem_node = node
            assert (
                unrelated_getitem_node is not None
            ), "did not find unrelated getitem node, bad test setup"
            assert (
                conv_bn_getitem_node is not None
            ), "did not find conv bn getitem node, bad test setup"
            return (unrelated_getitem_node, conv_bn_getitem_node)

        # Program capture
        example_inputs = (torch.randn(1, 3, 5, 5),)
        m = capture_pre_autograd_graph(
            M(),
            example_inputs,
        )
        m.graph.eliminate_dead_code()
        m.recompile()
        (_, original_conv_bn_getitem_node) = _get_getitem_nodes(m)

        # Prepare QAT
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(
            get_symmetric_quantization_config(is_per_channel=False, is_qat=True)
        )
        m = prepare_qat_pt2e(m, quantizer)
        (unrelated_getitem_node, conv_bn_getitem_node) = _get_getitem_nodes(m)

        # Verify that the metadata was copied from `conv_bn_getitem`, not `unrelated_getitem`
        original_conv_bn_getitem_meta = original_conv_bn_getitem_node.meta[
            "quantization_annotation"
        ]
        conv_bn_getitem_meta = conv_bn_getitem_node.meta["quantization_annotation"]
        self.assertEqual(conv_bn_getitem_meta, original_conv_bn_getitem_meta)
        self.assertTrue("quantization_annotation" not in unrelated_getitem_node.meta)

    def test_qat_update_shared_qspec(self):
        """
        Test the case where nodes used in SharedQuantizationSpec were replaced
        during QAT subgraph rewriting.
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.bn = torch.nn.BatchNorm2d(3)
                self.hardtanh = torch.nn.Hardtanh()

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = self.hardtanh(x)
                return x

        example_inputs = (torch.randn(1, 3, 5, 5),)
        self._verify_symmetric_xnnpack_qat_numerics(M(), example_inputs)


@skipIfNoQNNPACK
class TestQuantizePT2EQATModels(PT2EQATTestCase):
    @skip_if_no_torchvision
    @skipIfNoQNNPACK
    def test_qat_resnet18(self):
        import torchvision

        with override_quantized_engine("qnnpack"):
            example_inputs = (torch.randn(1, 3, 224, 224),)
            m = torchvision.models.resnet18()
            self._verify_symmetric_xnnpack_qat_numerics(m, example_inputs)

    @skip_if_no_torchvision
    @skipIfNoQNNPACK
    def test_qat_mobilenet_v2(self):
        import torchvision

        with override_quantized_engine("qnnpack"):
            example_inputs = (torch.randn(1, 3, 224, 224),)
            m = torchvision.models.mobilenet_v2()
            self._verify_symmetric_xnnpack_qat_numerics(m, example_inputs)
