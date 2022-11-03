# Owner(s): ["oncall: quantization"]

from collections import OrderedDict
import contextlib
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.reference as nnqr
import torch.ao.nn.quantized.dynamic as nnqd
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nniq
import torch.nn.intrinsic.quantized.dynamic as nniqd
import torch.multiprocessing as mp

# graph mode quantization based on fx
from torch.ao.quantization.quantize_fx import (
    prepare_fx,
    convert_fx,
    convert_to_reference_fx,
    _convert_to_reference_decomposed_fx,
    prepare_qat_fx,
    fuse_fx,
)


from torch.ao.quantization.fx.quantization_patterns import DefaultNodeQuantizeHandler

from torch.ao.quantization.fx.match_utils import (
    _is_match,
    MatchAllNode,
)

from torch.ao.quantization import (
    QuantType,
    _get_quant_type_to_str,
)

from torch.ao.quantization import (
    QuantStub,
    DeQuantStub,
    QuantWrapper,
    default_qconfig,
    default_dynamic_qconfig,
    default_qat_qconfig,
    default_reuse_input_qconfig,
    default_symmetric_qnnpack_qconfig,
    default_symmetric_qnnpack_qat_qconfig,
    per_channel_dynamic_qconfig,
    float16_dynamic_qconfig,
    float16_static_qconfig,
    float_qparams_weight_only_qconfig,
    float_qparams_weight_only_qconfig_4bit,
    get_default_qconfig,
    get_default_qat_qconfig,
    get_default_qconfig_mapping,
    get_default_qat_qconfig_mapping,
    fuse_modules,
    fuse_modules_qat,
    prepare,
    prepare_qat,
    convert,
    quantize_dynamic,
    default_placeholder_observer,
    default_weight_observer,
    PerChannelMinMaxObserver,
    FixedQParamsFakeQuantize,
    FixedQParamsObserver,
    FusedMovingAvgObsFakeQuantize,
    FakeQuantize,
    MovingAverageMinMaxObserver,
    HistogramObserver,
    ReuseInputObserver,
    QConfig,
    default_embedding_qat_qconfig,
)

from torch.ao.quantization.backend_config import (
    get_qnnpack_backend_config,
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    DTypeWithConstraints,
    ObservationType
)
from torch.ao.quantization.backend_config.native import (
    get_test_only_legacy_native_backend_config,
)

from torch.ao.quantization.qconfig_mapping import (
    _get_symmetric_qnnpack_qconfig_mapping,
    _GLOBAL_DICT_KEY,
    _MODULE_NAME_DICT_KEY,
    _MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY,
    _MODULE_NAME_REGEX_DICT_KEY,
    _OBJECT_TYPE_DICT_KEY,
    QConfigMapping,
)

from torch.ao.quantization.qconfig_mapping_utils import (
    _get_object_type_qconfig,
    _get_module_name_qconfig,
    _get_module_name_regex_qconfig,
)

from torch.ao.quantization.fx.pattern_utils import (
    _DEFAULT_FUSION_PATTERNS,
    _DEFAULT_QUANTIZATION_PATTERNS,
    _DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP,
    _DEFAULT_OUTPUT_OBSERVER_MAP,
    _register_fusion_pattern,
    _register_quant_pattern,
    get_default_output_activation_post_process_map
)

from torch.ao.quantization.fx.custom_config import (
    STANDALONE_MODULE_NAME_DICT_KEY,
    STANDALONE_MODULE_CLASS_DICT_KEY,
    FLOAT_TO_OBSERVED_DICT_KEY,
    OBSERVED_TO_QUANTIZED_DICT_KEY,
    NON_TRACEABLE_MODULE_NAME_DICT_KEY,
    NON_TRACEABLE_MODULE_CLASS_DICT_KEY,
    INPUT_QUANTIZED_INDEXES_DICT_KEY,
    OUTPUT_QUANTIZED_INDEXES_DICT_KEY,
    PRESERVED_ATTRIBUTES_DICT_KEY,
    FuseCustomConfig,
    ConvertCustomConfig,
    PrepareCustomConfig,
    StandaloneModuleConfigEntry,
)

from torch.ao.quantization.fx.qconfig_mapping_utils import (
    maybe_adjust_qconfig_for_module_name_object_type_order,
)

from torch.ao.quantization.fx.utils import (
    _reroute_tuple_getitem_pattern,
    NodeInfo,
)

from torch.ao.quantization.fake_quantize import (
    default_fixed_qparams_range_0to1_fake_quant,
    default_fixed_qparams_range_neg1to1_fake_quant,
)

from torch.ao.quantization.observer import (
    default_fixed_qparams_range_0to1_observer,
    default_fixed_qparams_range_neg1to1_observer,
    MinMaxObserver,
    _is_activation_post_process,
)

# test utils
from hypothesis import given, settings
from hypothesis import strategies as st
from torch.testing._internal.common_cuda import TEST_MULTIGPU, TEST_CUDA
from torch.testing._internal.common_quantization import (
    LinearReluLinearModel,
    LinearReluModel,
    QuantizationTestCase,
    skipIfNoFBGEMM,
    skip_if_no_torchvision,
    train_one_epoch,
    run_ddp,
    test_only_eval_fn,
    test_only_train_fn,
    ModelForConvTransposeBNFusion,
    get_supported_device_types,
)

from torch.testing._internal.common_quantization import (
    LinearModelWithSubmodule,
    ResNetBase,
    RNNDynamicModel,
    RNNCellDynamicModel,
)

from torch.testing._internal.common_quantized import (
    supported_qengines,
    override_qengines,
    override_quantized_engine,
)

from torch.testing._internal.common_utils import TemporaryFileName, IS_ARM64

from torch.testing._internal.common_quantization import NodeSpec as ns

from torch.testing import FileCheck

import copy
import itertools
import operator
import unittest
import io
from typing import Callable, Optional, List

class BinaryOp(torch.nn.Module):
    def __init__(self, binary_op, ibinary_op, is_inplace, is_scalar):
        """ ibinary_op means inplace binary op
        """
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1).float()
        self.conv2 = torch.nn.Conv2d(1, 1, 1).float()
        self.is_scalar = is_scalar
        self.op = ibinary_op if ibinary_op and is_inplace else binary_op

    def forward(self, x, y):
        x = self.conv1(x)
        y = 3 if self.is_scalar else self.conv2(y)
        # x = x + y
        x = self.op(x, y)
        # x = y + x
        x = self.op(y, x)
        return x

class BinaryOpNonQuantizedInput(torch.nn.Module):
    def __init__(self, binary_op, ibinary_op, is_inplace, is_scalar):
        """ ibinary_op means inplace binary op
        """
        super().__init__()
        self.is_scalar = is_scalar
        self.op = ibinary_op if ibinary_op and is_inplace else binary_op

    def forward(self, x, y):
        y = 3 if self.is_scalar else y
        x = self.op(x, y)
        return x

class BinaryOpRelu(torch.nn.Module):
    def __init__(self, binary_op, ibinary_op, is_inplace, relu_callable,
                 is_scalar):
        """ ibinary_op means inplace binary op
        """
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1).float()
        self.conv2 = torch.nn.Conv2d(1, 1, 1).float()
        self.op = ibinary_op if ibinary_op and is_inplace else binary_op
        self.relu_callable = relu_callable
        self.is_scalar = is_scalar
        if relu_callable is torch.nn.ReLU:
            self.relu = torch.nn.ReLU()
        else:
            self.relu = relu_callable

    def forward(self, x, y):
        x = self.conv1(x)
        y = 3 if self.is_scalar else self.conv2(y)
        x = self.op(x, y)
        x = self.relu(x)
        x = self.op(y, x)
        x = self.relu(x)
        return x

@torch.fx.wrap
def _user_func_with_complex_return_type(x):
    return list(torch.split(x, 1, 1))

class TestFuseFx(QuantizationTestCase):
    def test_fuse_conv_bn_relu(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1d = nn.Conv1d(1, 1, 1)
                self.conv2d = nn.Conv2d(1, 1, 1)
                self.conv3d = nn.Conv3d(1, 1, 1)
                self.bn1d = nn.BatchNorm1d(1)
                self.bn2d = nn.BatchNorm2d(1)
                self.bn3d = nn.BatchNorm3d(1)
                self.conv1d2 = nn.Conv1d(1, 1, 1)
                self.conv2d2 = nn.Conv2d(1, 1, 1)
                self.conv3d2 = nn.Conv3d(1, 1, 1)
                self.bn1d2 = nn.BatchNorm1d(1)
                self.bn2d2 = nn.BatchNorm2d(1)
                self.bn3d2 = nn.BatchNorm3d(1)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.conv1d(x)
                x = self.bn1d(x)
                x = self.conv2d(x)
                x = self.bn2d(x)
                x = self.conv3d(x)
                x = self.bn3d(x)
                x = self.conv1d2(x)
                x = self.bn1d2(x)
                x = self.relu(x)
                x = self.conv2d2(x)
                x = self.bn2d2(x)
                x = self.relu(x)
                x = self.conv3d2(x)
                x = self.bn3d2(x)
                x = self.relu(x)
                return x

        # test train mode
        m = M().train()
        # currently we don't check if the module are configured with qconfig before fusion
        # TODO: if we decide to do that in the future, this test needs to
        # be updated
        # train mode fuse_fx is called in prepare_qat_fx
        m = prepare_qat_fx(m, {}, example_inputs=(torch.randn(1, 1, 1, 1),))
        expected_nodes = [
            ns.call_module(nni.ConvBn1d),
            ns.call_module(nni.ConvBn2d),
            ns.call_module(nni.ConvBn3d),
            ns.call_module(nni.ConvBnReLU1d),
            ns.call_module(nni.ConvBnReLU2d),
            ns.call_module(nni.ConvBnReLU3d),
        ]
        expected_occurrence = {
            ns.call_module(nn.ReLU): 0
        }
        self.checkGraphModuleNodes(
            m,
            expected_node_list=expected_nodes,
            expected_node_occurrence=expected_occurrence)

        # test eval mode
        m = M().eval()
        # fuse_fx is a top level api and only supports eval mode
        m = fuse_fx(m)
        expected_nodes = [
            ns.call_module(nn.Conv1d),
            ns.call_module(nn.Conv2d),
            ns.call_module(nn.Conv3d),
            ns.call_module(nni.ConvReLU1d),
            ns.call_module(nni.ConvReLU2d),
            ns.call_module(nni.ConvReLU3d),
        ]
        # ConvBnRelu1d is not fused
        expected_occurrence = {
            ns.call_module(nn.ReLU): 0
        }
        self.checkGraphModuleNodes(
            m,
            expected_node_list=expected_nodes,
            expected_node_occurrence=expected_occurrence)

    def test_fuse_linear_bn_eval(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)
                self.bn1d = nn.BatchNorm1d(1)

            def forward(self, x):
                x = self.linear(x)
                x = self.bn1d(x)
                return x

        # test eval mode
        m = M().eval()
        # fuse_fx is a top level api and only supports eval mode
        m = fuse_fx(m)
        expected_nodes = [
            ns.call_module(nn.Linear),
        ]
        expected_occurrence = {
            ns.call_module(nn.BatchNorm1d): 0,
        }
        self.checkGraphModuleNodes(
            m,
            expected_node_list=expected_nodes,
            expected_node_occurrence=expected_occurrence)

    def test_fuse_convtranspose_bn_eval(self):

        m = ModelForConvTransposeBNFusion().eval()
        m = fuse_fx(m)

        expected_nodes = [
            ns.call_module(nn.ConvTranspose1d),
            ns.call_module(nn.ConvTranspose2d),
            ns.call_module(nn.ConvTranspose3d),
        ]
        expected_occurrence = {
            ns.call_module(nn.BatchNorm1d): 0,
            ns.call_module(nn.BatchNorm2d): 0,
            ns.call_module(nn.BatchNorm3d): 0,
        }
        self.checkGraphModuleNodes(
            m,
            expected_node_list=expected_nodes,
            expected_node_occurrence=expected_occurrence)


    def test_fuse_module_relu(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1d = nn.Conv1d(1, 1, 1)
                self.conv2d = nn.Conv2d(1, 1, 1)
                self.conv3d = nn.Conv3d(1, 1, 1)
                self.bn1d = nn.BatchNorm1d(1)
                self.bn2d = nn.BatchNorm2d(1)
                self.bn3d = nn.BatchNorm3d(1)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.conv1d(x)
                x = self.relu(x)
                x = self.conv2d(x)
                x = self.relu(x)
                x = self.conv3d(x)
                x = self.relu(x)
                x = self.bn1d(x)
                x = self.relu(x)
                x = self.bn2d(x)
                x = self.relu(x)
                x = self.bn3d(x)
                x = self.relu(x)
                return x

        m = M().eval()
        m = fuse_fx(m)
        expected_nodes = [
            ns.call_module(nni.ConvReLU1d),
            ns.call_module(nni.ConvReLU2d),
            ns.call_module(nni.ConvReLU3d),
            ns.call_module(nni.BNReLU2d),
            ns.call_module(nni.BNReLU3d),
        ]
        self.checkGraphModuleNodes(m, expected_node_list=expected_nodes)

    @skipIfNoFBGEMM
    def test_qconfig_fused_module(self):
        """ TODO: add test for all fused modules
        """
        qconfig_dict = {
            "": None,
            "object_type": [(nn.Linear, default_qconfig),
                            (nn.ReLU, default_qconfig),
                            (F.relu, default_qconfig)]
        }

        linearRelu_node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nniq.LinearReLU),
            ns.call_method('dequantize')
        ]

        linearReluLinear_node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nniq.LinearReLU),
            ns.call_module(nnq.Linear),
            ns.call_method('dequantize')
        ]

        tests = [(LinearReluModel, linearRelu_node_list),
                 (LinearReluLinearModel, linearReluLinear_node_list)]

        for M, node_list in tests:
            m = M().eval()
            example_inputs = (torch.rand(5, 5),)
            prepared = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)

            prepared(*example_inputs)
            quantized = convert_fx(prepared)

            self.checkGraphModuleNodes(quantized, expected_node_list=node_list)

    def test_problematic_fuse_example(self):
        class LinearRelu(nn.Sequential):
            def __init__(self):
                super().__init__(
                    nn.Linear(5, 5),
                    nn.ReLU(),
                )

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin_relu = LinearRelu()
                self.linear = nn.Linear(5, 5)

            def forward(self, x):
                x = self.lin_relu(x)
                x = self.linear(x)
                return x

        model = M().eval()
        # these qconfigs somehow fail equality where default_qconfig does not
        qconfig_dict = {
            "": None,
            "object_type": [
                (torch.nn.Linear, get_default_qconfig('fbgemm')),
                (torch.nn.ReLU, get_default_qconfig('fbgemm')),
            ],
        }
        m = prepare_fx(model, qconfig_dict, example_inputs=(torch.randn(1, 5),))

        self.checkGraphModuleNodes(m, expected_node=ns.call_module(torch.nn.intrinsic.modules.fused.LinearReLU))

    @unittest.skip("Temprorarily skipping the test case, will enable after the simple"
                   "pattern format is supported")
    def test_fuse_addtional_fuser_method(self):
        class MyConvReLU(torch.nn.Module):
            pass

        def my_conv_relu_fuser(conv, relu):
            return MyConvReLU()

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.conv(x))

        m = M().eval()
        m = fuse_fx(m, fuse_custom_config={
            "additional_fuser_method_mapping": {
                (torch.nn.Conv2d, torch.nn.ReLU): my_conv_relu_fuser
            }
        })
        self.checkGraphModuleNodes(m, expected_node=ns.call_module(MyConvReLU))

    def test_fuse_custom_pattern(self):
        class M(torch.nn.Module):
            def __init__(self, use_torch_add=True):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.bn = torch.nn.BatchNorm2d(3)
                self.relu = torch.nn.ReLU()
                self.maxpool = torch.nn.MaxPool2d(3)
                if use_torch_add:
                    self.add = torch.add
                else:
                    self.add = operator.add

            def forward(self, x):
                y = x
                y = self.maxpool(x)
                x = self.conv(x)
                x = self.bn(x)
                x = self.add(y, x)
                x = self.relu(x)
                return x

        for use_torch_add in [True, False]:
            m = M(use_torch_add).eval()

            def fuse_conv_bn_relu(is_qat, relu, add_pattern):
                _, _, bn_pattern = add_pattern
                bn, conv = bn_pattern
                return conv

            conv_bn_res_relu_config1 = BackendPatternConfig((nn.ReLU, (torch.add, MatchAllNode, (nn.BatchNorm2d, nn.Conv2d)))) \
                .set_fuser_method(fuse_conv_bn_relu)
            conv_bn_res_relu_config2 = BackendPatternConfig((nn.ReLU, (operator.add, MatchAllNode, (nn.BatchNorm2d, nn.Conv2d)))) \
                .set_fuser_method(fuse_conv_bn_relu)
            backend_config = BackendConfig() \
                .set_backend_pattern_config(conv_bn_res_relu_config1) \
                .set_backend_pattern_config(conv_bn_res_relu_config2)
            m = fuse_fx(m, backend_config=backend_config)
            self.assertEqual(type(m.conv), torch.nn.Conv2d)
            # check bn and relu are gone since we replaced the whole pattern to conv
            self.assertFalse(hasattr(m, "bn"))
            self.assertFalse(hasattr(m, "relu"))

    def test_fusion_pattern_with_multiple_inputs(self):
        """ This test tests two keys in backend_config: root_node_getter and
        extra_inputs_getter,
        root_node_getter is used to identify a "root" module in the node pattern,
        the node that we'll keep after fusion.
        extra_inputs_getter will return a list of node that needs to be added to the
        fused node as extra inputs.
        """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.bn = torch.nn.BatchNorm2d(3)
                self.relu = torch.nn.ReLU()
                self.maxpool = torch.nn.MaxPool2d(3)

            def forward(self, x):
                y = x
                y = self.maxpool(x)
                x = self.conv(x)
                x = self.bn(x)
                x = torch.add(x, y)
                x = self.relu(x)
                return x

        m = M().eval()

        def fuse_conv_bn_relu(is_qat, relu, add_pattern):
            _, bn_pattern, _ = add_pattern
            bn, conv = bn_pattern
            return conv

        def conv_bn_res_relu_root_node_getter(pattern):
            relu, add_pattern = pattern
            _, bn_pattern, _ = add_pattern
            bn, conv = bn_pattern
            return conv

        def conv_bn_res_relu_extra_inputs_getter(pattern):
            """ get inputs pattern for extra inputs, inputs for root node
            are assumed to be copied over from root node to the fused node
            """
            relu, add_pattern = pattern
            _, bn_pattern, extra_input = add_pattern
            bn, conv = bn_pattern
            return [extra_input]

        conv_bn_res_relu_config = BackendPatternConfig((nn.ReLU, (torch.add, (nn.BatchNorm2d, nn.Conv2d), MatchAllNode))) \
            .set_fuser_method(fuse_conv_bn_relu) \
            ._set_root_node_getter(conv_bn_res_relu_root_node_getter) \
            ._set_extra_inputs_getter(conv_bn_res_relu_extra_inputs_getter)
        backend_config = BackendConfig().set_backend_pattern_config(conv_bn_res_relu_config)
        m = fuse_fx(m, backend_config=backend_config)
        self.assertEqual(type(m.conv), torch.nn.Conv2d)
        # check bn and relu are gone since we replaced the whole pattern to conv
        self.assertFalse(hasattr(m, "bn"))
        self.assertFalse(hasattr(m, "relu"))

        # check conv module has two inputs
        named_modules = dict(m.named_modules())
        for node in m.graph.nodes:
            if node.op == "call_module" and type(named_modules[node.target]) == torch.nn.Conv2d:
                self.assertTrue(len(node.args) == 2), "Expecting the fused op to have two arguments"

    def test_fusion_pattern_with_matchallnode(self):
        """This test tests that the node matched by MatchAllNode will be regared as an input
        instead of a module to be fused. For instance, we have two patterns:
            (nn.ReLU, (torch.add, MatchAllNode, nn.Conv2d))
            (nn.ReLU, nn.Conv2d)
        And we wanna fuse the following model
            Conv2d -> ReLU +
            Conv2d ------ Add -> ReLU
        ReLU in the first row is matched as MatchAllNode in the residual pattern. But it won't be
        fused as part of that pattnern. It needs to be properly fused with the upstream Conv2d.
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3)
                self.relu1 = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv2d(3, 3, 3)
                self.relu2 = torch.nn.ReLU()

            def forward(self, x):
                y = self.conv1(x)
                y = self.relu1(y)

                x = self.conv2(x)
                x = torch.add(x, y)
                x = self.relu2(x)
                return x

        m = M().eval()

        def fuse_conv_relu(is_qat, relu, conv):
            return conv

        def fuse_conv_res_relu(is_qat, relu, add_pattern):
            _, conv, _ = add_pattern
            return conv

        def conv_res_relu_root_node_getter(pattern):
            relu, (_, conv, _) = pattern
            return conv

        def conv_res_relu_extra_inputs_getter(pattern):
            relu, (_, _, extra_input) = pattern
            return [extra_input]

        conv_relu_config = BackendPatternConfig((nn.ReLU, nn.Conv2d)) \
            .set_fuser_method(fuse_conv_relu)
        conv_res_relu_config = BackendPatternConfig((nn.ReLU, (torch.add, nn.Conv2d, MatchAllNode))) \
            .set_fuser_method(fuse_conv_res_relu) \
            ._set_root_node_getter(conv_res_relu_root_node_getter) \
            ._set_extra_inputs_getter(conv_res_relu_extra_inputs_getter)
        backend_config = BackendConfig() \
            .set_backend_pattern_config(conv_relu_config) \
            .set_backend_pattern_config(conv_res_relu_config)
        m = fuse_fx(m, backend_config=backend_config)
        self.assertEqual(type(m.conv1), torch.nn.Conv2d)
        self.assertEqual(type(m.conv2), torch.nn.Conv2d)
        # check relu are gone since we replaced the both patterns to conv
        self.assertFalse(hasattr(m, "relu1"))
        self.assertFalse(hasattr(m, "relu2"))


@skipIfNoFBGEMM
class TestQuantizeFx(QuantizationTestCase):
    def test_pattern_match(self):
        """ test MatchAllNode with
            conv - bn - add - relu pattern
        """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 1)
                self.bn = nn.BatchNorm2d(1)
                self.relu = nn.ReLU()

            def forward(self, x, y):
                x = self.conv(x)
                x = self.bn(x)
                x = x + y
                x = self.relu(x)
                return x

        pattern = (nn.ReLU, (operator.add, (nn.BatchNorm2d, nn.Conv2d), MatchAllNode))
        m = torch.fx.symbolic_trace(M())
        modules = dict(m.named_modules())
        for n in m.graph.nodes:
            if n.op == 'call_module' and type(modules[n.target]) == nn.ReLU:
                self.assertTrue(_is_match(modules, n, pattern))

    def test_fused_module_qat_swap(self):
        class Tmp(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tmp = torch.nn.Linear(5, 5)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.tmp(x)
                return self.relu(x)


        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mods1 = torch.nn.Sequential(Tmp(), torch.nn.Linear(5, 5))
                self.mods2 = torch.nn.Linear(5, 5)

            def forward(self, x):
                a = self.mods1(x)
                x = torch.add(x, 5)
                x = self.mods2(x)
                x = torch.add(x, 5)
                return a, x


        model = M().train()
        qconfig_dict = {
            "": None,
            "object_type": [
                (torch.nn.Linear, default_qat_qconfig),
                (torch.nn.ReLU, default_qat_qconfig),
            ],
        }
        prepared = prepare_qat_fx(model, qconfig_dict, example_inputs=(torch.randn(1, 5),))
        self.assertTrue(isinstance(getattr(prepared.mods1, "0").tmp, torch.ao.nn.intrinsic.qat.LinearReLU))

    def _get_conv_linear_test_cases(self, is_reference):
        """ Returns a list of test cases, with format:
        is_dynamic, ModuleClass, module_constructor_inputs,
        inputs, quantized_node, weight_prepack_op
        """
        class FunctionalConv1d(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = torch.nn.Parameter(weight)
                self.stride = 1
                self.padding = 0
                self.dilation = 1
                self.groups = 1

            def forward(self, x):
                return F.conv1d(x, self.weight, None, self.stride, self.padding, self.dilation, self.groups)


        class Conv1d(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.conv = torch.nn.Conv1d(*args)

            def forward(self, x):
                return self.conv(x)

        conv1d_input = torch.rand(1, 3, 224)
        conv1d_weight = torch.rand(3, 3, 3)
        conv1d_module_args = (3, 3, 3)

        class FunctionalConv2d(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = torch.nn.Parameter(weight)
                self.stride = (1, 1)
                self.padding = (0, 0)
                self.dilation = (1, 1)
                self.groups = 1

            def forward(self, x):
                return F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation, self.groups)

        class Conv2d(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.conv = torch.nn.Conv2d(*args)

            def forward(self, x):
                return self.conv(x)

        conv2d_input = torch.rand(1, 3, 224, 224)
        conv2d_weight = torch.rand(3, 3, 3, 3)
        conv2d_module_args = (3, 3, 3)

        class FunctionalConv3d(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = torch.nn.Parameter(weight)
                self.stride = (1, 1, 1)
                self.padding = (0, 0, 0)
                self.dilation = (1, 1, 1)
                self.groups = 1

            def forward(self, x):
                return F.conv3d(
                    x,
                    self.weight,
                    None,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )

        class Conv3d(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.conv = torch.nn.Conv3d(*args)

            def forward(self, x):
                return self.conv(x)

        conv3d_input = torch.rand(1, 3, 32, 224, 224)
        conv3d_weight = torch.rand(3, 3, 3, 3, 3)
        conv3d_module_args = (3, 3, 3)

        class Linear(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = torch.nn.Parameter(weight)

            def forward(self, x):
                return F.linear(x, self.weight)

        linear_input = torch.rand(8, 5)
        linear_weight = torch.rand(10, 5)

        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 10)

            def forward(self, x):
                return self.linear(x)

        linear_module_input = torch.rand(8, 5)

        # is_dynamic, ModuleClass, module_constructor_inputs,
        # inputs, quantized_node, weight_prepack_node
        tests = [
            (
                False,
                FunctionalConv1d,
                (conv1d_weight,),
                (conv1d_input,),
                ns.call_function(torch.nn.functional.conv1d if is_reference else torch.ops.quantized.conv1d) ,
                ns.call_function(torch.ops.quantized.conv1d_prepack),
            ),
            (
                False,
                FunctionalConv2d,
                (conv2d_weight,),
                (conv2d_input,),
                ns.call_function(torch.nn.functional.conv2d if is_reference else torch.ops.quantized.conv2d),
                ns.call_function(torch.ops.quantized.conv2d_prepack),
            ),
            (
                False,
                FunctionalConv3d,
                (conv3d_weight,),
                (conv3d_input,),
                ns.call_function(torch.nn.functional.conv3d if is_reference else torch.ops.quantized.conv3d),
                ns.call_function(torch.ops.quantized.conv3d_prepack),
            ),
            (
                False,
                Conv1d,
                conv1d_module_args,
                (conv1d_input,),
                ns.call_module(nnqr.Conv1d if is_reference else nnq.Conv1d),
                None
            ),
            (
                False,
                Conv2d,
                conv2d_module_args,
                (conv2d_input,),
                ns.call_module(nnqr.Conv2d if is_reference else nnq.Conv2d),
                None
            ),
            (
                False,
                Conv3d,
                conv3d_module_args,
                (conv3d_input,),
                ns.call_module(nnqr.Conv3d if is_reference else nnq.Conv3d),
                None
            ),
            (
                True,
                Linear,
                (linear_weight,),
                (linear_input,),
                None if is_reference else ns.call_function(torch.ops.quantized.linear_dynamic),
                ns.call_function(torch.ops.quantized.linear_prepack),
            ),
            (
                False,
                Linear,
                (linear_weight,),
                (linear_input,),
                ns.call_function(torch.nn.functional.linear if is_reference else torch.ops.quantized.linear),
                ns.call_function(torch.ops.quantized.linear_prepack),
            ),
            (
                True,
                LinearModule,
                (),
                (linear_module_input,),
                ns.call_module(nnqr.Linear) if is_reference else ns.call_module(nnqd.Linear),
                None,
            ),
            (
                False,
                LinearModule,
                (),
                (linear_module_input,),
                ns.call_module(nnqr.Linear if is_reference else nnq.Linear),
                None,
            ),
        ]
        return tests

    @skipIfNoFBGEMM
    def test_conv_linear_not_reference(self):
        """ Test quantizing conv and linear
        """
        tests = self._get_conv_linear_test_cases(is_reference=False)
        for (is_dynamic, ModuleClass, module_constructor_inputs,
             inputs, quantized_node, weight_prepack_node) in tests:
            quant_type = QuantType.DYNAMIC if is_dynamic else QuantType.STATIC
            node_occurrence = {}
            if weight_prepack_node:
                node_occurrence[weight_prepack_node] = 0
            self.checkGraphModeFxOp(
                ModuleClass(*module_constructor_inputs),
                inputs, quant_type,
                expected_node=quantized_node,
                expected_node_occurrence=node_occurrence,
                is_reference=False)

    @skipIfNoFBGEMM
    def test_conv_linear_reference(self):
        """ Test quantizing functional conv and linear with reference option
        """
        tests = self._get_conv_linear_test_cases(is_reference=True)

        def _get_keys(prefix, is_dynamic):
            all_keys = [prefix + "." + k for k in ["weight_qscheme", "weight_dtype"]]
            if not is_dynamic:
                all_keys.extend([prefix + "." + k for k in ["weight_scale", "weight_zero_point"]])
            return all_keys

        for (is_dynamic, ModuleClass, module_constructor_inputs,
             inputs, quantized_node, weight_prepack_node) in tests:
            quant_type = QuantType.DYNAMIC if is_dynamic else QuantType.STATIC
            node_occurrence = {}
            if weight_prepack_node:
                node_occurrence[weight_prepack_node] = 0
            result_dict = self.checkGraphModeFxOp(
                ModuleClass(*module_constructor_inputs),
                inputs, quant_type,
                expected_node=quantized_node,
                expected_node_occurrence=node_occurrence,
                is_reference=True)
            qr = result_dict["quantized_reference"]

            def checkWeightQParams(model):
                for module_name in ("linear", "conv"):
                    if hasattr(model, module_name):
                        self.assertTrue(hasattr(qr.get_submodule(module_name), "weight_qscheme"))
                        self.assertTrue(hasattr(qr.get_submodule(module_name), "weight_scale"))
                        self.assertTrue(hasattr(qr.get_submodule(module_name), "weight_zero_point"))
                        self.assertTrue("Reference" in qr.get_submodule(module_name)._get_name())

            def checkSerDeser(model, is_dynamic):
                for module_name in ("linear", "conv"):
                    if hasattr(model, module_name):
                        # make sure seralization works
                        state_dict = copy.deepcopy(model.state_dict())
                        all_keys = _get_keys(module_name, is_dynamic)
                        for key in all_keys:
                            self.assertTrue(key in state_dict)
                        # check load_state_dict restores states
                        module = getattr(model, module_name)
                        prev_scale = module.weight_scale
                        module.weight_scale = None
                        model.load_state_dict(state_dict)
                        module = getattr(model, module_name)
                        self.assertTrue(torch.equal(prev_scale, module.weight_scale))


            checkWeightQParams(qr)
            qr = copy.deepcopy(qr)
            # make sure the qparams are preserved after copy
            checkWeightQParams(qr)

            checkSerDeser(qr, is_dynamic)

    @skipIfNoFBGEMM
    def test_dynamic_quant_weight_observer(self):
        ''' Test that weight observer is run in convert step
        '''

        class M(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = torch.nn.Parameter(weight)

            def forward(self, x):
                return F.linear(x, self.weight)

        m = M(torch.rand(1, 1)).eval()
        qconfig = default_dynamic_qconfig
        qconfig_dict = {'': qconfig}
        example_inputs = (torch.rand(1, 1),)
        prepared = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        quantized = convert_to_reference_fx(prepared)
        qparams = (quantized._scale_0, quantized._zero_point_0)
        weight_obs = qconfig.weight()
        weight_obs(quantized.weight)
        # Get the actual value to avoid tensor size mismatch error, torch.Size([]) vs torch.Size([1])
        ref_qparams = (weight_obs.calculate_qparams()[0].item(), weight_obs.calculate_qparams()[1].item())
        self.assertEqual(qparams, ref_qparams)

    def test_conv_bn_relu(self):
        """ Tests fusion and quantization for "Conv - Bn" and "Conv - Bn - ReLU"
        """
        convs = {
            1: nn.Conv1d,
            2: nn.Conv2d,
            3: nn.Conv3d,
        }
        bns = {
            1: nn.BatchNorm1d,
            2: nn.BatchNorm2d,
            3: nn.BatchNorm3d,
        }
        quantized_convs = {
            1: nnq.Conv1d,
            2: nnq.Conv2d,
            3: nnq.Conv3d,
        }
        quantized_conv_relus = {
            1: nniq.ConvReLU1d,
            2: nniq.ConvReLU2d,
            3: nniq.ConvReLU3d,
        }

        class M(torch.nn.Module):
            def __init__(self, dim, has_relu):
                super().__init__()
                self.conv = convs[dim](3, 3, 3)
                self.bn = bns[dim](3)
                self.relu = nn.ReLU() if has_relu else nn.Identity()
                self.has_relu = has_relu
                self.quant = QuantStub()
                self.dequant = DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv(x)
                x = self.bn(x)
                if self.has_relu:
                    x = self.relu(x)
                x = self.dequant(x)
                return x

        options = itertools.product([1, 2, 3], [True, False], self.static_quant_types)
        for dim, has_relu, quant_type in options:
            expected_node = ns.call_module(
                quantized_conv_relus[dim] if has_relu
                else quantized_convs[dim])
            m = M(dim, has_relu)
            m_eager = copy.deepcopy(m)
            result_dict = self.checkGraphModeFxOp(
                m,
                self.img_data_dict[dim],
                quant_type,
                expected_node=expected_node,
            )
            result = result_dict["quantized_output"]

            # check numerics
            qengine = torch.backends.quantized.engine
            if quant_type == QuantType.STATIC:
                m_eager.eval()
                qconfig = get_default_qconfig(qengine)
                prepare_fn = prepare
                is_qat = False
            else:
                m_eager.train()
                qconfig = get_default_qat_qconfig(qengine)
                prepare_fn = prepare_qat
                is_qat = True

            fuse_list = ["conv", "bn"]
            if has_relu:
                fuse_list.append("relu")
            if is_qat:
                fuse_modules_qat(m_eager, fuse_list, inplace=True)
            else:
                fuse_modules(m_eager, fuse_list, inplace=True)
            m_eager.qconfig = qconfig
            m_eager = prepare_fn(m_eager)
            prepared_fx = result_dict["prepared"]

            m_eager(*self.img_data_dict[dim][0])
            m_eager = convert(m_eager)
            result_eager = m_eager(*self.img_data_dict[dim][0])
            self.assertEqual(result, result_eager)

    def test_linear_bn(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)
                self.bn = nn.BatchNorm1d(4)
                self.quant = QuantStub()
                self.dequant = DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.linear(x)
                x = self.bn(x)
                x = self.dequant(x)
                return x

        data = (torch.randn(4, 4),)
        for quant_type in self.static_quant_types:
            expected_node = ns.call_module(nnq.Linear)
            m = M()
            m_eager = copy.deepcopy(m)
            result_dict = self.checkGraphModeFxOp(m, data, quant_type, expected_node=expected_node)
            result = result_dict["quantized_output"]

            # check numerics vs eager mode
            fuse_list = ["linear", "bn"]
            qengine = torch.backends.quantized.engine
            if quant_type == QuantType.STATIC:
                m_eager.eval()
                qconfig = get_default_qconfig(qengine)
                prepare_fn = prepare
                fuse_modules(m_eager, fuse_list, inplace=True)
            else:
                m_eager.train()
                qconfig = get_default_qat_qconfig(qengine)
                prepare_fn = prepare_qat
                fuse_modules_qat(m_eager, fuse_list, inplace=True)
            m_eager.qconfig = qconfig
            m_eager = prepare_fn(m_eager)
            m_eager(*data)
            m_eager = convert(m_eager)
            result_eager = m_eager(*data)
            self.assertEqual(result, result_eager)

    @skipIfNoFBGEMM
    def test_dynamic_quant_fp16(self):
        with override_quantized_engine('fbgemm'):
            class Linear(torch.nn.Module):
                def __init__(self, weight):
                    super().__init__()
                    self.weight = torch.nn.Parameter(weight)

                def forward(self, x):
                    return F.linear(x, self.weight)

            linear_input = torch.rand(8, 5)
            linear_weight = torch.rand(10, 5)

            class LinearModule(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(5, 10)

                def forward(self, x):
                    return self.linear(x)

            linear_module_input = torch.rand(8, 5)

            tests = [
                (Linear, (linear_weight,), (linear_input,),
                 ns.call_function(torch.ops.quantized.linear_dynamic_fp16),
                 ns.call_function(torch.ops.quantized.linear_prepack_fp16)),
                (LinearModule, (), (linear_module_input,),
                 ns.call_module(nnqd.Linear),
                 None),
            ]
            for (ModuleClass, module_constructor_inputs,
                 inputs, quantized_node, weight_prepack_node) in tests:
                for is_reference in [True, False]:
                    node_occurrence = {}
                    if weight_prepack_node:
                        node_occurrence[weight_prepack_node] = 0
                    m = ModuleClass(*module_constructor_inputs).eval()
                    qconfig_dict = {"": float16_dynamic_qconfig}
                    m = prepare_fx(m, qconfig_dict, example_inputs=inputs)
                    convert_fn = convert_to_reference_fx if is_reference else convert_fx
                    m = convert_fn(m)
                    self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)



    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @override_qengines
    def test_qat_prepare_device_affinity(self):
        """
        Tests that FX QAT prepare pass respects device affinity
        """
        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.conv = nn.Conv2d(1, 1, 1)
                self.bn = nn.BatchNorm2d(1)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                return x

        model = Model()
        qengine = torch.backends.quantized.engine
        qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig(qengine)}
        device = torch.device('cuda:0')
        model.to(device)

        example_inputs = (torch.randn(4, 1, 4, 4, device=device),)
        # QAT prepare
        model = prepare_qat_fx(model, qconfig_dict, example_inputs=example_inputs)

        # ensure that running an input on CUDA works without any needed changes
        model(*example_inputs)

        # ensure all buffers and parameters are on the device we expect
        model_devices = {p.device for p in model.parameters()} | \
            {p.device for p in model.buffers()}
        self.assertEqual(len(model_devices), 1)
        model_device = next(iter(model_devices))
        self.assertEqual(model_device, device)

    @skipIfNoFBGEMM
    def test_dict_output(self):
        """ Make sure quantization runs for models with dictionary output
        """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                return {"output": self.conv(x["input"])}

        example_inputs = ({"input": torch.randn(1, 1, 1, 1)},)
        m = M().eval()
        qconfig_dict = {"": default_qconfig}
        m = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        m(*example_inputs)
        m = convert_fx(m)
        m(*example_inputs)

    @override_qengines
    def test_attention(self):
        """ Make sure quantization runs for a corner case in attention module
        """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv(x)
                q, k, v = x.chunk(3, dim=0)
                q = q.contiguous().view(-1, 1).transpose(0, 1)
                k = k.contiguous().view(-1, 1).transpose(0, 1)
                v = v.contiguous().view(-1, 1).transpose(0, 1)
                torch._assert(
                    k.size(1) == 1, "key size should be equal to 1"
                )
                r = torch.mm(k, v)
                return q * k + r

        example_inputs = (torch.randn(3, 1, 1, 1),)
        m = M().eval()
        qconfig_dict = {
            "": None,
            "object_type": [
                (nn.Conv2d, default_qconfig),
            ]
        }
        # make sure it runs
        m = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        m(*example_inputs)
        m = convert_fx(m)
        m(*example_inputs)

    def _test_standalone_module(
            self,
            interface_config,
            prepare_count_check,
            standalone_prepare_count_check,
            convert_count_check,
            standalone_convert_count_check):
        """ Test standalone module with different quantized input/quantized output
        configurations
        """
        class StandaloneModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                return self.conv(x)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.standalone = StandaloneModule()

            def forward(self, x):
                x = self.conv(x)
                x = self.standalone(x)
                return x

        class RefM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 1, 1)
                self.conv2 = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                return x

        example_inputs = (torch.randn(1, 1, 1, 1),)
        # instantiate M and RefM and align the parameters
        original_m = M().eval()
        original_ref_m = RefM().eval()
        original_ref_m.conv1.weight = torch.nn.Parameter(original_m.conv.weight.detach())
        original_ref_m.conv1.bias = torch.nn.Parameter(original_m.conv.bias.detach())
        original_ref_m.conv2.weight = torch.nn.Parameter(original_m.standalone.conv.weight.detach())
        original_ref_m.conv2.bias = torch.nn.Parameter(original_m.standalone.conv.bias.detach())

        for is_name in [True, False]:
            sm_example_inputs = example_inputs
            if is_name:
                prepare_config = {
                    "standalone_module_name": [("standalone", None, sm_example_inputs, interface_config, None)]
                }
            else:
                prepare_config = {
                    "standalone_module_class": [(StandaloneModule, None, sm_example_inputs, interface_config, None)]
                }

            original_m_copy = copy.deepcopy(original_m)
            original_ref_m_copy = copy.deepcopy(original_ref_m)

            qconfig_dict = {"": default_qconfig}
            # check prepared model
            m = prepare_fx(
                original_m_copy,
                qconfig_dict,
                example_inputs=example_inputs,
                prepare_custom_config=prepare_config)
            # calibration
            m(*example_inputs)
            self.checkGraphModuleNodes(m, expected_node_occurrence=prepare_count_check)
            self.checkGraphModuleNodes(m.standalone, expected_node_occurrence=standalone_prepare_count_check)

            # check converted/quantized model
            m = convert_fx(m)
            self.checkGraphModuleNodes(m, expected_node_occurrence=convert_count_check)
            self.checkGraphModuleNodes(m.standalone, expected_node_occurrence=standalone_convert_count_check)
            res = m(*example_inputs)

            # quantize the reference model
            ref_m = prepare_fx(
                original_ref_m_copy,
                qconfig_dict,
                example_inputs=example_inputs,
            )
            ref_m(*example_inputs)
            ref_m = convert_fx(ref_m)
            ref_res = ref_m(*example_inputs)
            self.assertEqual(res, ref_res)

    def test_standalone_module_float_interface(self):
        float_interface_config = {
            "input_quantized_idxs": [],  # float input
            "output_quantized_idxs": [],  # float output
        }
        interface_config = float_interface_config
        # input and output of first conv, observer for standalone module
        # will be inserted in the standalone module itself
        prepare_count_check = {
            ns.call_module(torch.ao.quantization.MinMaxObserver): 2
        }
        # for input and output of conv in the standalone module
        standalone_prepare_count_check = {
            ns.call_module(torch.ao.quantization.MinMaxObserver): 2
        }
        convert_count_check = {
            ns.call_function(torch.quantize_per_tensor) : 1,
            ns.call_module(nnq.Conv2d) : 1,
            ns.call_method("dequantize") : 1,
        }
        standalone_convert_count_check = {
            # standalone module will take float as input and output
            # so we'll see quantize and dequantize in the modoule
            ns.call_function(torch.quantize_per_tensor) : 1,
            ns.call_module(nnq.Conv2d): 1,
            ns.call_method("dequantize") : 1,
        }
        self._test_standalone_module(
            interface_config,
            prepare_count_check,
            standalone_prepare_count_check,
            convert_count_check,
            standalone_convert_count_check)

    def test_standalone_module_quantized_interface(self):
        quantized_interface_config = {
            "input_quantized_idxs": [0],  # quantized input
            "output_quantized_idxs": [0],  # quantized output
        }
        interface_config = quantized_interface_config
        # observer for input and output of first conv
        prepare_count_check = {
            ns.call_module(torch.ao.quantization.MinMaxObserver): 2
        }
        # for output of conv in the standalone module
        standalone_prepare_count_check = {
            ns.call_module(torch.ao.quantization.MinMaxObserver): 1
        }
        convert_count_check = {
            # quantizing input for conv
            ns.call_function(torch.quantize_per_tensor) : 1,
            ns.call_module(nnq.Conv2d) : 1,
            # dequantizing output of standalone module
            ns.call_method("dequantize") : 1,
        }
        standalone_convert_count_check = {
            # quantization of input happens in parent module
            # quantization of output happens in the quantized conv module
            ns.call_function(torch.quantize_per_tensor) : 0,
            ns.call_module(nnq.Conv2d): 1,
            # dequantization for output happens in parent module
            ns.call_method("dequantize") : 0,
        }
        self._test_standalone_module(
            interface_config,
            prepare_count_check,
            standalone_prepare_count_check,
            convert_count_check,
            standalone_convert_count_check)

    @skipIfNoFBGEMM
    def test_qconfig_none(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.conv2 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                return x

        m = M().eval()
        qconfig_dict = {"": default_qconfig,
                        "module_name": [("conv2", None)]}
        example_inputs = (torch.randn(1, 1, 1, 1),)
        m = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        m(*example_inputs)
        m = convert_fx(m)
        m(*example_inputs)
        # first conv is quantized, second conv is not quantized
        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Conv2d),
            ns.call_method("dequantize"),
            ns.call_module(nn.Conv2d),
        ]
        self.checkGraphModuleNodes(m, expected_node_list=node_list)

    def test_qconfig_module_type(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.conv2 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                return x

        m = M().eval()
        qconfig_dict = {"object_type": [(torch.nn.Conv2d, default_qconfig)]}
        example_inputs = (torch.randn(1, 1, 1, 1),)
        m = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        m(*example_inputs)
        m = convert_fx(m)
        m(*example_inputs)
        # first conv is quantized, second conv is not quantized
        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Conv2d),
            ns.call_module(nnq.Conv2d),
            ns.call_method("dequantize"),
        ]
        self.checkGraphModuleNodes(m, expected_node_list=node_list)

    def test_qconfig_qat_module_type(self):
        class LinearRelu(nn.Sequential):
            def __init__(self):
                super().__init__(
                    nn.Linear(5, 5),
                    nn.ReLU(),
                )

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin_relu = LinearRelu()
                self.linear = nn.Linear(5, 5)

            def forward(self, x):
                x = self.lin_relu(x)
                x = self.linear(x)
                return x

        model = M().train()

        qconfig_dict = {
            "": None,
            "object_type": [
                (torch.nn.Linear, default_qat_qconfig),
                (torch.nn.ReLU, default_qat_qconfig),
            ],
        }
        example_inputs = (torch.rand(5, 5),)
        m = prepare_qat_fx(model, qconfig_dict, example_inputs=example_inputs)
        m(*example_inputs)
        m = convert_fx(m)
        m(*example_inputs)
        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nniq.LinearReLU),
            ns.call_module(nnq.Linear),
            ns.call_method("dequantize"),
        ]
        self.checkGraphModuleNodes(m, expected_node_list=node_list)

    def test_qconfig_function(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()

            def forward(self, x, y):
                return x + y

        m = M().eval()
        qconfig_dict = {"object_type": [(operator.add, default_qconfig)]}
        data = torch.randn(1, 1, 1, 1)
        example_inputs = (data, data)
        m = prepare_fx(m, qconfig_dict, example_inputs)
        m(*example_inputs)
        m = convert_fx(m)
        m(*example_inputs)
        # first conv is quantized, second conv is not quantized
        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_function(torch.ops.quantized.add),
            ns.call_method("dequantize"),
        ]
        self.checkGraphModuleNodes(m, expected_node_list=node_list)

    def test_qconfig_module_name_regex(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.conv2 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                return x

        m = M().eval()
        qconfig_dict = {"module_name_regex": [("conv*", default_qconfig)]}
        example_inputs = (torch.randn(1, 1, 1, 1),)
        m = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        m(*example_inputs)
        m = convert_fx(m)
        m(*example_inputs)
        # first conv is quantized, second conv is not quantized
        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Conv2d),
            ns.call_module(nnq.Conv2d),
            ns.call_method("dequantize"),
        ]
        self.checkGraphModuleNodes(m, expected_node_list=node_list)

    def test_qconfig_precedence(self):
        for device in get_supported_device_types():
            class M(torch.nn.Module):
                def __init__(self):
                    super(M, self).__init__()
                    self.linear = nn.Linear(1, 1)
                    self.conv = nn.Conv2d(1, 1, 1)
                    self.module_conv1 = nn.Conv2d(1, 1, 1)
                    self.module_conv2 = nn.Conv2d(1, 1, 1)

                def forward(self, x):
                    # global
                    x = self.linear(x)
                    # global + object_type --> object_type
                    x = self.conv(x)
                    # global + object_type + module_name_regex --> module_name_regex
                    x = self.module_conv1(x)
                    # global + object_type + module_name_regex + module_name --> module_name
                    x = self.module_conv2(x)
                    return x

            m = M().to(device).eval()

            global_qconfig = default_qconfig
            object_type_qconfig = default_dynamic_qconfig
            module_name_regex_qconfig = float16_dynamic_qconfig
            module_name_qconfig = default_qat_qconfig
            qconfig_dict = {
                "": global_qconfig,
                "object_type": [(nn.Conv2d, object_type_qconfig)],
                "module_name_regex": [("module_conv*", module_name_regex_qconfig)],
                "module_name": [("module_conv2", module_name_qconfig)]}
            m_prep = prepare_fx(m, qconfig_dict, example_inputs=(torch.randn(1, 1),))
            self.assertEqual(m_prep.linear.qconfig.activation.p.func, global_qconfig.activation.p.func)
            self.assertEqual(m_prep.linear.qconfig.weight.p.func, global_qconfig.weight.p.func)
            self.assertEqual(m_prep.conv.qconfig.activation.p.func, object_type_qconfig.activation.p.func)
            self.assertEqual(m_prep.conv.qconfig.weight.p.func, object_type_qconfig.weight.p.func)
            self.assertEqual(m_prep.module_conv1.qconfig.activation.p.func, module_name_regex_qconfig.activation.p.func)
            self.assertEqual(m_prep.module_conv1.qconfig.weight.p.func, module_name_regex_qconfig.weight.p.func)
            self.assertEqual(m_prep.module_conv2.qconfig.activation.p.func, module_name_qconfig.activation.p.func)
            self.assertEqual(m_prep.module_conv2.qconfig.weight.p.func, module_name_qconfig.weight.p.func)

    def test_qconfig_module_name_object_type_order(self):
        class M1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(1, 1)
                self.fc2 = nn.Linear(1, 1)

            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                x = torch.add(x, x)
                x = torch.add(x, x)
                return x

        class M2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(1, 1)
                self.fc2 = nn.Linear(1, 1)
                self.m1 = M1()

            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                x = torch.add(x, x)
                x = torch.add(x, x)
                x = self.m1(x)
                return x

        class M3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(1, 1)
                self.fc2 = nn.Linear(1, 1)
                self.m2 = M2()

            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                x = torch.add(x, x)
                x = torch.add(x, x)
                x = self.m2(x)
                return x

        m = M3().eval()
        qconfig_dict = {
            "module_name_object_type_order": [
                # test various FQNs: global, single child, multiple children
                ("", nn.Linear, 0, torch.ao.quantization.default_qconfig),
                ("", torch.add, 0, torch.ao.quantization.default_qconfig),
                ("m2", nn.Linear, 1, torch.ao.quantization.default_qconfig),
                ("m2", torch.add, 1, torch.ao.quantization.default_qconfig),
                ("m2.m1", nn.Linear, 0, torch.ao.quantization.default_qconfig),
                ("m2.m1", torch.add, 0, torch.ao.quantization.default_qconfig),
            ],
        }
        example_inputs = (torch.randn(1, 1, 1, 1),)
        m = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        m(*example_inputs)
        m = convert_fx(m)
        m(*example_inputs)

        node_list = [
            # m3
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Linear),
            ns.call_method("dequantize"),
            ns.call_module(nn.Linear),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_function(torch.ops.quantized.add),
            ns.call_method("dequantize"),
            ns.call_function(torch.add),
            # m2
            ns.call_module(nn.Linear),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Linear),
            ns.call_method("dequantize"),
            ns.call_function(torch.add),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_function(torch.ops.quantized.add),
            # m1
            ns.call_module(nnq.Linear),
            ns.call_method("dequantize"),
            ns.call_module(nn.Linear),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_function(torch.ops.quantized.add),
            ns.call_method("dequantize"),
            ns.call_function(torch.add),
        ]
        self.checkGraphModuleNodes(m, expected_node_list=node_list)

        # test that function order overrides global qconfig
        class M4(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(1, 1)
                self.fc2 = nn.Linear(1, 1)

            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                x = torch.add(x, x)
                x = torch.add(x, x)
                return x

        m = M4().eval()
        qconfig_dict = {
            "": torch.ao.quantization.default_qconfig,
            "module_name_object_type_order": [
                ("", nn.Linear, 1, None),
                ("", torch.add, 1, None),
            ],
        }
        example_inputs = (torch.randn(1, 1, 1, 1),)
        m = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        m(*example_inputs)
        m = convert_fx(m)
        m(*example_inputs)

        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Linear),
            ns.call_method("dequantize"),
            ns.call_module(nn.Linear),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_function(torch.ops.quantized.add),
            ns.call_method("dequantize"),
            ns.call_function(torch.add),
        ]
        self.checkGraphModuleNodes(m, expected_node_list=node_list)


    def test_qconfig_dict_with_fused_modules(self):
        class LinearReLUModel(torch.nn.Module):
            def __init__(self, relu):
                super(LinearReLUModel, self).__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.relu = relu

            def forward(self, x):
                x = self.linear(x)
                x = self.relu(x)
                return x

        class ConvReLUModel(torch.nn.Module):
            def __init__(self, relu):
                super(ConvReLUModel, self).__init__()
                self.conv = torch.nn.Conv1d(3, 3, 3)
                self.relu = relu

            def forward(self, x):
                x = self.conv(x)
                x = self.relu(x)
                return x

        class ConvBnReLUModel(torch.nn.Module):
            def __init__(self, relu):
                super(ConvBnReLUModel, self).__init__()
                self.conv = torch.nn.Conv1d(3, 3, 3)
                self.bn = torch.nn.BatchNorm1d(3)
                self.relu = relu

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                return x

        for model in [LinearReLUModel, ConvReLUModel, ConvBnReLUModel]:
            for relu in [torch.nn.ReLU(), torch.nn.functional.relu, torch.relu]:
                m = model(relu).eval()
                qconfig_dict = torch.ao.quantization.get_default_qconfig_mapping("fbgemm")
                # should not crash as in https://github.com/pytorch/pytorch/issues/75825
                prepare_fx(m, qconfig_dict, example_inputs=(torch.randn(1, 3, 3, 3),))

    # TODO: move QConfigMapping tests to test/quantization/core
    def test_qconfig_mapping_set_global(self):
        qconfig = get_default_qconfig()
        qconfig_mapping = QConfigMapping()
        self.assertEqual(qconfig_mapping.global_qconfig, None)
        qconfig_mapping.set_global(qconfig)
        self.assertEqual(qconfig_mapping.global_qconfig, qconfig)

    def test_qconfig_mapping_set_object_type(self):
        qconfig1 = get_default_qconfig()
        qconfig2 = get_default_qconfig()
        qconfig3 = get_default_qconfig()
        self.assertNotEqual(qconfig1, qconfig2)
        self.assertNotEqual(qconfig1, qconfig3)
        qconfig_mapping = QConfigMapping()
        self.assertEqual(len(qconfig_mapping.object_type_qconfigs), 0)
        # Insert some entries
        qconfig_mapping.set_object_type(torch.nn.Linear, qconfig1)
        qconfig_mapping.set_object_type(torch.nn.ReLU, qconfig2)
        self.assertEqual(len(qconfig_mapping.object_type_qconfigs), 2)
        self.assertEqual(qconfig_mapping.object_type_qconfigs[torch.nn.Linear], qconfig1)
        self.assertEqual(qconfig_mapping.object_type_qconfigs[torch.nn.ReLU], qconfig2)
        # Override existing key
        qconfig_mapping.set_object_type(torch.nn.Linear, qconfig3)
        self.assertEqual(qconfig_mapping.object_type_qconfigs[torch.nn.Linear], qconfig3)
        self.assertEqual(qconfig_mapping.object_type_qconfigs[torch.nn.ReLU], qconfig2)
        self.assertEqual(_get_object_type_qconfig(qconfig_mapping, torch.nn.Linear, None), qconfig3)
        self.assertEqual(_get_object_type_qconfig(qconfig_mapping, torch.nn.ReLU, None), qconfig2)
        self.assertEqual(_get_object_type_qconfig(qconfig_mapping, "nomatch", None), None)

    def test_qconfig_mapping_set_module_name_regex(self):
        qconfig1 = get_default_qconfig()
        qconfig2 = get_default_qconfig()
        qconfig3 = get_default_qconfig()
        self.assertNotEqual(qconfig1, qconfig2)
        self.assertNotEqual(qconfig1, qconfig3)
        qconfig_mapping = QConfigMapping()
        self.assertEqual(len(qconfig_mapping.module_name_regex_qconfigs), 0)
        # Insert some entries
        qconfig_mapping.set_module_name_regex("foo.*bar", qconfig1)
        qconfig_mapping.set_module_name_regex("foo.*", qconfig2)
        self.assertEqual(len(qconfig_mapping.module_name_regex_qconfigs), 2)
        self.assertEqual(qconfig_mapping.module_name_regex_qconfigs["foo.*bar"], qconfig1)
        self.assertEqual(qconfig_mapping.module_name_regex_qconfigs["foo.*"], qconfig2)
        # Override existing key
        qconfig_mapping.set_module_name_regex("foo.*bar", qconfig3)
        self.assertEqual(qconfig_mapping.module_name_regex_qconfigs["foo.*bar"], qconfig3)
        self.assertEqual(qconfig_mapping.module_name_regex_qconfigs["foo.*"], qconfig2)
        self.assertEqual(_get_module_name_regex_qconfig(qconfig_mapping, "foo123bar", None), qconfig3)
        self.assertEqual(_get_module_name_regex_qconfig(qconfig_mapping, "foobar", None), qconfig3)
        self.assertEqual(_get_module_name_regex_qconfig(qconfig_mapping, "foobaz", None), qconfig2)
        self.assertEqual(_get_module_name_regex_qconfig(qconfig_mapping, "foo", None), qconfig2)
        self.assertEqual(_get_module_name_regex_qconfig(qconfig_mapping, "nomatch", None), None)

    def test_qconfig_mapping_set_module_name(self):
        qconfig1 = get_default_qconfig()
        qconfig2 = get_default_qconfig()
        qconfig3 = get_default_qconfig()
        self.assertNotEqual(qconfig1, qconfig2)
        self.assertNotEqual(qconfig1, qconfig3)
        qconfig_mapping = QConfigMapping()
        self.assertEqual(len(qconfig_mapping.module_name_qconfigs), 0)
        # Insert some entries
        qconfig_mapping.set_module_name("mod1", qconfig1)
        qconfig_mapping.set_module_name("mod2", qconfig2)
        self.assertEqual(len(qconfig_mapping.module_name_qconfigs), 2)
        self.assertEqual(qconfig_mapping.module_name_qconfigs["mod1"], qconfig1)
        self.assertEqual(qconfig_mapping.module_name_qconfigs["mod2"], qconfig2)
        # Override existing key
        qconfig_mapping.set_module_name("mod1", qconfig3)
        self.assertEqual(qconfig_mapping.module_name_qconfigs["mod1"], qconfig3)
        self.assertEqual(qconfig_mapping.module_name_qconfigs["mod2"], qconfig2)
        self.assertEqual(_get_module_name_qconfig(qconfig_mapping, "mod1", None), qconfig3)
        self.assertEqual(_get_module_name_qconfig(qconfig_mapping, "mod2", None), qconfig2)
        self.assertEqual(_get_module_name_qconfig(qconfig_mapping, "nomatch", None), None)

    def test_qconfig_mapping_set_module_name_object_type_order(self):
        qconfig1 = get_default_qconfig()
        qconfig2 = get_default_qconfig()
        qconfig3 = get_default_qconfig()
        self.assertNotEqual(qconfig1, qconfig2)
        self.assertNotEqual(qconfig1, qconfig3)
        qconfig_mapping = QConfigMapping()
        self.assertEqual(len(qconfig_mapping.module_name_object_type_order_qconfigs), 0)
        # Insert some entries
        qconfig_mapping.set_module_name_object_type_order("mod1", torch.nn.Linear, 0, qconfig1)
        qconfig_mapping.set_module_name_object_type_order("mod2", torch.nn.ReLU, 1, qconfig2)
        self.assertEqual(len(qconfig_mapping.module_name_object_type_order_qconfigs), 2)
        key1 = ("mod1", torch.nn.Linear, 0)
        key2 = ("mod2", torch.nn.ReLU, 1)
        self.assertEqual(list(qconfig_mapping.module_name_object_type_order_qconfigs)[0], key1)
        self.assertEqual(list(qconfig_mapping.module_name_object_type_order_qconfigs)[1], key2)
        self.assertEqual(qconfig_mapping.module_name_object_type_order_qconfigs[key1], qconfig1)
        self.assertEqual(qconfig_mapping.module_name_object_type_order_qconfigs[key2], qconfig2)
        self.assertEqual(maybe_adjust_qconfig_for_module_name_object_type_order(
                         qconfig_mapping, "mod1", torch.nn.Linear, 0, None), qconfig1)
        self.assertEqual(maybe_adjust_qconfig_for_module_name_object_type_order(
                         qconfig_mapping, "mod2", torch.nn.ReLU, 1, None), qconfig2)
        # Override existing key
        qconfig_mapping.set_module_name_object_type_order("mod1", torch.nn.Linear, 0, qconfig3)
        self.assertEqual(len(qconfig_mapping.module_name_object_type_order_qconfigs), 2)
        self.assertEqual(list(qconfig_mapping.module_name_object_type_order_qconfigs)[0], key1)
        self.assertEqual(list(qconfig_mapping.module_name_object_type_order_qconfigs)[1], key2)
        self.assertEqual(qconfig_mapping.module_name_object_type_order_qconfigs[key1], qconfig3)
        self.assertEqual(qconfig_mapping.module_name_object_type_order_qconfigs[key2], qconfig2)
        self.assertEqual(maybe_adjust_qconfig_for_module_name_object_type_order(
                         qconfig_mapping, "mod1", torch.nn.Linear, 0, None), qconfig3)
        self.assertEqual(maybe_adjust_qconfig_for_module_name_object_type_order(
                         qconfig_mapping, "mod2", torch.nn.ReLU, 1, None), qconfig2)
        # No match
        self.assertEqual(maybe_adjust_qconfig_for_module_name_object_type_order(
                         qconfig_mapping, "mod123", torch.nn.Linear, 0, None), None)
        self.assertEqual(maybe_adjust_qconfig_for_module_name_object_type_order(
                         qconfig_mapping, "mod1", torch.nn.Linear, 35, None), None)
        self.assertEqual(maybe_adjust_qconfig_for_module_name_object_type_order(
                         qconfig_mapping, "mod2", torch.nn.Conv2d, 1, None), None)

    def _get_qconfig_dict_for_qconfig_mapping_test(self, global_qconfig, qconfig1, qconfig2):
        """
        Return a dummy qconfig_dict to test QConfigMapping's to_dict and from_dict methods.
        """
        return {
            _GLOBAL_DICT_KEY: global_qconfig,
            _OBJECT_TYPE_DICT_KEY: [
                (torch.nn.Linear, qconfig1),
                (torch.nn.ReLU, qconfig2),
            ],
            _MODULE_NAME_REGEX_DICT_KEY: [
                ("foo.*bar", qconfig1),
                ("foo.*", qconfig2),
            ],
            _MODULE_NAME_DICT_KEY: [
                ("bazbaz", qconfig1),
                ("borbor", qconfig2),
            ],
            _MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY: [
                ("bazbaz", torch.nn.Linear, 0, qconfig1),
                ("foofoo", torch.nn.ReLU, 1, qconfig2),
            ],
        }

        with self.assertRaises(ValueError) as context:
            m = prepare_fx(m, qconfig_dict, example_inputs=(torch.randn(1, 3, 3, 3),))
        self.assertTrue(
            'Expected qconfig_dict to have the following keys:' in str(context.exception)
        )
        self.assertTrue('But found \'object_typo\' instead.' in str(context.exception))

    def test_qconfig_mapping_from_dict(self):
        global_qconfig = QConfig(123, "global")
        qconfig1 = QConfig(1, "one")
        qconfig2 = QConfig(2, "two")
        qconfig_dict = self._get_qconfig_dict_for_qconfig_mapping_test(global_qconfig, qconfig1, qconfig2)
        qconfig_dict["undefined_dict_key"] = [(123, qconfig1), (234, qconfig2)]
        qconfig_mapping = QConfigMapping.from_dict(qconfig_dict)
        self.assertEqual(qconfig_mapping.global_qconfig, global_qconfig)
        self.assertEqual(qconfig_mapping.object_type_qconfigs, OrderedDict({
            torch.nn.Linear: qconfig1,
            torch.nn.ReLU: qconfig2,
        }))
        self.assertEqual(qconfig_mapping.module_name_regex_qconfigs, OrderedDict({
            "foo.*bar": qconfig1,
            "foo.*": qconfig2,
        }))
        self.assertEqual(qconfig_mapping.module_name_qconfigs, OrderedDict({
            "bazbaz": qconfig1,
            "borbor": qconfig2,
        }))
        self.assertEqual(qconfig_mapping.module_name_object_type_order_qconfigs, OrderedDict({
            ("bazbaz", torch.nn.Linear, 0): qconfig1,
            ("foofoo", torch.nn.ReLU, 1): qconfig2,
        }))

    def test_qconfig_mapping_to_dict(self):
        global_qconfig = QConfig(123, "global")
        qconfig1 = QConfig(1, "one")
        qconfig2 = QConfig(2, "two")
        qconfig_mapping = QConfigMapping().set_global(global_qconfig) \
            .set_object_type(torch.nn.Linear, qconfig1) \
            .set_object_type(torch.nn.ReLU, qconfig2) \
            .set_module_name_regex("foo.*bar", qconfig1) \
            .set_module_name_regex("foo.*", qconfig2) \
            .set_module_name("bazbaz", qconfig1) \
            .set_module_name("borbor", qconfig2) \
            .set_module_name_object_type_order("bazbaz", torch.nn.Linear, 0, qconfig1) \
            .set_module_name_object_type_order("foofoo", torch.nn.ReLU, 1, qconfig2)
        qconfig_dict = self._get_qconfig_dict_for_qconfig_mapping_test(global_qconfig, qconfig1, qconfig2)
        self.assertEqual(qconfig_mapping.to_dict(), qconfig_dict)

    # Dummy classes for PrepareCustomConfig testing

    class _DummyStandaloneModule:
        pass

    class _DummyFloatModule:
        pass

    class _DummyObservedModule:
        pass

    class _DummyQuantizedModule:
        pass

    class _DummyNonTraceableModule1:
        pass

    class _DummyNonTraceableModule2:
        pass

    def test_prepare_custom_config_set_standalone_module_name(self):
        qconfig_mapping = QConfigMapping()
        example_inputs = (torch.randn(3),)
        child_prepare_custom_config = PrepareCustomConfig()
        backend_config = BackendConfig("my_backend")
        config_entry = StandaloneModuleConfigEntry(
            qconfig_mapping, example_inputs, child_prepare_custom_config, backend_config)
        prepare_custom_config = PrepareCustomConfig()
        self.assertEqual(len(prepare_custom_config.standalone_module_names), 0)
        prepare_custom_config.set_standalone_module_name(
            "module1", qconfig_mapping, example_inputs, child_prepare_custom_config, backend_config)
        self.assertEqual(list(prepare_custom_config.standalone_module_names.keys()), ["module1"])
        self.assertEqual(prepare_custom_config.standalone_module_names["module1"], config_entry)

    def test_prepare_custom_config_set_standalone_module_class(self):
        qconfig_mapping = QConfigMapping()
        example_inputs = (torch.randn(3),)
        child_prepare_custom_config = PrepareCustomConfig()
        backend_config = BackendConfig("my_backend")
        config_entry = StandaloneModuleConfigEntry(
            qconfig_mapping, example_inputs, child_prepare_custom_config, backend_config)
        prepare_custom_config = PrepareCustomConfig()
        self.assertEqual(len(prepare_custom_config.standalone_module_classes), 0)
        prepare_custom_config.set_standalone_module_class(
            self._DummyStandaloneModule, qconfig_mapping, example_inputs, child_prepare_custom_config, backend_config)
        self.assertEqual(len(prepare_custom_config.standalone_module_classes), 1)
        self.assertTrue(self._DummyStandaloneModule in prepare_custom_config.standalone_module_classes)
        self.assertEqual(prepare_custom_config.standalone_module_classes[self._DummyStandaloneModule], config_entry)

    def test_prepare_custom_config_set_float_to_observed_mapping(self):
        prepare_custom_config = PrepareCustomConfig()
        self.assertEqual(len(prepare_custom_config.float_to_observed_mapping), 0)
        prepare_custom_config.set_float_to_observed_mapping(self._DummyFloatModule, self._DummyObservedModule, QuantType.STATIC)
        self.assertEqual(len(prepare_custom_config.float_to_observed_mapping), 1)
        self.assertEqual(list(prepare_custom_config.float_to_observed_mapping.keys()), [QuantType.STATIC])
        self.assertEqual(len(prepare_custom_config.float_to_observed_mapping[QuantType.STATIC]), 1)
        self.assertTrue(self._DummyFloatModule in prepare_custom_config.float_to_observed_mapping[QuantType.STATIC])
        self.assertEqual(prepare_custom_config.float_to_observed_mapping[QuantType.STATIC][self._DummyFloatModule],
                         self._DummyObservedModule)

    def test_prepare_custom_config_set_non_traceable_module_names(self):
        prepare_custom_config = PrepareCustomConfig()
        self.assertEqual(len(prepare_custom_config.non_traceable_module_names), 0)
        prepare_custom_config.set_non_traceable_module_names(["module1", "module2"])
        self.assertEqual(prepare_custom_config.non_traceable_module_names, ["module1", "module2"])

    def test_prepare_custom_config_set_non_traceable_module_classes(self):
        prepare_custom_config = PrepareCustomConfig()
        self.assertEqual(len(prepare_custom_config.non_traceable_module_classes), 0)
        prepare_custom_config.set_non_traceable_module_classes([self._DummyNonTraceableModule1, self._DummyNonTraceableModule2])
        self.assertEqual(prepare_custom_config.non_traceable_module_classes,
                         [self._DummyNonTraceableModule1, self._DummyNonTraceableModule2])

    def test_prepare_custom_config_set_input_quantized_indexes(self):
        prepare_custom_config = PrepareCustomConfig()
        self.assertEqual(len(prepare_custom_config.input_quantized_indexes), 0)
        prepare_custom_config.set_input_quantized_indexes([0, 1])
        self.assertEqual(prepare_custom_config.input_quantized_indexes, [0, 1])

    def test_prepare_custom_config_set_output_quantized_indexes(self):
        prepare_custom_config = PrepareCustomConfig()
        self.assertEqual(len(prepare_custom_config.output_quantized_indexes), 0)
        prepare_custom_config.set_output_quantized_indexes([0, 1])
        self.assertEqual(prepare_custom_config.output_quantized_indexes, [0, 1])

    def test_prepare_custom_config_set_preserved_attributes(self):
        prepare_custom_config = PrepareCustomConfig()
        self.assertEqual(len(prepare_custom_config.preserved_attributes), 0)
        prepare_custom_config.set_preserved_attributes(["attr1", "attr2"])
        self.assertEqual(prepare_custom_config.preserved_attributes, ["attr1", "attr2"])

    def _get_dummy_prepare_custom_config_dict(self):
        """
        Return a dummy prepare_custom_config_dict to test PrepareCustomConfig's to_dict and from_dict methods.
        """
        return {
            STANDALONE_MODULE_NAME_DICT_KEY: [(
                "module1",
                QConfigMapping(),
                (torch.randn(3),),
                PrepareCustomConfig(),
                BackendConfig("my_backend"),
            )],
            STANDALONE_MODULE_CLASS_DICT_KEY: [(
                self._DummyStandaloneModule,
                QConfigMapping(),
                (torch.randn(10),),
                PrepareCustomConfig(),
                BackendConfig("my_backend"),
            )],
            FLOAT_TO_OBSERVED_DICT_KEY: {
                "static": {
                    self._DummyFloatModule: self._DummyObservedModule
                },
            },
            NON_TRACEABLE_MODULE_NAME_DICT_KEY: ["module2", "module3"],
            NON_TRACEABLE_MODULE_CLASS_DICT_KEY: [self._DummyNonTraceableModule1, self._DummyNonTraceableModule2],
            INPUT_QUANTIZED_INDEXES_DICT_KEY: [0, 1],
            OUTPUT_QUANTIZED_INDEXES_DICT_KEY: [0, 1],
            PRESERVED_ATTRIBUTES_DICT_KEY: ["attr1", "attr2"]
        }

    def test_prepare_custom_config_from_dict(self):
        prepare_custom_config_dict = self._get_dummy_prepare_custom_config_dict()
        (sm_name, qm1, ei1, pcc1, bcd1) = prepare_custom_config_dict[STANDALONE_MODULE_NAME_DICT_KEY][0]
        (sm_class, qm2, ei2, pcc2, bcd2) = prepare_custom_config_dict[STANDALONE_MODULE_CLASS_DICT_KEY][0]
        sm_config_entry1 = StandaloneModuleConfigEntry(qm1, ei1, pcc1, bcd1)
        sm_config_entry2 = StandaloneModuleConfigEntry(qm2, ei2, pcc2, bcd2)
        prepare_custom_config = PrepareCustomConfig.from_dict(prepare_custom_config_dict)

        # Standalone modules
        self.assertEqual(len(prepare_custom_config.standalone_module_names), 1)
        self.assertTrue(sm_name in prepare_custom_config.standalone_module_names)
        self.assertEqual(prepare_custom_config.standalone_module_names[sm_name], sm_config_entry1)
        self.assertEqual(len(prepare_custom_config.standalone_module_classes), 1)
        self.assertTrue(sm_class in prepare_custom_config.standalone_module_classes)
        self.assertEqual(prepare_custom_config.standalone_module_classes[sm_class], sm_config_entry2)

        # Float to observed mapping
        self.assertEqual(len(prepare_custom_config.float_to_observed_mapping), 1)
        self.assertEqual(list(prepare_custom_config.float_to_observed_mapping.keys()), [QuantType.STATIC])
        self.assertEqual(len(prepare_custom_config.float_to_observed_mapping[QuantType.STATIC]), 1)
        self.assertTrue(self._DummyFloatModule in prepare_custom_config.float_to_observed_mapping[QuantType.STATIC])
        self.assertEqual(prepare_custom_config.float_to_observed_mapping[QuantType.STATIC][self._DummyFloatModule],
                         self._DummyObservedModule)

        # Other
        self.assertEqual(prepare_custom_config.non_traceable_module_names, ["module2", "module3"])
        self.assertEqual(prepare_custom_config.non_traceable_module_classes,
                         [self._DummyNonTraceableModule1, self._DummyNonTraceableModule2])
        self.assertEqual(prepare_custom_config.input_quantized_indexes, [0, 1])
        self.assertEqual(prepare_custom_config.output_quantized_indexes, [0, 1])
        self.assertEqual(prepare_custom_config.preserved_attributes, ["attr1", "attr2"])

    def test_prepare_custom_config_to_dict(self):
        prepare_custom_config_dict = self._get_dummy_prepare_custom_config_dict()
        (sm_name, qm1, ei1, pcc1, bcd1) = prepare_custom_config_dict[STANDALONE_MODULE_NAME_DICT_KEY][0]
        (sm_class, qm2, ei2, pcc2, bcd2) = prepare_custom_config_dict[STANDALONE_MODULE_CLASS_DICT_KEY][0]
        prepare_custom_config = PrepareCustomConfig() \
            .set_standalone_module_name(sm_name, qm1, ei1, pcc1, bcd1) \
            .set_standalone_module_class(sm_class, qm2, ei2, pcc2, bcd2) \
            .set_float_to_observed_mapping(self._DummyFloatModule, self._DummyObservedModule) \
            .set_non_traceable_module_names(["module2", "module3"]) \
            .set_non_traceable_module_classes([self._DummyNonTraceableModule1, self._DummyNonTraceableModule2]) \
            .set_input_quantized_indexes([0, 1]) \
            .set_output_quantized_indexes([0, 1]) \
            .set_preserved_attributes(["attr1", "attr2"])
        # PrepareCustomConfig.to_dict also converts internal QConfigMappings and PrepareCustomConfigs to dicts
        prepare_custom_config_dict[STANDALONE_MODULE_NAME_DICT_KEY][0] = (sm_name, qm1.to_dict(), ei1, pcc1.to_dict(), bcd1)
        prepare_custom_config_dict[STANDALONE_MODULE_CLASS_DICT_KEY][0] = (sm_class, qm2.to_dict(), ei2, pcc2.to_dict(), bcd2)
        self.assertEqual(prepare_custom_config.to_dict(), prepare_custom_config_dict)

    def test_convert_custom_config_set_observed_to_quantized_mapping(self):
        convert_custom_config = ConvertCustomConfig()
        self.assertEqual(len(convert_custom_config.observed_to_quantized_mapping), 0)
        convert_custom_config.set_observed_to_quantized_mapping(
            self._DummyObservedModule, self._DummyQuantizedModule, QuantType.STATIC)
        self.assertEqual(len(convert_custom_config.observed_to_quantized_mapping), 1)
        self.assertEqual(list(convert_custom_config.observed_to_quantized_mapping.keys()), [QuantType.STATIC])
        self.assertTrue(self._DummyObservedModule in convert_custom_config.observed_to_quantized_mapping[QuantType.STATIC])
        self.assertEqual(convert_custom_config.observed_to_quantized_mapping[QuantType.STATIC][self._DummyObservedModule],
                         self._DummyQuantizedModule)

    def test_convert_custom_config_set_preserved_attributes(self):
        convert_custom_config = ConvertCustomConfig()
        self.assertEqual(len(convert_custom_config.preserved_attributes), 0)
        convert_custom_config.set_preserved_attributes(["attr1", "attr2"])
        self.assertEqual(convert_custom_config.preserved_attributes, ["attr1", "attr2"])

    def _get_dummy_convert_custom_config_dict(self):
        """
        Return a dummy convert_custom_config_dict to test ConvertCustomConfig's to_dict and from_dict methods.
        """
        return {
            OBSERVED_TO_QUANTIZED_DICT_KEY: {
                "static": {
                    self._DummyObservedModule: self._DummyQuantizedModule
                },
            },
            PRESERVED_ATTRIBUTES_DICT_KEY: ["attr1", "attr2"]
        }

    def test_convert_custom_config_from_dict(self):
        convert_custom_config_dict = self._get_dummy_convert_custom_config_dict()
        convert_custom_config = ConvertCustomConfig.from_dict(convert_custom_config_dict)
        self.assertEqual(len(convert_custom_config.observed_to_quantized_mapping), 1)
        self.assertEqual(list(convert_custom_config.observed_to_quantized_mapping.keys()), [QuantType.STATIC])
        self.assertEqual(len(convert_custom_config.observed_to_quantized_mapping[QuantType.STATIC]), 1)
        self.assertTrue(self._DummyObservedModule in convert_custom_config.observed_to_quantized_mapping[QuantType.STATIC])
        self.assertEqual(convert_custom_config.observed_to_quantized_mapping[QuantType.STATIC][self._DummyObservedModule],
                         self._DummyQuantizedModule)
        self.assertEqual(convert_custom_config.preserved_attributes, ["attr1", "attr2"])

    def test_convert_custom_config_to_dict(self):
        convert_custom_config = ConvertCustomConfig() \
            .set_observed_to_quantized_mapping(self._DummyObservedModule, self._DummyQuantizedModule) \
            .set_preserved_attributes(["attr1", "attr2"])
        self.assertEqual(convert_custom_config.to_dict(), self._get_dummy_convert_custom_config_dict())

    def test_fuse_custom_config_set_preserved_attributes(self):
        fuse_custom_config = FuseCustomConfig()
        self.assertEqual(len(fuse_custom_config.preserved_attributes), 0)
        fuse_custom_config.set_preserved_attributes(["attr1", "attr2"])
        self.assertEqual(fuse_custom_config.preserved_attributes, ["attr1", "attr2"])

    def test_fuse_custom_config_from_dict(self):
        fuse_custom_config_dict = {PRESERVED_ATTRIBUTES_DICT_KEY: ["attr1", "attr2"]}
        fuse_custom_config = FuseCustomConfig.from_dict(fuse_custom_config_dict)
        self.assertEqual(fuse_custom_config.preserved_attributes, ["attr1", "attr2"])

    def test_fuse_custom_config_to_dict(self):
        fuse_custom_config_dict = {PRESERVED_ATTRIBUTES_DICT_KEY: ["attr1", "attr2"]}
        fuse_custom_config = FuseCustomConfig().set_preserved_attributes(["attr1", "attr2"])
        self.assertEqual(fuse_custom_config.to_dict(), fuse_custom_config_dict)

    def test_remove_qconfig(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.avg_pool = torch.nn.AvgPool2d(1)

            def forward(self, x):
                return self.avg_pool(x)

        m = M().eval()
        qconfig_dict = {'': default_qconfig}
        example_inputs = (torch.randn(1, 1, 1, 1),)
        m = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        m(*example_inputs)
        m = convert_fx(m)
        m(*example_inputs)
        for name, module in m.named_modules():
            self.assertFalse(hasattr(module, 'qconfig'),
                             'qconfig is not removed for ' + name)

    def test_return_none(self):
        class M(torch.nn.Module):
            def forward(self, x):
                pass

        m = M().eval()
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        m = prepare_fx(m, qconfig_dict, example_inputs=(torch.randn(1),))
        m = convert_fx(m)

    def test_default_quant_after_none_qconfig(self):
        """ Make sure default quant is inserted properly"""
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 1, 1)
                self.conv2 = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv1(x)
                x = x.transpose(1, 2)
                x = self.conv2(x)

        m = M().eval()
        qconfig_dict = {
            "": default_qconfig,
            "module_name": [
                ("conv1", None)
            ]
        }
        m = prepare_fx(m, qconfig_dict, example_inputs=(torch.randn(1, 1, 1, 1),))
        m = convert_fx(m)

    def test_qconfig_for_call_method(self):
        class Sub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = x.transpose(2, 3)
                x = self.conv(x)
                return x.transpose(2, 3)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = Sub()
                self.conv1 = torch.nn.Conv2d(1, 1, 1)
                self.conv2 = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.sub(x)
                x = self.conv2(x)
                return x.transpose(2, 3)

        qconfig_dict1 = {"": default_qconfig, "module_name": [("sub", None)]}
        # since sub is configured to have qconfig None, we should dequantize the output
        # of self.conv1 and quantize the input of self.conv2
        # dequantize after conv2 should happen after transpose since
        # it is configured with default_qconfig
        # nodes in Sub module instance is not quantized
        node_list1 = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Conv2d),
            ns.call_method("dequantize"),
            ns.call_method("transpose"),
            ns.call_module(nn.Conv2d),
            ns.call_method("transpose"),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Conv2d),
            ns.call_method("transpose"),
            ns.call_method("dequantize")
        ]

        qconfig_dict2 = {"": None, "module_name": [("sub", default_qconfig)]}
        # Only nodes in Sub module instance are quantized
        # the first transpose is not quantized because the input is not quantized
        node_list2 = [
            ns.call_module(nn.Conv2d),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_method("transpose"),
            ns.call_module(nnq.Conv2d),
            ns.call_method("transpose"),
            ns.call_method("dequantize"),
            ns.call_module(nn.Conv2d),
            ns.call_method("transpose"),
        ]

        for qconfig_dict, node_list in [
                (qconfig_dict1, node_list1),
                (qconfig_dict2, node_list2)
        ]:
            example_inputs = (torch.randn(2, 1, 3, 3),)
            m = M().eval()
            m = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
            m(torch.randn(2, 1, 3, 3))
            m = convert_fx(m)
            self.checkGraphModuleNodes(m, expected_node_list=node_list)
            # make sure it runs
            m(*example_inputs)

    def test_qconfig_for_call_func(self):
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.ones(5, 5)
                self.b = torch.zeros(5)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.w, self.b)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mods1 = torch.nn.Sequential(
                    Linear(),
                    Linear()
                )
                self.mods2 = Linear()

            def forward(self, x):
                x = self.mods1(x)
                x = self.mods2(x)
                return x

        model = M().eval()
        example_inputs = (torch.rand(5, 5),)
        qconfig_dict = {"": default_qconfig, "module_name": [("mods2", None)]}
        m = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)
        m(*example_inputs)

        m = convert_fx(m)
        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_function(torch.ops.quantized.linear),
            ns.call_function(torch.ops.quantized.linear),
            ns.call_method('dequantize'),
            ns.call_function(torch.nn.functional.linear)
        ]
        self.checkGraphModuleNodes(m, expected_node_list=node_list)
        m(torch.rand(5, 5))

    def test_preserve_attributes(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                return self.conv(x)

        m = M()
        m.eval()
        m.preserved_attr = 3
        prepare_custom_config_dict = {
            "preserved_attributes": ["preserved_attr"]
        }
        example_inputs = (torch.randn(1, 1, 1, 1),)
        m = prepare_fx(
            m,
            {"": default_qconfig},
            example_inputs=example_inputs,
            prepare_custom_config=prepare_custom_config_dict)

        def assertAttrPreserved(m):
            self.assertTrue(hasattr(m, "preserved_attr"))
            self.assertEqual(m.preserved_attr, 3)

        assertAttrPreserved(m)
        convert_custom_config_dict = {
            "preserved_attributes": ["preserved_attr"]
        }
        m = convert_fx(m, convert_custom_config=convert_custom_config_dict)
        assertAttrPreserved(m)

    @skipIfNoFBGEMM
    def test_qat_and_script(self):
        model = LinearModelWithSubmodule().train()
        qengine = torch.backends.quantized.engine
        qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig(qengine)}
        x = torch.randn(5, 5)
        example_inputs = (x,)
        model = prepare_qat_fx(model, qconfig_dict, example_inputs=example_inputs)

        # ensure scripting works
        scripted = torch.jit.script(model)
        # run one round to make sure model runs
        scripted(x)
        FileCheck().check_count('FakeQuantize = prim::GetAttr[name="', 4, exactly=True) \
                   .run(scripted.graph)

        # disable fake_quant and observer
        for epoch in range(3):
            if epoch == 1:
                scripted.apply(torch.ao.quantization.disable_observer)
            if epoch == 2:
                scripted.apply(torch.ao.quantization.disable_fake_quant)

        # ensure the fake_quant and observer have been disabled.
        matches = ['.fake_quant_enabled', '.observer_enabled']
        for key, v in scripted.state_dict().items():
            if any(x in key for x in matches):
                self.assertEqual(v, torch.tensor([0], dtype=torch.int64))

        # enable them back
        scripted.apply(torch.ao.quantization.enable_fake_quant)
        scripted.apply(torch.ao.quantization.enable_observer)
        for key, v in scripted.state_dict().items():
            if any(x in key for x in matches):
                self.assertEqual(v, torch.tensor([1], dtype=torch.int64))

    @skipIfNoFBGEMM
    def test_save_observer_state_dict(self):
        orig = LinearModelWithSubmodule().eval()
        model = orig
        qconfig_dict = {'': torch.ao.quantization.get_default_qconfig('fbgemm')}
        x = torch.randn(5, 5)
        model = prepare_fx(model, qconfig_dict, example_inputs=(x,))

        # run it through input
        model(x)

        quant = convert_fx(model)

        # save state_dict of model
        obs_dict = torch.ao.quantization.get_observer_state_dict(model)
        b = io.BytesIO()
        torch.save(obs_dict, b)
        b.seek(0)

        # Load the stats into new model
        model_2 = orig
        model_2 = prepare_fx(model_2, qconfig_dict, example_inputs=(x,))

        loaded_dict = torch.load(b)
        torch.ao.quantization.load_observer_state_dict(model_2, loaded_dict)

        quant_2 = convert_fx(model_2)

        # Verify that loaded state dict produces same results.
        self.assertEqual(quant(x), quant_2(x))

    @skipIfNoFBGEMM
    def test_custom_module_class(self):
        class CustomModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return self.linear(x)

        class ObservedCustomModule(torch.nn.Module):
            def __init__(self, linear):
                super().__init__()
                self.linear = linear

            def forward(self, x):
                return self.linear(x)

            @classmethod
            def from_float(cls, float_module):
                assert hasattr(float_module, 'qconfig')
                observed = cls(float_module.linear)
                observed.qconfig = float_module.qconfig
                return observed

        class StaticQuantCustomModule(torch.nn.Module):
            def __init__(self, linear):
                super().__init__()
                self.linear = linear

            def forward(self, x):
                return self.linear(x)

            @classmethod
            def from_observed(cls, observed_module):
                assert hasattr(observed_module, 'qconfig')
                assert hasattr(observed_module, 'activation_post_process')
                observed_module.linear.activation_post_process = \
                    observed_module.activation_post_process
                quantized = cls(nnq.Linear.from_float(observed_module.linear))
                return quantized

        class DynamicQuantCustomModule(torch.nn.Module):
            def __init__(self, linear):
                super().__init__()
                self.linear = linear

            def forward(self, x):
                return self.linear(x)

            @classmethod
            def from_observed(cls, observed_module):
                assert hasattr(observed_module, 'qconfig')
                observed_module.linear.qconfig = observed_module.qconfig
                quantized = cls(nnqd.Linear.from_float(observed_module.linear))
                return quantized

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.custom = CustomModule()

            def forward(self, x):
                x = self.linear(x)
                x = self.custom(x)
                return x

        class RefM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(3, 3)
                self.linear2 = torch.nn.Linear(3, 3)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        # instantiate M and RefM and align the parameters
        original_m = M().eval()
        original_ref_m = RefM().eval()
        original_ref_m.linear1.weight = torch.nn.Parameter(original_m.linear.weight.detach())
        original_ref_m.linear1.bias = torch.nn.Parameter(original_m.linear.bias.detach())
        original_ref_m.linear2.weight = torch.nn.Parameter(original_m.custom.linear.weight.detach())
        original_ref_m.linear2.bias = torch.nn.Parameter(original_m.custom.linear.bias.detach())

        a16_qconfig = QConfig(
            activation=MinMaxObserver.with_args(dtype=torch.qint32, quant_min=0, quant_max=65536),
            weight=default_weight_observer,
        )
        test_configs = {
            "static": (default_qconfig, StaticQuantCustomModule, 3),
            "static_a16": (a16_qconfig, StaticQuantCustomModule, 3),
            "dynamic": (default_dynamic_qconfig, DynamicQuantCustomModule, 0)
        }

        for quant_type in [QuantType.STATIC, QuantType.DYNAMIC]:
            key = _get_quant_type_to_str(quant_type)
            qconfig, quantized_module_class, num_observers = test_configs[key]
            qconfig_dict = {"": qconfig}
            if key == "static":
                prepare_custom_config_dict = {
                    "float_to_observed_custom_module_class": {
                        "static": {
                            CustomModule: ObservedCustomModule
                        }
                    }
                }
                convert_custom_config_dict = {
                    "observed_to_quantized_custom_module_class": {
                        "static": {
                            ObservedCustomModule: quantized_module_class
                        }
                    }
                }
            else:
                prepare_custom_config_dict = {
                    "non_traceable_module_class": [
                        CustomModule
                    ]
                }
                convert_custom_config_dict = {
                    "observed_to_quantized_custom_module_class": {
                        "dynamic": {
                            CustomModule: quantized_module_class
                        }
                    }
                }

            example_inputs = (torch.randn(3, 3),)
            # check prepared model
            m = prepare_fx(
                copy.deepcopy(original_m),
                qconfig_dict,
                example_inputs=example_inputs,
                prepare_custom_config=prepare_custom_config_dict)
            # calibration
            m(*example_inputs)
            # all activation observers are inserted in the top level module
            count_check = {
                ns.call_module(torch.ao.quantization.MinMaxObserver): num_observers
            }
            self.checkGraphModuleNodes(m, expected_node_occurrence=count_check)

            # check converted/quantized model
            m = convert_fx(
                m,
                convert_custom_config=convert_custom_config_dict)
            if quant_type == QuantType.STATIC:
                count_check = {
                    ns.call_function(torch.quantize_per_tensor) : 1,
                    ns.call_module(nnq.Linear) : 1,
                    ns.call_method('dequantize') : 1,
                }
                self.checkGraphModuleNodes(m, expected_node_occurrence=count_check)
            self.assertEqual(type(m.custom), quantized_module_class)
            res = m(*example_inputs)

            # quantize the reference model
            ref_m = prepare_fx(
                copy.deepcopy(original_ref_m), qconfig_dict, example_inputs=example_inputs)
            ref_m(*example_inputs)
            ref_m = convert_fx(ref_m)
            ref_res = ref_m(*example_inputs)
            self.assertEqual(res, ref_res)

    @skipIfNoFBGEMM
    def test_custom_module_class_input_has_multiple_users(self):
        """ Tests that the flow still works when the input of custom module
        has multiple users
        """
        class CustomModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return self.linear(x)

        class ObservedCustomModule(torch.nn.Module):
            def __init__(self, linear):
                super().__init__()
                self.linear = linear

            def forward(self, x):
                return self.linear(x)

            @classmethod
            def from_float(cls, float_module):
                assert hasattr(float_module, 'qconfig')
                observed = cls(float_module.linear)
                observed.qconfig = float_module.qconfig
                return observed

        class StaticQuantCustomModule(torch.nn.Module):
            def __init__(self, linear):
                super().__init__()
                self.linear = linear

            def forward(self, x):
                return self.linear(x)

            @classmethod
            def from_observed(cls, observed_module):
                assert hasattr(observed_module, 'qconfig')
                assert hasattr(observed_module, 'activation_post_process')
                observed_module.linear.activation_post_process = \
                    observed_module.activation_post_process
                quantized = cls(nnq.Linear.from_float(observed_module.linear))
                return quantized

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.custom = CustomModule()

            def forward(self, x0):
                x1 = self.custom(x0)
                x2 = self.linear(x0)
                return x1 + x2

        prepare_custom_config_dict = {
            "float_to_observed_custom_module_class": {
                "static": {
                    CustomModule: ObservedCustomModule
                }
            }
        }
        convert_custom_config_dict = {
            "observed_to_quantized_custom_module_class": {
                "static": {
                    ObservedCustomModule: StaticQuantCustomModule
                }
            }
        }
        m = M().eval()
        example_inputs = (torch.randn(3, 3),)
        m = prepare_fx(
            m,
            {"": default_qconfig},
            example_inputs=example_inputs,
            prepare_custom_config=prepare_custom_config_dict)
        # make sure it works
        m = convert_fx(
            m,
            convert_custom_config=convert_custom_config_dict)
        # make sure it runs
        m(*example_inputs)

    @skipIfNoFBGEMM
    def test_non_traceable_module(self):
        class NonTraceable(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                for k in x.keys():
                    print(x[k])
                return x

        class NonTraceable2(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # data dependent control flow is not traceable
                for i in x:
                    print(i)
                return x

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m1 = NonTraceable()
                self.m2 = NonTraceable2()

            def forward(self, x):
                x = self.m1(x)
                x = self.m2(x)
                return x

        m = M().eval()
        qconfig_dict = {"": default_qconfig}
        prepare_custom_config_dict = {
            "non_traceable_module_name": [
                "m1"
            ],
            "non_traceable_module_class": [
                NonTraceable2
            ]
        }
        m = prepare_fx(
            m, qconfig_dict,
            example_inputs=({"key": torch.randn(1)},),
            prepare_custom_config=prepare_custom_config_dict)

        node_occurrence = {
            ns.call_module(NonTraceable) : 1,
            ns.call_module(NonTraceable2) : 1,
        }
        # make sure these modules are not traced
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_prepared_model_deepcopy(self):
        """Ensures that copy.deepcopy works correctly on a prepared model.
        """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self._foobar = 'foobar'
                self.foobar2 = 'foobar2'

            def forward(self, x):
                x = self.conv(x)
                return x

        m = M()
        m.eval()
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        example_inputs = (torch.randn(4, 1, 4, 4),)
        prepared = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        # calibrate
        prepared(*example_inputs)
        # copy
        prepared_copy = copy.deepcopy(prepared)
        # quantize, should run with no errors
        quantized = convert_fx(prepared_copy)

    def test_quantized_model_type(self):
        """ Test state_dict and deepcopy works properly in the quantized model
        """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x)

        example_inputs = (torch.rand(8, 5),)
        m = M().eval()
        m = prepare_fx(m, {"": default_qconfig}, example_inputs=example_inputs)
        m = convert_fx(m)
        # test deepcopy
        m_copy = copy.deepcopy(m)
        self.assertEqual(m_copy(*example_inputs), m(*example_inputs))

        # test state_dict
        state_dict = m.state_dict()
        m_new = M().eval()
        m_new = prepare_fx(m_new, {"": default_qconfig}, example_inputs=example_inputs)
        m_new = convert_fx(m_new)
        m_new.load_state_dict(state_dict)
        self.assertEqual(m_new(*example_inputs), m(*example_inputs))

    def test_dequantize(self):
        r""" Test to make sure dequantize node are placed before
        non-quantizable node
        """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.act = torch.nn.GELU()

            def forward(self, x):
                x = self.conv(x)
                return self.act(x)

        data = torch.rand(5, 1, 3, 3, dtype=torch.float)
        for quant_type in self.static_quant_types:
            node_list = [
                ns.call_module(nnq.Conv2d),
                ns.call_method("dequantize"),
                ns.call_module(nn.GELU),
            ]
            self.checkGraphModeFxOp(
                M().eval(), (data,), quant_type, expected_node_list=node_list)

    def test_sequential(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 1, 1),
                    torch.nn.Conv2d(1, 1, 1)
                )

            def forward(self, x):
                x = self.convs(x)
                return x

        data = torch.rand(5, 1, 3, 3, dtype=torch.float)
        for quant_type in self.static_quant_types:
            node_list = [
                ns.call_module(nnq.Conv2d),
                ns.call_module(nnq.Conv2d),
            ]
            self.checkGraphModeFxOp(
                M().eval(), (data,), quant_type, expected_node_list=node_list)

    def _test_quantized_inputs_outputs(
            self, prepare_custom_config_dict, prepare_count_check,
            convert_count_check):
        """
        Test the option to have inputs and outputs of the graph quantized
        """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 1, 1)
                self.conv2 = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                return x

        # quantized input, quantized output
        m = M()
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        example_inputs = (torch.randn(1, 1, 4, 4),)
        m.eval()
        mp = torch.ao.quantization.quantize_fx.prepare_fx(
            m, qconfig_dict,
            example_inputs=example_inputs,
            prepare_custom_config=prepare_custom_config_dict)
        self.checkGraphModuleNodes(mp, expected_node_occurrence=prepare_count_check)
        mp(*example_inputs)
        mq = torch.ao.quantization.quantize_fx.convert_fx(mp)
        self.checkGraphModuleNodes(mq, expected_node_occurrence=convert_count_check)

    def test_quantized_input_quantized_output(self):
        prepare_custom_config_dict = {
            'input_quantized_idxs': [0], 'output_quantized_idxs': [0]}
        prepare_count_check = {
            ns.call_module(torch.ao.quantization.MinMaxObserver): 2,
        }
        convert_count_check = {
            ns.call_function(torch.quantize_per_tensor): 0,
            ns.call_method('dequantize'): 0,
        }
        self._test_quantized_inputs_outputs(
            prepare_custom_config_dict, prepare_count_check, convert_count_check)

    def test_fp32_input_quantized_output(self):
        prepare_custom_config_dict = {
            'output_quantized_idxs': [0]}
        prepare_count_check = {
            ns.call_module(torch.ao.quantization.MinMaxObserver): 3,
        }
        convert_count_check = {
            ns.call_function(torch.quantize_per_tensor): 1,
            ns.call_method('dequantize'): 0,
        }
        self._test_quantized_inputs_outputs(
            prepare_custom_config_dict, prepare_count_check, convert_count_check)

    def test_quantized_input_fp32_output(self):
        prepare_custom_config_dict = {
            'input_quantized_idxs': [0]}
        prepare_count_check = {
            ns.call_module(torch.ao.quantization.MinMaxObserver): 2,
        }
        convert_count_check = {
            ns.call_function(torch.quantize_per_tensor): 0,
            ns.call_method('dequantize'): 1,
        }
        self._test_quantized_inputs_outputs(
            prepare_custom_config_dict, prepare_count_check, convert_count_check)

    def test_fp32_input_fp32_output(self):
        prepare_custom_config_dict = {}
        prepare_count_check = {
            ns.call_module(torch.ao.quantization.MinMaxObserver): 3,
        }
        convert_count_check = {
            ns.call_function(torch.quantize_per_tensor): 1,
            ns.call_method('dequantize'): 1,
        }
        self._test_quantized_inputs_outputs(
            prepare_custom_config_dict, prepare_count_check, convert_count_check)

    @skipIfNoFBGEMM
    def test_convtranspose_per_channel_fails_early(self):
        r"""
        Verifies that attempting to quantize a ConvTranspose module with per-Channel
        weight observers fails in the prepare step, as opposed to the convert step.
        """
        m = torch.nn.Sequential(torch.nn.ConvTranspose2d(1, 1, 1))
        m.eval()
        qconfig_dict = {'': torch.ao.quantization.get_default_qconfig('fbgemm')}
        with self.assertRaises(AssertionError) as context:
            mp = prepare_fx(m, qconfig_dict, example_inputs=(torch.randn(1, 1, 1, 1),))
        self.assertTrue(
            str(context.exception) ==
            'Per channel weight observer is not supported yet for ConvTranspose{n}d.')

    @skipIfNoFBGEMM
    def test_qparams_buffers(self):
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.ones(5, 5)
                self.b = torch.zeros(5)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.w, self.b)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mods1 = torch.nn.Sequential(
                    Linear(),
                    Linear()
                )
                self.mods2 = Linear()

            def forward(self, x):
                x = self.mods1(x)
                x = self.mods2(x)
                return x

        model = M().eval()
        qconfig_dict = {"": default_qconfig}
        example_inputs = (torch.rand(5, 5),)
        m = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)
        m(*example_inputs)
        m = convert_fx(m)
        keys = m.state_dict().keys()
        quant_scale_count = quant_zero_point = scale_count = zero_point_count = 0
        for k in keys:
            if 'input_scale' in k:
                quant_scale_count = quant_scale_count + 1
            elif 'input_zero_point' in k:
                quant_zero_point = quant_zero_point + 1
            elif 'scale' in k:
                scale_count = scale_count + 1
            elif 'zero_point' in k:
                zero_point_count = zero_point_count + 1

        # Expect each quantized linear op to have a scale and zero point
        self.assertTrue(scale_count == 3, "Expect each quantized linear op to have a scale in state_dict")
        self.assertTrue(zero_point_count == 3, "Expect each quantized linear op to have a zero_point in state_dict")
        # ensure it runs
        m(*example_inputs)
        # ensure it is scriptable
        scripted = torch.jit.script(m)
        scripted_keys = scripted.state_dict().keys()
        scripted.mods1_0_packed_weight_0 = m.state_dict()["mods1_0_packed_weight_0"]
        non_packed_weight_keys = [key for key in keys if "_packed_weight" not in key]
        self.assertTrue(
            set(scripted_keys) == set(non_packed_weight_keys),
            "Expected the scripted model to preserve the state_dict for non-packed weight attributes")
        # TODO: probably don't want to hardcode the attribute names, since they are generated
        for attr_name in [
                "mods1_0_input_scale_0", "mods1_0_input_zero_point_0",
                "mods1_0_scale_1", "mods1_0_zero_point_1",
                "mods1_1_scale_1", "mods1_1_zero_point_1",
                "mods2_scale_1", "mods2_zero_point_1"]:
            self.assertTrue(hasattr(m, attr_name), attr_name + " not found.")

    @skipIfNoFBGEMM
    def test_packed_weight_fused_op(self):
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.ones(5, 5)
                self.b = torch.zeros(5)

            def forward(self, x):
                return F.linear(x, self.w, self.b)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mods1 = torch.nn.Sequential(
                    Linear(),
                    Linear()
                )
                self.mods2 = Linear()
                self.relu = F.relu

            def forward(self, x):
                x = self.mods1(x)
                x = self.mods2(x)
                x = self.relu(x)
                return x

        model = M().eval()
        example_inputs = (torch.rand(5, 5),)
        qconfig_dict = {"": default_qconfig}
        m = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)
        m(*example_inputs)
        m = convert_fx(m)
        assert hasattr(m, "mods1_0_packed_weight_0")
        assert hasattr(m, "mods1_1_packed_weight_0")
        assert hasattr(m, "mods2_packed_weight_0")

    @skipIfNoFBGEMM
    def test_mul_add_fp16_config(self):
        with override_quantized_engine('fbgemm'):
            class Linear(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.w = torch.ones(5, 5)
                    self.b = torch.zeros(5)

                def forward(self, x):
                    return torch.nn.functional.linear(x, self.w, self.b)

            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mods1 = torch.nn.Sequential(
                        Linear(),
                        Linear()
                    )
                    self.mods2 = Linear()

                def forward(self, x):
                    x = x * 5
                    x = x + 5
                    x = self.mods1(x)
                    x = self.mods2(x)
                    return x
            model = M().eval()
            qconfig_dict = {"": float16_dynamic_qconfig}
            example_inputs = (torch.rand(5, 5),)
            m = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)
            m = convert_fx(m)
            # make sure it runs
            m(*example_inputs)

    def test_getattr_with_nontensor_result(self):
        """
        Verifies that binary ops get quantized correctly if some
        of the args are nodes but not Tensors, such as an `x.ndim`
        pattern.
        """
        class M1(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                dims = x.ndim
                dims_sub = dims - 1
                dims_sub2 = dims_sub - 1
                x = torch.add(x, dims_sub2)
                return x

        class M2(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                dims = x.ndim
                dims_sub = dims - 2
                mul = [1] * dims_sub
                dims_list = [-1, x.size(1)] + mul
                x = x.view(dims_list)
                return x

        class M3(torch.nn.Module):
            def forward(self, x):
                shape = x.shape
                x = x.view(shape)
                return x

        for cls in (M1, M2, M3):
            m = cls().eval()
            example_inputs = (torch.rand(4, 4, 4, 4),)
            m(*example_inputs)
            qconfig_dict = {'': torch.ao.quantization.default_qconfig}
            mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
            mp(torch.rand(4, 4, 4, 4))
            mc = convert_fx(mp)

    class _NonReferenceTestModel(nn.Module):
        def __init__(self, func, lin_in, lin_out):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.lin = nn.Linear(lin_in, lin_out)
            self.func = func

        def forward(self, x, y, z):
            x = self.pool(F.relu(self.conv1(x)))
            x = torch.flatten(x, 1)
            x = self.func(x, y, z)
            x = self.lin(x)
            return x

    # This function looks at the node specified by the NodeInfo in the key of
    # node_info_to_non_tensor_args and checks that the args at specified indices
    # are not observed (since they are non tensors). If the args at those indices
    # are a tuple/list (which do not show up as nodes) the function checks the
    # individual elements of the tuple/list recursively.
    def _check_not_observed(self, model, node_info_to_non_tensor_args):

        # this is a helper function (for easier recursion) that checks whether
        # arg_node is observed
        def _check_node_not_observed(model, arg_node, node):
            if isinstance(arg_node, tuple) or isinstance(arg_node, list):
                for new_node in arg_node:
                    _check_node_not_observed(model, new_node, node)
            elif arg_node.op == "call_module":
                self.assertTrue(
                    not _is_activation_post_process(getattr(model, arg_node.target)),
                    "Arg: {0} of node: {1} is observed but is not a float tensor".format(
                        arg_node, node
                    ),
                )

        for node in model.graph.nodes:
            indices = node_info_to_non_tensor_args.get(
                NodeInfo(node.op, node.target), []
            )
            for index in indices:
                if index < len(node.args):
                    arg_node = node.args[index]
                    _check_node_not_observed(model, arg_node, node)

    # This test checks that the model gets prepared correct, doesn't have observers
    # on specific ops (see _check_not_observed) and that the prepared model runs
    def _test_dtype_propagation(self, model, node_info_to_non_tensor_args, *args):
        model.eval()
        qconfig_dict = {"": torch.ao.quantization.get_default_qconfig("fbgemm")}
        prepared_model = prepare_fx(model, qconfig_dict, example_inputs=tuple(args))
        self._check_not_observed(prepared_model, node_info_to_non_tensor_args)
        prepared_model(*args)

    def test_masked_fill_nontensor_args_not_observed(self):
        def func(x, y, z):
            return x.masked_fill(y, z)

        model = self._NonReferenceTestModel(func, 1176, 1)
        args = [torch.randn(5, 3, 32, 32), torch.randn(1176) > 0, 0.1]
        node_info_to_non_tensor_args = {NodeInfo("call_method", "masked_fill"): [1, 2]}
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    def test_permute_nontensor_args_not_observed(self):
        def func(x, y, z):
            return x.permute(y, z)

        model = self._NonReferenceTestModel(func, 1176, 1)
        args = [torch.randn(5, 3, 32, 32), 0, 1]
        node_info_to_non_tensor_args = {NodeInfo("call_method", "permute"): [1, 2]}
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    def test_repeat_nontensor_args_not_observed(self):
        def func(x, y, z):
            return x.repeat(y, z)

        model = self._NonReferenceTestModel(func, 1176, 1)
        args = [torch.randn(5, 3, 32, 32), 2, 1]
        node_info_to_non_tensor_args = {NodeInfo("call_method", "repeat"): [1, 2]}
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    def test_reshape_nontensor_args_not_observed(self):
        def func(x, y, z):
            return x.reshape(-1, y)

        model = self._NonReferenceTestModel(func, 5, 1)
        args = [torch.randn(5, 3, 32, 32), 5, None]
        node_info_to_non_tensor_args = {NodeInfo("call_method", "reshape"): [2]}
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    def test_size_nontensor_args_not_observed(self):
        def func(x, y, z):
            return x.reshape((-1, x.size(y)))

        model = self._NonReferenceTestModel(func, 5, 1)
        args = [torch.randn(5, 3, 32, 32), 0, None]
        node_info_to_non_tensor_args = {NodeInfo("call_method", "size"): [1]}
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    def test_transpose_nontensor_args_not_observed(self):
        def func(x, y, z):
            return x.transpose(y, z)

        model = self._NonReferenceTestModel(func, 5, 1)
        args = [torch.randn(5, 3, 32, 32), 0, 1]
        node_info_to_non_tensor_args = {NodeInfo("call_method", "transpose"): [1, 2]}
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    def test_torch_transpose_nontensor_args_not_observed(self):
        # TODO: make torch.transpose traceable by fx when using
        # variable nontensor arguments
        # func = lambda x, y, z: torch.transpose(x, y, z) # error
        def func(x, y, z):
            return torch.transpose(x, 0, 1)

        model = self._NonReferenceTestModel(func, 5, 1)
        node_info_to_non_tensor_args = {
            NodeInfo("call_method", torch.transpose): [1, 2]
        }
        args = [torch.randn(5, 3, 32, 32), 0, 1]
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    def test_unsqueeze_nontensor_args_not_observed(self):
        def func(x, y, z):
            return x.unsqueeze(y)

        model = self._NonReferenceTestModel(func, 1176, 1)
        args = [torch.randn(5, 3, 32, 32), 1, None]
        node_info_to_non_tensor_args = {NodeInfo("call_method", "unsqueeze"): [1]}
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    def test_unsqueeze__nontensor_args_not_observed(self):
        def func(x, y, z):
            return x.unsqueeze_(y)

        model = self._NonReferenceTestModel(func, 1176, 1)
        args = [torch.randn(5, 3, 32, 32), 1, None]
        node_info_to_non_tensor_args = {NodeInfo("call_method", "unsqueeze_"): [1]}
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    def test_torch_unsqueeze_nontensor_args_not_observed(self):
        # TODO: make torch.unsqueeze scriptable by fx when using
        # variable nontensor arguments
        # func = lambda x, y, z: torch.unsqueeze(x, y) # error
        def func(x, y, z):
            return torch.unsqueeze(x, 1)

        model = self._NonReferenceTestModel(func, 1176, 1)
        args = [torch.randn(5, 3, 32, 32), 1, None]
        node_info_to_non_tensor_args = {NodeInfo("call_method", torch.unsqueeze): [1]}
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    def test_view_nontensor_args_not_observed(self):
        def func(x, y, z):
            return x.view(-1, y)

        model = self._NonReferenceTestModel(func, 5, 1)
        args = [torch.randn(5, 3, 32, 32), 5, None]
        node_info_to_non_tensor_args = {NodeInfo("call_method", "view"): [2]}
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    def test_propagate_dtypes_for_known_nodes_list_args(self):
        def func(x, y, z):
            return x.reshape(y)

        model = self._NonReferenceTestModel(func, 5, 1)
        args = [torch.randn(5, 3, 32, 32), [-1, 5], None]
        node_info_to_non_tensor_args = {NodeInfo("call_method", "reshape"): [1]}
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    def test_propagate_dtypes_for_known_nodes_split_list_args(self):
        def func(x, y, z):
            return x.reshape([y, z])

        model = self._NonReferenceTestModel(func, 5, 1)
        args = [torch.randn(5, 3, 32, 32), -1, 5]
        node_info_to_non_tensor_args = {NodeInfo("call_method", "reshape"): [1]}
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    def test_propagate_dtypes_for_known_nodes_tuple_args(self):
        def func(x, y, z):
            return x.reshape(y)

        model = self._NonReferenceTestModel(func, 5, 1)
        args = [torch.randn(5, 3, 32, 32), (-1, 5), None]
        node_info_to_non_tensor_args = {NodeInfo("call_method", "reshape"): [1]}
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    def test_propagate_dtypes_for_known_nodes_split_tuple_args(self):
        def func(x, y, z):
            return x.reshape((y, z))

        model = self._NonReferenceTestModel(func, 5, 1)
        args = [torch.randn(5, 3, 32, 32), -1, 5]
        node_info_to_non_tensor_args = {NodeInfo("call_method", "reshape"): [1]}
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    def test_propagate_dtypes_for_known_nodes_dict_args(self):
        def func(x, y, z):
            return x.transpose(y["first"], y["second"])

        model = self._NonReferenceTestModel(func, 5, 1)
        args = [torch.randn(5, 3, 32, 32), {"first": 0, "second": 1}, None]
        node_info_to_non_tensor_args = {NodeInfo("call_method", "transpose"): [1, 2]}
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    def test_propagate_dtypes_for_known_nodes_dict_tuple_args(self):
        class reshape_module(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, z):
                return x.reshape(y["shape"])

        model = self._NonReferenceTestModel(reshape_module(), 5, 1)
        args = [torch.randn(5, 3, 32, 32), {"shape": (-1, 5)}, None]
        node_info_to_non_tensor_args = {NodeInfo("call_method", "reshape"): [1]}
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    def test_propagate_dtypes_for_known_nodes_dict_split_tuple_args(self):
        def func(x, y, z):
            return x.reshape((y["first"], y["second"]))

        model = self._NonReferenceTestModel(func, 5, 1)
        args = [torch.randn(5, 3, 32, 32), {"first": -1, "second": 5}, None]
        node_info_to_non_tensor_args = {NodeInfo("call_method", "transpose"): [1]}
        self._test_dtype_propagation(model, node_info_to_non_tensor_args, *args)

    def test_assert_on_size_after_quant_layer(self):
        """
        Verifies that calculating a size of a quantized tensor works
        correctly in quantization passes.
        """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv1(x)
                torch._assert(x.size(1) == 1, 'foobar')
                return x

        m = M().eval()
        example_inputs = (torch.rand(4, 1, 4, 4),)
        m(*example_inputs)
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        mp(*example_inputs)
        mc = convert_fx(mp)
        mc(*example_inputs)

    def test_fp32_sum(self):
        """
        Verifies that fp32 sum works correctly if it's before or after
        quantized layers.
        """
        class M1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv1(x)
                x = torch.stack([x])
                x = torch.sum(x)
                return x

        class M2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.conv2 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv1(x)
                x1 = torch.stack([x])
                x1 = torch.sum(x1, dim=0)
                x2 = self.conv2(x1)
                return x2

        for cls in (M1, M2):
            m = cls().eval()
            example_inputs = (torch.rand(4, 1, 4, 4),)
            m(*example_inputs)
            qconfig_dict = {'': torch.ao.quantization.default_qconfig}
            mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
            mp(*example_inputs)
            mc = convert_fx(mp)
            mc(*example_inputs)

    def test_fusion_pattern_unquantized(self):
        """
        Ensure that leaving a possible fusion pattern of multiple nodes
        unquantized runs through the APIs without errors.
        """
        class Child(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = nn.ReLU()

            def forward(self, x):
                x = torch.add(x, 1.0)
                x = torch.nn.functional.relu(x)
                return x

        class Parent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.child = Child()
                self.conv = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.child(x)
                x = self.conv(x)
                return x

        m = Parent().eval()
        qconfig_dict = {
            '': torch.ao.quantization.default_qconfig,
            'module_name': [
                ('child', None),
            ],
        }
        example_inputs = (torch.rand(1, 1, 1, 1),)
        mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        mp(*example_inputs)
        mc = convert_fx(mp)

    def test_state_dict(self):
        """ Make sure packed params appear in state_dict
        """

        # test linear packed weight
        class M1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.rand(4, 30)
                self.b = torch.rand(4)

            def forward(self, x):
                return F.linear(x, self.w, self.b)

        m = M1().eval()
        qconfig_dict = {"": default_qconfig}
        m = prepare_fx(m, qconfig_dict, example_inputs=(torch.randn(1, 30),))
        m = convert_fx(m)
        state_dict = m.state_dict()
        self.assertTrue("_packed_weight_0" in state_dict)

        # test conv packed weight
        class M2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.rand(3, 3, 3, 3)
                self.b = torch.rand(3)
                self.stride = (1, 1)
                self.padding = (0, 0)
                self.dilation = (1, 1)
                self.groups = 1

            def forward(self, x):
                return F.conv2d(x, self.w, self.b, self.stride, self.padding, self.dilation, self.groups)

        m = M2().eval()
        qconfig_dict = {"": default_qconfig}
        m = prepare_fx(m, qconfig_dict, example_inputs=(torch.randn(1, 3, 3, 3),))
        m = convert_fx(m)
        state_dict = m.state_dict()
        self.assertTrue("_packed_weight_0" in state_dict)

        # test load
        ref_weight, ref_bias = torch.ops.quantized.conv2d_unpack(state_dict["_packed_weight_0"])
        data = torch.rand(1, 3, 5, 5)
        ref_res = m(data)
        m = M2().eval()
        m = prepare_fx(m, qconfig_dict, (data,))
        m = convert_fx(m)
        res = m(data)
        weight, bias = m._packed_weight_0.unpack()
        # check that random model weight/bias does not match ref weight/bias
        self.assertNotEqual(weight, ref_weight)
        self.assertNotEqual(bias, ref_bias)
        self.assertNotEqual(res, ref_res)
        m.load_state_dict(state_dict)

        def checkModel(m, data, ref_weight, ref_bias, ref_res):
            res = m(data)
            weight, bias = m._packed_weight_0.unpack()
            # check that weight/bias matches after load the state_dict
            self.assertEqual(weight, ref_weight)
            self.assertEqual(bias, ref_bias)
            self.assertEqual(res, ref_res)

        checkModel(m, data, ref_weight, ref_bias, ref_res)

        # Test save to disk and load back
        m = M2().eval()
        m = prepare_fx(m, qconfig_dict, example_inputs=(data,))
        m = convert_fx(m)
        m.load_state_dict(state_dict)
        with TemporaryFileName() as fname:
            torch.save(m.state_dict(), fname)
            m.load_state_dict(torch.load(fname))

        checkModel(m, data, ref_weight, ref_bias, ref_res)

    @skipIfNoFBGEMM
    def test_preserve_qconfig(self):
        """
        Test to make sure the temporary config option to preserve qconfig attributes
        in the model works
        """
        with override_quantized_engine('fbgemm'):
            class Linear(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.w = torch.ones(5, 5)
                    self.b = torch.zeros(5)

                def forward(self, x):
                    return torch.nn.functional.linear(x, self.w, self.b)

            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mods1 = torch.nn.Sequential(
                        Linear(),
                        Linear()
                    )
                    self.mods2 = torch.nn.Sigmoid()

                def forward(self, x):
                    x = self.mods1(x)
                    x = self.mods2(x)
                    return x

            model = M().eval()
            qconfig_dict = {
                "object_type": [
                    (torch.nn.functional.linear, float16_dynamic_qconfig),
                ],
            }
            example_inputs = (torch.rand(5, 5),)
            m = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)
            m(*example_inputs)
            m = convert_fx(m, _remove_qconfig=False)

            self.assertTrue(hasattr(m.mods2, 'qconfig'))

    def test_not_used(self):
        """ Test quantizing a not used value"""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = x + x
                x.sigmoid_()
                return x

        m = M().eval()
        qconfig_mapping = get_default_qconfig_mapping().set_global(float16_static_qconfig)
        # make sure quantization runs
        m = prepare_fx(m, qconfig_mapping, example_inputs=(torch.randn(1),))
        m = convert_fx(m)

    def test_qparams_fqn(self):
        """ Test that the FQN of input_scale/zero_point is set
        to that of first linear use. """
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.ones(5, 5)
                self.b = torch.zeros(5)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.w, self.b)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mods1 = torch.nn.Sequential(
                    Linear(),
                    Linear()
                )

            def forward(self, x):
                x = torch.cat((x,), 1)
                tmp = x.size()
                x = self.mods1(x)
                y = x * tmp[0]
                return y

        model = M().eval()
        qconfig_dict = {
            "": None,
            "object_type": [
                (torch.nn.functional.linear, default_qconfig),
                (torch.nn.functional.relu, default_qconfig),
            ],
        }
        example_inputs = (torch.rand(5, 5),)
        m = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)
        m(*example_inputs)
        m = convert_fx(m)
        keys = m.state_dict().keys()
        m(torch.randn(5, 5))
        # TODO: probably don't want to hardcode the attribute names, since they are generated
        for attr_name in [
                "mods1_0_input_scale_0", "mods1_0_input_zero_point_0",
                "mods1_0_scale_0", "mods1_0_zero_point_0",
                "mods1_1_scale_0", "mods1_1_zero_point_0"]:
            self.assertTrue(hasattr(m, attr_name), attr_name + " not found.")

    def test_no_obs_between_unmatched_node_and_copy_node(self):
        """
        Verifies that an observer is not inserted between an unmatched
        node and a node matched to CopyNodeQuantizeHandler.  This is done
        because observers require activations to be Tensors, and there is
        no guarantee that an output of an unmatched node is a Tensor.
        """

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = nn.ReLU()

            def forward(self, x):
                x = _user_func_with_complex_return_type(x)
                x1 = x[0] + 1
                return x1, x[1]

        m = M().eval()

        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        example_inputs = (torch.randn(4, 4, 4, 4),)
        mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        # if an observer is inserted after _user_func_with_complex_return_type,
        # the following call will fail
        mp(*example_inputs)
        mc = convert_fx(mp)
        mc(*example_inputs)

    def test_fold_quant_dequant(self):
        """ Test that the sequence of quant-dequant nodes in the
            graph, get folded and we erase the extra dequant nodes.
        """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.ones(5, 5)
                self.b = torch.zeros(5)

            def forward(self, x):
                x = torch.cat((x,), 1)
                tmp = x.size()
                x = torch.nn.functional.linear(x, self.w, self.b)
                y = x * tmp[0]
                return y

        model = M().eval()
        qconfig_dict = {
            "": None,
            "object_type": [
                (torch.nn.functional.linear, default_qconfig),
            ],
        }
        example_inputs = (torch.rand(5, 5),)
        m = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)
        m(*example_inputs)
        m = convert_fx(m)
        keys = m.state_dict().keys()
        m(*example_inputs)
        dequant = 0
        quant = 0
        for n in m.graph.nodes:
            if n.op == "call_method" and n.target == "dequantize":
                dequant = dequant + 1
            if n.op == "call_function" and n.target == torch.quantize_per_tensor:
                quant = quant + 1
        self.assertEqual(dequant, 1)
        self.assertEqual(quant, 1)

    def test_quant_output_always_observed(self):
        """
        If the output is hardcoded to be quantized, ensure that
        there is always an observer, even if the last non-output node is not
        quantizeable.
        """
        qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        prepare_custom_config_dict = {'output_quantized_idxs': [0]}
        example_inputs = (torch.randn(4, 1, 4, 4),)

        # non-quantizeable node, quantized output
        class M1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.identity = torch.nn.Identity()

            def forward(self, x):
                x = self.identity(x)
                return x

        m1 = M1()
        self.checkGraphModeFxOp(
            m1, example_inputs, QuantType.QAT,
            prepare_expected_node_occurrence={
                ns.call_module(torch.ao.quantization.FusedMovingAvgObsFakeQuantize): 2,
            },
            expected_node_occurrence={
                ns.call_function(torch.quantize_per_tensor): 1,
            },
            prepare_custom_config=prepare_custom_config_dict)

        # quantizeable node, quantized output
        class M2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv(x)
                return x

        m2 = M2()
        self.checkGraphModeFxOp(
            m2, example_inputs, QuantType.QAT,
            prepare_expected_node_occurrence={
                # one for weights, one for activations
                ns.call_module(torch.ao.quantization.FusedMovingAvgObsFakeQuantize): 2,
            },
            expected_node_occurrence={
                ns.call_function(torch.quantize_per_tensor): 1,
            },
            prepare_custom_config=prepare_custom_config_dict)

        # quantizeable node, quantized dictionary output
        class M3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv(x)
                return {"output": x}

        m3 = M3()
        self.checkGraphModeFxOp(
            m3, example_inputs, QuantType.QAT,
            prepare_expected_node_occurrence={
                # one for weights, one for activations
                ns.call_module(torch.ao.quantization.FusedMovingAvgObsFakeQuantize): 2,
            },
            expected_node_occurrence={
                ns.call_function(torch.quantize_per_tensor): 1,
            },
            prepare_custom_config=prepare_custom_config_dict)

    def test_deepcopy_preserve_attributes(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr = 3

            def forward(self, x):
                return x

        m = M().eval()
        m = prepare_fx(
            m,
            {"": default_qconfig},
            example_inputs=(torch.randn(1),),
            prepare_custom_config={"preserved_attributes": ["attr"]})
        self.assertTrue(hasattr(m, "attr"))
        m2 = copy.deepcopy(m)
        self.assertTrue(hasattr(m2, "attr"))
        m = convert_fx(m, convert_custom_config={"preserved_attributes": ["attr"]})
        self.assertTrue(hasattr(m, "attr"))
        m2 = copy.deepcopy(m)
        self.assertTrue(hasattr(m2, "attr"))

    def test_output_lists_and_dicts(self):
        """Verify that specifying complicated output types does not crash.
        """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv(x)
                return {'foo': [x]}, [{'foo': [[x]]}]

        m = M().eval()
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        mp = prepare_fx(m, qconfig_dict, example_inputs=(torch.randn(1, 1, 1, 1),))
        mc = convert_fx(mp)

    def test_shape_followed_by_quantized_op(self):
        """ Make sure that shape does not dequantize
        the Tensor before the next operator
        """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2)
                self.conv2 = torch.nn.Conv2d(2, 2, 2)

            def forward(self, x):
                x = self.conv1(x)
                s = x.shape
                torch._assert(s == x.shape, "")
                x = self.conv2(x)
                return x

        # make sure quantization runs
        m = M().eval()
        example_inputs = (torch.randn(2, 2, 4, 4),)
        m = prepare_fx(m, {"": default_qconfig}, example_inputs=example_inputs)
        m = convert_fx(m)
        m(*example_inputs)
        node_occurrence = {
            ns.call_function(torch.quantize_per_tensor): 1,
            ns.call_method("dequantize"): 1
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_trace_quantize_per_tensor(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv(x)
                return x

        m = M().eval()
        m = prepare_fx(m, {"": default_qconfig}, example_inputs=(torch.randn(1, 1, 3, 3),))
        m = convert_fx(m)
        # Make sure this runs without error
        m = torch.fx.Transformer(m).transform()

    def test_copy_node_has_shared_actpp_instance(self):
        """ Test the output of CopyNode to have the same
        observer/fake_quant instance as the input
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.avgpool2d = torch.nn.AvgPool2d(kernel_size=3)

            def forward(self, x):
                x = self.avgpool2d(x)
                return x

        for quant_type in self.static_quant_types:
            m = M()
            # Checks that we have an observer for both input and output
            occurrence_map = {
                QuantType.STATIC: {
                    ns.call_module(torch.ao.quantization.MinMaxObserver): 2
                },
                QuantType.QAT: {
                    ns.call_module(torch.ao.quantization.FakeQuantize): 2
                }
            }
            if quant_type == QuantType.QAT:
                m.train()
                prepare = prepare_qat_fx
                qconfig = default_qat_qconfig
                actpp_module_class = torch.ao.quantization.FakeQuantize
            else:
                m.eval()
                prepare = prepare_fx
                qconfig = default_qconfig
                actpp_module_class = torch.ao.quantization.MinMaxObserver

            example_inputs = (torch.randn(1, 3, 3, 3),)
            m = prepare(m, {"": qconfig}, example_inputs=example_inputs)
            # check that there is a duplicated observer instance
            actpp_module_count = 0
            for name, module in m.named_modules(remove_duplicate=False):
                if isinstance(module, actpp_module_class):
                    actpp_module_count += 1
            self.assertEqual(actpp_module_count, 2)

            actpp_module_count = 0
            for name, module in m.named_modules():
                if isinstance(module, actpp_module_class):
                    actpp_module_count += 1
            self.assertEqual(actpp_module_count, 1)

            m_copy = copy.deepcopy(m)
            m = convert_fx(m)
            m_reference = convert_to_reference_fx(m_copy)

            # checks for non-reference quantized model
            node_occurrence = {
                ns.call_function(torch.quantize_per_tensor): 1,
                ns.call_method("dequantize"): 1
            }
            node_list = [
                ns.call_function(torch.quantize_per_tensor),
                ns.call_module(torch.nn.AvgPool2d),
                ns.call_method("dequantize"),
            ]
            self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence, expected_node_list=node_list)

            # checks for reference quantized model, for copy nodes we'll have
            # dequant - copy_node - quant patterns which will be fused later
            # in the backend lowering step
            node_occurrence = {
                ns.call_function(torch.quantize_per_tensor): 2,
                ns.call_method("dequantize"): 2
            }
            node_list = [
                ns.call_function(torch.quantize_per_tensor),
                ns.call_method("dequantize"),
                ns.call_module(torch.nn.AvgPool2d),
                ns.call_function(torch.quantize_per_tensor),
                ns.call_method("dequantize"),
            ]
            self.checkGraphModuleNodes(m_reference, expected_node_occurrence=node_occurrence, expected_node_list=node_list)

    def test_linear_qint8_activation(self):
        """Test support for qint8 activation in reference pattern
        """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 2, 2, 2)
                self.linear = torch.nn.Linear(8, 5)

            def forward(self, x):
                x = self.conv(x)
                x = torch.flatten(x, 1)
                x = self.linear(x)
                return x

        m = M().eval()
        example_inputs = (torch.rand(2, 1, 5, 5),)
        m = prepare_fx(
            m,
            {"": torch.ao.quantization.QConfig(
                activation=torch.ao.quantization.HistogramObserver.with_args(
                    qscheme=torch.per_tensor_symmetric, dtype=torch.qint8
                ), weight=torch.ao.quantization.default_per_channel_weight_observer)},
            example_inputs=example_inputs)
        m = convert_to_reference_fx(m)
        m(*example_inputs)

    def test_preserve_tuple(self):
        """ Test tuple input type is preserved
        """

        class LSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(50, 50, 1)

            def forward(self, inputs: torch.Tensor, state: List[torch.Tensor]):
                h = state[0]
                c = state[1]
                return self.lstm(inputs, (h, c))

        m = LSTM().eval()
        example_inputs = (torch.randn(5, 3, 50), torch.randn(2, 3, 50), torch.randn(2, 3, 50))
        m = prepare_fx(m, {"": default_qconfig}, example_inputs=example_inputs)
        # make sure the arg[1] of lstm module is a tuple
        for n in m.graph.nodes:
            if n.target == "lstm":
                self.assertEqual(type(n.args[1]), tuple)

    def _test_static_lstm_helper(self, model, prepare_node_occurrence, convert_node_occurrence):
        """
        Helper method to validate the graph of a model with static LSTM.
        """
        qconfig_mapping = get_default_qconfig_mapping()
        prepare_custom_config = PrepareCustomConfig() \
            .set_float_to_observed_mapping(torch.nn.LSTM, torch.ao.nn.quantizable.LSTM)
        convert_custom_config = ConvertCustomConfig() \
            .set_observed_to_quantized_mapping(torch.ao.nn.quantizable.LSTM, torch.ao.nn.quantized.LSTM)
        example_inputs = (torch.rand(5, 3, 50), torch.rand(1, 3, 50), torch.randn(1, 3, 50))

        model = prepare_fx(model, qconfig_mapping, example_inputs, prepare_custom_config=prepare_custom_config)
        self.checkGraphModuleNodes(model, expected_node_occurrence=prepare_node_occurrence)
        model(*example_inputs)

        model = convert_fx(model, convert_custom_config=convert_custom_config)
        self.checkGraphModuleNodes(model, expected_node_occurrence=convert_node_occurrence)
        model(*example_inputs)

    def test_static_lstm(self):
        """
        Test statically quantized custom module LSTM followed by ops that consume individual
        tensors of the output tuple.
        """
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(50, 50, 1)
                self.linear1 = nn.Linear(50, 10)
                self.linear2 = nn.Linear(50, 10)
                self.linear3 = nn.Linear(50, 10)

            def forward(self, inputs: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor):
                (out, (h0_out, c0_out)) = self.lstm(inputs, (h0, c0))
                out = self.linear1(out)
                h0_out = self.linear2(h0_out)
                c0_out = self.linear3(c0_out)
                return (out, (h0_out, c0_out))

        m = MyModel()
        prepare_node_occurrence = {
            ns.call_module(torch.ao.nn.quantizable.LSTM): 1,
        }
        convert_node_occurrence = {
            ns.call_module(torch.ao.nn.quantized.LSTM): 1,
            ns.call_function(torch.quantize_per_tensor): 3,
            # lstm[0].dequantize()
            # lstm[1][0].dequantize()
            # lstm[1][1].dequantize()
            ns.call_method("dequantize"): 3,
            # lstm[0], lstm[1], lstm[1][0], lstm[1][1]
            ns.call_function(operator.getitem): 4,
            # No tuples are consumed
            ns.call_function(tuple): 0,
        }
        self._test_static_lstm_helper(m, prepare_node_occurrence, convert_node_occurrence)

    def test_static_lstm_consume_tuple(self):
        """
        Test statically quantized custom module LSTM followed by a module that consumes the
        output tuple, either as a whole or part of it.
        """
        class ModuleAfterLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.identity = torch.nn.Identity()

            def forward(self, x):
                return self.identity(x)

        class ConsumeWholeTuple(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(50, 50, 1)
                self.module_after_lstm = ModuleAfterLSTM()

            def forward(self, inputs: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor):
                x = self.lstm(inputs, (h0, c0))
                x = self.module_after_lstm(x)  # consume tuple (output, (hidden0, hidden1))
                return x

        class ConsumeHiddenTuple(ConsumeWholeTuple):
            def forward(self, inputs: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor):
                x = self.lstm(inputs, (h0, c0))
                x = self.module_after_lstm(x[1])  # consume tuple (hidden0, hidden1)
                return x

        # Test consuming the whole tuple (output, (hidden0, hidden1))
        m1 = ConsumeWholeTuple()
        prepare_node_occurrence = {
            ns.call_module(torch.ao.nn.quantizable.LSTM): 1,
        }
        convert_node_occurrence1 = {
            ns.call_module(torch.ao.nn.quantized.LSTM): 1,
            ns.call_function(torch.quantize_per_tensor): 3,
            # lstm[0].dequantize()
            # lstm[1][0].dequantize()
            # lstm[1][1].dequantize()
            ns.call_method("dequantize"): 3,
            # lstm[0], lstm[1], lstm[1][0], lstm[1][1]
            ns.call_function(operator.getitem): 4,
            # tuple(output_dq, tuple(hidden0_dq, hidden1_dq))
            ns.call_function(tuple): 2,
        }
        self._test_static_lstm_helper(m1, prepare_node_occurrence, convert_node_occurrence1)

        # Test consuming just the hidden tuple (hidden0, hidden1)
        m2 = ConsumeHiddenTuple()
        convert_node_occurrence2 = {
            ns.call_module(torch.ao.nn.quantized.LSTM): 1,
            ns.call_function(torch.quantize_per_tensor): 3,
            # lstm[1][0].dequantize()
            # lstm[1][1].dequantize()
            ns.call_method("dequantize"): 2,
            # lstm[1], lstm[1][0], lstm[1][1]
            ns.call_function(operator.getitem): 3,
            # tuple(hidden0_dq, hidden1_dq)
            ns.call_function(tuple): 1,
        }
        self._test_static_lstm_helper(m2, prepare_node_occurrence, convert_node_occurrence2)

    def test_reroute_tuple_getitem_patterns(self):
        """
        The following graph should redirect the output to `b`. After the transformation,
        all other nodes, including the inputs `a` and `c`, are no longer needed.

             a   b     c
             |   \\   /
             \\   tuple
              \\   /
               tuple
               /  \\
              /    \\
             |      \\
             |       \\
             |        \\
        getitem0    getitem1
             |      /     \\
             | getitem0  getitem1
             |     \\     /
             \\      tuple
              \\      /
               \\    /
                tuple
                  |
               getitem1
                  |
               getitem0
                  |
                output
        """
        # Construct graph manually because symbolic_trace does not insert tuple and getitem nodes
        graph = torch.fx.Graph()
        a = graph.create_node("placeholder", "a")
        b = graph.create_node("placeholder", "b")
        c = graph.create_node("placeholder", "c")
        bc = graph.call_function(tuple, args=([b, c],))
        abc = graph.call_function(tuple, args=([a, bc],))

        # Break down tuple and reconstruct it again
        a2 = graph.call_function(operator.getitem, args=(abc, 0))
        bc2 = graph.call_function(operator.getitem, args=(abc, 1))
        b2 = graph.call_function(operator.getitem, args=(bc2, 0))
        c2 = graph.call_function(operator.getitem, args=(bc2, 1))
        bc3 = graph.call_function(tuple, args=([b2, c2],))
        abc2 = graph.call_function(tuple, args=([a2, bc3],))

        # Output tuple[1][0]
        bc4 = graph.call_function(operator.getitem, args=(abc2, 1))
        b3 = graph.call_function(operator.getitem, args=(bc4, 0))
        output = graph.output(b3)

        # Do reroute
        _reroute_tuple_getitem_pattern(graph)

        # Assert that output reroutes to `b` directly, and all other nodes can be removed
        output_ancestors = []
        def gather_ancestors(current_node):  # noqa: E306
            for arg in current_node.args:
                output_ancestors.append(arg)
                gather_ancestors(arg)
        gather_ancestors(output)
        self.assertEqual(output_ancestors, [b])
        self.assertEqual(output.args[0], b)

    def test_relu_lowering(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.relu(x)

        m = M().eval()
        m = prepare_fx(m, {"": default_qconfig}, example_inputs=(torch.randn(1),))
        m_copy = copy.deepcopy(m)
        m = convert_fx(m)
        m_ref = convert_to_reference_fx(m_copy)
        node_occurrence = {
            ns.call_function(torch.quantize_per_tensor): 1,
            ns.call_method("dequantize"): 1
        }
        node_occurrence_ref = {
            ns.call_function(torch.quantize_per_tensor): 2,
            ns.call_method("dequantize"): 2
        }

        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)
        self.checkGraphModuleNodes(m_ref, expected_node_occurrence=node_occurrence_ref)

    @skipIfNoFBGEMM
    def test_dynamic_with_fusion(self):
        """
        Tests that dynamic quantization APIs work with Linear + Relu fusion
        """
        with override_quantized_engine('fbgemm'):
            class LinearRelu(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(5, 5)
                    self.relu = torch.nn.ReLU()

                def forward(self, x):
                    x = self.linear(x)
                    return self.relu(x)

            class Linear(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.w = torch.ones(5, 5)
                    self.b = torch.zeros(5)

                def forward(self, x):
                    return torch.nn.functional.linear(x, self.w, self.b)

            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mods1 = torch.nn.Sequential(LinearRelu(), LinearRelu())
                    self.mods2 = Linear()
                    self.relu = F.relu

                def forward(self, x):
                    x = self.mods1(x)
                    x = self.mods2(x)
                    x = self.relu(x)
                    return x

            dynamic_quantized_ops = {
                float16_dynamic_qconfig: torch.ops.quantized.linear_relu_dynamic_fp16,
                default_dynamic_qconfig: torch.ops.quantized.linear_relu_dynamic
            }
            for qconfig in [float16_dynamic_qconfig, default_dynamic_qconfig]:
                model = M().eval()
                qconfig_dict = {
                    "": qconfig
                }
                example_inputs = (torch.rand(5, 5),)
                m = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)
                m = convert_fx(m)
                m(*example_inputs)
                node_list = [
                    ns.call_module(nniqd.LinearReLU),
                    ns.call_module(nniqd.LinearReLU),
                    ns.call_function(dynamic_quantized_ops[qconfig]),
                ]
                self.checkGraphModuleNodes(m, expected_node_list=node_list)

    @skipIfNoFBGEMM
    def test_dynamic_with_fusion_multiple_uses(self):
        """
        Tests that dynamic quantization APIs work with Linear + Relu fusion
        """
        with override_quantized_engine('fbgemm'):
            class LinearRelu(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(5, 5)
                    self.relu = torch.nn.ReLU()

                def forward(self, x):
                    x = self.linear(x)
                    return self.relu(x)

            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear_relu = LinearRelu()

                def forward(self, x):
                    x = self.linear_relu(x)
                    x = self.linear_relu(x)
                    return x

            for qconfig in [float16_dynamic_qconfig, default_dynamic_qconfig]:
                model = M().eval()
                qconfig_dict = {
                    "": qconfig
                }
                example_inputs = (torch.randn(5, 5),)
                m = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)
                m = convert_fx(m)
                m(*example_inputs)
                node_list = [
                    ns.call_module(nniqd.LinearReLU),
                    ns.call_module(nniqd.LinearReLU),
                ]
                self.checkGraphModuleNodes(m, expected_node_list=node_list)

    @skipIfNoFBGEMM
    def test_dynamic_linear_input_multiple_use(self):
        """
        Tests input for dynamic linear being used by multiple ops
        """
        with override_quantized_engine('fbgemm'):
            class LinearRelu(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(5, 5)
                    self.relu = torch.nn.ReLU()

                def forward(self, x):
                    x = self.linear(x)
                    return self.relu(x)

            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mod1 = LinearRelu()
                    self.mod2 = LinearRelu()

                def forward(self, x):
                    y1 = self.mod1(x)
                    y2 = self.mod2(x)
                    return y1 + y2

            for qconfig in [float16_dynamic_qconfig, default_dynamic_qconfig]:
                model = M().eval()
                qconfig_dict = {
                    "": qconfig
                }
                example_inputs = (torch.rand(5, 5, 5),)
                m = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)
                m = convert_fx(m)
                m(*example_inputs)
                node_list = [
                    ns.call_module(nniqd.LinearReLU),
                    ns.call_module(nniqd.LinearReLU),
                ]
                self.checkGraphModuleNodes(m, expected_node_list=node_list)

    def test_ref_linear_module(self):
        """ Make sure the numerics for models with ref linear module
        matches models with fbgemm/qnnpack module
        """
        class M1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        class M2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.linear(x))

        for M in [M1, M2]:
            m = M().eval()
            example_inputs = (torch.randn(5, 10),)
            m = prepare_fx(m, {"": default_qconfig}, example_inputs=example_inputs)
            m_copy = copy.deepcopy(m)
            m = convert_fx(m)
            m_ref = convert_to_reference_fx(m_copy)
            result = m(*example_inputs)
            result_ref = m_ref(*example_inputs)
            self.assertTrue(torch.equal(result, result_ref))

    def test_ref_conv_module(self):
        """ Make sure the numerics for models with ref conv module
        matches models with fbgemm/qnnpack module
        """
        convs = {
            1: nn.Conv1d,
            2: nn.Conv2d,
            3: nn.Conv3d,
        }

        class M1(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.conv = convs[dim](3, 3, 3)

            def forward(self, x):
                return self.conv(x)

        class M2(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.conv = convs[dim](3, 3, 3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.conv(x))

        for dim, M in itertools.product([1, 2, 3], [M1, M2]):
            m = M(dim).eval()
            data = self.img_data_dict[dim][0][0]
            m = prepare_fx(m, {"": default_qconfig}, example_inputs=(data,))
            m_copy = copy.deepcopy(m)
            m = convert_fx(m)
            m_ref = convert_to_reference_fx(m_copy)
            result = m(data)
            result_ref = m_ref(data)
            self.assertTrue(torch.equal(result, result_ref))

    def test_sub_scalar(self):
        class M(torch.nn.Module):
            def forward(self, x):
                x = x + 1
                x = x - 1
                x = x + 3
                x = x - 4
                return x

        m = M().eval()
        m = prepare_fx(m, {"": default_qconfig}, example_inputs=(torch.rand(3),))
        m = convert_fx(m)
        occurrence = {
            ns.call_function(torch.quantize_per_tensor): 2,
            ns.call_method("dequantize"): 2
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=occurrence)

    def test_observer_fqn(self):
        """
        Test to make sure the observer FQN is based on the quantizable op/module that it is observing
        and uses the modules FQN to determine the observer name.
        """
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.ones(5, 5)
                self.b = torch.zeros(5)


            def forward(self, x):
                return torch.nn.functional.linear(x, self.w, self.b)


        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mods1 = torch.nn.Sequential(
                    Linear(),
                    Linear()
                )
                self.mods2 = Linear()
                self.mods3 = torch.nn.Linear(5, 5)

            def forward(self, x):
                x = self.mods1(x)
                x = torch.add(x, 4)
                x = self.mods2(x)
                y = torch.add(x, 2)
                z = torch.mul(x, 5)
                a = self.mods3(y)
                return a, z

        model = M().eval()

        prepared = prepare_fx(model, {"": default_qconfig}, example_inputs=(torch.randn(1, 5)))
        name_list = []
        for name, mod in prepared.named_modules():
            if isinstance(mod, torch.ao.quantization.observer.MinMaxObserver):
                name_list.append(name)
        expected_name_list = ['activation_post_process_0',
                              'activation_post_process_1',
                              'activation_post_process_2',
                              'activation_post_process_3',
                              'activation_post_process_4',
                              'activation_post_process_6',
                              'activation_post_process_7',
                              'activation_post_process_10']
        assert name_list == expected_name_list

    def test_conv_lowering(self):
        convs = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
        qconvs = {1: nn.quantized.Conv1d, 2: nn.quantized.Conv2d, 3: nn.quantized.Conv3d}

        class M(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.conv = convs[dim](3, 3, 3)

            def forward(self, x):
                return self.conv(x)

        for dim in range(1, len(convs) + 1):
            m = M(dim).eval()
            data = self.img_data_dict[dim][0][0]
            m = prepare_fx(m, {"": default_qconfig}, example_inputs=(data,))
            m_ref = copy.deepcopy(m)
            m_ref = convert_to_reference_fx(m_ref)
            m = convert_fx(m)
            out_ref = m_ref(data)
            out = m(data)
            # check that reference pattern for quantized conv module is fused
            expected_node_occurrence = {
                ns.call_function(torch.quantize_per_tensor): 1,
                ns.call_module(qconvs[dim]): 1,
                ns.call_method("dequantize"): 1
            }
            self.checkGraphModuleNodes(m, expected_node_occurrence=expected_node_occurrence)
            # checking result match
            self.assertTrue(torch.equal(out_ref, out))

    def test_convert_qconfig_mapping(self):
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.ones(5, 5)
                self.b = torch.zeros(5)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.w, self.b)


        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mods1 = torch.nn.Sequential(
                    Linear(),
                    Linear()
                )
                self.mods3 = torch.nn.Linear(5, 5)

            def forward(self, x):
                x = self.mods1(x)
                x = torch.add(x, 4)
                z = torch.mul(x, 5)
                x = self.mods3(z)
                return x

        model = M().train()

        for check in ["module_name", "object_type"]:
            qconfig_dict = {"": None,
                            "object_type": [
                                (nn.functional.linear, get_default_qat_qconfig("fbgemm")),
                                (torch.add, get_default_qat_qconfig("fbgemm")),
                                (nn.Linear, get_default_qat_qconfig("fbgemm")),
                            ],
                            }
            example_inputs = (torch.rand(5, 5),)
            prepared = prepare_qat_fx(model, qconfig_dict, example_inputs=example_inputs)
            prepared(*example_inputs)
            if check == "module_name":
                convert_qconfig_dict = {"": None,
                                        "object_type": [
                                            (nn.functional.linear, get_default_qat_qconfig("fbgemm")),
                                            (torch.add, get_default_qat_qconfig("fbgemm")),
                                            (nn.Linear, get_default_qat_qconfig("fbgemm")),
                                        ],
                                        "module_name": [("mods1.0", None)]}

                node_occurrence = {
                    ns.call_function(torch.quantize_per_tensor): 2,
                    ns.call_function(torch.nn.functional.linear): 1,
                    ns.call_function(torch.ops.quantized.linear): 1,
                    ns.call_function(torch.ops.quantized.add): 1,
                    ns.call_method("dequantize"): 2
                }
                order_check = [
                    ns.call_function(torch.nn.functional.linear),
                    ns.call_function(torch.quantize_per_tensor),
                    ns.call_function(torch.ops.quantized.linear),
                    ns.call_function(torch.ops.quantized.add),
                    ns.call_method("dequantize"),
                    ns.call_function(torch.quantize_per_tensor),
                    ns.call_module(nnq.Linear),
                    ns.call_method("dequantize"),
                ]
            elif check == "object_type":
                convert_qconfig_dict = {"": None,
                                        "object_type": [
                                            (nn.functional.linear, get_default_qat_qconfig("fbgemm")),
                                            (torch.add, get_default_qat_qconfig("fbgemm")),
                                            (nn.Linear, None),
                                        ]}

                node_occurrence = {
                    ns.call_function(torch.quantize_per_tensor): 1,
                    ns.call_function(torch.ops.quantized.linear): 2,
                    ns.call_function(torch.ops.quantized.add): 1,
                    ns.call_function(torch.mul): 1,
                    ns.call_method("dequantize"): 1
                }
                order_check = [
                    ns.call_function(torch.quantize_per_tensor),
                    ns.call_function(torch.ops.quantized.linear),
                    ns.call_function(torch.ops.quantized.linear),
                    ns.call_function(torch.ops.quantized.add),
                    ns.call_method("dequantize"),
                    ns.call_function(torch.mul),
                    ns.call_module(nn.Linear),
                ]

            converted = convert_fx(prepared, qconfig_mapping=convert_qconfig_dict)
            converted(torch.rand(5, 5))
            self.checkGraphModuleNodes(
                converted,
                expected_node_occurrence=node_occurrence,
                expected_node_list=order_check)

    def _assertFixedQParamsFakeQuantizeEqual(self, fq1, fq2):
        self.assertEqual(fq1()._observer_ctr, fq2()._observer_ctr)

    def test_register_patterns(self):
        @_register_fusion_pattern("dummy_fusion")
        class DummyFusion():
            pass

        @_register_quant_pattern("dummy_quant")
        class DummyQuant():
            pass

        @_register_quant_pattern("dummy_quant2", default_fixed_qparams_range_0to1_observer)
        class DummyQuant2():
            pass

        @_register_quant_pattern("dummy_quant3", default_fixed_qparams_range_neg1to1_observer)
        class DummyQuant3():
            pass

        self.assertEqual(_DEFAULT_FUSION_PATTERNS["dummy_fusion"], DummyFusion)
        self.assertEqual(_DEFAULT_QUANTIZATION_PATTERNS["dummy_quant"], DummyQuant)
        self.assertEqual(_DEFAULT_QUANTIZATION_PATTERNS["dummy_quant2"], DummyQuant2)
        self.assertEqual(_DEFAULT_QUANTIZATION_PATTERNS["dummy_quant3"], DummyQuant3)
        self.assertEqual(_DEFAULT_OUTPUT_OBSERVER_MAP["dummy_quant2"], default_fixed_qparams_range_0to1_observer)
        self.assertEqual(_DEFAULT_OUTPUT_OBSERVER_MAP["dummy_quant3"], default_fixed_qparams_range_neg1to1_observer)
        self._assertFixedQParamsFakeQuantizeEqual(_DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP["dummy_quant2"],
                                                  default_fixed_qparams_range_0to1_fake_quant)
        self._assertFixedQParamsFakeQuantizeEqual(_DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP["dummy_quant3"],
                                                  default_fixed_qparams_range_neg1to1_fake_quant)
        output_fake_quantize_map = get_default_output_activation_post_process_map(is_training=True)
        output_observer_map = get_default_output_activation_post_process_map(is_training=False)
        self.assertEqual(output_observer_map.get("dummy_quant3"), default_fixed_qparams_range_neg1to1_observer)
        self._assertFixedQParamsFakeQuantizeEqual(output_fake_quantize_map.get("dummy_quant3"),
                                                  default_fixed_qparams_range_neg1to1_fake_quant)



    def test_reuse_input_qconfig(self):
        class M1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                x = self.conv(x)
                x = x.reshape()
                return x

        class M2(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = x.reshape()
                return x

        options = itertools.product([M1, M2], [True, False])
        for M, is_qat in options:
            m = M1().eval()
            example_inputs = (torch.randn(1, 3, 3, 3),)
            m = prepare_fx(m, get_default_qconfig_mapping(), example_inputs=example_inputs)
            m = convert_fx(m)
            node_list = [
                ns.call_function(torch.quantize_per_tensor),
                ns.call_module(nnq.Conv2d),
                ns.call_method("reshape"),
                ns.call_method("dequantize"),
            ]
            self.checkGraphModuleNodes(
                m,
                expected_node_list=node_list)

            m = M2().eval()
            m = prepare_fx(m, get_default_qconfig_mapping(), example_inputs=example_inputs)
            m = convert_fx(m)
            node_occurrence = {
                ns.call_function(torch.quantize_per_tensor): 0,
                ns.call_method("dequnatize"): 0,
            }
            node_list = [
                ns.call_method("reshape"),
            ]
            self.checkGraphModuleNodes(
                m,
                expected_node_occurrence=node_occurrence,
                expected_node_list=node_list)

    def test_stack_trace_preserved_linear(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)

            def forward(self, x):
                x = self.linear(x)
                return x

        m = M().eval()
        mp = prepare_fx(m, get_default_qconfig_mapping(), example_inputs=(torch.randn(1, 1),))

        found_stack_trace = False
        for n in mp.graph.nodes:
            if n.op == 'call_module' and n.target == 'linear':
                found_stack_trace = n.stack_trace is not None
                break
        self.assertTrue(found_stack_trace)

        # test reference model
        mq = convert_to_reference_fx(copy.deepcopy(mp))
        found_stack_trace = False
        for n in mq.graph.nodes:
            if n.op == 'call_module' and n.target == 'linear':
                found_stack_trace = n.stack_trace is not None
                break
        self.assertTrue(found_stack_trace, f"stack trace not found, node: {n.format_node()}, is_reference: True")

        # test quantized model
        mq = convert_fx(mp)
        found_stack_trace = False
        for n in mq.graph.nodes:
            if n.op == 'call_module' and n.target == 'linear':
                found_stack_trace = n.stack_trace is not None
                break
        self.assertTrue(found_stack_trace, f"stack trace not found, node: {n.format_node()}, is_reference: False")

    def test_qat_skip_untraced(self):
        class UnTraceableModuleClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        class UnTraceableModuleName(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.untraceable_module_class = UnTraceableModuleClass()
                self.untraceable_module_name = UnTraceableModuleClass()

            def forward(self, x):
                x = self.untraceable_module_class(x)
                x = self.untraceable_module_name(x)
                return x

        mod = M()

        qconfig_dict = {"": torch.quantization.get_default_qat_qconfig()}
        prepare_custom_config_dict = {
            "non_traceable_module_class": [UnTraceableModuleClass],
            "non_traceable_module_name": ["untraceable_module_name"],
        }
        example_inputs = (torch.randn(2, 2),)
        mod_prep = torch.ao.quantization.quantize_fx.prepare_qat_fx(
            mod.train(), qconfig_dict, example_inputs=example_inputs,
            prepare_custom_config=prepare_custom_config_dict
        )
        mod_prep = torch.ao.quantization.quantize_fx.prepare_qat_fx(
            mod.train(), qconfig_dict, example_inputs=example_inputs,
            prepare_custom_config=prepare_custom_config_dict
        )
        self.assertTrue(
            isinstance(mod_prep.untraceable_module_class.linear, torch.nn.Linear)
        )
        self.assertTrue(
            isinstance(mod_prep.untraceable_module_name.linear, torch.nn.Linear)
        )
        self.assertTrue(
            type(mod_prep.untraceable_module_class.linear)
            is not torch.ao.nn.qat.modules.linear.Linear,
            "prepare_qat_fx shold not convert anything inside untraced module classes",
        )
        self.assertTrue(
            type(mod_prep.untraceable_module_name.linear)
            is not torch.ao.nn.qat.modules.linear.Linear,
            "prepare_qat_fx shold not convert anything inside modules named in untraced_module_names",
        )

    def test_qconfig_dict_setup(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.Conv1d = torch.nn.Conv1d(1, 1, 1)
                self.Conv2d = torch.nn.Conv2d(1, 1, 1)
                self.Conv3d = torch.nn.Conv3d(1, 1, 1)
                self.ConvTranspose1d = torch.nn.ConvTranspose1d(1, 1, 1)
                self.ConvTranspose2d = torch.nn.ConvTranspose2d(1, 1, 1)
                self.ConvTranspose3d = torch.nn.ConvTranspose3d(1, 1, 1)
                self.Linear = torch.nn.Linear(1, 1, 1)

            def forward(self, x):
                x = self.Conv1d(x)
                x = self.Conv2d(x)
                x = self.Conv3d(x)
                x = self.ConvTranspose1d(x)
                x = self.ConvTranspose2d(x)
                x = self.ConvTranspose3d(x)
                x = self.Linear(x)
                x = torch.nn.functional.conv1d(x, torch.rand(2, 2))
                x = torch.nn.functional.conv2d(x, torch.rand(2, 2))
                x = torch.nn.functional.conv3d(x, torch.rand(2, 2))
                x = torch.nn.functional.linear(x, torch.rand(2, 2))
                return x

        backends = ["qnnpack", "fbgemm"]
        for func in [get_default_qconfig_mapping, get_default_qat_qconfig_mapping]:
            for backend in backends:
                m = M().eval()
                qconfig_dict = func(backend)
                m = prepare_fx(m, qconfig_dict, example_inputs=(torch.randn(1, 1, 1, 1)))
                for name, mod in m.named_modules():
                    if _is_activation_post_process(mod) and mod.dtype == torch.quint8:
                        if backend == "fbgemm":
                            lower_bnd = 0
                            upper_bnd = 127
                        else:
                            lower_bnd = 0
                            upper_bnd = 255
                        if issubclass(type(mod), FakeQuantize):
                            self.assertEqual(mod.activation_post_process.quant_min, lower_bnd)
                            self.assertEqual(mod.activation_post_process.quant_max, upper_bnd)
                        else:
                            self.assertEqual(mod.quant_min, lower_bnd)
                            self.assertEqual(mod.quant_max, upper_bnd)

    def test_prepare_mode(self):
        class LinearModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 10)

            def forward(self, x):
                return self.linear(x)

        def _test(prepare_fn, qconfig_dict):
            m = LinearModel()
            m1 = copy.deepcopy(m)
            m1.train()
            example_inputs = (torch.randn(1, 5),)
            prepare_fn(m1, qconfig_dict, example_inputs=example_inputs)
            m2 = copy.deepcopy(m)
            m2.eval()
            prepare_fn(m2, qconfig_dict, example_inputs=example_inputs)

        # Ensure prepare_fx and prepare_qat_fx work in both training and eval modes
        _test(prepare_fx, get_default_qconfig_mapping())
        _test(prepare_qat_fx, get_default_qat_qconfig_mapping())

    def _validate_qconfig_against_backend_config_constraints(
            self,
            model: torch.nn.Module,
            qconfig: QConfig,
            backend_config: BackendConfig,
            satisfies_constraints: bool,
            qconfig_name: Optional[str] = None):
        """
        Helper method to validate whether `qconfig` satisfies the constraints specified in `backend_config`.
        """
        qconfig_mapping = QConfigMapping().set_object_type(torch.nn.Linear, qconfig)
        example_inputs = (torch.rand((1, 30), dtype=torch.float),)
        model = prepare_fx(model, qconfig_mapping, example_inputs, backend_config=backend_config)
        model(*example_inputs)
        model = convert_fx(model, backend_config=backend_config)
        if satisfies_constraints:
            expected_node_occurrence = {
                ns.call_module(torch.ao.nn.quantized.Linear) : 1,
                ns.call_module(torch.nn.Linear) : 0,
            }
        else:
            expected_node_occurrence = {
                ns.call_module(torch.ao.nn.quantized.Linear) : 0,
                ns.call_module(torch.nn.Linear) : 1,
            }
        try:
            self.checkGraphModuleNodes(model, expected_node_occurrence=expected_node_occurrence)
        except AssertionError as e:
            if qconfig_name is not None:
                print("ERROR: Validation for QConfig '%s' failed" % qconfig_name)
            raise e

    def test_backend_config_quantization_range(self):
        """
        Check that quantization ranges specified through the BackendConfig are reflected in
        the observers inserted into the model.
        """
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.linear = torch.nn.Linear(30, 4).float()

            def forward(self, x):
                return self.linear(x)

        dtype_config = DTypeConfig(
            input_dtype=DTypeWithConstraints(
                dtype=torch.quint8,
                quant_min_lower_bound=0,
                quant_max_upper_bound=31,
            ),
            output_dtype=DTypeWithConstraints(
                dtype=torch.quint8,
                quant_min_lower_bound=0,
                quant_max_upper_bound=31,
            ),
            weight_dtype=DTypeWithConstraints(
                dtype=torch.qint8,
                quant_min_lower_bound=-64,
                quant_max_upper_bound=63,
            ),
            bias_dtype=torch.float,
        )
        backend_config = BackendConfig() \
            .set_backend_pattern_config(BackendPatternConfig(torch.nn.Linear)
                .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # noqa: E128
                .add_dtype_config(dtype_config)
                .set_root_module(torch.nn.Linear)
                .set_reference_quantized_module(nnqr.Linear))

        def validate_qconfig(qconfig: QConfig, satisfies_constraints: bool):
            self._validate_qconfig_against_backend_config_constraints(
                MyModel(), qconfig, backend_config, satisfies_constraints)

        # Case 1: QConfig ranges fit within backend ranges, OK
        qconfig1 = QConfig(
            activation=MinMaxObserver.with_args(quant_min=0, quant_max=15, dtype=torch.quint8),
            weight=MinMaxObserver.with_args(quant_min=-32, quant_max=31, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
        validate_qconfig(qconfig1, satisfies_constraints=True)

        # Case 2: QConfig activation range falls outside backend range, should fail
        qconfig2 = QConfig(
            activation=MinMaxObserver.with_args(quant_min=0, quant_max=63, dtype=torch.quint8),
            weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
        validate_qconfig(qconfig2, satisfies_constraints=False)

        # Case 3: QConfig weight range falls outside backend range, should fail
        qconfig3 = QConfig(
            activation=MinMaxObserver.with_args(dtype=torch.quint8),
            weight=MinMaxObserver.with_args(quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
        validate_qconfig(qconfig3, satisfies_constraints=False)

        # Case 4: QConfig doesn't specify range, should fail
        qconfig4 = QConfig(activation=ReuseInputObserver, weight=ReuseInputObserver)
        validate_qconfig(qconfig4, satisfies_constraints=False)

    def test_backend_config_scale_min(self):
        """
        Test QConfig eps validation against the BackendConfig's min scale value.
        """
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.linear = torch.nn.Linear(30, 4).float()

            def forward(self, x):
                return self.linear(x)

        dtype_config = DTypeConfig(
            input_dtype=DTypeWithConstraints(dtype=torch.quint8, scale_min_lower_bound=2 ** -12),
            output_dtype=DTypeWithConstraints(dtype=torch.quint8, scale_min_lower_bound=2 ** -12),
            weight_dtype=DTypeWithConstraints(dtype=torch.qint8, scale_min_lower_bound=2 ** -12),
            bias_dtype=torch.float,
        )

        backend_config = BackendConfig() \
            .set_backend_pattern_config(BackendPatternConfig(torch.nn.Linear)
                .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # noqa: E128
                .add_dtype_config(dtype_config)
                .set_root_module(torch.nn.Linear)
                .set_reference_quantized_module(nnqr.Linear))

        def validate_qconfig(qconfig: QConfig, satisfies_constraints: bool):
            self._validate_qconfig_against_backend_config_constraints(
                MyModel(), qconfig, backend_config, satisfies_constraints)

        # Case 1: QConfig min scale value == backend min scale value, OK
        qconfig1 = QConfig(
            activation=MinMaxObserver.with_args(dtype=torch.quint8, eps=2 ** -12),
            weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, eps=2 ** -12))
        validate_qconfig(qconfig1, satisfies_constraints=True)

        # Case 2: QConfig min scale value > backend min scale value, OK
        qconfig2 = QConfig(
            activation=MinMaxObserver.with_args(dtype=torch.quint8, eps=2 ** -10),
            weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, eps=2 ** -10))
        validate_qconfig(qconfig2, satisfies_constraints=True)

        # Case 3: QConfig activation min scale value < backend min scale value, should fail
        qconfig3 = QConfig(
            activation=MinMaxObserver.with_args(dtype=torch.quint8, eps=2 ** -14),
            weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
        validate_qconfig(qconfig3, satisfies_constraints=False)

        # Case 3: QConfig weight min scale value < backend min scale value, should fail
        qconfig4 = QConfig(
            activation=MinMaxObserver.with_args(dtype=torch.quint8),
            weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, eps=2 ** -14))
        validate_qconfig(qconfig4, satisfies_constraints=False)

        # Case 5: QConfig doesn't specify eps, should fail
        qconfig5 = QConfig(
            activation=FixedQParamsObserver.with_args(scale=1.0, zero_point=0),
            weight=FixedQParamsObserver.with_args(scale=1.0, zero_point=0))
        validate_qconfig(qconfig5, satisfies_constraints=False)

    def test_qnnpack_backend_config(self):
        """
        Test whether default QNNPACK QConfigs are compatible with the QNNPACK BackendConfig.
        """
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.linear = torch.nn.Linear(30, 4).float()

            def forward(self, x):
                return self.linear(x)

        all_qconfigs: List[Tuple[QConfig, str]] = [
            (get_default_qconfig("qnnpack", version=0), "default_qnnpack_qconfig_v0"),
            (get_default_qat_qconfig("qnnpack", version=0), "default_qat_qnnpack_qconfig_v0"),
            (get_default_qat_qconfig("qnnpack", version=1), "default_qat_qnnpack_qconfig_v1"),
            (default_symmetric_qnnpack_qconfig, "default_symmetric_qnnpack_qconfig"),
            (default_symmetric_qnnpack_qat_qconfig, "default_symmetric_qnnpack_qat_qconfig"),
            # TODO: Test these QConfigs once they are fixed, see https://github.com/pytorch/pytorch/issues/85862
            # (default_per_channel_symmetric_qnnpack_qconfig, "default_per_channel_symmetric_qnnpack_qconfig"),
            # (default_per_channel_symmetric_qnnpack_qat_qconfig, "default_per_channel_symmetric_qnnpack_qat_qconfig"),
        ]
        backend_config = get_qnnpack_backend_config()
        for qconfig, qconfig_name in all_qconfigs:
            self._validate_qconfig_against_backend_config_constraints(
                MyModel(), qconfig, backend_config, satisfies_constraints=True, qconfig_name=qconfig_name)

    def test_symmetric_qnnpack_qconfig_mapping(self):
        """
        Test whether `torch.ao.quantization.qconfig_mapping._get_symmetric_qnnpack_qconfig_mapping`
        works with the QNNPACK BackendConfig.
        """
        if "qnnpack" not in supported_qengines:
            return

        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.linear = torch.nn.Linear(30, 4).float()

            def forward(self, x):
                return self.linear(x)

        with override_quantized_engine("qnnpack"):
            qconfig_mapping = _get_symmetric_qnnpack_qconfig_mapping()
            example_inputs = (torch.rand((1, 30), dtype=torch.float),)
            backend_config = get_qnnpack_backend_config()
            model = MyModel()
            model = prepare_fx(model, qconfig_mapping, example_inputs, backend_config=backend_config)
            model(*example_inputs)
            model = convert_fx(model, backend_config=backend_config)
            expected_node_occurrence = {
                ns.call_module(torch.ao.nn.quantized.Linear) : 1,
                ns.call_module(torch.nn.Linear) : 0,
            }
            self.checkGraphModuleNodes(model, expected_node_occurrence=expected_node_occurrence)
            model(*example_inputs)

    def test_get_executorch_backend_config(self):
        from torch.ao.quantization.backend_config import get_executorch_backend_config
        # make sure this runs
        executorch_backend_config = get_executorch_backend_config()

    def test_backend_config_check_for_weight_and_bias(self):
        """ Test to make sure the backend_config check for weight and bias
        runs when the qconfig is None for the ops with weight and bias
        previously the error was not hit because we first check input, and
        the check for weight and bias are skipped.
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.tensor((5, 5))
                self.bias = torch.tensor((5,))

            def forward(self, x):
                return torch.addmm(self.bias, x, self.weight)

        m = M().eval()
        qconfig_mapping = QConfigMapping()
        observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
        weighted_op_quint8_dtype_config = DTypeConfig(
            input_dtype=torch.quint8,
            output_dtype=torch.quint8,
            weight_dtype=torch.qint8,
            bias_dtype=torch.float,
        )
        dtype_configs = [weighted_op_quint8_dtype_config]
        backend_pattern_config = BackendPatternConfig(torch.addmm) \
            .set_observation_type(observation_type) \
            .set_dtype_configs(dtype_configs) \
            ._set_input_type_to_index({"weight": 2, "bias": 0})
        backend_config = BackendConfig() \
            .set_backend_pattern_config(backend_pattern_config)
        example_inputs = (torch.rand(1, 5),)
        # make sure this runs
        m = prepare_fx(m, qconfig_mapping, example_inputs, backend_config=backend_config)

    def test_get_default_qconfig_valid_backend(self):
        """ Checks that AssertionError is raised when non expected backend input is specified
        """
        invalid_backends = ["imaginary_backend", 3]
        for invalid_backend in invalid_backends:
            with self.assertRaisesRegex(AssertionError, "not supported"):
                qconfig = get_default_qconfig(invalid_backend)
            with self.assertRaisesRegex(AssertionError, "not supported"):
                qconfig = get_default_qat_qconfig(invalid_backend)
            with self.assertRaisesRegex(AssertionError, "not supported"):
                qconfig_mapping = get_default_qconfig_mapping(invalid_backend)
            with self.assertRaisesRegex(AssertionError, "not supported"):
                qconfig_mapping = get_default_qat_qconfig_mapping(invalid_backend)

    def test__convert_to_reference_decomposed_fx(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 10)

            def forward(self, x):
                return self.linear(x)

        m = M().eval()
        qconfig_mapping = get_default_qconfig_mapping("fbgemm")
        example_inputs = (torch.randn(1, 5),)
        m = prepare_fx(m, qconfig_mapping, example_inputs)
        m_ref = copy.deepcopy(m)
        m_ref = convert_to_reference_fx(m_ref)
        m = _convert_to_reference_decomposed_fx(m)
        expected_occurrence = {
            ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor): 2,
            ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor): 2,
        }
        self.checkGraphModuleNodes(
            m,
            expected_node_occurrence=expected_occurrence)
        # make sure it runs
        res_ref = m_ref(*example_inputs)
        res = m(*example_inputs)
        self.assertEqual(res, res_ref)

    def test_change_backend_config_for_fixed_qparam_ops(self):
        """ Making sure we can skip validation of qconfigs for fixedqparam ops based
        on BackendConfig
        """
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.tanh = torch.nn.Tanh()

            def forward(self, x: torch.Tensor):
                x = self.tanh(x)
                return x

        model = M().eval()
        # we set a global default_qconfig, which will be ignored since the backend
        # we defined doesn't support anything
        # this is to make sure we don't validate the qconfig when BackendConfig does not
        # have fixed qparam op related configurations
        qconfig_mapping = QConfigMapping().set_global(default_qconfig)
        backend_config = BackendConfig()
        # make sure this runs
        model = prepare_fx(
            model,
            qconfig_mapping=qconfig_mapping,
            example_inputs=(torch.randn(1, 2, 3, 4),),
            backend_config=backend_config
        )

@skipIfNoFBGEMM
class TestQuantizeFxOps(QuantizationTestCase):
    def setUp(self):
        super().setUp()
        self.custom_qconfig = torch.ao.quantization.QConfig(
            activation=torch.ao.quantization.observer.HistogramObserver.with_args(
                qscheme=torch.per_tensor_symmetric, dtype=torch.qint8
            ),
            weight=torch.ao.quantization.default_per_channel_weight_observer
        )
        self.common_quant_patterns = {
            torch.nn.ConvTranspose1d: DefaultNodeQuantizeHandler,
            torch.nn.ConvTranspose2d: DefaultNodeQuantizeHandler,
            torch.nn.ELU: DefaultNodeQuantizeHandler,
            torch.nn.LeakyReLU: DefaultNodeQuantizeHandler,
            torch.nn.Hardswish: DefaultNodeQuantizeHandler,
            torch.nn.InstanceNorm1d: DefaultNodeQuantizeHandler,
            torch.nn.InstanceNorm2d: DefaultNodeQuantizeHandler,
            torch.nn.InstanceNorm3d: DefaultNodeQuantizeHandler,
            torch.nn.LayerNorm: DefaultNodeQuantizeHandler,
            torch.nn.SiLU: DefaultNodeQuantizeHandler,
            torch.nn.Mish: DefaultNodeQuantizeHandler,
            torch.nn.GELU: DefaultNodeQuantizeHandler,
            torch.nn.Softmax: DefaultNodeQuantizeHandler,
            torch.nn.functional.elu: DefaultNodeQuantizeHandler,
            torch.nn.functional.hardswish: DefaultNodeQuantizeHandler,
            torch.nn.functional.instance_norm: DefaultNodeQuantizeHandler,
            torch.nn.functional.layer_norm: DefaultNodeQuantizeHandler,
            torch.nn.functional.leaky_relu: DefaultNodeQuantizeHandler,
            torch.nn.functional.silu: DefaultNodeQuantizeHandler,
            torch.nn.functional.mish: DefaultNodeQuantizeHandler,
            torch.nn.functional.gelu: DefaultNodeQuantizeHandler,
            torch.nn.functional.softmax: DefaultNodeQuantizeHandler,
            torch.sum: DefaultNodeQuantizeHandler
        }

    """Unit tests for individual ops
    """
    @skipIfNoFBGEMM
    def test_linear_module(self):
        with override_quantized_engine('fbgemm'):
            class LinearModel(torch.nn.Module):
                def __init__(self):
                    super(LinearModel, self).__init__()
                    self.linear = torch.nn.Linear(30, 4).float()

                def forward(self, x):
                    return self.linear(x)

            class LinearReLUModel(torch.nn.Module):
                def __init__(self, f_relu=False):
                    super(LinearReLUModel, self).__init__()
                    self.linear = torch.nn.Linear(30, 4).float()
                    if f_relu:
                        self.relu = F.relu
                    else:
                        self.relu = torch.nn.ReLU()

                def forward(self, x):
                    x = self.linear(x)
                    x = self.relu(x)
                    return x

            class LinearBnModel(torch.nn.Module):
                def __init__(self):
                    super(LinearBnModel, self).__init__()
                    self.linear = torch.nn.Linear(4, 4).float()
                    self.bn = torch.nn.BatchNorm1d(4)

                def forward(self, x):
                    x = self.linear(x)
                    x = self.bn(x)
                    return x

            # Test linear
            data = (torch.rand((1, 30), dtype=torch.float),)
            for quant_type in self.all_quant_types:
                model = LinearModel()
                quantized_module = nnqd.Linear if quant_type == QuantType.DYNAMIC else nnq.Linear
                quantized_node = ns.call_module(quantized_module)
                result_dict = self.checkGraphModeFxOp(model, data, quant_type, quantized_node)
                if quant_type in self.static_quant_types:
                    self.assertEqual(result_dict["quantized_output"], result_dict["quantized_reference_output"])

            # TODO: enable test for dynamic quant
            # Test linear-relu
            for f_relu, quant_type in itertools.product([True, False], [QuantType.STATIC, QuantType.QAT]):
                model = LinearReLUModel(f_relu)
                quantized_node = ns.call_module(nniq.LinearReLU)
                result_dict = self.checkGraphModeFxOp(model, data, quant_type, quantized_node)
                self.assertEqual(result_dict["quantized_output"], result_dict["quantized_reference_output"])

            # Test linear-bn
            data = (torch.rand((4, 4), dtype=torch.float),)
            for quant_type in self.static_quant_types:
                model = LinearBnModel()
                quantized_node = ns.call_module(nnq.Linear)
                result_dict = self.checkGraphModeFxOp(model, data, quant_type, quantized_node)
                self.assertEqual(result_dict["quantized_output"], result_dict["quantized_reference_output"])

    @skipIfNoFBGEMM
    def test_functional_linear(self):
        with override_quantized_engine('fbgemm'):
            class FuncLinear(torch.nn.Module):
                def __init__(self, use_bias, has_relu, f_relu):
                    super(FuncLinear, self).__init__()
                    self.w = torch.randn(4, 30)
                    self.b = torch.randn(4)
                    self.use_bias = use_bias
                    if has_relu:
                        if f_relu:
                            self.relu_or_id = F.relu
                        else:
                            self.relu_or_id = torch.nn.ReLU()
                    else:
                        self.relu_or_id = torch.nn.Identity()

                def forward(self, x):
                    if self.use_bias:
                        x = F.linear(x, self.w, self.b)
                    else:
                        x = F.linear(x, self.w)
                    x = self.relu_or_id(x)
                    return x

            data = (torch.rand((1, 30), dtype=torch.float),)
            quant_type_to_qlinear_fun = {
                QuantType.DYNAMIC: ns.call_function(torch.ops.quantized.linear_dynamic),
                QuantType.STATIC: ns.call_function(torch.ops.quantized.linear),
                QuantType.QAT: ns.call_function(torch.ops.quantized.linear),
            }
            quant_type_to_qlinear_relu_fun = {
                # we don't have linear_relu_dynamic
                QuantType.DYNAMIC: ns.call_function(torch.ops.quantized.linear_relu_dynamic),
                QuantType.STATIC: ns.call_function(torch.ops.quantized.linear_relu),
                QuantType.QAT: ns.call_function(torch.ops.quantized.linear_relu),
            }

            options = itertools.product(
                self.all_quant_types,
                (True, False),  # use_bias
                (True, False),  # has_relu
                (True, False),  # functional relu
            )
            for quant_type, use_bias, has_relu, f_relu in options:
                # when has_relu is False, we are using an nn.Identity and
                # we will insert observer/fake_quant for the output of nn.Identity since
                # it is a copy node, that's why we have extra observer/fake_quant
                # when has_relu is False
                quant_type_to_prepare_expected_node_occurrence = {
                    QuantType.DYNAMIC: {
                        ns.call_module(torch.ao.quantization.PlaceholderObserver): 1,
                        ns.call_module(torch.ao.quantization.MinMaxObserver): 1,
                    },
                    # There should be 3 observers: after input, weight and activation.
                    # one more observer for torch.nn.Identity when there is no relu
                    QuantType.STATIC: {
                        ns.call_module(torch.ao.quantization.HistogramObserver): 2 if has_relu else 3,
                        ns.call_module(torch.ao.quantization.PerChannelMinMaxObserver): 1,
                    },
                    # There should be 3 observers: after input, weight and activation.
                    QuantType.QAT: {
                        ns.call_module(torch.ao.quantization.FusedMovingAvgObsFakeQuantize): 3 if has_relu else 4,
                    },
                }
                model = FuncLinear(use_bias, has_relu, f_relu)
                if has_relu:
                    qlinear_fun = quant_type_to_qlinear_relu_fun[quant_type]
                else:
                    qlinear_fun = quant_type_to_qlinear_fun[quant_type]

                if quant_type != QuantType.DYNAMIC:
                    num_dequantize = 1
                else:
                    # we will have an extra quantize_per_tensor_dynamic + dequantize for
                    # nn.Identity right now, but it will be fixed after we use
                    # backend_config to configure the default pt backend
                    num_dequantize = int(not has_relu)

                convert_node_occurrence = {
                    ns.call_function(torch.quantize_per_tensor): 1 if quant_type != QuantType.DYNAMIC else 0,
                    qlinear_fun: 1,
                    ns.call_method("dequantize"): num_dequantize if quant_type != QuantType.DYNAMIC else 0,
                }
                prepare_expected_node_occurrence = \
                    quant_type_to_prepare_expected_node_occurrence[quant_type]
                result_dict = self.checkGraphModeFxOp(
                    model, data, quant_type, qlinear_fun,
                    prepare_expected_node_occurrence=prepare_expected_node_occurrence,
                    expected_node_occurrence=convert_node_occurrence)
                if quant_type != QuantType.DYNAMIC:
                    self.assertEqual(result_dict["quantized_output"], result_dict["quantized_reference_output"])
                    # Ensure packed weights in lowered models are folded
                    self.assertIn("_packed_weight_0", result_dict["quantized"].state_dict().keys())

    @skipIfNoFBGEMM
    def test_linear_dynamic_fp16(self):
        with override_quantized_engine('fbgemm'):
            class FuncLinear(torch.nn.Module):
                def __init__(self, use_bias, has_relu, f_relu):
                    super(FuncLinear, self).__init__()
                    self.w = torch.randn(4, 30)
                    self.b = torch.randn(4)
                    self.use_bias = use_bias
                    if has_relu:
                        if f_relu:
                            self.relu = F.relu
                        else:
                            self.relu = torch.nn.ReLU()
                    else:
                        self.relu = torch.nn.Identity()

                def forward(self, x):
                    if self.use_bias:
                        x = F.linear(x, self.w, self.b)
                    else:
                        x = F.linear(x, self.w)
                    x = self.relu(x)
                    return x

            data = (torch.rand((1, 30), dtype=torch.float),)
            options = itertools.product(
                (True, False),  # use_bias
                (True, False),  # has_relu
                (True, False),  # functional relu
                (True, False),  # is_reference
            )
            for use_bias, has_relu, f_relu, is_reference in options:
                model = FuncLinear(use_bias, has_relu, f_relu)
                if is_reference:
                    qlinear_fun = ns.call_function(torch.nn.functional.linear)
                else:
                    if has_relu:
                        qlinear_fun = ns.call_function(torch.ops.quantized.linear_relu_dynamic_fp16)
                    else:
                        qlinear_fun = ns.call_function(torch.ops.quantized.linear_dynamic_fp16)
                prepare_node_occurrence = {
                    # activation and weight
                    ns.call_module(torch.ao.quantization.PlaceholderObserver): 2
                }
                convert_node_occurrence = {
                    qlinear_fun: 1,
                    # weight
                    ns.call_method("to"): 1 if is_reference else 0
                }
                self.checkGraphModeFxOp(
                    model, data, QuantType.DYNAMIC, qlinear_fun,
                    is_reference=is_reference,
                    custom_qconfig_dict={"": float16_dynamic_qconfig},
                    prepare_expected_node_occurrence=prepare_node_occurrence,
                    expected_node_occurrence=convert_node_occurrence)

    def test_linear_static_fp16(self):
        class FuncLinear(torch.nn.Module):
            def __init__(self, use_bias, has_relu, f_relu):
                super(FuncLinear, self).__init__()
                self.w = torch.randn(4, 30)
                self.b = torch.randn(4)
                self.use_bias = use_bias
                if has_relu:
                    if f_relu:
                        self.relu = F.relu
                    else:
                        self.relu = torch.nn.ReLU()
                else:
                    self.relu = torch.nn.Identity()

            def forward(self, x):
                if self.use_bias:
                    x = F.linear(x, self.w, self.b)
                else:
                    x = F.linear(x, self.w)
                x = self.relu(x)
                return x

        data = (torch.rand((1, 30), dtype=torch.float),)
        options = itertools.product(
            (True, False),  # use_bias
            (True, False),  # has_relu
            (True, False),  # functional relu
            (True, False),  # is_reference
        )
        backend_config = get_test_only_legacy_native_backend_config()
        for use_bias, has_relu, f_relu, is_reference in options:
            model = FuncLinear(use_bias, has_relu, f_relu)
            linear_fun = ns.call_function(torch.nn.functional.linear)
            # when has_relu is False, we are using an nn.Identity and
            # we will insert observer/fake_quant for the output of nn.Identity since
            # it is a copy node, that's why we have extra observer/fake_quant
            # when has_relu is False
            prepare_node_occurrence = {
                # activation, weight, bias and output
                ns.call_module(torch.ao.quantization.PlaceholderObserver): 3 + int(use_bias) + int(not has_relu),
            }
            # We have extra to and dequantize when is_reference is True
            # and has_relu is False since when has_relu is False, we
            # have an nn.Identity in the model, which is a CopyNode
            # and we would add extra quant - dequant for CopyNode in
            # reference patterns
            convert_node_occurrence = {
                # we don't support static fp16 ops, so the linear function
                # is unfused
                linear_fun: 1,
                # activation, weight, bias and output
                ns.call_method("to"): 3 + int(use_bias) + int(not has_relu and is_reference),
                ns.call_method("dequantize"): 3 + int(use_bias) + int(not has_relu and is_reference)
            }
            self.checkGraphModeFxOp(
                model, data, QuantType.DYNAMIC, linear_fun,
                is_reference=is_reference,
                custom_qconfig_dict={"": float16_static_qconfig},
                prepare_expected_node_occurrence=prepare_node_occurrence,
                expected_node_occurrence=convert_node_occurrence,
                backend_config=backend_config)

    @skipIfNoFBGEMM
    def test_conv_module(self):
        conv_module = {1 : torch.nn.Conv1d, 2 : torch.nn.Conv2d, 3 : torch.nn.Conv3d}

        class ConvWrapper(torch.nn.Module):
            def __init__(self, dim):
                super(ConvWrapper, self).__init__()
                self.conv = conv_module[dim](3, 3, 3).float()

            def forward(self, x):
                return self.conv(x)

        options = itertools.product([1, 2, 3], self.static_quant_types)
        quantized_nodes = {
            # dim
            1: ns.call_module(nnq.Conv1d),
            2: ns.call_module(nnq.Conv2d),
            3: ns.call_module(nnq.Conv3d),
        }
        for dim, quant_type in options:
            self.checkGraphModeFxOp(
                ConvWrapper(dim), self.img_data_dict[dim], quant_type,
                quantized_nodes[dim])

    @skipIfNoFBGEMM
    def test_functional_conv(self):
        with override_quantized_engine('fbgemm'):
            """ Test for function conv and functional conv + relu
            """
            convs = {
                1: torch.nn.functional.conv1d,
                2: torch.nn.functional.conv2d,
                3: torch.nn.functional.conv3d,
            }

            class FuncConv(torch.nn.Module):
                def __init__(self, dim, use_bias, has_relu, f_relu):
                    super().__init__()
                    self.dim = dim
                    self.w = torch.randn(tuple([3] * (dim + 2)))
                    self.b = torch.randn(3) if use_bias else None
                    self.stride = tuple([1] * dim)
                    self.padding = tuple([0] * dim)
                    self.dilation = tuple([1] * dim)
                    self.groups = 1
                    self.use_bias = use_bias
                    if has_relu:
                        if f_relu:
                            self.relu = F.relu
                        else:
                            self.relu = torch.nn.ReLU()
                    else:
                        self.relu = torch.nn.Identity()

                def forward(self, x):
                    x = convs[self.dim](x, self.w, self.b, self.stride, self.padding, self.dilation, self.groups)
                    x = self.relu(x)
                    return x

            quant_type_to_qconv_fun = {
                QuantType.STATIC: {
                    1: ns.call_function(torch.ops.quantized.conv1d),
                    2: ns.call_function(torch.ops.quantized.conv2d),
                    3: ns.call_function(torch.ops.quantized.conv3d)
                },
                QuantType.QAT: {
                    1: ns.call_function(torch.ops.quantized.conv1d),
                    2: ns.call_function(torch.ops.quantized.conv2d),
                    3: ns.call_function(torch.ops.quantized.conv3d)
                },
            }
            quant_type_to_qconv_relu_fun = {
                QuantType.STATIC: {
                    1: ns.call_function(torch.ops.quantized.conv1d_relu),
                    2: ns.call_function(torch.ops.quantized.conv2d_relu),
                    3: ns.call_function(torch.ops.quantized.conv3d_relu)
                },
                QuantType.QAT: {
                    1: ns.call_function(torch.ops.quantized.conv1d_relu),
                    2: ns.call_function(torch.ops.quantized.conv2d_relu),
                    3: ns.call_function(torch.ops.quantized.conv3d_relu)
                },
            }

            options = itertools.product(
                [1, 2, 3],  # dims
                self.static_quant_types,
                (True, False),  # use_bias
                (True, False),  # has_relu
                (True, False),  # functional relu
            )
            for dim, quant_type, use_bias, has_relu, f_relu in options:
                # when has_relu is False, we are using an nn.Identity and
                # we will insert observer/fake_quant for the output of nn.Identity since
                # it is a copy node, that's why we have extra observer/fake_quant
                # when has_relu is False
                quant_type_to_prepare_expected_node_occurrence = {
                    QuantType.DYNAMIC: {},
                    # There should be 3 observers: after input, weight and activation.
                    QuantType.STATIC: {
                        ns.call_module(torch.ao.quantization.HistogramObserver): 2 if has_relu else 3,
                        ns.call_module(torch.ao.quantization.PerChannelMinMaxObserver): 1,
                    },
                    # There should be 3 observers: after input, weight and activation.
                    QuantType.QAT: {
                        ns.call_module(torch.ao.quantization.FusedMovingAvgObsFakeQuantize): 3 if has_relu else 4,
                    },
                }
                data_dims = [2, 3] + [4] * dim
                data = (torch.randn(tuple(data_dims), dtype=torch.float),)
                model = FuncConv(dim, use_bias, has_relu, f_relu)
                if has_relu:
                    qconv_fun = quant_type_to_qconv_relu_fun[quant_type][dim]
                else:
                    qconv_fun = quant_type_to_qconv_fun[quant_type][dim]

                convert_node_occurrence = {
                    ns.call_function(torch.quantize_per_tensor): 1,
                    qconv_fun: 1,
                    ns.call_method("dequantize"): 1
                }
                prepare_expected_node_occurrence = \
                    quant_type_to_prepare_expected_node_occurrence[quant_type]
                result_dict = self.checkGraphModeFxOp(
                    model, data, quant_type, qconv_fun,
                    prepare_expected_node_occurrence=prepare_expected_node_occurrence,
                    expected_node_occurrence=convert_node_occurrence)
                if quant_type != QuantType.DYNAMIC:
                    self.assertEqual(result_dict["quantized_output"], result_dict["quantized_reference_output"])
                    # Ensure packed weights in lowered models are folded
                    self.assertIn("_packed_weight_0", result_dict["quantized"].state_dict().keys())

    @skipIfNoFBGEMM
    def test_quantized_conv_relu(self):
        """tests for conv1d_relu/conv2d_relu/conv3d_relu"""
        conv_module = {1 : torch.nn.Conv1d, 2 : torch.nn.Conv2d, 3 : torch.nn.Conv3d}

        class ConvNdRelu(torch.nn.Module):
            def __init__(self, dim, inplace):
                super(ConvNdRelu, self).__init__()
                self.conv = conv_module[dim](3, 3, 3).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x):
                return self.relu(self.conv(x))

        class ConvNdFunctionalRelu(torch.nn.Module):
            def __init__(self, dim):
                super(ConvNdFunctionalRelu, self).__init__()
                self.conv = conv_module[dim](3, 3, 3).float()

            def forward(self, x):
                return F.relu(self.conv(x))

        class ConvNdInplaceFunctionalRelu(torch.nn.Module):
            def __init__(self, dim):
                super(ConvNdInplaceFunctionalRelu, self).__init__()
                self.conv = conv_module[dim](3, 3, 3).float()

            def forward(self, x):
                return F.relu(self.conv(x), True)

        options = itertools.product([1, 2, 3], self.static_quant_types)
        quantized_nodes = {
            # dim
            1: ns.call_module(nniq.ConvReLU1d),
            2: ns.call_module(nniq.ConvReLU2d),
            3: ns.call_module(nniq.ConvReLU3d),
        }
        for dim, quant_type in options:
            for m in [ConvNdRelu(dim, True),
                      ConvNdRelu(dim, False),
                      ConvNdFunctionalRelu(dim),
                      ConvNdInplaceFunctionalRelu(dim)]:
                self.checkGraphModeFxOp(
                    m, self.img_data_dict[dim], quant_type,
                    quantized_nodes[dim])


    def _test_binary_op_int8_impl(self, binary_op, ibinary_op, quantized_op):
        data = (torch.randn(1, 1, 1, 1, dtype=torch.float),
                torch.randn(1, 1, 1, 1, dtype=torch.float))
        options = itertools.product([True, False], [True, False], [True, False])
        quant_type = QuantType.STATIC
        # testing for default int8 static quant
        for is_inplace, is_scalar, is_reference in options:
            if is_reference:
                node_list = [
                    ns.call_method("dequantize"),
                    ns.call_function(binary_op),
                    ns.call_function(torch.quantize_per_tensor)
                ]
                quantized_node = None
            else:
                node_list = None
                quantized_node = ns.call_function(quantized_op)

            self.checkGraphModeFxOp(
                BinaryOp(binary_op, ibinary_op, is_inplace, is_scalar), data, quant_type,
                quantized_node, expected_node_list=node_list, is_reference=is_reference)
            # This tests the binary op should be quantized even when it is not feed with a
            # quantized input
            self.checkGraphModeFxOp(
                BinaryOpNonQuantizedInput(binary_op, ibinary_op, is_inplace, is_scalar),
                data, quant_type, quantized_node,
                expected_node_list=node_list, is_reference=is_reference)


    def _test_binary_op_float16_impl(self, binary_op, ibinary_op):
        data = (torch.randn(1, 1, 1, 1, dtype=torch.float),
                torch.randn(1, 1, 1, 1, dtype=torch.float))
        quant_type = QuantType.STATIC
        # testing for fp16 static quant
        # we are producing fp16 patterns
        options = itertools.product([True, False], [True, False])
        custom_qconfig_dict = {
            "object_type": [(binary_op, float16_static_qconfig)]
        }
        backend_config = get_test_only_legacy_native_backend_config()
        for is_inplace, is_scalar in options:
            node_occurrence = {
                # output_conv1, output_add1, output_add2 for scalar
                # output_conv1, output_conv2, output_add1, output_add2 for non-scalar
                ns.call_method("to"): 3 if is_scalar else 4
            }
            self.checkGraphModeFxOp(
                BinaryOp(binary_op, ibinary_op, is_inplace, is_scalar), data, quant_type,
                expected_node_occurrence=node_occurrence,
                custom_qconfig_dict=custom_qconfig_dict,
                backend_config=backend_config)

            node_occurrence = {
                # input_add, output_add for scalar
                # input_add1, input_add2, output_add for non-scalar
                ns.call_method("to"): 2 if is_scalar else 3
            }
            self.checkGraphModeFxOp(
                BinaryOpNonQuantizedInput(binary_op, ibinary_op, is_inplace, is_scalar), data, quant_type,
                expected_node_occurrence=node_occurrence,
                custom_qconfig_dict=custom_qconfig_dict,
                backend_config=backend_config)

    def _test_binary_op_relu_int8_impl(self, binary_op, ibinary_op, quantized_op):
        data = (torch.rand((1, 1, 1, 1), dtype=torch.float),
                torch.rand((1, 1, 1, 1), dtype=torch.float))
        quant_type = QuantType.STATIC
        quantized_node = ns.call_function(quantized_op)
        options = itertools.product(
            [True, False], [nn.ReLU, F.relu, torch.relu], [True, False])
        for is_inplace_op, relu_callable, is_scalar in options:
            model = BinaryOpRelu(
                binary_op, ibinary_op, is_inplace_op, relu_callable, is_scalar)
            self.checkGraphModeFxOp(
                model, data, quant_type, quantized_node)

    def _test_binary_op_relu_float16_impl(self, binary_op, ibinary_op):
        data = (torch.rand((1, 1, 1, 1), dtype=torch.float),
                torch.rand((1, 1, 1, 1), dtype=torch.float))
        quant_type = QuantType.STATIC
        options = itertools.product(
            [True, False], [nn.ReLU, F.relu, torch.relu], [True, False])
        custom_qconfig_dict = {
            "": float16_static_qconfig,
            "object_type": [(torch.nn.Conv2d, None)]
        }
        backend_config = get_test_only_legacy_native_backend_config()
        for is_inplace_op, is_functional_relu, is_scalar in options:
            node_occurrence = {
                ns.call_method("to"): 3 if is_scalar else 4
            }
            model = BinaryOpRelu(
                binary_op, ibinary_op, is_inplace_op, is_functional_relu, is_scalar)
            self.checkGraphModeFxOp(
                model, data, quant_type, custom_qconfig_dict=custom_qconfig_dict,
                expected_node_occurrence=node_occurrence,
                backend_config=backend_config)


    @skipIfNoFBGEMM
    def test_add(self):
        self._test_binary_op_int8_impl(
            operator.add, operator.iadd, torch.ops.quantized.add)
        self._test_binary_op_float16_impl(
            operator.add, operator.iadd)

    @unittest.skip("This is no longer needed right now, can enable later with new api")
    def test_sub(self):
        self._test_binary_op_float16_impl(operator.sub, operator.isub)
        self._test_binary_op_float16_impl(torch.sub, None)

    @unittest.skip("This is no longer needed right now, can enable later with new api")
    def test_div(self):
        self._test_binary_op_float16_impl(operator.truediv, operator.itruediv)
        self._test_binary_op_float16_impl(torch.div, None)

    @skipIfNoFBGEMM
    def test_mul(self):
        self._test_binary_op_int8_impl(
            operator.mul, operator.imul, torch.ops.quantized.mul)
        self._test_binary_op_float16_impl(operator.mul, operator.imul)

    @unittest.skip("This is no longer needed right now, can enable later with new api")
    def test_sum(self):
        class Sum(torch.nn.Module):
            def forward(self, x):
                x = torch.sum(x, [1], keepdim=True)
                x = torch.sum(x, [1])
                return x

        data = torch.randn(1, 2, 3, 4, dtype=torch.float)
        quant_type = QuantType.STATIC
        # testing for fp16 static quant
        # we are producing fp16 patterns
        custom_qconfig_dict = {
            "object_type": [(torch.sum, float16_static_qconfig)]
        }
        node_occurrence = {
            # input_sum1, output_sum1, output_sum2
            ns.call_method("to"): 3
        }
        self.checkGraphModeFxOp(
            Sum(), data, quant_type,
            expected_node_occurrence=node_occurrence,
            custom_qconfig_dict=custom_qconfig_dict)

    @unittest.skip("This is no longer needed right now, can enable later with new api")
    def test_bmm(self):
        class BMMMethod(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x.bmm(y)

        data = (torch.randn(1, 1, 1, dtype=torch.float),
                torch.randn(1, 1, 1, dtype=torch.float))
        quant_type = QuantType.STATIC
        # testing for fp16 static quant
        # we are producing fp16 patterns
        custom_qconfig_dict = {
            "object_type": [(torch.bmm, float16_static_qconfig),
                            ("bmm", float16_static_qconfig)]
        }
        node_occurrence = {
            # input_bmm1, input_bmm2, output_bmm
            ns.call_method("to"): 3
        }
        self.checkGraphModeFxOp(
            BinaryOpNonQuantizedInput(torch.bmm, None, False, False), data, quant_type,
            expected_node_occurrence=node_occurrence,
            custom_qconfig_dict=custom_qconfig_dict)

        # TODO: support call_method("bmm")
        # we can transform call_method("bmm") to call_function(torch.bmm)
        # self.checkGraphModeFxOp(
        #     BMMMethod(), data, quant_type,
        #     expected_node_occurrence=node_occurrence,
        #     custom_qconfig_dict=custom_qconfig_dict,
        #     print_debug_info=True)

    @skipIfNoFBGEMM
    def test_add_relu(self):
        self._test_binary_op_relu_int8_impl(
            operator.add, operator.iadd, torch.ops.quantized.add_relu)
        self._test_binary_op_relu_float16_impl(
            operator.add, operator.iadd)

    @skipIfNoFBGEMM
    def test_add_relu_multiple_uses_of_relu(self):
        class Sub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU(inplace=True)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = Sub()

            def forward(self, x, y):
                x = x + y
                x = self.sub.relu(x)
                x = x + y
                x = self.sub.relu(x)
                return x

        m = M().eval()
        example_inputs = (torch.randn(3), torch.randn(3))
        m = prepare_fx(m, {"": default_qconfig}, example_inputs=example_inputs)
        m = convert_fx(m)
        node_occurrence = {
            ns.call_function(torch.quantize_per_tensor): 2,
            ns.call_function(torch.ops.quantized.add_relu): 2,
            ns.call_method("dequantize"): 1,
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)
        # check the model is scriptable
        m = torch.jit.script(m)
        # check the model is runnable
        m(*example_inputs)

    @skipIfNoFBGEMM
    def test_mul_relu(self):
        self._test_binary_op_relu_int8_impl(
            operator.mul, operator.imul, torch.ops.quantized.mul_relu)
        self._test_binary_op_relu_float16_impl(
            operator.mul, operator.imul)

    # TODO(future PR): make more generic
    def _test_quantized_add_mul_qat(self, model, example_inputs, expected_node_occurrence):
        qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        mp = prepare_qat_fx(model, qconfig_dict, example_inputs=example_inputs)
        self.checkGraphModuleNodes(
            mp, expected_node_occurrence=expected_node_occurrence)

    @skipIfNoFBGEMM
    def test_quantized_add_qat(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 1, 1)
                self.conv2 = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = torch.add(x, 1.0)
                x = self.conv1(x)
                x = torch.add(x, 1.0)
                x = torch.relu(x)
                x = self.conv2(x)
                return x

        m = M()
        example_inputs = (torch.randn(1, 1, 1, 1),)
        expected_node_occurrence = {
            ns.call_module(torch.ao.quantization.FusedMovingAvgObsFakeQuantize): 5,
        }
        self._test_quantized_add_mul_qat(m, example_inputs, expected_node_occurrence)

    @skipIfNoFBGEMM
    def test_quantized_mul_qat(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 1, 1)
                self.conv2 = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = torch.mul(x, 1.0)
                x = self.conv1(x)
                x = torch.mul(x, 1.0)
                x = torch.relu(x)
                x = self.conv2(x)
                return x

        m = M()
        example_inputs = (torch.randn(1, 1, 1, 1),)
        expected_node_occurrence = {
            ns.call_module(torch.ao.quantization.FusedMovingAvgObsFakeQuantize): 5,
        }
        self._test_quantized_add_mul_qat(m, example_inputs, expected_node_occurrence)

    def test_int8_input_no_unnecessary_fq(self):
        """
        If the inputs to the graph are quantized and the only node
        does not need an activation observer, verifies that the
        activation observer is not inserted.
        """
        class M(nn.Module):
            def __init__(self, scalar):
                super().__init__()
                self.scalar = scalar
                self.add_func = torch.ao.nn.quantized.FloatFunctional()

            def forward(self, x):
                return self.add_func.add_scalar(x, self.scalar)

        m = M(0.5)
        mp = torch.ao.quantization.quantize_fx.prepare_qat_fx(
            m, {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')},
            example_inputs=(torch.randn(1),),
            prepare_custom_config={"input_quantized_idxs": [0]})
        expected_node_occurrence = {
            ns.call_module(torch.ao.quantization.FusedMovingAvgObsFakeQuantize): 1,
        }
        self.checkGraphModuleNodes(
            mp, expected_node_occurrence=expected_node_occurrence)

    @skipIfNoFBGEMM
    def test_cat(self):
        """ quantization of the output of cat will depend on the
        input of cat. we only quantize the output of cat when its inputs are quantized.
        """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                return torch.cat([x, y], 1)

        example_inputs = (torch.randn(1, 2, 5, 5, dtype=torch.float),
                          torch.randn(1, 2, 5, 5, dtype=torch.float))
        quantized_node = ns.call_function(torch.cat)
        options = itertools.product(self.static_quant_types, [True, False])
        for quant_type, is_reference in options:
            if is_reference:
                converted_node_list = [
                    ns.call_method("dequantize"),
                    ns.call_function(torch.cat),
                    ns.call_function(torch.quantize_per_tensor)
                ]
                converted_node_occurrence = {
                    # inputs and outputs of the two conv, and output of cat
                    ns.call_method("dequantize"): 5,
                    ns.call_function(torch.cat): 1,
                    # inputs and outputs of the two conv, and output of cat
                    ns.call_function(torch.quantize_per_tensor): 5,
                }
            else:
                converted_node_list = None
                converted_node_occurrence = {
                    # output of cat
                    ns.call_method("dequantize"): 1,
                    ns.call_function(torch.cat): 1,
                    # for two inputs
                    ns.call_function(torch.quantize_per_tensor): 2,
                }

            self.checkGraphModeFxOp(
                M(),
                example_inputs,
                quant_type,
                quantized_node,
                expected_node_list=converted_node_list,
                expected_node_occurrence=converted_node_occurrence,
                is_reference=is_reference)

        # check cat is using the same observer for input and output
        m = M().eval()
        m = prepare_fx(m, {"": default_qconfig}, example_inputs=example_inputs)
        # two inputs and one output of torch.cat are using same observer, so we have
        # 2 observers that's replicated
        all_observers = len(dict(m.named_modules(remove_duplicate=False)))
        distinct_observers = len(dict(m.named_modules()))
        self.assertEqual(all_observers, distinct_observers + 2)
        # make sure the converted model runs
        m = convert_fx(m)
        m(*example_inputs)

    @skipIfNoFBGEMM
    def test_qbatch_norm(self):
        bn_module = {
            # TODO: quantized batchnorm 1d module is missing
            # 1 : torch.nn.BatchNorm1d,
            2 : torch.nn.BatchNorm2d,
            3 : torch.nn.BatchNorm3d,
        }

        class M(torch.nn.Module):
            def __init__(self, dim):
                super(M, self).__init__()
                self.bn = bn_module[dim](3).to(torch.float)

            def forward(self, x):
                return self.bn(x)

        options = itertools.product(self.static_quant_types, [2, 3], [True, False])
        quantized_nodes = {
            False: {
                # 1: ns.call_module(nnq.BatchNorm1d),
                2: ns.call_module(nnq.BatchNorm2d),
                3: ns.call_module(nnq.BatchNorm3d),
            },
            True: {
                # 1: ns.call_module(nn.BatchNorm1d),
                2: ns.call_module(nn.BatchNorm2d),
                3: ns.call_module(nn.BatchNorm3d),
            }
        }
        for quant_type, dim, is_reference in options:
            self.checkGraphModeFxOp(
                M(dim), self.img_data_dict[dim], quant_type, quantized_nodes[is_reference][dim], is_reference=is_reference)

    @skipIfNoFBGEMM
    def test_qbatch_norm_relu(self):
        bn_module = {2 : torch.nn.BatchNorm2d, 3 : torch.nn.BatchNorm3d}

        class BNRelu(torch.nn.Module):
            def __init__(self, dim, inplace):
                super(BNRelu, self).__init__()
                self.bn = bn_module[dim](3).to(torch.float)
                self.relu = torch.nn.ReLU(inplace=inplace)

            def forward(self, x):
                return self.relu(self.bn(x))

        class BNFuncRelu(torch.nn.Module):
            def __init__(self, dim):
                super(BNFuncRelu, self).__init__()
                self.bn = bn_module[dim](3).to(torch.float)

            def forward(self, x):
                return F.relu(self.bn(x), False)

        class BNFuncInplaceRelu(torch.nn.Module):
            def __init__(self, dim):
                super(BNFuncInplaceRelu, self).__init__()
                self.bn = bn_module[dim](3).to(torch.float)

            def forward(self, x):
                return F.relu(self.bn(x), True)

        options = itertools.product(self.static_quant_types, [2, 3], [True, False])
        quantized_nodes = {
            True: {
                2: ns.call_module(nni.BNReLU2d),
                3: ns.call_module(nni.BNReLU3d),
            },
            False: {
                2: ns.call_module(nniq.BNReLU2d),
                3: ns.call_module(nniq.BNReLU3d),
            }
        }
        for quant_type, dim, is_reference in options:
            for instance in [BNRelu(dim, True), BNRelu(dim, False),
                             BNFuncRelu(dim), BNFuncInplaceRelu(dim)]:
                self.checkGraphModeFxOp(
                    instance, self.img_data_dict[dim], quant_type,
                    quantized_nodes[is_reference][dim], is_reference=is_reference)

    def _test_activation_impl(
            self, float_module, float_op, quantized_module, quantized_op):
        ''' Test for activation op(with inplace options), float_op can be
        torch op or functional op
        '''
        class M(torch.nn.Module):
            def __init__(self, is_module, inplace):
                super(M, self).__init__()
                self.is_module = is_module
                self.inplace = inplace
                if self.is_module:
                    self.op = float_module(self.inplace)
                else:
                    self.op = float_op

            def forward(self, input):
                if self.is_module:
                    return self.op(input)
                else:
                    return self.op(input, self.inplace)

        options = itertools.product([True, False], [True, False], self.static_quant_types, [True, False])
        quantized_nodes = {
            # is_module
            True: {
                # is_reference
                True: ns.call_module(float_module),
                False: ns.call_module(quantized_module),
            },
            False: {
                True: ns.call_function(float_op),
                False: ns.call_function(quantized_op),
            }
        }

        for is_module, is_inplace, quant_type, is_reference in options:
            self.checkGraphModeFxOp(
                M(is_module, is_inplace), self.img_data_2d,
                quant_type, quantized_nodes[is_module][is_reference], is_reference=is_reference)

    def test_hardswish(self):
        self._test_activation_impl(nn.Hardswish, F.hardswish, nnq.Hardswish, torch.ops.quantized.hardswish)

    def test_elu(self):
        self._test_activation_impl(nn.ELU, F.elu, nnq.ELU, torch.ops.quantized.elu)

    def test_leaky_relu(self):
        self._test_activation_impl(nn.LeakyReLU, F.leaky_relu, nnq.LeakyReLU, torch.ops.quantized.leaky_relu)

    def test_prelu(self):
        class M(torch.nn.Module):
            def __init__(self, num_param: int):
                super(M, self).__init__()
                self.op = torch.nn.PReLU(num_parameters=num_param)

            def forward(self, input):
                return self.op(input)

        X = [[torch.randn(4, 4, 4, 4, dtype=torch.float)]]
        options = itertools.product([1, 4], self.static_quant_types, [True, False])
        quantized_nodes = {
            # is_reference
            True: ns.call_module(torch.nn.PReLU),
            False: ns.call_module(torch.nn.quantized.PReLU),
        }

        for num_parameter, quant_type, is_reference in options:
            self.checkGraphModeFxOp(
                M(num_parameter), X, quant_type, quantized_nodes[is_reference],
                is_reference=is_reference)

    def _test_norm_impl(
            self, float_module, float_op, op_args, data, quantized_module, quantized_op,
            skip_op_arg_for_functional=False):
        ''' Test for normalization op, float_op can be torch op or functional op,
        op_args is a list of positional argument for the module/op
        '''
        class M(torch.nn.Module):
            def __init__(self, is_module):
                super(M, self).__init__()
                self.is_module = is_module
                if self.is_module:
                    self.op = float_module(*op_args)
                else:
                    self.op = float_op

            def forward(self, input):
                if self.is_module:
                    return self.op(input)
                else:
                    args = [input]
                    if not skip_op_arg_for_functional:
                        args += op_args
                    return self.op(*args)

        options = itertools.product([True, False], self.static_quant_types)
        quantized_nodes = {
            # is_module
            True: ns.call_module(quantized_module),
            False: ns.call_function(quantized_op),
        }

        for is_module, quant_type in options:
            self.checkGraphModeFxOp(
                M(is_module), data, quant_type, quantized_nodes[is_module])

    def _test_norm_float16_impl(
            self, float_module, float_op, op_args, data,
            skip_op_arg_for_functional=False):
        ''' Test for normalization op, float_op can be torch op or functional op,
        op_args is a list of positional argument for the module/op
        '''
        class M(torch.nn.Module):
            def __init__(self, is_module):
                super(M, self).__init__()
                self.is_module = is_module
                if self.is_module:
                    self.op = float_module(*op_args)
                else:
                    self.op = float_op

            def forward(self, input):
                if self.is_module:
                    return self.op(input)
                else:
                    args = [input]
                    if not skip_op_arg_for_functional:
                        args += op_args
                    return self.op(*args)

        options = itertools.product([True, False], self.static_quant_types)
        qconfig_dict = {
            "object_type": [
                (float_module, float16_static_qconfig),
                (float_op, float16_static_qconfig)
            ]
        }
        node_occurrence = {
            ns.call_method("to"): 2
        }
        for is_module, quant_type in options:
            self.checkGraphModeFxOp(
                M(is_module), data, quant_type, custom_qconfig_dict=qconfig_dict, expected_node_occurrence=node_occurrence)

    def test_layer_norm(self):
        data = (torch.rand((1, 2, 5, 5), dtype=torch.float),)
        self._test_norm_impl(
            nn.LayerNorm, F.layer_norm, [[2, 5, 5]], data, nnq.LayerNorm, torch.ops.quantized.layer_norm)

    def test_instance_norm(self):
        data_1d = (torch.rand((1, 4, 5), dtype=torch.float),)
        data_2d = (torch.rand((1, 4, 5, 1), dtype=torch.float),)
        data_3d = (torch.rand((1, 4, 5, 1, 1), dtype=torch.float),)
        data_dict = {1 : data_1d, 2 : data_2d, 3 : data_3d}
        instance_norm_modules = {1 : nn.InstanceNorm1d,
                                 2 : nn.InstanceNorm2d,
                                 3 : nn.InstanceNorm3d}
        quantized_instance_norm_modules = {
            1 : nnq.InstanceNorm1d,
            2 : nnq.InstanceNorm2d,
            3 : nnq.InstanceNorm3d
        }
        for dim in [1, 2, 3]:
            data = data_dict[dim]
            module = instance_norm_modules[dim]
            quantized_module = quantized_instance_norm_modules[dim]
            self._test_norm_impl(
                module, F.instance_norm, [4], data,
                quantized_module, torch.ops.quantized.instance_norm,
                skip_op_arg_for_functional=True)

    def test_norm_weight_bias(self):
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.ones(5, 5)
                self.b = torch.zeros(5)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.w, self.b)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mods1 = Linear()
                self.scale = torch.randn(5, 5)
                self.bias = torch.randn(5, 5)

            def forward(self, x):
                x1 = self.mods1(x)
                y = F.layer_norm(x1, [5, 5], weight=self.scale, bias=self.bias)
                return y

        model = M()
        expected_occurrence = {
            ns.call_function(torch.quantize_per_tensor): 1,
            ns.call_function(torch.ops.quantized.linear): 1,
            ns.call_function(torch.ops.quantized.layer_norm): 1,
            ns.call_method("dequantize"): 1,
        }

        self.checkGraphModeFxOp(
            model,
            (torch.rand(5, 5),),
            QuantType.STATIC,
            expected_node_occurrence=expected_occurrence,
            custom_qconfig_dict=get_default_qconfig_mapping().to_dict()
        )

    def _test_default_node_quant_handler_ops(
            self, module, functional, qconfig, is_reference=True, node_list=None, additional_quant_pattern_dict=None
    ):
        class M(torch.nn.Module):
            def __init__(self, mod, func):
                super().__init__()
                self.module = mod()
                self.functional = func

            def forward(self, x):
                x = self.module(x)
                x = self.functional(x)
                return x

        if node_list is None:
            node_list = []
        if additional_quant_pattern_dict is None:
            additional_quant_pattern_dict = {}

        data = torch.randn((2, 2, 2, 2))
        quant_type = QuantType.STATIC
        prepare_custom_qconfig_dict = {"additional_quant_pattern": additional_quant_pattern_dict}
        qconfig_dict = {"": qconfig}

        m = M(module, functional).eval()
        m_prep = prepare_fx(m, qconfig_dict, prepare_custom_qconfig_dict)
        m_prep(data)
        convert_fn = convert_to_reference_fx if is_reference else convert_fx
        m_quant = convert_fn(m_prep, is_reference=is_reference)
        m_quant(data)

        self.checkGraphModuleNodes(m_quant, expected_node_list=node_list)

    @unittest.skip("TODO: reenable with backend_config api")
    def test_gelu_normal(self):
        module = torch.nn.GELU
        functional = torch.nn.functional.gelu
        qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        is_reference = False
        node_list = [
            ns.call_module(module),
            ns.call_function(functional),
        ]
        self._test_default_node_quant_handler_ops(
            module, functional, qconfig, is_reference, node_list)

    @unittest.skip("TODO: reenable with backend_config api")
    def test_softmax_normal(self):
        module = torch.nn.Softmax
        functional = torch.nn.functional.softmax
        qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        is_reference = False
        node_list = [
            ns.call_module(torch.ao.nn.quantized.Softmax),
            ns.call_function(functional),
        ]
        self._test_default_node_quant_handler_ops(
            module, functional, qconfig, is_reference, node_list)

    @unittest.skip("This is no longer needed right now, can enable later with new api")
    def test_gelu_reference(self):
        module = torch.nn.GELU
        functional = torch.nn.functional.gelu
        qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        is_reference = True
        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_method("dequantize"),
            ns.call_module(module),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_method('dequantize'),
            ns.call_function(functional),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_method('dequantize')
        ]
        # TODO: change these to use backend_config
        additional_patterns = {torch.nn.GELU: DefaultNodeQuantizeHandler,
                               torch.nn.functional.gelu: DefaultNodeQuantizeHandler}
        self._test_default_node_quant_handler_ops(
            module, functional, qconfig, is_reference, node_list, additional_patterns)

        self._test_default_node_quant_handler_ops(module, functional, self.custom_qconfig, is_reference, node_list,
                                                  additional_quant_pattern_dict=self.common_quant_patterns)

    @unittest.skip("This is no longer needed right now, can enable later with new api")
    def test_softmax_reference(self):
        module = torch.nn.Softmax
        functional = torch.nn.functional.softmax
        qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        is_reference = True
        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_method("dequantize"),
            ns.call_module(module),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_method('dequantize'),
            ns.call_function(functional),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_method('dequantize')
        ]
        additional_patterns = {torch.nn.Softmax: DefaultNodeQuantizeHandler,
                               torch.nn.functional.softmax: DefaultNodeQuantizeHandler}
        self._test_default_node_quant_handler_ops(
            module, functional, qconfig, is_reference, node_list, additional_patterns)

        self._test_default_node_quant_handler_ops(module, functional, self.custom_qconfig, is_reference, node_list,
                                                  additional_quant_pattern_dict=self.common_quant_patterns)

    @unittest.skip("This is no longer needed right now, can enable later with new api")
    def test_silu_reference(self):
        module = torch.nn.SiLU
        functional = torch.nn.functional.silu
        qconfig = float16_static_qconfig
        is_reference = True
        node_list = [
            ns.call_method("to"),
            ns.call_method("dequantize"),
            ns.call_module(module),
            ns.call_method("to"),
            ns.call_method('dequantize'),
            ns.call_function(functional),
            ns.call_method("to"),
            ns.call_method('dequantize')
        ]
        self._test_default_node_quant_handler_ops(
            module, functional, qconfig, is_reference, node_list)

        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_method("dequantize"),
            ns.call_module(module),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_method("dequantize"),
            ns.call_function(functional),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_method("dequantize")
        ]
        self._test_default_node_quant_handler_ops(module, functional, self.custom_qconfig, is_reference, node_list,
                                                  additional_quant_pattern_dict=self.common_quant_patterns)

    @unittest.skip("This is no longer needed right now, can enable later with new api")
    def test_mish_reference(self):
        module = torch.nn.Mish
        functional = torch.nn.functional.mish
        qconfig = float16_static_qconfig
        is_reference = True
        node_list = [
            ns.call_method("to"),
            ns.call_method("dequantize"),
            ns.call_module(module),
            ns.call_method("to"),
            ns.call_method('dequantize'),
            ns.call_function(functional),
            ns.call_method("to"),
            ns.call_method('dequantize')
        ]
        self._test_default_node_quant_handler_ops(
            module, functional, qconfig, is_reference, node_list)

        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_method("dequantize"),
            ns.call_module(module),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_method("dequantize"),
            ns.call_function(functional),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_method("dequantize")
        ]
        self._test_default_node_quant_handler_ops(module, functional, self.custom_qconfig, is_reference, node_list,
                                                  additional_quant_pattern_dict=self.common_quant_patterns)

    def test_bmm_int_reference(self):
        """ int8 is not supported for bmm so we won't produce reference
            pattern for it
        """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bmm = torch.bmm

            def forward(self, x, y):
                out = self.bmm(x, y)
                return out

        data_x = torch.randn((2, 2, 2,))
        data_y = torch.randn((2, 2, 2,))
        example_inputs = (data_x, data_y)
        qconfig_dict = {"": torch.ao.quantization.get_default_qconfig("fbgemm")}
        is_reference = True
        node_list = [
            ns.call_function(torch.bmm),
        ]

        m = M().eval()
        m_prep = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        m_prep(*example_inputs)
        convert_fn = convert_to_reference_fx if is_reference else convert_fx
        m_quant = convert_fn(m_prep)
        m_quant(*example_inputs)

        self.checkGraphModuleNodes(m_quant, expected_node_list=node_list)

    @skipIfNoFBGEMM
    def test_clamp(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()
                self.relu6 = torch.nn.ReLU6()
                self.relu6_ = torch.nn.ReLU6(True)
                self.hardtanh = torch.nn.Hardtanh()
                self.hardtanh_ = torch.nn.Hardtanh(inplace=True)

            def forward(self, x):
                x = self.conv(x)
                x = self.relu6(x)
                self.relu6_(x)
                x = F.relu6(x)
                x = torch.clamp(x, -3, 3)
                x = x.clamp(-2.5, 2.5)
                # x = x.clamp_(-2, 2)  # Enable when quantized `clamp_` is ready
                x = self.hardtanh(x)
                self.hardtanh_(x)
                x = F.hardtanh(x)
                return x

        data = (torch.rand((1, 2, 5, 5), dtype=torch.float),)
        # list of node that should occur in order
        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Conv2d),
            ns.call_method('dequantize')
        ]
        for quant_type in self.static_quant_types:
            self.checkGraphModeFxOp(
                M(), data, quant_type, expected_node_list=node_list)

    def test_fixed_qparams_ops_fp16(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()
                self.tanh = torch.nn.Tanh()

            def forward(self, x):
                x = self.sigmoid(x)
                x = torch.sigmoid(x)
                x = x.sigmoid()
                x = self.tanh(x)
                x = torch.tanh(x)
                x = x.tanh()
                return x

        data = (torch.randn((2, 2, 2, 2), dtype=torch.float),)
        quant_type = QuantType.STATIC
        # TODO: use get_default_qconfig_mapping once it handles fp16
        qconfig_mapping = QConfigMapping().set_global(float16_static_qconfig)
        backend_config = get_test_only_legacy_native_backend_config()
        node_occurrence = {
            ns.call_method("to"): 7
        }
        self.checkGraphModeFxOp(
            M(), data, quant_type, custom_qconfig_dict=qconfig_mapping,
            expected_node_occurrence=node_occurrence,
            backend_config=backend_config)

    def test_fixed_qparams_ops_qint8(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()
                self.tanh = torch.nn.Tanh()

            def forward(self, x):
                x = self.sigmoid(x)
                x = torch.sigmoid(x)
                x = x.sigmoid()
                x = self.tanh(x)
                x = torch.tanh(x)
                x = x.tanh()
                return x

        data = (torch.randn((2, 2, 2, 2), dtype=torch.float),)
        quant_type = QuantType.STATIC
        qconfig = torch.ao.quantization.QConfig(
            activation=HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8),
            weight=default_weight_observer)
        qconfig_mapping = get_default_qconfig_mapping().set_global(qconfig)
        node_occurrence = {
            ns.call_function(torch.quantize_per_tensor): 7,
            ns.call_method("dequantize"): 7
        }
        self.checkGraphModeFxOp(
            M(), data, quant_type, custom_qconfig_dict=qconfig_mapping,
            expected_node_occurrence=node_occurrence, is_reference=True)

    def test_fixed_qparams_ops_qconfig_error(self):
        """ Test that a proper error message is shown when user don't specify the correct
        qconfig for fixed qaprams ops
        """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()
                self.tanh = torch.nn.Tanh()

            def forward(self, x):
                x = self.sigmoid(x)
                x = torch.sigmoid(x)
                x = x.sigmoid()
                x = self.tanh(x)
                x = torch.tanh(x)
                x = x.tanh()
                return x

        data = (torch.randn((2, 2, 2, 2), dtype=torch.float),)
        qconfig_mapping = QConfigMapping().set_global(default_qconfig)
        m = M().eval()
        with self.assertRaisesRegex(ValueError, "get_default_qconfig_mapping"):
            m = prepare_fx(m, qconfig_mapping, data)

    @skipIfNoFBGEMM
    def test_general_shape_ops(self):
        """ A test that checks dequantize will be swapped for
        all supported general shape ops like aten::flatten
        without actually checking for execution of these ops
        """
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.maxpool1d = torch.nn.MaxPool1d(kernel_size=3)
                self.maxpool2d = torch.nn.MaxPool2d(kernel_size=3)
                self.maxpool3d = torch.nn.MaxPool3d(kernel_size=3)
                self.dropout = torch.nn.Dropout()
                self.conv1 = torch.nn.Conv2d(3, 3, 3)
                self.conv2 = torch.nn.Conv2d(3, 3, 3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                # add_scalar
                x = x + 3
                # mul_scalar
                x = x * 3
                # add_scalar_out
                x += 3
                # mul_scalar_out
                x *= 3
                # add_scalar_relu
                x = x + 3
                x = F.relu(x)
                # add_scalar_relu_out
                x += 3
                x = F.relu(x)
                # mul_scalar_relu
                x = x * 3
                x = F.relu(x)
                # mul_scalar_relu_out
                x *= 3
                x = F.relu(x)
                x = self.maxpool1d(x)
                x = self.maxpool2d(x)
                x = self.maxpool3d(x)
                x = torch.flatten(x)
                x = x.reshape([-1])
                x = x.resize_(1, 1, x)
                x = x.view(-1)
                # prim::ListConstruct
                xs = [x, x]
                # prim::ListUnpack
                x, y = xs
                # prim::TupleConstruct
                xs = (x, x)
                # prim::TupleUnpack
                x, y = xs
                x = x.transpose(1, 2)
                x = x.contiguous()
                # chunk is not supported since observer only supports
                # observing single Tensor currently
                x, y = torch.chunk(x, 2)
                x = F.dropout(x)
                x = self.dropout(x)
                x = x.permute(0, 2, 3, 1)
                x = x.repeat_interleave(3, 1)
                x = torch.repeat_interleave(x, 3, 1)
                x = self.relu(x)
                x = F.relu(x)
                x = F.relu(x, inplace=True)
                x = x.relu()
                x.relu_()
                x = x.squeeze(0)
                x.squeeze_(0)
                x = torch.squeeze(x, 0)
                x = x.unsqueeze(0)
                x.unsqueeze_(0)
                x = torch.unsqueeze(x, 0)
                x = x.detach()
                x.detach_()
                x = x.repeat(4, 2)
                y = []
                y.append(x)
                z = torch.stack(y, 0)
                z = [z, z]
                x, _ = z
                x = self.conv2(x)
                return x

        example_inputs = (torch.rand(1, 3, 10, 10),)
        # This model is not executable since we just put all ops
        # in the same forward
        m = M().eval()
        qconfig_dict = {'': default_qconfig}
        prepared = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        # not runnable
        quantized = convert_fx(prepared)

        # This checks that the dequantize from the output of first conv
        # is being propagated to the end, so that we don't insert extra
        # observers and also successfully fused two quantized::conv2d
        # patterns
        # one quantize_per_tensor for input
        # check exact counts of quantize and dequantize
        count_check = {
            # input of conv and two outputs of getitem
            ns.call_function(torch.quantize_per_tensor) : 2,
            # output of the model and two outputs of getitem
            ns.call_method('dequantize') : 2
        }
        order_check = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Conv2d),
            ns.call_module(nnq.Conv2d),
            ns.call_method('dequantize'),
        ]
        self.checkGraphModuleNodes(
            quantized,
            expected_node_occurrence=count_check,
            expected_node_list=order_check)


        # Checking the is_reference output
        m = M().eval()
        qconfig_dict = {'': default_qconfig}
        prepared = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        # not runnable
        quantized = convert_to_reference_fx(prepared)


    @skipIfNoFBGEMM
    def test_ave_pool_with_custom_cfg(self):
        """ A test that checks correct patterns are produced for
        avg_pool2d with customized config
        """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.avg_pool2d = torch.nn.AvgPool2d(3)


            def forward(self, x):
                x = self.avg_pool2d(x)
                return x

        # This model is not executable since we just put all ops
        # in the same forward
        m = M().eval()
        # nothing to fuse so skipping the fuse step
        qconfig_dict = {'': default_qconfig}
        example_inputs = (torch.randn(1, 3, 3, 3),)
        prepared = prepare_fx(
            m, qconfig_dict, example_inputs=example_inputs,
            prepare_custom_config={"input_quantized_idxs": [0]})

        # not runnable
        quantized = convert_fx(prepared)

        # This checks that the dequantize from the output of first conv
        # is being propagated to the end, so that we don't insert extra
        # observers
        # check exact counts of quantize and dequantize
        count_check = {
            ns.call_method('dequantize') : 1
        }
        order_check = [
            ns.call_module(nn.AvgPool2d),
            ns.call_method('dequantize'),
        ]
        self.checkGraphModuleNodes(
            quantized,
            expected_node_occurrence=count_check,
            expected_node_list=order_check)

    @skipIfNoFBGEMM
    def test_general_value_ops(self):
        """ A test that checks correct patterns are produced for
        all supported general value ops like aten::avg_pool2d \
        without actually checking for execution of these ops
        """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.avg_pool1d = torch.nn.AvgPool1d(3)
                self.avg_pool2d = torch.nn.AvgPool2d(3)
                self.avg_pool3d = torch.nn.AvgPool3d(3)
                self.adaptive_avg_pool1d = torch.nn.AdaptiveAvgPool1d((1))
                self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.adaptive_avg_pool3d = torch.nn.AdaptiveAvgPool3d((1, 1, 1))

            def forward(self, x):
                x = self.conv(x)
                x = self.avg_pool1d(x)
                x = self.avg_pool2d(x)
                x = self.avg_pool3d(x)
                x = self.adaptive_avg_pool1d(x)
                x = self.adaptive_avg_pool2d(x)
                x = self.adaptive_avg_pool3d(x)
                x = F.avg_pool1d(x, 3)
                x = F.avg_pool2d(x, 3)
                x = F.avg_pool3d(x, 3)
                x = F.adaptive_avg_pool1d(x, (1))
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = F.adaptive_avg_pool3d(x, (1, 1, 1))
                x = torch.mean(x)
                x = torch.mean(x, [2, 3], False)
                x = x.mean()
                x = x.mean([2, 3], True)
                x = F.interpolate(x, 4, mode='nearest')
                x = F.interpolate(x, 4, mode='linear')
                x = self.conv(x)
                return x

        # This model is not executable since we just put all ops
        # in the same forward
        m = M().eval()
        # nothing to fuse so skipping the fuse step
        qconfig_dict = {'': default_qconfig}
        example_inputs = (torch.randn(1, 3, 3, 3),)
        prepared = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        # not runnable
        quantized = convert_fx(prepared)

        # This checks that the dequantize from the output of first conv
        # is being propagated to the end, so that we don't insert extra
        # observers
        # check exact counts of quantize and dequantize
        count_check = {
            ns.call_function(torch.quantize_per_tensor) : 1,
            ns.call_method('dequantize') : 1
        }
        order_check = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Conv2d),
            ns.call_module(nnq.Conv2d),
            ns.call_method('dequantize'),
        ]
        self.checkGraphModuleNodes(
            quantized,
            expected_node_occurrence=count_check,
            expected_node_list=order_check)

    def test_copy_node_fp32_input(self):
        """ CopyNode works for both fp32 and int8 inputs, this is a test to make
        sure that a CopyNode can be successfully quantized in both cases
        """
        class M(torch.nn.Module):
            def forward(self, x):
                x = x.relu()
                return x

        m = M().eval()
        m = prepare_fx(m, {"": default_reuse_input_qconfig}, example_inputs=(torch.randn(1),))
        m = convert_fx(m)
        # make sure it runs
        m(torch.rand(1))

    def test_getitem(self):
        """ Make sure we only insert observer for getitem if the following node is matched
        or needs to be quantized
        """
        class M(torch.nn.Module):
            def forward(self, xs):
                x = xs[0]
                return x

        m = M().eval()
        example_inputs = (torch.rand(1, 2),)
        qconfig_mapping = get_default_qconfig_mapping()
        m = prepare_fx(m, qconfig_mapping, example_inputs=example_inputs)
        self.checkGraphModuleNodes(m, expected_node_occurrence={
            ns.call_module(torch.ao.quantization.MinMaxObserver): 0
        })
        m = convert_fx(m)
        m(*example_inputs)

        class M2(torch.nn.Module):
            def forward(self, xs):
                x = xs[0]
                x = torch.sigmoid(x)
                return x

        m2 = M2().eval()
        example_inputs = ([torch.rand(1, 2)],)
        qconfig_mapping = get_default_qconfig_mapping()
        m2 = prepare_fx(m2, qconfig_mapping, example_inputs=example_inputs)
        self.checkGraphModuleNodes(m2, expected_node_occurrence={
            ns.call_module(torch.ao.quantization.FixedQParamsObserver): 2
        })
        m2 = convert_fx(m2)
        self.checkGraphModuleNodes(m2, expected_node_list=[
            ns.call_function(torch.quantize_per_tensor),
            ns.call_method("dequantize")
        ])
        m2(*example_inputs)

        # testing prepare recognizes non-Tensor input for getitem
        class M3(torch.nn.Module):
            def forward(self, x):
                s = x.shape
                n, c = s[:2]
                x = torch.sigmoid(x)
                return x

        m3 = M3().eval()
        example_inputs = (torch.rand(1, 2, 3, 4),)
        qconfig_mapping = get_default_qconfig_mapping()
        m3 = prepare_fx(m3, qconfig_mapping, example_inputs=example_inputs)
        self.checkGraphModuleNodes(m3, expected_node_occurrence={
            ns.call_module(torch.ao.quantization.FixedQParamsObserver): 2
        })
        m3 = convert_fx(m3)
        self.checkGraphModuleNodes(m3, expected_node_list=[
            ns.call_function(torch.quantize_per_tensor),
            ns.call_method("dequantize")
        ])
        m3(*example_inputs)


    @skipIfNoFBGEMM
    def test_fixed_qparams_ops(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.sigmoid = torch.nn.Sigmoid()
                self.hardsigmoid = torch.nn.Hardsigmoid()
                self.tanh = torch.nn.Tanh()
                self.softmax = torch.nn.Softmax(dim=0)

            def forward(self, x):
                x = self.conv(x)
                # F.sigmoid is deprecated
                x = self.sigmoid(x)
                x = torch.sigmoid(x)
                x = x.sigmoid()
                x = self.hardsigmoid(x)
                x = F.hardsigmoid(x)
                x = F.hardsigmoid(x, inplace=True)
                x = self.tanh(x)
                # F.tanh is deprecated
                x = torch.tanh(x)
                x = x.tanh()
                # TODO(future PR): handle F.softmax
                x = self.softmax(x)
                return x

        for eval_mode in [True, False]:
            # This model is not executable since we just put all ops
            # in the same forward
            m = M()
            if eval_mode:
                m.eval()
                qconfig_mapping = get_default_qconfig_mapping()
                prepare = prepare_fx
                fq_count = 10
            else:
                m.train()
                qconfig_mapping = get_default_qat_qconfig_mapping()
                prepare = prepare_qat_fx
                fq_count = 10
            # nothing to fuse so skipping the fuse step
            m_copy = copy.deepcopy(m)
            example_inputs = (torch.rand(3, 3, 3, 3),)
            prepared = prepare(m, qconfig_mapping, example_inputs=example_inputs)
            prepared_copy = copy.deepcopy(prepared)
            # check that prepare does not change model result
            if eval_mode:
                self.assertEqual(m_copy(*example_inputs), prepared_copy(*example_inputs))
            # check the correct number of activation_post_process is inserted
            expected_activation_post_process = FixedQParamsObserver if eval_mode else FixedQParamsFakeQuantize
            count_check = {
                ns.call_module(expected_activation_post_process) : fq_count,
            }
            self.checkGraphModuleNodes(
                prepared,
                expected_node_occurrence=count_check)
            # not runnable
            quantized = convert_fx(prepared)
            quantized_reference = convert_to_reference_fx(prepared_copy)

            # This checks that the dequantize from the output of first conv
            # is being propagated to the end, so that we don't insert extra
            # observers
            # check exact counts of quantize and dequantize
            count_check = {
                ns.call_function(torch.quantize_per_tensor) : 1,
                ns.call_method('dequantize') : 1
            }
            order_check = [
                ns.call_function(torch.quantize_per_tensor),
                ns.call_module(nnq.Conv2d),
                ns.call_module(nn.Sigmoid),
                ns.call_module(nnq.Softmax),
                ns.call_method('dequantize'),
            ]
            self.checkGraphModuleNodes(
                quantized,
                expected_node_occurrence=count_check,
                expected_node_list=order_check)

            reference_count_check = {
                ns.call_function(torch.quantize_per_tensor) : 12,
                ns.call_method('dequantize') : 12
            }
            reference_order_check = [
                ns.call_function(torch.quantize_per_tensor),
                ns.call_method('dequantize'),
                ns.call_module(nnqr.Conv2d),
                ns.call_function(torch.quantize_per_tensor),
                ns.call_method('dequantize'),
                ns.call_module(nn.Sigmoid),
                ns.call_function(torch.quantize_per_tensor),
                ns.call_method('dequantize'),
                ns.call_module(nn.Softmax),
                ns.call_function(torch.quantize_per_tensor),
                ns.call_method('dequantize'),
            ]
            self.checkGraphModuleNodes(
                quantized_reference,
                expected_node_occurrence=reference_count_check,
                expected_node_list=reference_order_check)

            # Verify that softmax scale and zero_point are correct
            self.assertTrue(quantized.softmax.scale - (1.0 / 256) <= 1e-8)
            self.assertTrue(quantized.softmax.zero_point == 0)

    def test_float_functional(self):
        class TorchAdd(nn.Module):
            """Wrapper around torch.add so that all ops can be found at build"""
            def __init__(self):
                super().__init__()
                self.add_func = nnq.FloatFunctional()

            def forward(self, x, y):
                return self.add_func.add(x, y)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ff1 = TorchAdd()
                self.ff2 = nnq.FloatFunctional()
                self.ff3 = nnq.FloatFunctional()
                self.ff4 = nnq.FloatFunctional()
                self.ff5 = nnq.FloatFunctional()
                self.ff6 = nnq.FloatFunctional()

            def forward(self, x):
                x = self.ff1(x, x)
                x = self.ff2.add_scalar(x, 3)
                x = self.ff3.mul(x, x)
                x = self.ff4.mul_scalar(x, 3)
                x = self.ff5.add_relu(x, x)
                x = self.ff6.cat([x])
                return x

        example_inputs = (torch.rand(3, 3),)
        # Note: QAT test succeeded by chance, to make it actually work
        # we need to fix eager mode FloatFunctional by removing
        # activation_post_process in add_scalar and mul_scalar
        for quant_type in self.static_quant_types:
            m = M()
            ref_m = torch.ao.quantization.QuantWrapper(M())
            is_qat = quant_type == QuantType.QAT
            if is_qat:
                m.train()
                ref_m.train()
                qconfig = default_qat_qconfig
                expected_act_post_process = torch.ao.quantization.FakeQuantize
            else:
                m.eval()
                ref_m.eval()
                qconfig = default_qconfig
                expected_act_post_process = torch.ao.quantization.MinMaxObserver

            prepare_fx_function = prepare_qat_fx if is_qat else prepare_fx
            qconfig_dict = {"": qconfig}
            m = prepare_fx_function(m, qconfig_dict, example_inputs=example_inputs)
            node_occurrence = {
                ns.call_module(expected_act_post_process): 7,
                ns.call_module(torch.ao.nn.quantized.FloatFunctional): 0
            }
            self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)
            m(*example_inputs)
            node_list = [
                ns.call_function(torch.quantize_per_tensor),
                ns.call_function(torch.ops.quantized.add),
                ns.call_function(torch.ops.quantized.add),
                ns.call_function(torch.ops.quantized.mul),
                ns.call_function(torch.ops.quantized.mul),
                ns.call_function(torch.ops.quantized.add_relu),
                ns.call_function(torch.cat),
                ns.call_method('dequantize')
            ]
            m = convert_fx(m)
            self.checkGraphModuleNodes(m, expected_node_list=node_list)

            # make sure numerics match with eager mode
            ref_m.qconfig = qconfig
            prepare_function = prepare_qat if is_qat else prepare
            ref_m = prepare_function(ref_m)
            ref_m(*example_inputs)
            ref_m = convert(ref_m)
            # FX Graph Mode and Eager Mode now diverages in numerics of add_scalar and mul_scalar
            # self.assertEqual(m(data), ref_m(data))

    def test_embedding(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=12)

            def forward(self, indices):
                return self.emb(indices)

        for qconfig_type in [float_qparams_weight_only_qconfig, float_qparams_weight_only_qconfig_4bit]:
            model = M().eval()
            indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])
            example_inputs = (indices,)
            quantized_node = ns.call_module(nnq.Embedding)
            configs = [
                (qconfig_type, ns.call_module(nnq.Embedding)),
                (None, ns.call_module(nn.Embedding)),
                (default_qconfig, ns.call_module(nn.Embedding)),
            ]

            for qconfig, node in configs:
                qconfig_dict = {"": qconfig}
                m = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)
                self.checkGraphModuleNodes(m, expected_node_occurrence={
                    ns.call_module(torch.ao.quantization.MinMaxObserver): 0
                })
                m = convert_fx(m)
                self.checkGraphModuleNodes(m, expected_node=node)
                # make sure it runs
                m(*example_inputs)

    def test_embedding_bag(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.EmbeddingBag(num_embeddings=10, embedding_dim=12, include_last_offset=True)

            def forward(self, indices, offsets):
                return self.emb(indices, offsets)

        indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])
        offsets = torch.tensor([0, 19, 20, 28, 28, 32])
        quantized_node = ns.call_module(nnq.EmbeddingBag)
        example_inputs = (indices, offsets)

        for dtype in [torch.quint8, torch.quint4x2]:
            model = M().eval()
            float_qparams_observer = PerChannelMinMaxObserver.with_args(dtype=dtype,
                                                                        qscheme=torch.per_channel_affine_float_qparams,
                                                                        ch_axis=0)
            float_qparams_qconfig = QConfig(activation=default_placeholder_observer,
                                            weight=float_qparams_observer)
            self.checkGraphModeFxOp(
                model,
                example_inputs,
                QuantType.DYNAMIC,
                quantized_node,
                custom_qconfig_dict={"": float_qparams_qconfig}
            )

        # check it works in None and static qconfig
        for qconfig in [None, default_qconfig]:
            qconfig_dict = {"": default_qconfig}
            m = M().eval()
            m = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)
            self.checkGraphModuleNodes(m, expected_node_occurrence={
                ns.call_module(torch.ao.quantization.MinMaxObserver): 0
            })
            m = convert_fx(m)
            self.checkGraphModuleNodes(m, expected_node=ns.call_module(nn.EmbeddingBag))
            # make sure it runs
            m(*example_inputs)

    def _test_rnn_impl(self, qconfigs, M, module_type_strs, module_types, sample_input):
        options = itertools.product(qconfigs, module_type_strs)
        for qconfig, module_type_str in options:
            model_eager = M(module_type_str).eval()
            model_graph = copy.deepcopy(model_eager)
            if torch.backends.quantized.engine == 'qnnpack' and \
               qconfig is float16_dynamic_qconfig:
                continue
                # fp16 dynamic quant is not supported for qnnpack

            eager_qconfig_dict = {x : qconfig for x in module_types}
            model_eager = quantize_dynamic(model_eager, qconfig_spec=eager_qconfig_dict)

            graph_qconfig_dict = {
                "object_type": [
                    (x, qconfig) for x in module_types
                ]
            }
            model_graph = prepare_fx(model_graph, graph_qconfig_dict, example_inputs=(sample_input,))
            model_graph = convert_fx(model_graph)
            self.assertEqual(model_eager(sample_input), model_graph(sample_input))
            self.checkScriptable(model_graph, [[sample_input]], True)

    @override_qengines
    def test_rnn_cell(self):
        if torch.backends.quantized.engine not in ('fbgemm', 'qnnpack'):
            return
        qconfigs = [per_channel_dynamic_qconfig, default_dynamic_qconfig, float16_dynamic_qconfig]
        module_type_strs = ['LSTMCell', 'GRUCell', 'RNNTanh', 'RNNReLU']
        module_types = [torch.nn.LSTMCell, torch.nn.GRUCell, torch.nn.RNNCell]
        sample_input = torch.tensor([[100, -155],
                                     [-155, 100],
                                     [100, -155]], dtype=torch.float)
        self._test_rnn_impl(qconfigs, RNNCellDynamicModel, module_type_strs, module_types, sample_input)

    @override_qengines
    def test_rnn(self):
        if torch.backends.quantized.engine not in ('fbgemm', 'qnnpack'):
            return
        qconfigs = [per_channel_dynamic_qconfig, default_dynamic_qconfig, float16_dynamic_qconfig]
        module_type_strs = ['LSTM']
        module_types = [torch.nn.LSTM]
        niter = 10
        sample_input = torch.tensor([[100, -155],
                                     [-155, 100],
                                     [100, -155]], dtype=torch.float).unsqueeze(0).repeat(niter, 1, 1)
        self._test_rnn_impl(qconfigs, RNNDynamicModel, module_type_strs, module_types, sample_input)

    def _test_conv_transpose_impl(
            self, float_cls: Callable, q_cls: Callable, data: torch.Tensor):
        with override_quantized_engine('qnnpack'):
            # Create fp32 versions of FX and Eager models
            m1 = torch.nn.Sequential(float_cls(1, 1, 1))
            m2 = torch.nn.Sequential(float_cls(1, 1, 1))
            m2.load_state_dict(m1.state_dict())
            m2 = torch.ao.quantization.QuantWrapper(m2)
            # FX graph
            result_dict = self.checkGraphModeFxOp(
                m1, (data,), QuantType.STATIC,
                expected_node_occurrence={
                    ns.call_module(q_cls): 1,
                })
            q_result1 = result_dict["quantized_output"]
            # Eager
            m2.qconfig = get_default_qconfig(torch.backends.quantized.engine)
            m2.eval()
            m2p = torch.ao.quantization.prepare(m2)
            m2p(data)
            m2q = torch.ao.quantization.convert(m2p)
            q_result2 = m2q(data)
            # verify results match
            self.assertEqual(q_result1, q_result2)

    @unittest.skipUnless('qnnpack' in supported_qengines,
                         "This Pytorch Build has not been built with or does not support QNNPACK")
    def test_conv_transpose_1d(self):
        self._test_conv_transpose_impl(
            torch.nn.ConvTranspose1d, nnq.ConvTranspose1d, torch.randn(4, 1, 4))

    @unittest.skipUnless('qnnpack' in supported_qengines,
                         "This Pytorch Build has not been built with or does not support QNNPACK")
    def test_conv_transpose_2d(self):
        self._test_conv_transpose_impl(
            torch.nn.ConvTranspose2d, nnq.ConvTranspose2d, torch.randn(4, 1, 4, 4))

    def test_reshape_fp16(self):
        class M(torch.nn.Module):
            def __init__(self, w, b):
                super().__init__()
                self.w = w
                self.b = b

            def forward(self, x):
                x = torch.nn.functional.linear(x, self.w)
                x = x.reshape(-1, 4)
                x = torch.nn.functional.linear(x, self.w)
                return x

        w = torch.randn(4, 4)
        b = torch.randn(4)
        m = M(w, b).eval()
        qconfig_dict = {
            # reshape will be quantized to fp16 as requested by this qconfig
            "": float16_static_qconfig,
            "object_type": [
                (torch.nn.functional.linear, default_qconfig)
            ]
        }
        backend_config = get_test_only_legacy_native_backend_config()
        example_inputs = (torch.randn(1, 4),)
        m = prepare_fx(
            m, qconfig_dict, example_inputs=example_inputs,
            backend_config=backend_config)
        expected_occurrence = {
            # input and weight of first and second linear, output of first and second linear
            ns.call_module(torch.ao.quantization.MinMaxObserver): 6,
            # we insert placeholder observer for both input and output of reshape
            ns.call_module(torch.ao.quantization.PlaceholderObserver): 2
        }
        self.checkGraphModuleNodes(
            m,
            expected_node_occurrence=expected_occurrence
        )
        m = convert_fx(m, backend_config=backend_config)
        expected_occurrence = {
            ns.call_function(torch.quantize_per_tensor): 2,
            # dequantize after first linear, before reshape and before output
            ns.call_method("dequantize"): 3,
            # before reshape, to(fp16)
            ns.call_method("to"): 1,
            ns.call_function(torch.ops.quantized.linear): 2
        }
        self.checkGraphModuleNodes(
            m,
            expected_node_occurrence=expected_occurrence
        )
        # make sure it runs
        m(torch.randn(2, 4))

    def test_multiple_qconfigs_for_single_value(self):
        """ Test multiple qconfigs for a single value"""
        class M(torch.nn.Module):
            def __init__(self, w, b):
                super().__init__()
                self.w = w
                self.b = b

            def forward(self, x):
                x = torch.nn.functional.linear(x, self.w)
                x = torch.sigmoid(x)
                return x

        w = torch.randn(4, 4)
        b = torch.randn(4)
        m = M(w, b).eval()
        # TODO: use get_default_qconfig_mapping once it handles fp16
        qconfig_mapping = QConfigMapping() \
            .set_global(float16_static_qconfig) \
            .set_object_type(torch.nn.functional.linear, default_qconfig)
        example_inputs = (torch.randn(1, 4),)
        backend_config = get_test_only_legacy_native_backend_config()
        m = prepare_fx(
            m, qconfig_mapping, example_inputs=example_inputs,
            backend_config=backend_config)
        expected_occurrence = {
            # input and weight of linear, output of linear
            ns.call_module(torch.ao.quantization.MinMaxObserver): 3,
            # input and output of sigmoid
            ns.call_module(torch.ao.quantization.PlaceholderObserver): 2,
        }
        self.checkGraphModuleNodes(
            m,
            expected_node_occurrence=expected_occurrence
        )
        # make sure it runs
        m = convert_fx(m)
        expected_occurrence = {
            ns.call_function(torch.quantize_per_tensor): 1,
            ns.call_method("dequantize"): 3,
            ns.call_method("to"): 2
        }
        self.checkGraphModuleNodes(
            m,
            expected_node_occurrence=expected_occurrence
        )

    def test_boolean_tensor(self):
        """ Make sure we don't insert observer for boolean Tensors """
        class M(torch.nn.Module):
            def forward(self, x, mask):
                mask = mask.unsqueeze(0)
                mask = mask.unsqueeze(1)
                x = x.masked_fill(mask, 1)
                return x

        m = M().eval()
        example_inputs = (torch.rand(1, 2, 3, 4), torch.rand(3, 4).bool())
        m = prepare_fx(m, {"": default_qconfig}, example_inputs=example_inputs)
        expected_occurrence = {
            ns.call_module(torch.ao.quantization.MinMaxObserver): 0
        }
        self.checkGraphModuleNodes(
            m,
            expected_node_occurrence=expected_occurrence)
        m = convert_fx(m)
        m(*example_inputs)

    def test_chunk(self):
        class M(torch.nn.Module):
            def forward(self, x):
                x, y = torch.chunk(x, 2)
                x = x + y
                return x
        m = M().eval()
        example_inputs = (torch.rand(2, 2, 2, 2),)
        m = prepare_fx(m, {"": default_qconfig}, example_inputs=example_inputs)
        m(*example_inputs)
        m = convert_fx(m)
        m(*example_inputs)
        # make sure everything runs

    def test_ref_pattern_multi_use(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
                self.linear1 = torch.nn.Linear(5, 5)

            def forward(self, x):
                y = self.linear(x)
                z = self.linear1(x)
                a = torch.mul(z, 5)
                b = torch.add(z, 5)
                return (y, a, b)

        m = M().eval()
        qconfig_dict = {
            "": None,
            "object_type": [
                (torch.nn.Linear, get_default_qconfig("fbgemm")),
                (torch.nn.ReLU, get_default_qconfig("fbgemm")),
            ],
        }
        example_inputs = (torch.randn(1, 5),)
        m = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        m = convert_fx(m)
        expected_occurrence = {
            ns.call_function(torch.quantize_per_tensor): 1,
            ns.call_module(nnq.Linear): 2,
            ns.call_method("dequantize"): 2,
            ns.call_function(torch.add): 1,
            ns.call_function(torch.mul): 1,
        }
        self.checkGraphModuleNodes(
            m,
            expected_node_occurrence=expected_occurrence)

    def test_qmatmul(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                z = torch.matmul(x, y)
                return z

        m = M().eval()
        example_inputs = (torch.randn(2, 2), torch.randn(2, 2))
        qconfig_dict = get_default_qconfig_mapping("fbgemm")
        mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        mp(*example_inputs)
        mq = convert_fx(mp)
        expected_occurrence = {
            ns.call_function(torch.matmul): 0,
            ns.call_function(torch.ops.quantized.matmul): 1,
        }
        self.checkGraphModuleNodes(
            mq,
            expected_node_occurrence=expected_occurrence)
        # verify no crash
        res = mq(*example_inputs)

class TestQuantizeFxModels(QuantizationTestCase):
    @skipIfNoFBGEMM
    @unittest.skipIf(not TEST_CUDA, "gpu is not available.")
    def test_static_gpu_convert_basic(self):

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.relu1 = nn.ReLU()
                self.conv1 = nn.Conv2d(1, 6, 5)
                self.linear1 = nn.Linear(120, 1)

            def forward(self, x):
                x = self.relu1(self.conv1(x))
                y = self.linear1(x.view(-1))
                return y

        input = torch.randn((5, 1, 6, 6)).to('cuda')
        example_inputs = (input,)
        model = Net().to('cuda').eval()
        qconfig_dict = {"": torch.ao.quantization.get_default_qconfig('fbgemm')}
        model_prepared = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)
        model_prepared(*example_inputs)
        model_quantized = convert_to_reference_fx(model_prepared)
        out = model_quantized(*example_inputs)
        self.assertEqual(out.device.type, 'cuda')

    @skipIfNoFBGEMM
    @unittest.skipIf(not TEST_CUDA, "gpu is not available.")
    def test_switch_device_prepare_convert(self):

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.relu1 = nn.ReLU()
                self.conv1 = nn.Conv2d(1, 6, 5)
                self.linear1 = nn.Linear(120, 1)

            def forward(self, x):
                x = self.relu1(self.conv1(x))
                y = self.linear1(x.view(-1))
                return y

        for device in ['cuda', 'cpu']:
            device_after = 'cuda' if device == 'cpu' else 'cpu'
            input = torch.randn((5, 1, 6, 6)).to(device)
            model = Net().to(device).eval()
            qconfig_dict = {"": torch.ao.quantization.get_default_qconfig('fbgemm')}
            model_prepared = prepare_fx(model, qconfig_dict, example_inputs=(input,))
            model_prepared(input)
            model_prepared.to(device_after)
            model_quantized = convert_to_reference_fx(model_prepared)
            out = model_quantized(input.to(device_after))
            self.assertEqual(out.device.type, device_after)

    @skipIfNoFBGEMM
    @unittest.skipIf(not TEST_CUDA, "gpu is not available.")
    def test_prepare_serialize_switch_device_convert(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 6, 5)
                self.linear1 = nn.Linear(120, 1)

            def forward(self, x):
                x = self.conv1(x)
                y = self.linear1(x.view(-1))
                return y

        for device in ['cuda', 'cpu']:
            for device_after in ['cuda', 'cpu']:
                input = torch.randn((5, 1, 6, 6)).to(device)
                model = Net().to(device).eval()
                qconfig_dict = {"": torch.ao.quantization.get_default_qconfig('fbgemm')}
                model_prepared_first = prepare_fx(model, qconfig_dict, example_inputs=(input,))
                model_prepared_second = prepare_fx(model, qconfig_dict, example_inputs=(input,))
                model_prepared_first(input)
                state_dict = model_prepared_first.state_dict()
                del model_prepared_first
                model_prepared_second.load_state_dict(state_dict)
                model_prepared_second.to(device_after)
                model_quantized = convert_to_reference_fx(model_prepared_second)
                out = model_quantized(input.to(device_after))
                self.assertEqual(out.device.type, device_after)

    @skip_if_no_torchvision
    def test_model_dropout(self):
        from torchvision import models
        m = models.mobilenet_v3_small()
        qconfig_mapping = torch.ao.quantization.get_default_qat_qconfig_mapping('fbgemm')
        example_inputs = (torch.randn(1, 3, 224, 224),)
        mp = prepare_qat_fx(m, qconfig_mapping, example_inputs=example_inputs)
        mp(*example_inputs)
        with override_quantized_engine("qnnpack") if IS_ARM64 else contextlib.nullcontext():
            mq = convert_fx(mp)
        mq(*example_inputs)

    def _test_model_impl(
            self, mode, name, model, eager_quantizable_model,
            check_with_eager=True,
            diff_of_quant=None,
            diff_from_eager=None):
        if diff_of_quant is None or diff_from_eager is None:
            diff_of_quant = {}
            diff_from_eager = {}

        if mode not in diff_of_quant or mode not in diff_from_eager:
            diff_of_quant[mode] = {}
            diff_from_eager[mode] = {}

        input_tensor = torch.rand(1, 3, 224, 224)
        input_tensor_inception = torch.rand(1, 3, 299, 299)
        output_value = torch.randint(0, 1, (1,))

        # print('quantizing:', name, ' mode:', mode)
        if name == 'inception_v3':
            input_value = input_tensor_inception
        else:
            input_value = input_tensor

        qconfig = default_qconfig if mode == 'static' else default_qat_qconfig
        qconfig_dict = {'': qconfig}
        script = torch.jit.script(model)

        # make sure graph module and script module are both runanble
        original_out = model(input_value)
        is_not_tuple_out = not isinstance(original_out, tuple)
        script_out = script(input_value)

        # set to train just before quantization
        prepare_fx_fn = prepare_fx
        if mode != 'static':
            model.train()
            prepare_fx_fn = prepare_qat_fx

        prepared = prepare_fx_fn(model, qconfig_dict)

        if mode == 'ddp':
            mp.spawn(run_ddp,
                     args=(world_size, prepared),
                     nprocs=world_size,
                     join=True)
        elif mode == 'qat':
            assert prepared.training, 'prepared must be in training mode for qat'
            optimizer = torch.optim.SGD(prepared.parameters(), lr=0.0001)
            criterion = nn.CrossEntropyLoss()
            train_one_epoch(prepared, criterion, optimizer, [(input_value, output_value)], torch.device('cpu'), 1)
        else:
            for i in range(10):
                prepared(input_value)

        # print('after observation root:', prepared.root)

        qgraph = convert_fx(prepared)
        # print('after quantization root:', qgraph.root)
        # print('after quantization code:', qgraph.src)
        qgraph.eval()
        qgraph_script = torch.jit.script(qgraph)
        # print('quantized and scripted:', qgraph_script.graph)

        qgraph_out = qgraph(input_value)
        qgraph_script = qgraph_script(input_value)

        if is_not_tuple_out:
            diff_of_quant[mode][name] = (original_out - qgraph_out).abs().max()
            assert torch.allclose(qgraph_out, qgraph_script), 'graph, scripted graph'
        else:
            print('tuple output')

        if eager_quantizable_model is not None:
            # comparing to eager mode quantization
            qeager = eager_quantizable_model
            ref_out = qeager(input_value)
            qeager.qconfig = qconfig
            if mode == 'static':
                qeager.fuse_model()
                prepare(qeager, inplace=True)
            else:
                qeager.train()
                qeager.fuse_model()
                prepare_qat(qeager, inplace=True)

            # calibration
            if mode == 'ddp':
                mp.spawn(run_ddp,
                         args=(world_size, qeager),
                         nprocs=world_size,
                         join=True)
            elif mode == 'qat':
                assert qeager.training, 'qeager should be in training mode for qat'
                optimizer = torch.optim.SGD(qeager.parameters(), lr=0.0001)
                train_one_epoch(qeager, criterion, optimizer, [(input_value, output_value)], torch.device('cpu'), 1)
            else:
                for i in range(10):
                    qeager(input_value)

            # print('ref after observation:', qeager)

            convert(qeager, inplace=True)
            qeager.eval()

            # print('ref after quantization:', qeager)
            qeager_out = qeager(input_value)
            qeager_script = torch.jit.script(qeager)
            qscript_out = qeager_script(input_value)
            if is_not_tuple_out:
                diff_from_eager[mode][name] = (qeager_out - qgraph_out).abs().max()
                if check_with_eager:
                    self.assertEqual(diff_from_eager[mode][name], 0,
                                     'Result of graph mode quantization and ' +
                                     'eager mode quantization on model: ' + name +
                                     ' should match. Mode: ' + mode +
                                     ' diff:' + str(diff_from_eager[mode][name]))

    def _test_building_block(self, quant_type, BB):
        eager = BB().float()
        graph = copy.deepcopy(eager)

        if quant_type == QuantType.STATIC:
            qconfig = default_qconfig
            eager_prepare = prepare
            graph_prepare = prepare_fx
            eager.eval()
            graph.eval()
            calibrate_or_train = test_only_eval_fn
            data = self.img_data_2d
            is_qat = False
        else:
            assert quant_type == QuantType.QAT
            qconfig = default_qat_qconfig
            eager_prepare = prepare_qat
            graph_prepare = prepare_qat_fx
            eager.train()
            graph.train()
            calibrate_or_train = test_only_train_fn
            data = self.img_data_2d_train
            is_qat = True

        if hasattr(eager, "fuse_model"):
            eager.fuse_model()
        eager = QuantWrapper(eager)
        eager.qconfig = qconfig
        eager = eager_prepare(eager)

        qconfig_dict = {"": qconfig}
        graph = graph_prepare(graph, qconfig_dict, example_inputs=(data[0][0],))

        eager_out = eager(data[0][0])
        graph_out = graph(data[0][0])
        # Eager Mode and FX Graph Mode QAT now differ in numerics both
        # in Post Training and QAT because FX Graph Mode uses same fake_quant instances
        # for input and output of CopyNode
        # self.assertEqual(eager_out, graph_out)

        calibrate_or_train(eager, data)
        calibrate_or_train(graph, data)

        eager = convert(eager)
        graph = convert_fx(graph)

        eager_out = eager(data[0][0])
        graph_out = graph(data[0][0])

    @override_qengines
    def test_resnet_base(self):
        models = [ResNetBase]
        options = itertools.product(self.static_quant_types, models)
        for quant_type, M in options:
            self._test_building_block(quant_type, M)

    @skip_if_no_torchvision
    @skipIfNoFBGEMM
    @unittest.skip("skip for now since tbb failed")
    def test_torchvision(self):
        from torchvision import models
        from torchvision.models import quantization as quantized_models
        from torchvision.models.quantization.utils import _replace_relu

        def get_available_classification_models(models):
            return [k for k, v in models.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]

        model_list = get_available_classification_models(models)
        quantized_model_list = get_available_classification_models(quantized_models)

        quantized_model_list = set(quantized_model_list)
        # test eager and graph consistency
        model_list = quantized_model_list
        # mobilenet/inception_v3/googlenet qat is not working due to AdaptiveAveragePool qat
        # we might observe the output of AdaptiveAveragePool in the future
        # and re-enable the test
        fx_eager_not_matching = [
            ("mobilenet_v2", "qat"),
            ("inception_v3", "qat"),
            ("googlenet", "qat")
        ]  # because relu6 is replaced as relu in mobilenetv2

        diff_of_quant = {}
        diff_from_eager = {}
        modes = ['static', 'qat']
        options = itertools.product(modes, model_list)
        for mode, name in options:
            pretrained = name in quantized_model_list  # load pretrained model to compare with quantized model
            kwargs = {}
            # turn off transform input for inception_v3 since
            # it's not quantized in eager mode and in fx graph
            # mode we can't skip quantizing a method right now
            # (might be supported in the future)
            if name in ["inception_v3", "googlenet"]:
                kwargs["transform_input"] = False
            eager_quantizable_model = None
            if name in quantized_model_list:
                eager_quantizable_model = quantized_models.__dict__[name](pretrained=False, quantize=False, **kwargs).eval().float()
            # compare with eager mode quantized model when it is available
            pretrained = eager_quantizable_model is not None
            model = models.__dict__[name](pretrained=pretrained, **kwargs).eval().float()
            if name == "mobilenet_v2":
                _replace_relu(model)
            # disable aux logits
            if hasattr(model, "aux_logits"):
                model.aux_logits = False
                model.AuxLogits = None
                if eager_quantizable_model:
                    eager_quantizable_model.aux_logits = False
                    eager_quantizable_model.AuxLogits = None

            check_with_eager = (name, mode) not in fx_eager_not_matching
            self._test_model_impl(
                mode, name, model, eager_quantizable_model,
                check_with_eager,
                diff_of_quant, diff_from_eager)

        def print_diffs(diffs):
            for mode, diffs_for_mode in diffs.items():
                print('mode:', mode)
                for name, diff in diffs_for_mode.items():
                    print(name, ':', diff)

        # print('differences between float and quantized')
        # print_diffs(diff_of_quant)
        # print('----------------------')
        # print('differences between graph mode and eager mode')
        # print_diffs(diff_from_eager)
        # print('----------------------')

    @skip_if_no_torchvision
    @skipIfNoFBGEMM
    @unittest.skip("TODO: Test is always failing - https://github.com/pytorch/pytorch/issues/54979")
    def test_resnet18_ddp(self):
        from torchvision import models
        from torchvision.models import quantization as quantized_models
        eager_quantizable_model = quantized_models.__dict__[name](pretrained=False, quantize=False).eval().float()
        model = models.__dict__[name](pretrained=False).eval().float()
        self._test_model_impl(
            'ddp', 'resnet18', model, eager_quantizable_model)

    @override_qengines
    def test_qat_embeddingbag_linear(self):
        for device in get_supported_device_types():
            class EmbeddingBagLinear(torch.nn.Module):
                def __init__(self):
                    super(EmbeddingBagLinear, self).__init__()
                    self.emb = torch.nn.EmbeddingBag(num_embeddings=10, embedding_dim=12, mode='sum')
                    self.linear = torch.nn.Linear(12, 1).to(dtype=torch.float)

                def forward(self, input: torch.Tensor, offsets: Optional[torch.Tensor] = None,
                            per_sample_weights: Optional[torch.Tensor] = None):
                    x = self.emb(input, offsets, per_sample_weights)
                    x = self.linear(x)
                    return x

            qengine = torch.backends.quantized.engine
            qconfig_dict = {"": get_default_qat_qconfig(qengine),
                            "object_type": [(torch.nn.EmbeddingBag, default_embedding_qat_qconfig)]}

            train_indices = [[torch.randint(0, 10, (12, 12)), torch.randn((12, 1))] for _ in range(2)]
            eval_output = [[torch.randint(0, 10, (12, 1))]]

            model = EmbeddingBagLinear().train()
            prepared_fx_model = prepare_qat_fx(model, qconfig_dict, example_inputs=(train_indices[0][0],))
            test_only_train_fn(prepared_fx_model, train_indices)
            quant_model = convert_fx(prepared_fx_model,
                                     qconfig_mapping=qconfig_dict)

            def checkQuantized(model):
                # Make sure EmbeddingBag is now a quantized EmbeddingBag.
                self.assertTrue(type(model.emb), nn.quantized.EmbeddingBag)
                # Also test that Linear has been quantized.
                self.assertTrue(type(model.linear), nnq.Linear)

                test_only_eval_fn(model, eval_output)
                self.checkScriptable(model, eval_output)
                self.checkNoQconfig(model)
            checkQuantized(quant_model)


    @override_qengines
    def test_qat_embedding_linear(self):
        for device in get_supported_device_types():
            class EmbeddingLinear(torch.nn.Module):
                def __init__(self):
                    super(EmbeddingLinear, self).__init__()
                    self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=12)
                    self.linear = torch.nn.Linear(12, 1).to(dtype=torch.float)

                def forward(self, input: torch.Tensor):
                    x = torch.sum(self.emb(input), dim=1)
                    x = self.linear(x)
                    return x

            qengine = torch.backends.quantized.engine
            qconfig_dict = {"": get_default_qat_qconfig(qengine),
                            "object_type": [(torch.nn.Embedding, default_embedding_qat_qconfig)]}


            train_indices = [[torch.randint(0, 10, (12, 12)), torch.randn((12, 1))] for _ in range(2)]
            eval_output = [[torch.randint(0, 10, (12, 1))]]

            model = EmbeddingLinear().train()
            prepared_fx_model = prepare_qat_fx(model, qconfig_dict, example_inputs=(train_indices[0][0],))
            test_only_train_fn(prepared_fx_model, train_indices)
            quant_model = convert_fx(prepared_fx_model,
                                     qconfig_mapping=qconfig_dict)

            def checkQuantized(model):
                # Make sure EmbeddingBag is now a quantized EmbeddingBag.
                self.assertTrue(type(model.emb), nn.quantized.Embedding)
                # Also test that Linear has been quantized.
                self.assertTrue(type(model.linear), nnq.Linear)

                test_only_eval_fn(model, eval_output)
                self.checkScriptable(model, eval_output)
                self.checkNoQconfig(model)
            checkQuantized(quant_model)

    @given(
        device=st.sampled_from(
            ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        )
    )
    @settings(deadline=None)
    @override_qengines
    def test_qat_functional_linear(self, device):
        if torch.backends.quantized.engine not in ('fbgemm', 'qnnpack'):
            return

        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.ones(5, 5)
                self.b = torch.zeros(5)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.w, self.b)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mods1 = torch.nn.Sequential(Linear(), Linear())
                self.mods2 = Linear()

            def forward(self, x):
                x = self.mods1(x)
                x = self.mods2(x)
                return x

        model = M().train()
        ref_fake_quant = FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            reduce_range=False,
        )
        ref_weight_fake_quant = FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            reduce_range=False,
        )
        ref_qat_qconfig = QConfig(
            activation=ref_fake_quant, weight=ref_weight_fake_quant
        )
        qconfig_dict = {"": ref_qat_qconfig}
        example_inputs = (torch.randn(1, 5),)
        prepared_ref = prepare_qat_fx(model, qconfig_dict, example_inputs=example_inputs)

        custom_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            reduce_range=False,
        )
        custom_weight_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            reduce_range=False,
        )
        custom_qconfig = QConfig(
            activation=custom_fake_quant, weight=custom_weight_fake_quant
        )
        custom_qconfig_dict = {"": custom_qconfig}
        prepared = prepare_qat_fx(model, custom_qconfig_dict, example_inputs=example_inputs)

        prepared.to(device)
        prepared_ref.to(device)

        prepared.apply(torch.ao.quantization.disable_fake_quant)
        prepared.apply(torch.ao.quantization.disable_observer)
        prepared_ref.apply(torch.ao.quantization.disable_fake_quant)
        prepared_ref.apply(torch.ao.quantization.disable_observer)

        inp = torch.randn(5, 5, device=device, requires_grad=True)
        for i in range(10):
            if i == 2:
                prepared.apply(torch.ao.quantization.enable_observer)
                prepared_ref.apply(torch.ao.quantization.enable_observer)
            if i == 4:
                prepared.apply(torch.ao.quantization.enable_fake_quant)
                prepared_ref.apply(torch.ao.quantization.enable_fake_quant)

            inp = torch.randn(5, 5, device=device, requires_grad=True)
            out_ref = prepared_ref(inp)
            out = prepared(inp)
            torch.testing.assert_close(out, out_ref)

            # try backward pass
            labels = torch.randn(5, 5, device=device)
            loss = (out - labels).sum()
            grad = torch.autograd.grad(loss, [inp])
            loss_ref = (out_ref - labels).sum()
            grad_ref = torch.autograd.grad(loss_ref, [inp])
            torch.testing.assert_close(grad[0], grad_ref[0])

        if 'fbgemm' in torch.backends.quantized.supported_engines:
            # During the lowering step in convert, fold_weight calls quantized::linear_prepack
            # which doesn't support QuantizedCuda backend
            prepared.cpu()
            prepared_ref.cpu()
            converted = convert_fx(prepared)
            converted_ref = convert_fx(prepared_ref)
            inp = torch.rand(5, 5)
            out = converted(inp)
            out_ref = converted_ref(inp)

            torch.testing.assert_close(out, out_ref)
if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_quantization.py TESTNAME\n\n"
                       "instead.")
