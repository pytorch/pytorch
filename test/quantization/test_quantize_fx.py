import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nniq
import torch.multiprocessing as mp

# graph mode quantization based on fx
from torch.quantization.quantize_fx import (
    prepare_fx,
    convert_fx,
    prepare_qat_fx,
)

from torch.quantization.fx.quantization_patterns import DefaultNodeQuantizeHandler

from torch.quantization.fx.pattern_utils import (
    is_match,
    MatchAllNode,
)

from torch.quantization import (
    QuantType,
    QuantStub,
    DeQuantStub,
    QuantWrapper,
    quant_type_to_str,
    default_qconfig,
    default_dynamic_qconfig,
    default_qat_qconfig,
    per_channel_dynamic_qconfig,
    float16_dynamic_qconfig,
    float16_static_qconfig,
    float_qparams_weight_only_qconfig,
    get_default_qconfig,
    get_default_qat_qconfig,
    fuse_modules,
    prepare,
    prepare_qat,
    convert,
    quantize_dynamic,
    default_placeholder_observer,
    PerChannelMinMaxObserver,
    QConfigDynamic,
    FixedQParamsFakeQuantize,
)

# test utils
from torch.testing._internal.common_cuda import TEST_MULTIGPU, TEST_CUDA
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    skipIfNoFBGEMM,
    skip_if_no_torchvision,
    train_one_epoch,
    run_ddp,
    test_only_eval_fn,
    test_only_train_fn,
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

from torch.testing._internal.common_utils import TemporaryFileName

from torch.testing._internal.common_quantization import NodeSpec as ns

from torch.testing import FileCheck

import copy
import itertools
import operator
import unittest
import io
from typing import Callable

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
    def __init__(self, binary_op, ibinary_op, is_inplace, is_functional_relu,
                 is_scalar):
        """ ibinary_op means inplace binary op
        """
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1).float()
        self.conv2 = torch.nn.Conv2d(1, 1, 1).float()
        self.op = ibinary_op if ibinary_op and is_inplace else binary_op
        self.is_functional_relu = is_functional_relu
        self.is_scalar = is_scalar
        self.relu = F.relu if self.is_functional_relu \
            else torch.nn.ReLU()

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
        m = prepare_qat_fx(m, {})
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
        from torch.quantization.quantize_fx import fuse_fx
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
        from torch.quantization.quantize_fx import fuse_fx
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
                self.assertTrue(is_match(modules, n, pattern))

    def _get_conv_linear_test_cases(self):
        """ Returns a list of test cases, with format:
        is_dynamic, ModuleClass, module_constructor_inputs,
        inputs, quantized_node, weight_prepack_op
        """
        class Conv1d(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = torch.nn.Parameter(weight)
                self.stride = 1
                self.padding = 0
                self.dilation = 1
                self.groups = 1

            def forward(self, x):
                return F.conv1d(x, self.weight, None, self.stride, self.padding, self.dilation, self.groups)

        conv1d_input = torch.rand(1, 3, 224)
        conv1d_weight = torch.rand(3, 3, 3)

        class Conv2d(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = torch.nn.Parameter(weight)
                self.stride = (1, 1)
                self.padding = (0, 0)
                self.dilation = (1, 1)
                self.groups = 1

            def forward(self, x):
                return F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation, self.groups)

        conv2d_input = torch.rand(1, 3, 224, 224)
        conv2d_weight = torch.rand(3, 3, 3, 3)

        class Conv3d(torch.nn.Module):
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

        conv3d_input = torch.rand(1, 3, 32, 224, 224)
        conv3d_weight = torch.rand(3, 3, 3, 3, 3)

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
            (
                False,
                Conv1d,
                (conv1d_weight,),
                (conv1d_input,),
                ns.call_function(torch.ops.quantized.conv1d),
                ns.call_function(torch.ops.quantized.conv1d_prepack),
            ),
            (
                False,
                Conv2d,
                (conv2d_weight,),
                (conv2d_input,),
                ns.call_function(torch.ops.quantized.conv2d),
                ns.call_function(torch.ops.quantized.conv2d_prepack),
            ),
            (
                False,
                Conv3d,
                (conv3d_weight,),
                (conv3d_input,),
                ns.call_function(torch.ops.quantized.conv3d),
                ns.call_function(torch.ops.quantized.conv3d_prepack),
            ),
            (
                True,
                Linear,
                (linear_weight,),
                (linear_input,),
                ns.call_function(torch.ops.quantized.linear_dynamic),
                ns.call_function(torch.ops.quantized.linear_prepack),
            ),
            (
                False,
                Linear,
                (linear_weight,),
                (linear_input,),
                ns.call_function(torch.ops.quantized.linear),
                ns.call_function(torch.ops.quantized.linear_prepack),
            ),
            (
                True,
                LinearModule,
                (),
                (linear_module_input,),
                ns.call_module(nnqd.Linear),
                None,
            ),
            (
                False,
                LinearModule,
                (),
                (linear_module_input,),
                ns.call_module(nnq.Linear),
                None,
            ),
        ]
        return tests

    """
    Unit tests for functionalities
    """
    @skipIfNoFBGEMM
    def test_functional_not_reference(self):
        """ Test quantizing functional conv and linear
        """
        tests = self._get_conv_linear_test_cases()
        for (is_dynamic, ModuleClass, module_constructor_inputs,
             inputs, quantized_node, weight_prepack_node) in tests:
            quant_type = QuantType.DYNAMIC if is_dynamic else QuantType.STATIC
            node_occurrence = dict()
            if weight_prepack_node:
                node_occurrence[weight_prepack_node] = 0
            self.checkGraphModeFxOp(
                ModuleClass(*module_constructor_inputs),
                inputs, quant_type,
                expected_node=quantized_node,
                expected_node_occurrence=node_occurrence,
                is_reference=False)

    @skipIfNoFBGEMM
    def test_functional_reference(self):
        """ Test quantizing functional conv and linear with reference option
        """
        tests = self._get_conv_linear_test_cases()
        for (is_dynamic, ModuleClass, module_constructor_inputs,
             inputs, quantized_node, weight_prepack_node) in tests:
            quant_type = QuantType.DYNAMIC if is_dynamic else QuantType.STATIC
            node_occurrence = dict()
            if weight_prepack_node:
                node_occurrence[weight_prepack_node] = 0
                node_occurrence[quantized_node] = 0
            self.checkGraphModeFxOp(
                ModuleClass(*module_constructor_inputs),
                inputs, quant_type,
                expected_node_occurrence=node_occurrence,
                is_reference=True)

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
        prepared = prepare_fx(m, qconfig_dict)
        quantized = convert_fx(prepared, is_reference=True)
        qparams = (quantized._input_scale_0, quantized._input_zero_point_0)
        weight_obs = qconfig.weight()
        weight_obs(quantized.weight)
        # Get the actual value to avoid tensor size mismatch error, torch.Size([]) vs torch.Size([1])
        ref_qparams = (weight_obs.calculate_qparams()[0].item(), weight_obs.calculate_qparams()[1].item())
        self.assertEqual(qparams, ref_qparams)

    def test_conv_bn_relu(self):
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
            result = self.checkGraphModeFxOp(
                m,
                self.img_data_dict[dim],
                quant_type,
                expected_node=expected_node,
            )

            # check numerics
            qengine = torch.backends.quantized.engine
            if quant_type == QuantType.STATIC:
                m_eager.eval()
                qconfig = get_default_qconfig(qengine)
                prepare_fn = prepare
            else:
                m_eager.train()
                qconfig = get_default_qat_qconfig(qengine)
                prepare_fn = prepare_qat

            fuse_list = ["conv", "bn"]
            if has_relu:
                fuse_list.append("relu")
            fuse_modules(m_eager, fuse_list, inplace=True)
            m_eager.qconfig = qconfig
            m_eager = prepare_fn(m_eager)
            m_eager(*self.img_data_dict[dim][0])
            m_eager = convert(m_eager)
            result_eager = m_eager(*self.img_data_dict[dim][0])
            self.assertEqual(result, result_eager)


    @skipIfNoFBGEMM
    def test_dynamic_quant_fp16(self):
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
                node_occurrence = dict()
                if weight_prepack_node:
                    node_occurrence[weight_prepack_node] = 0
                m = ModuleClass(*module_constructor_inputs).eval()
                qconfig_dict = {"": float16_dynamic_qconfig}
                m = prepare_fx(m, qconfig_dict)
                m = convert_fx(m, is_reference=is_reference)
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
        qconfig_dict = {'': torch.quantization.get_default_qat_qconfig(qengine)}
        device = torch.device('cuda:0')
        model.to(device)

        # QAT prepare
        model = prepare_qat_fx(model, qconfig_dict)

        # ensure that running an input on CUDA works without any needed changes
        input = torch.randn(4, 1, 4, 4, device=device)
        model(input)

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

        dict_input = {"input": torch.randn(1, 1, 1, 1)}
        m = M().eval()
        qconfig_dict = {"": default_qconfig}
        m = prepare_fx(m, qconfig_dict)
        m(dict_input)
        m = convert_fx(m)
        m(dict_input)

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

        tensor_input = torch.randn(3, 1, 1, 1)
        m = M().eval()
        qconfig_dict = {
            "": None,
            "object_type": [
                (nn.Conv2d, default_qconfig),
            ]
        }
        # make sure it runs
        m = prepare_fx(m, qconfig_dict)
        m(tensor_input)
        m = convert_fx(m)
        m(tensor_input)

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

        data = torch.randn(1, 1, 1, 1)
        # instantiate M and RefM and align the parameters
        original_m = M().eval()
        original_ref_m = RefM().eval()
        original_ref_m.conv1.weight = torch.nn.Parameter(original_m.conv.weight.detach())
        original_ref_m.conv1.bias = torch.nn.Parameter(original_m.conv.bias.detach())
        original_ref_m.conv2.weight = torch.nn.Parameter(original_m.standalone.conv.weight.detach())
        original_ref_m.conv2.bias = torch.nn.Parameter(original_m.standalone.conv.bias.detach())

        for is_name in [True, False]:
            if is_name:
                prepare_config = {
                    "standalone_module_name": [("standalone", None, interface_config)]
                }
            else:
                prepare_config = {
                    "standalone_module_class": [(StandaloneModule, None, interface_config)]
                }

            original_m_copy = copy.deepcopy(original_m)
            original_ref_m_copy = copy.deepcopy(original_ref_m)

            qconfig_dict = {"": default_qconfig}
            # check prepared model
            m = prepare_fx(
                original_m_copy, qconfig_dict, prepare_custom_config_dict=prepare_config)
            # calibration
            m(data)
            self.checkGraphModuleNodes(m, expected_node_occurrence=prepare_count_check)
            self.checkGraphModuleNodes(m.standalone, expected_node_occurrence=standalone_prepare_count_check)

            # check converted/quantized model
            m = convert_fx(m)
            self.checkGraphModuleNodes(m, expected_node_occurrence=convert_count_check)
            self.checkGraphModuleNodes(m.standalone, expected_node_occurrence=standalone_convert_count_check)
            res = m(data)

            # quantize the reference model
            ref_m = prepare_fx(original_ref_m_copy, qconfig_dict)
            ref_m(data)
            ref_m = convert_fx(ref_m)
            ref_res = ref_m(data)
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
            ns.call_module(torch.quantization.MinMaxObserver): 2
        }
        # for input and output of conv in the standalone module
        standalone_prepare_count_check = {
            ns.call_module(torch.quantization.MinMaxObserver): 2
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
            ns.call_module(torch.quantization.MinMaxObserver): 2
        }
        # for output of conv in the standalone module
        standalone_prepare_count_check = {
            ns.call_module(torch.quantization.MinMaxObserver): 1
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
        m = prepare_fx(m, qconfig_dict)
        data = torch.randn(1, 1, 1, 1)
        m(data)
        m = convert_fx(m)
        m(data)
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
        m = prepare_fx(m, qconfig_dict)
        data = torch.randn(1, 1, 1, 1)
        m(data)
        m = convert_fx(m)
        m(data)
        # first conv is quantized, second conv is not quantized
        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Conv2d),
            ns.call_module(nnq.Conv2d),
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
        m = prepare_fx(m, qconfig_dict)
        data = torch.randn(1, 1, 1, 1)
        m(data, data)
        m = convert_fx(m)
        m(data, data)
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
        m = prepare_fx(m, qconfig_dict)
        data = torch.randn(1, 1, 1, 1)
        m(data)
        m = convert_fx(m)
        m(data)
        # first conv is quantized, second conv is not quantized
        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Conv2d),
            ns.call_module(nnq.Conv2d),
            ns.call_method("dequantize"),
        ]
        self.checkGraphModuleNodes(m, expected_node_list=node_list)

    def test_qconfig_precedence(self):
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

        m = M().eval()
        global_qconfig = default_qconfig
        object_type_qconfig = default_dynamic_qconfig
        module_name_regex_qconfig = float16_dynamic_qconfig
        module_name_qconfig = default_qat_qconfig
        qconfig_dict = {
            "": global_qconfig,
            "object_type": [(nn.Conv2d, object_type_qconfig)],
            "module_name_regex": [("module_conv*", module_name_regex_qconfig)],
            "module_name": [("module_conv2", module_name_qconfig)]}
        m = prepare_fx(m, qconfig_dict)
        self.assertEqual(m.linear.qconfig, global_qconfig)
        self.assertEqual(m.conv.qconfig, object_type_qconfig)
        self.assertEqual(m.module_conv1.qconfig, module_name_regex_qconfig)
        self.assertEqual(m.module_conv2.qconfig, module_name_qconfig)

    def test_remove_qconfig(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.avg_pool = torch.nn.AvgPool2d(1)

            def forward(self, x):
                return self.avg_pool(x)

        m = M().eval()
        qconfig_dict = {'': default_qconfig}
        m = prepare_fx(m, qconfig_dict)
        data = torch.randn(1, 1, 1, 1)
        m(data)
        m = convert_fx(m)
        m(data)
        for name, module in m.named_modules():
            self.assertFalse(hasattr(module, 'qconfig'),
                             'qconfig is not removed for ' + name)

    def test_return_none(self):
        class M(torch.nn.Module):
            def forward(self, x):
                pass

        m = M().eval()
        qconfig_dict = {'': torch.quantization.default_qconfig}
        m = prepare_fx(m, qconfig_dict)
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
        m = prepare_fx(m, qconfig_dict)
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
            m = M().eval()
            m = prepare_fx(m, qconfig_dict)
            m(torch.randn(2, 1, 3, 3))
            m = convert_fx(m)
            self.checkGraphModuleNodes(m, expected_node_list=node_list)
            # make sure it runs
            m(torch.randn(2, 1, 3, 3))

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
        qconfig_dict = {"": default_qconfig, "module_name": [("mods2", None)]}
        m = prepare_fx(model, qconfig_dict)
        m(torch.rand(5, 5))

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
        m = prepare_fx(m, {"": default_qconfig}, prepare_custom_config_dict)

        def assertAttrPreserved(m):
            self.assertTrue(hasattr(m, "preserved_attr"))
            self.assertTrue(m.preserved_attr, 3)

        assertAttrPreserved(m)
        convert_custom_config_dict = {
            "preserved_attributes": ["preserved_attr"]
        }
        m = convert_fx(m, convert_custom_config_dict=convert_custom_config_dict)
        assertAttrPreserved(m)

    @skipIfNoFBGEMM
    def test_qat_and_script(self):
        model = LinearModelWithSubmodule().train()
        qengine = torch.backends.quantized.engine
        qconfig_dict = {'': torch.quantization.get_default_qat_qconfig(qengine)}
        model = prepare_qat_fx(model, qconfig_dict)

        # ensure scripting works
        scripted = torch.jit.script(model)
        # run one round to make sure model runs
        x = torch.randn(5, 5)
        scripted(x)
        FileCheck().check_count('FakeQuantize = prim::GetAttr[name="', 4, exactly=True) \
                   .run(scripted.graph)

        # disable fake_quant and observer
        for epoch in range(3):
            if epoch == 1:
                scripted.apply(torch.quantization.disable_observer)
            if epoch == 2:
                scripted.apply(torch.quantization.disable_fake_quant)

        # ensure the fake_quant and observer have been disabled.
        matches = ['.fake_quant_enabled', '.observer_enabled']
        for key, v in scripted.state_dict().items():
            if any(x in key for x in matches):
                self.assertEqual(v, torch.tensor([0], dtype=torch.uint8))

        # enable them back
        scripted.apply(torch.quantization.enable_fake_quant)
        scripted.apply(torch.quantization.enable_observer)
        for key, v in scripted.state_dict().items():
            if any(x in key for x in matches):
                self.assertEqual(v, torch.tensor([1], dtype=torch.uint8))

    @skipIfNoFBGEMM
    def test_save_observer_state_dict(self):
        orig = LinearModelWithSubmodule().eval()
        model = orig
        qconfig_dict = {'': torch.quantization.get_default_qconfig('fbgemm')}
        model = prepare_fx(model, qconfig_dict)

        # run it through input
        x = torch.randn(5, 5)
        model(x)

        quant = convert_fx(model)

        # save state_dict of model
        obs_dict = torch.quantization.get_observer_state_dict(model)
        b = io.BytesIO()
        torch.save(obs_dict, b)
        b.seek(0)

        # Load the stats into new model
        model_2 = orig
        model_2 = prepare_fx(model_2, qconfig_dict)

        loaded_dict = torch.load(b)
        torch.quantization.load_observer_state_dict(model_2, loaded_dict)

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

        data = torch.randn(3, 3)
        # instantiate M and RefM and align the parameters
        original_m = M().eval()
        original_ref_m = RefM().eval()
        original_ref_m.linear1.weight = torch.nn.Parameter(original_m.linear.weight.detach())
        original_ref_m.linear1.bias = torch.nn.Parameter(original_m.linear.bias.detach())
        original_ref_m.linear2.weight = torch.nn.Parameter(original_m.custom.linear.weight.detach())
        original_ref_m.linear2.bias = torch.nn.Parameter(original_m.custom.linear.bias.detach())

        test_configs = {
            "static": (default_qconfig, StaticQuantCustomModule, 3),
            "dynamic": (default_dynamic_qconfig, DynamicQuantCustomModule, 0)
        }

        for quant_type in [QuantType.DYNAMIC]:
            key = quant_type_to_str(quant_type)
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

            # check prepared model
            m = prepare_fx(
                original_m,
                qconfig_dict,
                prepare_custom_config_dict=prepare_custom_config_dict)
            # calibration
            m(data)
            # all activation observers are inserted in the top level module
            count_check = {
                ns.call_module(torch.quantization.MinMaxObserver): num_observers
            }
            self.checkGraphModuleNodes(m, expected_node_occurrence=count_check)

            # check converted/quantized model
            m = convert_fx(
                m,
                convert_custom_config_dict=convert_custom_config_dict)
            if quant_type == QuantType.STATIC:
                count_check = {
                    ns.call_function(torch.quantize_per_tensor) : 1,
                    ns.call_module(nnq.Linear) : 1,
                    ns.call_method('dequantize') : 1,
                }
                self.checkGraphModuleNodes(m, expected_node_occurrence=count_check)
            self.assertEqual(type(m.custom), quantized_module_class)
            res = m(data)

            # quantize the reference model
            ref_m = prepare_fx(original_ref_m, qconfig_dict)
            ref_m(data)
            ref_m = convert_fx(ref_m)
            ref_res = ref_m(data)
            self.assertEqual(res, ref_res)

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
            prepare_custom_config_dict=prepare_custom_config_dict)

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
        qconfig_dict = {'': torch.quantization.default_qconfig}
        prepared = prepare_fx(m, qconfig_dict)
        # calibrate
        prepared(torch.randn(4, 1, 4, 4))
        # copy
        prepared_copy = copy.deepcopy(prepared)
        # quantize, should run with no errors
        quantized = convert_fx(prepared_copy)

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
        qconfig_dict = {'': torch.quantization.default_qconfig}
        m.eval()
        mp = torch.quantization.quantize_fx.prepare_fx(
            m, qconfig_dict,
            prepare_custom_config_dict=prepare_custom_config_dict)
        self.checkGraphModuleNodes(mp, expected_node_occurrence=prepare_count_check)
        mp(torch.randn(1, 1, 4, 4))
        mq = torch.quantization.quantize_fx.convert_fx(mp)
        self.checkGraphModuleNodes(mq, expected_node_occurrence=convert_count_check)

    def test_quantized_input_quantized_output(self):
        prepare_custom_config_dict = {
            'input_quantized_idxs': [0], 'output_quantized_idxs': [0]}
        prepare_count_check = {
            ns.call_module(torch.quantization.MinMaxObserver): 2,
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
            ns.call_module(torch.quantization.MinMaxObserver): 3,
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
            ns.call_module(torch.quantization.MinMaxObserver): 2,
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
            ns.call_module(torch.quantization.MinMaxObserver): 3,
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
        qconfig_dict = {'': torch.quantization.get_default_qconfig('fbgemm')}
        with self.assertRaises(AssertionError) as context:
            mp = prepare_fx(m, qconfig_dict)
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
        m = prepare_fx(model, qconfig_dict)
        m(torch.rand(5, 5))
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
        m(torch.rand(5, 5))
        # ensure it is scriptable
        scripted = torch.jit.script(m)
        scripted_keys = scripted.state_dict().keys()
        scripted.mods1_0_packed_weight_0 = m.state_dict()["mods1_0_packed_weight_0"]
        non_packed_weight_keys = [key for key in keys if "_packed_weight" not in key]
        self.assertTrue(
            set(scripted_keys) == set(non_packed_weight_keys),
            "Expected the scripted model to preserve the state_dict for non-packed weight attributes")
        for attr_name in [
                "mods1_0_input_scale_0", "mods1_0_input_zero_point_0",
                "mods1_0_scale_0", "mods1_0_zero_point_0",
                "mods1_1_scale_0", "mods1_1_zero_point_0",
                "mods2_scale_0", "mods2_zero_point_0"]:
            self.assertTrue(hasattr(m, attr_name))

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
        qconfig_dict = {"": default_qconfig}
        m = prepare_fx(model, qconfig_dict)
        m(torch.rand(5, 5))
        m = convert_fx(m)
        assert hasattr(m, "mods1_0_packed_weight_0")
        assert hasattr(m, "mods1_1_packed_weight_0")
        assert hasattr(m, "mods2_packed_weight_0")

    def test_mul_add_fp16_config(self):
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
        m = prepare_fx(model, qconfig_dict)
        m = convert_fx(m)
        # make sure it runs
        m(torch.randn(5, 5))

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
            m(torch.rand(4, 4, 4, 4))
            qconfig_dict = {'': torch.quantization.default_qconfig}
            mp = prepare_fx(m, qconfig_dict)
            mp(torch.rand(4, 4, 4, 4))
            mc = convert_fx(mp)

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
        m(torch.rand(4, 1, 4, 4))
        qconfig_dict = {'': torch.quantization.default_qconfig}
        mp = prepare_fx(m, qconfig_dict)
        mp(torch.rand(4, 1, 4, 4))
        mc = convert_fx(mp)
        mc(torch.rand(4, 1, 4, 4))

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
            m(torch.rand(4, 1, 4, 4))
            qconfig_dict = {'': torch.quantization.default_qconfig}
            mp = prepare_fx(m, qconfig_dict)
            mp(torch.rand(4, 1, 4, 4))
            mc = convert_fx(mp)
            mc(torch.rand(4, 1, 4, 4))

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
            '': torch.quantization.default_qconfig,
            'module_name': [
                ('child', None),
            ],
        }
        mp = prepare_fx(m, qconfig_dict)
        mp(torch.rand(1, 1, 1, 1))
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
        m = prepare_fx(m, qconfig_dict)
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
        m = prepare_fx(m, qconfig_dict)
        m = convert_fx(m)
        state_dict = m.state_dict()
        self.assertTrue("_packed_weight_0" in state_dict)

        # test load
        ref_weight, ref_bias = torch.ops.quantized.conv2d_unpack(state_dict["_packed_weight_0"])
        data = torch.rand(1, 3, 5, 5)
        ref_res = m(data)
        m = M2().eval()
        m = prepare_fx(m, qconfig_dict)
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
        m = prepare_fx(m, qconfig_dict)
        m = convert_fx(m)
        m.load_state_dict(state_dict)
        with TemporaryFileName() as fname:
            torch.save(m.state_dict(), fname)
            m.load_state_dict(torch.load(fname))

        checkModel(m, data, ref_weight, ref_bias, ref_res)

    def test_preserve_qconfig(self):
        """
        Test to make sure the temporary config option to preserve qconfig attributes
        in the model works
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
        m = prepare_fx(model, qconfig_dict)
        m(torch.rand(5, 5))
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
        qconfig_dict = {"": float16_static_qconfig}
        # make sure quantization runs
        m = prepare_fx(m, qconfig_dict)
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
        m = prepare_fx(model, qconfig_dict)
        m(torch.rand(5, 5))
        m = convert_fx(m)
        keys = m.state_dict().keys()
        m(torch.randn(5, 5))
        for attr_name in [
                "mods1_0_input_scale_0", "mods1_0_input_zero_point_0",
                "mods1_0_scale_0", "mods1_0_zero_point_0",
                "mods1_1_scale_0", "mods1_1_zero_point_0"]:
            self.assertTrue(hasattr(m, attr_name))

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

        qconfig_dict = {'': torch.quantization.default_qconfig}
        mp = prepare_fx(m, qconfig_dict)
        # if an observer is inserted after _user_func_with_complex_return_type,
        # the following call will fail
        mp(torch.randn(4, 4, 4, 4))
        mc = convert_fx(mp)
        mc(torch.randn(4, 4, 4, 4))

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
        m = prepare_fx(model, qconfig_dict)
        m(torch.rand(5, 5))
        m = convert_fx(m)
        keys = m.state_dict().keys()
        m(torch.randn(5, 5))
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
        qconfig_dict = {'': torch.quantization.get_default_qat_qconfig('fbgemm')}
        prepare_custom_config_dict = {'output_quantized_idxs': [0]}
        data = (torch.randn(4, 1, 4, 4),)

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
            m1, data, QuantType.QAT,
            prepare_expected_node_occurrence={
                ns.call_module(torch.quantization.FakeQuantize): 1,
            },
            expected_node_occurrence={
                ns.call_function(torch.quantize_per_tensor): 1,
            },
            prepare_custom_config_dict=prepare_custom_config_dict,
            print_debug_info=True)

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
            m2, data, QuantType.QAT,
            prepare_expected_node_occurrence={
                # one for weights, one for activations
                ns.call_module(torch.quantization.FakeQuantize): 2,
            },
            expected_node_occurrence={
                ns.call_function(torch.quantize_per_tensor): 1,
            },
            prepare_custom_config_dict=prepare_custom_config_dict,
            print_debug_info=True)

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
            m3, data, QuantType.QAT,
            prepare_expected_node_occurrence={
                # one for weights, one for activations
                ns.call_module(torch.quantization.FakeQuantize): 2,
            },
            expected_node_occurrence={
                ns.call_function(torch.quantize_per_tensor): 1,
            },
            prepare_custom_config_dict=prepare_custom_config_dict)

    def test_deepcopy_preserve_attributes(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr = 3

            def forward(self, x):
                return x

        m = M().eval()
        m = prepare_fx(m, {"": default_qconfig}, prepare_custom_config_dict={"preserved_attributes": ["attr"]})
        self.assertTrue(hasattr(m, "attr"))
        m2 = copy.deepcopy(m)
        self.assertTrue(hasattr(m2, "attr"))
        m = convert_fx(m, convert_custom_config_dict={"preserved_attributes": ["attr"]})
        self.assertTrue(hasattr(m, "attr"))
        m2 = copy.deepcopy(m)
        self.assertTrue(hasattr(m2, "attr"))

@skipIfNoFBGEMM
class TestQuantizeFxOps(QuantizationTestCase):
    """Unit tests for individual ops
    """
    @skipIfNoFBGEMM
    def test_linear_module(self):
        class ModuleLinear(torch.nn.Module):
            def __init__(self, has_relu=False, f_relu=False):
                super(ModuleLinear, self).__init__()
                self.linear = torch.nn.Linear(30, 4).float()
                if has_relu:
                    if f_relu:
                        self.relu = F.relu
                    else:
                        self.relu = torch.nn.ReLU()
                else:
                    self.relu = torch.nn.Identity()

            def forward(self, x):
                return self.relu(self.linear(x))

        data = (torch.rand((1, 30), dtype=torch.float),)
        options = itertools.product(
            [ModuleLinear(has_relu=False)],
            self.all_quant_types)
        quantized_nodes = {
            # quant_type:
            QuantType.DYNAMIC: ns.call_module(nnqd.Linear),
            QuantType.STATIC: ns.call_module(nnq.Linear),
            # note that we are checking the final result
            QuantType.QAT: ns.call_module(nnq.Linear),
        }
        for model, quant_type in options:
            self.checkGraphModeFxOp(
                model, data, quant_type, quantized_nodes[quant_type])

        for f_relu, quant_type in itertools.product([True, False], [QuantType.STATIC, QuantType.QAT]):
            for model, quantized_node in [
                    (ModuleLinear(has_relu=True, f_relu=f_relu), ns.call_module(nniq.LinearReLU))]:
                self.checkGraphModeFxOp(model, data, quant_type, quantized_node)

    @skipIfNoFBGEMM
    def test_functional_linear(self):
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
        quant_type_to_prepare_expected_node_occurrence = {
            QuantType.DYNAMIC: {},
            # There should be 3 observers: after input, weight and activation.
            QuantType.STATIC: {
                ns.call_module(torch.quantization.HistogramObserver): 2,
                ns.call_module(torch.quantization.PerChannelMinMaxObserver): 1,
            },
            # There should be 3 observers: after input, weight and activation.
            QuantType.QAT: {
                ns.call_module(torch.quantization.FakeQuantize): 3,
            },
        }
        quant_type_to_qlinear_fun = {
            QuantType.DYNAMIC: ns.call_function(torch.ops.quantized.linear_dynamic),
            QuantType.STATIC: ns.call_function(torch.ops.quantized.linear),
            QuantType.QAT: ns.call_function(torch.ops.quantized.linear),
        }
        quant_type_to_qlinear_relu_fun = {
            # we don't have linear_relu_dynamic
            QuantType.DYNAMIC: ns.call_function(torch.ops.quantized.linear_dynamic),
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
            model = FuncLinear(use_bias, has_relu, f_relu)
            if has_relu:
                qlinear_fun = quant_type_to_qlinear_relu_fun[quant_type]
            else:
                qlinear_fun = quant_type_to_qlinear_fun[quant_type]

            convert_node_occurrence = {
                ns.call_function(torch.quantize_per_tensor): 1 if quant_type != QuantType.DYNAMIC else 0,
                qlinear_fun: 1,
                ns.call_method("dequantize"): 1 if quant_type != QuantType.DYNAMIC else 0
            }
            prepare_expected_node_occurrence = \
                quant_type_to_prepare_expected_node_occurrence[quant_type]
            self.checkGraphModeFxOp(
                model, data, quant_type, qlinear_fun,
                prepare_expected_node_occurrence=prepare_expected_node_occurrence,
                expected_node_occurrence=convert_node_occurrence)

    def test_linear_dynamic_fp16(self):
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
                qlinear_fun = ns.call_function(torch.ops.quantized.linear_dynamic_fp16)
            prepare_node_occurrence = {
                # weight
                ns.call_module(torch.quantization.PlaceholderObserver): 1
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
        for use_bias, has_relu, f_relu, is_reference in options:
            model = FuncLinear(use_bias, has_relu, f_relu)
            linear_fun = ns.call_function(torch.nn.functional.linear)
            prepare_node_occurrence = {
                # activation, weight, bias and output
                ns.call_module(torch.quantization.PlaceholderObserver): 3 + int(use_bias)
            }
            convert_node_occurrence = {
                # we don't support static fp16 ops, so the linear functino
                # is unfused
                linear_fun: 1,
                # activation, weight, bias and output
                ns.call_method("to"): 3 + int(use_bias),
                # TODO: because CopyNode is not handled properly currently, there is
                # a dequantize that is missing, will need to fix and
                # remove (- int(not has relu))
                ns.call_method("dequantize"): 3 + int(use_bias) - int(not has_relu)
            }
            self.checkGraphModeFxOp(
                model, data, QuantType.DYNAMIC, linear_fun,
                is_reference=is_reference,
                custom_qconfig_dict={"": float16_static_qconfig},
                prepare_expected_node_occurrence=prepare_node_occurrence,
                expected_node_occurrence=convert_node_occurrence)

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
            model = self.checkGraphModeFxOp(
                ConvWrapper(dim), self.img_data_dict[dim], quant_type,
                quantized_nodes[dim])

    @skipIfNoFBGEMM
    def test_functional_conv(self):
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

        quant_type_to_prepare_expected_node_occurrence = {
            QuantType.DYNAMIC: {},
            # There should be 3 observers: after input, weight and activation.
            QuantType.STATIC: {
                ns.call_module(torch.quantization.HistogramObserver): 2,
                ns.call_module(torch.quantization.PerChannelMinMaxObserver): 1,
            },
            # There should be 3 observers: after input, weight and activation.
            QuantType.QAT: {
                ns.call_module(torch.quantization.FakeQuantize): 3,
            },
        }
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
            self.checkGraphModeFxOp(
                model, data, quant_type, qconv_fun,
                prepare_expected_node_occurrence=prepare_expected_node_occurrence,
                expected_node_occurrence=convert_node_occurrence)


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
        quantized_node = ns.call_function(quantized_op)
        options = itertools.product([True, False], [True, False])
        quant_type = QuantType.STATIC
        # testing for default int8 static quant
        for is_inplace, is_scalar in options:
            self.checkGraphModeFxOp(
                BinaryOp(binary_op, ibinary_op, is_inplace, is_scalar), data, quant_type, quantized_node)
            # This tests the binary op should be quantized even when it is not feed with a
            # quantized input
            self.checkGraphModeFxOp(
                BinaryOpNonQuantizedInput(binary_op, ibinary_op, is_inplace, is_scalar), data, quant_type, quantized_node)


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
        for is_inplace, is_scalar in options:
            node_occurrence = {
                # output_conv1, output_add1, output_add2 for scalar
                # output_conv1, output_conv2, output_add1, output_add2 for non-scalar
                ns.call_method("to"): 3 if is_scalar else 4
            }
            self.checkGraphModeFxOp(
                BinaryOp(binary_op, ibinary_op, is_inplace, is_scalar), data, quant_type,
                expected_node_occurrence=node_occurrence,
                custom_qconfig_dict=custom_qconfig_dict)

            node_occurrence = {
                # input_add, output_add for scalar
                # input_add1, input_add2, output_add for non-scalar
                ns.call_method("to"): 2 if is_scalar else 3
            }
            self.checkGraphModeFxOp(
                BinaryOpNonQuantizedInput(binary_op, ibinary_op, is_inplace, is_scalar), data, quant_type,
                expected_node_occurrence=node_occurrence,
                custom_qconfig_dict=custom_qconfig_dict)

    def _test_binary_op_relu_int8_impl(self, binary_op, ibinary_op, quantized_op):
        data = (torch.rand((1, 1, 1, 1), dtype=torch.float),
                torch.rand((1, 1, 1, 1), dtype=torch.float))
        quant_type = QuantType.STATIC
        quantized_node = ns.call_function(quantized_op)
        options = itertools.product(
            [True, False], [True, False], [True, False])
        for is_inplace_op, is_functional_relu, is_scalar in options:
            self.checkGraphModeFxOp(
                BinaryOpRelu(binary_op, ibinary_op, is_inplace_op, is_functional_relu, is_scalar),
                data, quant_type, quantized_node)

    def _test_binary_op_relu_float16_impl(self, binary_op, ibinary_op):
        data = (torch.rand((1, 1, 1, 1), dtype=torch.float),
                torch.rand((1, 1, 1, 1), dtype=torch.float))
        quant_type = QuantType.STATIC
        options = itertools.product(
            [True, False], [True, False], [True, False])
        custom_qconfig_dict = {
            "": float16_static_qconfig,
            "object_type": [(torch.nn.Conv2d, None)]
        }
        for is_inplace_op, is_functional_relu, is_scalar in options:
            node_occurrence = {
                ns.call_method("to"): 3 if is_scalar else 4
            }
            self.checkGraphModeFxOp(
                BinaryOpRelu(binary_op, ibinary_op, is_inplace_op, is_functional_relu, is_scalar),
                data, quant_type, custom_qconfig_dict=custom_qconfig_dict,
                expected_node_occurrence=node_occurrence)


    @skipIfNoFBGEMM
    def test_add(self):
        self._test_binary_op_int8_impl(
            operator.add, operator.iadd, torch.ops.quantized.add)
        self._test_binary_op_float16_impl(
            operator.add, operator.iadd)

    def test_sub(self):
        self._test_binary_op_float16_impl(operator.sub, operator.isub)
        self._test_binary_op_float16_impl(torch.sub, None)

    def test_div(self):
        self._test_binary_op_float16_impl(operator.truediv, operator.itruediv)
        self._test_binary_op_float16_impl(torch.div, None)

    @skipIfNoFBGEMM
    def test_mul(self):
        self._test_binary_op_int8_impl(
            operator.mul, operator.imul, torch.ops.quantized.mul)
        self._test_binary_op_float16_impl(operator.mul, operator.imul)

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
    def test_mul_relu(self):
        self._test_binary_op_relu_int8_impl(
            operator.mul, operator.imul, torch.ops.quantized.mul_relu)
        self._test_binary_op_relu_float16_impl(
            operator.mul, operator.imul)

    # TODO(future PR): make more generic
    def _test_quantized_add_mul_qat(self, model, expected_node_occurrence):
        qconfig_dict = {'': torch.quantization.get_default_qat_qconfig('fbgemm')}
        mp = torch.quantization.quantize_fx.prepare_qat_fx(model, qconfig_dict)
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
        expected_node_occurrence = {
            ns.call_module(torch.quantization.FakeQuantize): 4,
        }
        self._test_quantized_add_mul_qat(m, expected_node_occurrence)

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
        expected_node_occurrence = {
            ns.call_module(torch.quantization.FakeQuantize): 4,
        }
        self._test_quantized_add_mul_qat(m, expected_node_occurrence)

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
                self.add_func = torch.nn.quantized.FloatFunctional()

            def forward(self, x):
                return self.add_func.add_scalar(x, self.scalar)

        m = M(0.5)
        mp = torch.quantization.quantize_fx.prepare_qat_fx(
            m, {'': torch.quantization.get_default_qat_qconfig('fbgemm')},
            prepare_custom_config_dict={"input_quantized_idxs": [0]})
        expected_node_occurrence = {
            ns.call_module(torch.quantization.FakeQuantize): 0,
        }
        self.checkGraphModuleNodes(
            mp, expected_node_occurrence=expected_node_occurrence)

    @skipIfNoFBGEMM
    def test_cat(self):
        """ quantization of the output of cat will depend on the
        input of cat. we only quantize the output of cat when its inputs are quantized.
        """
        class QuantizedInput(torch.nn.Module):
            def __init__(self):
                super(QuantizedInput, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                return torch.cat([x, y], 1)

        class NonQuantizedInput(torch.nn.Module):
            def __init__(self):
                super(NonQuantizedInput, self).__init__()

            def forward(self, x, y):
                return torch.cat([x, y], 1)

        data = (torch.randn(1, 2, 5, 5, dtype=torch.float),
                torch.randn(1, 2, 5, 5, dtype=torch.float))
        quantized_node = ns.call_function(torch.cat)
        for quant_type in self.static_quant_types:
            self.checkGraphModeFxOp(QuantizedInput(), data, quant_type, quantized_node)
            self.checkGraphModeFxOp(NonQuantizedInput(), data, quant_type, quantized_node)

        # check cat is using the same observer for input and output
        m = QuantizedInput().eval()
        m = prepare_fx(m, {"": default_qconfig})
        # two inputs and one output of torch.cat are using same observer, so we have
        # 2 observers that's replicated
        all_observers = len(dict(m.named_modules(remove_duplicate=False)))
        distinct_observers = len(dict(m.named_modules()))
        self.assertEqual(all_observers, distinct_observers + 2)
        # make sure the converted model runs
        m = convert_fx(m)
        m(*data)

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

        options = itertools.product(self.static_quant_types, [2, 3])
        quantized_nodes = {
            # 1: ns.call_module(nnq.BatchNorm1d),
            2: ns.call_module(nnq.BatchNorm2d),
            3: ns.call_module(nnq.BatchNorm3d),
        }
        for quant_type, dim in options:
            model = self.checkGraphModeFxOp(
                M(dim), self.img_data_dict[dim], quant_type, quantized_nodes[dim])

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

        options = itertools.product(self.static_quant_types, [2, 3])
        quantized_nodes = {
            2: ns.call_module(nniq.BNReLU2d),
            3: ns.call_module(nniq.BNReLU3d),
        }
        for quant_type, dim in options:
            for instance in [BNRelu(dim, True), BNRelu(dim, False),
                             BNFuncRelu(dim), BNFuncInplaceRelu(dim)]:
                self.checkGraphModeFxOp(
                    instance, self.img_data_dict[dim], quant_type,
                    quantized_nodes[dim])

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

        options = itertools.product([True, False], [True, False], self.static_quant_types)
        quantized_nodes = {
            # is_module
            True: ns.call_module(quantized_module),
            False: ns.call_function(quantized_op),
        }

        for is_module, is_inplace, quant_type in options:
            self.checkGraphModeFxOp(
                M(is_module, is_inplace), self.img_data_2d,
                quant_type, quantized_nodes[is_module])

    def test_hardswish(self):
        self._test_activation_impl(nn.Hardswish, F.hardswish, nnq.Hardswish, torch.ops.quantized.hardswish)

    def test_elu(self):
        self._test_activation_impl(nn.ELU, F.elu, nnq.ELU, torch.ops.quantized.elu)

    def test_leaky_relu(self):
        self._test_activation_impl(nn.LeakyReLU, F.leaky_relu, nnq.LeakyReLU, torch.ops.quantized.leaky_relu)

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

        self._test_norm_float16_impl(
            nn.LayerNorm, F.layer_norm, [[2, 5, 5]], data)

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
        m_prep = torch.quantization.quantize_fx.prepare_fx(m, qconfig_dict, prepare_custom_qconfig_dict)
        m_prep(data)
        m_quant = torch.quantization.quantize_fx.convert_fx(m_prep, is_reference=is_reference)
        m_quant(data)

        self.checkGraphModuleNodes(m_quant, expected_node_list=node_list)

    def test_gelu_normal(self):
        module = torch.nn.GELU
        functional = torch.nn.functional.gelu
        qconfig = torch.quantization.get_default_qconfig("fbgemm")
        is_reference = False
        node_list = [
            ns.call_module(module),
            ns.call_function(functional),
        ]
        self._test_default_node_quant_handler_ops(
            module, functional, qconfig, is_reference, node_list)

    def test_softmax_normal(self):
        module = torch.nn.Softmax
        functional = torch.nn.functional.softmax
        qconfig = torch.quantization.get_default_qconfig("fbgemm")
        is_reference = False
        node_list = [
            ns.call_module(module),
            ns.call_function(functional),
        ]
        self._test_default_node_quant_handler_ops(
            module, functional, qconfig, is_reference, node_list)

    def test_gelu_reference(self):
        module = torch.nn.GELU
        functional = torch.nn.functional.gelu
        qconfig = torch.quantization.get_default_qconfig("fbgemm")
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
        additional_patterns = {torch.nn.GELU: DefaultNodeQuantizeHandler,
                               torch.nn.functional.gelu: DefaultNodeQuantizeHandler}
        self._test_default_node_quant_handler_ops(
            module, functional, qconfig, is_reference, node_list, additional_patterns)

    def test_softmax_reference(self):
        module = torch.nn.Softmax
        functional = torch.nn.functional.softmax
        qconfig = torch.quantization.get_default_qconfig("fbgemm")
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

    def test_bmm_int_reference(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bmm = torch.bmm

            def forward(self, x, y):
                out = self.bmm(x, y)
                return out

        data_x = torch.randn((2, 2, 2,))
        data_y = torch.randn((2, 2, 2,))
        qconfig_dict = {"": torch.quantization.get_default_qconfig("fbgemm")}
        is_reference = True
        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_method('dequantize'),
            ns.call_method('dequantize'),
            ns.call_function(torch.bmm),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_method('dequantize'),
        ]

        m = M().eval()
        m_prep = torch.quantization.quantize_fx.prepare_fx(m, qconfig_dict)
        m_prep(data_x, data_y)
        m_quant = torch.quantization.quantize_fx.convert_fx(m_prep, is_reference=is_reference)
        m_quant(data_x, data_y)

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
                F.hardtanh_(x)
                return x

        data = (torch.rand((1, 2, 5, 5), dtype=torch.float),)
        # list of node that should occur in order
        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Conv2d),
            ns.call_function(F.hardtanh_),
            ns.call_method('dequantize')
        ]
        for quant_type in self.static_quant_types:
            m = self.checkGraphModeFxOp(
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
        qconfig_dict = {
            "": float16_static_qconfig
        }
        node_occurrence = {
            ns.call_method("to"): 7
        }
        m = self.checkGraphModeFxOp(
            M(), data, quant_type, custom_qconfig_dict=qconfig_dict,
            expected_node_occurrence=node_occurrence)


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
                x = torch.max(x)
                x = torch.min(x)
                x = x.reshape([-1])
                x = x.resize_(1, 1, x.numel())
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
                x, y = torch.chunk(x, 2)
                x = F.dropout(x)
                x = self.dropout(x)
                x, _ = torch.sort(x)
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

        data = torch.rand(1, 3, 10, 10)
        # This model is not executable since we just put all ops
        # in the same forward
        m = M().eval()
        # nothing to fuse so skipping the fuse step
        qconfig_dict = {'': default_qconfig}
        prepared = prepare_fx(m, qconfig_dict)
        # not runnable
        quantized = convert_fx(prepared)

        # This checks that the dequantize from the output of first conv
        # is being propagated to the end, so that we don't insert extra
        # observers and also successfully fused two quantized::conv2d
        # patterns
        # one quantize_per_tensor for input
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
        prepared = prepare_fx(m, qconfig_dict)
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

    def test_getitem(self):
        """ Make sure we only insert observer for getitem if the following node is matched
        or needs to be quantized
        """
        class M(torch.nn.Module):
            def forward(self, xs):
                x = xs[0]
                return x

        m = M().eval()
        m = prepare_fx(m, {"": default_qconfig})
        self.checkGraphModuleNodes(m, expected_node_occurrence={
            ns.call_module(torch.quantization.MinMaxObserver): 0
        })
        m = convert_fx(m)
        m(torch.rand(1, 2))

        class M2(torch.nn.Module):
            def forward(self, xs):
                x = xs[0]
                x = torch.sigmoid(x)
                return x

        m2 = M2().eval()
        m2 = prepare_fx(m2, {"": default_qconfig})
        self.checkGraphModuleNodes(m2, expected_node_occurrence={
            ns.call_module(torch.quantization.MinMaxObserver): 1
        })
        m2 = convert_fx(m2)
        self.checkGraphModuleNodes(m2, expected_node_list=[
            ns.call_function(torch.quantize_per_tensor),
            ns.call_method("dequantize")
        ])
        m2([torch.rand(1, 2)])

        # testing prepare recognizes non-Tensor input for getitem
        class M3(torch.nn.Module):
            def forward(self, x):
                s = x.shape
                n, c = s[:2]
                x = torch.sigmoid(x)
                return x

        m3 = M3().eval()
        m3 = prepare_fx(m3, {"": default_qconfig})
        self.checkGraphModuleNodes(m3, expected_node_occurrence={
            ns.call_module(torch.quantization.MinMaxObserver): 1
        })
        m3 = convert_fx(m3)
        self.checkGraphModuleNodes(m3, expected_node_list=[
            ns.call_function(torch.quantize_per_tensor),
            ns.call_method("dequantize")
        ])
        m3(torch.rand(1, 2, 3, 4))


    @skipIfNoFBGEMM
    def test_fixed_qparams_ops(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.sigmoid = torch.nn.Sigmoid()
                self.hardsigmoid = torch.nn.Hardsigmoid()
                self.tanh = torch.nn.Tanh()

            def forward(self, x):
                x = self.conv(x)
                # F.sigmoid is deprecated
                x = self.sigmoid(x)
                x = torch.sigmoid(x)
                x = x.sigmoid()
                x.sigmoid_()
                x = self.hardsigmoid(x)
                x = F.hardsigmoid(x)
                x = F.hardsigmoid(x, inplace=True)
                x = x.hardsigmoid()
                x.hardsigmoid_()
                x = self.tanh(x)
                # F.tanh is deprecated
                x = torch.tanh(x)
                x = x.tanh()
                x.tanh_()
                x = self.conv(x)
                return x

        for eval_mode in [True, False]:
            # This model is not executable since we just put all ops
            # in the same forward
            m = M()
            if eval_mode:
                m.eval()
                qconfig = default_qconfig
                prepare = prepare_fx
                fq_count = 0
            else:
                m.train()
                qconfig = default_qat_qconfig
                prepare = prepare_qat_fx
                fq_count = 13

            # nothing to fuse so skipping the fuse step
            qconfig_dict = {'': qconfig}
            prepared = prepare(m, qconfig_dict)
            # check the correct number of activation_post_process is inserted
            count_check = {
                ns.call_module(FixedQParamsFakeQuantize) : fq_count,
            }
            self.checkGraphModuleNodes(
                prepared,
                expected_node_occurrence=count_check)
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
                ns.call_module(nn.Sigmoid),
                ns.call_module(nnq.Conv2d),
                ns.call_method('dequantize'),
            ]
            self.checkGraphModuleNodes(
                quantized,
                expected_node_occurrence=count_check,
                expected_node_list=order_check)

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

        data = torch.rand(3, 3)
        # Note: QAT test succeeded by chance, to make it actually work
        # we need to fix eager mode FloatFunctional by removing
        # activation_post_process in add_scalar and mul_scalar
        for quant_type in self.static_quant_types:
            m = M()
            ref_m = torch.quantization.QuantWrapper(M())
            is_qat = quant_type == QuantType.QAT
            if is_qat:
                m.train()
                ref_m.train()
                qconfig = default_qat_qconfig
                expected_act_post_process = torch.quantization.FakeQuantize
            else:
                m.eval()
                ref_m.eval()
                qconfig = default_qconfig
                expected_act_post_process = torch.quantization.MinMaxObserver

            prepare_fx_function = prepare_qat_fx if is_qat else prepare_fx
            qconfig_dict = {"": qconfig}
            m = prepare_fx_function(m, qconfig_dict)
            node_occurrence = {
                ns.call_module(expected_act_post_process): 5,
                ns.call_module(torch.nn.quantized.FloatFunctional): 0
            }
            self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)
            m(data)
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
            ref_m(data)
            ref_m = convert(ref_m)
            self.assertEqual(m(data), ref_m(data))

    def test_embedding(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=12)

            def forward(self, indices):
                return self.emb(indices)

        model = M().eval()
        indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])
        quantized_node = ns.call_module(nnq.Embedding)
        configs = [
            (float_qparams_weight_only_qconfig, ns.call_module(nnq.Embedding)),
            (None, ns.call_module(nn.Embedding)),
            (default_qconfig, ns.call_module(nn.Embedding)),
        ]

        for qconfig, node in configs:
            qconfig_dict = {"": qconfig}
            m = prepare_fx(model, qconfig_dict)
            self.checkGraphModuleNodes(m, expected_node_occurrence={
                ns.call_module(torch.quantization.MinMaxObserver): 0
            })
            m = convert_fx(m)
            self.checkGraphModuleNodes(m, expected_node=node)
            # make sure it runs
            m(indices)

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
        inputs = (indices, offsets)

        for dtype in [torch.quint8, torch.quint4x2]:
            model = M().eval()
            float_qparams_observer = PerChannelMinMaxObserver.with_args(dtype=dtype,
                                                                        qscheme=torch.per_channel_affine_float_qparams,
                                                                        ch_axis=0)
            float_qparams_qconfig = QConfigDynamic(activation=default_placeholder_observer,
                                                   weight=float_qparams_observer)
            self.checkGraphModeFxOp(
                model,
                inputs,
                QuantType.DYNAMIC,
                quantized_node,
                custom_qconfig_dict={"": float_qparams_qconfig}
            )

        # check it works in None and static qconfig
        for qconfig in [None, default_qconfig]:
            qconfig_dict = {"": default_qconfig}
            m = M().eval()
            m = prepare_fx(model, qconfig_dict)
            self.checkGraphModuleNodes(m, expected_node_occurrence={
                ns.call_module(torch.quantization.MinMaxObserver): 0
            })
            m = convert_fx(m)
            self.checkGraphModuleNodes(m, expected_node=ns.call_module(nn.EmbeddingBag))
            # make sure it runs
            m(*inputs)

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
            model_graph = prepare_fx(model_graph, graph_qconfig_dict)
            model_graph = convert_fx(model_graph)
            self.assertEqual(model_eager(sample_input), model_graph(sample_input))
            self.checkScriptable(model_graph, [[sample_input]], True)

    def test_rnn_cell(self):
        qconfigs = [per_channel_dynamic_qconfig, default_dynamic_qconfig, float16_dynamic_qconfig]
        module_type_strs = ['LSTMCell', 'GRUCell', 'RNNTanh', 'RNNReLU']
        module_types = [torch.nn.LSTMCell, torch.nn.GRUCell, torch.nn.RNNCell]
        sample_input = torch.tensor([[100, -155],
                                     [-155, 100],
                                     [100, -155]], dtype=torch.float)
        self._test_rnn_impl(qconfigs, RNNCellDynamicModel, module_type_strs, module_types, sample_input)

    def test_rnn(self):
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
            m2 = torch.quantization.QuantWrapper(m2)
            # FX graph
            q_result1 = self.checkGraphModeFxOp(
                m1, (data,), QuantType.STATIC,
                expected_node_occurrence={
                    ns.call_module(q_cls): 1,
                })
            # Eager
            m2.qconfig = get_default_qconfig(torch.backends.quantized.engine)
            m2.eval()
            m2p = torch.quantization.prepare(m2)
            m2p(data)
            m2q = torch.quantization.convert(m2p)
            q_result2 = m2q(data)
            # verify results match
            self.assertTrue(torch.allclose(q_result1, q_result2))

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
            # this has no effect on reshape since it's a CopyNode
            "": float16_static_qconfig,
            "object_type": [
                (torch.nn.functional.linear, default_qconfig)
            ]
        }
        m = prepare_fx(m, qconfig_dict)
        expected_occurrence = {
            # input and weight of first and second linear, output of first and second linear
            ns.call_module(torch.quantization.MinMaxObserver): 6,
        }
        self.checkGraphModuleNodes(
            m,
            expected_node_occurrence=expected_occurrence
        )
        # make sure it runs
        m = convert_fx(m)
        expected_occurrence = {
            ns.call_function(torch.quantize_per_tensor): 2,
            ns.call_method("dequantize"): 2,
            ns.call_method("to"): 1,
            ns.call_function(torch.ops.quantized.linear): 2
        }
        self.checkGraphModuleNodes(
            m,
            expected_node_occurrence=expected_occurrence
        )

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
        qconfig_dict = {
            "": float16_static_qconfig,
            "object_type": [
                (torch.nn.functional.linear, default_qconfig)
            ]
        }
        m = prepare_fx(m, qconfig_dict)
        expected_occurrence = {
            # input and weight of linear, output of linear
            ns.call_module(torch.quantization.MinMaxObserver): 3,
            # input and output of sigmoid
            ns.call_module(torch.quantization.PlaceholderObserver): 2,
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
        m = prepare_fx(m, {"": default_qconfig})
        expected_occurrence = {
            ns.call_module(torch.quantization.MinMaxObserver): 0
        }
        self.checkGraphModuleNodes(
            m,
            expected_node_occurrence=expected_occurrence)
        m = convert_fx(m)
        m(torch.rand(1, 2, 3, 4), torch.rand(3, 4).bool())
        return m


class TestQuantizeFxModels(QuantizationTestCase):
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
        else:
            assert quant_type == QuantType.QAT
            qconfig = default_qat_qconfig
            eager_prepare = prepare_qat
            graph_prepare = prepare_qat_fx
            eager.train()
            graph.train()
            calibrate_or_train = test_only_train_fn
            data = self.img_data_2d_train

        if hasattr(eager, "fuse_model"):
            eager.fuse_model()
        eager = QuantWrapper(eager)
        eager.qconfig = qconfig
        eager = eager_prepare(eager)

        qconfig_dict = {"": qconfig}
        graph = graph_prepare(graph, qconfig_dict)

        eager_out = eager(data[0][0])
        graph_out = graph(data[0][0])
        self.assertEqual(eager_out, graph_out)

        calibrate_or_train(eager, data)
        calibrate_or_train(graph, data)

        eager = convert(eager)
        graph = convert_fx(graph)

        eager_out = eager(data[0][0])
        graph_out = graph(data[0][0])
        self.assertEqual(eager_out, graph_out)

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

        no_pretrained_model = set(['shufflenet_v2_x0_5', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'])
        quantized_model_list = set(quantized_model_list) - no_pretrained_model
        # test eager and graph consistency
        model_list = quantized_model_list
        model_list = set(model_list)
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
                eager_quantizable_model = quantized_models.__dict__[name](pretrained=True, quantize=False, **kwargs).eval().float()
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
        eager_quantizable_model = quantized_models.__dict__[name](pretrained=True, quantize=False).eval().float()
        model = models.__dict__[name](pretrained=True).eval().float()
        self._test_model_impl(
            'ddp', 'resnet18', model, eager_quantizable_model)
