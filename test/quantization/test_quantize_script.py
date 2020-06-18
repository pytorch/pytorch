# -*- coding: utf-8 -*-
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
from torch._C import parse_ir

# torch.quantization
from torch.quantization import QConfig
from torch.quantization import default_dynamic_qconfig
from torch.quantization import per_channel_dynamic_qconfig
from torch.quantization import default_observer
from torch.quantization import default_per_channel_weight_observer
from torch.quantization import default_qconfig

# torch.quantization.quantize_script
from torch.quantization.quantize_script import script_qconfig
from torch.quantization.quantize_script import prepare_script
from torch.quantization.quantize_script import convert_script
from torch.quantization.quantize_script import quantize_script
from torch.quantization.quantize_script import prepare_dynamic_script
from torch.quantization.quantize_script import convert_dynamic_script
from torch.quantization.quantize_script import quantize_dynamic_script

# Testing utils
from torch.testing._internal.common_quantization import test_only_eval_fn as _test_only_eval_fn
from torch.testing._internal.common_quantized import override_qengines

from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    skipIfNoFBGEMM,
    get_script_module,
)

from torch.testing import FileCheck
from torch.testing._internal.jit_utils import attrs_with_prefix
from torch.testing._internal.jit_utils import get_forward
from torch.testing._internal.jit_utils import get_forward_graph

from torch.jit._recursive import wrap_cpp_module

# Standard library
import itertools
import unittest

class TestQuantizeScriptJitPasses(QuantizationTestCase):
    """ Test graph mode quantization passes used by quantize_script
    """
    def test_foldbn_trivial(self):
        # Test trivial case
        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.conv = torch.nn.Conv2d(1, 20, 5, 1)
                self.bn = torch.nn.BatchNorm2d(num_features=20)
                self.bn.eps = 0.0023

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        # Check that the transformation doesn't change numerics
        for tracing_mode in [True, False]:
            eager = TestModule()
            eager.eval()
            if tracing_mode:
                x = torch.rand(1, 1, 6, 6)
                scripted_or_traced = torch.jit.trace(eager, x)
            else:
                scripted_or_traced = torch.jit.script(eager)
            scripted_or_traced.eval()

            # Check that in the original script module's forward we have two
            # CallMethod nodes. One of them should be for conv.forward and the other
            # for bn.forward.
            FileCheck().check_count("prim::CallMethod[name=\"forward\"]", 2, exactly=True) \
                .run(str(get_forward(scripted_or_traced._c).graph))

            # Run FoldConvBatchnorm2d pass.
            scripted_or_traced = wrap_cpp_module(torch._C._jit_pass_fold_convbn(scripted_or_traced._c))

            # Check that after the pass one of the CallMethods is gone (supposedly,
            # the bn.forward).
            FileCheck().check_count("prim::CallMethod[name=\"forward\"]", 1, exactly=True) \
                .run(str(get_forward_graph(scripted_or_traced._c)))

            # Check that the transformation doesn't change numerics
            x = torch.rand(1, 1, 6, 6)
            self.assertEqual(eager(x), scripted_or_traced(x))

    def test_foldbn_trivial_nobias(self):
        # Test trivial case
        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.conv = torch.nn.Conv2d(1, 20, 5, 1, bias=False)
                self.bn = torch.nn.BatchNorm2d(num_features=20)
                # to make sure new bias is not zero
                self.bn.eps = 0.0027
                self.bn.bias = torch.nn.Parameter(torch.rand([20]))

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        for tracing_mode in [True, False]:
            eager = TestModule()
            eager.eval()
            if tracing_mode:
                x = torch.rand(1, 1, 6, 6)
                scripted_or_traced = torch.jit.trace(eager, x)
            else:
                scripted_or_traced = torch.jit.script(eager)
            scripted_or_traced.eval()

            # Check that in the original script module's forward we have two
            # CallMethod nodes. One of them should be for conv.forward and the other
            # for bn.forward.
            FileCheck().check_count("prim::CallMethod[name=\"forward\"]", 2, exactly=True) \
                .run(str(get_forward_graph(scripted_or_traced._c)))

            # Run FoldConvBatchnorm2d pass.
            scripted_or_traced = wrap_cpp_module(torch._C._jit_pass_fold_convbn(scripted_or_traced._c))

            # Check that after the pass one of the CallMethods is gone (supposedly,
            # the bn.forward).
            FileCheck().check_count("prim::CallMethod[name=\"forward\"]", 1, exactly=True) \
                .run(str(get_forward_graph(scripted_or_traced._c)))

            # Check that the transformation doesn't change numerics
            x = torch.rand(1, 1, 6, 6)
            self.assertEqual(eager(x), scripted_or_traced(x))

    def test_foldbn_in_submodule(self):
        # Test that we find Conv-BN patterns in submodules
        class SubModule(torch.nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.conv = torch.nn.Conv2d(1, 20, 5, 1)
                self.bn = torch.nn.BatchNorm2d(num_features=20)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.sub = SubModule()

            def forward(self, x):
                x = self.sub(x)
                return x

        for tracing_mode in [True, False]:
            eager = TestModule()
            eager.eval()
            if tracing_mode:
                x = torch.rand(1, 1, 10, 10)
                scripted_or_traced = torch.jit.trace(eager, x)
            else:
                scripted_or_traced = torch.jit.script(eager)
            scripted_or_traced.eval()

            FileCheck().check_count("prim::CallMethod[name=\"forward\"]", 2, exactly=True) \
                .run(str(get_forward_graph(scripted_or_traced.sub._c)))

            scripted_or_traced = wrap_cpp_module(torch._C._jit_pass_fold_convbn(scripted_or_traced._c))

            FileCheck().check_count("prim::CallMethod[name=\"forward\"]", 1, exactly=True) \
                .run(str(get_forward_graph(scripted_or_traced.sub._c)))

            x = torch.rand(1, 1, 10, 10)
            self.assertEqual(eager(x), scripted_or_traced(x))

    def test_foldbn_in_customConv2D(self):
        # Make sure a custom Conv2D class is not folded
        # as we do not know it does.
        class CustomConv2D(torch.nn.Module):
            def __init__(self, a, b, c, d):
                super(CustomConv2D, self).__init__()

            def forward(self, x):
                return F.relu(x)

        class SubModule(torch.nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.conv = CustomConv2D(1, 20, 5, 1)
                self.bn = torch.nn.BatchNorm2d(num_features=20)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.sub = SubModule()

            def forward(self, x):
                x = self.sub(x)
                return x

        for tracing_mode in [True, False]:
            eager = TestModule()
            eager.eval()
            if tracing_mode:
                x = torch.rand(1, 20, 10, 10)
                scripted_or_traced = torch.jit.trace(eager, x)
            else:
                scripted_or_traced = torch.jit.script(eager)
            scripted_or_traced.eval()

            FileCheck().check_count("prim::CallMethod[name=\"forward\"]", 2, exactly=True) \
                .run(str(get_forward_graph(scripted_or_traced.sub._c)))

            scripted_or_traced = wrap_cpp_module(torch._C._jit_pass_fold_convbn(scripted_or_traced._c))

            FileCheck().check_count("prim::CallMethod[name=\"forward\"]", 2, exactly=True) \
                .run(str(get_forward_graph(scripted_or_traced.sub._c)))

            x = torch.rand(1, 20, 10, 10)
            self.assertEqual(eager(x), scripted_or_traced(x))

    def test_foldbn_shared_classtype(self):
        class TestModule(torch.nn.Module):
            def __init__(self, bias=False):
                super(TestModule, self).__init__()
                self.conv1 = torch.nn.Conv2d(5, 5, 3, bias=bias)
                self.bn1 = torch.nn.BatchNorm2d(num_features=5)
                self.bn1.running_mean.fill_(-0.2)
                self.bn1.bias = torch.nn.Parameter(torch.rand([5]))
                # to make sure new bias is not zero
                self.bn1.eps = 0.0023
                self.conv2 = torch.nn.Conv2d(5, 5, 3, bias=bias)
                self.bn2 = torch.nn.BatchNorm2d(num_features=5)
                self.bn2.eps = 0.0029
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)
                return x

        for tracing_mode in [True, False]:
            for bias in [True, False]:
                eager = TestModule(bias).eval()
                if tracing_mode:
                    x = torch.rand(1, 5, 6, 6)
                    scripted_or_traced = torch.jit.trace(eager, x).copy()
                else:
                    scripted_or_traced = torch.jit.script(eager).copy()
                torch._C._jit_pass_dedup_module_uses(scripted_or_traced ._c)
                folded = wrap_cpp_module(torch._C._jit_pass_fold_convbn(scripted_or_traced ._c))
                x = torch.rand(1, 5, 6, 6)
                self.assertEqual(eager(x), scripted_or_traced(x))

    def test_foldbn_complex_cases(self):
        # This test case attempt to try combinations of conv2d with bias/nobias
        # as well as BatchNorm with affine/no-affine along with varying the
        # number of layers.
        # this only works when default dtype is double
        torch.set_default_dtype(torch.double)

        class SubModule(torch.nn.Module):
            def __init__(self, num_blocks, enable_bias, enable_affine):
                super(SubModule, self).__init__()
                layers = []
                for i in range(num_blocks):
                    layers.append(torch.nn.Conv2d(20, 20, 5, 1, bias=enable_bias))
                    bn_obj = torch.nn.BatchNorm2d(num_features=20, affine=enable_affine)
                    if enable_affine:
                        bn_obj.weight = torch.nn.Parameter(torch.rand_like(bn_obj.weight))
                        bn_obj.bias = torch.nn.Parameter(torch.rand_like(bn_obj.bias))
                    bn_obj.running_mean = torch.rand_like(bn_obj.running_mean)
                    bn_obj.running_var = torch.rand_like(bn_obj.running_var)
                    layers.append(bn_obj)
                self.layers = nn.Sequential(*layers)

            def forward(self, x):
                return self.layers(x)

        class TestModule(torch.nn.Module):
            def __init__(self, num_blocks, enable_bias, enable_affine):
                super(TestModule, self).__init__()
                self.sub = SubModule(num_blocks, enable_bias, enable_affine)

            def forward(self, x):
                x = self.sub(x)
                return x

        bias_affine_options = itertools.product([True, False], [True, False], [True, False], [1, 2])
        for (tracing_mode, enable_bias, enable_bn_affine, num_layers) in bias_affine_options:
            eager = TestModule(num_layers, enable_bias, enable_bn_affine)
            eager.eval()

            if tracing_mode:
                x = torch.rand(1, 20, 10, 10)
                scripted_or_traced = torch.jit.trace(eager, x)
            else:
                scripted_or_traced = torch.jit.script(eager)
            scripted_or_traced.eval()

            FileCheck().check_count("prim::CallMethod[name=\"forward\"]", num_layers * 2, exactly=True) \
                .run(str(get_forward_graph(scripted_or_traced.sub.layers._c)))

            scripted_or_traced = wrap_cpp_module(torch._C._jit_pass_fold_convbn(scripted_or_traced._c))

            FileCheck().check_count("prim::CallMethod[name=\"forward\"]", num_layers, exactly=True) \
                .run(str(get_forward_graph(scripted_or_traced.sub.layers._c)))

            x = torch.rand(1, 20, 10, 10)
            self.assertEqual(eager(x), scripted_or_traced(x))
        torch.set_default_dtype(torch.float)

    def test_fuse_linear(self):
        input_strs = ["""
graph(%input, %weight, %bias, %4):
    # CHECK-NOT: aten::t
    # CHECK-NOT: aten::addmm
    # CHECK: aten::linear
    %weight_t = aten::t(%weight)
    %res = aten::addmm(%bias, %input, %weight_t, %4, %4)
    return (%res)""", """
graph(%input, %weight, %bias, %4):
    # CHECK-NOT: aten::t
    # CHECK-NOT: aten::matmul
    # CHECK-NOT: aten::add_
    # CHECK: aten::linear
    %weight_t = aten::t(%weight)
    %output = aten::matmul(%input, %weight_t)
    %res = aten::add_(%output, %bias, %4)
    return (%res)""", """
graph(%input, %weight):
    # CHECK-NOT: aten::t
    # CHECK-NOT: aten::matmul
    # CHECK: aten::linear
    %weight_t = aten::t(%weight)
    %output = aten::matmul(%input, %weight_t)
    return (%output)"""]
        for input_str in input_strs:
            graph = parse_ir(input_str)
            torch._C._jit_pass_fuse_linear(graph)
            FileCheck().run(input_str, graph)

    def test_insert_observers(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)

            def forward(self, x):
                return self.conv(x)

        m = torch.jit.script(M())
        qconfig_dict = {'': default_qconfig}
        m = prepare_script(m, qconfig_dict)
        # for input and output of conv
        assert len(attrs_with_prefix(m, '_observer_')) == 2
        # for weight
        assert len(attrs_with_prefix(m.conv, '_observer_')) == 1

    def test_insert_observers_child_qconfig(self):
        class Sub(torch.nn.Module):
            def __init__(self):
                super(Sub, self).__init__()
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.fc(x)

        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)
                self.sub = Sub()

            def forward(self, x):
                return self.sub(self.conv(x))

        m = torch.jit.script(M())
        qconfig_dict = {'sub.fc': default_qconfig}
        m = prepare_script(m, qconfig_dict)
        # input and output of sub
        assert len(attrs_with_prefix(m, '_observer_')) == 2
        # not quantized
        assert len(attrs_with_prefix(m.conv, '_observer_')) == 0
        # no observers since we observe in the outer most call site
        assert len(attrs_with_prefix(m.sub, '_observer_')) == 0
        # weight of linear
        assert len(attrs_with_prefix(m.sub.fc, '_observer_')) == 1

    @unittest.skipUnless('fbgemm' in torch.backends.quantized.supported_engines,
                         " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
                         " with instruction set support avx2 or newer.")
    def test_insert_observers_skip_values(self):
        class ConvFunctionalReLU(torch.nn.Module):
            def __init__(self):
                super(ConvFunctionalReLU, self).__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)

            def forward(self, x):
                return F.relu(self.conv(x))

        class ConvReLUModule(torch.nn.Module):
            def __init__(self):
                super(ConvReLUModule, self).__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.conv(x))

        class AddReLUModule(torch.nn.Module):
            def __init__(self):
                super(AddReLUModule, self).__init__()
                self.relu = torch.nn.ReLU()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                out = self.conv(x)
                out += x
                return self.relu(out)

        class AddFunctionalReLU(torch.nn.Module):
            def __init__(self):
                super(AddFunctionalReLU, self).__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                out = self.conv(x)
                out += x
                return F.relu(out)

        def attrs_with_prefix(module, prefix):
            return [x for x, _ in module._modules._c.items()
                    if x.startswith(prefix)]

        qconfig_dict = {'': default_qconfig}
        m = torch.jit.script(ConvFunctionalReLU())
        m = prepare_script(m, qconfig_dict)
        # observer for weight of conv
        assert len(attrs_with_prefix(m.conv, '_observer_')) == 1
        # observer for input of conv and output of relu
        assert len(attrs_with_prefix(m, '_observer_')) == 2

        m = torch.jit.script(ConvReLUModule())
        m = prepare_script(m, qconfig_dict)
        # observer for input of conv and output of relu
        assert len(attrs_with_prefix(m, '_observer_')) == 2
        # observer for weight of conv
        assert len(attrs_with_prefix(m.conv, '_observer_')) == 1
        # observer for output of relu
        assert len(attrs_with_prefix(m.relu, '_observer_')) == 0

        m = torch.jit.script(AddReLUModule())
        qconfig_dict = {'': default_qconfig}
        m = prepare_script(m, qconfig_dict)
        assert len(attrs_with_prefix(m, '_observer')) == 3
        assert len(attrs_with_prefix(m.relu, '_observer')) == 0
        FileCheck().check('aten::add_') \
                   .check_not('Observer = prim::GetAttr[name="_observer_') \
                   .check('ReLU = prim::GetAttr') \
                   .run(str(get_forward_graph(m._c)))

        m = torch.jit.script(AddFunctionalReLU())
        qconfig_dict = {'': default_qconfig}
        m = prepare_script(m, qconfig_dict)
        assert len(attrs_with_prefix(m, '_observer')) == 3
        FileCheck().check('aten::add_') \
                   .check_not('Observer = prim::GetAttr[name="_observer_') \
                   .check('CallFunction') \
                   .check('Observer = prim::GetAttr[name="_observer_') \
                   .run(str(get_forward_graph(m._c)))

    def test_insert_observers_weight_dtype(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)

            def forward(self, x):
                return F.relu(self.conv(x))

        m = torch.jit.script(M())
        qconfig_dict = {'': default_qconfig}
        m = prepare_script(m, qconfig_dict)
        activation_dtypes = set(obs.getattr('dtype') for x, obs in m._modules._c.items()
                                if x.startswith('_observer_'))
        weight_dtypes = set(obs.getattr('dtype') for x, obs in m.conv._modules._c.items()
                            if x.startswith('_observer_'))
        assert len(activation_dtypes) == 1, 'Expected to have 1 activation dtype'
        assert len(weight_dtypes) == 1, 'Expected to have 1 weight dtype'
        assert list(activation_dtypes)[0] != list(weight_dtypes)[0], 'Expected activation dtype to '
        ' be different from wegiht dtype'

    def test_insert_observers_for_reused_weight(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()

            def forward(self, x, y, weight):
                x = F.conv2d(x, weight)
                y = F.conv2d(y, weight)
                return x + y

        m = torch.jit.script(M()).eval()
        m = prepare_script(m, {'': default_qconfig})
        # 3 for x, y, weight, one for output of each F.conv2d and one for output of add
        assert len(attrs_with_prefix(m, '_observer')) == 6

    def test_insert_observers_shared_class_type(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 5, 3).float()
                self.conv2 = torch.nn.Conv2d(3, 5, 3).float()

            def forward(self, x):
                return self.conv2(self.conv1(x))

        m = torch.jit.script(M())
        qconfig_dict = {'': default_qconfig}
        m = prepare_script(m, qconfig_dict)
        # conv1 and conv2 shares the same type, we need to
        # make sure we didn't quantize the type twice
        conv1_observers = attrs_with_prefix(m.conv1, '_observer_')
        conv2_observers = attrs_with_prefix(m.conv2, '_observer_')
        assert len(conv1_observers) == 1, \
            'Expected to have 1 observer submodules'
        assert len(conv2_observers) == 1, \
            'Expected to have 1 observer submodules'
        assert conv1_observers == conv2_observers, \
            'Expect conv1 and conv2 to have same observers since the class type is shared'

    def test_insert_observers_for_general_ops(self):
        """ Make sure we skip observers for ops that doesn't require
            observation, e.g. flatten
        """
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                x = self.conv(x)
                x = torch.flatten(x)
                return x

        m = torch.jit.script(M())
        qconfig_dict = {'': default_qconfig}
        m = prepare_script(m, qconfig_dict)
        # input and output of conv
        assert len(attrs_with_prefix(m, '_observer_')) == 2
        FileCheck().check('Observer = prim::GetAttr[name="_observer_') \
                   .check('prim::GetAttr[name="conv"]') \
                   .check('prim::CallMethod') \
                   .check('Observer = prim::GetAttr[name="_observer_') \
                   .check('aten::flatten') \
                   .check_not('Observer = prim::GetAttr[name="_observer_') \
                   .run(m.graph)

    # TODO: this is too long, split this to test_insert_observers.py and remove
    # insrt_observers prefix
    def test_insert_observers_propagate_observed(self):
        """ Make sure we propagate observed property through general ops
        """
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                x = self.conv1(x)
                x = torch.flatten(x)
                # we don't want to insert observer for input of self.conv2
                # because output of self.conv1 is already observed
                x = self.conv2(x)
                return x

        m = torch.jit.script(M())
        qconfig_dict = {'': default_qconfig}
        m = prepare_script(m, qconfig_dict)
        # input and output of conv
        assert len(attrs_with_prefix(m, '_observer_')) == 3
        FileCheck().check('Observer = prim::GetAttr[name="_observer_') \
                   .check('prim::GetAttr[name="conv1"]') \
                   .check('prim::CallMethod') \
                   .check('Observer = prim::GetAttr[name="_observer_') \
                   .check('aten::flatten') \
                   .check_not('Observer = prim::GetAttr[name="_observer_') \
                   .check('prim::GetAttr[name="conv2"]') \
                   .check('Observer = prim::GetAttr[name="_observer_') \
                   .run(m.graph)

    def test_insert_observers_propagate_observed_in_submodule(self):
        """ Make sure we propagate observed property through general ops
        """
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 3).float()
                self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                x = self.conv1(x)
                x = self.avgpool(x)
                # we don't want to insert observer for input of self.conv2
                # because output of self.conv1 is already observed
                x = self.conv2(x)
                return x

        m = torch.jit.script(M())
        qconfig_dict = {'': default_qconfig}
        m = prepare_script(m, qconfig_dict)
        # input and output of conv
        assert len(attrs_with_prefix(m, '_observer_')) == 3
        FileCheck().check('Observer = prim::GetAttr[name="_observer_') \
                   .check('prim::GetAttr[name="conv1"]') \
                   .check('prim::CallMethod') \
                   .check('Observer = prim::GetAttr[name="_observer_') \
                   .check('prim::CallMethod') \
                   .check_not('Observer = prim::GetAttr[name="_observer_') \
                   .check('prim::GetAttr[name="conv2"]') \
                   .check('Observer = prim::GetAttr[name="_observer_') \
                   .run(m.graph)

    def test_insert_observers_propagate_observed_for_function(self):
        def channel_shuffle(x, groups):
            # type: (torch.Tensor, int) -> torch.Tensor
            batchsize, num_channels, height, width = x.data.size()
            channels_per_group = num_channels // groups
            # reshape
            x = x.view(batchsize, groups,
                       channels_per_group, height, width)
            x = torch.transpose(x, 1, 2).contiguous()
            # flatten
            x = x.view(batchsize, -1, height, width)
            return x

        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 1).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 1).float()

            def forward(self, x):
                x = self.conv1(x)
                x = channel_shuffle(x, 1)
                x = self.conv2(x)
                return x

        data = [(torch.rand((1, 3, 10, 10), dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        m = torch.jit.script(M()).eval()
        m = prepare_script(m, {'': default_qconfig})
        # we want to test that channel_shuffle is going to pass
        # the observed property from the output of conv1 to input of conv2
        # so that we don't insert observers for input of conv2
        assert len(attrs_with_prefix(m, '_observer_',)) == 3

    def test_insert_observers_for_if(self):
        class QuantProp(torch.nn.Module):
            def __init__(self, use_skip):
                super(QuantProp, self).__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()
                self.use_skip = use_skip

            def forward(self, x):
                if self.use_skip:
                    x = self.conv(x)
                    return torch.reshape(x, x.shape)
                else:
                    x = self.conv(x)
                    return torch.reshape(x, x.shape)

        class Res(torch.nn.Module):
            def __init__(self, use_skip):
                super(Res, self).__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()
                self.use_skip = use_skip

            def forward(self, x):
                if self.use_skip:
                    return self.conv(x)
                else:
                    return self.conv(x)

        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.quant_prop = QuantProp(True)
                self.res = Res(False)

            def forward(self, x):
                x = self.quant_prop(x)
                x = self.res(x)
                return x

        data = [torch.rand(1, 3, 10, 10, dtype=torch.float)]
        result = {False : [1, 2, 2], True : [2, 1, 0]}
        for tracing in [True, False]:
            if tracing:
                m = torch.jit.trace(M(), data).eval()
            else:
                m = torch.jit.script(M()).eval()
            m = prepare_script(m, {'': default_qconfig})
            assert len(attrs_with_prefix(m, '_observer_',)) == result[tracing][0]
            assert len(attrs_with_prefix(m.quant_prop, '_observer_',)) == result[tracing][1]
            assert len(attrs_with_prefix(m.res, '_observer_',)) == result[tracing][2]

    def test_insert_observers_for_nested_if(self):
        class Res(torch.nn.Module):
            def __init__(self, use_skip):
                super(Res, self).__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()
                self.cond = use_skip
                self.use_skip = use_skip

            def forward(self, x):
                if self.use_skip:
                    if self.cond:
                        return self.conv(x)
                    else:
                        return self.conv(x)
                else:
                    return self.conv(x)

        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.res1 = Res(True)
                self.res2 = Res(False)

            def forward(self, x):
                x = self.res1(x)
                x = self.res2(x)
                return x

        data = torch.rand((1, 3, 10, 10), dtype=torch.float)
        result = {True : 3, False : 1}
        for tracing in [True, False]:
            if tracing:
                m = torch.jit.trace(M(), data).eval()
            else:
                m = torch.jit.script(M()).eval()
            m = prepare_script(m, {'': default_qconfig})
            assert len(attrs_with_prefix(m, '_observer_')) == result[tracing]

    def test_insert_observers_for_if_consistent_observation(self):
        """ check quantization for if works as long as
        output of all branches are quantized/observed consistently
        """
        class M(torch.nn.Module):
            def __init__(self, cond):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()
                self.cond = cond

            def forward(self, x):
                x = self.conv(x)
                # x is already observed
                if self.cond:
                    x = torch.flatten(x)
                return x

        class M2(torch.nn.Module):
            def __init__(self, cond):
                super(M2, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 3).float()
                self.cond = cond

            def forward(self, x):
                x = self.conv1(x)
                if self.cond:
                    x = self.conv2(x)
                    # x will be observed in the branch
                else:
                    x = torch.flatten(x)
                # since output for both branch are quantized
                # the if node is quantized consistently
                return x

        data = torch.rand((1, 3, 5, 5), dtype=torch.float)
        options = list(itertools.product([True, False], [True, False]))
        for cond, tracing in options:
            if tracing:
                m = torch.jit.trace(M(cond), data)
            else:
                m = torch.jit.script(M(cond))
            m = prepare_script(m, {'': default_qconfig})
            assert len(attrs_with_prefix(m, '_observer_')) == 2

        for cond, tracing in options:
            if tracing:
                m = torch.jit.trace(M2(cond), data)
            else:
                m = torch.jit.script(M2(cond))
            m = prepare_script(m, {'': default_qconfig})
            num_observers = 2 if tracing and not cond else 3
            assert len(attrs_with_prefix(m, '_observer_')) == num_observers

    def test_insert_quant_dequant(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3).float()

            def forward(self, x):
                return self.conv(x)

        for is_per_channel in [True, False]:
            m = torch.jit.script(M())
            observer = default_per_channel_weight_observer.with_args(ch_axis=1) \
                if is_per_channel else default_observer
            qconfig_dict = {'': QConfig(activation=observer, weight=observer)}
            m = prepare_script(m, qconfig_dict)
            data = torch.randn(1, 3, 10, 10, dtype=torch.float)

            m(data)
            m = convert_script(m, debug=True)
            assert len(m._modules._c.items()) == 1, \
                'Expected to have single submodule of conv'
            # make sure the quantized model is executable
            m(data)
            quant_func = "aten::quantize_per_channel" if is_per_channel \
                else "aten::quantize_per_tensor"
            FileCheck().check_count(quant_func, 3, exactly=True) \
                       .run(m.graph)

    def test_insert_quant_dequant_shared_class_type(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                return self.conv2(self.conv1(x))

        for is_per_channel in [True, False]:
            m = torch.jit.script(M())
            observer = default_per_channel_weight_observer.with_args(ch_axis=1) \
                if is_per_channel else default_observer
            qconfig = QConfig(activation=observer, weight=observer)
            qconfig_dict = {'': qconfig}
            m = prepare_script(m, qconfig_dict)
            # observers for input, output and value between conv1/conv2
            assert len(attrs_with_prefix(m, '_observer_')) == 3, \
                'Expected to have 3 obervers'
            # observer for weight
            assert len(attrs_with_prefix(m.conv1, '_observer_')) == 1, \
                'Expected to have 1 obervers'
            # observer for weight
            assert len(attrs_with_prefix(m.conv2, '_observer_')) == 1, \
                'Expected to have 1 obervers'

            data = torch.randn(1, 3, 10, 10, dtype=torch.float)
            m(data)
            m = convert_script(m, debug=True)
            m(data)
            assert m.conv1._c._type() == m.conv2._c._type()

            # check all observers have been removed
            assert len(attrs_with_prefix(m, '_observer_')) == 0, \
                'Expected to have 0 obervers'
            assert len(attrs_with_prefix(m.conv1, '_observer_')) == 0, \
                'Expected to have 0 obervers'
            assert len(attrs_with_prefix(m.conv2, '_observer_')) == 0, \
                'Expected to have 0 obervers'

            quant_func = "aten::quantize_per_channel" if is_per_channel \
                else "aten::quantize_per_tensor"
            for module in ['conv1', 'conv2']:
                conv = m._c.getattr(module)
                # quantize weight
                FileCheck().check(quant_func) \
                           .check_next("aten::dequantize") \
                           .check("prim::CallMethod[name=\"_conv_forward\"]") \
                           .check("return") \
                           .run(get_forward_graph(conv))
                # no quantize node in _conv_forward
                FileCheck().check_not(quant_func) \
                           .check("aten::conv2d") \
                           .check_not(quant_func) \
                           .check("return") \
                           .run(conv._get_method('_conv_forward').graph)

    def test_dedup_module_uses(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.relu(x)
                x -= 0.5
                return self.relu(x)

        data = torch.randn((2, 2))
        m = torch.jit.script(M())
        ref_res = m(data)
        assert len([x for x, _ in m._modules._c.items()
                    if x.startswith('relu')]) == 1, \
            "Expected to have 1 relu modules after dedup module uses"
        torch._C._jit_pass_dedup_module_uses(m._c)
        m = torch.jit._recursive.wrap_cpp_module(m._c)
        res = m(data)
        assert len([x for x, _ in m._modules._c.items()
                    if x.startswith('relu')]) == 2, \
            "Expected to have 2 relu modules after dedup module uses"
        self.assertEqual(res, ref_res)

    def test_replicate_dequantize(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()

            def forward(self, x):
                x = torch.dequantize(x)
                r = self.conv(x)
                r += x
                return r
        x = torch.randn([1, 3, 10, 10], dtype=torch.float)
        x = torch.quantize_per_tensor(x, 0.5, 1, torch.quint8)
        m = torch.jit.script(M())
        ref_res = m(x)
        FileCheck().check_count("aten::dequantize", 1, exactly=True) \
                   .run(m.graph)
        torch._C._jit_pass_replicate_dequantize(m.graph)
        FileCheck().check_count("aten::dequantize", 2, exactly=True) \
                   .run(m.graph)
        res = get_forward(m._c)(x)
        self.assertEqual(res, ref_res)

    def test_replicate_dequantize_in_block(self):
        class M(torch.nn.Module):
            def __init__(self, cond):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()

                self.cond = cond

            def forward(self, x):
                x = torch.dequantize(x)
                if self.cond:
                    x = self.conv(x)
                else:
                    x = x + 3
                return x

        x = torch.randn([1, 3, 10, 10], dtype=torch.float)
        x = torch.quantize_per_tensor(x, 0.5, 1, torch.quint8)
        m = torch.jit.script(M(True))
        ref_res = m(x)
        FileCheck().check_count("aten::dequantize", 1, exactly=True) \
                   .run(m.graph)
        torch._C._jit_pass_replicate_dequantize(m.graph)
        FileCheck().check_count("aten::dequantize", 2, exactly=True) \
                   .run(m.graph)
        # check dequantize is right before CallMethod of conv
        FileCheck().check("aten::dequantize") \
                   .check_next("CallMethod") \
                   .run(m.graph)
        # check dequantize is right before add
        FileCheck().check("aten::dequantize") \
                   .check("aten::dequantize") \
                   .check_next("aten::add") \
                   .run(m.graph)
        res = get_forward(m._c)(x)
        self.assertEqual(res, ref_res)

    def test_swap_functional_linear(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()

            def forward(self, x, weight, bias):
                x = torch.dequantize(x)
                weight = torch.dequantize(weight)
                x = F.linear(x, weight, bias)
                x = torch.quantize_per_tensor(x, scale=1.0, zero_point=0, dtype=torch.quint8)
                return x

        x = torch.rand((10, 5), dtype=torch.float)
        x = torch.quantize_per_tensor(x, scale=0.5, zero_point=1, dtype=torch.quint8)
        weight = torch.rand((5, 5), dtype=torch.float)
        weight = torch.quantize_per_tensor(weight, scale=0.5, zero_point=1, dtype=torch.qint8)
        bias = torch.rand((5), dtype=torch.float)
        m = torch.jit.script(M())
        ref_res = m(x, weight, bias)
        FileCheck().check("CallFunction") \
                   .run(m.graph)
        torch._C._jit_pass_swap_functional_linear(m.graph)
        FileCheck().check("aten::linear") \
                   .check_not("CallFunction") \
                   .run(m.graph)
        res = m(x, weight, bias)
        self.assertEqual(res, ref_res)

    def test_replicate_quantize_for_if(self):
        """ We want to move quantize nodes for output of prim::If
        inside the prim::If blocks so that we can match quantization
        patterns.
        """
        class Res(torch.nn.Module):
            def __init__(self):
                super(Res, self).__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()
                self.use_skip = True

            def forward(self, x, cond):
                # type: (Tensor, bool) -> Tensor
                # to avoid being frozen
                self.use_skip = cond
                if self.use_skip:
                    return self.conv(x)
                else:
                    return self.conv(x)

        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.res1 = Res()
                self.res2 = Res()

            def forward(self, x):
                x = self.res1(x, True)
                x = self.res2(x, False)
                return x

        data = [(torch.rand((1, 3, 10, 10), dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        qconfig_dict = {'': default_qconfig}
        m = torch.jit.script(M()).eval()
        m = quantize_script(m, qconfig_dict, _test_only_eval_fn, [data])
        # make sure patterns in both branches are fused
        FileCheck().check_count("quantized::conv2d(", 4, exactly=True) \
                   .run(m.graph)

    def test_finalize_for_linear(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.fc = torch.nn.Linear(5, 5).float()

            def forward(self, x):
                return self.fc(x)

        data = [(torch.rand((1, 5), dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        qconfig_dict = {'': default_qconfig}
        model = torch.jit.script(M()).eval()
        model = quantize_script(model, qconfig_dict, _test_only_eval_fn, [data])
        # make sure there is only one quantize_per_tensor for input
        # and linear_prepack is folded
        FileCheck().check_count("aten::quantize_per_tensor", 1, exactly=True) \
                   .check_not("quantized::linear_prepack") \
                   .check("quantized::linear") \
                   .run(model.graph)

    def test_finalize_debug(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()
                self.avgpool = torch.nn.AvgPool2d(3)

            def forward(self, x):
                x = self.conv(x)
                x = self.avgpool(x)
                return x

        data = [(torch.rand((1, 3, 10, 10), dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        qconfig_dict = {'': default_qconfig}
        model = torch.jit.script(M()).eval()
        model = quantize_script(model, qconfig_dict, _test_only_eval_fn, [data], debug=True)
        FileCheck().check_not("quantized::conv2d") \
                   .check("aten::conv2d") \
                   .check("aten::avg_pool2d") \
                   .check("aten::q_scale") \
                   .check_next("aten::q_zero_point") \
                   .check_next("prim::dtype") \
                   .check_next("aten::quantize_per_tensor") \
                   .check("aten::dequantize") \
                   .run(model.graph)

    def test_finalize_no_extra_dequantize(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                x = self.conv(x)
                return x.size(0) * x

        model = torch.jit.script(M()).eval()
        model = quantize_script(model, {'': default_qconfig}, _test_only_eval_fn, [self.img_data])
        FileCheck().check_not("aten::dequantize(") \
                   .run(model.graph)

    def test_module_list(self):
        class SimpleLinearLayer(torch.nn.Module):
            def __init__(self):
                super(SimpleLinearLayer, self).__init__()
                self.fc = torch.nn.Linear(5, 5).float()

            def forward(self, x):
                return self.fc(x)

        class ComplexModel(torch.nn.Module):
            def __init__(self):
                super(ComplexModel, self).__init__()
                self.layers = torch.nn.ModuleList([SimpleLinearLayer() for i in range(2)])

            def forward(self, x):
                # type: (torch.Tensor) -> List[torch.Tensor]
                states = []
                for layer in self.layers:
                    val = layer(x)
                    states.append(val)
                return states

        data = torch.rand((1, 5), dtype=torch.float)
        qconfig_dict = {'': default_qconfig}
        model = torch.jit.script(ComplexModel()).eval()
        model = prepare_script(model, qconfig_dict)
        assert len(attrs_with_prefix(model, '_observer')) == 3
        model(data)
        model = convert_script(model, debug=False)
        FileCheck().check("quantized::linear") \
                   .check("quantized::linear") \
                   .run(model.graph)

    def test_conv_trace(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv1d = torch.nn.Conv1d(3, 3, 3).float()
                self.conv2d = torch.nn.Conv2d(3, 3, 3).float()
                self.conv3d = torch.nn.Conv3d(3, 3, 3).float()

            def forward(self, x, y, z):
                a = self.conv1d(x)
                b = self.conv2d(y)
                c = self.conv3d(z)
                return (a, b, c)

        qconfig_dict = {'': default_qconfig}
        inputs = (torch.rand((1, 3, 10), dtype=torch.float),
                  torch.rand((1, 3, 10, 10), dtype=torch.float),
                  torch.rand((1, 3, 10, 10, 10), dtype=torch.float))
        model = torch.jit.trace(M(), inputs).eval()
        m = prepare_script(model, qconfig_dict)
        FileCheck().check('aten::conv1d') \
                   .check_not("aten::_convolution") \
                   .run(str(get_forward_graph(m.conv1d._c)))
        FileCheck().check('aten::conv2d') \
                   .check_not("aten::_convolution") \
                   .run(str(get_forward_graph(m.conv2d._c)))
        FileCheck().check('aten::conv3d') \
                   .check_not("aten::_convolution") \
                   .run(str(get_forward_graph(m.conv3d._c)))

    @unittest.skipUnless('fbgemm' in torch.backends.quantized.supported_engines,
                         " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
                         " with instruction set support avx2 or newer.")
    def test_replicate_dequant_same_value(self):
        class Mul(torch.nn.Module):
            def __init__(self):
                super(Mul, self).__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                x = self.conv(x)
                return x * x

        data = [(torch.rand((1, 3, 10, 10), dtype=torch.float),
                 torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]

        qconfig_dict = {'': default_qconfig}
        model = torch.jit.script(Mul()).eval()
        m = quantize_script(model, qconfig_dict, _test_only_eval_fn, [data])
        FileCheck().check("quantized::mul(") \
                   .check_not("aten::mul") \
                   .run(m.graph)

class TestQuantizeScriptPTSQOps(QuantizationTestCase):
    """ Test graph mode post training static quantization works
    for individual ops end to end.
    """
    @skipIfNoFBGEMM
    def test_quantized_conv(self):
        conv_module = {1 : torch.nn.Conv1d, 2 : torch.nn.Conv2d, 3 : torch.nn.Conv3d}

        class Conv(torch.nn.Module):
            def __init__(self, dim):
                super(Conv, self).__init__()
                self.conv = conv_module[dim](3, 3, 3).float()

            def forward(self, x):
                return self.conv(x)

        options = itertools.product([1, 2, 3], [True, False])
        for dim, tracing in options:
            model = self.checkGraphModeOp(
                Conv(dim), self.img_data_dict[dim],
                "quantized::conv{}d".format(dim), tracing)
            # make sure there is only one quantize_per_tensor for input
            # and conv2d_prepack is folded
            FileCheck().check_count("aten::quantize_per_tensor", 1, exactly=True) \
                       .run(model.graph)

            FileCheck().check_not("quantized::conv{}d_prepack".format(dim)) \
                       .run(model.graph)

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

        options = itertools.product([1, 2, 3], [True, False])
        for dim, tracing in options:
            for orig_m in [ConvNdRelu(dim, True),
                           ConvNdRelu(dim, False),
                           ConvNdFunctionalRelu(dim),
                           ConvNdInplaceFunctionalRelu(dim)]:
                conv_name = "conv{}d".format(dim)
                m = self.checkGraphModeOp(
                    orig_m, self.img_data_dict[dim], "quantized::conv{}d_relu(".format(dim), tracing=tracing)

                FileCheck().check_not("aten::conv{}d(".format(dim)) \
                           .check_not("aten::relu") \
                           .check_not("quantized::conv{}d(".format(dim)) \
                           .check_not("quantized::relu(") \
                           .run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_add_alpha(self):
        """ Test quant fusion for multiple aten::add using same
        constant alpha as the third argument
        """
        class QuantizedAdd(torch.nn.Module):
            def __init__(self):
                super(QuantizedAdd, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                z = x + y
                w = y + z
                return z + w

        data = [(torch.randn(1, 2, 5, 5, dtype=torch.float),
                 torch.randn(1, 2, 5, 5, dtype=torch.float),
                 torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        for tracing in [True, False]:
            m = self.checkGraphModeOp(QuantizedAdd(), data, "quantized::add", tracing)
            FileCheck().check_count("quantized::add", 3, exactly=True) \
                       .run(m.graph)
            FileCheck().check_not("aten::add") \
                       .check_not("aten::add_") \
                       .run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_add_relu_alpha(self):
        """ Test quant fusion for multiple aten::add using same
        constant alpha as the third argument in add_relu pattern
        """
        class AddRelu(torch.nn.Module):
            def __init__(self, inplace):
                super(AddRelu, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                x = x + y
                x = self.relu(x)
                x = x + y
                return self.relu(x)

        class InplaceAddRelu(torch.nn.Module):
            def __init__(self, inplace):
                super(InplaceAddRelu, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                x += y
                x = self.relu(x)
                x += y
                return self.relu(x)

        class AddFunctionalRelu(torch.nn.Module):
            def __init__(self):
                super(AddFunctionalRelu, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                x = x + y
                x = F.relu(x)
                x = x + y
                return F.relu(x)

        class InplaceAddFunctionalRelu(torch.nn.Module):
            def __init__(self):
                super(InplaceAddFunctionalRelu, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                x += y
                x = F.relu(x)
                x += y
                return F.relu(x)

        class AddInplaceFunctionalRelu(torch.nn.Module):
            def __init__(self):
                super(AddInplaceFunctionalRelu, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                x = x + y
                x = F.relu(x, True)
                x = x + y
                return F.relu(x, True)

        class InplaceAddInplaceFunctionalRelu(torch.nn.Module):
            def __init__(self):
                super(InplaceAddInplaceFunctionalRelu, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                x += y
                x = F.relu(x, True)
                x += y
                return F.relu(x, True)

        data = [(torch.rand((1, 2, 5, 5), dtype=torch.float),
                 torch.rand((1, 2, 5, 5), dtype=torch.float),
                 torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        for m_orig in [AddRelu(True), AddRelu(False),
                       InplaceAddRelu(True), InplaceAddRelu(False),
                       AddFunctionalRelu(), InplaceAddFunctionalRelu(),
                       AddInplaceFunctionalRelu(), InplaceAddInplaceFunctionalRelu()]:
            for tracing in [True, False]:
                m = self.checkGraphModeOp(m_orig, data, "quantized::add_relu(", tracing=tracing)
                FileCheck().check_count("quantized::add_relu(", 2, exactly=True) \
                           .run(m.graph)
                FileCheck().check_not("aten::add(") \
                           .check_not("aten::add_(") \
                           .check_not("aten::relu(") \
                           .check_not("aten::relu_(") \
                           .check_not("quantized::add(") \
                           .check_not("quantized::relu(") \
                           .run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_add(self):
        class QuantizedAdd(torch.nn.Module):
            def __init__(self):
                super(QuantizedAdd, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                return x + y

        class QuantizedInplaceAdd(torch.nn.Module):
            def __init__(self):
                super(QuantizedInplaceAdd, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                x += y
                return x

        class NonQuantizedAdd(torch.nn.Module):
            def __init__(self):
                super(NonQuantizedAdd, self).__init__()

            def forward(self, x, y):
                return x + y

        class NonQuantizedInplaceAdd(torch.nn.Module):
            def __init__(self):
                super(NonQuantizedInplaceAdd, self).__init__()

            def forward(self, x, y):
                x += y
                return x

        data = [(torch.randn(1, 2, 5, 5, dtype=torch.float),
                 torch.randn(1, 2, 5, 5, dtype=torch.float),
                 torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        for m, quantized in [(QuantizedAdd(), True),
                             (QuantizedInplaceAdd(), True),
                             (NonQuantizedAdd(), False),
                             (NonQuantizedInplaceAdd(), False)]:
            for tracing in [True, False]:
                op = "quantized::add" if quantized else "aten::add"
                m = self.checkGraphModeOp(m, data, op, tracing)
                # TODO: remove after refactor of checkGraphModeOp
                if quantized:
                    FileCheck().check_not("aten::add") \
                               .check_not("aten::add_") \
                               .run(m.graph)
                else:
                    FileCheck().check_not("quantized::add") \
                               .run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_add_scalar(self):
        class QuantizedAddScalar(torch.nn.Module):
            def __init__(self):
                super(QuantizedAddScalar, self).__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                x = self.conv(x)
                return x + 3

        class QuantizedInplaceAddScalar(torch.nn.Module):
            def __init__(self):
                super(QuantizedInplaceAddScalar, self).__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                x = self.conv(x)
                x += 3
                return x

        class NonQuantizedAddScalar(torch.nn.Module):
            def __init__(self):
                super(NonQuantizedAddScalar, self).__init__()

            def forward(self, x):
                return x + 3

        class NonQuantizedInplaceAddScalar(torch.nn.Module):
            def __init__(self):
                super(NonQuantizedInplaceAddScalar, self).__init__()

            def forward(self, x):
                x += 3
                return x

        data = [(torch.randn(1, 2, 5, 5, dtype=torch.float),
                 torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        for m, quantized in [(QuantizedAddScalar(), True),
                             (QuantizedInplaceAddScalar(), True),
                             (NonQuantizedAddScalar(), False),
                             (NonQuantizedInplaceAddScalar(), False)]:
            for tracing in [True, False]:
                op = "quantized::add_scalar" if quantized else "aten::add"
                # TODO: fix debug=True numerics
                m = self.checkGraphModeOp(m, data, op, tracing, check=False)
                # TODO: remove after refactor of checkGraphModeOp
                if quantized:
                    FileCheck().check_not("aten::add") \
                               .check_not("aten::add_") \
                               .run(m.graph)
                else:
                    FileCheck().check_not("quantized::add_scalar") \
                               .run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_add_relu(self):
        class AddRelu(torch.nn.Module):
            def __init__(self, inplace):
                super(AddRelu, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                x = x + y
                return self.relu(x)

        class InplaceAddRelu(torch.nn.Module):
            def __init__(self, inplace):
                super(InplaceAddRelu, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                x += y
                return self.relu(x)

        class AddFunctionalRelu(torch.nn.Module):
            def __init__(self):
                super(AddFunctionalRelu, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                x = x + y
                return F.relu(x)

        class InplaceAddFunctionalRelu(torch.nn.Module):
            def __init__(self):
                super(InplaceAddFunctionalRelu, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                x += y
                return F.relu(x)

        class AddInplaceFunctionalRelu(torch.nn.Module):
            def __init__(self):
                super(AddInplaceFunctionalRelu, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                x = x + y
                return F.relu(x, True)

        class InplaceAddInplaceFunctionalRelu(torch.nn.Module):
            def __init__(self):
                super(InplaceAddInplaceFunctionalRelu, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                x += y
                return F.relu(x, True)

        data = [(torch.rand((1, 2, 5, 5), dtype=torch.float),
                 torch.rand((1, 2, 5, 5), dtype=torch.float),
                 torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        for m in [AddRelu(True), AddRelu(False),
                  InplaceAddRelu(True), InplaceAddRelu(False),
                  AddFunctionalRelu(), InplaceAddFunctionalRelu(),
                  AddInplaceFunctionalRelu(), InplaceAddInplaceFunctionalRelu()]:
            for tracing in [True, False]:
                m = self.checkGraphModeOp(m, data, "quantized::add_relu(", tracing)
                FileCheck().check_not("aten::add(") \
                           .check_not("aten::add_(") \
                           .check_not("aten::relu(") \
                           .check_not("aten::relu_(") \
                           .check_not("quantized::add(") \
                           .check_not("quantized::relu(") \
                           .run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_add_scalar_relu(self):
        class AddScalarRelu(torch.nn.Module):
            def __init__(self, inplace):
                super(AddScalarRelu, self).__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x):
                x = self.conv(x)
                return self.relu(x + 3)

        class InplaceAddScalarRelu(torch.nn.Module):
            def __init__(self, inplace):
                super(InplaceAddScalarRelu, self).__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x):
                x = self.conv(x)
                x += 3
                return self.relu(x)

        class AddScalarFunctionalRelu(torch.nn.Module):
            def __init__(self):
                super(AddScalarFunctionalRelu, self).__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                x = self.conv(x)
                return F.relu(x + 3)

        class InplaceAddScalarFunctionalRelu(torch.nn.Module):
            def __init__(self):
                super(InplaceAddScalarFunctionalRelu, self).__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                x = self.conv(x)
                x += 3
                return F.relu(x)

        class AddScalarInplaceFunctionalRelu(torch.nn.Module):
            def __init__(self):
                super(AddScalarInplaceFunctionalRelu, self).__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                x = self.conv(x)
                return F.relu(x + 3, True)

        class InplaceAddScalarInplaceFunctionalRelu(torch.nn.Module):
            def __init__(self):
                super(InplaceAddScalarInplaceFunctionalRelu, self).__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                x = self.conv(x)
                x += 3
                return F.relu(x, True)

        data = [(torch.rand((1, 2, 5, 5), dtype=torch.float),
                 torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        for m in [AddScalarRelu(True), AddScalarRelu(False),
                  InplaceAddScalarRelu(True), InplaceAddScalarRelu(False),
                  AddScalarFunctionalRelu(),
                  InplaceAddScalarFunctionalRelu(),
                  AddScalarInplaceFunctionalRelu(),
                  InplaceAddScalarInplaceFunctionalRelu()]:
            for tracing in [True, False]:
                # quantized::add_scalar_relu or quantized::add_scalar_relu_out
                # TODO: split this after refactor of checkGraphModeOp
                # TODO: fix debug=True numerics
                m = self.checkGraphModeOp(m, data, "quantized::add_scalar_relu", tracing, check=False)
                FileCheck().check_not("aten::add(") \
                           .check_not("aten::add_(") \
                           .check_not("aten::relu(") \
                           .check_not("aten::relu_(") \
                           .check_not("quantized::add_scalar(") \
                           .check_not("quantized::relu(") \
                           .run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_cat(self):
        """ quantization of the output of cat will be depend on the
        input of cat. we only quantize the output of cat when its inputs are quantized.
        """
        class QuantizedCat(torch.nn.Module):
            def __init__(self):
                super(QuantizedCat, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                return torch.cat([x, y], 1)

        class NonQuantizedCat(torch.nn.Module):
            def __init__(self):
                super(NonQuantizedCat, self).__init__()

            def forward(self, x, y):
                return torch.cat([x, y], 1)

        data = [(torch.randn(1, 2, 5, 5, dtype=torch.float),
                 torch.randn(1, 2, 5, 5, dtype=torch.float),
                 torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        for tracing in [True, False]:
            m = self.checkGraphModeOp(QuantizedCat(), data, "quantized::cat", tracing)
            FileCheck().check_not("aten::cat") \
                       .run(m.graph)

            m = self.checkGraphModeOp(NonQuantizedCat(), data, "aten::cat", tracing)
            FileCheck().check_not("quantized::cat") \
                       .run(m.graph)

    @skipIfNoFBGEMM
    def test_qbatch_norm(self):
        bn_module = {2 : torch.nn.BatchNorm2d, 3 : torch.nn.BatchNorm3d}

        class M(torch.nn.Module):
            def __init__(self, dim):
                super(M, self).__init__()
                self.bn = bn_module[dim](3).to(torch.float)

            def forward(self, x):
                return self.bn(x)

        options = itertools.product([True, False], [2, 3])
        for tracing, dim in options:
            model = self.checkGraphModeOp(M(dim), self.img_data_dict[dim], "quantized::batch_norm", tracing)

            FileCheck().check_not("aten::batch_norm") \
                       .run(model.graph)

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

        options = itertools.product([True, False], [2, 3])
        for tracing, dim in options:
            for instance in [BNRelu(dim, True), BNRelu(dim, False),
                             BNFuncRelu(dim), BNFuncInplaceRelu(dim)]:
                model = self.checkGraphModeOp(instance, self.img_data_dict[dim],
                                              "quantized::batch_norm_relu", tracing)
                FileCheck().check_not("aten::batch_norm") \
                           .check_not("aten::relu") \
                           .check_not("aten::relu_") \
                           .run(model.graph)

    @skipIfNoFBGEMM
    def test_quantized_mul(self):
        class QuantizedMul(torch.nn.Module):
            def __init__(self):
                super(QuantizedMul, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                return x * y

        class QuantizedInplaceMul(torch.nn.Module):
            def __init__(self):
                super(QuantizedInplaceMul, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                x *= y
                return x

        class NonQuantizedMul(torch.nn.Module):
            def __init__(self):
                super(NonQuantizedMul, self).__init__()

            def forward(self, x, y):
                return x * y

        class NonQuantizedInplaceMul(torch.nn.Module):
            def __init__(self):
                super(NonQuantizedInplaceMul, self).__init__()

            def forward(self, x, y):
                x *= y
                return x

        data = [(torch.randn(1, 2, 10, 10, dtype=torch.float),
                 torch.randn(1, 2, 10, 10, dtype=torch.float),
                 torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        for m, quantized in [(QuantizedMul(), True),
                             (QuantizedInplaceMul(), True),
                             (NonQuantizedMul(), False),
                             (NonQuantizedInplaceMul(), False)]:
            for tracing in [True, False]:
                op = "quantized::mul" if quantized else "aten::mul"
                m = self.checkGraphModeOp(m, data, op, tracing)
                # TODO: remove after refactor of checkGraphModeOp
                if quantized:
                    FileCheck().check_not("aten::mul") \
                               .check_not("aten::mul_") \
                               .run(m.graph)
                else:
                    FileCheck().check_not("quantized::mul") \
                               .run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_mul_scalar(self):
        class QuantizedMulScalar(torch.nn.Module):
            def __init__(self):
                super(QuantizedMulScalar, self).__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                x = self.conv(x)
                return x * 3

        class QuantizedInplaceMulScalar(torch.nn.Module):
            def __init__(self):
                super(QuantizedInplaceMulScalar, self).__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                x = self.conv(x)
                x *= 3
                return x

        class NonQuantizedMulScalar(torch.nn.Module):
            def __init__(self):
                super(NonQuantizedMulScalar, self).__init__()

            def forward(self, x):
                return x * 3

        class NonQuantizedInplaceMulScalar(torch.nn.Module):
            def __init__(self):
                super(NonQuantizedInplaceMulScalar, self).__init__()

            def forward(self, x):
                x *= 3
                return x

        data = [(torch.randn(1, 2, 5, 5, dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        for m, quantized in [(QuantizedMulScalar(), True),
                             (QuantizedInplaceMulScalar(), True),
                             (NonQuantizedMulScalar(), False),
                             (NonQuantizedInplaceMulScalar(), False)]:
            for tracing in [True, False]:
                op = "quantized::mul_scalar" if quantized else "aten::mul"
                # TODO: fix debug=True numerics
                m = self.checkGraphModeOp(m, data, op, tracing, check=False)
                # TODO: remove after refactor of checkGraphModeOp
                if quantized:
                    FileCheck().check_not("aten::mul") \
                               .check_not("aten::mul_") \
                               .run(m.graph)
                else:
                    FileCheck().check_not("quantized::mul_scalar") \
                               .run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_mul_relu(self):
        class MulRelu(torch.nn.Module):
            def __init__(self, inplace):
                super(MulRelu, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                x = x * y
                return self.relu(x)

        class InplaceMulRelu(torch.nn.Module):
            def __init__(self, inplace):
                super(InplaceMulRelu, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                x *= y
                return self.relu(x)

        class MulFunctionalRelu(torch.nn.Module):
            def __init__(self):
                super(MulFunctionalRelu, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                x = x * y
                return F.relu(x)

        class InplaceMulFunctionalRelu(torch.nn.Module):
            def __init__(self):
                super(InplaceMulFunctionalRelu, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                x *= y
                return F.relu(x)

        class MulInplaceFunctionalRelu(torch.nn.Module):
            def __init__(self):
                super(MulInplaceFunctionalRelu, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                x = x * y
                return F.relu(x, True)

        class InplaceMulInplaceFunctionalRelu(torch.nn.Module):
            def __init__(self):
                super(InplaceMulInplaceFunctionalRelu, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                x *= y
                return F.relu(x, True)

        data = [(torch.rand((1, 2, 5, 5), dtype=torch.float),
                 torch.rand((1, 2, 5, 5), dtype=torch.float),
                 torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        for m in [MulRelu(True), MulRelu(False),
                  InplaceMulRelu(True), InplaceMulRelu(False),
                  MulFunctionalRelu(), InplaceMulFunctionalRelu(),
                  MulInplaceFunctionalRelu(), InplaceMulInplaceFunctionalRelu()]:
            for tracing in [True, False]:
                m = self.checkGraphModeOp(m, data, "quantized::mul_relu(", tracing)
                FileCheck().check_not("aten::mul(") \
                           .check_not("aten::mul_(") \
                           .check_not("aten::relu(") \
                           .check_not("aten::relu_(") \
                           .check_not("quantized::mul(") \
                           .check_not("quantized::relu(") \
                           .run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_mul_scalar_relu(self):
        class MulScalarRelu(torch.nn.Module):
            def __init__(self, inplace):
                super(MulScalarRelu, self).__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x):
                x = self.conv(x)
                return self.relu(x * 3)

        class InplaceMulScalarRelu(torch.nn.Module):
            def __init__(self, inplace):
                super(InplaceMulScalarRelu, self).__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x):
                x = self.conv(x)
                x *= 3
                return self.relu(x)

        class MulScalarFunctionalRelu(torch.nn.Module):
            def __init__(self):
                super(MulScalarFunctionalRelu, self).__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                x = self.conv(x)
                return F.relu(x * 3)

        class InplaceMulScalarFunctionalRelu(torch.nn.Module):
            def __init__(self):
                super(InplaceMulScalarFunctionalRelu, self).__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                x = self.conv(x)
                x *= 3
                return F.relu(x)

        class MulScalarInplaceFunctionalRelu(torch.nn.Module):
            def __init__(self):
                super(MulScalarInplaceFunctionalRelu, self).__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                x = self.conv(x)
                return F.relu(x * 3, True)

        class InplaceMulScalarInplaceFunctionalRelu(torch.nn.Module):
            def __init__(self):
                super(InplaceMulScalarInplaceFunctionalRelu, self).__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                x = self.conv(x)
                x *= 3
                return F.relu(x, True)

        data = [(torch.randn(1, 2, 5, 5, dtype=torch.float),
                 torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        for m in [MulScalarRelu(True), MulScalarRelu(False),
                  InplaceMulScalarRelu(True), InplaceMulScalarRelu(False),
                  MulScalarFunctionalRelu(),
                  InplaceMulScalarFunctionalRelu(),
                  MulScalarInplaceFunctionalRelu(),
                  InplaceMulScalarInplaceFunctionalRelu()]:
            for tracing in [True, False]:
                # quantized::mul_scalar_relu or quantized::mul_scalar_relu_out
                # TODO: fix debug=True numerics
                m = self.checkGraphModeOp(m, data, "quantized::mul_scalar_relu", tracing, check=False)
                FileCheck().check_not("aten::mul(") \
                           .check_not("aten::mul_(") \
                           .check_not("aten::relu(") \
                           .check_not("aten::relu_(") \
                           .check_not("quantized::mul_scalar(") \
                           .check_not("quantized::relu(") \
                           .run(m.graph)

    def test_hardswish(self):
        data = [(torch.rand((1, 2, 5, 5), dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        hardswish = torch.nn.Hardswish()
        for tracing in [True, False]:
            m = self.checkGraphModeOp(hardswish, data, "quantized::hardswish", tracing)
            FileCheck().check_not("aten::hardswish") \
                       .run(m.graph)

    def test_layer_norm(self):
        data = [(torch.rand((1, 2, 5, 5), dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        layer_norm = torch.nn.LayerNorm([2, 5, 5])
        for tracing in [True, False]:
            m = self.checkGraphModeOp(layer_norm, data, "quantized::layer_norm", tracing)
            FileCheck().check_not("aten::layer_norm") \
                       .run(m.graph)

    def test_group_norm(self):
        data = [(torch.rand((1, 4, 5, 5), dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        group_norm = torch.nn.GroupNorm(2, 4)
        for tracing in [True, False]:
            m = self.checkGraphModeOp(group_norm, data, "quantized::group_norm", tracing)
            FileCheck().check_not("aten::group_norm") \
                       .run(m.graph)

    def test_instance_norm(self):
        data_1d = [(torch.rand((1, 4, 5), dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        data_2d = [(torch.rand((1, 4, 5, 1), dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        data_3d = [(torch.rand((1, 4, 5, 1, 1), dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        data = {1 : data_1d, 2 : data_2d, 3 : data_3d}
        instance_norm_modules = {1 : torch.nn.InstanceNorm1d,
                                 2 : torch.nn.InstanceNorm2d,
                                 3 : torch.nn.InstanceNorm3d}

        options = itertools.product([1, 2, 3], [True, False])
        for dim, tracing in options:
            instance_norm = instance_norm_modules[dim](4)
            m = self.checkGraphModeOp(
                instance_norm, data[dim], "quantized::instance_norm", tracing)
            FileCheck().check_not("aten::instance_norm") \
                       .run(m.graph)

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

        data = [(torch.rand((1, 2, 5, 5), dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        options = itertools.product(["aten::clamp", "aten::hardtanh", "aten::hardtanh_"], [True, False])
        for op, tracing in options:
            m = self.checkGraphModeOp(M(), data, op, tracing)
            FileCheck().check_count("aten::quantize_per_tensor", 1, exactly=True) \
                       .run(m.graph)

            FileCheck().check_count("aten::dequantize", 1, exactly=True) \
                       .run(m.graph)

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
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
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
                x = torch.repeat_interleave(x, 3, 1)
                x = self.relu(x)
                x = F.relu(x)
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
                x, _ = y
                x = self.conv(x)
                return x

        data = torch.rand(1, 3, 10, 10)
        # This model is not executable since we just put all ops
        # in the same forward, therefore we only test scripting
        m = torch.jit.script(M())
        qconfig = script_qconfig(default_qconfig)
        # dummy data to suppress warning
        get_forward(qconfig.activation)(data)
        get_forward(qconfig.weight)(data)

        m = wrap_cpp_module(torch._C._jit_pass_insert_observers(
            m._c, 'forward', {'': qconfig}, inplace=False))
        m = convert_script(m)
        # This checks that the dequantize from the output of first conv
        # is being propagated to the end, so that we don't insert extra
        # observers and also successfully fused two quantized::conv2d
        # patterns
        # one quantize_per_tensor for input
        FileCheck().check_count("aten::quantize_per_tensor", 1, exactly=True) \
                   .check_count("quantized::conv2d", 2, exactly=True) \
                   .check("aten::dequantize") \
                   .run(m.graph)

    def test_general_value_ops(self):
        """ A test that checks correct patterns are produced for
        all supported general value ops like aten::avg_pool2d \
        without actually checking for execution of these ops
        """
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.avg_pool1d = torch.nn.AvgPool1d(3)
                self.avg_pool2d = torch.nn.AvgPool2d(3)
                self.avg_pool3d = torch.nn.AvgPool3d(3)
                self.adaptive_avg_pool1d = torch.nn.AdaptiveAvgPool1d((1))
                self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.adaptive_avg_pool3d = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
                self.leaky_relu = torch.nn.LeakyReLU()
                self.hardsigmoid = torch.nn.Hardsigmoid()
                self.sigmoid = torch.nn.Sigmoid()
                self.tanh = torch.nn.Tanh()

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
                # interpolate node will introduce 3 quantize_per_tensor ops
                x = F.interpolate(x, 4, mode='nearest')  # interpolate node
                x = F.upsample(x, (32, 32))  # interpolate node
                x = F.upsample_nearest(x, (32, 32))  # interpolate node
                x = F.interpolate(x, 4, mode='linear')  # common node
                x = F.upsample_bilinear(x, (32, 32))  # common node
                x = self.leaky_relu(x)
                x = F.leaky_relu(x)
                x.leaky_relu_()
                x = self.hardsigmoid(x)
                x = F.hardsigmoid(x)
                x.hardsigmoid_()
                x = self.sigmoid(x)
                x = torch.sigmoid(x)
                # F.sigmoid is deprecated
                x = x.sigmoid()
                x.sigmoid_()
                x = self.tanh(x)
                # F.tanh is deprecated
                x = torch.tanh(x)
                x = x.tanh()
                x.tanh_()
                x = self.conv(x)
                return x

        # This model is not executable since we just put all ops
        # in the same forward, therefore we only test scripting
        m = torch.jit.script(M())
        qconfig = script_qconfig(default_qconfig)
        # dummy data to suppress warning
        data = torch.rand(1, 3, 10, 10)
        get_forward(qconfig.activation)(data)
        get_forward(qconfig.weight)(data)

        m = wrap_cpp_module(torch._C._jit_pass_insert_observers(
            m._c, 'forward', {'': qconfig}, inplace=False))
        # Checking the model before fianlize contain unfused patterns
        # that numerically matches the model after quantize by checking
        # number of aten::quantize_per_tensor functions
        # conv has 3 quantize_per_tensor for activations and 1 for weight
        # and for N general value op between conv we should have

        # N + 1 quantize_per_tensor between these ops
        m1 = convert_script(m, debug=True)
        # NB: This Needs to be updated when we add more ops to test
        # mapping from number of quant for the op to the number of these ops
        # for example, for `3` in the key means for this type of op
        # we'll have 3 quantize_per_tensor
        num_op_by_num_quant = {1: 35, 2: 2, 3: 3}
        num_quantize_per_tensor = 1  # for output
        for num_quant, num_op in num_op_by_num_quant.items():
            num_quantize_per_tensor += num_op * num_quant
        FileCheck().check_count("aten::quantize_per_tensor(", num_quantize_per_tensor, exactly=True) \
                   .run(m1.graph)

        # This checks that the dequantize from the output of first conv
        # is being propagated to the end, so that we don't insert extra
        # observers and also successfully fused two quantized::conv2d
        # patterns
        # one quantize_per_tensor for input
        m2 = convert_script(m, debug=False)
        FileCheck().check_count("aten::quantize_per_tensor(", 1, exactly=True) \
                   .run(m2.graph)
        FileCheck().check_count("quantized::conv2d(", 2, exactly=True) \
                   .check("aten::dequantize(") \
                   .run(m2.graph)

class TestQuantizeDynamicScriptJitPasses(QuantizationTestCase):
    def test_prepare_dynamic(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.fc(x)

        m = torch.jit.script(M())
        m = prepare_dynamic_script(m, {'': default_dynamic_qconfig})
        # for input of FC for dynamic quant
        assert len(attrs_with_prefix(m, '_observer_')) == 1
        # for weight
        assert len(attrs_with_prefix(m.fc, '_observer_')) == 1
        FileCheck().check('DynamicQuantObserver = prim::GetAttr[name="_observer_') \
                   .check('prim::GetAttr[name="fc"]') \
                   .check('prim::CallMethod') \
                   .check_not('Observer = prim::GetAttr[name="_observer_') \
                   .run(m.graph)


    def test_prepare_dynamic_child_qconfig(self):
        class Sub(torch.nn.Module):
            def __init__(self):
                super(Sub, self).__init__()
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.fc(x)

        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)
                self.sub = Sub()

            def forward(self, x):
                return self.sub(self.conv(x))

        m = torch.jit.script(M())
        # only quantize child module.
        m = prepare_dynamic_script(m, {'sub.fc': default_dynamic_qconfig})

        # input of sub for dynamic quant
        assert len(attrs_with_prefix(m, '_observer_')) == 1
        # not quantized
        assert len(attrs_with_prefix(m.conv, '_observer_')) == 0
        # no observers since we observe in the outer most call site
        assert len(attrs_with_prefix(m.sub, '_observer_')) == 0
        # weight of linear
        assert len(attrs_with_prefix(m.sub.fc, '_observer_')) == 1
        FileCheck().check('prim::GetAttr[name="sub') \
                   .check('prim::CallMethod') \
                   .check('DynamicQuantObserver = prim::GetAttr[name="_observer_') \
                   .check('prim::CallMethod') \
                   .check_not('Observer = prim::GetAttr[name="_observer_') \
                   .run(m.graph)

    def test_insert_quant_dequant_linear_dynamic(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.fc1 = torch.nn.Linear(5, 5).float()
                self.fc2 = torch.nn.Linear(5, 5).float()

            def forward(self, x):
                x = self.fc1(x)
                return self.fc2(x)
        for is_per_channel in [True, False]:
            m = torch.jit.script(M())
            qconfig = per_channel_dynamic_qconfig if is_per_channel is True else default_dynamic_qconfig
            m = prepare_dynamic_script(m, {'': qconfig})
            m = convert_dynamic_script(m, debug=True)
            assert len(m._modules._c.items()) == 2, \
                'Expected to have two submodule of linear'

            wt_quant_func = "aten::quantize_per_channel" if is_per_channel \
                else "aten::quantize_per_tensor"
            act_quant_func = "aten::quantize_per_tensor"
            # quantizing activations
            FileCheck().check("aten::_choose_qparams_per_tensor") \
                       .check_next(act_quant_func) \
                       .check_next("aten::dequantize") \
                       .check("aten::_choose_qparams_per_tensor") \
                       .check_next(act_quant_func) \
                       .check_next("aten::dequantize") \
                       .check(wt_quant_func) \
                       .check_next("aten::dequantize") \
                       .check_not(wt_quant_func) \
                       .check("return") \
                       .run(m.graph)

    @override_qengines
    def test_dynamic_multi_op(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)

            def forward(self, x):
                x = x + 5
                return self.fc1(x)

        x = torch.randn(5, 5)
        for tracing in [True, False]:
            model = self.checkGraphModeOp(M(), x, "quantized::linear_dynamic", tracing=tracing, dynamic=True)
            # add op is not dynamically quantized.
            FileCheck().check("aten::add") \
                       .run(model.graph)

    @override_qengines
    def test_dynamic_quant_multi_uses(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.fc = torch.nn.Linear(5, 5).float()

            def forward(self, x):
                size1 = x.size()
                size2 = x.size()
                return self.fc(x), size1, size2

        x = torch.randn(5, 5)
        for tracing in [True, False]:
            model = self.checkGraphModeOp(M(), x, "quantized::linear_dynamic", tracing=tracing, dynamic=True)
            FileCheck().check_not("aten::_choose_qparams_per_tensor") \
                       .run(model.graph)

    @override_qengines
    def test_dynamic_shared_weights(self):
        class myMod(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.linear = nn.Linear(5, 5)
                self.linear.weight = weight

            def forward(self, x):
                return self.linear(x)

        class DynamicModel(torch.nn.Module):
            def __init__(self):
                super(DynamicModel, self).__init__()
                self.weight = torch.nn.Parameter(torch.ones(5, 5))
                self.mod1 = myMod(self.weight)

            def forward(self, x):
                y = self.mod1(x)
                z = torch.nn.functional.linear(y, self.weight)
                return z

        model = torch.jit.script(DynamicModel()).eval()
        data = torch.randn(5, 5, dtype=torch.float)
        quant_ops = ['mod1', '']
        counts = [1, 2]
        for op, count in zip(quant_ops, counts):
            qconfig_dict = {op: default_dynamic_qconfig}
            m1 = quantize_dynamic_script(model, qconfig_dict)
            out_graph = m1(data)

            FileCheck().check_count("quantized::linear_dynamic(", count, exactly=True) \
                       .check_not("aten::_choose_qparams_per_tensor") \
                       .run(m1.graph)

            # Explicitly call forward on model before convert
            m2 = prepare_dynamic_script(model, qconfig_dict)
            m2(data)
            m2 = convert_dynamic_script(m2, debug=False)
            out_ref = m2(data)
            self.assertEqual(out_graph, out_ref)

    @override_qengines
    def test_dynamic_with_if(self):
        class Res(torch.nn.Module):
            def __init__(self):
                super(Res, self).__init__()
                self.weight = torch.nn.Parameter(torch.ones(5, 5))

            def forward(self, x, cond):
                # type: (Tensor, bool) -> Tensor
                if cond:
                    return torch.nn.functional.linear(x, self.weight)
                else:
                    return torch.nn.functional.linear(x, self.weight)

        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.res1 = Res()
                self.res2 = Res()

            def forward(self, x):
                x = self.res1(x, True)
                x = self.res2(x, False)
                return x

        model = torch.jit.script(M()).eval()
        data = torch.randn(5, 5, dtype=torch.float)
        qconfig_dict = {'': default_dynamic_qconfig}
        for tracing in [True, False]:
            m1 = self.checkGraphModeOp(M(), data, "quantized::linear_dynamic", tracing=tracing, dynamic=True)
            FileCheck().check_count("quantized::linear_dynamic(", 2, exactly=True) \
                       .check_not("aten::_choose_qparams_per_tensor") \
                       .run(m1.graph)

        # Check to make sure weight observers run correctly
        ref_qparams = []
        qconfig = script_qconfig(default_dynamic_qconfig)
        wt_module = wrap_cpp_module(qconfig.weight)
        for wt in [model.res1.weight, model.res2.weight]:
            wt_module(wt)
            qparams = wt_module.calculate_qparams()
            ref_qparams.append((qparams[0].item(), qparams[1].item()))

        m2 = prepare_dynamic_script(model, qconfig_dict)
        m2 = convert_dynamic_script(m2, debug=True)
        graph_params = []
        for x, obs in m2._modules._c.items():
            if x == 'res1':
                graph_params.append((obs.getattr('6_scale_0'), obs.getattr('6_zero_point_0')))
            elif x == 'res2':
                graph_params.append((obs.getattr('10_scale_0'), obs.getattr('10_zero_point_0')))
        self.assertEqual(ref_qparams, graph_params)

    def test_dynamic_weight_observer(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.fc = torch.nn.Linear(5, 5).float()
                self.fc2 = torch.nn.Linear(5, 5).float()

            def forward(self, x):
                x = self.fc(x)
                return self.fc2(x)

        qconfig_dict = {'': default_dynamic_qconfig}
        eager_model = M().eval()
        x = torch.rand(5, 5)
        for tracing in [True, False]:
            model = get_script_module(eager_model, tracing, x)
            qconfig = script_qconfig(default_dynamic_qconfig)
            ref_qparams = []
            wt_module = wrap_cpp_module(qconfig.weight)
            for wt in [model.fc.weight, model.fc2.weight]:
                wt_module(wt)
                qparams = wt_module.calculate_qparams()
                ref_qparams.append((qparams[0].item(), qparams[1].item()))
            model = prepare_dynamic_script(model, qconfig_dict)
            model = convert_dynamic_script(model, debug=True)
            graph_params = []
            for x, obs in model._modules._c.items():
                if tracing:
                    graph_params.append((obs.getattr('4_scale_0'), obs.getattr('4_zero_point_0')))
                else:
                    graph_params.append((obs.getattr('3_scale_0'), obs.getattr('3_zero_point_0')))
            self.assertEqual(ref_qparams, graph_params)

class TestQuantizeScriptPTDQOps(QuantizationTestCase):
    """ Test graph mode post training dynamic quantization works
    for individual ops end to end.
    """
    @override_qengines
    def test_quantized_linear_dynamic(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.fc = torch.nn.Linear(5, 5).float()

            def forward(self, x):
                return self.fc(x)

        x = torch.rand(5, 5)
        for tracing in [True, False]:
            model = self.checkGraphModeOp(M(), x, "quantized::linear_dynamic", tracing=tracing, dynamic=True)
