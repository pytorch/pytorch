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
from torch.quantization import QConfigDynamic
from torch.quantization import default_observer
from torch.quantization import default_per_channel_weight_observer
from torch.quantization import default_qconfig
from torch.quantization import get_default_qconfig

# torch.quantization._quantize_script
from torch.quantization._quantize_script import script_qconfig
from torch.quantization._quantize_script import prepare_script
from torch.quantization._quantize_script import convert_script
from torch.quantization._quantize_script import quantize_script
from torch.quantization._quantize_script import prepare_dynamic_script
from torch.quantization._quantize_script import quantize_dynamic_script

# Testing utils
from torch.testing._internal.common_quantization import test_only_eval_fn as _test_only_eval_fn

from torch.testing import FileCheck
from torch.testing._internal.jit_utils import attrs_with_prefix
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing._internal.jit_utils import get_forward
from torch.testing._internal.jit_utils import get_forward_graph
from torch.testing._internal.jit_utils import get_module_method

from torch.jit._recursive import wrap_cpp_module

# Standard library
import itertools
import unittest

class TestQuantizeScriptJitPasses(JitTestCase):
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
        observer = torch.jit.script(default_observer())
        qconfig_dict = {'': script_qconfig(default_qconfig)}
        m = wrap_cpp_module(torch._C._jit_pass_insert_observers(m._c, "forward", qconfig_dict, False))
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
        qconfig = script_qconfig(default_qconfig)

        qconfig_dict = {
            'sub.fc': qconfig
        }
        m = wrap_cpp_module(torch._C._jit_pass_insert_observers(m._c, "forward",
                                                                qconfig_dict,
                                                                False))
        # input and output of sub
        assert len(attrs_with_prefix(m, '_observer_')) == 2
        # not quantized
        assert len(attrs_with_prefix(m.conv, '_observer_')) == 0
        # no observers since we observe in the outer most call site
        assert len(attrs_with_prefix(m.sub, '_observer_')) == 0
        # weight of linear
        assert len(attrs_with_prefix(m.sub.fc, '_observer_')) == 1

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

            def forward(self, x):
                out = x
                out += x
                return self.relu(out)

        class AddFunctionalReLU(torch.nn.Module):
            def __init__(self):
                super(AddFunctionalReLU, self).__init__()

            def forward(self, x):
                out = x
                out += x
                return F.relu(out)

        def attrs_with_prefix(module, prefix):
            return [x for x, _ in module._modules._c.items()
                    if x.startswith(prefix)]

        qconfig_dict = {'': script_qconfig(default_qconfig)}
        m = torch.jit.script(ConvFunctionalReLU())
        m = wrap_cpp_module(torch._C._jit_pass_insert_observers(m._c, "forward", qconfig_dict, False))
        # observer for weight of conv
        assert len(attrs_with_prefix(m.conv, '_observer_')) == 1
        # observer for input of conv and output of relu
        assert len(attrs_with_prefix(m, '_observer_')) == 2

        m = torch.jit.script(ConvReLUModule())
        m = wrap_cpp_module(torch._C._jit_pass_insert_observers(m._c, "forward", qconfig_dict, False))
        # observer for input of conv and output of relu
        assert len(attrs_with_prefix(m, '_observer_')) == 2
        # observer for weight of conv
        assert len(attrs_with_prefix(m.conv, '_observer_')) == 1
        # observer for output of relu
        assert len(attrs_with_prefix(m.relu, '_observer_')) == 0

        m = torch.jit.script(AddReLUModule())
        qconfig_dict = {'': script_qconfig(default_qconfig)}
        m = wrap_cpp_module(torch._C._jit_pass_insert_observers(m._c, "forward", qconfig_dict, False))
        assert len(attrs_with_prefix(m, '_observer')) == 2
        assert len(attrs_with_prefix(m.relu, '_observer')) == 0
        FileCheck().check('aten::add_') \
                   .check_not('Observer = prim::GetAttr[name="_observer_') \
                   .check('ReLU = prim::GetAttr') \
                   .run(str(get_forward_graph(m._c)))

        m = torch.jit.script(AddFunctionalReLU())
        qconfig_dict = {'': script_qconfig(default_qconfig)}
        m = wrap_cpp_module(torch._C._jit_pass_insert_observers(m._c, "forward", qconfig_dict, False))
        assert len(attrs_with_prefix(m, '_observer')) == 2
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
        qconfig_dict = {
            '': script_qconfig(default_qconfig)
        }
        m = wrap_cpp_module(torch._C._jit_pass_insert_observers(m._c, "forward", qconfig_dict, False))
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
        m = prepare_script(m, {'': default_qconfig}, False)
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
        qconfig_dict = {'': script_qconfig(default_qconfig)}
        m = wrap_cpp_module(torch._C._jit_pass_insert_observers(m._c, "forward", qconfig_dict, False))
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

        qconfig_dict = {'': script_qconfig(default_qconfig)}
        m = torch.jit.script(M())
        m = wrap_cpp_module(torch._C._jit_pass_insert_observers(m._c, "forward", qconfig_dict, False))
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

        qconfig_dict = {'': script_qconfig(default_qconfig)}
        m = torch.jit.script(M())
        m = wrap_cpp_module(torch._C._jit_pass_insert_observers(m._c, "forward", qconfig_dict, False))
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

        qconfig_dict = {'': script_qconfig(default_qconfig)}
        m = torch.jit.script(M())
        m = wrap_cpp_module(torch._C._jit_pass_insert_observers(m._c, "forward", qconfig_dict, False))
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
        m = prepare_script(m, {'': default_qconfig}, inplace=False)
        # we want to test that channel_shuffle is going to pass
        # the observed property from the output of conv1 to input of conv2
        # so that we don't insert observers for input of conv2
        assert len(attrs_with_prefix(m, '_observer_',)) == 3

    def test_insert_observers_for_if(self):
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
                self.res1 = Res(True)
                self.res2 = Res(False)

            def forward(self, x):
                x = self.res1(x)
                x = self.res2(x)
                return x

        data = [(torch.rand((1, 3, 10, 10), dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        m = torch.jit.script(M()).eval()
        m = prepare_script(m, {'': default_qconfig}, inplace=False)
        assert len(attrs_with_prefix(m, '_observer_',)) == 3

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
            qconfig = QConfig(activation=observer, weight=observer)
            qconfig_dict = {
                '': script_qconfig(qconfig)
            }
            m = wrap_cpp_module(torch._C._jit_pass_insert_observers(m._c, "forward", qconfig_dict, False))
            data = torch.randn(1, 3, 10, 10, dtype=torch.float)

            m(data)
            m = wrap_cpp_module(torch._C._jit_pass_insert_quant_dequant(m._c, "forward", False))
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
            m = wrap_cpp_module(torch._C._jit_pass_insert_quant_dequant(m._c, "forward", False))
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

    def test_swap_dequantize(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.maxpool = torch.nn.MaxPool2d(kernel_size=3)
                self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                x = torch.dequantize(x)
                x = self.maxpool(x)
                x = self.avgpool(x)
                return x
        x = torch.randn([1, 3, 10, 10], dtype=torch.float)
        x = torch.quantize_per_tensor(x, 0.5, 1, torch.quint8)
        m = torch.jit.script(M())
        ref_res = m(x)
        torch._C._jit_pass_inline(m.graph)
        FileCheck().check("aten::dequantize") \
                   .check("aten::max_pool2d") \
                   .check("aten::adaptive_avg_pool2d") \
                   .run(m.graph)
        torch._C._jit_pass_swap_dequantize(m.graph)
        FileCheck().check("aten::max_pool2d") \
                   .check("aten::adaptive_avg_pool2d") \
                   .check("dequantize") \
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
        m = quantize_script(m, qconfig_dict, _test_only_eval_fn, [data], inplace=False)
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
        model = quantize_script(model, qconfig_dict, _test_only_eval_fn, [data], inplace=False)
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
        model = quantize_script(model, qconfig_dict, _test_only_eval_fn, [data], inplace=False, debug=True)
        FileCheck().check_not("quantized::conv2d") \
                   .check("aten::conv2d") \
                   .check("aten::avg_pool2d") \
                   .check("aten::q_scale") \
                   .check_next("aten::q_zero_point") \
                   .check_next("prim::dtype") \
                   .check_next("aten::quantize_per_tensor") \
                   .check("aten::dequantize") \
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

    def test_replicate_dequant_same_value(self):
        class Mul(torch.nn.Module):
            def __init__(self):
                super(Mul, self).__init__()

            def forward(self, x):
                return x * x

        data = [(torch.rand((1, 3, 10, 10), dtype=torch.float),
                 torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]

        qconfig_dict = {'': default_qconfig}
        model = torch.jit.script(Mul()).eval()
        m = quantize_script(model, qconfig_dict, _test_only_eval_fn, [data])
        FileCheck().check("quantized::mul") \
                   .check_not("aten::mul") \
                   .run(m.graph)

class TestQuantizeScriptPTSQOps(JitTestCase):
    """ Test graph mode post training static quantization works
    for individual ops end to end.
    """
    def _test_op_impl(self, module, data, quantized_op):
        qconfig_dict = {'': get_default_qconfig('fbgemm')}
        model = torch.jit.script(module).eval()
        model = quantize_script(model, qconfig_dict, _test_only_eval_fn, [data], inplace=False)
        FileCheck().check(quantized_op) \
                   .run(model.graph)

        # make sure it runs
        *inputs, target = data[0]
        model(*inputs)

        return model

    @unittest.skipUnless(
        'fbgemm' in torch.backends.quantized.supported_engines,
        " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
        " with instruction set support avx2 or newer.",
    )
    def test_quantized_conv2d(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                return self.conv(x)

        data = [(torch.rand((1, 3, 10, 10), dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        model = self._test_op_impl(M(), data, "quantized::conv2d")
        # make sure there is only one quantize_per_tensor for input
        # and conv2d_prepack is folded
        FileCheck().check_count("aten::quantize_per_tensor", 1, exactly=True) \
                   .run(model.graph)

        FileCheck().check_not("quantized::conv2d_prepack") \
                   .run(model.graph)


    @unittest.skipUnless(
        'fbgemm' in torch.backends.quantized.supported_engines,
        " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
        " with instruction set support avx2 or newer.",
    )
    def test_quantized_conv3d(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = torch.nn.Conv3d(3, 3, 3).float()

            def forward(self, x):
                return self.conv(x)

        data = [(torch.rand((1, 3, 10, 10, 10), dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        model = self._test_op_impl(M(), data, "quantized::conv3d")
        # make sure there is only one quantize_per_tensor for input
        # and conv3d_prepack is folded
        FileCheck().check_count("aten::quantize_per_tensor", 1, exactly=True) \
                   .run(model.graph)

    @unittest.skipUnless('fbgemm' in torch.backends.quantized.supported_engines,
                         " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
                         " with instruction set support avx2 or newer.")
    def test_quantized_conv2d_relu(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(1, 4, 2, 3).float()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.conv(x))

        data = [(torch.randn(1, 1, 10, 10, dtype=torch.float),
                 torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        model = self._test_op_impl(M(), data, "quantized::conv2d_relu")

        FileCheck().check_not("aten::conv2d") \
                   .check_not("aten::relu") \
                   .check_not("quantized::conv2d(") \
                   .check_not("quantized::relu(") \
                   .run(model.graph)

    @unittest.skipUnless('fbgemm' in torch.backends.quantized.supported_engines,
                         " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
                         " with instruction set support avx2 or newer.")
    def test_quantized_conv3d_relu(self):
        class M(torch.nn.Module):
            def __init__(self, functional):
                super(M, self).__init__()
                self.conv = torch.nn.Conv3d(1, 4, 2, 3).float()
                if functional:
                    self.relu = F.relu
                else:
                    self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.conv(x))

        data = [(torch.randn(1, 1, 5, 5, 5, dtype=torch.float),
                 torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        model = self._test_op_impl(M(functional=False), data,
                                   "quantized::conv3d_relu")
        model_functional = self._test_op_impl(M(functional=True), data,
                                              "quantized::conv3d_relu")

        checker = FileCheck().check_not("aten::conv3d") \
                             .check_not("aten::relu") \
                             .check_not("quantized::conv3d(") \
                             .check_not("quantized::relu(")
        checker.run(model.graph)
        checker.run(model_functional.graph)

    def test_quantized_add(self):
        class Add(torch.nn.Module):
            def __init__(self):
                super(Add, self).__init__()

            def forward(self, x, y):
                return x + y

        class InplaceAdd(torch.nn.Module):
            def __init__(self):
                super(InplaceAdd, self).__init__()

            def forward(self, x, y):
                x += y
                return x

        for M in [Add, InplaceAdd]:
            m = torch.jit.script(M()).eval()
            m = prepare_script(m, {'': default_qconfig}, True)
            # two for input tensor, one for output
            assert len(attrs_with_prefix(m, '_observer')) == 3
            data = torch.randn(1, 1, 10, 10, dtype=torch.float)
            m(data, data)
            m = convert_script(m, True)
            FileCheck().check_not("aten::add") \
                       .check_not("aten::add_") \
                       .check("quantized::add") \
                       .run(m.graph_for(data, data))

    def test_quantized_add_scalar(self):
        class AddScalar(torch.nn.Module):
            def __init__(self):
                super(AddScalar, self).__init__()

            def forward(self, x):
                return x + 3

        class InplaceAddScalar(torch.nn.Module):
            def __init__(self):
                super(InplaceAddScalar, self).__init__()

            def forward(self, x):
                x += 3
                return x

        for M in [AddScalar, InplaceAddScalar]:
            m = torch.jit.script(M()).eval()
            m = prepare_script(m, {'': default_qconfig}, True)
            # for input tensor
            assert len(attrs_with_prefix(m, '_observer')) == 1
            data = torch.randn(1, 1, 10, 10, dtype=torch.float)
            m(data)
            m = convert_script(m, True)
            FileCheck().check_not("aten::add") \
                       .check_not("aten::add_") \
                       .check("quantized::add_scalar") \
                       .run(m.graph_for(data))

    def test_quantized_add_relu(self):
        class M(torch.nn.Module):
            def __init__(self, inplace):
                super(M, self).__init__()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x, y):
                x += y
                return self.relu(x)

        for inplace in [True, False]:
            m = torch.jit.script(M(inplace).eval())
            m = prepare_script(m, {'': default_qconfig}, True)
            data = torch.randn(1, 1, 10, 10, dtype=torch.float)
            m(data, data)
            m = convert_script(m, True)
            FileCheck().check_not("aten::add") \
                       .check_not("aten::relu") \
                       .check_not("aten::relu_") \
                       .check_not("quantized::add") \
                       .check_not("quantized::relu") \
                       .check("quantized::add_relu") \
                       .run(m.graph_for(data, data))

    def test_quantized_add_scalar_relu(self):
        class AddScalar(torch.nn.Module):
            def __init__(self):
                super(AddScalar, self).__init__()

            def forward(self, x):
                return F.relu(x + 3)

        class InplaceAddScalar(torch.nn.Module):
            def __init__(self):
                super(InplaceAddScalar, self).__init__()

            def forward(self, x):
                x += 3
                return F.relu(x)

        for M in [AddScalar, InplaceAddScalar]:
            m = torch.jit.script(M()).eval()
            m = prepare_script(m, {'': default_qconfig}, True)
            # for input tensor
            assert len(attrs_with_prefix(m, '_observer')) == 1
            data = torch.randn(1, 1, 10, 10, dtype=torch.float)
            m(data)
            m = convert_script(m, True)
            FileCheck().check_not("aten::add") \
                       .check_not("aten::add_") \
                       .check_not("aten::relu") \
                       .check_not("quantized::add_scalar") \
                       .check_not("quantized::relu") \
                       .check("quantized::add_scalar_relu") \
                       .run(m.graph_for(data))

    @unittest.skipUnless('fbgemm' in torch.backends.quantized.supported_engines,
                         " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
                         " with instruction set support avx2 or newer.")
    def test_quantized_cat(self):
        """ Note that we to support the case that torch.cat is quantized
        indepdently, we need to have an observer that works
        for list of Tensors.
        """
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv1 = torch.nn.Conv2d(1, 1, 1).float()
                self.conv2 = torch.nn.Conv2d(1, 1, 1).float()

            def forward(self, x, y):
                x = self.conv1(x)
                y = self.conv2(y)
                return torch.cat([x, y], 1)

        m = torch.jit.script(M().eval())
        m = prepare_script(m, {'': default_qconfig}, True)
        # four for input and output of conv and one for output of cat
        # this also tests the ListConstruct can preserve the observed property so that
        # torch.cat knows that inputs are observed
        assert len(attrs_with_prefix(m, '_observer_')) == 5
        data = torch.randn(1, 1, 10, 10, dtype=torch.float)
        m(data, data)
        m = convert_script(m, True)

        FileCheck().check_not("aten::cat") \
                   .check("quantized::cat") \
                   .run(m.graph_for(data, data))

    def test_qbatch_norm(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.bn = torch.nn.BatchNorm2d(3).to(torch.float)

            def forward(self, x):
                return self.bn(x)

        data = [(torch.rand((1, 3, 10, 10), dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        model = self._test_op_impl(M(), data, "quantized::batch_norm2d")

        FileCheck().check_not("aten::batch_norm") \
                   .run(model.graph)

    def test_qbatch_norm_relu(self):
        class BNRelu(torch.nn.Module):
            def __init__(self, inplace):
                super(BNRelu, self).__init__()
                self.bn = torch.nn.BatchNorm2d(3).to(torch.float)
                self.relu = torch.nn.ReLU(inplace=inplace)

            def forward(self, x):
                return self.relu(self.bn(x))

        # Note Fusion for functional Relu with inplace argument isn't currently supported in fusion patterns.
        class BNFuncRelu(torch.nn.Module):
            def __init__(self):
                super(BNFuncRelu, self).__init__()
                self.bn = torch.nn.BatchNorm2d(3).to(torch.float)

            def forward(self, x):
                return F.relu(self.bn(x), False)

        class BNFuncInplaceRelu(torch.nn.Module):
            def __init__(self):
                super(BNFuncInplaceRelu, self).__init__()
                self.bn = torch.nn.BatchNorm2d(3).to(torch.float)

            def forward(self, x):
                return F.relu(self.bn(x), True)

        data = [(torch.rand((1, 3, 10, 10), dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        for instance in [BNRelu(True), BNRelu(False), BNFuncRelu(), BNFuncInplaceRelu()]:
            model = self._test_op_impl(instance, data, "quantized::batch_norm2d_relu")
            FileCheck().check_not("aten::batch_norm") \
                       .check_not("aten::relu") \
                       .check_not("aten::relu_") \
                       .run(model.graph)

    def test_quantized_mul(self):
        class Mul(torch.nn.Module):
            def __init__(self):
                super(Mul, self).__init__()

            def forward(self, x, y):
                return x * y

        class InplaceMul(torch.nn.Module):
            def __init__(self):
                super(InplaceMul, self).__init__()

            def forward(self, x, y):
                x *= y
                return x

        data = [(torch.rand((1, 3, 10, 10), dtype=torch.float), torch.rand((1, 3, 10, 10), dtype=torch.float),
                 torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        for M in [Mul(), InplaceMul()]:
            m = self._test_op_impl(M, data, "quantized::mul")
            FileCheck().check_not("aten::mul") \
                       .check_not("aten::mul_") \
                       .run(m.graph)

    def test_quantized_mul_scalar(self):
        class MulScalar(torch.nn.Module):
            def __init__(self):
                super(MulScalar, self).__init__()

            def forward(self, x):
                return x * 3

        class InplaceMulScalar(torch.nn.Module):
            def __init__(self):
                super(InplaceMulScalar, self).__init__()

            def forward(self, x):
                x *= 3
                return x

        data = [(torch.rand((1, 3, 10, 10), dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        for M in [MulScalar(), InplaceMulScalar()]:
            m = self._test_op_impl(M, data, "quantized::mul_scalar")
            FileCheck().check_not("aten::mul") \
                       .check_not("aten::mul_") \
                       .run(m.graph)

    def test_quantized_mul_relu(self):
        class MulRelu(torch.nn.Module):
            def __init__(self, inplace):
                super(MulRelu, self).__init__()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x, y):
                x = x * y
                return self.relu(x)

        class InplaceMulRelu(torch.nn.Module):
            def __init__(self, inplace):
                super(InplaceMulRelu, self).__init__()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x, y):
                x *= y
                return self.relu(x)

        data = [(torch.rand((1, 3, 10, 10), dtype=torch.float), torch.rand((1, 3, 10, 10), dtype=torch.float),
                 torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        for M in [MulRelu(True), MulRelu(False), InplaceMulRelu(True), InplaceMulRelu(False)]:
            m = self._test_op_impl(M, data, "quantized::mul_relu")
            FileCheck().check_not("aten::mul") \
                       .check_not("aten::mul_") \
                       .check_not("aten::relu") \
                       .check_not("aten::relu_") \
                       .check_not("quantized::relu") \
                       .run(m.graph)

    def test_quantized_mul_scalar_relu(self):
        class MulScalar(torch.nn.Module):
            def __init__(self):
                super(MulScalar, self).__init__()

            def forward(self, x):
                return F.relu(x * 3)

        class InplaceMulScalar(torch.nn.Module):
            def __init__(self):
                super(InplaceMulScalar, self).__init__()

            def forward(self, x):
                x *= 3
                return F.relu(x)

        data = [(torch.rand((1, 3, 10, 10), dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        for M in [MulScalar(), InplaceMulScalar()]:
            m = self._test_op_impl(M, data, "quantized::mul_scalar_relu")
            FileCheck().check_not("aten::mul") \
                       .check_not("aten::mul_") \
                       .check_not("aten::relu") \
                       .check_not("quantized::relu") \
                       .run(m.graph)

    def test_hardswish(self):
        data = [(torch.rand((1, 3, 10, 10), dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        m = self._test_op_impl(torch.nn.Hardswish(), data, "quantized::hardswish")
        FileCheck().check_not("aten::hardswish") \
                   .run(m.graph)

    def test_layer_norm(self):
        data = [(torch.rand((1, 3, 10, 10), dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        layer_norm = torch.nn.LayerNorm([3, 10, 10])
        m = self._test_op_impl(layer_norm, data, "quantized::layer_norm")
        FileCheck().check_not("aten::layer_norm") \
                   .run(m.graph)

    def test_quantize_general_shape_ops(self):
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
                self.sigmoid = torch.nn.Sigmoid()
                self.tanh = torch.nn.Tanh()
                self.hardtanh = torch.nn.Hardtanh()
                self.elu = torch.nn.ELU()
                self.hardsigmoid = torch.nn.Hardsigmoid()
                self.relu = torch.nn.ReLU()
                self.relu6 = torch.nn.ReLU6()
                self.leaky_relu = torch.nn.LeakyReLU()

            def forward(self, x):
                x = self.conv(x)
                x = self.maxpool1d(x)
                x = self.maxpool2d(x)
                x = self.maxpool3d(x)
                x = torch.flatten(x)
                x = torch.max(x)
                x = torch.min(x)
                x = torch.sigmoid(x)
                x = x.reshape([-1])
                x = x.resize_(1, 1, x.numel())
                x = self.sigmoid(x)
                x = x.view(-1)
                x = x.transpose(1, 2)
                x = x.contiguous()
                x, y = torch.chunk(x, 2)
                x = F.dropout(x)
                x = self.dropout(x)
                x, _ = torch.sort(x)
                x = F.sigmoid(x)
                x = x.permute(0, 2, 3, 1)
                x = torch.repeat_interleave(x, 3, 1)
                x = self.tanh(x)
                x = F.tanh(x)
                x = torch.tanh(x)
                x = self.hardtanh(x)
                x = F.hardtanh(x)
                x.hardtanh_()
                x = self.elu(x)
                x = F.elu(x)
                x.elu_()
                x = self.hardsigmoid(x)
                x = F.hardsigmoid(x)
                x.hardsigmoid_()
                x = self.relu(x)
                x = F.relu(x)
                x.relu_()
                x = self.relu6(x)
                x = F.relu6(x)
                x = self.leaky_relu(x)
                x = F.leaky_relu(x)
                x.leaky_relu_()
                x = self.conv(x)
                return x

        m = torch.jit.script(M())
        qconfig = script_qconfig(default_qconfig)
        # dummy data to suppress warning
        data = torch.rand((1, 3, 10, 10))
        get_forward(qconfig.activation)(data)
        get_forward(qconfig.weight)(data)

        m = wrap_cpp_module(torch._C._jit_pass_insert_observers(
            m._c, 'forward', {'': qconfig}, inplace=False))
        m = convert_script(m, True)
        # This checks that the dequantize from the output of first conv
        # is being propagated to the end, so that we don't insert extra
        # observers and also successfully fused two quantized::conv2d
        # patterns
        # one quantize_per_tensor for input
        FileCheck().check_count("aten::quantize_per_tensor", 1, exactly=True) \
                   .check_count("quantized::conv2d", 2, exactly=True) \
                   .check("aten::dequantize") \
                   .run(m.graph)

    def test_quantize_general_value_ops(self):
        """ A test that checks dequantize will be swapped for \
        all supported general value ops like aten::avg_pool2d \
        without actually checking for execution of these ops
        """
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.avg_pool1d = torch.nn.AvgPool1d(3)
                self.avg_pool2d = torch.nn.AvgPool2d(3)
                self.avg_pool3d = torch.nn.AvgPool2d(3)
                self.adaptive_avg_pool1d = torch.nn.AdaptiveAvgPool1d((1))
                self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.adaptive_avg_pool3d = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
                self.conv = torch.nn.Conv2d(3, 3, 3)

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
                x = x.mean()
                # interpolate node will introduce 3 quantize_per_tensor ops
                x = F.interpolate(x, 4, mode='nearest')  # interpolate node
                x = F.upsample(x, (32, 32))  # interpolate node
                x = F.upsample_nearest(x, (32, 32))  # interpolate node
                x = F.interpolate(x, 4, mode='linear')  # common node
                x = F.upsample_bilinear(x, (32, 32))  # common node
                x = torch.clamp(x, -3, 3)
                x = x.clamp(-2.5, 2.5)
                # x = x.clamp_(-2, 2)  # Enable when quantized `clamp_` is ready
                x = self.conv(x)
                return x

        m = torch.jit.script(M())
        qconfig = script_qconfig(default_qconfig)
        # dummy data to suppress warning
        data = torch.rand((1, 3, 10, 10))
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
        # number of quantize_per_tensor op for type
        num_quant_by_op_type = {'conv': 2, 'common': 1, 'interpolate': 3}
        # number of ops for each type
        num_op_by_op_type = {'conv': 2, 'common': 18, 'interpolate': 3}
        num_quantize_per_tensor = 1  # for output
        for op_type, num_op in num_op_by_op_type.items():
            num_quantize_per_tensor += num_op * num_quant_by_op_type[op_type]
        FileCheck().check_count("aten::quantize_per_tensor(", num_quantize_per_tensor, exactly=True) \
                   .run(m1.graph)

        # This checks that the dequantize from the output of first conv
        # is being propagated to the end, so that we don't insert extra
        # observers and also successfully fused two quantized::conv2d
        # patterns
        # one quantize_per_tensor for input
        m2 = convert_script(m, debug=False)
        FileCheck().check_count("aten::quantize_per_tensor(", 1, exactly=True) \
                   .check_count("quantized::conv2d(", 2, exactly=True) \
                   .check("aten::dequantize(") \
                   .run(m2.graph)

class TestQuantizeDynamicScript(JitTestCase):
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

        m = torch.jit.script(M())

        m = prepare_dynamic_script(m, {'': default_dynamic_qconfig})
        data = torch.randn(5, 5, dtype=torch.float)

        m(data)
        m = wrap_cpp_module(torch._C._jit_pass_insert_quant_dequant(m._c, "forward", False, True))

        assert len(m._modules._c.items()) == 2, \
            'Expected to have two submodule of linear'

        m(data)
        quant_func = "aten::quantize_per_tensor"

        # quantizing activations
        FileCheck().check("aten::_choose_qparams_per_tensor") \
                   .check_next(quant_func) \
                   .check_next("aten::dequantize") \
                   .check("aten::_choose_qparams_per_tensor") \
                   .check_next(quant_func) \
                   .check_next("aten::dequantize") \
                   .check(quant_func) \
                   .check_next("aten::dequantize") \
                   .check_not(quant_func) \
                   .check("return") \
                   .run(m.graph)

    def test_finalize_for_linear_dynamic(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.fc = torch.nn.Linear(5, 5).float()

            def forward(self, x):
                return self.fc(x)

        data = torch.rand((1, 5), dtype=torch.float)
        qconfig_dict = {'': default_dynamic_qconfig}
        model = torch.jit.script(M()).eval()
        model = quantize_dynamic_script(model, qconfig_dict, data)
        FileCheck().check("quantized::linear_dynamic") \
                   .run(model.graph)

    def test_dynamic_multi_op(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)

            def forward(self, x):
                x = x + 5
                return self.fc1(x)

        m = torch.jit.script(M())
        data = torch.randn((1, 5), dtype=torch.float)
        qconfig_dict = {'' : default_dynamic_qconfig}
        model = quantize_dynamic_script(m, qconfig_dict, data)
        # add op is not dynamically quantized.
        FileCheck().check("aten::add") \
                   .check("quantized::linear_dynamic") \
                   .run(model.graph)

    def test_dynamic_quant_multi_uses(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.fc = torch.nn.Linear(5, 5).float()

            def forward(self, x):
                size1 = x.size()
                size2 = x.size()
                return self.fc(x), size1, size2

        model = torch.jit.script(M()).eval()
        data = torch.rand((1, 5), dtype=torch.float)
        qconfig_dict = {'': default_dynamic_qconfig}

        model = quantize_dynamic_script(model, qconfig_dict, [data])
        FileCheck().check("quantized::linear_dynamic") \
                   .check_not("aten::_choose_qparams_per_tensor") \
                   .run(model.graph)

    def test_prepare_dynamic_lstm(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.lstm = torch.nn.LSTM(2, 2).to(dtype=torch.float)

            def forward(self, x):
                return self.lstm(x)
        from torch.quantization.observer import default_dynamic_quant_observer, _MinMaxTensorListObserver
        qconfig = QConfigDynamic(activation=default_dynamic_quant_observer,
                                 weight=_MinMaxTensorListObserver)
        m = torch.jit.script(M())
        m = prepare_dynamic_script(m, {'': qconfig})
        assert len(attrs_with_prefix(m.lstm, '_observer_')) == 1
        FileCheck().check('_MinMaxTensorListObserver = prim::GetAttr[name="_observer_0') \
                   .check("aten::lstm") \
                   .check("return") \
                   .run(str(get_module_method(m, 'lstm', 'forward__0').graph))
