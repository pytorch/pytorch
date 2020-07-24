import unittest
import torch
import torch.backends.xnnpack
import torch.utils.bundled_inputs
import torch.nn as nn
from torch.testing._internal.jit_utils import get_forward, get_forward_graph
from torch.utils.mobile_optimizer import *
from torch.nn import functional as F
from torch._C import MobileOptimizerType

FileCheck = torch._C.FileCheck

class TestOptimizer(unittest.TestCase):

    @unittest.skipUnless(torch.backends.xnnpack.enabled,
                         " XNNPACK must be enabled for these tests."
                         " Please build with USE_XNNPACK=1.")
    def test_optimize_for_mobile(self):
        batch_size = 2
        input_channels_per_group = 6
        height = 16
        width = 16
        output_channels_per_group = 6
        groups = 4
        kernel_h = kernel_w = 3
        stride_h = stride_w = 1
        pad_h = pad_w = 1
        dilation = 1
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        kernels = (kernel_h, kernel_w)
        strides = (stride_h, stride_w)
        paddings = (pad_h, pad_w)
        dilations = (dilation, dilation)
        conv_weight_shape = (output_channels, input_channels_per_group, kernel_h, kernel_w)
        conv_bias_shape = (output_channels)

        input_data = torch.rand((batch_size, input_channels, height, width))
        conv_weight = torch.rand((output_channels, input_channels_per_group, kernel_h, kernel_w))
        conv_bias = torch.rand((output_channels))
        result = F.conv2d(input_data, conv_weight, conv_bias, strides, paddings, dilations, groups)
        weight_output_dim = 24
        linear_input_shape = result.shape[1]
        linear_weight_shape = (weight_output_dim, linear_input_shape)

        class MyTestModule(torch.nn.Module):
            def __init__(self):
                super(MyTestModule, self).__init__()
                self.conv_weight = torch.nn.Parameter(torch.Tensor(torch.rand(conv_weight_shape)))
                self.conv_bias = torch.nn.Parameter(torch.Tensor(torch.rand((conv_bias_shape))))
                self.linear_weight = torch.nn.Parameter(torch.Tensor(torch.rand(linear_weight_shape)))
                self.linear_bias = torch.nn.Parameter(torch.Tensor(torch.rand((weight_output_dim))))
                self.strides = strides
                self.paddings = paddings
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                o = F.conv2d(x, self.conv_weight, self.conv_bias,
                             self.strides, self.paddings, self.dilations, self.groups)
                o = F.relu(o)
                x = o.permute([0, 2, 3, 1])
                o = F.linear(x, self.linear_weight, self.linear_bias)
                o = o + x
                return F.relu(o)

        class BNTestModule(torch.nn.Module):
            def __init__(self):
                super(BNTestModule, self).__init__()
                self.conv = torch.nn.Conv2d(1, 20, 5, 1)
                self.bn = torch.nn.BatchNorm2d(num_features=20)
                self.bn.eps = 0.0023

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        data_shape = (batch_size, input_channels, height, width)
        input_data = torch.normal(1, 20, size=data_shape)

        scripted_model = torch.jit.script(MyTestModule())
        scripted_model.eval()
        initial_result = scripted_model(input_data)

        optimized_scripted_model = optimize_for_mobile(scripted_model)
        optimized_result = optimized_scripted_model(input_data)

        FileCheck().check_not("Tensor = aten::conv2d") \
                   .check_not("Tensor = prim::CallFunction") \
                   .check_not("prepacked::conv2d_clamp_prepack") \
                   .check_count("prepacked::conv2d_clamp_run", 1, exactly=True) \
                   .check_not("prepacked::linear_clamp_prepack") \
                   .check_count("prepacked::linear_clamp_run", 1, exactly=True) \
                   .check_not("aten::add(") \
                   .check_not("aten::relu(") \
                   .check_count("aten::add_relu(", 1, exactly=True) \
                   .run(optimized_scripted_model.graph)
        torch.testing.assert_allclose(initial_result, optimized_result, rtol=1e-2, atol=1e-3)


        optimization_blacklist_no_prepack = {MobileOptimizerType.INSERT_FOLD_PREPACK_OPS}
        optimized_scripted_model_no_prepack = optimize_for_mobile(scripted_model, optimization_blacklist_no_prepack)
        optimized_result_no_prepack = optimized_scripted_model_no_prepack(input_data)

        FileCheck().check_count("Tensor = aten::conv2d", 1, exactly=True) \
                   .check_not("prepacked::linear_clamp_run") \
                   .check_not("prepacked::conv2d_clamp_run") \
                   .run(optimized_scripted_model_no_prepack.graph)
        torch.testing.assert_allclose(initial_result, optimized_result_no_prepack, rtol=1e-2, atol=1e-3)


        bn_test_module = BNTestModule()
        bn_scripted_module = torch.jit.script(bn_test_module)
        bn_scripted_module.eval()
        self.assertEqual(len(torch.jit.export_opnames(bn_scripted_module)), 13)
        FileCheck().check_count("prim::CallMethod[name=\"forward\"]", 2, exactly=True) \
                   .run(str(get_forward(bn_scripted_module._c).graph))

        optimization_blacklist_no_prepack = {MobileOptimizerType.INSERT_FOLD_PREPACK_OPS}
        bn_fold_scripted_module = optimize_for_mobile(bn_scripted_module, optimization_blacklist_no_prepack)
        self.assertEqual(len(torch.jit.export_opnames(bn_fold_scripted_module)), 1)
        FileCheck().check_count("prim::CallMethod[name=\"forward\"]", 1, exactly=True) \
                   .run(str(get_forward_graph(bn_fold_scripted_module._c)))
        bn_input = torch.rand(1, 1, 6, 6)
        torch.testing.assert_allclose(bn_scripted_module(bn_input), bn_fold_scripted_module(bn_input), rtol=1e-2, atol=1e-3)

        optimization_blacklist_no_fold_bn = {MobileOptimizerType.CONV_BN_FUSION}
        no_bn_fold_scripted_module = optimize_for_mobile(bn_scripted_module, optimization_blacklist_no_fold_bn)
        FileCheck().check_count("aten::batch_norm", 1, exactly=True) \
                   .run(str(get_forward_graph(no_bn_fold_scripted_module._c)))
        bn_input = torch.rand(1, 1, 6, 6)
        torch.testing.assert_allclose(bn_scripted_module(bn_input), no_bn_fold_scripted_module(bn_input), rtol=1e-2, atol=1e-3)

        class MyPreserveMethodsTest(torch.nn.Module):
            def __init__(self):
                super(MyPreserveMethodsTest, self).__init__()
                self.linear_weight = torch.nn.Parameter(torch.Tensor(torch.rand(linear_weight_shape)))
                self.linear_bias = torch.nn.Parameter(torch.Tensor(torch.rand((weight_output_dim))))

            def forward(self, x):
                o = F.linear(x, self.linear_weight, self.linear_bias)
                return F.relu(o)

            @torch.jit.export
            def preserveThis(self):
                pass

        preserve_method_module = MyPreserveMethodsTest()
        m = torch.jit.script(preserve_method_module)
        m.eval()
        opt_m = optimize_for_mobile(m)
        no_preserveThis = getattr(opt_m, "preserveThis", None)
        self.assertEqual(no_preserveThis, None)
        opt_m = optimize_for_mobile(m, preserved_methods=["preserveThis"])
        preserveThis = getattr(opt_m, "preserveThis", None)
        self.assertNotEqual(preserveThis, None)


    def test_generate_mobile_module_lints(self):
        class MyTestModule(torch.nn.Module):
            def __init__(self):
                super(MyTestModule, self).__init__()
                self.fc = torch.nn.Linear(4, 4)
                self.dropout = torch.nn.Dropout(p=0.5)

            def forward(self, inputs):
                out = self.fc(inputs)
                out = self.dropout(out)
                return out

        class MyBNModule(torch.nn.Module):
            def __init__(self):
                super(MyBNModule, self).__init__()
                self.bn = torch.nn.BatchNorm2d(4, affine=True)

            def forward(self, inputs):
                bn = self.bn(inputs)
                return bn

        class MyBundledInputModule(torch.nn.Module):
            def __init__(self):
                super(MyBundledInputModule, self).__init__()

            def forward(self, inputs):
                return inputs

        def get_lint_count_by_type(lint_type, module_lint_List):
            return len([lint_dict for lint_dict in module_lint_List if lint_dict['name'] == lint_type.name])

        test_module = torch.jit.script(MyTestModule())
        test_module_lint_list = generate_mobile_module_lints(test_module)
        self.assertEqual(len(test_module_lint_list), 4)
        self.assertEqual(get_lint_count_by_type(LintCode.BUNDLED_INPUT, test_module_lint_list), 1)
        self.assertEqual(get_lint_count_by_type(LintCode.DROPOUT, test_module_lint_list), 1)
        self.assertEqual(get_lint_count_by_type(LintCode.REQUIRES_GRAD, test_module_lint_list), 2)

        bn_module = torch.jit.script(MyBNModule())
        bn_module_lint_list = generate_mobile_module_lints(bn_module)
        self.assertEqual(len(bn_module_lint_list), 4)
        self.assertEqual(get_lint_count_by_type(LintCode.BUNDLED_INPUT, bn_module_lint_list), 1)
        self.assertEqual(get_lint_count_by_type(LintCode.BATCHNORM, bn_module_lint_list), 1)
        self.assertEqual(get_lint_count_by_type(LintCode.REQUIRES_GRAD, bn_module_lint_list), 2)

        bi_module = torch.jit.script(MyBundledInputModule())
        torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
            bi_module, [(torch.tensor([1]),)], [])
        bi_module_lint_list = generate_mobile_module_lints(bi_module)
        self.assertEqual(len(bi_module_lint_list), 0)

    @unittest.skipUnless(torch.backends.xnnpack.enabled,
                         " XNNPACK must be enabled for these tests."
                         " Please build with USE_XNNPACK=1.")
    def test_hoist_conv_packed_params(self):

        class Standalone(nn.Module):
            def __init__(self):
                super(Standalone, self).__init__()
                self.quant = torch.quantization.QuantStub()
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.dequant = torch.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv1(x)
                x = self.dequant(x)
                return x

        class Child(nn.Module):
            def __init__(self):
                super(Child, self).__init__()
                self.conv1 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv1(x)
                return x

        class Parent(nn.Module):
            def __init__(self):
                super(Parent, self).__init__()
                self.quant = torch.quantization.QuantStub()
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.child = Child()
                # TODO: test nn.Sequential after #42039 is fixed
                self.dequant = torch.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv1(x)
                x = self.child(x)
                x = self.dequant(x)
                return x

        def _static_quant(model):
            model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
            torch.quantization.prepare(model, inplace=True)
            model(torch.randn(4, 1, 4, 4))
            torch.quantization.convert(model, inplace=True)
            return model

        # basic case

        m = Standalone()
        m = _static_quant(m)
        m = torch.jit.script(m)
        m_optim = optimize_for_mobile(m)

        FileCheck().check_not("Conv2d = prim::GetAttr[name=\"conv1\"]") \
                   .check_count("_jit_pass_hoist_conv_packed_params", 1, exactly=True) \
                   .run(m_optim.graph)

        data = torch.randn(4, 1, 4, 4)
        m_res = m(data)
        m_optim_res = m_optim(data)
        torch.testing.assert_allclose(m_res, m_optim_res, rtol=1e-2, atol=1e-3)

        # generic case

        m = Parent()
        m = _static_quant(m)
        m = torch.jit.script(m)
        m_optim = optimize_for_mobile(m)

        FileCheck().check_not("Conv2d = prim::GetAttr[name=\"conv1\"]") \
                   .check_count("_jit_pass_hoist_conv_packed_params", 2, exactly=True) \
                   .run(m_optim.graph)

        data = torch.randn(4, 1, 4, 4)
        m_res = m(data)
        m_optim_res = m_optim(data)
        torch.testing.assert_allclose(m_res, m_optim_res, rtol=1e-2, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
