import unittest
import torch
import torch.nn as nn
import torch.backends.xnnpack
import torch.utils.bundled_inputs
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.jit_utils import get_forward, get_forward_graph
from torch.utils.mobile_optimizer import *
from torch.nn import functional as F
from torch._C import MobileOptimizerType
from torch.testing._internal.common_quantized import override_quantized_engine

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

FileCheck = torch._C.FileCheck

class TestOptimizer(TestCase):

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
                   .check_count("aten::_add_relu(", 1, exactly=True) \
                   .run(optimized_scripted_model.graph)
        torch.testing.assert_allclose(initial_result, optimized_result, rtol=1e-2, atol=1e-3)


        optimization_blocklist_no_prepack = {MobileOptimizerType.INSERT_FOLD_PREPACK_OPS}
        optimized_scripted_model_no_prepack = optimize_for_mobile(scripted_model, optimization_blocklist_no_prepack)
        optimized_result_no_prepack = optimized_scripted_model_no_prepack(input_data)

        FileCheck().check_count("Tensor = aten::conv2d", 1, exactly=True) \
                   .check_not("prepacked::linear_clamp_run") \
                   .check_not("prepacked::conv2d_clamp_run") \
                   .run(optimized_scripted_model_no_prepack.graph)
        torch.testing.assert_allclose(initial_result, optimized_result_no_prepack, rtol=1e-2, atol=1e-3)


        bn_test_module = BNTestModule()
        bn_scripted_module = torch.jit.script(bn_test_module)
        bn_scripted_module.eval()

        self.assertEqual(len(torch.jit.export_opnames(bn_scripted_module)), 14)
        FileCheck().check_count("prim::CallMethod[name=\"forward\"]", 2, exactly=True) \
                   .run(str(get_forward(bn_scripted_module._c).graph))

        optimization_blocklist_no_prepack = {MobileOptimizerType.INSERT_FOLD_PREPACK_OPS}
        bn_fold_scripted_module = optimize_for_mobile(bn_scripted_module, optimization_blocklist_no_prepack)
        self.assertEqual(len(torch.jit.export_opnames(bn_fold_scripted_module)), 1)
        bn_input = torch.rand(1, 1, 6, 6)
        torch.testing.assert_allclose(bn_scripted_module(bn_input), bn_fold_scripted_module(bn_input), rtol=1e-2, atol=1e-3)

        optimization_blocklist_no_fold_bn = {MobileOptimizerType.CONV_BN_FUSION}
        no_bn_fold_scripted_module = optimize_for_mobile(bn_scripted_module, optimization_blocklist_no_fold_bn)
        FileCheck().check_count("aten::batch_norm", 1, exactly=True) \
                   .run(str(get_forward_graph(no_bn_fold_scripted_module._c)))
        bn_input = torch.rand(1, 1, 6, 6)
        torch.testing.assert_allclose(bn_scripted_module(bn_input), no_bn_fold_scripted_module(bn_input), rtol=1e-2, atol=1e-3)

        class MyMobileOptimizedTagTest(torch.nn.Module):
            def __init__(self):
                super(MyMobileOptimizedTagTest, self).__init__()
                self.linear_weight = torch.nn.Parameter(torch.Tensor(torch.rand(linear_weight_shape)))
                self.linear_bias = torch.nn.Parameter(torch.Tensor(torch.rand((weight_output_dim))))

            def forward(self, x):
                o = F.linear(x, self.linear_weight, self.linear_bias)
                return F.relu(o)

        mobile_optimized_tag_module = MyMobileOptimizedTagTest()
        m = torch.jit.script(mobile_optimized_tag_module)
        m.eval()
        opt_m = optimize_for_mobile(m)
        tag = getattr(opt_m, "mobile_optimized", None)
        self.assertTrue(tag)

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

        class OptimizeNoForwardTest(torch.nn.Module):
            def __init__(self):
                super(OptimizeNoForwardTest, self).__init__()
                self.l = nn.Linear(10, 100)
                self.l2 = nn.Linear(100, 1)
                self.d = nn.Dropout(p=0.2)

            @torch.jit.export
            def foo(self, x):
                x = self.d(F.relu(self.l(x)))
                x = self.l2(x)
                x = x + torch.ones(1, 100)
                return F.relu(x)
        input_data = torch.ones(1, 10)
        m = torch.jit.script(OptimizeNoForwardTest())
        m.eval()
        initial_result = m.foo(input_data)

        optimized_scripted_model = optimize_for_mobile(m, methods_to_optimize=['foo'])
        optimized_result = optimized_scripted_model.foo(input_data)

        FileCheck().check_not("dropout.__") \
            .check_count("aten::_add_relu(", 1, exactly=True) \
            .run(optimized_scripted_model.foo.graph)
        torch.testing.assert_allclose(initial_result, optimized_result, rtol=1e-2, atol=1e-3)

        class BNTestNoForwardModule(torch.nn.Module):
            def __init__(self):
                super(BNTestNoForwardModule, self).__init__()
                self.conv = torch.nn.Conv2d(1, 20, 5, 1)
                self.bn = torch.nn.BatchNorm2d(num_features=20)
                self.bn.eps = 0.0023

            @torch.jit.export
            def foo(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        bn_test_no_forward_module = BNTestNoForwardModule()
        bn_no_forward_scripted_module = torch.jit.script(bn_test_no_forward_module)
        bn_no_forward_scripted_module.eval()

        self.assertEqual(len(torch.jit.export_opnames(bn_no_forward_scripted_module)), 14)
        FileCheck().check_count("prim::CallMethod[name=\"forward\"]", 2, exactly=True) \
                   .run(bn_no_forward_scripted_module.foo.graph)

        bn_fold_no_foward_scripted_module = optimize_for_mobile(bn_no_forward_scripted_module, methods_to_optimize=['foo'])
        self.assertEqual(len(torch.jit.export_opnames(bn_fold_no_foward_scripted_module)), 1)
        bn_input = torch.rand(1, 1, 6, 6)
        torch.testing.assert_allclose(
            bn_no_forward_scripted_module.foo(bn_input),
            bn_fold_no_foward_scripted_module.foo(bn_input),
            rtol=1e-2,
            atol=1e-3)

    @unittest.skipUnless(torch.backends.xnnpack.enabled,
                         " XNNPACK must be enabled for these tests."
                         " Please build with USE_XNNPACK=1.")
    def test_quantized_conv_no_asan_failures(self):
        # There were ASAN failures when fold_conv_bn was run on
        # already quantized conv modules. Verifying that this does
        # not happen again.

        if 'qnnpack' not in torch.backends.quantized.supported_engines:
            return

        class Child(nn.Module):
            def __init__(self):
                super(Child, self).__init__()
                self.conv2 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv2(x)
                return x

        class Parent(nn.Module):
            def __init__(self):
                super(Parent, self).__init__()
                self.quant = torch.quantization.QuantStub()
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.child = Child()
                self.dequant = torch.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv1(x)
                x = self.child(x)
                x = self.dequant(x)
                return x

        with override_quantized_engine('qnnpack'):
            model = Parent()
            model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
            torch.quantization.prepare(model, inplace=True)
            model(torch.randn(4, 1, 4, 4))
            torch.quantization.convert(model, inplace=True)
            model = torch.jit.script(model)
            # this line should not have ASAN failures
            model_optim = optimize_for_mobile(model)

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

    def test_preserve_bundled_inputs_methods(self):
        class MyBundledInputModule(torch.nn.Module):
            def __init__(self):
                super(MyBundledInputModule, self).__init__()

            def forward(self, inputs):
                return inputs

        class MyIncompleteBundledInputModule(torch.nn.Module):
            def __init__(self):
                super(MyIncompleteBundledInputModule, self).__init__()

            def forward(self, inputs):
                return inputs

            @torch.jit.export
            def get_all_bundled_inputs(self):
                pass

        bi_module = torch.jit.script(MyBundledInputModule())
        module_optim_bi_not_preserved = optimize_for_mobile(bi_module)

        # Expected to be False since no bundled inputs methods were added
        self.assertFalse(
            hasattr(module_optim_bi_not_preserved, 'get_all_bundled_inputs') or
            hasattr(module_optim_bi_not_preserved, 'get_num_bundled_inputs') or
            hasattr(module_optim_bi_not_preserved, 'run_on_bundled_input')
        )

        # We expect an exception here
        with self.assertRaises(AttributeError):
            module_optim_bi_not_preserved.run_on_bundled_input(0)

        # Add bundled inputs methods to the module
        torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
            bi_module, [(torch.tensor([1]),)], [])
        # Now they should be preserved
        module_optim_bi_preserved = optimize_for_mobile(bi_module)

        # All of the bundled inputs methods were preserved
        self.assertTrue(
            hasattr(module_optim_bi_preserved, 'get_all_bundled_inputs') and
            hasattr(module_optim_bi_preserved, 'get_num_bundled_inputs') and
            hasattr(module_optim_bi_preserved, 'run_on_bundled_input')
        )

        # We do not expect an exception here
        module_optim_bi_preserved.run_on_bundled_input(0)

        bundled_input = module_optim_bi_preserved.get_all_bundled_inputs()[0]
        module_optim_bi_preserved(*bundled_input)

        # If not all 3 bundled inputs methods are present in the module,
        # we will not try to preserve them unless specified by the user.
        incomplete_bi_module = torch.jit.script(MyIncompleteBundledInputModule())
        incomplete_bi_module_optim = optimize_for_mobile(incomplete_bi_module)
        self.assertFalse(hasattr(incomplete_bi_module_optim, 'get_all_bundled_inputs'))

        # Specifically preserve get_all_bundled_inputs even if it's the only one
        # bundled inputs method available.
        incomplete_bi_module_optim = optimize_for_mobile(incomplete_bi_module, preserved_methods=['get_all_bundled_inputs'])
        self.assertTrue(hasattr(incomplete_bi_module_optim, 'get_all_bundled_inputs'))

    @unittest.skipUnless(torch.backends.xnnpack.enabled,
                         " XNNPACK must be enabled for these tests."
                         " Please build with USE_XNNPACK=1.")
    def test_hoist_conv_packed_params(self):

        if 'qnnpack' not in torch.backends.quantized.supported_engines:
            return

        class Standalone(nn.Module):
            def __init__(self):
                super(Standalone, self).__init__()
                self.quant = torch.quantization.QuantStub()
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.conv2 = nn.Conv2d(1, 1, 1)
                self.relu = nn.ReLU()
                self.dequant = torch.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.relu(x)
                x = self.dequant(x)
                return x

            def fuse_model(self):
                torch.quantization.fuse_modules(self, [['conv2', 'relu']], inplace=True)
                pass

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

            def fuse_model(self):
                pass

        with override_quantized_engine('qnnpack'):
            def _quant_script_and_optimize(model):
                model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
                model.fuse_model()
                torch.quantization.prepare(model, inplace=True)
                model(torch.randn(4, 1, 4, 4))
                torch.quantization.convert(model, inplace=True)
                model = torch.jit.script(model)
                model_optim = optimize_for_mobile(model)
                return model, model_optim

            # basic case

            m, m_optim = _quant_script_and_optimize(Standalone())
            FileCheck().check_not("Conv2d = prim::GetAttr[name=\"conv1\"]") \
                       .check_count("__torch__.torch.classes.quantized.Conv2dPackedParamsBase = prim::Constant", 2, exactly=True) \
                       .run(m_optim.graph)
            self.assertFalse(hasattr(m_optim, "conv1"))
            self.assertFalse(hasattr(m_optim, "conv2"))

            data = torch.randn(4, 1, 4, 4)
            m_res = m(data)
            m_optim_res = m_optim(data)
            torch.testing.assert_allclose(m_res, m_optim_res, rtol=1e-2, atol=1e-3)

            # generic case

            m, m_optim = _quant_script_and_optimize(Parent())
            FileCheck().check_not("Conv2d = prim::GetAttr[name=\"conv1\"]") \
                       .check_count("__torch__.torch.classes.quantized.Conv2dPackedParamsBase = prim::Constant", 2, exactly=True) \
                       .run(m_optim.graph)
            self.assertFalse(hasattr(m_optim, "conv1"))
            self.assertFalse(hasattr(m_optim, "child"))

            data = torch.randn(4, 1, 4, 4)
            m_res = m(data)
            m_optim_res = m_optim(data)
            torch.testing.assert_allclose(m_res, m_optim_res, rtol=1e-2, atol=1e-3)

    @unittest.skipUnless(HAS_TORCHVISION, "Needs torchvision")
    def test_mobilenet_optimize_for_mobile(self):
        m = torchvision.models.mobilenet_v3_small()
        m = torch.jit.script(m)
        m = optimize_for_mobile(m)

        # run forward 3 times until segfault, see https://github.com/pytorch/pytorch/issues/52463
        x = torch.zeros(1, 3, 56, 56)
        self.assertEqual(m(x).numel(), 1000)
        self.assertEqual(m(x).numel(), 1000)
        self.assertEqual(m(x).numel(), 1000)



if __name__ == '__main__':
    run_tests()
