from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn._intrinsic as nni
import torch.nn._intrinsic.quantized as nniq
import torch.nn._intrinsic.qat as nniqat
from torch.quantization import \
    QConfig_dynamic, default_weight_observer, \
    quantize, prepare, convert, prepare_qat, quantize_qat, fuse_modules, \
    quantize_dynamic, default_qconfig, default_qat_qconfig, \
    default_dynamic_qconfig, Observer

from common_utils import run_tests
from common_quantization import QuantizationTestCase, SingleLayerLinearModel, \
    SkipQuantModel, QuantStubModel, \
    ModelForFusion, ManualLinearQATModel, ManualConvLinearQATModel, \
    ModForWrapping, \
    test_only_eval_fn, test_only_train_fn, \
    prepare_dynamic, convert_dynamic, SingleLayerLinearDynamicModel, TwoLayerLinearModel, NestedModel

from common_quantization import AnnotatedTwoLayerLinearModel, AnnotatedNestedModel, \
    AnnotatedSubNestedModel, AnnotatedCustomConfigNestedModel

from hypothesis import given
from hypothesis import strategies as st


@unittest.skipIf(
    not torch.fbgemm_is_cpu_supported(),
    " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
    " with instruction set support avx2 or newer.",
)
class PostTrainingQuantTest(QuantizationTestCase):
    def test_single_layer(self):
        r"""Quantize SingleLayerLinearModel which has one Linear module, make sure it is swapped
        to nnq.Linear which is the quantized version of the module
        """
        model = SingleLayerLinearModel()
        model = prepare(model)
        # Check if observers and quant/dequant nodes are inserted
        self.checkNoPrepModules(model)
        self.checkHasPrepModules(model.fc1)
        self.checkObservers(model)

        test_only_eval_fn(model, self.calib_data)
        convert(model)

        def checkQuantized(model):
            self.checkNoPrepModules(model)
            self.checkHasPrepModules(model.fc1)
            self.checkWrappedQuantizedLinear(model.fc1)
            test_only_eval_fn(model, self.calib_data)
            self.checkScriptable(model, self.calib_data)

        checkQuantized(model)

        # test one line API
        model = quantize(SingleLayerLinearModel(), test_only_eval_fn, self.calib_data)
        checkQuantized(model)

    def test_two_layers(self):
        r"""TwoLayerLinearModel has two Linear modules but we only quantize the second one
        `fc2`, and `fc1`is not quantized
        """
        model = AnnotatedTwoLayerLinearModel()
        model = prepare(model)

        self.checkNoPrepModules(model)
        self.checkObservers(model)
        self.checkNoPrepModules(model.fc1)
        self.checkHasPrepModules(model.fc2)

        test_only_eval_fn(model, self.calib_data)
        convert(model)

        def checkQuantized(model):
            self.checkNoPrepModules(model)
            self.checkNoPrepModules(model.fc1)
            self.checkHasPrepModules(model.fc2)
            self.assertEqual(type(model.fc1), torch.nn.Linear)
            self.checkWrappedQuantizedLinear(model.fc2)
            test_only_eval_fn(model, self.calib_data)
            self.checkScriptable(model, self.calib_data)

        checkQuantized(model)

        # test one line API
        model = quantize(AnnotatedTwoLayerLinearModel(), test_only_eval_fn, self.calib_data)
        checkQuantized(model)

    def test_nested1(self):
        r"""Test quantization for nested model, top level 'fc3' and
        'fc1' of submodule 'sub2', 'sub2.fc2' is not quantized
        """
        model = AnnotatedNestedModel()

        def checkPrepModules(model, before_calib=False):
            if before_calib:
                self.checkObservers(model)
            self.checkNoPrepModules(model)
            self.checkNoPrepModules(model.sub1)
            self.checkNoPrepModules(model.sub1.fc)
            self.checkNoPrepModules(model.sub1.relu)
            self.checkNoPrepModules(model.sub2)
            self.checkHasPrepModules(model.sub2.fc1)
            self.checkNoPrepModules(model.sub2.fc2)
            self.checkHasPrepModules(model.fc3)

        model = prepare(model)
        checkPrepModules(model, True)
        test_only_eval_fn(model, self.calib_data)
        convert(model)

        def checkQuantized(model):
            checkPrepModules(model)
            self.checkLinear(model.sub1.fc)
            self.checkWrappedQuantizedLinear(model.fc3)
            self.checkWrappedQuantizedLinear(model.sub2.fc1)
            self.checkLinear(model.sub2.fc2)
            test_only_eval_fn(model, self.calib_data)
            self.checkScriptable(model, self.calib_data)

        checkQuantized(model)

        # test one line API
        model = quantize(AnnotatedNestedModel(), test_only_eval_fn, self.calib_data)
        checkQuantized(model)


    def test_nested2(self):
        model = AnnotatedSubNestedModel()
        model = prepare(model)

        def checkPrepModules(model, before_calib=False):
            if before_calib:
                self.checkObservers(model)
            self.checkNoPrepModules(model)
            self.checkNoPrepModules(model.sub1)
            self.checkNoPrepModules(model.sub1.fc)
            self.checkNoPrepModules(model.sub1.relu)
            self.checkHasPrepModules(model.sub2)
            self.checkNoPrepModules(model.sub2.module.fc1)
            self.checkNoPrepModules(model.sub2.module.fc2)
            self.checkHasPrepModules(model.fc3)

        checkPrepModules(model, True)

        test_only_eval_fn(model, self.calib_data)
        convert(model)

        def checkQuantized(model):
            checkPrepModules(model)
            self.checkLinear(model.sub1.fc)
            self.assertEqual(type(model.sub1.relu), torch.nn.ReLU)
            self.checkQuantizedLinear(model.sub2.module.fc1)
            self.checkQuantizedLinear(model.sub2.module.fc2)
            self.checkWrappedQuantizedLinear(model.fc3)
            test_only_eval_fn(model, self.calib_data)
            self.checkScriptable(model, self.calib_data)

        checkQuantized(model)

        # test one line API
        model = quantize(AnnotatedSubNestedModel(), test_only_eval_fn, self.calib_data)
        checkQuantized(model)

    def test_nested3(self):
        r"""More complicated nested test case with child qconfig overrides
        parent qconfig
        """
        model = AnnotatedCustomConfigNestedModel()
        model = prepare(model)

        def checkPrepModules(model, before_calib=False):
            if before_calib:
                self.checkObservers(model)
            self.checkNoPrepModules(model)
            self.checkNoPrepModules(model.sub1)
            self.checkNoPrepModules(model.sub1.fc)
            self.checkNoPrepModules(model.sub1.relu)
            self.checkNoPrepModules(model.sub2)
            self.checkHasPrepModules(model.sub2.fc1)
            self.checkHasPrepModules(model.sub2.fc2)
            self.checkHasPrepModules(model.fc3)

        checkPrepModules(model, True)

        test_only_eval_fn(model, self.calib_data)
        convert(model)

        def checkQuantized(model):
            checkPrepModules(model)
            self.checkWrappedQuantizedLinear(model.sub2.fc1)
            self.checkWrappedQuantizedLinear(model.sub2.fc2)
            self.checkWrappedQuantizedLinear(model.fc3)
            test_only_eval_fn(model, self.calib_data)
            self.checkScriptable(model, self.calib_data)

        checkQuantized(model)

        # test one line API
        model = quantize(AnnotatedCustomConfigNestedModel(), test_only_eval_fn, self.calib_data)
        checkQuantized(model)

    def test_skip_quant(self):
        r"""The case when we want to skip quantizing some layers
        """

        model = SkipQuantModel()
        prepare(model)
        self.checkObservers(model)

        test_only_eval_fn(model, self.calib_data)
        convert(model)

        def checkQuantized(model):
            self.checkLinear(model.fc)
            self.checkQuantDequant(model.sub)
            self.checkQuantizedLinear(model.sub.module.fc1)
            self.checkQuantizedLinear(model.sub.module.fc2)
            self.assertEqual(type(model.sub.module.relu), nnq.ReLU)
            self.checkScriptable(model, self.calib_data)

        checkQuantized(model)

        # test one line API
        model = quantize(SkipQuantModel(), test_only_eval_fn, self.calib_data)
        checkQuantized(model)


    def test_manual(self):
        r"""User inserts QuantStub and DeQuantStub in model code
        and call the quantization utility functions.
        """
        model = QuantStubModel()
        # propagate the qconfig of parents to children, model is changed
        # inplace
        prepare(model)
        self.checkObservers(model)

        test_only_eval_fn(model, self.calib_data)
        convert(model)

        def checkQuantized(model):
            self.assertEqual(type(model.fc), nnq.Linear)
            test_only_eval_fn(model, self.calib_data)
            self.checkScriptable(model, self.calib_data)

        checkQuantized(model)

        # test one line API
        model = quantize(QuantStubModel(), test_only_eval_fn, self.calib_data)
        checkQuantized(model)


@unittest.skipIf(
    not torch.fbgemm_is_cpu_supported(),
    " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
    " with instruction set support avx2 or newer.",
)
class PostTrainingDynamicQuantTest(QuantizationTestCase):
    def test_single_layer(self):
        r"""Dynamic Quantize SingleLayerLinearDynamicModel which has one Linear module,
        make sure it is swapped to nnqd.Linear which is the quantized version of
        the module
        """
        model = SingleLayerLinearDynamicModel().eval()
        qconfig_dict = {
            '': default_dynamic_qconfig
        }
        model = prepare_dynamic(model, qconfig_dict)
        convert_dynamic(model)

        def checkQuantized(model):
            self.checkDynamicQuantizedLinear(model.fc1)

        checkQuantized(model)

        # test one line API
        model = quantize_dynamic(SingleLayerLinearDynamicModel().eval(), qconfig_dict)
        checkQuantized(model)

    def test_two_layers(self):
        r"""TwoLayerLinearModel has two Linear modules but we only quantize the second one
        `fc2`, and `fc1`is not quantized
        """
        model = TwoLayerLinearModel().eval()
        qconfig_dict = {
            'fc2': default_dynamic_qconfig
        }
        model = prepare_dynamic(model, qconfig_dict)

        convert_dynamic(model)

        def checkQuantized(model):
            self.assertEqual(type(model.fc1), torch.nn.Linear)
            self.checkDynamicQuantizedLinear(model.fc2)

        checkQuantized(model)

        # test one line API
        model = quantize_dynamic(TwoLayerLinearModel().eval(), qconfig_dict)
        checkQuantized(model)

    def test_nested1(self):
        r"""Test quantization for nested model, top level 'fc3' and
        'fc1' of submodule 'sub2', 'sub2.fc2' is not quantized
        """
        model = NestedModel().eval()
        qconfig_dict = {
            'fc3': default_dynamic_qconfig,
            'sub2.fc1': default_dynamic_qconfig
        }

        model = prepare_dynamic(model, qconfig_dict)
        convert_dynamic(model)

        def checkQuantized(model):
            self.checkLinear(model.sub1.fc)
            self.checkDynamicQuantizedLinear(model.fc3)
            self.checkDynamicQuantizedLinear(model.sub2.fc1)
            self.checkLinear(model.sub2.fc2)

        checkQuantized(model)

        # test one line API
        model = quantize_dynamic(NestedModel().eval(), qconfig_dict)
        checkQuantized(model)

    def test_nested2(self):
        r"""Another test case for quantized, we will quantize all submodules
        of submodule sub2
        """
        model = NestedModel().eval()
        qconfig_dict = {
            'fc3': default_dynamic_qconfig,
            'sub2': default_dynamic_qconfig
        }
        model = prepare_dynamic(model, qconfig_dict)

        convert_dynamic(model)

        def checkQuantized(model):
            self.checkLinear(model.sub1.fc)
            self.assertEqual(type(model.sub1.relu), torch.nn.ReLU)
            self.checkDynamicQuantizedLinear(model.sub2.fc1)
            self.checkDynamicQuantizedLinear(model.sub2.fc2)
            self.checkDynamicQuantizedLinear(model.fc3)

        checkQuantized(model)

        # test one line API
        model = quantize_dynamic(NestedModel().eval(), qconfig_dict)
        checkQuantized(model)

    def test_nested3(self):
        r"""More complicated nested test case with child qconfig overrides
        parent qconfig
        """
        model = NestedModel().eval()
        custum_options = {
            'dtype': torch.quint8,
            'qscheme': torch.per_tensor_affine
        }
        custom_dynamic_qconfig = QConfig_dynamic(weight=default_weight_observer())
        qconfig_dynamic_dict = {
            'fc3': default_dynamic_qconfig,
            'sub2': default_dynamic_qconfig,
            'sub2.fc1': custom_dynamic_qconfig
        }
        model = prepare_dynamic(model, qconfig_dynamic_dict)

        convert_dynamic(model)

        def checkQuantized(model):
            self.checkDynamicQuantizedLinear(model.sub2.fc1)
            self.checkDynamicQuantizedLinear(model.sub2.fc2)
            self.checkDynamicQuantizedLinear(model.fc3)

        checkQuantized(model)

        # test one line API
        model = quantize_dynamic(NestedModel().eval(), qconfig_dynamic_dict)
        checkQuantized(model)

    def test_type_match_rule(self):
        r"""Test quantization for nested model, top level 'fc3' and
        'fc1' of submodule 'sub2', All 'torch.nn.Linear' modules are quantized
        """
        model = NestedModel().eval()
        qconfig_dict = {
            'fc3': None,
            'sub2.fc1': None,
            torch.nn.Linear: default_dynamic_qconfig
        }

        model = prepare_dynamic(model, qconfig_dict)
        test_only_eval_fn(model, self.calib_data)
        convert_dynamic(model)

        def checkQuantized(model):
            self.checkDynamicQuantizedLinear(model.sub1.fc)
            self.checkLinear(model.fc3)
            self.checkLinear(model.sub2.fc1)
            self.checkDynamicQuantizedLinear(model.sub2.fc2)
            test_only_eval_fn(model, self.calib_data)

        checkQuantized(model)

        # test one line API
        model = quantize_dynamic(NestedModel().eval(), qconfig_dict)
        checkQuantized(model)


@unittest.skipIf(
    not torch.fbgemm_is_cpu_supported(),
    " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
    " with instruction set support avx2 or newer.",
)
class QuantizationAwareTrainingTest(QuantizationTestCase):
    def test_manual(self):
        model = ManualLinearQATModel()
        model = prepare_qat(model)
        self.checkObservers(model)
        test_only_train_fn(model, self.train_data)
        convert(model)

        def checkQuantized(model):
            self.assertEqual(type(model.fc1), nnq.Linear)
            self.assertEqual(type(model.fc2), nnq.Linear)
            test_only_eval_fn(model, self.calib_data)
            self.checkScriptable(model, self.calib_data)

        checkQuantized(model)

        model = quantize_qat(ManualLinearQATModel(), test_only_train_fn, self.train_data)
        checkQuantized(model)

    def test_eval_only_fake_quant(self):
        r"""Using FakeQuant in evaluation only mode,
        this is useful for estimating accuracy loss when we quantize the
        network
        """
        model = ManualLinearQATModel()

        model = prepare_qat(model)
        self.checkObservers(model)

        model.eval()
        test_only_eval_fn(model, self.calib_data)

    def test_conv_linear(self):
        model = ManualConvLinearQATModel()

        model = prepare_qat(model)
        self.checkObservers(model)

        test_only_train_fn(model, self.img_data)
        convert(model)

        def checkQuantized(model):
            self.assertEqual(type(model.conv), nnq.Conv2d)
            self.assertEqual(type(model.fc1), nnq.Linear)
            self.assertEqual(type(model.fc2), nnq.Linear)
            test_only_eval_fn(model, self.img_data)
            self.checkScriptable(model, self.img_data)

        checkQuantized(model)

        model = ManualConvLinearQATModel()
        model = quantize_qat(model, test_only_train_fn, self.img_data)
        checkQuantized(model)


class ScriptabilityTest(QuantizationTestCase):
    def setUp(self):
        self.model_under_test = ModForWrapping(quantized=False)
        self.qmodel_under_test = ModForWrapping(quantized=True)
        self.qmodel_under_test = self.qmodel_under_test.from_float(
            self.model_under_test)
        self.x = torch.rand(10)
        self.qx = torch.quantize_linear(self.x.to(torch.float), scale=1.0,
                                        zero_point=0, dtype=torch.qint32)

    def test_quantized(self):
        qtraced_model = torch.jit.trace(self.qmodel_under_test, self.qx,
                                        check_trace=False)
        self.assertEqual(qtraced_model(self.qx), self.qmodel_under_test(self.qx))

        qscripted_model = torch.jit.script(self.qmodel_under_test)
        self.assertEqual(qscripted_model(self.qx), self.qmodel_under_test(self.qx))

    def test_float(self):
        traced_model = torch.jit.trace(self.model_under_test, self.x,
                                       check_trace=False)
        self.assertEqual(traced_model(self.x), self.model_under_test(self.x))

        scripted_model = torch.jit.script(self.model_under_test)
        self.assertEqual(scripted_model(self.x), self.model_under_test(self.x))


@unittest.skipIf(not torch.fbgemm_is_cpu_supported(),
                 'Quantization requires FBGEMM. FBGEMM does not play'
                 ' well with UBSAN at the moment, so we skip the test if'
                 ' we are in a UBSAN environment.')
class FusionTest(QuantizationTestCase):
    def test_fuse_module_train(self):
        model = ModelForFusion(default_qat_qconfig).train()
        fuse_modules(model, [['conv1', 'bn1', 'relu1'],
                             ['sub1.conv', 'sub1.bn']])
        self.assertEqual(type(model.conv1), nni.ConvBnReLU2d,
                         "Fused Conv + BN + Relu first layer")
        self.assertEqual(type(model.bn1), torch.nn.Identity,
                         "Fused Conv + BN + Relu (skipped BN)")
        self.assertEqual(type(model.relu1), torch.nn.Identity,
                         "Fused Conv + BN + Relu (skipped Relu)")

        self.assertEqual(type(model.sub1.conv), nni.ConvBn2d,
                         "Fused submodule Conv + BN")
        self.assertEqual(type(model.sub1.bn), torch.nn.Identity,
                         "Fused submodule Conv + BN (skipped BN)")
        self.assertEqual(type(model.sub2.conv), torch.nn.Conv2d,
                         "Non-fused submodule Conv")
        self.assertEqual(type(model.sub2.relu), torch.nn.ReLU,
                         "Non-fused submodule ReLU")
        model = prepare_qat(model)
        self.checkObservers(model)

        def checkQAT(model):
            self.assertEqual(type(model.conv1), nniqat.ConvBnReLU2d)
            self.assertEqual(type(model.bn1), nn.Identity)
            self.assertEqual(type(model.relu1), nn.Identity)
            self.assertEqual(type(model.sub1.conv), nniqat.ConvBn2d)
            self.assertEqual(type(model.sub1.bn), nn.Identity)
            self.assertEqual(type(model.sub2.conv), nn.Conv2d)
            self.assertEqual(type(model.sub2.relu), nn.ReLU)

        checkQAT(model)
        test_only_train_fn(model, self.img_data)
        convert(model)

        def checkQuantized(model):
            self.assertEqual(type(model.conv1), nniq.ConvReLU2d)
            self.assertEqual(type(model.bn1), nn.Identity)
            self.assertEqual(type(model.relu1), nn.Identity)
            self.assertEqual(type(model.sub1.conv), nnq.Conv2d)
            self.assertEqual(type(model.sub1.bn), nn.Identity)
            self.assertEqual(type(model.sub2.conv), nn.Conv2d)
            self.assertEqual(type(model.sub2.relu), nn.ReLU)
            test_only_eval_fn(model, self.img_data)
        checkQuantized(model)

        model = ModelForFusion(default_qat_qconfig).train()
        fuse_modules(model, [['conv1', 'bn1', 'relu1'],
                             ['sub1.conv', 'sub1.bn']])
        model = quantize_qat(model, test_only_train_fn, self.img_data)
        checkQuantized(model)


    def test_fuse_module_eval(self):
        model = ModelForFusion(default_qconfig)
        model.eval()
        fuse_modules(model, [['conv1', 'bn1', 'relu1'] ,
                             ['sub1.conv', 'sub1.bn']])
        self.assertEqual(type(model.conv1), nni.ConvReLU2d,
                         "Fused Conv + BN + Relu first layer (BN is folded)")
        self.assertEqual(type(model.conv1[0]), nn.Conv2d,
                         "Fused Conv + BN + Relu (Conv + folded BN only)")
        self.assertEqual(type(model.conv1[1]), nn.ReLU,
                         "Fused Conv + BN + Relu second layer (Relu only)")
        self.assertEqual(type(model.bn1), nn.Identity,
                         "Fused Conv + BN + Relu second layer (Skipped BN)")
        self.assertEqual(type(model.relu1), nn.Identity,
                         "Fused Conv + BN + Relu second layer (Skipped Relu)")

        self.assertEqual(type(model.sub1.conv), nn.Conv2d,
                         "Fused submodule Conv + folded BN")
        self.assertEqual(type(model.sub1.bn), nn.Identity,
                         "Fused submodule (skipped BN)")
        self.assertEqual(type(model.sub2.conv), nn.Conv2d,
                         "Non-fused submodule Conv")
        self.assertEqual(type(model.sub2.relu), torch.nn.ReLU,
                         "Non-fused submodule ReLU")

        model = prepare(model)
        self.checkObservers(model)
        test_only_eval_fn(model, self.img_data)
        convert(model)

        def checkQuantized(model):
            self.assertEqual(type(model.conv1), nniq.ConvReLU2d)
            self.assertEqual(type(model.bn1), nn.Identity)
            self.assertEqual(type(model.relu1), nn.Identity)
            self.assertEqual(type(model.sub1.conv), nnq.Conv2d)
            self.assertEqual(type(model.sub1.bn), nn.Identity)
            self.assertEqual(type(model.sub2.conv), nn.Conv2d)
            self.assertEqual(type(model.sub2.relu), nn.ReLU)
            test_only_eval_fn(model, self.img_data)
        checkQuantized(model)

        model = ModelForFusion(default_qat_qconfig).eval()
        fuse_modules(model, [['conv1', 'bn1', 'relu1'],
                             ['sub1.conv', 'sub1.bn']])
        model = quantize(model, test_only_eval_fn, self.img_data)
        checkQuantized(model)

class ObserverTest(QuantizationTestCase):
    @given(qdtype=st.sampled_from((torch.qint8, torch.quint8)),
           qscheme=st.sampled_from((torch.per_tensor_affine, torch.per_tensor_symmetric)))
    def test_observer(self, qdtype, qscheme):
        myobs = Observer(dtype=qdtype, qscheme=qscheme)
        x = torch.tensor([1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y = torch.tensor([4.0, 5.0, 5.0, 6.0, 7.0, 8.0])
        result = myobs(x)
        result = myobs(y)
        self.assertEqual(result, y)
        self.assertEqual(myobs.min_val, 1.0)
        self.assertEqual(myobs.max_val, 8.0)
        qparams = myobs.calculate_qparams()
        if qscheme == torch.per_tensor_symmetric:
            ref_scale = 0.062745
            ref_zero_point = 0 if qdtype is torch.qint8 else 128
        else:
            ref_scale = 0.0313725
            ref_zero_point = -128 if qdtype is torch.qint8 else 0
        self.assertEqual(qparams[1].item(), ref_zero_point)
        self.assertAlmostEqual(qparams[0].item(), ref_scale, delta=1e-5)



if __name__ == '__main__':
    run_tests()
