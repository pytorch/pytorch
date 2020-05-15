import unittest
import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nniq
import torch.nn.intrinsic.qat as nniqat
from torch.nn.utils.rnn import PackedSequence
from torch.quantization import (
    quantize,
    prepare,
    convert,
    prepare_qat,
    quantize_qat,
    fuse_modules,
    quantize_dynamic,
    QuantWrapper,
    QConfig,
    default_qconfig,
    default_per_channel_qconfig,
    default_qat_qconfig,
    default_dynamic_qconfig,
    per_channel_dynamic_qconfig,
    default_eval_fn,
    float16_dynamic_qconfig,
    default_observer,
    default_weight_observer,
    default_per_channel_weight_observer,
    default_histogram_observer,
)

from torch.quantization._quantize_script import (
    quantize_script,
    quantize_dynamic_script
)

from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    AnnotatedSingleLayerLinearModel,
    SingleLayerLinearModel,
    AnnotatedConvModel,
    ConvModel,
    AnnotatedConvBnModel,
    ConvBnModel,
    SkipQuantModel,
    QuantStubModel,
    ModelForFusion,
    ModelWithSequentialFusion,
    ManualLinearQATModel,
    ManualConvLinearQATModel,
    ModelWithFunctionals,
    ModelMultipleOps,
    ModelMultipleOpsNoAvgPool,
    SingleLayerLinearDynamicModel,
    TwoLayerLinearModel,
    NestedModel,
    ResNetBase,
    LSTMDynamicModel,
    ModelForFusionWithBias,
    ActivationsTestModel,
    ActivationsQATTestModel,
    NormalizationTestModel,
    test_only_eval_fn,
    test_only_train_fn,
    prepare_dynamic,
    convert_dynamic,
    skipIfNoFBGEMM,
)

# annotated models
from torch.testing._internal.common_quantization import (
    AnnotatedTwoLayerLinearModel,
    AnnotatedNestedModel,
    AnnotatedSubNestedModel,
    AnnotatedCustomConfigNestedModel,
    AnnotatedSkipQuantModel,
)

from torch.testing._internal.common_quantized import (
    override_quantized_engine,
    supported_qengines,
)
from hypothesis import given
from hypothesis import strategies as st
import torch.testing._internal.hypothesis_utils as hu
hu.assert_deadline_disabled()
import io
import copy

class TestPostTrainingStatic(QuantizationTestCase):
    def test_single_layer(self):
        r"""Quantize SingleLayerLinearModel which has one Linear module, make sure it is swapped
        to nnq.Linear which is the quantized version of the module
        """
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                qconfig = torch.quantization.get_default_qconfig(qengine)
                model = AnnotatedSingleLayerLinearModel(qengine)
                model.qconfig = qconfig
                model = prepare(model)
                # Check if observers and quant/dequant nodes are inserted
                self.checkNoPrepModules(model)
                self.checkHasPrepModules(model.fc1)
                self.checkObservers(model)

                test_only_eval_fn(model, self.calib_data)
                model = convert(model)

                def checkQuantized(model):
                    self.checkNoPrepModules(model)
                    self.checkHasPrepModules(model.fc1)
                    self.checkWrappedQuantizedLinear(model.fc1)
                    test_only_eval_fn(model, self.calib_data)
                    self.checkScriptable(model, self.calib_data)

                checkQuantized(model)

                # test one line API - out of place version
                base = AnnotatedSingleLayerLinearModel(qengine)
                base.qconfig = qconfig
                keys_before = set(list(base.state_dict().keys()))
                model = quantize(base, test_only_eval_fn, self.calib_data)
                checkQuantized(model)
                keys_after = set(list(base.state_dict().keys()))
                self.assertEqual(keys_before, keys_after)  # simple check that nothing changed

                # in-place version
                model = AnnotatedSingleLayerLinearModel(qengine)
                model.qconfig = qconfig
                quantize(model, test_only_eval_fn, self.calib_data, inplace=True)
                checkQuantized(model)

    @skipIfNoFBGEMM
    def test_two_layers(self):
        r"""TwoLayerLinearModel has two Linear modules but we only quantize the second one
        `fc2`, and `fc1`is not quantized
        """
        with override_quantized_engine('fbgemm'):
            model = AnnotatedTwoLayerLinearModel()
            model = prepare(model)

            self.checkNoPrepModules(model)
            self.checkObservers(model)
            self.checkNoPrepModules(model.fc1)
            self.checkHasPrepModules(model.fc2)

            test_only_eval_fn(model, self.calib_data)
            model = convert(model)

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
            model = quantize(AnnotatedTwoLayerLinearModel(), test_only_eval_fn,
                             self.calib_data)
            checkQuantized(model)

    def test_nested1(self):
        r"""Test quantization for nested model, top level 'fc3' and
        'fc1' of submodule 'sub2', 'sub2.fc2' is not quantized
        """
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = AnnotatedNestedModel(qengine)

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
                model = convert(model)

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
                model = quantize(AnnotatedNestedModel(qengine), test_only_eval_fn,
                                 self.calib_data)
                checkQuantized(model)


    @skipIfNoFBGEMM
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
        model = convert(model)

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
        model = quantize(AnnotatedSubNestedModel(), test_only_eval_fn,
                         self.calib_data)
        checkQuantized(model)

    def test_nested3(self):
        r"""More complicated nested test case with child qconfig overrides
        parent qconfig
        """
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
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
                model = convert(model)

                def checkQuantized(model):
                    checkPrepModules(model)
                    self.checkWrappedQuantizedLinear(model.sub2.fc1)
                    self.checkWrappedQuantizedLinear(model.sub2.fc2)
                    self.checkWrappedQuantizedLinear(model.fc3)
                    test_only_eval_fn(model, self.calib_data)
                    self.checkScriptable(model, self.calib_data)

                checkQuantized(model)

                # test one line API
                model = quantize(AnnotatedCustomConfigNestedModel(), test_only_eval_fn,
                                 self.calib_data)
                checkQuantized(model)

    def test_skip_quant(self):
        r"""The case when we want to skip quantizing some layers
        """
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = AnnotatedSkipQuantModel(qengine)
                model = prepare(model)
                self.checkObservers(model)

                test_only_eval_fn(model, self.calib_data)
                model = convert(model)

                def checkQuantized(model):
                    self.checkLinear(model.fc)
                    self.checkQuantDequant(model.sub)
                    self.checkQuantizedLinear(model.sub.module.fc1)
                    self.checkQuantizedLinear(model.sub.module.fc2)
                    self.assertEqual(type(model.sub.module.relu1), nnq.ReLU)
                    self.assertEqual(type(model.sub.module.relu2), nnq.ReLU)
                    self.checkScriptable(model, self.calib_data)

                checkQuantized(model)

                # test one line API
                model = quantize(AnnotatedSkipQuantModel(qengine), test_only_eval_fn, self.calib_data)
                checkQuantized(model)

    @skipIfNoFBGEMM
    def test_manual(self):
        r"""User inserts QuantStub and DeQuantStub in model code
        and call the quantization utility functions.
        """
        model = QuantStubModel()
        # propagate the qconfig of parents to children, model is changed
        # inplace
        model = prepare(model)
        self.checkObservers(model)

        test_only_eval_fn(model, self.calib_data)
        model = convert(model)

        def checkQuantized(model):
            self.assertEqual(type(model.fc), nnq.Linear)
            test_only_eval_fn(model, self.calib_data)
            self.checkScriptable(model, self.calib_data)

        checkQuantized(model)

        # test one line API
        model = quantize(QuantStubModel(), test_only_eval_fn, self.calib_data)
        checkQuantized(model)

    def test_resnet_base(self):
        r"""Test quantization for bottleneck topology used in resnet/resnext
        and add coverage for conversion of average pool and float functional
        """
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                qconfig = torch.quantization.get_default_qconfig(qengine)
                model = ResNetBase().float().eval()
                model = QuantWrapper(model)
                model.qconfig = qconfig
                fuse_list = ['module.conv1', 'module.bn1', 'module.relu1']
                fuse_modules(model, fuse_list, inplace=True)
                model = prepare(model)
                self.checkObservers(model)
                test_only_eval_fn(model, self.img_data)
                model = convert(model)

                def checkQuantized(model):
                    self.assertEqual(type(model.module.conv1), nn.intrinsic.quantized.ConvReLU2d)
                    self.assertEqual(type(model.module.myop), nn.quantized.QFunctional)
                    self.assertEqual(type(model.module.avgpool), nn.AdaptiveAvgPool2d)
                    test_only_eval_fn(model, self.img_data)

                checkQuantized(model)

    @skipIfNoFBGEMM
    def test_normalization(self):
        r"""
        Test quantization of normalization layers
        """
        model = NormalizationTestModel()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        prepare(model, inplace=True)
        self.checkObservers(model)
        test_only_eval_fn(model, self.calib_data)
        model = convert(model)

        def checkQuantized(model):
            self.checkNoPrepModules(model.layer_norm)
            self.assertEqual(type(model.layer_norm), nnq.LayerNorm)
            test_only_eval_fn(model, self.calib_data)
            self.checkScriptable(model, self.calib_data)

        checkQuantized(model)

        model_oneline = quantize(
            NormalizationTestModel(), test_only_eval_fn, self.calib_data)
        checkQuantized(model)

    def test_save_load_state_dict(self):
        r"""Test PTQ flow of creating a model and quantizing it and saving the quantized state_dict
        Load the quantized state_dict for eval and compare results against original model
        """

        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = TwoLayerLinearModel()
                model = torch.quantization.QuantWrapper(model)
                model.qconfig = torch.quantization.get_default_qconfig(qengine)

                model = prepare(model)
                # calibrate
                test_only_eval_fn(model, self.calib_data)
                model = convert(model)
                x = torch.rand(2, 5, dtype=torch.float)
                ref = model(x)

                quant_state_dict = model.state_dict()

                # Create model again for eval
                model = TwoLayerLinearModel()
                model = torch.quantization.QuantWrapper(model)
                model.qconfig = torch.quantization.get_default_qconfig(qengine)
                model = prepare(model)
                model = convert(model)
                new_state_dict = model.state_dict()

                # Check to make sure the state dict keys match original model after convert.
                self.assertEqual(set(new_state_dict.keys()), set(quant_state_dict.keys()))

                model.load_state_dict(quant_state_dict)

                out = model(x)
                self.assertEqual(ref, out)

    @skipIfNoFBGEMM
    def test_activations(self):
        r"""
        Test quantization of activations
        """
        model = ActivationsTestModel()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        prepare(model, inplace=True)
        self.checkObservers(model)
        test_only_eval_fn(model, self.calib_data)
        model = convert(model)

        def checkQuantized(model):
            self.checkNoPrepModules(model.hardswish)
            self.assertEqual(type(model.hardswish), nnq.Hardswish)
            test_only_eval_fn(model, self.calib_data)
            self.checkScriptable(model, self.calib_data)

        checkQuantized(model)

        # test one line API
        model_oneline = quantize(ActivationsTestModel(), test_only_eval_fn,
                                 self.calib_data)
        checkQuantized(model_oneline)


@skipIfNoFBGEMM
class TestPostTrainingDynamic(QuantizationTestCase):
    def test_single_layer(self):
        r"""Dynamic Quantize SingleLayerLinearDynamicModel which has one Linear module,
        make sure it is swapped to nnqd.Linear which is the quantized version of
        the module
        """
        for dtype in [torch.qint8, torch.float16]:
            model = SingleLayerLinearDynamicModel().eval()
            qconfig = float16_dynamic_qconfig if dtype == torch.float16 else default_dynamic_qconfig
            qconfig_dict = {
                'fc1': qconfig
            }
            prepare_dynamic(model, qconfig_dict)
            convert_dynamic(model)

            def checkQuantized(model):
                self.checkDynamicQuantizedLinear(model.fc1, dtype)
                self.checkScriptable(model, self.calib_data, check_save_load=True)

            checkQuantized(model)

            # test one line API - out of place version
            base = SingleLayerLinearDynamicModel()
            keys_before = set(list(base.state_dict().keys()))
            model = quantize_dynamic(base, qconfig_dict)
            checkQuantized(model)
            keys_after = set(list(base.state_dict().keys()))
            self.assertEqual(keys_before, keys_after)  # simple check that nothing changed

            # in-place version
            model = SingleLayerLinearDynamicModel()
            quantize_dynamic(model, qconfig_dict, inplace=True)
            checkQuantized(model)

            # Test set qconfig
            model = SingleLayerLinearDynamicModel()
            quantize_dynamic(model, set([nn.Linear]), inplace=True, dtype=dtype)
            checkQuantized(model)

    def test_two_layers(self):
        r"""TwoLayerLinearModel has two Linear modules but we only quantize the second one
        `fc2`, and `fc1`is not quantized
        """
        for dtype in [torch.qint8, torch.float16]:
            model = TwoLayerLinearModel().eval()
            qconfig = float16_dynamic_qconfig if dtype == torch.float16 else default_dynamic_qconfig
            qconfig_dict = {
                'fc2': qconfig
            }
            prepare_dynamic(model, qconfig_dict)

            convert_dynamic(model)

            def checkQuantized(model):
                self.assertEqual(type(model.fc1), torch.nn.Linear)
                self.checkDynamicQuantizedLinear(model.fc2, dtype=dtype)
                self.checkScriptable(model, self.calib_data, check_save_load=True)

            checkQuantized(model)

            # test one line API
            model = quantize_dynamic(TwoLayerLinearModel().eval(), qconfig_dict)
            checkQuantized(model)

            # Test set API
            model = quantize_dynamic(TwoLayerLinearModel().eval(), {'fc2'}, dtype=dtype)
            checkQuantized(model)

    def test_nested1(self):
        r"""Test quantization for nested model, top level 'fc3' and
        'fc1' of submodule 'sub2', 'sub2.fc2' is not quantized
        """
        for dtype in [torch.qint8, torch.float16]:
            model = NestedModel().eval()
            qconfig = float16_dynamic_qconfig if dtype == torch.float16 else default_dynamic_qconfig
            qconfig_dict = {
                'fc3': qconfig,
                'sub2.fc1': qconfig
            }

            prepare_dynamic(model, qconfig_dict)
            convert_dynamic(model)

            def checkQuantized(model):
                self.checkLinear(model.sub1.fc)
                self.checkDynamicQuantizedLinear(model.fc3, dtype=dtype)
                self.checkDynamicQuantizedLinear(model.sub2.fc1, dtype=dtype)
                self.checkLinear(model.sub2.fc2)
                self.checkScriptable(model, self.calib_data, check_save_load=True)

            checkQuantized(model)

            # test one line API
            model = quantize_dynamic(NestedModel().eval(), qconfig_dict)
            checkQuantized(model)

            model = quantize_dynamic(NestedModel().eval(), {'fc3', 'sub2.fc1'}, dtype=dtype)
            checkQuantized(model)

    def test_nested2(self):
        r"""Another test case for quantized, we will quantize all submodules
        of submodule sub2
        """
        for dtype in [torch.qint8, torch.float16]:
            model = NestedModel().eval()
            qconfig = float16_dynamic_qconfig if dtype == torch.float16 else default_dynamic_qconfig
            qconfig_dict = {
                'fc3': qconfig,
                'sub2': qconfig
            }
            prepare_dynamic(model, qconfig_dict)

            convert_dynamic(model)

            def checkQuantized(model):
                self.checkLinear(model.sub1.fc)
                self.assertEqual(type(model.sub1.relu), torch.nn.ReLU)
                self.checkDynamicQuantizedLinear(model.sub2.fc1, dtype=dtype)
                self.checkDynamicQuantizedLinear(model.sub2.fc2, dtype=dtype)
                self.checkDynamicQuantizedLinear(model.fc3, dtype=dtype)
                self.checkScriptable(model, self.calib_data, check_save_load=True)

            checkQuantized(model)

            # test one line API
            model = quantize_dynamic(NestedModel().eval(), qconfig_dict, dtype=dtype)
            checkQuantized(model)

            # Test set API
            model = quantize_dynamic(NestedModel().eval(), {'fc3', 'sub2'}, dtype=dtype)
            checkQuantized(model)

    def test_nested3(self):
        r"""More complicated nested test case with child qconfig overrides
        parent qconfig
        """
        for dtype in [torch.qint8, torch.float16]:
            model = NestedModel().eval()
            qconfig = float16_dynamic_qconfig if dtype == torch.float16 else default_dynamic_qconfig
            qconfig_dynamic_dict = {
                'fc3': qconfig,
                'sub2': qconfig,
                'sub2.fc1': qconfig
            }
            prepare_dynamic(model, qconfig_dynamic_dict)

            convert_dynamic(model)

            def checkQuantized(model):
                self.checkDynamicQuantizedLinear(model.sub2.fc1, dtype=dtype)
                self.checkDynamicQuantizedLinear(model.sub2.fc2, dtype=dtype)
                self.checkDynamicQuantizedLinear(model.fc3, dtype=dtype)
                self.checkScriptable(model, self.calib_data, check_save_load=True)

            checkQuantized(model)

            # test one line API
            model = quantize_dynamic(NestedModel().eval(), qconfig_dynamic_dict)
            checkQuantized(model)

            # Test set API
            model = quantize_dynamic(NestedModel().eval(), {'fc3', 'sub2', 'sub2.fc1'}, dtype=dtype)
            checkQuantized(model)

    def test_type_match_rule(self):
        r"""Test quantization for nested model, top level 'fc3' and
        'fc1' of submodule 'sub2', All 'torch.nn.Linear' modules are quantized
        """
        for dtype in [torch.qint8, torch.float16]:
            model = NestedModel().eval()
            qconfig = float16_dynamic_qconfig if dtype == torch.float16 else default_dynamic_qconfig
            qconfig_dict = {
                'fc3': None,
                'sub2.fc1': None,
                torch.nn.Linear: qconfig
            }

            prepare_dynamic(model, qconfig_dict)
            test_only_eval_fn(model, self.calib_data)
            convert_dynamic(model)

            def checkQuantized(model):
                self.checkDynamicQuantizedLinear(model.sub1.fc, dtype=dtype)
                self.checkLinear(model.fc3)
                self.checkLinear(model.sub2.fc1)
                self.checkDynamicQuantizedLinear(model.sub2.fc2, dtype=dtype)
                test_only_eval_fn(model, self.calib_data)
                self.checkScriptable(model, self.calib_data, check_save_load=True)

            checkQuantized(model)

            # test one line API
            model = quantize_dynamic(NestedModel().eval(), qconfig_dict, dtype=dtype)
            checkQuantized(model)

    def test_per_channel_quantize(self):
        r"""Test quantization for per_channel dynamic quantization
        """
        model = NestedModel().eval()
        qconfig_dict = {
            torch.nn.Linear: per_channel_dynamic_qconfig
        }

        prepare_dynamic(model, qconfig_dict)
        test_only_eval_fn(model, self.calib_data)
        convert_dynamic(model)

        def checkQuantized(model):
            self.checkDynamicQuantizedLinear(model.sub1.fc, dtype=torch.qint8)
            self.checkDynamicQuantizedLinear(model.fc3, dtype=torch.qint8)
            self.checkDynamicQuantizedLinear(model.sub2.fc1, dtype=torch.qint8)
            self.checkDynamicQuantizedLinear(model.sub2.fc2, dtype=torch.qint8)
            test_only_eval_fn(model, self.calib_data)
            self.checkScriptable(model, self.calib_data, check_save_load=True)

        checkQuantized(model)
        # test one line API
        model = quantize_dynamic(NestedModel().eval(), qconfig_dict)
        checkQuantized(model)

    def test_quantized_rnn(self):
        r"""Test execution and serialization for dynamic quantized lstm modules on int8 and fp16
        """
        d_in, d_hid = 2, 2
        model = LSTMDynamicModel().eval()
        cell = model.lstm

        # Replace parameter values s.t. the range of values is exactly
        # 255, thus we will have 0 quantization error in the quantized
        # GEMM call. This i s for testing purposes.
        #
        # Note that the current implementation does not support
        # accumulation values outside of the range representable by a
        # 16 bit integer, instead resulting in a saturated value. We
        # must take care that in our test we do not end up with a dot
        # product that overflows the int16 range, e.g.
        # (255*127+255*127) = 64770. So, we hardcode the test values
        # here and ensure a mix of signedness.
        vals = [[100, -155],
                [100, -155],
                [-155, 100],
                [-155, 100],
                [100, -155],
                [-155, 100],
                [-155, 100],
                [100, -155]]
        if isinstance(cell, torch.nn.LSTM):
            num_chunks = 4
        vals = vals[:d_hid * num_chunks]
        cell.weight_ih_l0 = torch.nn.Parameter(
            torch.tensor(vals, dtype=torch.float),
            requires_grad=False)
        cell.weight_hh_l0 = torch.nn.Parameter(
            torch.tensor(vals, dtype=torch.float),
            requires_grad=False)

        ref = copy.deepcopy(cell)
        niter = 10
        x = torch.tensor([[100, -155],
                          [-155, 100],
                          [100, -155]], dtype=torch.float).unsqueeze(0).repeat(niter, 1, 1)

        h0_vals = [[-155, 100],
                   [-155, 155],
                   [100, -155]]

        hx = torch.tensor(h0_vals, dtype=torch.float).unsqueeze(0)
        cx = torch.tensor(h0_vals, dtype=torch.float).unsqueeze(0)

        if isinstance(ref, torch.nn.LSTM):
            hiddens = (hx, cx)

        ref_out, ref_hid = ref(x, hiddens)

        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                for dtype in [torch.qint8, torch.float16]:
                    if dtype == torch.float16 and qengine == "qnnpack":
                        # fp16 dynamic quant is not supported for qnnpack
                        continue
                    model_quantized = quantize_dynamic(model=model, dtype=dtype)

                    # Smoke test extra reprs
                    self.assertTrue('DynamicQuantizedLSTM' in str(model_quantized))
                    cell_quantized = model_quantized.lstm

                    assert type(cell_quantized) == torch.nn.quantized.dynamic.LSTM, \
                        'torch.nn.LSTM should be converted to torch.nn.quantized.dynamic.LSTM after quantize_dynamic'

                    # Compare int8/fp16 quantized to unquantized
                    output_quantized, final_hiddens_quantized = cell_quantized(x, hiddens)

                    torch.testing.assert_allclose(output_quantized, ref_out)
                    self.assertEqual(output_quantized, ref_out)
                    for out_val, ref_val in zip(final_hiddens_quantized, ref_hid):
                        torch.testing.assert_allclose(out_val, ref_val)

                    if dtype == torch.qint8:
                        # TODO: Revisit serialization tests once torchbind support lands

                        class ScriptWrapper(torch.nn.Module):
                            def __init__(self, cell):
                                super(ScriptWrapper, self).__init__()
                                self.cell = cell

                            def forward(self, x, hiddens):
                                # type: (torch.Tensor, Tuple[torch.Tensor, torch.Tensor])
                                # -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
                                return self.cell(x, hiddens)

                        # TODO: TorchScript overloads don't work without this wrapper
                        cell_script = torch.jit.script(ScriptWrapper(cell_quantized))
                        out_script, hid_script = cell_script(x, hiddens)
                        self.assertEqual(len(out_script), len(ref_out))
                        for out_val, ref_val in zip(out_script, ref_out):
                            torch.testing.assert_allclose(out_val, ref_val)

                        # Test save/load
                        b = io.BytesIO()
                        torch.jit.save(cell_script, b)
                        b.seek(0)
                        loaded = torch.jit.load(b)
                        out_loaded, hid_loaded = loaded(x, hiddens)
                        for loaded_val, ref_val in zip(out_loaded, ref_out):
                            torch.testing.assert_allclose(loaded_val, ref_val)

                        # Test tracing
                        # TODO: TorchScript overloads don't work without this wrapper
                        cell_trace = torch.jit.trace(ScriptWrapper(cell_quantized), (x, (hx, cx)))
                        out_script, hid_script = cell_trace(x, hiddens)
                        for out_val, ref_val in zip(out_script, ref_out):
                            torch.testing.assert_allclose(out_val, ref_val)
                        # Test save/load
                        b = io.BytesIO()
                        torch.jit.save(cell_trace, b)
                        b.seek(0)
                        loaded = torch.jit.load(b)
                        out_loaded, hid_loaded = loaded(x, hiddens)
                        for loaded_val, ref_val in zip(out_loaded, ref_out):
                            torch.testing.assert_allclose(loaded_val, ref_val)


                        class ScriptWrapperPacked(torch.nn.Module):
                            def __init__(self, cell):
                                super(ScriptWrapperPacked, self).__init__()
                                self.cell = cell

                            def forward(self,
                                        x,  # type: PackedSequence
                                        hiddens  # type: Tuple[torch.Tensor, torch.Tensor]
                                        ):
                                # type: (...) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]
                                return self.cell(x, hiddens)

                        cell_packed = torch.jit.script(ScriptWrapperPacked(cell_quantized))
                        packed_input = torch.nn.utils.rnn.pack_padded_sequence(x, torch.tensor([10, 5, 2]))
                        ref_out_packed, ref_hid_packed = ref(packed_input, hiddens)
                        output_packed, hiddens_packed = cell_packed(packed_input, hiddens)

                        for packed_val, ref_val in zip(output_packed, ref_out_packed):
                            if isinstance(packed_val, torch.Tensor):
                                torch.testing.assert_allclose(packed_val, ref_val)
                            else:
                                self.assertEqual(packed_val, ref_val)

                        # Test save/load
                        b = io.BytesIO()
                        torch.jit.save(cell_packed, b)
                        b.seek(0)
                        loaded_packed = torch.jit.load(b)
                        out_loaded_packed, hid_loaded_packed = loaded_packed(packed_input, hiddens)
                        for packed_val, ref_val in zip(out_loaded_packed, ref_out_packed):
                            if isinstance(packed_val, torch.Tensor):
                                torch.testing.assert_allclose(packed_val, ref_val)
                            else:
                                self.assertEqual(packed_val, ref_val)


    def test_default_quantized_lstm(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                # Test default instantiation
                seq_len = 128
                batch = 16
                input_size = 3
                hidden_size = 7
                num_layers = 2
                bias = True
                bidirectional = False

                x = torch.rand(seq_len, batch, input_size)
                h = torch.rand(num_layers * (bidirectional + 1), batch, hidden_size)
                c = torch.rand(num_layers * (bidirectional + 1), batch, hidden_size)

                dtype = torch.qint8

                cell_dq = torch.nn.quantized.dynamic.LSTM(input_size=input_size,
                                                          hidden_size=hidden_size,
                                                          num_layers=num_layers,
                                                          bias=bias,
                                                          batch_first=False,
                                                          dropout=0.0,
                                                          bidirectional=bidirectional,
                                                          dtype=dtype)

                y, (h, c) = cell_dq(x, (h, c))


class TestQuantizationAwareTraining(QuantizationTestCase):
    def test_manual(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = ManualLinearQATModel(qengine)
                model = prepare_qat(model)
                self.checkObservers(model)
                test_only_train_fn(model, self.train_data)
                model = convert(model)

                def checkQuantized(model):
                    self.assertEqual(type(model.fc1), nnq.Linear)
                    self.assertEqual(type(model.fc2), nnq.Linear)
                    test_only_eval_fn(model, self.calib_data)
                    self.checkScriptable(model, self.calib_data)

                checkQuantized(model)

                model = quantize_qat(ManualLinearQATModel(qengine), test_only_train_fn,
                                     self.train_data)
                checkQuantized(model)

    def test_activations(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = ActivationsQATTestModel(qengine)
                model = prepare_qat(model)

                self.assertEqual(type(model.fc1), torch.nn.qat.modules.Linear)
                self.assertEqual(type(model.hardswish), torch.nn.qat.modules.Hardswish)

                self.checkObservers(model)
                test_only_train_fn(model, self.train_data)
                model = convert(model)

                def checkQuantized(model):
                    self.assertEqual(type(model.fc1), nnq.Linear)
                    self.assertEqual(type(model.hardswish), nnq.Hardswish)
                    test_only_eval_fn(model, self.calib_data)
                    self.checkScriptable(model, self.calib_data)

                checkQuantized(model)

                model = quantize_qat(ActivationsQATTestModel(qengine), test_only_train_fn,
                                     self.train_data)
                checkQuantized(model)

    def test_eval_only_fake_quant(self):
        r"""Using FakeQuant in evaluation only mode,
        this is useful for estimating accuracy loss when we quantize the
        network
        """
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = ManualLinearQATModel(qengine)

                model = prepare_qat(model)
                self.checkObservers(model)

                model.eval()
                test_only_eval_fn(model, self.calib_data)

    def test_conv_linear(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = ManualConvLinearQATModel()

                model = prepare_qat(model)
                self.checkObservers(model)

                test_only_train_fn(model, self.img_data)
                model = convert(model)

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

    def test_train_save_load_eval(self):
        r"""Test QAT flow of creating a model, doing QAT and saving the quantized state_dict
        During eval, we first call prepare_qat and conver on the model and then load the state_dict
        and compare results against original model
        """
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = TwoLayerLinearModel()
                model = torch.quantization.QuantWrapper(model)
                model.qconfig = torch.quantization.get_default_qat_qconfig(qengine)
                model = prepare_qat(model)

                fq_state_dict = model.state_dict()

                test_only_train_fn(model, self.train_data)
                model = convert(model)

                quant_state_dict = model.state_dict()

                x = torch.rand(2, 5, dtype=torch.float)
                ref = model(x)

                # Create model again for eval. Check result using quantized state_dict
                model = TwoLayerLinearModel()
                model = torch.quantization.QuantWrapper(model)
                model.qconfig = torch.quantization.get_default_qat_qconfig(qengine)
                torch.quantization.prepare_qat(model, inplace=True)
                new_state_dict = model.state_dict()

                # Check to make sure the model after prepare_qat has the same state_dict as original.
                self.assertEqual(set(fq_state_dict.keys()), set(new_state_dict.keys()))

                torch.quantization.convert(model, inplace=True)
                model.eval()
                model.load_state_dict(quant_state_dict)
                out = model(x)
                self.assertEqual(ref, out)

                # Check model created using prepare has same state dict as quantized state_dict
                model = TwoLayerLinearModel()
                model.eval()
                model = torch.quantization.QuantWrapper(model)
                model.qconfig = torch.quantization.get_default_qconfig(qengine)
                torch.quantization.prepare(model, inplace=True)
                torch.quantization.convert(model, inplace=True)
                self.assertEqual(set(model.state_dict().keys()), set(quant_state_dict.keys()))
                model.eval()
                model.load_state_dict(quant_state_dict)
                out = model(x)
                self.assertEqual(ref, out)


class TestGraphModePostTrainingStatic(QuantizationTestCase):
    def test_single_linear(self):
        r"""Compare the result of quantizing single linear layer in
        eager mode and graph mode
        """
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                # eager mode
                annotated_linear_model = AnnotatedSingleLayerLinearModel(qengine).eval()
                linear_model = SingleLayerLinearModel().eval()
                # copy the weight from eager mode so that we can
                # compare the result of the two quantized models later
                linear_model.fc1.weight = torch.nn.Parameter(annotated_linear_model.fc1.module.weight.detach())
                linear_model.fc1.bias = torch.nn.Parameter(annotated_linear_model.fc1.module.bias.detach())
                model_eager = quantize(annotated_linear_model, test_only_eval_fn,
                                       self.calib_data)

                qconfig_dict = {'': torch.quantization.get_default_qconfig(qengine)}
                model_traced = torch.jit.trace(linear_model, self.calib_data[0][0])
                model_script = torch.jit.script(linear_model)
                result_eager = model_eager(self.calib_data[0][0])
                for model_under_test in [model_traced, model_script]:
                    model_quantized = quantize_script(
                        model_under_test,
                        qconfig_dict,
                        test_only_eval_fn,
                        [self.calib_data],
                        inplace=False)
                    self.assertEqual(model_quantized(self.calib_data[0][0]), result_eager)

    @skipIfNoFBGEMM
    def test_observer_with_ignored_function(self):
        r"""Test observers with ignored function and make sure it works in
        graph mode
        """
        # eager mode
        annotated_linear_model = AnnotatedSingleLayerLinearModel('fbgemm').eval()
        for qconfig in [
                QConfig(
                    activation=default_observer,
                    weight=default_weight_observer),
                QConfig(
                    activation=default_histogram_observer,
                    weight=default_weight_observer),
                QConfig(
                    activation=default_observer,
                    weight=default_per_channel_weight_observer),
        ]:
            annotated_linear_model.qconfig = qconfig
            linear_model = SingleLayerLinearModel().eval()
            # copy the weight from eager mode so that we can
            # compare the result of the two quantized models later
            linear_model.fc1.weight = torch.nn.Parameter(annotated_linear_model.fc1.module.weight.detach())
            linear_model.fc1.bias = torch.nn.Parameter(annotated_linear_model.fc1.module.bias.detach())
            model_eager = quantize(annotated_linear_model, test_only_eval_fn,
                                   self.calib_data)

            qconfig_dict = {'': qconfig}
            model_traced = torch.jit.trace(linear_model, self.calib_data[0][0])
            model_script = torch.jit.script(linear_model)
            result_eager = model_eager(self.calib_data[0][0])
            for model_under_test in [model_traced, model_script]:
                model_quantized = quantize_script(
                    model_under_test,
                    qconfig_dict,
                    test_only_eval_fn,
                    [self.calib_data],
                    inplace=False)
                self.assertEqual(model_quantized(self.calib_data[0][0]), result_eager)

    def test_conv(self):
        r"""Compare the result of quantizing conv layer in
        eager mode and graph mode
        """
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                # eager mode
                annotated_conv_model = AnnotatedConvModel(qengine).eval()
                conv_model = ConvModel().eval()
                # copy the weight from eager mode so that we can
                # compare the result of the two quantized models later
                conv_model.conv.weight = torch.nn.Parameter(annotated_conv_model.conv.weight.detach())
                model_eager = quantize(annotated_conv_model, default_eval_fn,
                                       self.img_data)
                qconfig_dict = {'': torch.quantization.get_default_qconfig(qengine)}
                model_traced = torch.jit.trace(conv_model, self.img_data[0][0])
                model_script = torch.jit.script(conv_model)
                result_eager = model_eager(self.img_data[0][0])
                for model_under_test in [model_traced, model_script]:
                    model_quantized = quantize_script(
                        model_under_test,
                        qconfig_dict,
                        default_eval_fn,
                        [self.img_data],
                        inplace=False)
                    self.assertEqual(model_quantized(self.img_data[0][0]), result_eager)

    @unittest.skip("This doesn't work right now, re-enable after fold_convbn is fixed")
    def test_conv_bn(self):
        r"""Compare the result of quantizing conv + bn layer in
        eager mode and graph mode
        """
        # eager mode
        conv_model = AnnotatedConvBnModel().eval()
        conv_model_to_script = ConvBnModel().eval()
        # copy the weight from eager mode so that we can
        # compare the result of the two quantized models later
        conv_model_to_script.conv.weight = torch.nn.Parameter(conv_model.conv.weight.detach())
        fuse_modules(conv_model, ['conv', 'bn'], inplace=True)
        model_eager = quantize(conv_model, default_eval_fn,
                               self.img_data)
        qconfig_dict = {
            '': default_qconfig
        }
        model_script = quantize_script(
            torch.jit.script(conv_model_to_script),
            qconfig_dict,
            default_eval_fn,
            [self.img_data],
            inplace=False)
        result_eager = model_eager(self.img_data[0][0])
        result_script = model_script(self.img_data[0][0])
        self.assertEqual(result_eager, result_script)

    def test_nested(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                # Eager mode
                eager_model = AnnotatedNestedModel(qengine).eval()

                # Graph mode
                script_model = NestedModel().eval()
                # Copy weights for eager_model
                script_model.sub1.fc.weight = torch.nn.Parameter(eager_model.sub1.fc.weight.detach())
                script_model.sub1.fc.bias = torch.nn.Parameter(eager_model.sub1.fc.bias.detach())
                script_model.sub2.fc1.weight = torch.nn.Parameter(eager_model.sub2.fc1.module.weight.detach())
                script_model.sub2.fc1.bias = torch.nn.Parameter(eager_model.sub2.fc1.module.bias.detach())
                script_model.sub2.fc2.weight = torch.nn.Parameter(eager_model.sub2.fc2.weight.detach())
                script_model.sub2.fc2.bias = torch.nn.Parameter(eager_model.sub2.fc2.bias.detach())
                script_model.fc3.weight = torch.nn.Parameter(eager_model.fc3.module.weight.detach())
                script_model.fc3.bias = torch.nn.Parameter(eager_model.fc3.module.bias.detach())

                model_eager = quantize(eager_model, test_only_eval_fn, self.calib_data)
                qconfig_dict = {
                    'sub2.fc1': default_per_channel_qconfig if qengine == 'fbgemm' else default_qconfig,
                    'fc3': default_qconfig
                }
                model_traced = torch.jit.trace(script_model, self.calib_data[0][0])
                model_script = torch.jit.script(script_model)
                result_eager = model_eager(self.calib_data[0][0])
                for model_under_test in [model_traced, model_script]:
                    model_quantized = quantize_script(
                        model_under_test,
                        qconfig_dict,
                        test_only_eval_fn,
                        [self.calib_data],
                        inplace=False)
                    self.assertEqual(model_quantized(self.calib_data[0][0]), result_eager)

    def test_skip_quant(self):
        """ Test None qconfig
        """
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                # Eager mode
                eager_model = AnnotatedSkipQuantModel(qengine).eval()

                # Graph mode
                script_model = SkipQuantModel().eval()
                # Copy weights for eager_model
                script_model.sub.fc1.weight = torch.nn.Parameter(eager_model.sub.module.fc1.weight.detach())
                script_model.sub.fc1.bias = torch.nn.Parameter(eager_model.sub.module.fc1.bias.detach())
                script_model.sub.fc2.weight = torch.nn.Parameter(eager_model.sub.module.fc2.weight.detach())
                script_model.sub.fc2.bias = torch.nn.Parameter(eager_model.sub.module.fc2.bias.detach())
                script_model.fc.weight = torch.nn.Parameter(eager_model.fc.weight.detach())
                script_model.fc.bias = torch.nn.Parameter(eager_model.fc.bias.detach())

                model_eager = quantize(eager_model, test_only_eval_fn, self.calib_data)
                qconfig_dict = {
                    '': torch.quantization.get_default_qconfig(qengine),
                    'fc': None
                }
                model_traced = torch.jit.trace(script_model, self.calib_data[0][0])
                model_script = torch.jit.script(script_model)
                result_eager = model_eager(self.calib_data[0][0])
                for model_under_test in [model_traced, model_script]:
                    model_quantized = quantize_script(
                        model_under_test,
                        qconfig_dict,
                        test_only_eval_fn,
                        [self.calib_data],
                        inplace=False)
                    self.assertEqual(model_quantized(self.calib_data[0][0]), result_eager)

    def test_single_linear_dynamic(self):
        r"""Compare the result of dynamic quantization of single linear layer in
        eager mode and graph mode.
        """
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                # eager mode
                annotated_linear_model = AnnotatedSingleLayerLinearModel('qnnpack').eval()
                linear_model = SingleLayerLinearModel().eval()
                # copy the weight from eager mode so that we can
                # compare the result of the two quantized models later
                linear_model.fc1.weight = torch.nn.Parameter(annotated_linear_model.fc1.module.weight.detach())
                linear_model.fc1.bias = torch.nn.Parameter(annotated_linear_model.fc1.module.bias.detach())
                qconfig_dict = {'': default_dynamic_qconfig}
                model_eager = quantize_dynamic(annotated_linear_model, qconfig_dict)

                model_traced = torch.jit.trace(linear_model, self.calib_data[0][0])
                model_script = torch.jit.script(linear_model)
                result_eager = model_eager(self.calib_data[0][0])

                for model_under_test in [model_traced, model_script]:
                    model_quantized = quantize_dynamic_script(
                        model_under_test,
                        qconfig_dict,
                        [self.calib_data[0][0]])
                    self.assertEqual(model_quantized(self.calib_data[0][0]), result_eager)

                    # Check to make sure choose_qparams->quant->dequant->linear is numerically
                    # equivalent to the final quantized model.
                    model_fake_quantized = quantize_dynamic_script(
                        model_under_test,
                        qconfig_dict,
                        [self.calib_data[0][0]],
                        debug=True)
                    self.assertEqual(model_fake_quantized(self.calib_data[0][0]), result_eager)


class TestFunctionalModule(QuantizationTestCase):
    # Histogram Observers are slow, so have no-deadline to ensure test doesn't time out
    @given(train_mode=st.booleans())
    def test_functional_module(self, train_mode):
        model = ModelWithFunctionals()
        x = torch.rand(10, 1, dtype=torch.float)
        xq = torch.quantize_per_tensor(x, 0.01, 30, torch.quint8)
        self.checkScriptable(model, [(x, x)], check_save_load=True)
        if train_mode:
            model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            model = prepare_qat(model)
        else:
            model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
            model = prepare(model)
        # Check if observers and quant/dequant nodes are inserted
        self.checkNoPrepModules(model)
        self.checkObservers(model)
        # Calibrate
        model(xq.dequantize())
        model = convert(model)

        def checkQuantized(model):
            self.checkNoPrepModules(model)
            self.assertEqual(type(model.myadd), torch.nn.quantized.QFunctional)
            self.assertEqual(type(model.mycat), torch.nn.quantized.QFunctional)
            self.assertEqual(type(model.myadd_relu), torch.nn.quantized.QFunctional)

        checkQuantized(model)
        self.checkScriptable(model, [(xq, xq)], check_save_load=True)

# TODO: figure out why this is not running on devgpu?
#@skipIfNoFBGEMM
class TestFusion(QuantizationTestCase):
    def test_fuse_module_train(self):
        model = ModelForFusion(default_qat_qconfig).train()
        # Test step by step fusion
        model = fuse_modules(model, ['conv1', 'bn1', 'relu1'])
        model = fuse_modules(model, ['sub1.conv', 'sub1.bn'])
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
        model = convert(model)

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
        model = fuse_modules(model, [['conv1', 'bn1', 'relu1'],
                             ['sub1.conv', 'sub1.bn']])
        model = quantize_qat(model, test_only_train_fn, self.img_data)
        checkQuantized(model)


    def test_fuse_module_eval(self):
        model = ModelForFusion(default_qconfig)
        model.eval()
        model = fuse_modules(model, [['conv1', 'bn1', 'relu1'] ,
                             ['conv2', 'relu2'],
                             ['bn2', 'relu3'],
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
        self.assertEqual(type(model.conv2), nni.ConvReLU3d,
                         "Fused Conv + BN + Relu first layer (BN is folded)")
        self.assertEqual(type(model.bn2), nni.BNReLU3d,
                         "Fused BN + Relu first layer (Relu is folded))")
        self.assertEqual(type(model.relu3), nn.Identity,
                         "Fused BN + Relu second layer (Skipped Relu)")
        self.assertEqual(type(model.conv2[0]), nn.Conv3d,
                         "Fused Conv + BN + Relu (Conv + folded BN only)")
        self.assertEqual(type(model.conv2[1]), nn.ReLU,
                         "Fused Conv + BN + Relu second layer (Relu only)")
        self.assertEqual(type(model.relu2), nn.Identity,
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
        model = convert(model)

        def checkQuantized(model):
            self.assertEqual(type(model.conv1), nniq.ConvReLU2d)
            self.assertEqual(type(model.bn1), nn.Identity)
            self.assertEqual(type(model.relu1), nn.Identity)
            self.assertEqual(type(model.sub1.conv), nnq.Conv2d)
            self.assertEqual(type(model.sub1.bn), nn.Identity)
            self.assertEqual(type(model.sub2.conv), nn.Conv2d)
            self.assertEqual(type(model.sub2.relu), nn.ReLU)
            self.assertEqual(type(model.bn2), nniq.BNReLU3d)
            test_only_eval_fn(model, self.img_data)
        checkQuantized(model)

        model = ModelForFusion(default_qconfig).eval()
        model = fuse_modules(model, [['conv1', 'bn1', 'relu1'],
                             ['conv2', 'relu2'],
                             ['bn2', 'relu3'],
                             ['sub1.conv', 'sub1.bn']])
        model = quantize(model, test_only_eval_fn, self.img_data)
        checkQuantized(model)

    def test_fusion_sequential_model_train(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = ModelWithSequentialFusion().train()
                model.to(torch.float)
                fuse_modules(model, [['conv1', 'relu1'] ,
                                     ['features.0.0', 'features.0.1', 'features.0.2'],
                                     ['features.1.0', 'features.1.1', 'features.1.2'],
                                     ['features.2.0', 'features.2.1', 'features.2.2'],
                                     ['classifier.0', 'classifier.1']], inplace=True)
                self.assertEqual(type(model.conv1), nni.ConvReLU2d,
                                 "Fused Conv + Relu: nni.ConvReLU2d")
                self.assertEqual(type(model.conv1[0]), nn.Conv2d,
                                 "Fused Conv + Relu: Conv2d")
                self.assertEqual(type(model.conv1[1]), nn.ReLU,
                                 "Fused Conv + Relu: Relu")
                self.assertEqual(type(model.relu1), nn.Identity,
                                 "Fused Conv + Relu: Identity")
                for i in range(3):
                    self.assertEqual(type(model.features[i][0]), nni.ConvBnReLU2d,
                                     "Fused submodule Conv + folded BN")
                    self.assertEqual(type(model.features[i][1]), nn.Identity,
                                     "Fused submodule (skipped BN)")
                    self.assertEqual(type(model.features[i][2]), nn.Identity,
                                     "Non-fused submodule Conv")
                self.assertEqual(type(model.classifier[0]), nni.LinearReLU)
                self.assertEqual(type(model.classifier[1]), nn.Identity)
                model.qconfig = torch.quantization.get_default_qat_qconfig(qengine)
                prepare_qat(model, inplace=True)
                self.checkObservers(model)
                model(self.img_data[0][0])


                def checkQAT(model):
                    self.assertEqual(type(model.conv1), nniqat.ConvReLU2d)
                    self.assertEqual(type(model.relu1), nn.Identity)
                for i in range(3):
                    self.assertEqual(type(model.features[i][0]), nniqat.ConvBnReLU2d,
                                     "Fused submodule Conv + folded BN")
                    self.assertEqual(type(model.features[i][1]), nn.Identity,
                                     "Fused submodule (skipped BN)")
                    self.assertEqual(type(model.features[i][2]), nn.Identity,
                                     "Non-fused submodule Conv")
                self.assertEqual(type(model.classifier[0]), nniqat.LinearReLU)
                self.assertEqual(type(model.classifier[1]), nn.Identity)

                checkQAT(model)
                model(self.img_data[1][0])
                convert(model, inplace=True)
                model(self.img_data[1][0])
                self.checkModelWithSequentialQuantized(model)

    def test_fusion_sequential_model_eval(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = ModelWithSequentialFusion().eval()
                model.to(torch.float)
                fuse_modules(model, [['conv1', 'relu1'] ,
                                     ['features.0.0', 'features.0.1', 'features.0.2'],
                                     ['features.1.0', 'features.1.1', 'features.1.2'],
                                     ['features.2.0', 'features.2.1', 'features.2.2'],
                                     ['classifier.0', 'classifier.1']], inplace=True)
                self.assertEqual(type(model.conv1), nni.ConvReLU2d,
                                 "Fused Conv + Relu: nni.ConvReLU2d")
                self.assertEqual(type(model.conv1[0]), nn.Conv2d,
                                 "Fused Conv + Relu: Conv2d")
                self.assertEqual(type(model.conv1[1]), nn.ReLU,
                                 "Fused Conv + Relu: Relu")
                self.assertEqual(type(model.relu1), nn.Identity,
                                 "Fused Conv + Relu: Identity")
                for i in range(3):
                    self.assertEqual(type(model.features[i][0]), nni.ConvReLU2d,
                                     "Fused submodule Conv + folded BN")
                    self.assertEqual(type(model.features[i][1]), nn.Identity,
                                     "Fused submodule (skipped BN)")
                    self.assertEqual(type(model.features[i][2]), nn.Identity,
                                     "Non-fused submodule Conv")
                self.assertEqual(type(model.classifier[0]), nni.LinearReLU)
                self.assertEqual(type(model.classifier[1]), nn.Identity)
                model.qconfig = torch.quantization.get_default_qconfig(qengine)
                prepare(model, inplace=True)
                self.checkObservers(model)
                model(self.img_data[0][0])
                convert(model, inplace=True)
                model(self.img_data[1][0])
                self.checkModelWithSequentialQuantized(model)

    def checkModelWithSequentialQuantized(self, model):
        self.assertEqual(type(model.conv1), nniq.ConvReLU2d)
        self.assertEqual(type(model.relu1), nn.Identity)
        for i in range(3):
            self.assertEqual(type(model.features[i][0]), nniq.ConvReLU2d)
            self.assertEqual(type(model.features[i][1]), nn.Identity)
            self.assertEqual(type(model.features[i][2]), nn.Identity)
        self.assertEqual(type(model.classifier[0]), nniq.LinearReLU)
        self.assertEqual(type(model.classifier[1]), nn.Identity)

    def test_fusion_conv_with_bias(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = ModelForFusionWithBias().train()
                # output with no fusion.
                out_ref = model(self.img_data[0][0])

                model.qconfig = QConfig(activation=torch.nn.Identity,
                                        weight=torch.nn.Identity)
                model = fuse_modules(model, [["conv1", "bn1", "relu1"],
                                             ["conv2", "bn2"]])
                prep_model = prepare_qat(model, inplace=False)
                # output with fusion but no observers.
                out_fused = prep_model(self.img_data[0][0])
                # TODO: fix this failure
                self.assertEqual(out_ref, out_fused)

                model.qconfig = torch.quantization.get_default_qconfig(qengine)
                prepare_qat(model, inplace=True)

                model(self.img_data[0][0])

                def checkQAT(model):
                    self.assertEqual(type(model.conv1), nniqat.ConvBnReLU2d)
                    self.assertEqual(type(model.bn1), nn.Identity)
                    self.assertEqual(type(model.relu1), nn.Identity)
                    self.assertEqual(type(model.conv2), nniqat.ConvBn2d)
                    self.assertEqual(type(model.bn2), nn.Identity)

                checkQAT(model)

class TestModelNumerics(QuantizationTestCase):
    def test_float_quant_compare_per_tensor(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                torch.manual_seed(42)
                my_model = ModelMultipleOps().to(torch.float32)
                my_model.eval()
                calib_data = torch.rand(1024, 3, 15, 15, dtype=torch.float32)
                eval_data = torch.rand(1, 3, 15, 15, dtype=torch.float32)
                out_ref = my_model(eval_data)
                qModel = torch.quantization.QuantWrapper(my_model)
                qModel.eval()
                qModel.qconfig = torch.quantization.default_qconfig
                torch.quantization.fuse_modules(qModel.module, [['conv1', 'bn1', 'relu1']], inplace=True)
                torch.quantization.prepare(qModel, inplace=True)
                qModel(calib_data)
                torch.quantization.convert(qModel, inplace=True)
                out_q = qModel(eval_data)
                SQNRdB = 20 * torch.log10(torch.norm(out_ref) / torch.norm(out_ref - out_q))
                # Quantized model output should be close to floating point model output numerically
                # Setting target SQNR to be 30 dB so that relative error is 1e-3 below the desired
                # output
                self.assertGreater(SQNRdB, 30, msg='Quantized model numerics diverge from float, expect SQNR > 30 dB')

    def test_float_quant_compare_per_channel(self):
        # Test for per-channel Quant
        torch.manual_seed(67)
        my_model = ModelMultipleOps().to(torch.float32)
        my_model.eval()
        calib_data = torch.rand(2048, 3, 15, 15, dtype=torch.float32)
        eval_data = torch.rand(10, 3, 15, 15, dtype=torch.float32)
        out_ref = my_model(eval_data)
        q_model = torch.quantization.QuantWrapper(my_model)
        q_model.eval()
        q_model.qconfig = torch.quantization.default_per_channel_qconfig
        torch.quantization.fuse_modules(q_model.module, [['conv1', 'bn1', 'relu1']], inplace=True)
        torch.quantization.prepare(q_model)
        q_model(calib_data)
        torch.quantization.convert(q_model)
        out_q = q_model(eval_data)
        SQNRdB = 20 * torch.log10(torch.norm(out_ref) / torch.norm(out_ref - out_q))
        # Quantized model output should be close to floating point model output numerically
        # Setting target SQNR to be 35 dB
        self.assertGreater(SQNRdB, 35, msg='Quantized model numerics diverge from float, expect SQNR > 35 dB')

    def test_fake_quant_true_quant_compare(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                torch.manual_seed(67)
                my_model = ModelMultipleOpsNoAvgPool().to(torch.float32)
                calib_data = torch.rand(2048, 3, 15, 15, dtype=torch.float32)
                eval_data = torch.rand(10, 3, 15, 15, dtype=torch.float32)
                my_model.eval()
                out_ref = my_model(eval_data)
                fq_model = torch.quantization.QuantWrapper(my_model)
                fq_model.train()
                fq_model.qconfig = torch.quantization.default_qat_qconfig
                torch.quantization.fuse_modules(fq_model.module, [['conv1', 'bn1', 'relu1']], inplace=True)
                torch.quantization.prepare_qat(fq_model)
                fq_model.eval()
                fq_model.apply(torch.quantization.disable_fake_quant)
                fq_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
                fq_model(calib_data)
                fq_model.apply(torch.quantization.enable_fake_quant)
                fq_model.apply(torch.quantization.disable_observer)
                out_fq = fq_model(eval_data)
                SQNRdB = 20 * torch.log10(torch.norm(out_ref) / torch.norm(out_ref - out_fq))
                # Quantized model output should be close to floating point model output numerically
                # Setting target SQNR to be 35 dB
                self.assertGreater(SQNRdB, 35, msg='Quantized model numerics diverge from float, expect SQNR > 35 dB')
                torch.quantization.convert(fq_model)
                out_q = fq_model(eval_data)
                SQNRdB = 20 * torch.log10(torch.norm(out_fq) / (torch.norm(out_fq - out_q) + 1e-10))
                self.assertGreater(SQNRdB, 60, msg='Fake quant and true quant numerics diverge, expect SQNR > 60 dB')

    # Test to compare weight only quantized model numerics and
    # activation only quantized model numerics with float
    def test_weight_only_activation_only_fakequant(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                torch.manual_seed(67)
                calib_data = torch.rand(2048, 3, 15, 15, dtype=torch.float32)
                eval_data = torch.rand(10, 3, 15, 15, dtype=torch.float32)
                qconfigset = set([torch.quantization.default_weight_only_qconfig,
                                  torch.quantization.default_activation_only_qconfig])
                SQNRTarget = [35, 45]
                for idx, qconfig in enumerate(qconfigset):
                    my_model = ModelMultipleOpsNoAvgPool().to(torch.float32)
                    my_model.eval()
                    out_ref = my_model(eval_data)
                    fq_model = torch.quantization.QuantWrapper(my_model)
                    fq_model.train()
                    fq_model.qconfig = qconfig
                    torch.quantization.fuse_modules(fq_model.module, [['conv1', 'bn1', 'relu1']], inplace=True)
                    torch.quantization.prepare_qat(fq_model)
                    fq_model.eval()
                    fq_model.apply(torch.quantization.disable_fake_quant)
                    fq_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
                    fq_model(calib_data)
                    fq_model.apply(torch.quantization.enable_fake_quant)
                    fq_model.apply(torch.quantization.disable_observer)
                    out_fq = fq_model(eval_data)
                    SQNRdB = 20 * torch.log10(torch.norm(out_ref) / torch.norm(out_ref - out_fq))
                    self.assertGreater(SQNRdB, SQNRTarget[idx], msg='Quantized model numerics diverge from float')

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_quantization.py TESTNAME\n\n"
                       "instead.")
