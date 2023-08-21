# Owner(s): ["oncall: quantization"]

import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
from torch.nn.utils.rnn import PackedSequence
from torch.ao.quantization import (
    quantize,
    prepare,
    convert,
    prepare_qat,
    quantize_dynamic,
    QuantWrapper,
    QuantStub,
    DeQuantStub,
    default_qconfig,
    default_dynamic_qconfig,
    per_channel_dynamic_qconfig,
    float16_dynamic_qconfig,
    float_qparams_weight_only_qconfig,
    float_qparams_weight_only_qconfig_4bit,
    FixedQParamsObserver,
    PerChannelMinMaxObserver,
    default_dynamic_quant_observer,
    default_weight_observer,
    QConfig,
)

from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    AnnotatedSingleLayerLinearModel,
    QuantStubModel,
    ModelWithFunctionals,
    SingleLayerLinearDynamicModel,
    TwoLayerLinearModel,
    NestedModel,
    ResNetBase,
    RNNDynamicModel,
    RNNCellDynamicModel,
    ActivationsTestModel,
    NormalizationTestModel,
    test_only_eval_fn,
    prepare_dynamic,
    convert_dynamic,
    skipIfNoFBGEMM,
    EmbeddingBagModule,
    EmbeddingModule,
    EmbeddingWithStaticLinear,
    LinearReluLinearModel,
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
    override_qengines,
)

from hypothesis import given
from hypothesis import strategies as st
import torch.testing._internal.hypothesis_utils as hu
hu.assert_deadline_disabled()

# Standard library
from typing import Tuple
import numpy as np

class TestQuantizeEagerOps(QuantizationTestCase):
    @override_qengines
    def _test_reference_module_impl(self,
                                    float_module_class,
                                    quantized_module_class,
                                    extra_module_kwargs,
                                    input_size):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = float_module_class(**extra_module_kwargs)
                self.quant = QuantStub()
                self.dequant = DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv(x)
                x = self.dequant(x)
                return x

        class RefM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = float_module_class(**extra_module_kwargs)
                self.quant1 = QuantStub()
                self.dequant1 = DeQuantStub()
                self.quant2 = QuantStub()
                self.dequant2 = DeQuantStub()

            def forward(self, x):
                x = self.quant1(x)
                x = self.dequant1(x)
                x = self.conv(x)
                x = self.quant2(x)
                x = self.dequant2(x)
                return x

        qengine = torch.backends.quantized.engine
        if qengine not in supported_qengines or qengine == 'qnnpack':
            return   # qnnpack does not support nnq.ConvTranspose3d

        data = torch.randn(*input_size, dtype=torch.float)
        original_m = M()
        original_ref_m = RefM()

        original_ref_m.conv.weight = torch.nn.Parameter(original_m.conv.weight.detach())
        original_ref_m.conv.bias = torch.nn.Parameter(original_m.conv.bias.detach())

        original_m.qconfig = torch.ao.quantization.default_qconfig

        m = prepare(original_m)
        # calibration
        m(data)
        m = convert(m)
        # check if the module is properly quantized
        self.assertEqual(type(m.quant), nnq.Quantize)
        self.assertEqual(type(m.conv), quantized_module_class)
        self.assertEqual(type(m.dequant), nnq.DeQuantize)
        res = m(data)

        # quantize the reference model
        original_ref_m.eval()
        original_ref_m.qconfig = torch.ao.quantization.default_qconfig

        ref_m = prepare(original_ref_m)
        ref_m(data)
        ref_m = convert(ref_m, is_reference=True)
        ref_res = ref_m(data)
        self.assertEqual(res, ref_res)

    def test_conv_1d(self):
        self._test_reference_module_impl(
            nn.Conv1d,
            nnq.Conv1d,
            {'in_channels': 1, 'out_channels': 1, 'kernel_size': 1},
            (16, 1, 1)
        )

    def test_conv_2d(self):
        self._test_reference_module_impl(
            nn.Conv2d,
            nnq.Conv2d,
            {'in_channels': 1, 'out_channels': 1, 'kernel_size': 1},
            (16, 1, 10, 10)
        )

    def test_conv_3d(self):
        self._test_reference_module_impl(
            nn.Conv3d,
            nnq.Conv3d,
            {'in_channels': 1, 'out_channels': 1, 'kernel_size': 1},
            (16, 1, 10, 10, 10)
        )

    def test_conv_transpose_1d(self):
        self._test_reference_module_impl(
            nn.ConvTranspose1d,
            nnq.ConvTranspose1d,
            {'in_channels': 1, 'out_channels': 1, 'kernel_size': 1},
            (16, 1, 1)
        )

    def test_conv_transpose_2d(self):
        self._test_reference_module_impl(
            nn.ConvTranspose2d,
            nnq.ConvTranspose2d,
            {'in_channels': 1, 'out_channels': 1, 'kernel_size': 1},
            (16, 1, 10, 10)
        )

    def test_conv_transpose_3d(self):
        self._test_reference_module_impl(
            nn.ConvTranspose3d,
            nnq.ConvTranspose3d,
            {'in_channels': 1, 'out_channels': 1, 'kernel_size': 1},
            (16, 1, 10, 10, 10)
        )

    def test_linear(self):
        self._test_reference_module_impl(
            nn.Linear,
            nnq.Linear,
            {'in_features': 5, 'out_features': 10},
            (16, 5)
        )

    @override_qengines
    def test_int16_reference_module(self):

        class RefM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.ConvTranspose2d(1, 1, 1)
                self.quant1 = QuantStub()
                self.dequant1 = DeQuantStub()
                self.quant2 = QuantStub()
                self.dequant2 = DeQuantStub()

            def forward(self, x):
                x = self.quant1(x)
                x = self.dequant1(x)
                x = self.conv(x)
                x = self.quant2(x)
                x = self.dequant2(x)
                return x


        input_size = (16, 1, 10, 10)
        data = torch.randn(*input_size, dtype=torch.float)

        original_ref_m = RefM()
        rand_w = torch.randn_like(original_ref_m.conv.weight)
        rand_b = torch.randn_like(original_ref_m.conv.bias)
        original_ref_m.conv.weight = torch.nn.Parameter(rand_w, requires_grad=False)
        original_ref_m.conv.bias = torch.nn.Parameter(rand_b, requires_grad=False)

        qengine = torch.backends.quantized.engine
        if qengine not in supported_qengines:
            return
        from torch.ao.quantization.observer import MovingAverageMinMaxObserver

        weight_obs = MovingAverageMinMaxObserver.with_args(
            dtype=torch.qint32,
            # set qmin and qmax to represent qint16
            quant_min=-1 * (2 ** 15),
            quant_max=(2 ** 15) - 1,
            qscheme=torch.per_tensor_symmetric,
        )
        act_obs = MovingAverageMinMaxObserver.with_args(
            dtype=torch.qint32,
            quant_min=-1 * (2 ** 15),
            quant_max=(2 ** 15) - 1,
        )
        custom_qconfig = QConfig(activation=act_obs, weight=weight_obs)

        # quantize the reference model
        original_ref_m.eval()
        original_ref_m.qconfig = custom_qconfig

        ref_m = prepare(original_ref_m)
        # calibration
        ref_m(torch.randn(*input_size, dtype=torch.float))

        ref_m = convert(ref_m, is_reference=True)

        myobs = MovingAverageMinMaxObserver(averaging_constant=0.5,
                                            dtype=torch.qint32,
                                            # set qmin and qmax to represent qint16
                                            quant_min=-1 * (2 ** 15),
                                            quant_max=(2 ** 15) - 1,
                                            qscheme=torch.per_tensor_symmetric,
                                            )
        result = myobs(rand_w)
        qparams = myobs.calculate_qparams()
        self.assertEqual(ref_m.conv.weight_scale, qparams[0])


    def _test_activation_op_impl(
            self, float_module_class, quantized_module_class, extra_module_kwargs):
        """ Implementation for testing common activation ops like leaky relu
        Args:
            extra_module_kwargs: keyword args to instantiate the float module
        """
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.activation_op = float_module_class(**extra_module_kwargs)
                self.quant = QuantStub()
                self.dequant = DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.activation_op(x)
                x = self.dequant(x)
                return x

        m = M().eval()
        m.qconfig = default_qconfig
        m = prepare(m)
        self.checkObservers(m)
        m = convert(m)
        self.assertEqual(type(m.activation_op), quantized_module_class)

    def test_leaky_relu(self):
        self._test_activation_op_impl(nn.LeakyReLU, nnq.LeakyReLU, {'negative_slope': 0.1, 'inplace': False})

    def test_relu(self):
        self._test_activation_op_impl(nn.ReLU, nn.ReLU, {'inplace': False})

    # Histogram Observers are slow, so have no-deadline to ensure test doesn't time out
    @given(train_mode=st.booleans())
    def test_functional_module(self, train_mode):
        model = ModelWithFunctionals()
        x = torch.rand(10, 1, dtype=torch.float)
        xq = torch.quantize_per_tensor(x, 0.01, 30, torch.quint8)
        self.checkScriptable(model, [[x]], check_save_load=True)
        if train_mode:
            model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
            model = prepare_qat(model)
        else:
            model.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
            model = prepare(model)
        # Check if observers and quant/dequant nodes are inserted
        self.checkNoPrepModules(model)
        self.checkObservers(model)
        # Calibrate
        model(xq.dequantize())
        model = convert(model)

        def checkQuantized(model):
            self.checkNoPrepModules(model)
            self.assertEqual(type(model.myadd), torch.ao.nn.quantized.QFunctional)
            self.assertEqual(type(model.mycat), torch.ao.nn.quantized.QFunctional)
            self.assertEqual(type(model.myadd_relu), torch.ao.nn.quantized.QFunctional)
            self.assertEqual(type(model.mymatmul), torch.ao.nn.quantized.QFunctional)
            self.checkNoQconfig(model)

        checkQuantized(model)
        self.checkScriptable(model, [[xq]], check_save_load=True)

class TestQuantizeEagerPTQStatic(QuantizationTestCase):

    def test_single_layer(self):
        r"""Quantize SingleLayerLinearModel which has one Linear module, make sure it is swapped
        to nnq.Linear which is the quantized version of the module
        """
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                qconfig = torch.ao.quantization.get_default_qconfig(qengine)
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
                    self.checkNoQconfig(model)

                checkQuantized(model)

                # test one line API - out of place version
                base = AnnotatedSingleLayerLinearModel(qengine)
                base.qconfig = qconfig
                keys_before = set(base.state_dict().keys())
                model = quantize(base, test_only_eval_fn, [self.calib_data])
                checkQuantized(model)
                keys_after = set(base.state_dict().keys())
                self.assertEqual(keys_before, keys_after)  # simple check that nothing changed

                # in-place version
                model = AnnotatedSingleLayerLinearModel(qengine)
                model.qconfig = qconfig
                quantize(model, test_only_eval_fn, [self.calib_data], inplace=True)
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
                self.checkNoQconfig(model)

            checkQuantized(model)

            # test one line API
            model = quantize(AnnotatedTwoLayerLinearModel(), test_only_eval_fn,
                             [self.calib_data])
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
                    self.checkNoQconfig(model)

                checkQuantized(model)

                # test one line API
                model = quantize(AnnotatedNestedModel(qengine), test_only_eval_fn,
                                 [self.calib_data])
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
            self.checkNoQconfig(model)

        checkQuantized(model)

        # test one line API
        model = quantize(AnnotatedSubNestedModel(), test_only_eval_fn,
                         [self.calib_data])
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
                    self.checkNoQconfig(model)

                checkQuantized(model)

                # test one line API
                model = quantize(AnnotatedCustomConfigNestedModel(), test_only_eval_fn,
                                 [self.calib_data])
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
                    self.assertEqual(type(model.sub.module.relu1), nn.ReLU)
                    self.assertEqual(type(model.sub.module.relu2), nn.ReLU)
                    self.checkScriptable(model, self.calib_data)
                    self.checkNoQconfig(model)

                checkQuantized(model)

                # test one line API
                model = quantize(AnnotatedSkipQuantModel(qengine), test_only_eval_fn, [self.calib_data])
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
            self.checkNoQconfig(model)

        checkQuantized(model)

        # test one line API
        model = quantize(QuantStubModel(), test_only_eval_fn, [self.calib_data])
        checkQuantized(model)

    def test_resnet_base(self):
        r"""Test quantization for bottleneck topology used in resnet/resnext
        and add coverage for conversion of average pool and float functional
        """
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                qconfig = torch.ao.quantization.get_default_qconfig(qengine)
                model = ResNetBase().float().eval()
                model.fuse_model()
                model = QuantWrapper(model)
                model.qconfig = qconfig
                model = prepare(model)
                self.checkObservers(model)
                test_only_eval_fn(model, self.img_data_2d)
                model = convert(model)

                def checkQuantized(model):
                    self.assertEqual(type(model.module.conv1), nn.intrinsic.quantized.ConvReLU2d)
                    self.assertEqual(type(model.module.myop), nn.quantized.QFunctional)
                    self.assertEqual(type(model.module.avgpool), nn.AdaptiveAvgPool2d)
                    self.assertEqual(type(model.module.fc), nnq.Linear)

                    test_only_eval_fn(model, self.img_data_2d)
                    self.checkNoQconfig(model)

                checkQuantized(model)

    @skipIfNoFBGEMM
    def test_normalization(self):
        r"""
        Test quantization of normalization layers
        """
        model = NormalizationTestModel()
        model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        prepare(model, inplace=True)
        self.checkObservers(model)
        test_only_eval_fn(model, self.calib_data)
        model = convert(model)

        def checkQuantized(model):
            self.checkNoPrepModules(model.layer_norm)
            self.checkNoPrepModules(model.group_norm)
            self.checkNoPrepModules(model.instance_norm1d)
            self.checkNoPrepModules(model.instance_norm2d)
            self.checkNoPrepModules(model.instance_norm3d)
            self.assertEqual(type(model.layer_norm), nnq.LayerNorm)
            self.assertEqual(type(model.group_norm), nnq.GroupNorm)
            self.assertEqual(type(model.instance_norm1d), nnq.InstanceNorm1d)
            self.assertEqual(type(model.instance_norm2d), nnq.InstanceNorm2d)
            self.assertEqual(type(model.instance_norm3d), nnq.InstanceNorm3d)
            test_only_eval_fn(model, self.calib_data)
            self.checkScriptable(model, self.calib_data)
            self.checkNoQconfig(model)

        checkQuantized(model)

        model_oneline = quantize(
            NormalizationTestModel(), test_only_eval_fn, [self.calib_data])
        checkQuantized(model)

    def test_save_load_state_dict(self):
        r"""Test PTQ flow of creating a model and quantizing it and saving the quantized state_dict
        Load the quantized state_dict for eval and compare results against original model
        """

        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = TwoLayerLinearModel()
                model = torch.ao.quantization.QuantWrapper(model)
                model.qconfig = torch.ao.quantization.get_default_qconfig(qengine)

                model = prepare(model)
                # calibrate
                test_only_eval_fn(model, self.calib_data)
                model = convert(model)
                x = torch.rand(2, 5, dtype=torch.float)
                ref = model(x)

                quant_state_dict = model.state_dict()

                # Create model again for eval
                model = TwoLayerLinearModel()
                model = torch.ao.quantization.QuantWrapper(model)
                model.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
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
        model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        prepare(model, inplace=True)
        self.checkObservers(model)
        test_only_eval_fn(model, self.calib_data)
        model = convert(model)

        def checkQuantized(model):
            self.checkNoPrepModules(model.hardswish)
            self.assertEqual(type(model.hardswish), nnq.Hardswish)
            self.assertEqual(type(model.elu), nnq.ELU)
            test_only_eval_fn(model, self.calib_data)
            self.checkScriptable(model, self.calib_data)
            self.checkNoQconfig(model)

        checkQuantized(model)

        # test one line API
        model_oneline = quantize(ActivationsTestModel(), test_only_eval_fn,
                                 [self.calib_data])
        checkQuantized(model_oneline)

    @override_qengines
    def test_forward_hooks_preserved(self):
        r"""Test post-training static quantization on preserving
        pre forward and post forward hooks of original model
        """
        qengine = torch.backends.quantized.engine
        model = QuantStubModel()
        counter = {
            'pre_forwards': 0,
            'forwards': 0,
        }

        def fw_pre_hook(h_module, input):
            counter['pre_forwards'] += 1

        def fw_hook(h_module, input, output):
            counter['forwards'] += 1

        model.fc.register_forward_pre_hook(fw_pre_hook)
        model.fc.register_forward_hook(fw_hook)

        model.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
        model = prepare(model)

        def checkHooksIsPresent(model, before_convert=True):
            num_fwd_hooks = 1
            if before_convert:
                self.assertEqual(len(model.quant._forward_hooks.values()), 1,
                                 "Quantization observer hook has disappeared")
                num_fwd_hooks = 2

            self.assertObjectIn(fw_pre_hook, model.fc._forward_pre_hooks.values())
            self.assertObjectIn(fw_hook, model.fc._forward_hooks.values())
            self.assertEqual(len(model.fc._forward_pre_hooks.values()), 1,
                             "Extra pre forward hooks have appeared on a layer")
            # During static quantization non stub layers are provided with quantization observer hook too
            self.assertEqual(len(model.fc._forward_hooks.values()), num_fwd_hooks,
                             "Extra post forward hooks have appeared on a layer")
            # Implicitly check that fw_hook goes after _observer_forward_hook
            self.assertEqual(list(model.fc._forward_hooks.values())[-1], fw_hook,
                             "_observer_forward_hook is not a first entry of the hooks list")

        checkHooksIsPresent(model, True)
        test_only_eval_fn(model, self.calib_data)
        torch.ao.quantization.convert(model, inplace=True)
        checkHooksIsPresent(model, False)

    @skipIfNoFBGEMM
    def test_quantized_embedding(self):
        r""" Test the post-training quantization flow, serialization and scripting
        of embedding modules
        """

        for qconfig in [float_qparams_weight_only_qconfig, float_qparams_weight_only_qconfig_4bit]:
            model = EmbeddingModule().eval()
            indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])
            weights = torch.randn(10, 12, dtype=torch.float32)
            model.qconfig = qconfig
            prepare(model, inplace=True)
            convert(model, inplace=True)
            self.assertTrue('QuantizedEmbedding' in str(model))
            self.assertEqual(type(model.emb), torch.ao.nn.quantized.Embedding)
            self.checkScriptable(model, [[indices]], check_save_load=True)

            idx = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
            offsets = torch.LongTensor([0, 4])
            x = torch.randn(2, 4)
            model = EmbeddingWithStaticLinear().eval()
            prepare(model, inplace=True)
            convert(model, inplace=True)
            self.assertTrue('QuantizedEmbedding' in str(model))
            self.assertTrue('QuantizedLinear' in str(model))
            self.checkQuantizedLinear(model.fc)
            model(idx, offsets, x)

    @skipIfNoFBGEMM
    def test_dequant_stub(self):
        m = QuantStubModel().eval()
        prepare(m, inplace=True)
        self.checkObservers(m)
        convert(m, inplace=True)
        self.assertEqual(type(m.quant), nnq.Quantize)
        self.assertEqual(type(m.fc), nnq.Linear)
        self.assertEqual(type(m.dequant), nnq.DeQuantize)

        # check DeQuantStub is not swapped when it doesn't have a qconfig
        m2 = QuantStubModel().eval()
        m2.dequant.qconfig = None
        prepare(m2, inplace=True)
        self.checkObservers(m2)
        convert(m2, inplace=True)
        self.assertEqual(type(m2.quant), nnq.Quantize)
        self.assertEqual(type(m2.fc), nnq.Linear)
        self.assertEqual(type(m2.dequant), DeQuantStub)


    def test_quantized_embedding_bag(self):
        r""" Test the post-training quantization flow, serialization and scripting
        of embedding_bag modules
        """
        indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])
        offsets = torch.tensor([0, 19, 20, 28, 28, 32])
        weights = torch.randn(10, 12, dtype=torch.float32)

        for dtype in [torch.quint8, torch.quint4x2]:
            model = EmbeddingBagModule().eval()
            float_qparams_observer = PerChannelMinMaxObserver.with_args(dtype=dtype,
                                                                        qscheme=torch.per_channel_affine_float_qparams,
                                                                        ch_axis=0)
            float_qparams_qconfig = QConfig(activation=default_dynamic_quant_observer,
                                            weight=float_qparams_observer)
            model.qconfig = float_qparams_qconfig

            prepare(model, inplace=True)
            quantized_model = convert(model)

            per_sample_weights = torch.from_numpy(np.random.uniform(
                low=0.01, high=0.5, size=[len(indices)]).astype(np.float32))

            # Test to make sure module is quantized correctly.
            self.assertTrue('QuantizedEmbeddingBag' in str(quantized_model))
            self.checkDynamicQuantizedModule(quantized_model.emb, torch.ao.nn.quantized.EmbeddingBag, torch.quint8)
            self.checkScriptable(quantized_model, [[indices, offsets, per_sample_weights]], check_save_load=True)

            class EmbeddingBagWithLinear(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.emb = torch.nn.EmbeddingBag(num_embeddings=10, embedding_dim=12,
                                                     include_last_offset=True, scale_grad_by_freq=False, mode='sum')
                    self.fc = torch.nn.Linear(5, 5)

                def forward(self, indices, offsets, per_sample_weights, linear_in):
                    return self.emb(indices, offsets, per_sample_weights), self.fc(linear_in)

            # Test quantization of embedding_bag layer only
            model2 = EmbeddingBagWithLinear().eval()
            model2.emb.qconfig = float_qparams_qconfig
            prepare(model2, inplace=True)
            quantized_model = convert(model2)

            self.assertTrue('QuantizedEmbeddingBag' in str(quantized_model))
            self.checkLinear(model2.fc)
            self.checkDynamicQuantizedModule(quantized_model.emb, torch.ao.nn.quantized.EmbeddingBag, torch.quint8)

    @skipIfNoFBGEMM
    def test_custom_module_class(self):
        class CustomModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                return self.conv(x)

        class ObservedCustomModule(torch.nn.Module):
            def __init__(self, conv):
                super().__init__()
                self.conv = conv

            def forward(self, x):
                return self.conv(x)

            @classmethod
            def from_float(cls, float_module):
                assert hasattr(float_module, 'qconfig')
                observed = cls(float_module.conv)
                observed.qconfig = float_module.qconfig
                return observed

        class QuantizedCustomModule(torch.nn.Module):
            def __init__(self, conv):
                super().__init__()
                self.conv = conv

            def forward(self, x):
                return self.conv(x)

            @classmethod
            def from_observed(cls, observed_module):
                assert hasattr(observed_module, 'qconfig')
                assert hasattr(observed_module, 'activation_post_process')
                observed_module.conv.activation_post_process = \
                    observed_module.activation_post_process
                quantized = cls(nnq.Conv2d.from_float(observed_module.conv))
                return quantized

        class Sub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.custom = CustomModule()

            def forward(self, x):
                return self.custom(x)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = QuantStub()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.sub = Sub()
                self.dequant = DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv(x)
                x = self.sub(x)
                x = self.dequant(x)
                return x

        class RefM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = QuantStub()
                self.conv1 = torch.nn.Conv2d(1, 1, 1)
                self.conv2 = torch.nn.Conv2d(1, 1, 1)
                self.dequant = DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.dequant(x)
                return x

        data = torch.randn(1, 1, 1, 1)
        # instantiate M and RefM and align the parameters
        original_m = M()
        original_ref_m = RefM()
        original_ref_m.conv1.weight = torch.nn.Parameter(original_m.conv.weight.detach())
        original_ref_m.conv1.bias = torch.nn.Parameter(original_m.conv.bias.detach())
        original_ref_m.conv2.weight = torch.nn.Parameter(original_m.sub.custom.conv.weight.detach())
        original_ref_m.conv2.bias = torch.nn.Parameter(original_m.sub.custom.conv.bias.detach())

        original_m.qconfig = default_qconfig
        prepare_custom_config_dict = {
            "float_to_observed_custom_module_class": {
                CustomModule: ObservedCustomModule
            }
        }
        convert_custom_config_dict = {
            "observed_to_quantized_custom_module_class": {
                ObservedCustomModule: QuantizedCustomModule
            }
        }
        m = prepare(
            original_m,
            prepare_custom_config_dict=prepare_custom_config_dict)
        self.checkObservers(m, None, prepare_custom_config_dict)
        # calibration
        m(data)
        # all activation observers are inserted in the top level module

        # check converted/quantized model
        m = convert(
            m,
            convert_custom_config_dict=convert_custom_config_dict)
        # check if the module is properly quantized
        self.assertEqual(type(m.quant), nnq.Quantize)
        self.assertEqual(type(m.conv), nnq.Conv2d)
        self.assertEqual(type(m.sub), Sub)
        self.assertEqual(type(m.sub.custom), QuantizedCustomModule)
        self.assertEqual(type(m.sub.custom.conv), nnq.Conv2d)
        self.assertEqual(type(m.dequant), nnq.DeQuantize)
        res = m(data)

        # quantize the reference model
        original_ref_m.eval()
        original_ref_m.qconfig = default_qconfig
        ref_m = prepare(original_ref_m)
        ref_m(data)
        ref_m = convert(ref_m)
        ref_res = ref_m(data)
        self.assertEqual(res, ref_res)

    @skipIfNoFBGEMM
    def test_convtranspose_per_channel_fails_early(self):
        r"""
        Verifies that attempting to quantize a ConvTranspose module with per-Channel
        weight observers fails in the prepare step, as opposed to the convert step.
        """
        m = torch.nn.Sequential(torch.nn.ConvTranspose2d(1, 1, 1))
        m.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        with self.assertRaises(AssertionError) as context:
            mp = torch.ao.quantization.prepare(m)
        self.assertTrue(
            str(context.exception) ==
            'Per channel weight observer is not supported yet for ConvTranspose{n}d.')

    @skipIfNoFBGEMM
    def test_convtranspose_per_channel_qconfig_none(self):
        r"""
        Verifies that having qconfig==None for conv transpose does not crash
        """
        m = torch.nn.Sequential(torch.nn.ConvTranspose2d(1, 1, 1))
        m.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        m[0].qconfig = None
        mp = torch.ao.quantization.prepare(m)

    @skipIfNoFBGEMM
    def test_quantwrapper_attaches_qconfig_to_dequant(self):
        qconfig = torch.ao.quantization.default_qconfig

        m = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()
        for i in range(len(m)):
            m[i].qconfig = qconfig
            m[i] = torch.ao.quantization.QuantWrapper(m[i])

        mp = torch.ao.quantization.prepare(m)
        mq = torch.ao.quantization.convert(mp)
        self.assertTrue(isinstance(mq[0].dequant, nnq.DeQuantize))

    def test_activations_in_non_leaf_module_list(self):
        """
        Ensure activations like `nn.Sigmoid` and `nn.Tanh` are properly handled in
        `non_leaf_module_list`.
        """
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = QuantStub()
                self.sigmoid = torch.nn.Sigmoid()
                self.hardsigmoid = torch.nn.Hardsigmoid()
                self.softmax = torch.nn.Softmax()
                self.tanh = torch.nn.Tanh()
                self.dequant = DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.sigmoid(x)
                x = self.hardsigmoid(x)
                x = self.softmax(x)
                x = self.tanh(x)
                x = self.dequant(x)
                return x

        qconfig = QConfig(
            activation=FixedQParamsObserver.with_args(scale=123.0, zero_point=0),
            weight=default_weight_observer
        )
        m = MyModel()
        m.qconfig = qconfig
        m = prepare(m, observer_non_leaf_module_list=[
            torch.nn.Sigmoid,
            torch.nn.Hardsigmoid,
            torch.nn.Softmax,
            torch.nn.Tanh,
        ])

        # Should use the observer specified in the QConfig instead of the default (FixedQParamsFakeQuantize)
        self.assertTrue(isinstance(m.sigmoid.activation_post_process, FixedQParamsObserver))
        self.assertTrue(isinstance(m.hardsigmoid.activation_post_process, FixedQParamsObserver))
        self.assertTrue(isinstance(m.softmax.activation_post_process, FixedQParamsObserver))
        self.assertTrue(isinstance(m.tanh.activation_post_process, FixedQParamsObserver))

    @skipIfNoFBGEMM
    def test_mha_batch_first_attr_is_copied_in_prepare(self):
        class TransformerDecoderLayer(nn.Module):
            def __init__(self, d_model, nhead, batch_first):
                super().__init__()
                self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=batch_first)

        qengine = torch.backends.quantized.engine
        for batch_first in [True, False]:
            model = TransformerDecoderLayer(512, 8, batch_first)
            quantization_config = torch.ao.quantization.get_default_qconfig(qengine)
            model.qconfig = quantization_config
            prepared_model = torch.ao.quantization.prepare(model, inplace=False)
            self.assertTrue(prepared_model.self_attn.batch_first == model.self_attn.batch_first)

@skipIfNoFBGEMM
class TestQuantizeEagerPTQDynamic(QuantizationTestCase):
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
                self.checkNoQconfig(model)

            checkQuantized(model)

            # test one line API - out of place version
            base = SingleLayerLinearDynamicModel()
            keys_before = set(base.state_dict().keys())
            model = quantize_dynamic(base, qconfig_dict)
            checkQuantized(model)
            keys_after = set(base.state_dict().keys())
            self.assertEqual(keys_before, keys_after)  # simple check that nothing changed

            # in-place version
            model = SingleLayerLinearDynamicModel()
            quantize_dynamic(model, qconfig_dict, inplace=True)
            checkQuantized(model)

            # Test set qconfig
            model = SingleLayerLinearDynamicModel()
            quantize_dynamic(model, {nn.Linear}, inplace=True, dtype=dtype)
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
                self.checkNoQconfig(model)

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
                self.checkNoQconfig(model)

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
                self.checkNoQconfig(model)

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
                self.checkNoQconfig(model)

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
                self.checkNoQconfig(model)

            checkQuantized(model)

            # test one line API
            model = quantize_dynamic(NestedModel().eval(), qconfig_dict, dtype=dtype)
            checkQuantized(model)

    def test_per_channel_linear_quantize(self):
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
            self.checkNoQconfig(model)

        checkQuantized(model)
        # test one line API
        model = quantize_dynamic(NestedModel().eval(), qconfig_dict)
        checkQuantized(model)

    def test_linear_relu_fusion(self):
        dtype = torch.qint8
        model = LinearReluLinearModel().eval()
        qconfig = default_dynamic_qconfig
        qconfig_dict = {'' : qconfig}
        torch.ao.quantization.fuse_modules(model, [['fc1', 'relu']], inplace=True)
        prepare_dynamic(model, qconfig_dict)
        convert_dynamic(model)

        def checkQuantized(model):
            self.checkDynamicQuantizedLinearRelu(model.fc1, dtype)
            self.checkDynamicQuantizedLinear(model.fc2, dtype)
            self.checkScriptable(model, self.calib_data, check_save_load=True)
            self.checkNoQconfig(model)

        checkQuantized(model)

    @given(qconfig=st.sampled_from([per_channel_dynamic_qconfig, default_dynamic_qconfig]),
           dtype=st.sampled_from([torch.qint8, torch.float16]))
    def test_quantized_rnn(self, qconfig, dtype):
        r"""Test dynamic quantization, scriptability and serialization for dynamic quantized lstm modules on int8 and fp16
        """
        niter = 10
        x = torch.tensor([[100, -155],
                          [-155, 100],
                          [100, -155]], dtype=torch.float).unsqueeze(0).repeat(niter, 1, 1)
        qconfig_dict = {
            torch.nn.LSTM : qconfig,
            torch.nn.GRU: qconfig
        }

        def checkQuantized(model, module_type):
            mod_type_map = {'LSTM': torch.ao.nn.quantized.dynamic.LSTM,
                            'GRU': torch.ao.nn.quantized.dynamic.GRU}
            mod_repr_map = {'LSTM': 'DynamicQuantizedLSTM',
                            'GRU': 'DynamicQuantizedGRU'}
            self.assertTrue(mod_repr_map[module_type] in str(model_quantized))
            self.checkDynamicQuantizedModule(model_quantized.mod, mod_type_map[module_type], dtype)

        for module_type in ['LSTM', 'GRU']:
            model = RNNDynamicModel(module_type).eval()

            if dtype == torch.float16:
                model_quantized = quantize_dynamic(model=model, dtype=dtype)
            else:
                model_quantized = quantize_dynamic(model=model, qconfig_spec=qconfig_dict, dtype=dtype)

            checkQuantized(model_quantized, module_type)
            self.checkScriptable(model_quantized, [[x]], check_save_load=True)

            class ScriptWrapperPackedLSTM(torch.nn.Module):
                def __init__(self, cell):
                    super().__init__()
                    self.cell = cell

                def forward(self, x: PackedSequence) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]:
                    return self.cell(x)

            class ScriptWrapperPackedGRU(torch.nn.Module):
                def __init__(self, cell):
                    super().__init__()
                    self.cell = cell

                def forward(self, x: PackedSequence) -> Tuple[PackedSequence, torch.Tensor]:
                    return self.cell(x)

            script_wrapper_map = {'LSTM': ScriptWrapperPackedLSTM,
                                  'GRU': ScriptWrapperPackedGRU}
            packed_input = torch.nn.utils.rnn.pack_padded_sequence(x, torch.tensor([10, 5, 2]))
            model_with_packed_input = script_wrapper_map[module_type](model_quantized.mod)
            model_with_packed_input(packed_input)
            scripted = torch.jit.script(model_with_packed_input)
            scripted(packed_input)
            # We cannot trace with input dtype being a packed sequence
            self._checkScriptable(model_with_packed_input, scripted, [[packed_input]], True)


    @given(qconfig=st.sampled_from([per_channel_dynamic_qconfig, default_dynamic_qconfig]),
           dtype=st.sampled_from([torch.qint8, torch.float16]))
    def test_quantized_rnn_cell(self, qconfig, dtype):
        r"""Test dynamic quantization, scriptability and serialization for dynamic quantized rnn cell modules on int8 and fp16
        """
        qconfig_dict = {
            torch.nn.LSTMCell : qconfig,
            torch.nn.GRUCell : qconfig,
            torch.nn.RNNCell : qconfig
        }

        for module_type in ['LSTMCell', 'GRUCell', 'RNNTanh', 'RNNReLU']:
            model = RNNCellDynamicModel(module_type).eval()
            x = torch.tensor([[100, -155],
                             [-155, 100],
                             [100, -155]], dtype=torch.float)

            if torch.backends.quantized.engine == 'qnnpack' and dtype == torch.float16:
                continue
                # fp16 dynamic quant is not supported for qnnpack

            if dtype == torch.float16:
                model_quantized = quantize_dynamic(model=model, dtype=dtype)
            else:
                model_quantized = quantize_dynamic(model=model, qconfig_spec=qconfig_dict, dtype=dtype)

            def checkQuantized(model, module_type):
                mod_type_map = {'LSTMCell': torch.ao.nn.quantized.dynamic.LSTMCell,
                                'GRUCell': torch.ao.nn.quantized.dynamic.GRUCell,
                                'RNNTanh': torch.ao.nn.quantized.dynamic.RNNCell,
                                'RNNReLU': torch.ao.nn.quantized.dynamic.RNNCell}

                mod_repr_map = {'LSTMCell': 'DynamicQuantizedLSTMCell',
                                'GRUCell': 'DynamicQuantizedGRUCell',
                                'RNNTanh': 'DynamicQuantizedRNNCell',
                                'RNNReLU': 'DynamicQuantizedRNNCell'}

                self.assertTrue(mod_repr_map[module_type] in str(model_quantized))
                self.checkDynamicQuantizedModule(model_quantized.mod, mod_type_map[module_type], dtype)
                self.checkNoQconfig(model)

            # Smoke test extra reprs
            checkQuantized(model_quantized, module_type)
            self.checkScriptable(model_quantized, [[x]], check_save_load=True)


    def test_forward_hooks_preserved(self):
        r"""Test post-training dynamic quantization on preserving
        pre forward and post forward hooks of original model
        """
        for dtype in [torch.qint8, torch.float16]:
            model = SingleLayerLinearDynamicModel().eval()
            qconfig = float16_dynamic_qconfig if dtype == torch.float16 else default_dynamic_qconfig
            qconfig_dict = {
                'fc1': qconfig
            }
            convert_dynamic(model)

            counter = {
                'pre_forwards': 0,
                'forwards': 0,
            }

            def fw_pre_hook(h_module, input):
                counter['pre_forwards'] += 1

            def fw_hook(h_module, input, output):
                counter['forwards'] += 1

            model.fc1.register_forward_pre_hook(fw_pre_hook)
            model.fc1.register_forward_hook(fw_hook)
            prepare_dynamic(model, qconfig_dict)

            def checkHooksIsPresent(model):
                self.assertObjectIn(fw_pre_hook, model.fc1._forward_pre_hooks.values())
                self.assertObjectIn(fw_hook, model.fc1._forward_hooks.values())
                self.assertEqual(len(model.fc1._forward_pre_hooks.values()), 1,
                                 "Extra pre forward hooks have appeared on a layer")
                self.assertEqual(len(model.fc1._forward_hooks.values()), 1,
                                 "Extra post forward hooks have appeared on a layer")

            checkHooksIsPresent(model)
            test_only_eval_fn(model, self.calib_data)
            convert_dynamic(model)
            checkHooksIsPresent(model)

    @skipIfNoFBGEMM
    def test_embedding_bag_dynamic(self):
        class EmbeddingBagWithLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.EmbeddingBag(num_embeddings=10, embedding_dim=12,
                                                 include_last_offset=True, scale_grad_by_freq=False, mode='sum')
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, indices, offsets, linear_in):
                return self.emb(indices, offsets), self.fc(linear_in)
        model = EmbeddingBagWithLinear().eval()

        qconfig_dict = {
            torch.nn.EmbeddingBag : float_qparams_weight_only_qconfig,
            torch.nn.Linear: default_dynamic_qconfig
        }
        indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])
        offsets = torch.tensor([0, 19, 20, 28, 28, 32])
        q_model = quantize_dynamic(model, qconfig_dict)

        q_model(indices, offsets, torch.randn(5, 5))
        self.assertTrue('QuantizedEmbeddingBag' in str(q_model.emb))
        self.assertTrue('DynamicQuantizedLinear' in str(q_model.fc))

    @skipIfNoFBGEMM
    def test_embedding_op_dynamic(self):
        class EmbeddingWithLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=12,
                                                scale_grad_by_freq=False)
                self.fc = torch.nn.Linear(5, 5)
            def forward(self, indices,linear_in):
                return self.emb(indices), self.fc(linear_in)
        model = EmbeddingWithLinear().eval()
        qconfig_dict = {
            torch.nn.Embedding : float_qparams_weight_only_qconfig,
            torch.nn.Linear: default_dynamic_qconfig
        }
        indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])
        q_model = quantize_dynamic(model, qconfig_dict)
        self.assertTrue('QuantizedEmbedding' in str(q_model.emb))
        self.assertTrue('DynamicQuantizedLinear' in str(q_model.fc))
        q_model(indices, torch.randn(5, 5))

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_quantization.py TESTNAME\n\n"
                       "instead.")
