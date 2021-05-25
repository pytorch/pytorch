
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nniq
import torch.nn.intrinsic.quantized._reference as nniqr
import torch.nn.quantized.functional as qF
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
import torch.nn.quantized._reference as nnqr
from torch.nn.utils.rnn import PackedSequence
from torch.quantization import (
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
    default_float_qparams_observer,
    per_channel_dynamic_qconfig,
    float16_dynamic_qconfig,
    float_qparams_weight_only_qconfig,
    PerChannelMinMaxObserver,
    QConfigDynamic,
    default_dynamic_quant_observer,
    get_default_static_quant_module_mappings,
)

from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    AnnotatedSingleLayerLinearModel,
    QuantStubModel,
    ModelWithFunctionals,
    ModelMultipleOps,
    ModelMultipleOpsNoAvgPool,
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
    EmbeddingWithLinear,
    _make_conv_test_input,
    lengths_to_offsets
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
    _calculate_dynamic_qparams,
)

from torch.testing._internal.common_utils import (
    IS_PPC,
    TEST_WITH_UBSAN,
)

from torch.testing._internal.jit_utils import JitTestCase
from hypothesis import assume, given
from hypothesis import strategies as st
import torch.testing._internal.hypothesis_utils as hu
hu.assert_deadline_disabled()

# Standard library
from typing import Tuple
import copy
import io
import unittest
import numpy as np
import itertools

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
                    self.checkNoQconfig(model)

                checkQuantized(model)

                # test one line API - out of place version
                base = AnnotatedSingleLayerLinearModel(qengine)
                base.qconfig = qconfig
                keys_before = set(list(base.state_dict().keys()))
                model = quantize(base, test_only_eval_fn, [self.calib_data])
                checkQuantized(model)
                keys_after = set(list(base.state_dict().keys()))
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
                qconfig = torch.quantization.get_default_qconfig(qengine)
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
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
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

        model.qconfig = torch.quantization.get_default_qconfig(qengine)
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
        torch.quantization.convert(model, inplace=True)
        checkHooksIsPresent(model, False)

    @skipIfNoFBGEMM
    def test_quantized_embedding(self):
        r""" Test the post-training quantization flow, serialization and scripting
        of embedding modules
        """
        model = EmbeddingModule().eval()
        indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])
        weights = torch.randn(10, 12, dtype=torch.float32)
        model.qconfig = float_qparams_weight_only_qconfig
        prepare(model, inplace=True)
        convert(model, inplace=True)
        self.assertTrue('QuantizedEmbedding' in str(model))
        self.assertEqual(type(model.emb), torch.nn.quantized.Embedding)
        self.checkScriptable(model, [[indices]], check_save_load=True)

        model = EmbeddingWithLinear().eval()
        prepare(model, inplace=True)
        convert(model, inplace=True)
        self.assertTrue('QuantizedEmbedding' in str(model))
        self.assertTrue('QuantizedLinear' in str(model))
        self.checkQuantizedLinear(model.fc)

    @skipIfNoFBGEMM
    def test_embedding_linear_dynamic(self):
        class EmbeddingWithLinearDynamic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=12)
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, indices, linear_in):
                return self.emb(indices), self.fc(linear_in)

        model = EmbeddingWithLinearDynamic()
        qconfig_dict = {'fc' : default_dynamic_qconfig}
        model = EmbeddingWithLinear()
        quantize_dynamic(model, qconfig_dict, inplace=True)

        model.emb.qconfig = float_qparams_weight_only_qconfig
        prepare(model, inplace=True)
        convert(model, inplace=True)
        self.assertTrue('QuantizedEmbedding' in str(model))
        self.assertTrue('DynamicQuantizedLinear' in str(model))


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
            float_qparams_qconfig = QConfigDynamic(activation=default_dynamic_quant_observer,
                                                   weight=float_qparams_observer)
            model.qconfig = float_qparams_qconfig

            prepare(model, inplace=True)
            quantized_model = convert(model)

            per_sample_weights = torch.from_numpy(np.random.uniform(
                low=0.01, high=0.5, size=[len(indices)]).astype(np.float32))

            # Test to make sure module is quantized correctly.
            self.assertTrue('QuantizedEmbeddingBag' in str(quantized_model))
            self.checkDynamicQuantizedModule(quantized_model.emb, torch.nn.quantized.EmbeddingBag, torch.quint8)
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
            self.checkDynamicQuantizedModule(quantized_model.emb, torch.nn.quantized.EmbeddingBag, torch.quint8)

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

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = QuantStub()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.custom = CustomModule()
                self.dequant = DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv(x)
                x = self.custom(x)
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
        original_ref_m.conv2.weight = torch.nn.Parameter(original_m.custom.conv.weight.detach())
        original_ref_m.conv2.bias = torch.nn.Parameter(original_m.custom.conv.bias.detach())

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
        self.assertEqual(type(m.custom.conv), nnq.Conv2d)
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
        m.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        with self.assertRaises(AssertionError) as context:
            mp = torch.quantization.prepare(m)
        self.assertTrue(
            str(context.exception) ==
            'Per channel weight observer is not supported yet for ConvTranspose{n}d.')

    @skipIfNoFBGEMM
    def test_convtranspose_per_channel_qconfig_none(self):
        r"""
        Verifies that having qconfig==None for conv transpose does not crash
        """
        m = torch.nn.Sequential(torch.nn.ConvTranspose2d(1, 1, 1))
        m.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        m[0].qconfig = None
        mp = torch.quantization.prepare(m)


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
                self.checkNoQconfig(model)

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
            mod_type_map = {'LSTM': torch.nn.quantized.dynamic.LSTM,
                            'GRU': torch.nn.quantized.dynamic.GRU}
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
                    super(ScriptWrapperPackedLSTM, self).__init__()
                    self.cell = cell

                def forward(self, x: PackedSequence) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]:
                    return self.cell(x)

            class ScriptWrapperPackedGRU(torch.nn.Module):
                def __init__(self, cell):
                    super(ScriptWrapperPackedGRU, self).__init__()
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
                mod_type_map = {'LSTMCell': torch.nn.quantized.dynamic.LSTMCell,
                                'GRUCell': torch.nn.quantized.dynamic.GRUCell,
                                'RNNTanh': torch.nn.quantized.dynamic.RNNCell,
                                'RNNReLU': torch.nn.quantized.dynamic.RNNCell}

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

class TestEagerModeActivationOps(QuantizationTestCase):
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

class TestFunctionalModule(QuantizationTestCase):
    # Histogram Observers are slow, so have no-deadline to ensure test doesn't time out
    @given(train_mode=st.booleans())
    def test_functional_module(self, train_mode):
        model = ModelWithFunctionals()
        x = torch.rand(10, 1, dtype=torch.float)
        xq = torch.quantize_per_tensor(x, 0.01, 30, torch.quint8)
        self.checkScriptable(model, [[x]], check_save_load=True)
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
            self.checkNoQconfig(model)

        checkQuantized(model)
        self.checkScriptable(model, [[xq]], check_save_load=True)

class TestQuantizedFunctionalOps(QuantizationTestCase):
    def test_relu_api(self):
        X = torch.arange(-5, 5, dtype=torch.float)
        scale = 2.0
        zero_point = 1
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point, dtype=torch.quint8)
        qY = torch.relu(qX)
        qY_hat = F.relu(qX)
        self.assertEqual(qY, qY_hat)

    def _test_conv_api_impl(
        self, qconv_fn, conv_fn, batch_size, in_channels_per_group,
        input_feature_map_size, out_channels_per_group, groups, kernel_size,
        stride, padding, dilation, X_scale, X_zero_point, W_scale, W_zero_point,
        Y_scale, Y_zero_point, use_bias, use_channelwise,
    ):
        for i in range(len(kernel_size)):
            assume(input_feature_map_size[i] + 2 * padding[i]
                   >= dilation[i] * (kernel_size[i] - 1) + 1)
        (X, X_q, W, W_q, b) = _make_conv_test_input(
            batch_size, in_channels_per_group, input_feature_map_size,
            out_channels_per_group, groups, kernel_size, X_scale,
            X_zero_point, W_scale, W_zero_point, use_bias, use_channelwise)

        Y_exp = conv_fn(X, W, b, stride, padding, dilation, groups)
        Y_exp = torch.quantize_per_tensor(
            Y_exp, scale=Y_scale, zero_point=Y_zero_point, dtype=torch.quint8)
        Y_act = qconv_fn(
            X_q, W_q, b, stride, padding, dilation, groups,
            padding_mode="zeros", scale=Y_scale, zero_point=Y_zero_point)

        # Make sure the results match
        # assert_array_almost_equal compares using the following formula:
        #     abs(desired-actual) < 1.5 * 10**(-decimal)
        # (https://docs.scipy.org/doc/numpy/reference/generated/numpy.testing.assert_almost_equal.html)
        # We use decimal = 0 to ignore off-by-1 differences between reference
        # and test. Off-by-1 differences arise due to the order of round and
        # zero_point addition operation, i.e., if addition followed by round is
        # used by reference and round followed by addition is used by test, the
        # results may differ by 1.
        # For example, the result of round(2.5) + 1 is 3 while round(2.5 + 1) is
        # 4 assuming the rounding mode is round-to-nearest, ties-to-even.
        np.testing.assert_array_almost_equal(
            Y_exp.int_repr().numpy(), Y_act.int_repr().numpy(), decimal=0)

    @given(batch_size=st.integers(1, 3),
           in_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           L=st.integers(4, 16),
           out_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           groups=st.integers(1, 4),
           kernel=st.integers(1, 7),
           stride=st.integers(1, 2),
           pad=st.integers(0, 2),
           dilation=st.integers(1, 2),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.integers(0, 4),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(-5, 5), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.integers(0, 4),
           use_bias=st.booleans(),
           use_channelwise=st.booleans(),
           qengine=st.sampled_from(("qnnpack", "fbgemm")))
    def test_conv1d_api(
        self, batch_size, in_channels_per_group, L, out_channels_per_group,
        groups, kernel, stride, pad, dilation,
        X_scale, X_zero_point, W_scale, W_zero_point, Y_scale, Y_zero_point,
        use_bias, use_channelwise, qengine,
    ):
        # Tests the correctness of the conv1d function.
        if qengine not in torch.backends.quantized.supported_engines:
            return
        if qengine == 'qnnpack':
            if IS_PPC or TEST_WITH_UBSAN:
                return
            use_channelwise = False

        input_feature_map_size = (L, )
        kernel_size = (kernel, )
        stride = (stride, )
        padding = (pad, )
        dilation = (dilation, )

        with override_quantized_engine(qengine):
            qconv_fn = qF.conv1d
            conv_fn = F.conv1d
            self._test_conv_api_impl(
                qconv_fn, conv_fn, batch_size, in_channels_per_group,
                input_feature_map_size, out_channels_per_group, groups,
                kernel_size, stride, padding, dilation, X_scale, X_zero_point,
                W_scale, W_zero_point, Y_scale, Y_zero_point, use_bias,
                use_channelwise)

    @given(batch_size=st.integers(1, 3),
           in_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           H=st.integers(4, 16),
           W=st.integers(4, 16),
           out_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           groups=st.integers(1, 4),
           kernel_h=st.integers(1, 7),
           kernel_w=st.integers(1, 7),
           stride_h=st.integers(1, 2),
           stride_w=st.integers(1, 2),
           pad_h=st.integers(0, 2),
           pad_w=st.integers(0, 2),
           dilation=st.integers(1, 2),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.integers(0, 4),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(-5, 5), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.integers(0, 4),
           use_bias=st.booleans(),
           use_channelwise=st.booleans(),
           qengine=st.sampled_from(("qnnpack", "fbgemm")))
    def test_conv2d_api(
        self, batch_size, in_channels_per_group, H, W, out_channels_per_group,
        groups, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation,
        X_scale, X_zero_point, W_scale, W_zero_point, Y_scale, Y_zero_point,
        use_bias, use_channelwise, qengine,
    ):
        # Tests the correctness of the conv2d function.

        if qengine not in torch.backends.quantized.supported_engines:
            return
        if qengine == 'qnnpack':
            if IS_PPC or TEST_WITH_UBSAN:
                return

        input_feature_map_size = (H, W)
        kernel_size = (kernel_h, kernel_w)
        stride = (stride_h, stride_w)
        padding = (pad_h, pad_w)
        dilation = (dilation, dilation)

        with override_quantized_engine(qengine):
            qconv_fn = qF.conv2d
            conv_fn = F.conv2d
            self._test_conv_api_impl(
                qconv_fn, conv_fn, batch_size, in_channels_per_group,
                input_feature_map_size, out_channels_per_group, groups,
                kernel_size, stride, padding, dilation, X_scale, X_zero_point,
                W_scale, W_zero_point, Y_scale, Y_zero_point, use_bias,
                use_channelwise)

    @given(batch_size=st.integers(1, 3),
           in_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           D=st.integers(4, 8),
           H=st.integers(4, 8),
           W=st.integers(4, 8),
           out_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           groups=st.integers(1, 4),
           kernel_d=st.integers(1, 4),
           kernel_h=st.integers(1, 4),
           kernel_w=st.integers(1, 4),
           stride_d=st.integers(1, 2),
           stride_h=st.integers(1, 2),
           stride_w=st.integers(1, 2),
           pad_d=st.integers(0, 2),
           pad_h=st.integers(0, 2),
           pad_w=st.integers(0, 2),
           dilation=st.integers(1, 2),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.integers(0, 4),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(-5, 5), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.integers(0, 4),
           use_bias=st.booleans(),
           use_channelwise=st.booleans(),
           qengine=st.sampled_from(("fbgemm",)))
    def test_conv3d_api(
        self, batch_size, in_channels_per_group, D, H, W,
        out_channels_per_group, groups, kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dilation, X_scale,
        X_zero_point, W_scale, W_zero_point, Y_scale, Y_zero_point, use_bias,
        use_channelwise, qengine,
    ):
        # Tests the correctness of the conv3d function.
        # Currently conv3d only supports FbGemm engine

        if qengine not in torch.backends.quantized.supported_engines:
            return

        input_feature_map_size = (D, H, W)
        kernel_size = (kernel_d, kernel_h, kernel_w)
        stride = (stride_d, stride_h, stride_w)
        padding = (pad_d, pad_h, pad_w)
        dilation = (dilation, dilation, dilation)

        with override_quantized_engine(qengine):
            qconv_fn = qF.conv3d
            conv_fn = F.conv3d
            self._test_conv_api_impl(
                qconv_fn, conv_fn, batch_size, in_channels_per_group,
                input_feature_map_size, out_channels_per_group, groups,
                kernel_size, stride, padding, dilation, X_scale, X_zero_point,
                W_scale, W_zero_point, Y_scale, Y_zero_point, use_bias,
                use_channelwise)

class TestStaticQuantizedModule(QuantizationTestCase):
    """
    Note that tests in this class are just API test, to make sure we wrapped the
    quantized operator implementations correctly in the user facing APIs, these are
    not correctness test for the underlying quantized operators. For correctness
    test please see `test/quantization/test_quantized_op.py`.
    """
    def test_relu(self):
        relu_module = nn.ReLU()
        relu6_module = nnq.ReLU6()

        x = torch.arange(-10, 10, dtype=torch.float)
        y_ref = torch.relu(x)
        y6_ref = torch.nn.modules.ReLU6()(x)

        qx = torch.quantize_per_tensor(x, 1.0, 0, dtype=torch.qint32)
        qy = relu_module(qx)
        qy6 = relu6_module(qx)

        self.assertEqual(y_ref, qy.dequantize(),
                         msg="ReLU module API failed")
        self.assertEqual(y6_ref, qy6.dequantize(),
                         msg="ReLU6 module API failed")

    @override_qengines
    def test_linear_api(self):
        """test API functionality for nn.quantized.linear and nn.intrinsic.quantized.linear_relu"""
        options = itertools.product(
            [1, 5],
            [16, 32],
            [4, 8],
            [True, False],
            [True, False],
            [True, False],
            [True, False])
        for (batch_size, in_features, out_features, use_bias,
             use_fused, per_channel, is_reference) in options:
            self._test_linear_api_impl(
                batch_size, in_features, out_features, use_bias, use_fused,
                per_channel, is_reference)

    def _test_linear_api_impl(self, batch_size, in_features, out_features, use_bias, use_fused, per_channel, is_reference):
        if torch.backends.quantized.engine == 'qnnpack':
            per_channel = False

        # (use_fused, is_reference) -> quantized class
        class_map = {
            (True, True) : nniqr.LinearReLU,
            (True, False) : nniq.LinearReLU,
            (False, True) : nnqr.Linear,
            (False, False) : nnq.Linear,
        }

        W = torch.rand(out_features, in_features).float()
        if per_channel:
            scale_tensor = torch.ones(out_features, dtype=torch.double)
            zero_point_tensor = torch.zeros(out_features, dtype=torch.long)
            for i in range(len(scale_tensor)):
                scale_tensor[i] = (i + 1.0) / 255.0
            W_q = torch.quantize_per_channel(W, scales=scale_tensor,
                                             zero_points=zero_point_tensor,
                                             axis=0, dtype=torch.qint8)
        else:
            W_q = torch.quantize_per_tensor(W, 0.1, 4, torch.qint8)

        X = torch.rand(batch_size, in_features).float()
        X_q = torch.quantize_per_tensor(X, 0.2, 10, torch.quint8)
        B = torch.rand(out_features).float() if use_bias else None
        scale = 0.5
        zero_point = 3
        qlinear = class_map[(use_fused, is_reference)](in_features, out_features)

        qlinear_copy = qlinear  # deepcopy does not work right now
        # qlinear_copy = copy.deepcopy(qlinear)
        self.checkScriptable(qlinear_copy, [[X_q]], check_save_load=True)
        # Run module with default-initialized parameters.
        # This tests that the constructor is correct.
        qlinear(X_q)

        qlinear.set_weight_bias(W_q, B)
        # Simple round-trip test to ensure weight()/set_weight() API
        self.assertEqual(qlinear.weight(), W_q, atol=1e-5, rtol=0)

        # testing packed param implementation
        qlinear.scale = float(scale)
        qlinear.zero_point = int(zero_point)
        Z_q = qlinear(X_q)

        # Check if the module implementation matches calling the
        # ops directly
        if is_reference:
            weight = qlinear._qweight
            bias = qlinear._bias
            weight_dequant = weight.dequantize()
            X_q_dq = X_q.dequantize()
            Z_ref = F.linear(X_q_dq, weight_dequant, bias)
            if use_fused:
                Z_ref = F.relu(Z_ref, inplace=True)
            Z_ref = torch.quantize_per_tensor(Z_ref, scale, zero_point, torch.quint8)
        else:
            W_pack = qlinear._packed_params._packed_params
            if use_fused:
                Z_ref = torch.ops.quantized.linear_relu(X_q, W_pack, scale, zero_point)
            else:
                Z_ref = torch.ops.quantized.linear(X_q, W_pack, scale, zero_point)

        self.assertEqual(Z_ref, Z_q)
        self.assertTrue(
            ("QuantizedLinearReLU" if use_fused else "QuantizedLinear") in str(qlinear))

        # Test serialization of quantized Linear Module using state_dict
        model_dict = qlinear.state_dict()
        b = io.BytesIO()
        torch.save(model_dict, b)
        b.seek(0)
        loaded_dict = torch.load(b)
        for key in model_dict:
            if isinstance(model_dict[key], torch._C.ScriptObject):
                assert isinstance(loaded_dict[key], torch._C.ScriptObject)
                w_model, b_model = torch.ops.quantized.linear_unpack(model_dict[key])
                w_loaded, b_loaded = torch.ops.quantized.linear_unpack(loaded_dict[key])
                self.assertEqual(w_model, w_loaded)
                self.assertEqual(b_model, b_loaded)
            else:
                self.assertEqual(model_dict[key], loaded_dict[key])

        loaded_qlinear = class_map[(use_fused, is_reference)](
            in_features, out_features)
        loaded_qlinear.load_state_dict(loaded_dict)
        if is_reference:
            self.assertEqual(qlinear._qweight, loaded_qlinear._qweight)
            self.assertEqual(qlinear._bias, loaded_qlinear._bias)
        else:
            linear_unpack = torch.ops.quantized.linear_unpack
            self.assertEqual(linear_unpack(qlinear._packed_params._packed_params),
                             linear_unpack(loaded_qlinear._packed_params._packed_params))
        self.assertEqual(qlinear.scale, loaded_qlinear.scale)
        self.assertEqual(qlinear.zero_point, loaded_qlinear.zero_point)
        # make sure loaded_qlinear has the same dir as qlinear since
        # scripting the module will add __overloads__ to __dict__
        self.checkScriptable(loaded_qlinear, [[X_q]], check_save_load=True)
        self.assertTrue(dir(qlinear) == dir(loaded_qlinear))
        self.assertEqual(qlinear._weight_bias(), loaded_qlinear._weight_bias())
        if not is_reference:
            self.assertEqual(qlinear._weight_bias(), torch.ops.quantized.linear_unpack(qlinear._packed_params._packed_params))
        Z_q2 = loaded_qlinear(X_q)
        self.assertEqual(Z_q, Z_q2)

        b = io.BytesIO()
        torch.save(qlinear, b)
        b.seek(0)
        loaded = torch.load(b)
        self.assertEqual(qlinear.weight(), loaded.weight())
        self.assertEqual(qlinear.scale, loaded.scale)
        self.assertEqual(qlinear.zero_point, loaded.zero_point)

        # Test JIT
        self.checkScriptable(qlinear, [[X_q]], check_save_load=True)

        # Make sure `from_float` works for all linear variants
        modules_under_test = [torch.nn.Linear, torch.nn.modules.linear._LinearWithBias]

        for mut in modules_under_test:
            # Test from_float.
            float_linear = mut(in_features, out_features).float()
            float_linear.qconfig = torch.quantization.default_qconfig
            torch.quantization.prepare(float_linear, inplace=True)
            float_linear(X.float())
            # Sequential allows swapping using "convert".
            quantized_float_linear = torch.nn.Sequential(float_linear)
            quantized_float_linear = torch.quantization.convert(quantized_float_linear, inplace=True)

            # Smoke test to make sure the module actually runs
            quantized_float_linear(X_q)

            # Smoke test extra_repr
            self.assertTrue('QuantizedLinear' in str(quantized_float_linear))

    def test_quant_dequant_api(self):
        r = torch.tensor([[1., -1.], [1., -1.]], dtype=torch.float)
        scale, zero_point, dtype = 1.0, 2, torch.qint8
        # testing Quantize API
        qr = torch.quantize_per_tensor(r, scale, zero_point, dtype)
        quant_m = nnq.Quantize(scale, zero_point, dtype)
        qr2 = quant_m(r)
        self.assertEqual(qr, qr2)
        # testing Dequantize API
        rqr = qr.dequantize()
        dequant_m = nnq.DeQuantize()
        rqr2 = dequant_m(qr2)
        self.assertEqual(rqr, rqr2)

    def _test_conv_api_impl(
        self, module_name, qconv_module, conv_module, batch_size,
        in_channels_per_group, input_feature_map_size, out_channels_per_group,
        groups, kernel_size, stride, padding, padding_mode, dilation,
        X_scale, X_zero_point, W_scale, W_zero_point, Y_scale, Y_zero_point,
        use_bias, use_fused, use_channelwise, is_reference
    ):
        for i in range(len(kernel_size)):
            assume(input_feature_map_size[i] + 2 * padding[i]
                   >= dilation[i] * (kernel_size[i] - 1) + 1)

        in_channels = in_channels_per_group * groups
        out_channels = out_channels_per_group * groups
        (X, X_q, W, W_q, b) = _make_conv_test_input(
            batch_size, in_channels_per_group, input_feature_map_size,
            out_channels_per_group, groups, kernel_size, X_scale, X_zero_point,
            W_scale, W_zero_point, use_bias, use_channelwise)

        qconv_module.set_weight_bias(W_q, b)
        qconv_module.scale = Y_scale
        qconv_module.zero_point = Y_zero_point

        if use_fused:
            conv_module[0].weight.data = W
            if use_bias:
                conv_module[0].bias.data = b
        else:
            conv_module.weight.data = W
            if use_bias:
                conv_module.bias.data = b

        # Test members
        self.assertTrue(module_name == qconv_module._get_name(), module_name + " " + qconv_module._get_name())
        if not is_reference:
            self.assertTrue(hasattr(qconv_module, '_packed_params'))
        self.assertTrue(hasattr(qconv_module, 'scale'))
        self.assertTrue(hasattr(qconv_module, 'zero_point'))

        # Test properties
        self.assertEqual(W_q, qconv_module.weight())
        if use_bias:
            self.assertEqual(b, qconv_module.bias())
        self.assertEqual(Y_scale, qconv_module.scale)
        self.assertEqual(Y_zero_point, qconv_module.zero_point)

        # Test forward
        Y_exp = conv_module(X)
        Y_exp = torch.quantize_per_tensor(
            Y_exp, scale=Y_scale, zero_point=Y_zero_point, dtype=torch.quint8)
        Y_act = qconv_module(X_q)

        # Make sure the results match
        # assert_array_almost_equal compares using the following formula:
        #     abs(desired-actual) < 1.5 * 10**(-decimal)
        # (https://docs.scipy.org/doc/numpy/reference/generated/numpy.testing.assert_almost_equal.html)
        # We use decimal = 0 to ignore off-by-1 differences between reference
        # and test. Off-by-1 differences arise due to the order of round and
        # zero_point addition operation, i.e., if addition followed by round is
        # used by reference and round followed by addition is used by test, the
        # results may differ by 1.
        # For example, the result of round(2.5) + 1 is 3 while round(2.5 + 1) is
        # 4 assuming the rounding mode is round-to-nearest, ties-to-even.
        # skip numerics checking for reference module
        if not is_reference:
            np.testing.assert_array_almost_equal(
                Y_exp.int_repr().numpy(), Y_act.int_repr().numpy(), decimal=0)

        # Test serialization of quantized Conv Module using state_dict
        model_dict = qconv_module.state_dict()
        self.assertEqual(model_dict['weight'], W_q)
        if use_bias:
            self.assertEqual(model_dict['bias'], b)
        bytes_io = io.BytesIO()
        torch.save(model_dict, bytes_io)
        bytes_io.seek(0)
        loaded_dict = torch.load(bytes_io)
        for key in loaded_dict:
            self.assertEqual(model_dict[key], loaded_dict[key])
        loaded_qconv_module = type(qconv_module)(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, use_bias, padding_mode=padding_mode)
        loaded_qconv_module.load_state_dict(loaded_dict)

        self.assertTrue(dir(loaded_qconv_module) == dir(qconv_module))
        self.assertTrue(module_name == loaded_qconv_module._get_name())
        if not is_reference:
            self.assertTrue(hasattr(loaded_qconv_module, '_packed_params'))
        self.assertTrue(hasattr(loaded_qconv_module, '_weight_bias'))

        self.assertEqual(qconv_module.weight(), loaded_qconv_module.weight())
        if use_bias:
            self.assertEqual(qconv_module.bias(), loaded_qconv_module.bias())
        self.assertEqual(qconv_module.scale, loaded_qconv_module.scale)
        self.assertEqual(qconv_module.zero_point,
                         loaded_qconv_module.zero_point)
        Y_loaded = loaded_qconv_module(X_q)
        if not is_reference:
            np.testing.assert_array_almost_equal(
                Y_exp.int_repr().numpy(), Y_loaded.int_repr().numpy(), decimal=0)

        # Test serialization
        b = io.BytesIO()
        torch.save(qconv_module, b)
        b.seek(0)
        loaded_conv = torch.load(b)

        self.assertEqual(loaded_conv.bias(), qconv_module.bias())
        self.assertEqual(loaded_conv.scale, qconv_module.scale)
        self.assertEqual(loaded_conv.zero_point,
                         qconv_module.zero_point)

        # Test copy and deepcopy
        copied_conv = copy.copy(qconv_module)
        self.assertEqual(copied_conv.bias(), qconv_module.bias())
        self.assertEqual(copied_conv.scale, qconv_module.scale)
        self.assertEqual(copied_conv.zero_point,
                         qconv_module.zero_point)
        Y_copied = copied_conv(X_q)
        if not is_reference:
            np.testing.assert_array_almost_equal(
                Y_exp.int_repr().numpy(), Y_copied.int_repr().numpy(), decimal=0)

        deepcopied_conv = copy.deepcopy(qconv_module)
        self.assertEqual(deepcopied_conv.bias(), qconv_module.bias())
        self.assertEqual(deepcopied_conv.scale, qconv_module.scale)
        self.assertEqual(deepcopied_conv.zero_point,
                         qconv_module.zero_point)
        Y_deepcopied = copied_conv(X_q)
        if not is_reference:
            np.testing.assert_array_almost_equal(
                Y_exp.int_repr().numpy(), Y_deepcopied.int_repr().numpy(), decimal=0)

        # JIT testing
        self.checkScriptable(
            qconv_module, [[X_q]],
            check_save_load=True)

        # Test from_float
        fused_conv_module = torch.nn.intrinsic._FusedModule(conv_module)
        fused_conv_module.qconfig = torch.quantization.default_qconfig
        torch.quantization.prepare(fused_conv_module, inplace=True)
        fused_conv_module(X.float())
        converted_qconv_module = fused_conv_module
        reference_mapping = get_default_static_quant_module_mappings()
        reference_mapping[type(conv_module)] = type(qconv_module)
        torch.quantization.convert(converted_qconv_module, mapping=reference_mapping, inplace=True)

        # Smoke test to make sure the module actually runs
        if use_bias:
            if use_fused:
                self.assertEqual(conv_module[0].bias,
                                 converted_qconv_module[0].bias())
            else:
                self.assertEqual(conv_module.bias,
                                 converted_qconv_module[0].bias())
        # Smoke test extra_repr
        self.assertTrue(module_name == converted_qconv_module[0]._get_name())

    @override_qengines
    def test_conv1d_api(self):
        options = itertools.product(
            ["zeros", "reflect"],  # pad_mode
            [True, False],  # use_bias
            [True, False],  # use_fused
            [True, False],  # use_channelwise
            [True, False]  # is_reference
        )
        for pad_mode, use_bias, use_fused, use_channelwise, is_reference in options:
            if torch.backends.quantized.engine == "qnnpack":
                use_channelwise = False
            batch_size = 2
            in_channels_per_group = 2
            length = 8
            out_channels_per_group = 2
            groups = 3
            kernel = 3
            stride = 2
            pad = 1
            dilation = 1
            # Tests the correctness of the conv2d module.
            in_channels = in_channels_per_group * groups
            out_channels = out_channels_per_group * groups
            input_feature_map_size = (length,)
            kernel_size = (kernel, )
            stride = (stride, )
            pad = (pad, )
            dilation = (dilation, )
            X_scale = 1.3
            X_zero_point = 2
            W_scale = [0.5]
            W_zero_point = [3]
            Y_scale = 5.0
            Y_zero_point = 4
            if torch.backends.quantized.engine == 'qnnpack':
                use_channelwise = False
            # (use_fused, is_reference) -> quantized class
            class_map = {
                (True, True): (nniqr.ConvReLU1d, "QuantizedConvReLU1d(Reference)"),
                (True, False): (nniq.ConvReLU1d, "QuantizedConvReLU1d"),
                (False, True): (nnqr.Conv1d, "QuantizedConv1d(Reference)"),
                (False, False): (nnq.Conv1d, "QuantizedConv1d")
            }

            qconv_cls, module_name = class_map[(use_fused, is_reference)]
            qconv_module = qconv_cls(
                in_channels, out_channels, kernel, stride, pad,
                dilation, groups, use_bias, padding_mode=pad_mode
            )

            conv_module = nn.Conv1d(
                in_channels, out_channels, kernel, stride, pad,
                dilation, groups, use_bias, padding_mode=pad_mode)
            if use_fused:
                relu_module = nn.ReLU()
                conv_module = nni.ConvReLU1d(conv_module, relu_module)
            conv_module = conv_module.float()

            self._test_conv_api_impl(
                module_name, qconv_module, conv_module, batch_size,
                in_channels_per_group, input_feature_map_size,
                out_channels_per_group, groups, kernel_size, stride, pad, pad_mode,
                dilation, X_scale, X_zero_point, W_scale, W_zero_point, Y_scale,
                Y_zero_point, use_bias, use_fused, use_channelwise, is_reference)

    @override_qengines
    def test_conv2d_api(self):
        options = itertools.product(
            ["zeros", "reflect"],  # pad_mode
            [True, False],  # use_bias
            [True, False],  # use_fused
            [True, False],  # use_channelwise
            [True, False]  # is_reference
        )
        for pad_mode, use_bias, use_fused, use_channelwise, is_reference in options:
            if torch.backends.quantized.engine == "qnnpack":
                use_channelwise = False
            batch_size = 2
            in_channels_per_group = 2
            H = 8
            W = 8
            out_channels_per_group = 2
            groups = 3
            kernel_h = 3
            kernel_w = 3
            stride_h = 2
            stride_w = 2
            pad_h = 1
            pad_w = 1
            dilation = 1
            # Tests the correctness of the conv2d module.
            in_channels = in_channels_per_group * groups
            out_channels = out_channels_per_group * groups
            input_feature_map_size = (H, W)
            kernel_size = (kernel_h, kernel_w)
            stride = (stride_h, stride_w)
            padding = (pad_h, pad_w)
            dilation = (dilation, dilation)
            X_scale = 1.3
            X_zero_point = 2
            W_scale = [0.5]
            W_zero_point = [3]
            Y_scale = 5.0
            Y_zero_point = 4
            # (use_fused, is_reference) -> quantized class
            class_map = {
                (True, True): (nniqr.ConvReLU2d, "QuantizedConvReLU2d(Reference)"),
                (True, False): (nniq.ConvReLU2d, "QuantizedConvReLU2d"),
                (False, True): (nnqr.Conv2d, "QuantizedConv2d(Reference)"),
                (False, False): (nnq.Conv2d, "QuantizedConv2d")
            }

            qconv_cls, module_name = class_map[(use_fused, is_reference)]
            qconv_module = qconv_cls(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation, groups, use_bias, padding_mode=pad_mode
            )

            conv_module = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation, groups, use_bias, padding_mode=pad_mode)
            if use_fused:
                relu_module = nn.ReLU()
                conv_module = nni.ConvReLU2d(conv_module, relu_module)
            conv_module = conv_module.float()

            self._test_conv_api_impl(
                module_name, qconv_module, conv_module, batch_size,
                in_channels_per_group, input_feature_map_size,
                out_channels_per_group, groups, kernel_size, stride, padding,
                pad_mode, dilation, X_scale, X_zero_point, W_scale, W_zero_point,
                Y_scale, Y_zero_point, use_bias, use_fused, use_channelwise, is_reference)

    @skipIfNoFBGEMM
    def test_conv3d_api(self):
        options = itertools.product(
            [True, False],  # use_bias
            [True, False],  # use_fused
            [True, False],  # use_channelwise
            [True, False]  # is_reference
        )
        for use_bias, use_fused, use_channelwise, is_reference in options:
            if torch.backends.quantized.engine == "qnnpack":
                use_channelwise = False
            batch_size = 2
            in_channels_per_group = 2
            H = 8
            W = 8
            D = 8
            out_channels_per_group = 2
            groups = 3
            kernel_h = 3
            kernel_w = 3
            kernel_d = 3
            stride_h = 2
            stride_w = 2
            stride_d = 2
            pad_mode = "zeros"  # 3d doesn't support reflect padding
            pad_h = 1
            pad_w = 1
            pad_d = 1
            dilation = 1
            # Tests the correctness of the conv3d module.
            in_channels = in_channels_per_group * groups
            out_channels = out_channels_per_group * groups
            input_feature_map_size = (D, H, W)
            kernel_size = (kernel_d, kernel_h, kernel_w)
            stride = (stride_d, stride_h, stride_w)
            padding = (pad_d, pad_h, pad_w)
            dilation = (dilation, dilation, dilation)
            X_scale = 1.3
            X_zero_point = 2
            W_scale = [0.5]
            W_zero_point = [3]
            Y_scale = 5.0
            Y_zero_point = 4
            # (use_fused, is_reference) -> quantized class
            class_map = {
                (True, True): (nniqr.ConvReLU3d, "QuantizedConvReLU3d(Reference)"),
                (True, False): (nniq.ConvReLU3d, "QuantizedConvReLU3d"),
                (False, True): (nnqr.Conv3d, "QuantizedConv3d(Reference)"),
                (False, False): (nnq.Conv3d, "QuantizedConv3d")
            }

            with override_quantized_engine('fbgemm'):
                qconv_cls, module_name = class_map[(use_fused, is_reference)]
                qconv_module = qconv_cls(
                    in_channels, out_channels, kernel_size, stride, padding,
                    dilation, groups, use_bias, padding_mode=pad_mode
                )

                conv_module = nn.Conv3d(
                    in_channels, out_channels, kernel_size, stride, padding,
                    dilation, groups, use_bias, padding_mode=pad_mode)
                if use_fused:
                    relu_module = nn.ReLU()
                    conv_module = nni.ConvReLU3d(conv_module, relu_module)
                conv_module = conv_module.float()

                self._test_conv_api_impl(
                    module_name, qconv_module, conv_module, batch_size,
                    in_channels_per_group, input_feature_map_size,
                    out_channels_per_group, groups, kernel_size, stride, padding,
                    pad_mode, dilation, X_scale, X_zero_point, W_scale,
                    W_zero_point, Y_scale, Y_zero_point, use_bias, use_fused,
                    use_channelwise, is_reference)

    def test_pool_api(self):
        """Tests the correctness of the pool module.
        The correctness is defined against the functional implementation.
        """
        N, C, H, W = 10, 10, 10, 3
        kwargs = {
            'kernel_size': 2,
            'stride': None,
            'padding': 0,
            'dilation': 1
        }

        scale, zero_point = 1.0 / 255, 128

        X = torch.randn(N, C, H, W, dtype=torch.float32)
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch.quint8)
        qX_expect = torch.nn.functional.max_pool2d(qX, **kwargs)

        pool_under_test = torch.nn.quantized.MaxPool2d(**kwargs)
        qX_hat = pool_under_test(qX)
        self.assertEqual(qX_expect, qX_hat)

        # JIT Testing
        self.checkScriptable(pool_under_test, [[X]])

    def test_batch_norm2d(self):
        """Tests the correctness of the batchnorm2d module.
        The correctness is defined against the functional implementation.
        """
        x = torch.randn((2, 4, 6, 8), dtype=torch.float)
        float_mod = torch.nn.BatchNorm2d(4)
        float_mod.training = False

        y_ref = float_mod(x)
        quant_ref = torch.quantize_per_tensor(y_ref, 1.0, 0, dtype=torch.quint8)

        quant_mod = nnq.BatchNorm2d(4)
        qx = torch.quantize_per_tensor(x, 1.0, 0, dtype=torch.quint8)
        qy = quant_mod(qx)

        self.assertEqual(quant_ref.int_repr().numpy(), qy.int_repr().numpy(),
                         msg="BatchNorm2d module API failed")

    def test_batch_norm3d(self):
        """Tests the correctness of the batchnorm3d module.
        The correctness is defined against the functional implementation.
        """
        x = torch.randn((2, 4, 6, 8, 10), dtype=torch.float)
        float_mod = torch.nn.BatchNorm3d(4)
        float_mod.training = False

        y_ref = float_mod(x)
        quant_ref = torch.quantize_per_tensor(y_ref, 1.0, 0, dtype=torch.quint8)

        quant_mod = nnq.BatchNorm3d(4)
        qx = torch.quantize_per_tensor(x, 1.0, 0, dtype=torch.quint8)
        qy = quant_mod(qx)

        self.assertEqual(quant_ref.int_repr().numpy(), qy.int_repr().numpy(),
                         msg="BatchNorm3d module API failed")

    def test_layer_norm(self):
        """Tests the correctness of the layernorm module.
        The correctness is defined against the functional implementation.
        """
        x_scale = 10.0 / 256
        x_zero_point = 0
        y_scale = 5.0 / 256
        y_zero_point = 127

        dims = (1, 4, 8)

        X = (torch.randn(dims, dtype=torch.float) - 0.5) * 10
        qX = torch.quantize_per_tensor(X, x_scale, x_zero_point, dtype=torch.quint8)
        dqX = qX.dequantize()

        float_mod = torch.nn.LayerNorm(dqX.size()[1:]).float()
        float_mod.weight = torch.nn.Parameter(torch.rand(*dims[1:]))
        float_mod.bias = torch.nn.Parameter(torch.rand(*dims[1:]))

        dqY_ref = float_mod(dqX)
        qY_ref = torch.quantize_per_tensor(
            dqY_ref, y_scale, y_zero_point, dtype=torch.quint8)

        quant_mod = nnq.LayerNorm(
            qX.size()[1:], float_mod.weight, float_mod.bias, y_scale, y_zero_point)
        qY = quant_mod(qX)

        self.assertEqual(qY_ref.int_repr().numpy(), qY.int_repr().numpy(),
                         msg="LayerNorm module API failed, qY_ref\n{} vs qY\n{}"
                         .format(qY_ref, qY))

    def test_group_norm(self):
        """Tests the correctness of the groupnorm module.
        The correctness is defined against the functional implementation.
        """
        x_scale = 10.0 / 256
        x_zero_point = 0
        y_scale = 5.0 / 256
        y_zero_point = 127

        dims = (1, 4, 8)

        X = (torch.randn(dims, dtype=torch.float) - 0.5) * 10
        qX = torch.quantize_per_tensor(X, x_scale, x_zero_point, dtype=torch.quint8)
        dqX = qX.dequantize()

        float_mod = torch.nn.GroupNorm(2, 4).float()
        float_mod.weight = torch.nn.Parameter(torch.rand(dims[1]))
        float_mod.bias = torch.nn.Parameter(torch.rand(dims[1]))

        dqY_ref = float_mod(dqX)
        qY_ref = torch.quantize_per_tensor(
            dqY_ref, y_scale, y_zero_point, dtype=torch.quint8)

        quant_mod = nnq.GroupNorm(
            2, 2, float_mod.weight, float_mod.bias, y_scale, y_zero_point)
        qY = quant_mod(qX)

        self.assertEqual(qY_ref.int_repr().numpy(), qY.int_repr().numpy(),
                         msg="GroupNorm module API failed, qY_ref\n{} vs qY\n{}"
                         .format(qY_ref, qY))

    def test_instance_norm(self):
        """Tests the correctness of the instancenorm{n}d modules.
        The correctness is defined against the functional implementation.
        """
        x_scale = 10.0 / 256
        x_zero_point = 0
        y_scale = 5.0 / 256
        y_zero_point = 127

        dims_to_modules = [
            ((1, 4, 8), torch.nn.InstanceNorm1d, nnq.InstanceNorm1d),
            ((1, 4, 8, 1), torch.nn.InstanceNorm2d, nnq.InstanceNorm2d),
            ((1, 4, 8, 1, 1), torch.nn.InstanceNorm3d, nnq.InstanceNorm3d),
        ]

        for dim_to_modules in dims_to_modules:
            dims, float_cls, q_cls = dim_to_modules

            X = (torch.randn(dims, dtype=torch.float) - 0.5) * 10
            qX = torch.quantize_per_tensor(
                X, x_scale, x_zero_point, dtype=torch.quint8)
            dqX = qX.dequantize()

            float_mod = float_cls(dims[1]).float()
            float_mod.weight = torch.nn.Parameter(torch.rand(dims[1]))
            float_mod.bias = torch.nn.Parameter(torch.rand(dims[1]))

            dqY_ref = float_mod(dqX)
            qY_ref = torch.quantize_per_tensor(
                dqY_ref, y_scale, y_zero_point, dtype=torch.quint8)

            quant_mod = q_cls(
                dims[1], float_mod.weight, float_mod.bias, y_scale,
                y_zero_point)
            qY = quant_mod(qX)

            self.assertEqual(
                qY_ref.int_repr().numpy(), qY.int_repr().numpy(),
                msg="InstanceNorm module API failed, qY_ref\n{} vs qY\n{}"
                .format(qY_ref, qY))

    def _test_activation_module_impl(self, name, float_module_class, quantized_module_class, extra_kwargs):
        """Tests the correctness of the ELU module.
        The correctness is defined against the functional implementation.
        """
        x_scale = 10.0 / 256
        x_zero_point = 0
        y_scale = 5.0 / 256
        y_zero_point = 127
        alpha = 1.5

        dims = (1, 4, 8)

        X = (torch.randn(dims, dtype=torch.float) - 0.5) * 10
        qX = torch.quantize_per_tensor(X, x_scale, x_zero_point, dtype=torch.quint8)
        dqX = qX.dequantize()

        float_mod = float_module_class(**extra_kwargs).float()

        dqY_ref = float_mod(dqX)
        qY_ref = torch.quantize_per_tensor(
            dqY_ref, y_scale, y_zero_point, dtype=torch.quint8)

        quant_mod = quantized_module_class(y_scale, y_zero_point, **extra_kwargs)
        qY = quant_mod(qX)
        self.assertEqual(qY_ref.int_repr().numpy(), qY.int_repr().numpy(),
                         msg="{} module API failed, qY_ref\n{} vs qY\n{}"
                         .format(name, qY_ref, qY))

    def _test_leaky_relu_serialization(self):
        scale_original = 10.0 / 256
        zero_point_original = 1.0

        quant_mod_original = nnq.LeakyReLU(scale_original, zero_point_original)
        state_dict = quant_mod_original.state_dict()

        scale_new = 5.0 / 256
        zero_point_new = 2.0
        quant_mod_new = nnq.LeakyReLU(scale_new, zero_point_new)
        quant_mod_new.load_state_dict(state_dict)

        self.assertEqual(quant_mod_original.scale, quant_mod_new.scale)
        self.assertEqual(quant_mod_original.zero_point, quant_mod_new.zero_point)

    def test_elu(self):
        """Tests the correctness of the ELU module.
        The correctness is defined against the functional implementation.
        """
        self._test_activation_module_impl("ELU", nn.ELU, nnq.ELU, {"alpha": 1.5})

    def test_leaky_relu(self):
        self._test_activation_module_impl("LeakyReLU", nn.LeakyReLU, nnq.LeakyReLU, {"negative_slope": 0.2})
        self._test_leaky_relu_serialization()

    def test_sigmoid(self):
        self._test_activation_module_impl("Sigmoid", nn.Sigmoid, nnq.Sigmoid, {})

    @given(
        num_embeddings=st.integers(10, 50),
        embedding_dim=st.integers(5, 50).filter(lambda x: x % 4 == 0),
        set_qconfig=st.booleans(),
    )
    @skipIfNoFBGEMM
    def test_embedding_api(self, num_embeddings, embedding_dim, set_qconfig):
        num_lengths = np.random.randint(1, 6)
        lengths = np.random.randint(0, 21, size=num_lengths).astype(np.int32)
        num_indices = np.sum(lengths)
        indices = torch.from_numpy(np.random.randint(low=0, high=num_embeddings, size=num_indices, dtype=np.int64))
        weights = torch.from_numpy((np.random.random_sample((num_embeddings, embedding_dim)) + 1).astype(np.float32))

        obs = default_float_qparams_observer()
        obs(weights)
        qparams = obs.calculate_qparams()
        # Quantize the weights to 8bits
        qweight = torch.quantize_per_channel(weights, qparams[0], qparams[1], axis=0, dtype=torch.quint8)
        qemb = nnq.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        qemb.set_weight(qweight)
        qemb(indices)

        # Ensure the module has the correct weights
        self.assertEqual(qweight, qemb.weight())

        w_packed = qemb._packed_params._packed_weight
        module_out = qemb(indices)

        # Call the qembedding operator directly
        ref = torch.ops.quantized.embedding_byte(w_packed, indices, pruned_weights=False)
        self.assertEqual(module_out, ref)
        self.checkEmbeddingSerialization(qemb, num_embeddings, embedding_dim, indices, None, set_qconfig=False, is_emb_bag=False)


    @given(
        num_embeddings=st.integers(10, 50),
        embedding_dim=st.integers(5, 50).filter(lambda x: x % 4 == 0),
        num_offsets=st.integers(1, 20),
        set_qconfig=st.booleans(),
    )
    @skipIfNoFBGEMM
    def test_embedding_bag_api(self, num_embeddings, embedding_dim, num_offsets, set_qconfig):
        r"""Test execution and serialization for dynamic quantized embedding_bag modules on int8
        """

        num_lengths = np.random.randint(1, 6)
        lengths = np.random.randint(0, 21, size=num_lengths).astype(np.int32)
        num_indices = np.sum(lengths)
        indices = torch.from_numpy(np.random.randint(low=0, high=num_embeddings, size=num_indices, dtype=np.int64))

        offsets = lengths_to_offsets(lengths)
        # include the last offset
        offsets = torch.cat((offsets, torch.tensor([indices.size(0)], dtype=torch.long)), 0)
        weights = torch.from_numpy((np.random.random_sample((num_embeddings, embedding_dim)) + 1).astype(np.float32))

        for qdtype in [torch.quint8, torch.quint4x2]:
            obs = PerChannelMinMaxObserver(dtype=qdtype, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0)
            obs(weights)
            # Get the scale and zero point for the weight tensor
            qparams = obs.calculate_qparams()
            # Quantize the weights to 8bits
            qweight = torch.quantize_per_channel(weights, qparams[0], qparams[1], axis=0, dtype=qdtype)
            qemb = nnq.EmbeddingBag(num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                                    include_last_offset=True, mode='sum', _weight=qweight, dtype=qdtype)
            qemb(indices, offsets)

            # Ensure the module has the correct weights
            self.assertEqual(qweight, qemb.weight())

            w_packed = qemb._packed_params._packed_weight
            module_out = qemb(indices, offsets)

            # Call the qembedding_bag operator directly
            if qdtype == torch.quint8:
                ref = torch.ops.quantized.embedding_bag_byte(w_packed, indices, offsets, mode=0,
                                                             per_sample_weights=None,
                                                             include_last_offset=True)
            else:
                ref = torch.ops.quantized.embedding_bag_4bit(w_packed, indices, offsets, mode=0,
                                                             per_sample_weights=None,
                                                             include_last_offset=True)

            self.assertEqual(module_out, ref)
            self.checkEmbeddingSerialization(qemb, num_embeddings, embedding_dim, indices,
                                             offsets, set_qconfig, is_emb_bag=True, dtype=qdtype)

class TestDynamicQuantizedModule(QuantizationTestCase):
    """
    Note that tests in this class are just API test, to make sure we wrapped the
    quantized operator implementations correctly in the user facing APIs, these are
    not correctness test for the underlying quantized operators. For correctness
    test please see `test/quantization/test_quantized_op.py`.
    """
    @given(
        batch_size=st.integers(1, 5),
        in_features=st.integers(16, 32),
        out_features=st.integers(4, 8),
        use_bias=st.booleans(),
        use_default_observer=st.booleans(),
    )
    @override_qengines
    def test_linear_api(self, batch_size, in_features, out_features, use_bias, use_default_observer):
        """test API functionality for nn.quantized.dynamic.Linear"""
        W = torch.rand(out_features, in_features).float()
        W_scale, W_zp = _calculate_dynamic_qparams(W, torch.qint8)
        W_q = torch.quantize_per_tensor(W, W_scale, W_zp, torch.qint8)
        X = torch.rand(batch_size, in_features).float()
        B = torch.rand(out_features).float() if use_bias else None
        qlinear = nnqd.Linear(in_features, out_features)
        # Run module with default-initialized parameters.
        # This tests that the constructor is correct.
        qlinear.set_weight_bias(W_q, B)
        qlinear(X)

        # Simple round-trip test to ensure weight()/set_weight() API
        self.assertEqual(qlinear.weight(), W_q)
        W_pack = qlinear._packed_params._packed_params
        Z_dq = qlinear(X)

        # Check if the module implementation matches calling the
        # ops directly
        Z_ref = torch.ops.quantized.linear_dynamic(X, W_pack, reduce_range=True)
        self.assertEqual(Z_ref, Z_dq)

        # Test serialization of dynamic quantized Linear Module using state_dict
        model_dict = qlinear.state_dict()
        b = io.BytesIO()
        torch.save(model_dict, b)
        b.seek(0)
        loaded_dict = torch.load(b)
        for key in model_dict:
            if isinstance(model_dict[key], torch._C.ScriptObject):
                assert isinstance(loaded_dict[key], torch._C.ScriptObject)
                w_model, b_model = torch.ops.quantized.linear_unpack(model_dict[key])
                w_loaded, b_loaded = torch.ops.quantized.linear_unpack(loaded_dict[key])
                self.assertEqual(w_model, w_loaded)
                self.assertEqual(b_model, b_loaded)
            else:
                self.assertEqual(model_dict[key], loaded_dict[key])
        loaded_qlinear = nnqd.Linear(in_features, out_features)
        loaded_qlinear.load_state_dict(loaded_dict)

        linear_unpack = torch.ops.quantized.linear_unpack
        self.assertEqual(linear_unpack(qlinear._packed_params._packed_params),
                         linear_unpack(loaded_qlinear._packed_params._packed_params))
        if use_bias:
            self.assertEqual(qlinear.bias(), loaded_qlinear.bias())
        self.assertTrue(dir(qlinear) == dir(loaded_qlinear))
        self.assertTrue(hasattr(qlinear, '_packed_params'))
        self.assertTrue(hasattr(loaded_qlinear, '_packed_params'))
        self.assertTrue(hasattr(qlinear, '_weight_bias'))
        self.assertTrue(hasattr(loaded_qlinear, '_weight_bias'))

        self.assertEqual(qlinear._weight_bias(), loaded_qlinear._weight_bias())
        self.assertEqual(qlinear._weight_bias(), torch.ops.quantized.linear_unpack(qlinear._packed_params._packed_params))
        Z_dq2 = qlinear(X)
        self.assertEqual(Z_dq, Z_dq2)

        b = io.BytesIO()
        torch.save(qlinear, b)
        b.seek(0)
        loaded = torch.load(b)
        self.assertEqual(qlinear.weight(), loaded.weight())
        self.assertEqual(qlinear.zero_point, loaded.zero_point)

        # Test JIT
        self.checkScriptable(qlinear, [[X]], check_save_load=True)

        modules_under_test = [torch.nn.Linear, torch.nn.modules.linear._LinearWithBias]
        for mut in modules_under_test:
            # Test from_float
            float_linear = mut(in_features, out_features).float()
            if use_default_observer:
                float_linear.qconfig = torch.quantization.default_dynamic_qconfig
            prepare_dynamic(float_linear)
            float_linear(X.float())
            quantized_float_linear = nnqd.Linear.from_float(float_linear)

            # Smoke test to make sure the module actually runs
            quantized_float_linear(X)

        # Smoke test extra_repr
        self.assertTrue('QuantizedLinear' in str(quantized_float_linear))

    @given(
        dtype=st.sampled_from([torch.qint8, torch.float16]),
        bidirectional=st.booleans(),
    )
    @override_qengines
    def test_lstm_api(self, dtype, bidirectional):
        r"""Test execution and serialization for dynamic quantized lstm modules on int8 and fp16
        """
        # Check that module matches the numerics of the op and ensure that module can be
        # instantiated for all engines and dtypes
        seq_len = 4
        batch = 2
        input_size = 3
        hidden_size = 7
        num_layers = 2
        bias = True
        weight_keys = []
        bias_keys = []
        num_directions = 2 if bidirectional else 1
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                key_name1 = 'weight_ih_l{layer_idx}{suffix}'.format(layer_idx=layer, suffix=suffix)
                key_name2 = 'weight_hh_l{layer_idx}{suffix}'.format(layer_idx=layer, suffix=suffix)
                weight_keys.append(key_name1)
                weight_keys.append(key_name2)
                key_name1 = 'bias_ih_l{layer_idx}{suffix}'.format(layer_idx=layer, suffix=suffix)
                key_name2 = 'bias_hh_l{layer_idx}{suffix}'.format(layer_idx=layer, suffix=suffix)
                bias_keys.append(key_name1)
                bias_keys.append(key_name2)

        if not (dtype == torch.float16 and torch.backends.quantized.engine == "qnnpack"):
            # fp16 dynamic quant is not supported for qnnpack
            x = torch.randn(seq_len, batch, input_size)
            h = torch.randn(num_layers * (bidirectional + 1), batch, hidden_size)
            c = torch.randn(num_layers * (bidirectional + 1), batch, hidden_size)
            cell_dq = torch.nn.quantized.dynamic.LSTM(input_size=input_size,
                                                      hidden_size=hidden_size,
                                                      num_layers=num_layers,
                                                      bias=bias,
                                                      batch_first=False,
                                                      dropout=0.0,
                                                      bidirectional=bidirectional,
                                                      dtype=dtype)
            ref_dq = torch.nn.quantized.dynamic.LSTM(input_size=input_size,
                                                     hidden_size=hidden_size,
                                                     num_layers=num_layers,
                                                     bias=bias,
                                                     batch_first=False,
                                                     dropout=0.0,
                                                     bidirectional=bidirectional,
                                                     dtype=dtype)

            _all_params = ([m.param for m in cell_dq._all_weight_values])
            result = torch.quantized_lstm(x, (h, c),
                                          _all_params,
                                          cell_dq.bias,
                                          cell_dq.num_layers,
                                          float(cell_dq.dropout),
                                          False,
                                          bidirectional,
                                          False,
                                          dtype=dtype,
                                          use_dynamic=True)


            y, (h, c) = cell_dq(x, (h, c))
            self.assertEqual(result[0], y)
            self.assertEqual(result[1], h)
            self.assertEqual(result[2], c)
            x = torch.randn(10, 20, 3)
            self.check_eager_serialization(cell_dq, ref_dq, [x])
            self.check_weight_bias_api(cell_dq, weight_keys, bias_keys)

    @override_qengines
    def test_gru_api(self):
        r"""Test execution and serialization for dynamic quantized lstm modules on int8 and fp16
        """
        # Check that module matches the numerics of the op and ensure that module can be
        # instantiated for all engines and dtypes

        for dtype in [torch.qint8, torch.float16]:
            if dtype == torch.float16 and torch.backends.quantized.engine == "qnnpack":
                # fp16 dynamic quant is not supported for qnnpack
                continue
                # Test default instantiation
            seq_len = 4
            batch = 2
            input_size = 3
            hidden_size = 7
            num_layers = 2
            bias = True
            bidirectional = False

            x = torch.rand(seq_len, batch, input_size)
            h = torch.rand(num_layers * (bidirectional + 1), batch, hidden_size)


            cell_dq = torch.nn.quantized.dynamic.GRU(input_size=input_size,
                                                     hidden_size=hidden_size,
                                                     num_layers=num_layers,
                                                     bias=bias,
                                                     batch_first=False,
                                                     dropout=0.0,
                                                     bidirectional=bidirectional,
                                                     dtype=dtype)

            _all_params = ([m.param for m in cell_dq._all_weight_values])
            result = torch.quantized_gru(x,
                                         h,
                                         _all_params,
                                         cell_dq.bias,
                                         cell_dq.num_layers,
                                         float(cell_dq.dropout),
                                         False,
                                         bidirectional,
                                         False)


            y, h = cell_dq(x, h)
            self.assertEqual(result[0], y, msg="GRU module API failed")
            self.assertEqual(result[1], h, msg="GRU module API failed")

    @given(
        dtype=st.sampled_from([torch.qint8, torch.float16]),
    )
    @override_qengines
    def test_cell_api(self, dtype):
        r"""Test execution and serialization for dynamic quantized lstm modules on int8 and fp16
        """
        # Check that module matches the numerics of the op and ensure that module can be
        # instantiated for all engines and dtypes
        batch = 7
        input_size = 3
        hidden_size = 7
        bias = True

        x = torch.rand(batch, input_size)
        h = torch.rand(batch, hidden_size)
        cell_dict = {'LSTMCell': torch.nn.quantized.dynamic.LSTMCell,
                     'GRUCell': torch.nn.quantized.dynamic.GRUCell,
                     'RNNTanh': torch.nn.quantized.dynamic.RNNCell,
                     'RNNReLU': torch.nn.quantized.dynamic.RNNCell
                     }
        state = {'LSTMCell': (h, h),
                 'GRUCell': h,
                 'RNNTanh': h,
                 'RNNReLU': h}

        qfn_dict = {'LSTMCell': torch.ops.quantized.quantized_lstm_cell_dynamic,
                    'GRUCell': torch.ops.quantized.quantized_gru_cell_dynamic,
                    'RNNTanh': torch.ops.quantized.quantized_rnn_tanh_cell_dynamic,
                    'RNNReLU': torch.ops.quantized.quantized_rnn_relu_cell_dynamic}

        for rnn_type in cell_dict.keys():
            if not (dtype == torch.float16 and torch.backends.quantized.engine == "qnnpack"):
                # fp16 dynamic quant is not supported for qnnpack
                kwargs = {'input_size': input_size, 'hidden_size': hidden_size, 'bias': bias, 'dtype': dtype}
                if rnn_type == 'RNNReLU':
                    kwargs['nonlinearity'] = "relu"
                elif rnn_type == 'RNNTanh':
                    kwargs['nonlinearity'] = "tanh"

                cell_dq = cell_dict[rnn_type](**kwargs)
                result = qfn_dict[rnn_type](x, state[rnn_type],
                                            cell_dq._packed_weight_ih, cell_dq._packed_weight_hh,
                                            cell_dq.bias_ih, cell_dq.bias_hh)
                result_module = cell_dq(x, state[rnn_type])
                self.assertEqual(result[0], result_module[0], msg="RNNCell module API failed")
                self.assertEqual(result[1], result_module[1], msg="RNNCell module API failed")
                weight_keys = ['weight_ih', 'weight_hh']
                bias_keys = ['bias_ih', 'bias_hh']
                self.check_eager_serialization(cell_dq, cell_dict[rnn_type](**kwargs), [x])
                self.check_weight_bias_api(cell_dq, weight_keys, bias_keys)

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

class TestQuantizeONNXExport(JitTestCase):
    def _test_lower_graph_impl(self, model, data):
        model.qconfig = torch.quantization.default_qconfig
        model = torch.quantization.prepare(model)
        model = torch.quantization.convert(model)

        outputs = model(data)
        input_names = ["x"]

        def export_to_onnx(model, input, input_names):
            outputs = model(input)

            traced = torch.jit.trace(model, input)
            buf = io.BytesIO()
            torch.jit.save(traced, buf)
            buf.seek(0)

            model = torch.jit.load(buf)
            f = io.BytesIO()
            torch.onnx.export(model, input, f, input_names=input_names, example_outputs=outputs,
                              operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
        onnx_model = export_to_onnx(model, data, input_names)

    @skipIfNoFBGEMM
    def test_lower_graph_linear(self):
        model = torch.quantization.QuantWrapper(torch.nn.Linear(5, 10, bias=True)).to(dtype=torch.float)
        data_numpy = np.random.rand(1, 2, 5).astype(np.float32)
        data = torch.from_numpy(data_numpy).to(dtype=torch.float)
        self._test_lower_graph_impl(model, data)

    @skipIfNoFBGEMM
    def test_lower_graph_conv2d(self):
        model = torch.quantization.QuantWrapper(torch.nn.Conv2d(3, 5, 2, bias=True)).to(dtype=torch.float)
        data_numpy = np.random.rand(1, 3, 6, 6).astype(np.float32)
        data = torch.from_numpy(data_numpy).to(dtype=torch.float)
        self._test_lower_graph_impl(model, data)

    @skipIfNoFBGEMM
    @unittest.skip("onnx opset9 does not support quantize_per_tensor and caffe2 \
    does not support conv3d")
    def test_lower_graph_conv3d(self):
        model = torch.quantization.QuantWrapper(torch.nn.Conv3d(3, 5, 2, bias=True)).to(dtype=torch.float)
        data_numpy = np.random.rand(1, 3, 6, 6, 6).astype(np.float32)
        data = torch.from_numpy(data_numpy).to(dtype=torch.float)
        self._test_lower_graph_impl(model, data)


if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_quantization.py TESTNAME\n\n"
                       "instead.")
