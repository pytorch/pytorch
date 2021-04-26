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
    QuantStub,
    DeQuantStub,
    QConfig,
    default_qconfig,
    default_qat_qconfig,
    default_dynamic_qconfig,
    per_channel_dynamic_qconfig,
    float16_dynamic_qconfig,
    float_qparams_weight_only_qconfig,
    PerChannelMinMaxObserver,
    QConfigDynamic,
    default_dynamic_quant_observer,
    FixedQParamsFakeQuantize,
)

from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    AnnotatedSingleLayerLinearModel,
    QuantStubModel,
    ModelForFusion,
    ModelWithSequentialFusion,
    ModelForLinearBNFusion,
    ManualLinearQATModel,
    ManualConvLinearQATModel,
    ModelWithFunctionals,
    ModelMultipleOps,
    ModelMultipleOpsNoAvgPool,
    SingleLayerLinearDynamicModel,
    TwoLayerLinearModel,
    NestedModel,
    ResNetBase,
    RNNDynamicModel,
    RNNCellDynamicModel,
    ModelForFusionWithBias,
    ActivationsTestModel,
    NormalizationTestModel,
    test_only_eval_fn,
    test_only_train_fn,
    prepare_dynamic,
    convert_dynamic,
    skipIfNoFBGEMM,
    EmbeddingBagModule,
    EmbeddingModule,
    EmbeddingWithLinear,
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
from torch.testing._internal.common_utils import TemporaryFileName
from torch.testing._internal.common_utils import suppress_warnings
from torch.testing._internal.jit_utils import JitTestCase
from hypothesis import given
from hypothesis import strategies as st
import torch.testing._internal.hypothesis_utils as hu
hu.assert_deadline_disabled()

# Standard library
from typing import Tuple
import copy
import io
import unittest
import numpy as np

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
                    self.checkNoQconfig(model)

                checkQuantized(model)

                model = quantize_qat(ManualLinearQATModel(qengine), test_only_train_fn,
                                     [self.train_data])
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

                test_only_train_fn(model, self.img_data_2d_train)
                model = convert(model)

                def checkQuantized(model):
                    self.assertEqual(type(model.conv), nnq.Conv2d)
                    self.assertEqual(type(model.fc1), nnq.Linear)
                    self.assertEqual(type(model.fc2), nnq.Linear)
                    test_only_eval_fn(model, self.img_data_2d)
                    self.checkScriptable(model, self.img_data_2d)
                    self.checkNoQconfig(model)

                checkQuantized(model)

                model = ManualConvLinearQATModel()
                model = quantize_qat(model, test_only_train_fn, [self.img_data_2d_train])
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

    @override_qengines
    def test_forward_hooks_preserved(self):
        r"""Test QAT on preserving pre forward and post forward hooks of original model
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

        model.qconfig = torch.quantization.get_default_qat_qconfig(qengine)
        model = prepare_qat(model)

        def checkHooksIsPresent(model, before_convert=True):
            forward_hooks = 1
            if before_convert:
                self.assertEqual(len(model.quant._forward_hooks.values()), 1,
                                 "Quantization observer hook has disappeared")
                forward_hooks = 2
            self.assertObjectIn(fw_pre_hook, model.fc._forward_pre_hooks.values())
            self.assertObjectIn(fw_hook, model.fc._forward_hooks.values())
            self.assertEqual(len(model.fc._forward_pre_hooks.values()), 1,
                             "Extra pre forward hooks have appeared on a layer")
            self.assertEqual(len(model.fc._forward_hooks.values()), forward_hooks,
                             "Extra post forward hooks have appeared on a layer")

        checkHooksIsPresent(model, True)
        x = torch.rand(2, 5, dtype=torch.float)
        model(x)
        torch.quantization.convert(model, inplace=True)
        checkHooksIsPresent(model, False)

    def test_add_scalar_uses_input_qparams(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = torch.quantization.QuantStub()
                self.ff = torch.nn.quantized.FloatFunctional()

            def forward(self, x):
                x = self.quant(x)
                x = self.ff.add_scalar(x, 1.0)
                return x

        m = M()
        m.qconfig = torch.quantization.default_qconfig
        mp = torch.quantization.prepare_qat(m)
        mp(torch.randn(4, 4))
        mq = torch.quantization.convert(mp)
        res = mq(torch.randn(4, 4))
        eps = 1e-5
        self.assertTrue(torch.abs(mq.quant.scale - res.q_scale()) < eps)

    def test_mul_scalar_uses_input_qparams(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = torch.quantization.QuantStub()
                self.ff = torch.nn.quantized.FloatFunctional()

            def forward(self, x):
                x = self.quant(x)
                x = self.ff.mul_scalar(x, 2.0)
                return x

        m = M()
        m.qconfig = torch.quantization.default_qconfig
        mp = torch.quantization.prepare_qat(m)
        mp(torch.randn(4, 4))
        mq = torch.quantization.convert(mp)
        res = mq(torch.randn(4, 4))
        eps = 1e-5
        self.assertTrue(torch.abs(mq.quant.scale * 2 - res.q_scale()) < eps)


class TestEagerModeOps(QuantizationTestCase):
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


class TestEagerModeQATOps(QuantizationTestCase):
    def _test_activation_convert_numerics_impl(self, Act, data):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.act = Act()
                self.quant = QuantStub()
                self.dequant = DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.act(x)
                x = self.dequant(x)
                return x

        m = M().train()
        m.qconfig = default_qat_qconfig
        m = prepare_qat(m)
        before_convert = m(data)
        m = convert(m)
        after_convert = m(data)
        self.assertEqual(before_convert, after_convert)

    def test_fixed_qparam_ops(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()
                self.hardsigmoid = torch.nn.Hardsigmoid()
                self.tanh = torch.nn.Tanh()
                self.quant = QuantStub()
                self.dequant = DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.sigmoid(x)
                x = self.hardsigmoid(x)
                x = self.tanh(x)
                x = self.dequant(x)
                return x

        m = M().train()
        m.qconfig = default_qat_qconfig
        m = prepare_qat(m)
        for attr in ['sigmoid', 'hardsigmoid', 'tanh']:
            self.assertEqual(type(getattr(m, attr).activation_post_process), FixedQParamsFakeQuantize)
        data = torch.randn(1, 3, 2, 4)
        before_convert = m(data)
        m = convert(m)
        after_convert = m(data)
        self.assertEqual(before_convert, after_convert)
        # make sure activation post process is removed
        for attr in ['sigmoid', 'hardsigmoid', 'tanh']:
            # verify fake quant module is removd
            self.assertFalse(hasattr(getattr(m, attr), 'activation_post_process'))
            # verify that hooks are removed
            self.assertTrue(len(getattr(m, attr)._forward_hooks.items()) == 0)

        # make sure no fake quantize module is inserted for eval mode

        def checkNoFQModule(m):
            for attr in ['sigmoid', 'hardsigmoid', 'tanh']:
                self.assertFalse(hasattr(getattr(m, attr), "activation_post_process"))
                self.assertTrue(len(getattr(m, attr)._forward_hooks.items()) == 0)

        m = M().eval()
        m.qconfig = default_qconfig
        m = prepare(m)
        checkNoFQModule(m)
        m = convert(m)
        checkNoFQModule(m)

    def test_leaky_relu(self):
        data = torch.randn(1, 3, 2, 4)
        self._test_activation_convert_numerics_impl(nn.LeakyReLU, data)

    def test_relu(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(x)
                return x

        m = M().train()
        m.qconfig = default_qconfig
        m = prepare_qat(m)
        # make sure no activation_post_process is inserted for relu
        self.assertFalse(hasattr(m, "activation_post_process"))
        m = convert(m)
        # make sure ReLU module is not changed
        self.assertTrue(type(m.relu), nn.ReLU)

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

@skipIfNoFBGEMM
class TestFusion(QuantizationTestCase):
    def test_fuse_module_train(self):
        model = ModelForFusion(default_qat_qconfig).train()
        # Test step by step fusion
        model = fuse_modules(model, ['conv1', 'bn1', 'relu1'])
        model = fuse_modules(model, ['sub1.conv', 'sub1.bn'])
        self.assertEqual(type(model.conv1), nni.ConvBnReLU2d,
                         msg="Fused Conv + BN + Relu first layer")
        self.assertEqual(type(model.bn1), torch.nn.Identity,
                         msg="Fused Conv + BN + Relu (skipped BN)")
        self.assertEqual(type(model.relu1), torch.nn.Identity,
                         msg="Fused Conv + BN + Relu (skipped Relu)")

        self.assertEqual(type(model.sub1.conv), nni.ConvBn2d,
                         msg="Fused submodule Conv + BN")
        self.assertEqual(type(model.sub1.bn), torch.nn.Identity,
                         msg="Fused submodule Conv + BN (skipped BN)")
        self.assertEqual(type(model.sub2.conv), torch.nn.Conv2d,
                         msg="Non-fused submodule Conv")
        self.assertEqual(type(model.sub2.relu), torch.nn.ReLU,
                         msg="Non-fused submodule ReLU")
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
        test_only_train_fn(model, self.img_data_1d_train)
        model = convert(model)

        def checkQuantized(model):
            self.assertEqual(type(model.conv1), nniq.ConvReLU2d)
            self.assertEqual(type(model.bn1), nn.Identity)
            self.assertEqual(type(model.relu1), nn.Identity)
            self.assertEqual(type(model.sub1.conv), nnq.Conv2d)
            self.assertEqual(type(model.sub1.bn), nn.Identity)
            self.assertEqual(type(model.sub2.conv), nn.Conv2d)
            self.assertEqual(type(model.sub2.relu), nn.ReLU)
            test_only_eval_fn(model, self.img_data_1d)
            self.checkNoQconfig(model)

        with self.assertRaisesRegex(RuntimeError, "Could not run 'aten::native_batch_norm' with arguments from the 'QuantizedCPU'"):
            checkQuantized(model)

        model = ModelForFusion(default_qat_qconfig).train()
        model = fuse_modules(model, [['conv1', 'bn1', 'relu1'],
                             ['sub1.conv', 'sub1.bn']])
        model = quantize_qat(model, test_only_train_fn, [self.img_data_1d_train])
        with self.assertRaisesRegex(RuntimeError, "Could not run 'aten::native_batch_norm' with arguments from the 'QuantizedCPU'"):
            checkQuantized(model)


    def test_fuse_module_eval(self):
        model = ModelForFusion(default_qconfig)
        model.eval()
        model = fuse_modules(model, [['conv3', 'bn3', 'relu4'],
                             ['conv1', 'bn1', 'relu1'],
                             ['conv2', 'relu2'],
                             ['bn2', 'relu3'],
                             ['sub1.conv', 'sub1.bn']])
        self.assertEqual(type(model.conv1), nni.ConvReLU2d,
                         msg="Fused Conv + BN + Relu first layer (BN is folded)")
        self.assertEqual(type(model.conv1[0]), nn.Conv2d,
                         msg="Fused Conv + BN + Relu (Conv + folded BN only)")
        self.assertEqual(type(model.conv1[1]), nn.ReLU,
                         msg="Fused Conv + BN + Relu second layer (Relu only)")
        self.assertEqual(type(model.bn1), nn.Identity,
                         msg="Fused Conv + BN + Relu second layer (Skipped BN)")
        self.assertEqual(type(model.relu1), nn.Identity,
                         msg="Fused Conv + BN + Relu second layer (Skipped Relu)")
        self.assertEqual(type(model.conv2), nni.ConvReLU3d,
                         msg="Fused Conv + BN + Relu first layer (BN is folded)")
        self.assertEqual(type(model.bn2), nni.BNReLU3d,
                         msg="Fused BN + Relu first layer (Relu is folded))")
        self.assertEqual(type(model.relu3), nn.Identity,
                         msg="Fused BN + Relu second layer (Skipped Relu)")
        self.assertEqual(type(model.conv2[0]), nn.Conv3d,
                         msg="Fused Conv + BN + Relu (Conv + folded BN only)")
        self.assertEqual(type(model.conv2[1]), nn.ReLU,
                         msg="Fused Conv + BN + Relu second layer (Relu only)")
        self.assertEqual(type(model.relu2), nn.Identity,
                         msg="Fused Conv + BN + Relu second layer (Skipped Relu)")

        self.assertEqual(type(model.conv3), nni.ConvReLU1d,
                         msg="Fused Conv + Relu for Conv1d (folded BN)")
        self.assertEqual(type(model.conv3[0]), nn.Conv1d,
                         msg="Fused Conv + Relu for Conv1d ")
        self.assertEqual(type(model.conv3[1]), nn.ReLU,
                         msg="Fused Conv + Relu for Conv1d")
        self.assertEqual(type(model.bn3), nn.Identity,
                         msg="Fused Conv + BN + Relu for Conv1d (Skipped BN)")

        self.assertEqual(type(model.sub1.conv), nn.Conv2d,
                         msg="Fused submodule Conv + folded BN")
        self.assertEqual(type(model.sub1.bn), nn.Identity,
                         msg="Fused submodule (skipped BN)")
        self.assertEqual(type(model.sub2.conv), nn.Conv2d,
                         msg="Non-fused submodule Conv")
        self.assertEqual(type(model.sub2.relu), torch.nn.ReLU,
                         msg="Non-fused submodule ReLU")

        model = prepare(model)
        self.checkObservers(model)
        test_only_eval_fn(model, self.img_data_1d)
        model = convert(model)

        def checkQuantized(model):
            self.assertEqual(type(model.conv3), nniq.ConvReLU1d)
            self.assertEqual(type(model.conv1), nniq.ConvReLU2d)
            self.assertEqual(type(model.bn1), nn.Identity)
            self.assertEqual(type(model.relu1), nn.Identity)
            self.assertEqual(type(model.sub1.conv), nnq.Conv2d)
            self.assertEqual(type(model.sub1.bn), nn.Identity)
            self.assertEqual(type(model.sub2.conv), nn.Conv2d)
            self.assertEqual(type(model.sub2.relu), nn.ReLU)
            self.assertEqual(type(model.bn2), nniq.BNReLU3d)
            test_only_eval_fn(model, self.img_data_1d)
            self.checkNoQconfig(model)

        checkQuantized(model)

        model = ModelForFusion(default_qconfig).eval()
        model = fuse_modules(model, [['conv1', 'bn1', 'relu1'],
                             ['conv2', 'relu2'],
                             ['bn2', 'relu3'],
                             ['sub1.conv', 'sub1.bn'],
                             ['conv3', 'bn3', 'relu4']])
        model = quantize(model, test_only_eval_fn, [self.img_data_1d])
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
                                 msg="Fused Conv + Relu: nni.ConvReLU2d")
                self.assertEqual(type(model.conv1[0]), nn.Conv2d,
                                 msg="Fused Conv + Relu: Conv2d")
                self.assertEqual(type(model.conv1[1]), nn.ReLU,
                                 msg="Fused Conv + Relu: Relu")
                self.assertEqual(type(model.relu1), nn.Identity,
                                 msg="Fused Conv + Relu: Identity")
                for i in range(3):
                    self.assertEqual(type(model.features[i][0]), nni.ConvBnReLU2d,
                                     msg="Fused submodule Conv + folded BN")
                    self.assertEqual(type(model.features[i][1]), nn.Identity,
                                     msg="Fused submodule (skipped BN)")
                    self.assertEqual(type(model.features[i][2]), nn.Identity,
                                     msg="Non-fused submodule Conv")
                self.assertEqual(type(model.classifier[0]), nni.LinearReLU)
                self.assertEqual(type(model.classifier[1]), nn.Identity)
                model.qconfig = torch.quantization.get_default_qat_qconfig(qengine)
                prepare_qat(model, inplace=True)
                self.checkObservers(model)
                model(self.img_data_2d[0][0])


                def checkQAT(model):
                    self.assertEqual(type(model.conv1), nniqat.ConvReLU2d)
                    self.assertEqual(type(model.relu1), nn.Identity)
                for i in range(3):
                    self.assertEqual(type(model.features[i][0]), nniqat.ConvBnReLU2d,
                                     msg="Fused submodule Conv + folded BN")
                    self.assertEqual(type(model.features[i][1]), nn.Identity,
                                     msg="Fused submodule (skipped BN)")
                    self.assertEqual(type(model.features[i][2]), nn.Identity,
                                     msg="Non-fused submodule Conv")
                self.assertEqual(type(model.classifier[0]), nniqat.LinearReLU)
                self.assertEqual(type(model.classifier[1]), nn.Identity)

                checkQAT(model)
                model(self.img_data_2d[1][0])
                convert(model, inplace=True)
                model(self.img_data_2d[1][0])
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
                                 msg="Fused Conv + Relu: nni.ConvReLU2d")
                self.assertEqual(type(model.conv1[0]), nn.Conv2d,
                                 msg="Fused Conv + Relu: Conv2d")
                self.assertEqual(type(model.conv1[1]), nn.ReLU,
                                 msg="Fused Conv + Relu: Relu")
                self.assertEqual(type(model.relu1), nn.Identity,
                                 msg="Fused Conv + Relu: Identity")
                for i in range(3):
                    self.assertEqual(type(model.features[i][0]), nni.ConvReLU2d,
                                     msg="Fused submodule Conv + folded BN")
                    self.assertEqual(type(model.features[i][1]), nn.Identity,
                                     msg="Fused submodule (skipped BN)")
                    self.assertEqual(type(model.features[i][2]), nn.Identity,
                                     msg="Non-fused submodule Conv")
                self.assertEqual(type(model.classifier[0]), nni.LinearReLU)
                self.assertEqual(type(model.classifier[1]), nn.Identity)
                model.qconfig = torch.quantization.get_default_qconfig(qengine)
                prepare(model, inplace=True)
                self.checkObservers(model)
                model(self.img_data_2d[0][0])
                convert(model, inplace=True)
                model(self.img_data_2d[1][0])
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
                out_ref = model(self.img_data_2d[0][0])

                model.qconfig = QConfig(activation=torch.nn.Identity,
                                        weight=torch.nn.Identity)
                model = fuse_modules(model, [["conv1", "bn1", "relu1"],
                                             ["conv2", "bn2"]])
                prep_model = prepare_qat(model, inplace=False)
                # output with fusion but no observers.
                out_fused = prep_model(self.img_data_2d[0][0])
                self.assertEqual(out_ref, out_fused)

                model.qconfig = torch.quantization.get_default_qconfig(qengine)
                prepare_qat(model, inplace=True)

                model(self.img_data_2d[0][0])

                def checkQAT(model):
                    self.assertEqual(type(model.conv1), nniqat.ConvBnReLU2d)
                    self.assertEqual(type(model.bn1), nn.Identity)
                    self.assertEqual(type(model.relu1), nn.Identity)
                    self.assertEqual(type(model.conv2), nniqat.ConvBn2d)
                    self.assertEqual(type(model.bn2), nn.Identity)

                checkQAT(model)


    def test_fusion_linear_bn_eval(self):
        model = ModelForLinearBNFusion().train()
        inp1 = torch.randn(8, 20)
        inp2 = torch.randn(8, 20)

        # Get some interesting values into the running mean and variance.
        model(inp1)
        model.eval()
        golden = model(inp2)

        model = fuse_modules(model, [["fc", "bn"]])
        self.assertEqual(type(model.bn), nn.Identity)
        self.assertEqual(golden, model(inp2))

    def test_forward_hooks_preserved(self):
        r"""Test case that checks whether forward pre hooks of the first module and
        post forward hooks of the last module in modules list passed to fusion function preserved.
        (e.g. before fusion: [nn.Conv2d (with pre forward hooks), nn.BatchNorm2d, nn.ReLU (with post forward hooks)]
        after fusion: [nni.ConvBnReLU2d (with pre and post hooks), nn.Identity, nn.Identity])
        """
        model = ModelForFusion(default_qat_qconfig).train()

        counter = {
            'pre_forwards': 0,
            'forwards': 0,
        }
        fused = False

        def fw_pre_hook(fused_module_class, h_module, input):
            if fused:
                self.assertEqual(type(h_module), fused_module_class,
                                 "After fusion owner of the first module's forward pre hook is not a fused module")
            counter['pre_forwards'] += 1

        def fw_hook(fused_module_class, h_module, input, output):
            if fused:
                self.assertEqual(type(h_module), fused_module_class,
                                 "After fusion owner of the last module's forward hook is not a fused module")
            counter['forwards'] += 1

        # Registering two pre and two post forward hooks, thus expecting counter increment by two each inference
        model.conv1.register_forward_pre_hook(lambda *args: fw_pre_hook(nni.ConvBnReLU2d, *args))
        model.sub1.conv.register_forward_pre_hook(lambda *args: fw_pre_hook(nni.ConvBn2d, *args))
        model.relu1.register_forward_hook(lambda *args: fw_hook(nni.ConvBnReLU2d, *args))
        model.sub1.bn.register_forward_hook(lambda *args: fw_hook(nni.ConvBn2d, *args))

        test_only_eval_fn(model, self.img_data_1d)
        self.assertEqual(counter['pre_forwards'], 2 * len(self.img_data_1d))
        self.assertEqual(counter['forwards'], 2 * len(self.img_data_1d))

        model = fuse_modules(model, ['conv1', 'bn1', 'relu1'])
        model = fuse_modules(model, ['sub1.conv', 'sub1.bn'])

        fused = True
        before_fusion_pre_count = counter['pre_forwards']
        before_fusion_post_count = counter['forwards']
        test_only_eval_fn(model, self.img_data_1d)
        self.assertEqual(counter['pre_forwards'] - before_fusion_pre_count, 2 * len(self.img_data_1d))
        self.assertEqual(counter['forwards'] - before_fusion_post_count, 2 * len(self.img_data_1d))

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


class TestDeprecatedJitQuantized(JitTestCase):
    @skipIfNoFBGEMM
    def test_rnn_cell_quantized(self):
        d_in, d_hid = 2, 2

        for cell in [
            torch.nn.LSTMCell(d_in, d_hid).float(),
            torch.nn.GRUCell(d_in, d_hid).float(),
            torch.nn.RNNCell(d_in, d_hid).float(),
        ]:
            if isinstance(cell, torch.nn.LSTMCell):
                num_chunks = 4
            elif isinstance(cell, torch.nn.GRUCell):
                num_chunks = 3
            elif isinstance(cell, torch.nn.RNNCell):
                num_chunks = 1

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
            vals = vals[:d_hid * num_chunks]
            cell.weight_ih = torch.nn.Parameter(
                torch.tensor(vals, dtype=torch.float),
                requires_grad=False)
            cell.weight_hh = torch.nn.Parameter(
                torch.tensor(vals, dtype=torch.float),
                requires_grad=False)

            ref = copy.deepcopy(cell)

            cell = torch.jit.quantized.quantize_rnn_cell_modules(cell)
            x = torch.tensor([[100, -155],
                              [-155, 100],
                              [100, -155]], dtype=torch.float)
            h0_vals = [[-155, 100],
                       [-155, 155],
                       [100, -155]]
            hx = torch.tensor(h0_vals, dtype=torch.float)
            if isinstance(cell, torch.jit.quantized.QuantizedLSTMCell):
                cx = torch.tensor(h0_vals, dtype=torch.float)
                hiddens = (hx, cx)
            else:
                hiddens = hx

            if isinstance(cell, torch.jit.quantized.QuantizedLSTMCell):
                class ScriptWrapper(torch.jit.ScriptModule):
                    def __init__(self, cell):
                        super(ScriptWrapper, self).__init__()
                        self.cell = cell

                    @torch.jit.script_method
                    def forward(self, x: torch.Tensor,
                                hiddens: Tuple[torch.Tensor, torch.Tensor]
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
                        return self.cell(x, hiddens)
            else:

                class ScriptWrapper(torch.jit.ScriptModule):
                    def __init__(self, cell):
                        super(ScriptWrapper, self).__init__()
                        self.cell = cell

                    @torch.jit.script_method
                    def forward(self, x: torch.Tensor, hiddens: torch.Tensor) -> torch.Tensor:
                        return self.cell(x, hiddens)

            cell = ScriptWrapper(cell)
            outs = cell(x, hiddens)
            cell = self.getExportImportCopyWithPacking(cell)

            outs = cell(x, hiddens)
            ref_outs = ref(x, hiddens)

            self.assertEqual(len(outs), len(ref_outs))
            for out, ref_out in zip(outs, ref_outs):
                torch.testing.assert_allclose(out, ref_out)

    @skipIfNoFBGEMM
    def test_rnn_quantized(self):
        d_in, d_hid = 2, 2

        for cell in [
            torch.nn.LSTM(d_in, d_hid).float(),
            torch.nn.GRU(d_in, d_hid).float(),
        ]:

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
            elif isinstance(cell, torch.nn.GRU):
                num_chunks = 3
            vals = vals[:d_hid * num_chunks]
            cell.weight_ih_l0 = torch.nn.Parameter(
                torch.tensor(vals, dtype=torch.float),
                requires_grad=False)
            cell.weight_hh_l0 = torch.nn.Parameter(
                torch.tensor(vals, dtype=torch.float),
                requires_grad=False)

            ref = copy.deepcopy(cell)
            cell_int8 = torch.jit.quantized.quantize_rnn_modules(cell, dtype=torch.int8)
            cell_fp16 = torch.jit.quantized.quantize_rnn_modules(cell, dtype=torch.float16)

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
            elif isinstance(ref, torch.nn.GRU):
                hiddens = hx

            ref_out, ref_hid = ref(x, hiddens)

            # Compare int8 quantized to unquantized
            output_int8, final_hiddens_int8 = cell_int8(x, hiddens)

            torch.testing.assert_allclose(output_int8, ref_out)
            for out, ref in zip(final_hiddens_int8, ref_hid):
                torch.testing.assert_allclose(out, ref)

            # Compare fp16 quantized to unquantized
            output_fp16, final_hiddens_fp16 = cell_fp16(x, hiddens)

            torch.testing.assert_allclose(output_fp16, ref_out)
            for out, ref in zip(final_hiddens_fp16, ref_hid):
                torch.testing.assert_allclose(out, ref)

            def compare_quantized_unquantized(ScriptWrapper, cell):
                wrapper = ScriptWrapper(cell)

                # Compare quantize scripted module to unquantized
                script_out, script_hid = wrapper(x, hiddens)
                torch.testing.assert_allclose(script_out, ref_out)
                for out, ref in zip(script_hid, ref_hid):
                    torch.testing.assert_allclose(out, ref)

                # Compare export/import to unquantized
                export_import_wrapper = self.getExportImportCopyWithPacking(wrapper)
                ei_out, ei_hid = export_import_wrapper(x, hiddens)
                torch.testing.assert_allclose(ei_out, ref_out)
                for out, ref in zip(ei_hid, ref_hid):
                    torch.testing.assert_allclose(out, ref)

            if isinstance(cell, torch.jit.quantized.QuantizedGRU):
                class ScriptWrapper(torch.jit.ScriptModule):
                    def __init__(self, cell):
                        super(ScriptWrapper, self).__init__()
                        self.cell = cell

                    @torch.jit.script_method
                    def forward(self, x: torch.Tensor, hiddens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                        return self.cell(x, hiddens)

                compare_quantized_unquantized(ScriptWrapper, cell)
            elif isinstance(cell, torch.jit.quantized.QuantizedLSTM):
                for cell in [cell_int8, cell_fp16]:
                    class ScriptWrapper(torch.jit.ScriptModule):
                        def __init__(self, cell):
                            super(ScriptWrapper, self).__init__()
                            self.cell = cell

                        @torch.jit.script_method
                        def forward(self, x, hiddens):
                            # type: (torch.Tensor, Tuple[torch.Tensor, torch.Tensor])
                            #        -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
                            return self.cell(x, hiddens)
                    compare_quantized_unquantized(ScriptWrapper, cell)

    if 'fbgemm' in torch.backends.quantized.supported_engines:
        # Suppression: using deprecated quant api
        @suppress_warnings
        def test_quantization_modules(self):
            K1, N1 = 2, 2

            class FooBar(torch.nn.Module):
                def __init__(self):
                    super(FooBar, self).__init__()
                    self.linear1 = torch.nn.Linear(K1, N1).float()

                def forward(self, x):
                    x = self.linear1(x)
                    return x

            fb = FooBar()
            fb.linear1.weight = torch.nn.Parameter(
                torch.tensor([[-150, 100], [100, -150]], dtype=torch.float), requires_grad=False)
            fb.linear1.bias = torch.nn.Parameter(torch.zeros_like(fb.linear1.bias), requires_grad=False)

            x = (torch.rand(1, K1).float() - 0.5) / 10.0
            value = torch.tensor([[100, -150]], dtype=torch.float)

            y_ref = fb(value)

            fb_int8 = torch.jit.quantized.quantize_linear_modules(fb)
            traced_int8 = torch.jit.trace(fb_int8, (x,))
            fb_int8 = self.getExportImportCopyWithPacking(traced_int8)
            y_int8 = fb_int8(value)

            fb_fp16 = torch.jit.quantized.quantize_linear_modules(fb, torch.float16)
            traced_fp16 = torch.jit.trace(fb_fp16, (x,))
            fb_fp16 = self.getExportImportCopyWithPacking(traced_fp16)
            y_fp16 = fb_fp16(value)

            torch.testing.assert_allclose(y_int8, y_ref, rtol=0.0001, atol=1e-3)
            torch.testing.assert_allclose(y_fp16, y_ref, rtol=0.0001, atol=1e-3)

    def _test_pickle_checkpoint_qtensor(self, device):
        with TemporaryFileName() as fname:
            class M(torch.jit.ScriptModule):
                __constants__ = ['fname']

                def __init__(self):
                    super(M, self).__init__()
                    self.fname = fname

                @torch.jit.script_method
                def forward(self, x, y):
                    torch.save((x, y), self.fname)
                    return y

            q = torch.quantize_per_tensor(
                torch.rand(2, 3, dtype=torch.float), scale=0.1, zero_point=10, dtype=torch.quint8).to(device)
            qc = torch.quantize_per_channel(
                torch.rand(2, 3, dtype=torch.float),
                scales=torch.tensor([0.1, 0.5, 0.01]),
                zero_points=torch.tensor([10, 0, 20]),
                axis=1, dtype=torch.quint8).to(device)
            m = M()
            m(q, qc)
            with open(fname, "rb") as handle:
                loaded_q, loaded_qc = torch.load(fname)
                self.assertEqual(loaded_q, q)
                self.assertEqual(loaded_qc, qc)

    def test_pickle_checkpoint_qtensor(self):
        self._test_pickle_checkpoint_qtensor('cpu')

    def test_serialize_qtensor(self):
        class SimpleQTensor(torch.jit.ScriptModule):
            def __init__(self, per_channel):
                super(SimpleQTensor, self).__init__()
                x = torch.rand(5, 5).float()
                if not per_channel:
                    x_q = torch.quantize_per_tensor(x, 0.2, 10, torch.quint8)
                else:
                    s = torch.rand(5, dtype=torch.float64) + 0.1
                    zp = torch.randint(5, 15, (5,))
                    x_q = torch.quantize_per_channel(x, s, zp, 1, torch.quint8)
                self.register_buffer('x', x_q)

            @torch.jit.script_method
            def forward(self):
                return self.x

        for per_channel in [False, True]:
            model = SimpleQTensor(per_channel)
            buffer = io.BytesIO()
            torch.jit.save(model, buffer)
            buffer.seek(0)
            model_loaded = torch.jit.load(buffer)
            self.assertEqual(model_loaded(), model())

    @skipIfNoFBGEMM
    def test_erase_class_tensor_shapes(self):
        class Linear(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super(Linear, self).__init__()
                qweight = torch._empty_affine_quantized(
                    [out_features, in_features], scale=1, zero_point=0,
                    dtype=torch.qint8)
                self._packed_weight = torch.ops.quantized.linear_prepack(qweight)

            @torch.jit.export
            def __getstate__(self):
                return (torch.ops.quantized.linear_unpack(self._packed_weight)[0], self.training)

            def forward(self):
                return self._packed_weight

            @torch.jit.export
            def __setstate__(self, state):
                self._packed_weight = torch.ops.quantized.linear_prepack(state[0])
                self.training = state[1]

            @property
            def weight(self):
                return torch.ops.quantized.linear_unpack(self._packed_weight)[0]

            @weight.setter
            def weight(self, w):
                self._packed_weight = torch.ops.quantized.linear_prepack(w)

        with torch._jit_internal._disable_emit_hooks():
            x = torch.jit.script(Linear(10, 10))
            torch._C._jit_pass_erase_shape_information(x.graph)

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_quantization.py TESTNAME\n\n"
                       "instead.")
