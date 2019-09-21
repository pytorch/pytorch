import unittest
import math
import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn._intrinsic as nni
import torch.nn._intrinsic.quantized as nniq
import torch.nn._intrinsic.qat as nniqat
from torch.quantization import \
    QConfig_dynamic, default_weight_observer, get_observer_dict,\
    quantize, prepare, convert, prepare_qat, quantize_qat, fuse_modules, \
    quantize_dynamic, default_qconfig, default_debug_qconfig, default_qat_qconfig, \
    default_dynamic_qconfig, HistogramObserver, MinMaxObserver, PerChannelMinMaxObserver, RecordingObserver, QuantWrapper

from common_utils import run_tests
from common_quantization import QuantizationTestCase, SingleLayerLinearModel, \
    SkipQuantModel, QuantStubModel, \
    ModelForFusion, ManualLinearQATModel, ManualConvLinearQATModel, \
    ModForWrapping, \
    test_only_eval_fn, test_only_train_fn, \
    prepare_dynamic, convert_dynamic, SingleLayerLinearDynamicModel, \
    TwoLayerLinearModel, NestedModel, ResNetBase, LSTMDynamicModel

from common_quantization import AnnotatedTwoLayerLinearModel, AnnotatedNestedModel, \
    AnnotatedSubNestedModel, AnnotatedCustomConfigNestedModel

from hypothesis import given
from hypothesis import strategies as st
from hypothesis_utils import no_deadline
import io
import copy

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
        prepare(model)
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
        model = quantize(SingleLayerLinearModel(), test_only_eval_fn,
                         self.calib_data)
        checkQuantized(model)

    def test_two_layers(self):
        r"""TwoLayerLinearModel has two Linear modules but we only quantize the second one
        `fc2`, and `fc1`is not quantized
        """
        model = AnnotatedTwoLayerLinearModel()
        prepare(model)

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
        model = quantize(AnnotatedTwoLayerLinearModel(), test_only_eval_fn,
                         self.calib_data)
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

        prepare(model)
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
        model = quantize(AnnotatedNestedModel(), test_only_eval_fn,
                         self.calib_data)
        checkQuantized(model)


    def test_nested2(self):
        model = AnnotatedSubNestedModel()
        prepare(model)

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
        model = quantize(AnnotatedSubNestedModel(), test_only_eval_fn,
                         self.calib_data)
        checkQuantized(model)

    def test_nested3(self):
        r"""More complicated nested test case with child qconfig overrides
        parent qconfig
        """
        model = AnnotatedCustomConfigNestedModel()
        prepare(model)

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
        model = quantize(AnnotatedCustomConfigNestedModel(), test_only_eval_fn,
                         self.calib_data)
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

    def test_resnet_base(self):
        r"""Test quantization for bottleneck topology used in resnet/resnext
        and add coverage for conversion of average pool and float functional
        """
        model = ResNetBase().float().eval()
        model = QuantWrapper(model)
        model.qconfig = default_qconfig
        fuse_list = [['module.conv1', 'module.bn1', 'module.relu1']]
        fuse_modules(model, fuse_list)
        prepare(model)
        self.checkObservers(model)
        test_only_eval_fn(model, self.img_data)
        convert(model)

        def checkQuantized(model):
            self.assertEqual(type(model.module.conv1), nn._intrinsic.quantized.ConvReLU2d)
            self.assertEqual(type(model.module.myop), nn.quantized.QFunctional)
            self.assertEqual(type(model.module.avgpool), nn.AdaptiveAvgPool2d)
            test_only_eval_fn(model, self.img_data)

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
        prepare_dynamic(model, qconfig_dict)
        convert_dynamic(model)

        def checkQuantized(model):
            self.checkDynamicQuantizedLinear(model.fc1)
            self.checkScriptable(model, self.calib_data, check_save_load=True)

        checkQuantized(model)

        # test one line API
        model = quantize_dynamic(SingleLayerLinearDynamicModel().eval(),
                                 qconfig_dict)
        checkQuantized(model)

    def test_two_layers(self):
        r"""TwoLayerLinearModel has two Linear modules but we only quantize the second one
        `fc2`, and `fc1`is not quantized
        """
        model = TwoLayerLinearModel().eval()
        qconfig_dict = {
            'fc2': default_dynamic_qconfig
        }
        prepare_dynamic(model, qconfig_dict)

        convert_dynamic(model)

        def checkQuantized(model):
            self.assertEqual(type(model.fc1), torch.nn.Linear)
            self.checkDynamicQuantizedLinear(model.fc2)
            self.checkScriptable(model, self.calib_data, check_save_load=True)

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

        prepare_dynamic(model, qconfig_dict)
        convert_dynamic(model)

        def checkQuantized(model):
            self.checkLinear(model.sub1.fc)
            self.checkDynamicQuantizedLinear(model.fc3)
            self.checkDynamicQuantizedLinear(model.sub2.fc1)
            self.checkLinear(model.sub2.fc2)
            self.checkScriptable(model, self.calib_data, check_save_load=True)

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
        prepare_dynamic(model, qconfig_dict)

        convert_dynamic(model)

        def checkQuantized(model):
            self.checkLinear(model.sub1.fc)
            self.assertEqual(type(model.sub1.relu), torch.nn.ReLU)
            self.checkDynamicQuantizedLinear(model.sub2.fc1)
            self.checkDynamicQuantizedLinear(model.sub2.fc2)
            self.checkDynamicQuantizedLinear(model.fc3)
            self.checkScriptable(model, self.calib_data, check_save_load=True)

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
        prepare_dynamic(model, qconfig_dynamic_dict)

        convert_dynamic(model)

        def checkQuantized(model):
            self.checkDynamicQuantizedLinear(model.sub2.fc1)
            self.checkDynamicQuantizedLinear(model.sub2.fc2)
            self.checkDynamicQuantizedLinear(model.fc3)
            self.checkScriptable(model, self.calib_data, check_save_load=True)

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

        prepare_dynamic(model, qconfig_dict)
        test_only_eval_fn(model, self.calib_data)
        convert_dynamic(model)

        def checkQuantized(model):
            self.checkDynamicQuantizedLinear(model.sub1.fc)
            self.checkLinear(model.fc3)
            self.checkLinear(model.sub2.fc1)
            self.checkDynamicQuantizedLinear(model.sub2.fc2)
            test_only_eval_fn(model, self.calib_data)
            self.checkScriptable(model, self.calib_data, check_save_load=True)

        checkQuantized(model)

        # test one line API
        model = quantize_dynamic(NestedModel().eval(), qconfig_dict)
        checkQuantized(model)

    def test_quantized_rnn(self):
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

        qconfig_dynamic_dict = {
            torch.nn.LSTM: default_dynamic_qconfig,
        }
        default_dynamic_module_mapping = {
            torch.nn.LSTM: torch.nn.quantized.dynamic.LSTM,
        }
        model_int8 = quantize_dynamic(
            model=model, qconfig_dict=qconfig_dynamic_dict, mapping=default_dynamic_module_mapping,
            dtype=torch.qint8
        )
        model_fp16 = quantize_dynamic(
            model=model, qconfig_dict=qconfig_dynamic_dict, mapping=default_dynamic_module_mapping,
            dtype=torch.float16
        )
        cell_int8 = model_int8.lstm
        cell_fp16 = model_fp16.lstm

        assert type(cell_int8) == torch.nn.quantized.dynamic.LSTM, \
            'torch.nn.LSTM should be converted to torch.nn.quantized.dynamic.LSTM after quantize_dynamic'
        assert type(cell_fp16) == torch.nn.quantized.dynamic.LSTM, \
            'torch.nn.LSTM should be converted to torch.nn.quantized.dynamic.LSTM after quantize_dynamic'

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

        # Compare int8 quantized to unquantized
        output_int8, final_hiddens_int8 = cell_int8(x, hiddens)

        torch.testing.assert_allclose(output_int8, ref_out)
        self.assertEqual(output_int8, ref_out)
        for out_val, ref_val in zip(final_hiddens_int8, ref_hid):
            torch.testing.assert_allclose(out_val, ref_val)

        class ScriptWrapper(torch.nn.Module):
            def __init__(self, cell):
                super(ScriptWrapper, self).__init__()
                self.cell = cell

            def forward(self, x, hiddens):
                # type: (torch.Tensor, Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
                return self.cell(x, hiddens)

        # TODO: TorchScript overloads don't work without this wrapper
        cell_script = torch.jit.script(ScriptWrapper(cell_int8))
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

        # Compare fp16 quantized to unquantized
        output_fp16, final_hiddens_fp16 = cell_fp16(x, hiddens)

        torch.testing.assert_allclose(output_fp16, ref_out)
        self.assertEqual(output_fp16, ref_out)
        for out, ref in zip(final_hiddens_fp16, ref_hid):
            torch.testing.assert_allclose(out, ref)

@unittest.skipIf(
    not torch.fbgemm_is_cpu_supported(),
    " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
    " with instruction set support avx2 or newer.",
)
class QuantizationAwareTrainingTest(QuantizationTestCase):
    def test_manual(self):
        model = ManualLinearQATModel()
        prepare_qat(model)
        self.checkObservers(model)
        test_only_train_fn(model, self.train_data)
        convert(model)

        def checkQuantized(model):
            self.assertEqual(type(model.fc1), nnq.Linear)
            self.assertEqual(type(model.fc2), nnq.Linear)
            test_only_eval_fn(model, self.calib_data)
            self.checkScriptable(model, self.calib_data)

        checkQuantized(model)

        model = quantize_qat(ManualLinearQATModel(), test_only_train_fn,
                             self.train_data)
        checkQuantized(model)

    def test_eval_only_fake_quant(self):
        r"""Using FakeQuant in evaluation only mode,
        this is useful for estimating accuracy loss when we quantize the
        network
        """
        model = ManualLinearQATModel()

        prepare_qat(model)
        self.checkObservers(model)

        model.eval()
        test_only_eval_fn(model, self.calib_data)

    def test_conv_linear(self):
        model = ManualConvLinearQATModel()

        prepare_qat(model)
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
        self.qx = torch.quantize_per_tensor(self.x.to(torch.float), scale=1.0,
                                            zero_point=0, dtype=torch.qint32)

    def test_scriptability_serialization(self):
        # test serialization of quantized functional modules
        b = io.BytesIO()
        torch.save(self.qmodel_under_test, b)
        b.seek(0)
        loaded = torch.load(b)
        self.assertEqual(self.qmodel_under_test.myadd.zero_point, loaded.myadd.zero_point)
        state_dict = self.qmodel_under_test.state_dict()
        self.assertTrue('myadd.zero_point' in state_dict.keys(),
                        'zero point not in state dict for functional modules')

        x = torch.rand(10, 1, dtype=torch.float)
        xq = torch.quantize_per_tensor(x, 1.0, 0, torch.qint8)
        self.checkScriptable(self.qmodel_under_test, [(xq, xq)], check_save_load=True)
        self.checkScriptable(self.model_under_test, [(xq.dequantize(), xq.dequantize())], check_save_load=True)

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
        prepare_qat(model)
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

        prepare(model)
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
           qscheme=st.sampled_from((torch.per_tensor_affine, torch.per_tensor_symmetric)),
           reduce_range=st.booleans())
    def test_minmax_observer(self, qdtype, qscheme, reduce_range):
        # reduce_range cannot be true for symmetric quantization with uint8
        if qdtype == torch.quint8 and qscheme == torch.per_tensor_symmetric:
            reduce_range = False
        myobs = MinMaxObserver(dtype=qdtype, qscheme=qscheme, reduce_range=reduce_range)
        x = torch.tensor([1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y = torch.tensor([4.0, 5.0, 5.0, 6.0, 7.0, 8.0])
        result = myobs(x)
        result = myobs(y)
        self.assertEqual(result, y)
        self.assertEqual(myobs.min_val, 1.0)
        self.assertEqual(myobs.max_val, 8.0)
        qparams = myobs.calculate_qparams()
        if reduce_range:
            if qscheme == torch.per_tensor_symmetric:
                ref_scale = 0.062745 * 255 / 127
                ref_zero_point = 0 if qdtype is torch.qint8 else 128
            else:
                ref_scale = 0.0313725 * 255 / 127
                ref_zero_point = -64 if qdtype is torch.qint8 else 0
        else:
            if qscheme == torch.per_tensor_symmetric:
                ref_scale = 0.062745
                ref_zero_point = 0 if qdtype is torch.qint8 else 128
            else:
                ref_scale = 0.0313725
                ref_zero_point = -128 if qdtype is torch.qint8 else 0
        self.assertEqual(qparams[1].item(), ref_zero_point)
        self.assertAlmostEqual(qparams[0].item(), ref_scale, delta=1e-5)

    @given(qdtype=st.sampled_from((torch.qint8, torch.quint8)),
           qscheme=st.sampled_from((torch.per_channel_affine, torch.per_channel_symmetric)),
           ch_axis=st.sampled_from((0, 1, 2, 3)), reduce_range=st.booleans())
    def test_per_channel_minmax_observer(self, qdtype, qscheme, ch_axis, reduce_range):
        # reduce_range cannot be true for symmetric quantization with uint8
        if qdtype == torch.quint8 and qscheme == torch.per_channel_symmetric:
            reduce_range = False
        myobs = PerChannelMinMaxObserver(reduce_range=reduce_range, ch_axis=ch_axis, dtype=qdtype, qscheme=qscheme)
        x = torch.tensor(
            [
                [[[1.0, 2.0], [2.0, 2.5]], [[3.0, 4.0], [4.5, 6.0]]],
                [[[-4.0, -3.0], [5.0, 5.0]], [[6.0, 3.0], [7.0, 8.0]]],
            ]
        )
        result = myobs(x)
        self.assertEqual(result, x)
        qparams = myobs.calculate_qparams()
        ref_min_vals = [[1.0, -4.0], [-4.0, 3.0], [-4.0, 2.0], [-4.0, -3.0]]
        ref_max_vals = [[6.0, 8.0], [5.0, 8.0], [6.0, 8.0], [7.0, 8.0]]
        per_channel_symmetric_ref_scales = [
            [0.04705882, 0.06274509],
            [0.03921569, 0.0627451],
            [0.04705882, 0.0627451],
            [0.05490196, 0.0627451],
        ]
        per_channel_affine_ref_scales = [
            [0.02352941, 0.04705882],
            [0.03529412, 0.03137255],
            [0.03921569, 0.03137255],
            [0.04313726, 0.04313726],
        ]
        per_channel_affine_qint8_zp = [
            [-128, -43],
            [-15, -128],
            [-26, -128],
            [-35, -58],
        ]
        per_channel_affine_quint8_zp = [[0, 85], [113, 0], [102, 0], [93, 70]]

        self.assertEqual(myobs.min_vals, ref_min_vals[ch_axis])
        self.assertEqual(myobs.max_vals, ref_max_vals[ch_axis])
        if qscheme == torch.per_channel_symmetric:
            ref_scales = per_channel_symmetric_ref_scales[ch_axis]
            ref_zero_points = [0, 0] if qdtype is torch.qint8 else [128, 128]
        else:
            ref_scales = per_channel_affine_ref_scales[ch_axis]
            ref_zero_points = (
                per_channel_affine_qint8_zp[ch_axis]
                if qdtype is torch.qint8
                else per_channel_affine_quint8_zp[ch_axis]
            )

        if reduce_range:
            ref_scales = [s * 255 / 127 for s in ref_scales]
            ref_zero_points = [math.floor(z / 2) for z in ref_zero_points]

        self.assertTrue(torch.allclose(qparams[0], torch.tensor(ref_scales, dtype=qparams[0].dtype)))
        self.assertTrue(torch.allclose(qparams[1], torch.tensor(ref_zero_points, dtype=qparams[1].dtype)))

    def test_observer_scriptable(self):
        obs = torch.quantization.default_observer()()
        scripted = torch.jit.script(obs)

        x = torch.rand(3, 4)
        obs(x)
        scripted(x)

        self.assertEqual(obs.calculate_qparams(), scripted.calculate_qparams())

        buf = io.BytesIO()
        torch.jit.save(scripted, buf)
        buf.seek(0)
        loaded = torch.jit.load(buf)
        self.assertEqual(obs.calculate_qparams(), loaded.calculate_qparams())

@unittest.skipIf(not torch.fbgemm_is_cpu_supported(),
                 'Quantization requires FBGEMM. FBGEMM does not play'
                 ' well with UBSAN at the moment, so we skip the test if'
                 ' we are in a UBSAN environment.')
class QuantizationDebugTest(QuantizationTestCase):
    def test_record_observer(self):
        model = SingleLayerLinearModel()
        model.qconfig = default_debug_qconfig
        prepare(model)
        # run the evaluation and dump all tensors
        test_only_eval_fn(model, self.calib_data)
        test_only_eval_fn(model, self.calib_data)
        observer_dict = {}
        get_observer_dict(model, observer_dict)

        self.assertTrue('fc1.module.observer' in observer_dict.keys(),
                        'observer is not recorded in the dict')
        self.assertEqual(len(observer_dict['fc1.module.observer'].get_tensor_value()), 2 * len(self.calib_data))
        self.assertEqual(observer_dict['fc1.module.observer'].get_tensor_value()[0], model(self.calib_data[0][0]))


    @given(qdtype=st.sampled_from((torch.qint8, torch.quint8)),
           qscheme=st.sampled_from((torch.per_tensor_affine, torch.per_tensor_symmetric)))
    def test_observer_observer_scriptable(self, qdtype, qscheme):
        obs = RecordingObserver(dtype=qdtype, qscheme=qscheme)
        scripted = torch.jit.script(obs)

        x = torch.rand(3, 4)
        obs(x)
        scripted(x)
        self.assertTrue(torch.equal(obs.get_tensor_value()[0], scripted.get_tensor_value()[0]))
        buf = io.BytesIO()
        torch.jit.save(scripted, buf)
        buf.seek(0)
        loaded = torch.jit.load(buf)
        self.assertTrue(torch.equal(obs.get_tensor_value()[0], loaded.get_tensor_value()[0]))

    @no_deadline
    @given(qdtype=st.sampled_from((torch.qint8, torch.quint8)),
           qscheme=st.sampled_from((torch.per_tensor_affine, torch.per_tensor_symmetric)),
           reduce_range=st.booleans())
    def test_histogram_observer(self, qdtype, qscheme, reduce_range):
        myobs = HistogramObserver(bins=3, dtype=qdtype, qscheme=qscheme, reduce_range=reduce_range)
        x = torch.tensor([2.0, 3.0, 4.0, 5.0])
        y = torch.tensor([5.0, 6.0, 7.0, 8.0])
        myobs(x)
        myobs(y)
        self.assertEqual(myobs.min_val, 2.0)
        self.assertEqual(myobs.max_val, 8.0)
        self.assertEqual(myobs.histogram, [2., 3., 3.])

        qparams = myobs.calculate_qparams()

        if reduce_range:
            if qscheme == torch.per_tensor_symmetric:
                ref_scale = 0.0470588 * 255 / 127
                ref_zero_point = 0 if qdtype is torch.qint8 else 128
            else:
                ref_scale = 0.0235294 * 255 / 127
                ref_zero_point = -64 if qdtype is torch.qint8 else 0
        else:
            if qscheme == torch.per_tensor_symmetric:
                ref_scale = 0.0470588
                ref_zero_point = 0 if qdtype is torch.qint8 else 128
            else:
                ref_scale = 0.0235294
                ref_zero_point = -128 if qdtype is torch.qint8 else 0

        self.assertEqual(qparams[1].item(), ref_zero_point)
        self.assertAlmostEqual(qparams[0].item(), ref_scale, delta=1e-5)

if __name__ == '__main__':
    run_tests()
