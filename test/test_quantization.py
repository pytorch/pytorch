import unittest
import math
import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nniq
import torch.nn.intrinsic.qat as nniqat
from torch.quantization import \
    QConfigDynamic, get_observer_dict, default_weight_observer, \
    quantize, prepare, convert, prepare_qat, quantize_qat, fuse_modules, \
    quantize_dynamic, default_qconfig, default_debug_qconfig, default_qat_qconfig, \
    default_dynamic_qconfig, HistogramObserver, MinMaxObserver, PerChannelMinMaxObserver,\
    RecordingObserver, MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver, \
    QuantWrapper, default_eval_fn

from torch.quantization import QConfig
from torch.quantization import default_histogram_observer
from torch.quantization._quantize_script import quantize_script

from common_utils import run_tests
from common_quantization import QuantizationTestCase, \
    AnnotatedSingleLayerLinearModel, SingleLayerLinearModel, \
    AnnotatedConvModel, ConvModel, \
    AnnotatedConvBnModel, ConvBnModel, \
    SkipQuantModel, QuantStubModel, \
    ModelForFusion, ModelWithSequentialFusion, ManualLinearQATModel, ManualConvLinearQATModel, \
    ModelWithFunctionals, \
    test_only_eval_fn, test_only_train_fn, \
    prepare_dynamic, convert_dynamic, SingleLayerLinearDynamicModel, \
    TwoLayerLinearModel, NestedModel, ResNetBase, LSTMDynamicModel, \
    ModelWithNoQconfigPropagation

from common_quantization import AnnotatedTwoLayerLinearModel, AnnotatedNestedModel, \
    AnnotatedSubNestedModel, AnnotatedCustomConfigNestedModel

from jit_utils import _tmp_donotuse_dont_inline_everything
from jit_utils import get_forward

from hypothesis import given
from hypothesis import strategies as st
from hypothesis_utils import no_deadline
import io
import copy

@unittest.skipUnless('fbgemm' in torch.backends.quantized.supported_engines,
                     " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
                     " with instruction set support avx2 or newer.")
class EagerModePostTrainingQuantTest(QuantizationTestCase):
    @no_deadline
    @given(qconfig=st.sampled_from((torch.quantization.default_qconfig, torch.quantization.default_per_channel_qconfig)))
    def test_single_layer(self, qconfig):
        r"""Quantize SingleLayerLinearModel which has one Linear module, make sure it is swapped
        to nnq.Linear which is the quantized version of the module
        """
        model = AnnotatedSingleLayerLinearModel()
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
        base = AnnotatedSingleLayerLinearModel()
        base.qconfig = qconfig
        keys_before = set(list(base.state_dict().keys()))
        model = quantize(base, test_only_eval_fn, self.calib_data)
        checkQuantized(model)
        keys_after = set(list(base.state_dict().keys()))
        self.assertEqual(keys_before, keys_after)  # simple check that nothing changed

        # in-place version
        model = AnnotatedSingleLayerLinearModel()
        model.qconfig = qconfig
        quantize(model, test_only_eval_fn, self.calib_data, inplace=True)
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
        model = quantize(AnnotatedNestedModel(), test_only_eval_fn,
                         self.calib_data)
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

        model = SkipQuantModel()
        model = prepare(model)
        self.checkObservers(model)

        test_only_eval_fn(model, self.calib_data)
        model = convert(model)

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

    @given(qconfig=st.sampled_from((torch.quantization.default_qconfig, torch.quantization.default_per_channel_qconfig)))
    def test_resnet_base(self, qconfig):
        r"""Test quantization for bottleneck topology used in resnet/resnext
        and add coverage for conversion of average pool and float functional
        """
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

@unittest.skipUnless('fbgemm' in torch.backends.quantized.supported_engines,
                     " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
                     " with instruction set support avx2 or newer.")
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
        quantize_dynamic(model, set([nn.Linear]), inplace=True)
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

        # Test set API
        model = quantize_dynamic(TwoLayerLinearModel().eval(), {'fc2'})
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

        model = quantize_dynamic(NestedModel().eval(), {'fc3', 'sub2.fc1'})
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

        # Test set API
        model = quantize_dynamic(NestedModel().eval(), {'fc3', 'sub2'})
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
        custom_dynamic_qconfig = QConfigDynamic(weight=default_weight_observer)
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

        # Test set API
        model = quantize_dynamic(NestedModel().eval(), {'fc3', 'sub2', 'sub2.fc1'})
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

        model_int8 = quantize_dynamic(model=model, dtype=torch.qint8)
        model_fp16 = quantize_dynamic(model=model, dtype=torch.float16)

        # Smoke test extra reprs
        self.assertTrue('DynamicQuantizedLSTM' in str(model_int8))
        self.assertTrue('DynamicQuantizedLSTM' in str(model_fp16))
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

@unittest.skipUnless('fbgemm' in torch.backends.quantized.supported_engines,
                     " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
                     " with instruction set support avx2 or newer.")
class EagerModeQuantizationAwareTrainingTest(QuantizationTestCase):
    def test_manual(self):
        model = ManualLinearQATModel()
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

        model = quantize_qat(ManualLinearQATModel(), test_only_train_fn,
                             self.train_data)
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


@unittest.skipUnless(
    'fbgemm' in torch.backends.quantized.supported_engines,
    " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
    " with instruction set support avx2 or newer.",
)
class GraphModePostTrainingQuantTest(QuantizationTestCase):
    @_tmp_donotuse_dont_inline_everything
    def test_single_linear(self):
        r"""Compare the result of quantizing single linear layer in
        eager mode and graph mode
        """
        # eager mode
        annotated_linear_model = AnnotatedSingleLayerLinearModel()
        linear_model = SingleLayerLinearModel()
        # copy the weight from eager mode so that we can
        # compare the result of the two quantized models later
        linear_model.fc1.weight = torch.nn.Parameter(annotated_linear_model.fc1.module.weight.detach())
        linear_model.fc1.bias = torch.nn.Parameter(annotated_linear_model.fc1.module.bias.detach())
        model_eager = quantize(annotated_linear_model, test_only_eval_fn,
                               self.calib_data)

        qconfig_dict = {
            '': default_qconfig
        }
        model_script = quantize_script(
            torch.jit.script(linear_model),
            qconfig_dict,
            test_only_eval_fn,
            [self.calib_data],
            inplace=False)
        result_eager = model_eager(self.calib_data[0][0])
        result_script = model_script._c._get_method('forward')(self.calib_data[0][0])
        self.assertEqual(result_eager, result_script)

    def test_observer_with_ignored_function(self):
        r"""Test observers with ignored fucntion and make sure it works in
        graph mode
        """
        # eager mode
        annotated_linear_model = AnnotatedSingleLayerLinearModel().eval()
        qconfig = QConfig(
            activation=default_histogram_observer,
            weight=default_weight_observer)
        annotated_linear_model.qconfig = qconfig
        linear_model = SingleLayerLinearModel().eval()
        # copy the weight from eager mode so that we can
        # compare the result of the two quantized models later
        linear_model.fc1.weight = torch.nn.Parameter(annotated_linear_model.fc1.module.weight.detach())
        linear_model.fc1.bias = torch.nn.Parameter(annotated_linear_model.fc1.module.bias.detach())
        model_eager = quantize(annotated_linear_model, test_only_eval_fn,
                               self.calib_data)

        qconfig_dict = {
            '': qconfig
        }
        model_script = quantize_script(
            torch.jit.script(linear_model),
            qconfig_dict,
            test_only_eval_fn,
            [self.calib_data],
            inplace=False)
        result_eager = model_eager(self.calib_data[0][0])
        result_script = get_forward(model_script._c)(self.calib_data[0][0])
        self.assertEqual(result_eager, result_script)

    @_tmp_donotuse_dont_inline_everything
    def test_conv(self):
        r"""Compare the result of quantizing conv layer in
        eager mode and graph mode
        """
        # eager mode
        conv_model = AnnotatedConvModel().eval()
        conv_model_to_script = ConvModel().eval()
        # copy the weight from eager mode so that we can
        # compare the result of the two quantized models later
        conv_model_to_script.conv.weight = torch.nn.Parameter(conv_model.conv.weight.detach())
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

    @unittest.skip("quantization for inlined linear is not working right now")
    def test_nested(self):
        # Eager mode
        eager_model = AnnotatedNestedModel()
        # default_per_channel_qconfig is not scriptable right now,
        # temporarily change to default_qconfig until default_per_channel_qconfig is fixed
        eager_model.sub2.fc1.qconfig = default_qconfig

        # Graph mode
        script_model = NestedModel()
        # Copy weights for eager_model
        script_model.sub1.fc.weight = torch.nn.Parameter(eager_model.sub1.fc.weight.detach())
        script_model.sub1.fc.bias = torch.nn.Parameter(eager_model.sub1.fc.bias.detach())
        script_model.sub2.fc1.weight = torch.nn.Parameter(eager_model.sub2.fc1.module.weight.detach())
        script_model.sub2.fc1.bias = torch.nn.Parameter(eager_model.sub2.fc1.module.bias.detach())
        script_model.sub2.fc2.weight = torch.nn.Parameter(eager_model.sub2.fc2.weight.detach())
        script_model.sub2.fc2.bias = torch.nn.Parameter(eager_model.sub2.fc2.bias.detach())
        script_model.fc3.weight = torch.nn.Parameter(eager_model.fc3.module.weight.detach())
        script_model.fc3.bias = torch.nn.Parameter(eager_model.fc3.module.bias.detach())
        # Quantize eager module
        quantized_eager_model = quantize(eager_model, test_only_eval_fn, self.calib_data)

        qconfig_dict = {
            'sub2.fc1': default_qconfig,
            'fc3': default_qconfig
        }
        quantized_script_model = quantize_script(
            torch.jit.script(script_model),
            qconfig_dict,
            test_only_eval_fn,
            [self.calib_data],
            inplace=False)

        eager_result = quantized_eager_model(self.calib_data[0][0])
        script_result = get_forward(quantized_script_model._c)(self.calib_data[0][0])
        self.assertEqual(eager_result, script_result)


class FunctionalModuleTest(QuantizationTestCase):
    # Histogram Observers are slow, so have no-deadline to ensure test doesn't time out
    @no_deadline
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

@unittest.skipUnless('fbgemm' in torch.backends.quantized.supported_engines,
                     " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
                     " with instruction set support avx2 or newer.")
class FusionTest(QuantizationTestCase):
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

        model = ModelForFusion(default_qconfig).eval()
        model = fuse_modules(model, [['conv1', 'bn1', 'relu1'],
                             ['sub1.conv', 'sub1.bn']])
        model = quantize(model, test_only_eval_fn, self.img_data)
        checkQuantized(model)

    def test_fusion_sequential_model_train(self):
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
        model.qconfig = default_qat_qconfig
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
        model.qconfig = default_qconfig
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


class ObserverTest(QuantizationTestCase):
    @given(qdtype=st.sampled_from((torch.qint8, torch.quint8)),
           qscheme=st.sampled_from((torch.per_tensor_affine, torch.per_tensor_symmetric)),
           reduce_range=st.booleans())
    def test_per_tensor_observers(self, qdtype, qscheme, reduce_range):
        # reduce_range cannot be true for symmetric quantization with uint8
        if qdtype == torch.quint8 and qscheme == torch.per_tensor_symmetric:
            reduce_range = False
        ObserverList = [MinMaxObserver(dtype=qdtype, qscheme=qscheme, reduce_range=reduce_range),
                        MovingAverageMinMaxObserver(averaging_constant=0.5,
                                                    dtype=qdtype,
                                                    qscheme=qscheme,
                                                    reduce_range=reduce_range)]
        for myobs in ObserverList:
            # Calculate Qparams should return with a warning for observers with no data
            qparams = myobs.calculate_qparams()
            if type(myobs) == MinMaxObserver:
                x = torch.tensor([1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                y = torch.tensor([4.0, 5.0, 5.0, 6.0, 7.0, 8.0])
            else:
                # Moving average of min/max for x and y matches that of
                # extreme values for x/y used for minmax observer
                x = torch.tensor([0.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                y = torch.tensor([2.0, 5.0, 5.0, 6.0, 7.0, 10.0])

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
            state_dict = myobs.state_dict()
            b = io.BytesIO()
            torch.save(state_dict, b)
            b.seek(0)
            loaded_dict = torch.load(b)
            for key in state_dict:
                self.assertEqual(state_dict[key], loaded_dict[key])
            loaded_obs = MinMaxObserver(dtype=qdtype, qscheme=qscheme, reduce_range=reduce_range)
            loaded_obs.load_state_dict(loaded_dict)
            loaded_qparams = loaded_obs.calculate_qparams()
            self.assertEqual(myobs.min_val, loaded_obs.min_val)
            self.assertEqual(myobs.max_val, loaded_obs.max_val)
            self.assertEqual(myobs.calculate_qparams(), loaded_obs.calculate_qparams())

    @given(qdtype=st.sampled_from((torch.qint8, torch.quint8)),
           qscheme=st.sampled_from((torch.per_channel_affine, torch.per_channel_symmetric)),
           ch_axis=st.sampled_from((0, 1, 2, 3)), reduce_range=st.booleans())
    def test_per_channel_observers(self, qdtype, qscheme, ch_axis, reduce_range):
        # reduce_range cannot be true for symmetric quantization with uint8
        if qdtype == torch.quint8 and qscheme == torch.per_channel_symmetric:
            reduce_range = False
        ObserverList = [PerChannelMinMaxObserver(reduce_range=reduce_range,
                                                 ch_axis=ch_axis,
                                                 dtype=qdtype,
                                                 qscheme=qscheme),
                        MovingAveragePerChannelMinMaxObserver(averaging_constant=0.5,
                                                              reduce_range=reduce_range,
                                                              ch_axis=ch_axis,
                                                              dtype=qdtype,
                                                              qscheme=qscheme)]

        for myobs in ObserverList:
            # Calculate qparams should work for empty observers
            qparams = myobs.calculate_qparams()
            x = torch.tensor(
                [
                    [[[1.0, 2.0], [2.0, 2.5]], [[3.0, 4.0], [4.5, 6.0]]],
                    [[[-4.0, -3.0], [5.0, 5.0]], [[6.0, 3.0], [7.0, 8.0]]],
                ]
            )
            if type(myobs) == MovingAveragePerChannelMinMaxObserver:
                # Scaling the input tensor to model change in min/max values
                # across batches
                result = myobs(0.5 * x)
                result = myobs(1.5 * x)
                self.assertEqual(result, 1.5 * x)
            else:
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

            # Test for serializability
            state_dict = myobs.state_dict()
            b = io.BytesIO()
            torch.save(state_dict, b)
            b.seek(0)
            loaded_dict = torch.load(b)
            for key in state_dict:
                self.assertEqual(state_dict[key], loaded_dict[key])
            loaded_obs = PerChannelMinMaxObserver(reduce_range=reduce_range, ch_axis=ch_axis, dtype=qdtype, qscheme=qscheme)
            loaded_obs.load_state_dict(loaded_dict)
            loaded_qparams = loaded_obs.calculate_qparams()
            self.assertEqual(myobs.min_vals, loaded_obs.min_vals)
            self.assertEqual(myobs.max_vals, loaded_obs.max_vals)
            self.assertEqual(myobs.calculate_qparams(), loaded_obs.calculate_qparams())

    def test_observer_scriptable(self):
        obs_list = [MinMaxObserver(), MovingAverageMinMaxObserver()]
        for obs in obs_list:
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

    def test_no_qconfig_propagation(self):
        model = ModelWithNoQconfigPropagation()
        model.qconfig = torch.quantization.default_qconfig

        model = prepare(model)
        self.assertTrue(hasattr(model.fc1, 'qconfig'),
                        "QConfig is expected to propagate")
        self.assertFalse(hasattr(model.no_quant_module, 'qconfig'),
                         "QConfig is expected to NOT propagate")


@unittest.skipUnless('fbgemm' in torch.backends.quantized.supported_engines,
                     " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
                     " with instruction set support avx2 or newer.")
class RecordHistogramObserverTest(QuantizationTestCase):
    def test_record_observer(self):
        model = AnnotatedSingleLayerLinearModel()
        model.qconfig = default_debug_qconfig
        model = prepare(model)
        # run the evaluation and dump all tensors
        test_only_eval_fn(model, self.calib_data)
        test_only_eval_fn(model, self.calib_data)
        observer_dict = {}
        get_observer_dict(model, observer_dict)

        self.assertTrue('fc1.module.activation_post_process' in observer_dict.keys(),
                        'observer is not recorded in the dict')
        self.assertEqual(len(observer_dict['fc1.module.activation_post_process'].get_tensor_value()), 2 * len(self.calib_data))
        self.assertEqual(observer_dict['fc1.module.activation_post_process'].get_tensor_value()[0], model(self.calib_data[0][0]))

    @no_deadline
    @given(qdtype=st.sampled_from((torch.qint8, torch.quint8)),
           qscheme=st.sampled_from((torch.per_tensor_affine, torch.per_tensor_symmetric)))
    def test_observer_scriptable(self, qdtype, qscheme):
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
        # Calculate qparams should work for empty observers
        qparams = myobs.calculate_qparams()
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
        # Test for serializability
        state_dict = myobs.state_dict()
        b = io.BytesIO()
        torch.save(state_dict, b)
        b.seek(0)
        loaded_dict = torch.load(b)
        for key in state_dict:
            self.assertEqual(state_dict[key], loaded_dict[key])
        loaded_obs = HistogramObserver(bins=3, dtype=qdtype, qscheme=qscheme, reduce_range=reduce_range)
        loaded_obs.load_state_dict(loaded_dict)
        loaded_qparams = loaded_obs.calculate_qparams()
        self.assertEqual(myobs.min_val, loaded_obs.min_val)
        self.assertEqual(myobs.max_val, loaded_obs.max_val)
        self.assertEqual(myobs.histogram, loaded_obs.histogram)
        self.assertEqual(myobs.bins, loaded_obs.bins)
        self.assertEqual(myobs.calculate_qparams(), loaded_obs.calculate_qparams())


if __name__ == '__main__':
    run_tests()
