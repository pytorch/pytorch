# Torch
import torch
from torch.quantization import (
    MinMaxObserver,
    PerChannelMinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    MinMaxDynamicQuantObserver,
    HistogramObserver,
    RecordingObserver,
    FakeQuantize,
    default_debug_qconfig,
    default_observer,
    default_per_channel_weight_observer,
    get_observer_dict,
    prepare,
)
import torch.nn as nn

# Standard library
import copy
import io
import unittest
import math
import numpy as np

# Testing utils
from hypothesis import given
from hypothesis import strategies as st
import torch.testing._internal.hypothesis_utils as hu
hu.assert_deadline_disabled()
from torch.testing._internal.common_cuda import TEST_MULTIGPU, TEST_CUDA
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    ModelWithNoQconfigPropagation,
    AnnotatedSingleLayerLinearModel,
    test_only_eval_fn,
)

from torch.testing._internal.common_quantized import (
    override_quantized_engine,
    supported_qengines,
)

# Reference method for fake quantize
def _fake_quantize_per_tensor_affine_reference(X, scale, zero_point, quant_min, quant_max):
    res = (torch.clamp(torch.round(X * (1.0 / scale) + zero_point), quant_min, quant_max) - zero_point) * scale
    return res

# Reference method for the gradient of the fake quantize operator
def _fake_quantize_per_tensor_affine_grad_reference(dY, X, scale, zero_point, quant_min, quant_max):
    Xq = torch.round(X * (1.0 / scale) + zero_point)
    mask = (Xq >= quant_min) * (Xq <= quant_max)
    res = torch.zeros_like(dY)
    res[mask] = dY[mask]
    return res

# Helper function used to simulate per-channel fake-quant against any axis
def _permute_to_axis_zero(X, axis):
    new_axis_list = list(range(X.dim()))
    new_axis_list[axis] = 0
    new_axis_list[0] = axis
    y = X.permute(tuple(new_axis_list))
    return y, new_axis_list

# Reference method for fake quantize
def _fake_quantize_per_channel_affine_reference(X, per_channel_scale, per_channel_zero_point, axis, quant_min, quant_max):
    X, permute_axis_list = _permute_to_axis_zero(X, axis)
    res = torch.zeros_like(X)

    for i in range(X.size()[0]):
        res[i] = (torch.clamp(torch.round(X[i] * (1.0 / per_channel_scale[i]) +
                  per_channel_zero_point[i]), quant_min, quant_max) - per_channel_zero_point[i]) * per_channel_scale[i]

    out = res.permute(tuple(permute_axis_list))
    return out

# Reference method for the gradient of the fake quantize operator
def _fake_quantize_per_channel_affine_grad_reference(dY, X, per_channel_scale, per_channel_zero_point, axis, quant_min, quant_max):
    X, permute_axis_list = _permute_to_axis_zero(X, axis)
    Xq = torch.zeros_like(X)
    for i in range(X.size()[0]):
        Xq[i] = torch.round(X[i] * (1.0 / per_channel_scale[i]) + per_channel_zero_point[i])
    Xq = Xq.permute(tuple(permute_axis_list))
    mask = (Xq >= quant_min) * (Xq <= quant_max)
    res = torch.zeros_like(dY)
    res[mask] = dY[mask]
    return res

def to_tensor(X, device):
    return torch.tensor(X).to(device=torch.device(device), dtype=torch.float32)

NP_RANDOM_SEED = 19
tolerance = 1e-6


class TestObserver(QuantizationTestCase):
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


    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=2, max_dims=4,
                                              min_side=1, max_side=10),
                       qparams=hu.qparams()),
           reduce_range=st.booleans())
    def test_per_tensor_dynamic_quant_observers(self, X, reduce_range):

        X, (scale, zero_point, torch_type) = X
        x = torch.from_numpy(X)

        obs = MinMaxDynamicQuantObserver(dtype=torch.quint8, reduce_range=reduce_range)

        result = obs(x)
        qparams = obs.calculate_qparams()
        ref = torch._choose_qparams_per_tensor(x, reduce_range)

        self.assertEqual(ref[0], qparams[0])
        self.assertEqual(ref[1], qparams[1])

    def test_tensor_list_observer(self):
        from torch.quantization.observer import _MinMaxTensorListObserver
        x = [torch.tensor([1.0, 2.5, 3.5]),
             torch.tensor([2.0, 4.5, 3.5]),
             torch.tensor([4.0, 2.5, 3.5]), ]
        obs = _MinMaxTensorListObserver()
        obs(x)
        qparams = obs.calculate_qparams()
        ref_min_val = []
        ref_max_val = []
        ref_qparams = []
        for i in x:
            obs_ref = MinMaxObserver()
            obs_ref(i)
            ref_min_val.append(obs_ref.min_val)
            ref_max_val.append(obs_ref.max_val)
            ref_qparams.append(obs_ref.calculate_qparams())
        for i in range(len(x)):
            self.assertEqual(obs.min_val[i], ref_min_val[i])
            self.assertEqual(obs.max_val[i], ref_max_val[i])
            self.assertEqual(qparams[0][i], ref_qparams[i][0])
            self.assertEqual(qparams[1][i], ref_qparams[i][1])

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
        obs_list = [MinMaxObserver(), MovingAverageMinMaxObserver(), MinMaxDynamicQuantObserver()]
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

        # Check TensorListObserver
        from torch.quantization.observer import _MinMaxTensorListObserver
        obs = _MinMaxTensorListObserver()
        scripted = torch.jit.script(obs)
        x = [torch.rand(3, 4), torch.rand(4, 5)]
        obs(x)
        scripted(x)
        self.assertEqual(obs.calculate_qparams(), scripted.calculate_qparams())

    # TODO: move this to test_quantize.py
    def test_no_qconfig_propagation(self):
        model = ModelWithNoQconfigPropagation()
        model.qconfig = torch.quantization.default_qconfig

        model = prepare(model)
        self.assertTrue(hasattr(model.fc1, 'qconfig'),
                        "QConfig is expected to propagate")
        self.assertFalse(hasattr(model.no_quant_module, 'qconfig'),
                         "QConfig is expected to NOT propagate")


class TestRecordHistogramObserver(QuantizationTestCase):
    # TODO: move this to quantize.py
    def test_record_observer(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
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
                self.assertEqual(len(observer_dict['fc1.module.activation_post_process'].get_tensor_value()),
                                 2 * len(self.calib_data))
                self.assertEqual(observer_dict['fc1.module.activation_post_process'].get_tensor_value()[0],
                                 model(self.calib_data[0][0]))

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

    @given(qdtype=st.sampled_from((torch.qint8, torch.quint8)),
           qscheme=st.sampled_from((torch.per_tensor_affine, torch.per_tensor_symmetric)),
           reduce_range=st.booleans())
    def test_histogram_observer(self, qdtype, qscheme, reduce_range):
        myobs = HistogramObserver(bins=3, dtype=qdtype, qscheme=qscheme, reduce_range=reduce_range)
        # Calculate qparams should work for empty observers
        qparams = myobs.calculate_qparams()
        x = torch.tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
        y = torch.tensor([5.0, 6.0, 7.0, 8.0])
        out_x = myobs(x)
        self.assertTrue(out_x.requires_grad)
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

    def test_histogram_observer_one_sided(self):
        myobs = HistogramObserver(bins=8, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=True)
        x = torch.tensor([0.0, 0.3, 1.2, 1.7])
        y = torch.tensor([0.1, 1.3, 2.0, 2.7])
        myobs(x)
        myobs(y)
        self.assertEqual(myobs.min_val, 0)
        qparams = myobs.calculate_qparams()
        self.assertEqual(qparams[1].item(), 0)

class TestFakeQuantizePerTensor(TestCase):
    @given(device=st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']),
           X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       qparams=hu.qparams(dtypes=torch.quint8)))
    def test_forward_per_tensor(self, device, X):
        r"""Tests the forward path of the FakeQuantizePerTensorAffine op.
        """
        np.random.seed(NP_RANDOM_SEED)
        X, (scale, zero_point, torch_type) = X
        quant_min = torch.iinfo(torch_type).min
        quant_max = torch.iinfo(torch_type).max

        X = to_tensor(X, device)
        Y = _fake_quantize_per_tensor_affine_reference(X.cpu(), scale, zero_point, quant_min, quant_max)
        Y_prime = torch.fake_quantize_per_tensor_affine(
            X, scale, zero_point, quant_min, quant_max)
        np.testing.assert_allclose(Y, Y_prime.cpu(), rtol=tolerance, atol=tolerance)

    @given(device=st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']),
           X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       qparams=hu.qparams(dtypes=torch.quint8)))
    @unittest.skip("temporarily disable the test")
    def test_backward_per_tensor(self, device, X):
        r"""Tests the backward method.
        """
        np.random.seed(NP_RANDOM_SEED)
        X, (scale, zero_point, torch_type) = X
        quant_min = torch.iinfo(torch_type).min
        quant_max = torch.iinfo(torch_type).max

        X = to_tensor(X, device)
        X.requires_grad_()
        Y = _fake_quantize_per_tensor_affine_reference(X.cpu(), scale, zero_point, quant_min, quant_max)
        Y_prime = torch.fake_quantize_per_tensor_affine(
            X, scale, zero_point, quant_min, quant_max)
        dout = torch.rand(X.shape, dtype=torch.float).to(device)
        dX = _fake_quantize_per_tensor_affine_grad_reference(
            dout, X, scale, zero_point, quant_min, quant_max)
        Y_prime.backward(dout)
        np.testing.assert_allclose(dX.cpu(), X.grad.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)

    @given(device=st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']),
           X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       qparams=hu.qparams(dtypes=torch.quint8)))
    # https://github.com/pytorch/pytorch/issues/30604
    @unittest.skip("temporarily disable the test")
    def test_numerical_consistency_per_tensor(self, device, X):
        r"""Comparing numerical consistency between CPU quantize/dequantize op and the CPU fake quantize op
        """
        np.random.seed(NP_RANDOM_SEED)
        X, (scale, zero_point, torch_type) = X
        quant_min = torch.iinfo(torch_type).min
        quant_max = torch.iinfo(torch_type).max

        X = to_tensor(X, device)
        # quantize_per_tensor and dequantize are only implemented in CPU
        Y = torch.dequantize(torch.quantize_per_tensor(X.cpu(), scale, zero_point, torch_type))
        Y_prime = torch.fake_quantize_per_tensor_affine(
            X, scale, zero_point, quant_min, quant_max)
        np.testing.assert_allclose(Y, Y_prime.cpu(), rtol=tolerance, atol=tolerance)

    @given(device=st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']),
           X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       qparams=hu.qparams(dtypes=[torch.quint8])),
           )
    def test_fq_module(self, device, X):
        np.random.seed(NP_RANDOM_SEED)
        X, (scale, zero_point, torch_type) = X
        quant_min = torch.iinfo(torch_type).min
        quant_max = torch.iinfo(torch_type).max

        X = to_tensor(X, device)
        X.requires_grad_()
        fq_module = torch.quantization.default_fake_quant().to(device)
        Y_prime = fq_module(X)
        assert fq_module.scale is not None
        assert fq_module.zero_point is not None
        Y = _fake_quantize_per_tensor_affine_reference(X, fq_module.scale, fq_module.zero_point, quant_min, quant_max)
        np.testing.assert_allclose(Y.cpu().detach().numpy(), Y_prime.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)

        # Test backward
        dout = torch.rand(X.shape, dtype=torch.float, device=device)
        Y_prime.backward(dout)
        dX = _fake_quantize_per_tensor_affine_grad_reference(dout, X, fq_module.scale, fq_module.zero_point, quant_min, quant_max)
        np.testing.assert_allclose(dX.cpu().numpy(), X.grad.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)

    def test_fq_serializable(self):
        observer = default_observer
        quant_min = 0
        quant_max = 255
        fq_module = FakeQuantize(observer, quant_min, quant_max)
        X = torch.tensor([-5, -3.5, -2, 0, 3, 5, 7], dtype=torch.float32)
        y_ref = fq_module(X)
        state_dict = fq_module.state_dict()
        self.assertEqual(state_dict['scale'], 0.094488)
        self.assertEqual(state_dict['zero_point'], 53)
        b = io.BytesIO()
        torch.save(state_dict, b)
        b.seek(0)
        loaded_dict = torch.load(b)
        loaded_fq_module = FakeQuantize(observer, quant_min, quant_max)
        loaded_fq_module.load_state_dict(loaded_dict)
        for key in state_dict:
            self.assertEqual(state_dict[key], loaded_fq_module.state_dict()[key])

        self.assertEqual(loaded_fq_module.calculate_qparams(), fq_module.calculate_qparams())

    def test_fake_quant_control(self):
        torch.manual_seed(42)
        X = torch.rand(20, 10, dtype=torch.float32)
        fq_module = torch.quantization.default_fake_quant()
        # Output of fake quant is not identical to input
        Y = fq_module(X)
        self.assertNotEqual(Y, X)
        torch.quantization.disable_fake_quant(fq_module)
        X = torch.rand(20, 10, dtype=torch.float32)
        Y = fq_module(X)
        # Fake quant is disabled,output is identical to input
        self.assertEqual(Y, X)

        # Explicit copy at this point in time, because FakeQuant keeps internal
        # state in mutable buffers.
        scale = fq_module.scale.clone().detach()
        zero_point = fq_module.zero_point.clone().detach()

        torch.quantization.disable_observer(fq_module)
        torch.quantization.enable_fake_quant(fq_module)
        X = 10.0 * torch.rand(20, 10, dtype=torch.float32) - 5.0
        Y = fq_module(X)
        self.assertNotEqual(Y, X)
        # Observer is disabled, scale and zero-point do not change
        self.assertEqual(fq_module.scale, scale)
        self.assertEqual(fq_module.zero_point, zero_point)
        torch.quantization.enable_observer(fq_module)
        Y = fq_module(X)
        self.assertNotEqual(Y, X)
        # Observer is enabled, scale and zero-point are different
        self.assertNotEqual(fq_module.scale, scale)
        self.assertNotEqual(fq_module.zero_point, zero_point)



class TestFakeQuantizePerChannel(TestCase):

    @given(device=st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']),
           X=hu.per_channel_tensor(shapes=hu.array_shapes(1, 5,),
           qparams=hu.qparams(dtypes=torch.quint8)))
    def test_forward_per_channel(self, device, X):
        r"""Tests the forward path of the FakeQuantizePerTensorAffine op.
        """
        np.random.seed(NP_RANDOM_SEED)
        X, (scale, zero_point, axis, torch_type) = X
        quant_min = torch.iinfo(torch_type).min
        quant_max = torch.iinfo(torch_type).max

        X = to_tensor(X, device)
        scale = to_tensor(scale, device)
        zero_point = torch.tensor(zero_point).to(dtype=torch.int64, device=device)
        Y = _fake_quantize_per_channel_affine_reference(X.cpu(), scale.cpu(), zero_point.cpu(), axis, quant_min, quant_max)
        Y_prime = torch.fake_quantize_per_channel_affine(
            X, scale, zero_point, axis, quant_min, quant_max)
        np.testing.assert_allclose(Y, Y_prime.cpu(), rtol=tolerance, atol=tolerance)

    @given(device=st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']),
           X=hu.per_channel_tensor(shapes=hu.array_shapes(1, 5,),
           qparams=hu.qparams(dtypes=torch.quint8)))
    def test_backward_per_channel(self, device, X):
        r"""Tests the backward method.
        """
        np.random.seed(NP_RANDOM_SEED)
        X, (scale, zero_point, axis, torch_type) = X
        quant_min = torch.iinfo(torch_type).min
        quant_max = torch.iinfo(torch_type).max

        X = to_tensor(X, device)
        scale = to_tensor(scale, device)
        zero_point = torch.tensor(zero_point).to(dtype=torch.int64, device=device)
        X.requires_grad_()
        Y_prime = torch.fake_quantize_per_channel_affine(
            X, scale, zero_point, axis, quant_min, quant_max)
        dout = torch.rand(X.shape, dtype=torch.float).to(device)
        dX = _fake_quantize_per_channel_affine_grad_reference(
            dout, X, scale, zero_point, axis, quant_min, quant_max)
        Y_prime.backward(dout)
        np.testing.assert_allclose(dX.cpu().detach().numpy(), X.grad.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)

    @given(device=st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']),
           X=hu.per_channel_tensor(shapes=hu.array_shapes(1, 5,),
           qparams=hu.qparams(dtypes=torch.quint8)))
    @unittest.skip("temporarily disable the test")
    def test_numerical_consistency_per_channel(self, device, X):
        r"""Comparing numerical consistency between CPU quantize/dequantize op and the CPU fake quantize op
        """
        np.random.seed(NP_RANDOM_SEED)
        X, (scale, zero_point, axis, torch_type) = X
        quant_min = torch.iinfo(torch_type).min
        quant_max = torch.iinfo(torch_type).max

        X = to_tensor(X, device)
        scale = to_tensor(scale, device)
        zero_point = torch.tensor(zero_point).to(dtype=torch.int64, device=device)
        # quantize_linear and dequantize are only implemented in CPU
        Y = torch.dequantize(torch.quantize_per_channel(X.cpu(), scale.cpu(), zero_point.cpu(), axis, torch_type))
        Y_prime = torch.fake_quantize_per_channel_affine(
            X, scale, zero_point, axis, quant_min, quant_max)
        np.testing.assert_allclose(Y, Y_prime.cpu(), rtol=tolerance, atol=tolerance)

    @given(device=st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']),
           X=hu.per_channel_tensor(shapes=hu.array_shapes(2, 5,),
           qparams=hu.qparams(dtypes=torch.qint8)))
    def test_fq_module(self, device, X):
        np.random.seed(NP_RANDOM_SEED)
        X, (scale, zero_point, axis, torch_type) = X
        quant_min = torch.iinfo(torch_type).min
        quant_max = torch.iinfo(torch_type).max

        X = to_tensor(X, device)
        X.requires_grad_()
        fq_module = FakeQuantize(default_per_channel_weight_observer, quant_min, quant_max, ch_axis=axis).to(device)
        Y_prime = fq_module(X)
        assert fq_module.scale is not None
        assert fq_module.zero_point is not None
        Y = _fake_quantize_per_channel_affine_reference(X, fq_module.scale,
                                                        fq_module.zero_point, axis, quant_min, quant_max)
        np.testing.assert_allclose(Y.cpu().detach().numpy(), Y_prime.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)

        # Test backward
        dout = torch.rand(X.shape, dtype=torch.float, device=device)
        Y_prime.backward(dout)
        dX = _fake_quantize_per_channel_affine_grad_reference(dout, X, fq_module.scale,
                                                              fq_module.zero_point, axis, quant_min, quant_max)
        np.testing.assert_allclose(dX.cpu().numpy(), X.grad.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)

    def test_fq_serializable(self):
        observer = default_per_channel_weight_observer
        quant_min = -128
        quant_max = 127
        fq_module = FakeQuantize(observer, quant_min, quant_max)
        X = torch.tensor([[-5, -3.5, -2, 0, 3, 5, 7], [1, 3, 2, 5, 6.5, 8, 10]], dtype=torch.float32)
        y_ref = fq_module(X)
        state_dict = fq_module.state_dict()
        self.assertEqual(state_dict['scale'], [0.054902, 0.078431])
        self.assertEqual(state_dict['zero_point'], [0, 0])
        b = io.BytesIO()
        torch.save(state_dict, b)
        b.seek(0)
        loaded_dict = torch.load(b)
        for key in state_dict:
            self.assertEqual(state_dict[key], loaded_dict[key])

def _get_buffer_ids(module):
    """
    Object addresses stay constant if and only if all modifications are in-place
    """
    return [id(v) for k, v in module._buffers.items()]

class TestDistributed(QuantizationTestCase):

    def test_observers_preserve_buffers(self):
        """
        Tests that observers only modify buffers in place. Note: this is important
        because nn.DataParallel depends on this assumption to work correctly.
        However, DataParallel does not expose IDs of the replicas, so we test it
        without DataParallel in order to easily access the object IDs.
        """
        observer_types = [
            torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8),
            torch.quantization.MovingAverageMinMaxObserver.with_args(dtype=torch.qint8),
            torch.quantization.MinMaxDynamicQuantObserver.with_args(dtype=torch.qint8),
            torch.quantization.PerChannelMinMaxObserver.with_args(dtype=torch.qint8),
            torch.quantization.MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8),
            torch.quantization.HistogramObserver.with_args(dtype=torch.qint8),
            torch.quantization.RecordingObserver.with_args(dtype=torch.qint8),
            torch.quantization.NoopObserver.with_args(dtype=torch.float16),
        ]

        for observer_type in observer_types:
            observer = observer_type()
            buffer_ids_before = _get_buffer_ids(observer)
            for _i in range(5):
                inputs = torch.rand((4, 4, 4))
                observer(inputs)
            buffer_ids_after = _get_buffer_ids(observer)
            self.assertEqual(
                buffer_ids_before,
                buffer_ids_after,
                "{}: Buffers must be modified in place".format(str(observer)))

    def test_fake_quant_preserves_buffers(self):
        """
        Tests that fake quant only modifies buffers in place. Note: this is important
        because nn.DataParallel depends on this assumption to work correctly.
        However, DataParallel does not expose IDs of the replicas, so we test it
        without DataParallel in order to easily access the object IDs.
        """
        model = torch.quantization.FakeQuantize()
        buffer_ids_before = _get_buffer_ids(model)
        for _i in range(5):
            inputs = torch.rand((4, 4, 4))
            model(inputs)
        buffer_ids_after = _get_buffer_ids(model)
        self.assertEqual(
            buffer_ids_before,
            buffer_ids_after,
            "FakeQuant: Buffers must be modified in place")

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_qat_data_parallel(self):
        """
        Tests that doing QAT in nn.DataParallel does not crash.
        """
        if 'fbgemm' not in torch.backends.quantized.supported_engines:
            return
        with override_quantized_engine('fbgemm'):
            device = torch.device('cuda')

            model = nn.Sequential(
                torch.quantization.QuantStub(),
                nn.Conv2d(3, 1, 1, bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(),
                nn.Conv2d(1, 2, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(2),
                nn.AvgPool2d(14),
                nn.Sigmoid(),
                torch.quantization.DeQuantStub(),
            )

            torch.quantization.fuse_modules(model, [['1', '2', '3'], ['4', '5']], inplace=True)

            model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            torch.quantization.prepare_qat(model, inplace=True)
            model = nn.DataParallel(model, device_ids=[0, 1])
            model.to(device)
            model.train()

            for epoch in range(3):
                inputs = torch.rand(2, 3, 28, 28).to(device)
                model(inputs)
                if epoch >= 1:
                    model.apply(torch.quantization.disable_observer)
                if epoch >= 2:
                    model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
                quant_model = copy.deepcopy(model.module)
                quant_model = torch.quantization.convert(quant_model.eval().cpu(), inplace=False)
                with torch.no_grad():
                    out = quant_model(torch.rand(1, 3, 28, 28))

    def test_qat_convbn_fused_syncbn_replacement(self):
        """
        Tests that SyncBatchNorm replacement works for fused ConvBN.
        """
        if 'fbgemm' not in torch.backends.quantized.supported_engines:
            return
        with override_quantized_engine('fbgemm'):
            # create conv-bn
            class Model(nn.Module):
                def __init__(self):
                    super(Model, self).__init__()
                    self.conv = nn.Conv2d(4, 1, 3, padding=1)
                    self.bn = nn.BatchNorm2d(1)

                def forward(self, x):
                    x = self.conv(x)
                    x = self.bn(x)
                    return x

            model = Model()
            # fuse it
            fused_model = torch.quantization.fuse_modules(
                model,
                [['conv', 'bn']],
            )
            # convert to QAT
            fused_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare_qat(fused_model, inplace=True)
            # replace with DDP
            fused_model = nn.SyncBatchNorm.convert_sync_batchnorm(fused_model)
            self.assertTrue(
                isinstance(fused_model.conv.bn, nn.SyncBatchNorm),
                "Expected BN to be converted to SyncBN")
