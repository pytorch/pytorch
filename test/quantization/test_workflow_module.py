# Torch
import torch
from torch.quantization import (
    MinMaxObserver,
    PerChannelMinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    HistogramObserver,
    RecordingObserver,
    PlaceholderObserver,
    NoopObserver,
    FakeQuantize,
    FixedQParamsFakeQuantize,
    default_debug_qconfig,
    default_observer,
    default_histogram_observer,
    default_per_channel_weight_observer,
    default_affine_fixed_qparams_fake_quant,
    get_observer_dict,
    prepare,
    QConfig,
)

from torch.quantization._learnable_fake_quantize import _LearnableFakeQuantize

import torch.nn as nn

# Standard library
import copy
import io
import itertools
import unittest
import math
import numpy as np

# Testing utils
from hypothesis import given, settings
from hypothesis import strategies as st
import torch.testing._internal.hypothesis_utils as hu
hu.assert_deadline_disabled()
from torch.testing._internal.common_cuda import TEST_MULTIGPU, TEST_CUDA
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    AnnotatedSingleLayerLinearModel,
    test_only_eval_fn,
    SingleLayerLinearModel,
)

from torch.testing._internal.common_quantized import (
    override_quantized_engine,
    supported_qengines,
    override_qengines,
)

# Reference method for fake quantize
# Note: because scale/zero_point are left as float in the actual kernel, this mimics how fake_quant works for float16/64
def _fake_quantize_per_tensor_affine_reference(X, scale, zero_point, quant_min, quant_max):
    dtype = X.dtype
    res = ((torch.clamp(torch.round(X.to(torch.float32) * (1.0 / scale) + zero_point), quant_min, quant_max) - zero_point) * scale)
    return res.to(dtype)

# Reference method for the gradient of the fake quantize operator
# Note: because scale/zero_point are left as float in the actual kernel, this mimics how fake_quant works for float16/64
def _fake_quantize_per_tensor_affine_grad_reference(dY, X, scale, zero_point, quant_min, quant_max):
    dtype = X.dtype
    Xq = torch.round(X.to(torch.float32) * (1.0 / scale) + zero_point)
    mask = (Xq >= quant_min) * (Xq <= quant_max)
    res = torch.zeros_like(dY)
    res[mask] = dY[mask]
    return res.to(dtype)

# Reference method for the gradients of the fake quantize operator
def _fake_quantize_learnable_per_tensor_affine_grad_reference(dY, X, scale, zero_point, quant_min, quant_max, device):
    r"""This method references the following literatures for back propagation on scale and zero point.
    - https://arxiv.org/pdf/1902.08153.pdf
    - https://arxiv.org/pdf/1903.08066.pdf
    """
    zero_point_rounded = int((zero_point + 0.5).clamp(quant_min, quant_max).item())
    Xq = torch.round(X * (1.0 / scale) + zero_point_rounded)

    indicate_small_scale = (Xq < quant_min).float().to(device)
    indicate_big_scale = (Xq > quant_max).float().to(device)
    indicate_middle_scale = torch.ones(indicate_small_scale.shape).to(device) - \
        indicate_small_scale - indicate_big_scale

    indicate_saturate_zp = ((Xq < quant_min).float() + (Xq > quant_max).float()).to(device)
    indicate_unsaturate_zp = torch.ones(indicate_saturate_zp.shape).to(device) - indicate_saturate_zp

    Xq = Xq.clamp(quant_min, quant_max)
    Xfq = (Xq - zero_point_rounded) * scale

    grad_small_scale = quant_min - zero_point_rounded
    grad_big_scale = quant_max - zero_point_rounded
    grad_middle_scale = ((Xfq - X) / scale).to(device)

    grad_saturate_zp = -scale.to(device)
    grad_unsaturate_zp = 0

    grad_scale = indicate_small_scale * grad_small_scale + \
        indicate_big_scale * grad_big_scale + \
        indicate_middle_scale * grad_middle_scale
    grad_zp = indicate_saturate_zp * grad_saturate_zp + \
        indicate_unsaturate_zp * grad_unsaturate_zp
    grad_X = _fake_quantize_per_tensor_affine_grad_reference(
        dY, X, scale, zero_point, quant_min, quant_max).to(device)

    grad_scale = (grad_scale * dY).sum().unsqueeze(dim=0)
    grad_zp = (grad_zp * dY).sum().unsqueeze(dim=0)
    return grad_X, grad_scale, grad_zp

# Helper function used to simulate per-channel fake-quant against any axis
def _permute_to_axis_zero(X, axis):
    new_axis_list = list(range(X.dim()))
    new_axis_list[axis] = 0
    new_axis_list[0] = axis
    y = X.permute(tuple(new_axis_list))
    return y, new_axis_list

# Reference method for fake quantize
# Note: because scale/zero_point are left as float in the actual kernel, this mimics how fake_quant works for float16/64
def _fake_quantize_per_channel_affine_reference(X, per_channel_scale, per_channel_zero_point, axis, quant_min, quant_max):
    dtype = X.dtype
    X, permute_axis_list = _permute_to_axis_zero(X.to(torch.float32), axis)
    res = torch.zeros_like(X)

    for i in range(X.size()[0]):
        res[i] = (torch.clamp(torch.round(X[i] * (1.0 / per_channel_scale[i]) +
                  per_channel_zero_point[i]), quant_min, quant_max) - per_channel_zero_point[i]) * per_channel_scale[i]

    out = res.permute(tuple(permute_axis_list))
    return out.to(dtype)

# Reference method for the gradient of the fake quantize operator
# Note: because scale/zero_point are left as float in the actual kernel, this mimics how fake_quant works for float16/64
def _fake_quantize_per_channel_affine_grad_reference(dY, X, per_channel_scale, per_channel_zero_point, axis, quant_min, quant_max):
    dtype = X.dtype
    X, permute_axis_list = _permute_to_axis_zero(X.to(torch.float32), axis)
    Xq = torch.zeros_like(X)
    for i in range(X.size()[0]):
        Xq[i] = torch.round(X[i] * (1.0 / per_channel_scale[i]) + per_channel_zero_point[i])
    Xq = Xq.permute(tuple(permute_axis_list))
    mask = (Xq >= quant_min) * (Xq <= quant_max)
    res = torch.zeros_like(dY)
    res[mask] = dY[mask]
    return res.to(dtype)

# Reference method for quantization.
def _quantize_per_tensor(x, scale, zero_point, quant_min, quant_max):
    return ((x / scale) + zero_point).round().clamp(quant_min, quant_max)

# Reference method for the per channel gradients of the learnable fake quantize operator
def _fake_quantize_learnable_per_channel_affine_grad_reference(
        dY, X, per_channel_scale, per_channel_zero_point, axis, quant_min, quant_max, device):
    r"""This method references the following literatures for back propagation on scale and zero point.
    - https://arxiv.org/pdf/1902.08153.pdf
    - https://arxiv.org/pdf/1903.08066.pdf
    """
    per_channel_zero_point = ((per_channel_zero_point.detach() + 0.5).clamp(quant_min, quant_max)).type(torch.int64)
    grad_X = _fake_quantize_per_channel_affine_grad_reference(
        dY, X, per_channel_scale, per_channel_zero_point, axis, quant_min, quant_max).to(device)
    per_channel_scale = per_channel_scale.detach().type(torch.float)

    grad_scale = torch.zeros([per_channel_scale.size(0)]).to(device)
    grad_zero_point = torch.zeros([per_channel_zero_point.size(0)]).to(device)

    X_flattened = torch.unbind(X, dim=axis)
    dY_flattened = torch.unbind(dY, dim=axis)

    for i, X_i in enumerate(torch.unbind(X, dim=axis), 0):
        scale_i = per_channel_scale[i]
        zero_point_i = per_channel_zero_point[i]
        X_i = X_flattened[i]
        dY_i = dY_flattened[i]

        Xq_i = ((X_i / scale_i) + zero_point_i).round()
        Xfq_i = (Xq_i - zero_point_i) * scale_i

        indicate_small_scale_i = (Xq_i < quant_min).float().to(device)
        indicate_big_scale_i = (Xq_i > quant_max).float().to(device)
        indicate_middle_scale_i = torch.ones(indicate_small_scale_i.shape).to(device) - \
            indicate_small_scale_i - indicate_big_scale_i

        indicate_saturate_zp_i = ((Xq_i < quant_min).float() +
                                  (Xq_i > quant_max).float()).to(device)
        indicate_unsaturate_zp_i = torch.ones(indicate_saturate_zp_i.shape).to(device) - \
            indicate_saturate_zp_i

        Xq_i = Xq_i.clamp(quant_min, quant_max)
        Xfq_i = (Xq_i - zero_point_i) * scale_i

        grad_small_scale_i = quant_min - zero_point_i
        grad_big_scale_i = quant_max - zero_point_i
        grad_middle_scale_i = ((Xfq_i - X_i) / scale_i).to(device)

        grad_saturate_zp_i = -scale_i.to(device)
        grad_unsaturate_zp_i = 0

        grad_scale_i = indicate_small_scale_i * grad_small_scale_i + \
            indicate_middle_scale_i * grad_middle_scale_i + \
            indicate_big_scale_i * grad_big_scale_i
        grad_zp_i = indicate_saturate_zp_i * grad_saturate_zp_i + \
            indicate_unsaturate_zp_i * grad_unsaturate_zp_i

        grad_scale_i = (grad_scale_i * dY_i).sum().unsqueeze(dim=0)
        grad_zp_i = (grad_zp_i * dY_i).sum().unsqueeze(dim=0)

        grad_scale[i] = grad_scale_i
        grad_zero_point[i] = grad_zp_i
    return grad_X, grad_scale, grad_zero_point

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
            self.assertEqual(qparams[0].item(), ref_scale, atol=1e-5, rtol=0)
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
           qscheme=st.sampled_from((torch.per_channel_affine, torch.per_channel_symmetric, torch.per_channel_affine_float_qparams)),
           ch_axis=st.sampled_from((0, 1, 2, 3)), reduce_range=st.booleans())
    def test_per_channel_observers(self, qdtype, qscheme, ch_axis, reduce_range):
        # reduce_range cannot be true for symmetric quantization with uint8
        if qscheme == torch.per_channel_affine_float_qparams:
            reduce_range = False
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
            per_channel_affine_float_qparams_ref_scales = [
                [0.0196, 0.0471],
                [0.0353, 0.0196],
                [0.0392, 0.0235],
                [0.0431, 0.0431],
            ]
            per_channel_affine_quint8_zp = [[0, 85], [113, 0], [102, 0], [93, 70]]

            self.assertEqual(myobs.min_vals, ref_min_vals[ch_axis])
            self.assertEqual(myobs.max_vals, ref_max_vals[ch_axis])
            if qscheme == torch.per_channel_symmetric:
                ref_scales = per_channel_symmetric_ref_scales[ch_axis]
                ref_zero_points = [0, 0] if qdtype is torch.qint8 else [128, 128]
            elif qscheme == torch.per_channel_affine_float_qparams:
                ref_scales = per_channel_affine_float_qparams_ref_scales[ch_axis]
                ref_zero_points = [-1 * ref_min_vals[ch_axis][i] / ref_scales[i] for i in range(len(ref_scales))]
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
            self.assertTrue(torch.allclose(qparams[0], torch.tensor(ref_scales, dtype=qparams[0].dtype), atol=0.0001))
            if qscheme == torch.per_channel_affine_float_qparams:
                self.assertTrue(torch.allclose(qparams[1], torch.tensor(ref_zero_points, dtype=qparams[1].dtype), atol=1))
            else:
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

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @override_qengines
    def test_state_dict_respects_device_affinity(self):
        """
        Tests that loading from a state dict loads buffers to the correct
        device.
        """
        device_cpu = torch.device('cpu')
        device_cuda = torch.device('cuda:0')
        test_cases = itertools.product(
            [device_cpu, device_cuda],
            [device_cpu, device_cuda],
            [MinMaxObserver, MovingAverageMinMaxObserver,
             PerChannelMinMaxObserver,
             MovingAveragePerChannelMinMaxObserver,
             # TODO: enable this (separate PR)
             # HistogramObserver,
             PlaceholderObserver, RecordingObserver, NoopObserver,
             FakeQuantize])

        for device_source, device_target, obs_cls in test_cases:
            # calibrated source model
            model = obs_cls()
            model.to(device_source)
            model(torch.randn(4, 1, 4, 4, device=device_source))
            # target model
            model2 = obs_cls()
            model2.to(device_target)
            model2.load_state_dict(model.state_dict())
            # verify that buffers stayed on model2's device
            model_devices = {p.device for p in model2.parameters()} | \
                {p.device for p in model2.buffers()}
            # some observers do not have any buffers, so lessEqual instead of
            # Equal
            self.assertLessEqual(len(model_devices), 1)
            if len(model_devices) == 1:
                model_device = next(iter(model_devices))
                self.assertEqual(model_device, device_target)

    def test_histogram_observer_consistent_buffer_shape(self):
        """
        Ensures that the buffer shapes do not change from uninitialized to
        initialized states for HistogramObserver.
        """
        obs = HistogramObserver()
        min_shape_before = obs.min_val.shape
        max_shape_before = obs.max_val.shape
        for _ in range(2):
            obs(torch.randn(4, 4, 4, 4))
        self.assertEqual(min_shape_before, obs.min_val.shape)
        self.assertEqual(max_shape_before, obs.max_val.shape)

    def test_histogram_observer_save_load_state_dict(self):
        """
        Smoke test on saving/loading state_dict
        """
        obs1 = HistogramObserver()
        obs1(torch.randn(4, 4, 4, 4))
        obs2 = HistogramObserver()
        obs2.load_state_dict(obs1.state_dict())
        self.assertEqual(obs2.min_val.shape, torch.Size([]))
        self.assertEqual(obs2.max_val.shape, torch.Size([]))


    def test_save_load_state_dict_script(self):
        """
        Tests that we can save and load state_dict for observers that are scripted
        in a quantized model.
        """
        obs_list = [MinMaxObserver, MovingAverageMinMaxObserver,
                    PerChannelMinMaxObserver,
                    MovingAveragePerChannelMinMaxObserver, HistogramObserver]

        for obs in obs_list:
            model = SingleLayerLinearModel().eval()
            qconfig = QConfig(activation=default_observer, weight=obs)
            qconfig_dict = {'' : qconfig}
            scripted = torch.jit.script(model)
            scripted = torch.quantization.prepare_jit(scripted, qconfig_dict)
            x = torch.rand(5, 5)
            scripted(x)
            obs_dict = torch.quantization.get_observer_state_dict(scripted)

            # Load stats
            scripted_2 = torch.jit.script(model)
            scripted_2 = torch.quantization.prepare_jit(scripted_2, qconfig_dict)
            torch.quantization.load_observer_state_dict(scripted_2, obs_dict)
            # Verify that state_dict matches exactly with original one.
            self.assertEqual(scripted.state_dict(), scripted_2.state_dict())


    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_observer_qparams_respects_device_affinity(self):
        """
        Ensure that the scale and zero_point returned by the observer
        are on the same device as the input tensor.
        """
        observerList = [MinMaxObserver(),
                        MovingAverageMinMaxObserver(),
                        PerChannelMinMaxObserver(),
                        MovingAveragePerChannelMinMaxObserver()]
        for obs in observerList:
            device = torch.device('cuda:1')
            x = torch.randn(1, 2, device=device)
            obs.to(device)
            result = obs(x)
            scale, zero_point = obs.calculate_qparams()

            self.assertEqual(x.device, scale.device)
            self.assertEqual(x.device, zero_point.device)

    def test_zero_numel(self):
        obs_list = [MinMaxObserver, MovingAverageMinMaxObserver,
                    PerChannelMinMaxObserver,
                    MovingAveragePerChannelMinMaxObserver, HistogramObserver,
                    FakeQuantize, FixedQParamsFakeQuantize]
        for obs_cls in obs_list:
            if obs_cls is FixedQParamsFakeQuantize:
                obs = obs_cls(0.1, 0)
            else:
                obs = obs_cls()
            x = torch.tensor([])
            # verify no crash
            x = obs(x)


# HistogramObserver that works like it does on master
class _ReferenceHistogramObserver(HistogramObserver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.jit.ignore
    def _non_linear_param_search(self):
        r"""Non-linear parameter search.

        An approximation for L2 error minimization for selecting min/max.
        By selecting new min/max, we filter out outliers in input distribution.
        This follows the implementation of NormMinimization::NonlinearQuantizationParamsSearch in
        caffe2/quantization/server/norm_minimization.cc
        """
        def _get_norm(delta_begin, delta_end, density, norm_type):
            r"""
            Compute the norm of the values uniformaly distributed between
            delta_begin and delta_end.

            norm = density * (integral_{begin, end} x^2)
                 = density * (end^3 - begin^3) / 3
            """
            assert norm_type == "L2", "Only L2 norms are currently supported"
            norm = 0.0
            if norm_type == "L2":
                norm = (
                    delta_end * delta_end * delta_end
                    - delta_begin * delta_begin * delta_begin
                ) / 3
            return density * norm

        def _compute_quantization_error(next_start_bin, next_end_bin, norm_type):
            r"""
            Compute the quantization error if we use start_bin to end_bin as the
            min and max to do the quantization.
            """
            bin_width = (self.max_val.item() - self.min_val.item()) / self.bins

            norm = 0.0
            dst_bin_width = bin_width * (next_end_bin - next_start_bin + 1) / self.dst_nbins
            if dst_bin_width == 0.0:
                return 0.0
            for src_bin in range(self.bins):
                # distances from the beginning of first dst_bin to the beginning and
                # end of src_bin
                src_bin_begin = (src_bin - next_start_bin) * bin_width
                src_bin_end = src_bin_begin + bin_width

                # which dst_bins the beginning and end of src_bin belong to?
                dst_bin_of_begin = min(
                    self.dst_nbins - 1, max(0.0, math.floor(src_bin_begin / dst_bin_width))
                )
                dst_bin_of_end = min(
                    self.dst_nbins - 1, max(0.0, math.floor(src_bin_end / dst_bin_width))
                )
                dst_bin_of_begin_center = (
                    dst_bin_of_begin * dst_bin_width + dst_bin_width / 2
                )

                density = self.histogram[src_bin] / bin_width
                if dst_bin_of_begin == dst_bin_of_end:
                    # if src_bin is entirely within 1 dst_bin
                    delta_begin = src_bin_begin - dst_bin_of_begin_center
                    delta_end = src_bin_end - dst_bin_of_begin_center
                    norm = norm + _get_norm(delta_begin, delta_end, density, norm_type)
                else:
                    delta_begin = src_bin_begin - dst_bin_of_begin_center
                    delta_end = dst_bin_width / 2
                    norm = norm + _get_norm(delta_begin, delta_end, density, norm_type)

                    norm = norm + (dst_bin_of_end - dst_bin_of_begin - 1) * _get_norm(
                        -dst_bin_width / 2, dst_bin_width / 2, density, norm_type
                    )

                    dst_bin_of_end_center = (
                        dst_bin_of_end * dst_bin_width + dst_bin_width / 2
                    )

                    delta_begin = -dst_bin_width / 2
                    delta_end = src_bin_end - dst_bin_of_end_center
                    norm = norm + _get_norm(delta_begin, delta_end, density, norm_type)
            return norm

        assert self.histogram.size()[0] == self.bins, "bins mistmatch"
        bin_width = (self.max_val - self.min_val) / self.bins

        # cumulative sum
        total = sum(self.histogram)
        cSum = torch.cumsum(self.histogram, dim=0)

        stepsize = 1e-5  # granularity
        alpha = 0.0  # lower bound
        beta = 1.0  # upper bound
        start_bin = 0
        end_bin = self.bins - 1
        norm_min = float("inf")

        while alpha < beta:
            # Find the next step
            next_alpha = alpha + stepsize
            next_beta = beta - stepsize

            # find the left and right bins between the quantile bounds
            l = start_bin
            r = end_bin
            while l < end_bin and cSum[l] < next_alpha * total:
                l = l + 1
            while r > start_bin and cSum[r] > next_beta * total:
                r = r - 1

            # decide the next move
            next_start_bin = start_bin
            next_end_bin = end_bin
            if (l - start_bin) > (end_bin - r):
                # move the start bin
                next_start_bin = l
                alpha = next_alpha
            else:
                # move the end bin
                next_end_bin = r
                beta = next_beta

            if next_start_bin == start_bin and next_end_bin == end_bin:
                continue

            # calculate the quantization error using next_start_bin and next_end_bin
            norm = _compute_quantization_error(next_start_bin, next_end_bin, "L2")

            if norm > norm_min:
                break
            norm_min = norm
            start_bin = next_start_bin
            end_bin = next_end_bin

        new_min = self.min_val + bin_width * start_bin
        new_max = self.min_val + bin_width * (end_bin + 1)
        return new_min, new_max

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

class TestHistogramObserver(QuantizationTestCase):
    @given(qdtype=st.sampled_from((torch.qint8, torch.quint8)),
           qscheme=st.sampled_from(
               (torch.per_tensor_affine, torch.per_tensor_symmetric))
           )
    def test_observer_scriptable(self, qdtype, qscheme):
        ob_list = [
            HistogramObserver(dtype=qdtype, qscheme=qscheme),
            default_histogram_observer()
        ]
        for obs in ob_list:
            scripted = torch.jit.script(obs)

            x = torch.rand(3, 4)
            obs(x)
            scripted(x)
            self.assertTrue(torch.equal(obs.histogram, scripted.histogram))
            buf = io.BytesIO()
            torch.jit.save(scripted, buf)
            buf.seek(0)
            loaded = torch.jit.load(buf)
            self.assertTrue(torch.equal(obs.histogram, scripted.histogram))

    @given(qdtype=st.sampled_from((torch.qint8, torch.quint8)),
           qscheme=st.sampled_from((torch.per_tensor_affine, torch.per_tensor_symmetric)),
           reduce_range=st.booleans())
    @settings(max_examples=10)
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
        self.assertEqual(qparams[0].item(), ref_scale, atol=1e-5, rtol=0)
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

    def test_histogram_observer_same_inputs(self):
        myobs = HistogramObserver(bins=3, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False)
        w = torch.ones(4, requires_grad=True)
        x = torch.zeros(4, requires_grad=True)
        y = torch.tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
        z = torch.tensor([5.0, 6.0, 7.0, 8.0])
        myobs(w)
        myobs(x)
        myobs(x)
        myobs(y)
        myobs(z)
        qparams = myobs.calculate_qparams()
        self.assertEqual(myobs.min_val, 2.0)
        self.assertEqual(myobs.max_val, 8.0)
        self.assertEqual(myobs.histogram, [2., 3., 3.])

    @given(N=st.sampled_from([10, 1000]),
           bins=st.sampled_from([256, 512, 1024, 2048]),
           dtype=st.sampled_from([torch.qint8, torch.quint8]),
           qscheme=st.sampled_from([torch.per_tensor_affine, torch.per_tensor_symmetric]),
           reduce_range=st.booleans())
    def test_histogram_observer_against_reference(self, N, bins, dtype, qscheme, reduce_range):

        ref_obs = _ReferenceHistogramObserver(bins=bins, dtype=dtype, qscheme=qscheme, reduce_range=reduce_range)
        my_obs = HistogramObserver(bins=bins, dtype=dtype, qscheme=qscheme, reduce_range=reduce_range)

        for _ in range(10):
            X = torch.randn(N)
            my_obs(X)
            ref_obs(X)

        ref_qparams = ref_obs.calculate_qparams()
        my_qparams = my_obs.calculate_qparams()

        self.assertEqual(ref_qparams, my_qparams)


class TestFakeQuantize(TestCase):
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
        dout = torch.rand_like(X, dtype=torch.float).to(device)
        dX = _fake_quantize_per_tensor_affine_grad_reference(
            dout, X, scale, zero_point, quant_min, quant_max)
        Y_prime.backward(dout)
        np.testing.assert_allclose(dX.cpu(), X.grad.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)

    def test_forward_backward_per_tensor_with_amp(self):
        net = nn.Sequential(nn.Conv2d(1, 1, 3))
        net.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        net_prep = torch.quantization.prepare_qat(net)

        with torch.cuda.amp.autocast():
            x = torch.randn(4, 1, 5, 5)
            out = net_prep(x).sum()
            out.backward()
            self.assertTrue(net_prep[0].weight.grad is not None)

    def test_forward_per_tensor_half_precision_numerics(self):
        scale = .1
        zero = 0
        maxi = 255
        mini = 0

        for i in range(20):
            X1 = torch.randn(5, 5).to(torch.float16)
            Y1 = torch.fake_quantize_per_tensor_affine(X1, scale, zero, mini, maxi)
            Y1r = _fake_quantize_per_tensor_affine_reference(X1, scale, zero, mini, maxi)
            self.assertTrue(torch.allclose(Y1, Y1r, rtol=tolerance, atol=tolerance))

        # to force overflow
        X2 = torch.tensor(2**15 + .01).to(torch.float16)
        Y2 = torch.fake_quantize_per_tensor_affine(X2, scale, zero, mini, maxi)
        Y2r = _fake_quantize_per_tensor_affine_reference(X2, scale, zero, mini, maxi)
        self.assertTrue(torch.allclose(Y2, Y2r, rtol=tolerance, atol=tolerance))

        scale = 10

        # to force underflow
        X3 = torch.tensor(2**-24).to(torch.float16)
        Y3 = torch.fake_quantize_per_tensor_affine(X3, scale, zero, mini, maxi)
        Y3r = _fake_quantize_per_tensor_affine_reference(X3, scale, zero, mini, maxi)
        self.assertTrue(torch.allclose(Y3, Y3r, rtol=tolerance, atol=tolerance))

    def _test_forward_per_tensor_cachemask_impl(self, device):
        float_types = (torch.float32, torch.float16, torch.float64)
        torch_types = (torch.qint8, torch.quint8)
        Xs = (torch.randn(4, 8, device=device), torch.randn(4, 16, device=device)[:, ::2])
        for float_type, torch_type, X in itertools.product(float_types, torch_types, Xs):
            # pick the scale + zp so that some values get clipped
            X = X.to(float_type)
            obs = torch.quantization.MinMaxObserver(torch_type)
            obs(X * 0.75)
            scale, zero_point = obs.calculate_qparams()
            scale, zero_point = float(scale), int(zero_point)
            quant_min, quant_max = obs._calculate_qmin_qmax()

            Y_test = torch.fake_quantize_per_tensor_affine(
                X, scale, zero_point, quant_min, quant_max)
            Y_ref = _fake_quantize_per_tensor_affine_reference(
                X.cpu(), scale, zero_point, quant_min, quant_max).to(device)
            self.assertTrue(torch.allclose(Y_test, Y_ref, rtol=tolerance, atol=tolerance))
            self.assertTrue(Y_test.dtype == float_type)

    def test_forward_per_tensor_cachemask_cpu(self):
        device = torch.device('cpu')
        self._test_forward_per_tensor_cachemask_impl(device)

    @unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
    def test_forward_per_tensor_cachemask_cuda(self):
        device = torch.device('cuda')
        self._test_forward_per_tensor_cachemask_impl(device)

    def _test_backward_per_tensor_cachemask_impl(self, device):
        float_types = (torch.float32, torch.float16, torch.float64)
        torch_types = (torch.qint8, torch.quint8)
        for float_type, torch_type in itertools.product(float_types, torch_types):
            X = torch.randn(4, 8).to(device).to(float_type)
            X.requires_grad_()
            # pick the scale + zp so that some values get clipped
            obs = torch.quantization.MinMaxObserver(torch_type)
            obs(X * 0.75)
            scale, zero_point = obs.calculate_qparams()
            scale, zero_point = float(scale), int(zero_point)
            quant_min, quant_max = obs._calculate_qmin_qmax()

            # forward pass
            Y_test = torch.fake_quantize_per_tensor_affine(
                X, scale, zero_point, quant_min, quant_max)
            Y_ref = _fake_quantize_per_tensor_affine_reference(
                X.cpu(), scale, zero_point, quant_min, quant_max).to(device)
            self.assertTrue(torch.allclose(Y_test, Y_ref, rtol=tolerance, atol=tolerance))

            # backward pass
            dout = torch.rand_like(X, dtype=torch.float).to(device)
            dX = _fake_quantize_per_tensor_affine_grad_reference(
                dout, X, scale, zero_point, quant_min, quant_max)
            Y_test.backward(dout)
            self.assertTrue(torch.allclose(dX, X.grad))
            self.assertTrue(X.grad.dtype == float_type)

    def test_backward_per_tensor_cachemask_cpu(self):
        device = torch.device('cpu')
        self._test_backward_per_tensor_cachemask_impl(device)

    @unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
    def test_backward_per_tensor_cachemask_cuda(self):
        device = torch.device('cuda')
        self._test_backward_per_tensor_cachemask_impl(device)

    def _test_learnable_forward_per_tensor(self, X, device, scale_base, zero_point_base):
        X_base = torch.tensor(X).to(device)

        for n_bits in (4, 8):
            quant_min, quant_max = 0, 2 ** n_bits - 1

            X = X_base.clone().float()
            scale_base = scale_base.to(device).float()
            zero_point_base = zero_point_base.to(dtype=torch.int64, device=device)
            scale = scale_base.clone()
            zero_point = zero_point_base.clamp(quant_min, quant_max)

            Y = _fake_quantize_per_tensor_affine_reference(
                X, scale, zero_point, quant_min, quant_max).to(device)
            for grad_factor in [0.1, 1.0, 10.0]:
                Y_prime = torch._fake_quantize_learnable_per_tensor_affine(
                    X, scale, zero_point, quant_min, quant_max, grad_factor).to(device)
                self.assertTrue(
                    torch.allclose(Y, Y_prime, rtol=tolerance, atol=tolerance),
                    "Expected kernel forward function to have results match the reference forward function")

    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       elements=hu.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams(dtypes=torch.quint8)))
    def test_learnable_forward_per_tensor_cpu(self, X):
        X, (_, _, _) = X
        scale_base = torch.normal(mean=0, std=1, size=(1,)).clamp(1e-4, 100)
        zero_point_base = torch.normal(mean=0, std=128, size=(1,))
        self._test_learnable_forward_per_tensor(
            X, 'cpu', scale_base, zero_point_base)

    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       elements=hu.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams(dtypes=torch.quint8)))
    @unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
    def test_learnable_forward_per_tensor_cuda(self, X):
        X, (_, _, _) = X
        scale_base = torch.normal(mean=0, std=1, size=(1,)).clamp(1e-4, 100)
        zero_point_base = torch.normal(mean=0, std=128, size=(1,))
        self._test_learnable_forward_per_tensor(
            X, 'cuda', scale_base, zero_point_base)

    def _test_learnable_backward_per_tensor(self, X, device, scale_base, zero_point_base):
        r"""Tests the backward method with additional backprop support for scale and zero point.
        """
        X_base = torch.tensor(X).to(device)

        for n_bits in (4, 8):
            quant_min, quant_max = 0, 2 ** n_bits - 1

            X = X_base.clone().float().to(device)
            X.requires_grad_()
            scale_base = scale_base.to(device)
            zero_point_base = zero_point_base.to(device)
            scale = scale_base.clone()
            scale.requires_grad_()
            zero_point = zero_point_base.clone().clamp(quant_min, quant_max)
            zero_point.requires_grad_()
            for grad_factor in [0.1, 1.0, 10.0]:
                Y_prime = torch._fake_quantize_learnable_per_tensor_affine(
                    X, scale, zero_point, quant_min, quant_max, grad_factor).to(device)
                dout = torch.rand_like(X, dtype=torch.float).to(device)
                dX, dScale, dZeroPoint = _fake_quantize_learnable_per_tensor_affine_grad_reference(
                    dout, X, scale, zero_point, quant_min, quant_max, device)
                Y_prime.backward(dout)

                expected_dX = dX.to(device).detach()
                actual_dX = X.grad.to(device).detach()
                expected_dScale = dScale.to(device).detach()
                actual_dScale = scale.grad.to(device).detach()
                expected_dZeroPoint = dZeroPoint.to(device).detach()
                actual_dZeroPoint = zero_point.grad.to(device).detach()

                self.assertTrue(
                    torch.allclose(
                        expected_dX, actual_dX, rtol=tolerance, atol=tolerance),
                    "Expected dX to match X.grad")
                self.assertTrue(
                    torch.allclose(
                        expected_dScale * grad_factor, actual_dScale, rtol=tolerance, atol=tolerance),
                    "Expected dScale to match scale.grad")
                self.assertTrue(
                    torch.allclose(
                        expected_dZeroPoint * grad_factor, actual_dZeroPoint, rtol=tolerance, atol=tolerance),
                    "Expected dZeroPoint to match zero_point.grad")
                X.grad.data.zero_()
                scale.grad.data.zero_()
                zero_point.grad.data.zero_()

    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       elements=hu.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams(dtypes=torch.quint8)))
    def test_learnable_backward_per_tensor_cpu(self, X):
        torch.random.manual_seed(NP_RANDOM_SEED)
        X, (_, _, _) = X
        scale_base = torch.normal(mean=0, std=1, size=(1,)).clamp(1e-4, 100)
        zero_point_base = torch.normal(mean=0, std=128, size=(1,))
        self._test_learnable_backward_per_tensor(
            X, 'cpu', scale_base, zero_point_base)

    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       elements=hu.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams(dtypes=torch.quint8)))
    @unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
    def test_learnable_backward_per_tensor_cuda(self, X):
        torch.random.manual_seed(NP_RANDOM_SEED)
        X, (_, _, _) = X
        scale_base = torch.normal(mean=0, std=1, size=(1,)).clamp(1e-4, 100)
        zero_point_base = torch.normal(mean=0, std=128, size=(1,))
        self._test_learnable_backward_per_tensor(
            X, 'cuda', scale_base, zero_point_base)

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
    def test_fq_module_per_tensor(self, device, X):
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
        dout = torch.rand_like(X, dtype=torch.float, device=device)
        Y_prime.backward(dout)
        dX = _fake_quantize_per_tensor_affine_grad_reference(dout, X, fq_module.scale, fq_module.zero_point, quant_min, quant_max)
        np.testing.assert_allclose(dX.cpu().numpy(), X.grad.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)

    @given(device=st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']),
           X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       qparams=hu.qparams(dtypes=torch.quint8)))
    def test_fixed_qparams_fq_module(self, device, X):
        X, (scale, zero_point, torch_type) = X
        X = to_tensor(X, device)
        fq_module = default_affine_fixed_qparams_fake_quant()
        fixed_scale = fq_module.scale.clone()
        fixed_zero_point = fq_module.zero_point.clone()
        # run fq module and make sure the quantization parameters does not change
        torch.quantization.enable_observer(fq_module)
        fq_module(X)
        self.assertEqual(fixed_scale, fq_module.scale)
        self.assertEqual(fixed_zero_point, fq_module.zero_point)

    def test_fq_serializable_per_tensor(self):
        observer = default_observer
        quant_min = 0
        quant_max = 255
        for FakeQuantizeClass in [FakeQuantize, _LearnableFakeQuantize]:
            fq_module = FakeQuantizeClass(observer, quant_min, quant_max)
            X = torch.tensor([-5, -3.5, -2, 0, 3, 5, 7], dtype=torch.float32)
            y_ref = fq_module(X)
            state_dict = fq_module.state_dict()
            self.assertEqual(state_dict['scale'], 0.094488)
            self.assertEqual(state_dict['zero_point'], 53)
            b = io.BytesIO()
            torch.save(state_dict, b)
            b.seek(0)
            loaded_dict = torch.load(b)
            loaded_fq_module = FakeQuantizeClass(observer, quant_min, quant_max)
            loaded_fq_module.load_state_dict(loaded_dict)
            for key in state_dict:
                self.assertEqual(state_dict[key], loaded_fq_module.state_dict()[key])

            self.assertEqual(loaded_fq_module.calculate_qparams(), fq_module.calculate_qparams())

    def test_fake_quant_control(self):
        for fq_module in [torch.quantization.default_fake_quant(),
                          _LearnableFakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0,
                                                           quant_max=255,
                                                           dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                                                           reduce_range=True)()]:
            torch.manual_seed(42)
            X = torch.rand(20, 10, dtype=torch.float32)
            # Output of fake quant is not identical to input
            Y = fq_module(X)
            self.assertNotEqual(Y, X)
            if type(fq_module) == _LearnableFakeQuantize:
                fq_module.toggle_fake_quant(False)
            else:
                torch.quantization.disable_fake_quant(fq_module)
            X = torch.rand(20, 10, dtype=torch.float32)
            Y = fq_module(X)
            # Fake quant is disabled,output is identical to input
            self.assertEqual(Y, X)

            # Explicit copy at this point in time, because FakeQuant keeps internal
            # state in mutable buffers.
            scale = fq_module.scale.clone().detach()
            zero_point = fq_module.zero_point.clone().detach()

            if type(fq_module) == _LearnableFakeQuantize:
                fq_module.toggle_observer_update(False)
                fq_module.toggle_fake_quant(True)
            else:
                torch.quantization.disable_observer(fq_module)
                torch.quantization.enable_fake_quant(fq_module)
            X = 10.0 * torch.rand(20, 10, dtype=torch.float32) - 5.0
            Y = fq_module(X)
            self.assertNotEqual(Y, X)
            # Observer is disabled, scale and zero-point do not change
            self.assertEqual(fq_module.scale, scale)
            self.assertEqual(fq_module.zero_point, zero_point)
            if type(fq_module) == _LearnableFakeQuantize:
                fq_module.toggle_observer_update(True)
            else:
                torch.quantization.enable_observer(fq_module)
            Y = fq_module(X)
            self.assertNotEqual(Y, X)
            # Observer is enabled, scale and zero-point are different
            self.assertNotEqual(fq_module.scale, scale)
            self.assertNotEqual(fq_module.zero_point, zero_point)

    def test_fake_quant_preserves_qparam_shapes_for_activations(self):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, x):
                x = self.linear(x)
                return x

        m = Model()

        m.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(m, inplace=True)

        scale_shape_before = m.linear.activation_post_process.scale.shape
        zero_point_shape_before = m.linear.activation_post_process.zero_point.shape

        x = torch.rand(4, 4, 4, 4)
        m(x)
        scale_shape_after = m.linear.activation_post_process.scale.shape
        zero_point_shape_after = m.linear.activation_post_process.zero_point.shape
        self.assertEqual(
            scale_shape_before, scale_shape_after,
            msg="FakeQuant scale shape must stay consistent")
        self.assertEqual(
            zero_point_shape_before, zero_point_shape_after,
            msg="FakeQuant zero_point shape must stay consistent")

    def fake_quant_scriptable(self):
        observer = default_observer
        quant_min = 0
        quant_max = 255
        for FakeQuantizeClass in [FakeQuantize, _LearnableFakeQuantize]:
            fq_module = FakeQuantizeClass(observer, quant_min, quant_max)
            scripted_module = torch.jit.script(fq_module)

            X = torch.tensor([-5, -3.5, -2, 0, 3, 5, 7], dtype=torch.float32)

            fq_module(X)
            scripted_module(X)
            self.assertEqual(fq_module.calculate_qparams(), scripted_module.calculate_qparams())

            buf = io.BytesIO()
            torch.jit.save(scripted_module, buf)
            buf.seek(0)
            loaded_module = torch.jit.load(buf)
            self.assertEqual(fq_module.calculate_qparams(), loaded_module.calculate_qparams())


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

    def _test_forward_per_channel_cachemask_impl(self, device):
        torch_types = (torch.qint8, torch.quint8)
        float_types = (torch.float32, torch.float16, torch.float64)
        for torch_type, float_type in itertools.product(torch_types, float_types):
            X = torch.randn(1, 2, 4, 4, dtype=float_type).to(device)
            # pick the scale + zp so that some values get clipped
            axis = 1
            obs = torch.quantization.PerChannelMinMaxObserver(axis, torch_type).to(device)
            obs(X * 0.75)
            scale, zero_point = obs.calculate_qparams()
            # TODO(future PR): fix the wrong dtype in obs.calculate_qparams and remove the cast
            zero_point = zero_point.to(torch.int64)
            quant_min, quant_max = obs._calculate_qmin_qmax()

            Y = _fake_quantize_per_channel_affine_reference(
                X.cpu(), scale.cpu(), zero_point.cpu(), axis, quant_min, quant_max)
            Y_prime = torch.fake_quantize_per_channel_affine(
                X, scale, zero_point, axis, quant_min, quant_max)
            np.testing.assert_allclose(Y, Y_prime.cpu(), rtol=tolerance, atol=tolerance)
            self.assertTrue(Y.dtype == float_type)

    def test_forward_per_channel_cachemask_cpu(self):
        self._test_forward_per_channel_cachemask_impl('cpu')

    @unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
    def test_forward_per_channel_cachemask_cuda(self):
        self._test_forward_per_channel_cachemask_impl('cuda')

    def test_forward_per_channel_half_precision_numerics(self):
        scale = torch.randn(5).abs()
        zero = torch.randn(5).to(dtype=torch.long)
        axis = 1
        mini = 0
        maxi = 255

        for i in range(20):
            X1 = torch.randn(4, 5).to(torch.float16)
            Y1 = torch.fake_quantize_per_channel_affine(X1, scale, zero, axis, mini, maxi)
            Y1r = _fake_quantize_per_channel_affine_reference(X1, scale, zero, axis, mini, maxi)
            self.assertTrue(torch.allclose(Y1, Y1r, rtol=tolerance, atol=tolerance))

        # to force overflow
        X2 = torch.randn(4, 5).to(torch.float16)
        X2[0, 0] = 2**15 + .01
        Y2 = torch.fake_quantize_per_channel_affine(X2, scale, zero, axis, mini, maxi)
        Y2r = _fake_quantize_per_channel_affine_reference(X2, scale, zero, axis, mini, maxi)
        self.assertTrue(torch.allclose(Y2, Y2r, rtol=tolerance, atol=tolerance))

        scale = torch.zeros(5) + 10

        # to force underflow
        X3 = torch.randn(4, 5).to(torch.float16)
        X3[0, 0] = 2**-24
        Y3 = torch.fake_quantize_per_channel_affine(X3, scale, zero, axis, mini, maxi)
        Y3r = _fake_quantize_per_channel_affine_reference(X3, scale, zero, axis, mini, maxi)
        self.assertTrue(torch.allclose(Y3, Y3r, rtol=tolerance, atol=tolerance))

    def _test_learnable_forward_per_channel(self, X_base, device, scale_base, zero_point_base, axis):
        r"""Tests the forward path of the learnable FakeQuantizePerTensorAffine op.
        """
        for n_bits in (4, 8):
            quant_min, quant_max = 0, 2 ** (n_bits) - 1

            scale_base = scale_base.to(device)
            zero_point_base = zero_point_base.to(device)

            X_curr = X_base.clone()
            scale_curr = scale_base.clone()
            zero_point_curr = zero_point_base.clone()

            Y = _fake_quantize_per_channel_affine_reference(
                X_curr, scale_curr, zero_point_curr.round().clamp(quant_min, quant_max), axis, quant_min, quant_max).to(device)
            for grad_factor in [0.1, 1.0, 10.0]:
                Y_prime = torch._fake_quantize_learnable_per_channel_affine(
                    X_curr, scale_curr, zero_point_curr, axis, quant_min, quant_max, grad_factor).to(device)
                self.assertTrue(
                    torch.allclose(Y, Y_prime, rtol=tolerance, atol=tolerance),
                    "Expected kernel forward function to have results match the reference forward function")

    @given(X=hu.per_channel_tensor(shapes=hu.array_shapes(1, 5,),
                                   qparams=hu.qparams(dtypes=torch.quint8)))
    def test_learnable_forward_per_channel_cpu(self, X):
        torch.random.manual_seed(NP_RANDOM_SEED)
        X, (_, _, axis, _) = X
        X_base = torch.tensor(X).to('cpu')
        channel_size = X_base.size(axis)
        scale_base = torch.normal(mean=0, std=1, size=(channel_size,)).clamp(1e-4, 100)
        zero_point_base = torch.normal(mean=0, std=128, size=(channel_size,))
        self._test_learnable_forward_per_channel(
            X_base, 'cpu', scale_base, zero_point_base, axis)

    @given(X=hu.per_channel_tensor(shapes=hu.array_shapes(1, 5,),
                                   qparams=hu.qparams(dtypes=torch.quint8)))
    @unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
    def test_learnable_forward_per_channel_cuda(self, X):
        torch.random.manual_seed(NP_RANDOM_SEED)
        X, (_, _, axis, _) = X
        X_base = torch.tensor(X).to('cuda')
        channel_size = X_base.size(axis)
        scale_base = torch.normal(mean=0, std=1, size=(channel_size,)).clamp(1e-4, 100)
        zero_point_base = torch.normal(mean=0, std=128, size=(channel_size,))
        self._test_learnable_forward_per_channel(
            X_base, 'cuda', scale_base, zero_point_base, axis)

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
        dout = torch.rand_like(X, dtype=torch.float).to(device)
        dX = _fake_quantize_per_channel_affine_grad_reference(
            dout, X, scale, zero_point, axis, quant_min, quant_max)
        Y_prime.backward(dout)
        np.testing.assert_allclose(dX.cpu().detach().numpy(), X.grad.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)

    def _test_backward_per_channel_cachemask_impl(self, device):
        torch_types = (torch.qint8, torch.quint8)
        float_types = (torch.float32, torch.float16, torch.float64)
        for torch_type, float_type in itertools.product(torch_types, float_types):
            X = torch.randn(1, 2, 4, 4, dtype=float_type).to(device)
            # pick the scale + zp so that some values get clipped
            axis = 1
            obs = torch.quantization.PerChannelMinMaxObserver(axis, torch_type).to(device)
            obs(X * 0.75)
            scale, zero_point = obs.calculate_qparams()
            # TODO(future PR): fix the wrong dtype in obs.calculate_qparams and remove the cast
            zero_point = zero_point.to(torch.int64)
            quant_min, quant_max = obs._calculate_qmin_qmax()
            X.requires_grad_()
            Y_prime = torch.fake_quantize_per_channel_affine(
                X, scale, zero_point, axis, quant_min, quant_max)
            dout = torch.rand_like(X, dtype=float_type).to(device)
            dX = _fake_quantize_per_channel_affine_grad_reference(
                dout, X, scale, zero_point, axis, quant_min, quant_max)
            Y_prime.backward(dout)
            np.testing.assert_allclose(
                dX.cpu().detach().numpy(), X.grad.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)
            assert(X.grad.dtype == float_type)


    def test_backward_per_channel_cachemask_cpu(self):
        self._test_backward_per_channel_cachemask_impl('cpu')

    @unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
    def test_backward_per_channel_cachemask_cuda(self):
        self._test_backward_per_channel_cachemask_impl('cuda')

    def _test_learnable_backward_per_channel(self, X_base, device, scale_base, zero_point_base, axis):
        r"""Tests the backward path of the learnable FakeQuantizePerTensorAffine op.
        """
        for n_bits in (4, 8):
            quant_min, quant_max = 0, 2 ** n_bits - 1

            scale_base = scale_base.to(device)
            zero_point_base = zero_point_base.to(device=device)

            X_curr = X_base.clone()
            X_curr.requires_grad_()
            scale_curr = scale_base.clone()
            scale_curr.requires_grad_()
            zero_point_curr = zero_point_base.clone()
            zero_point_curr.requires_grad_()

            for grad_factor in [0.1, 1.0, 10.0]:
                Y_prime = torch._fake_quantize_learnable_per_channel_affine(
                    X_curr, scale_curr, zero_point_curr, axis, quant_min, quant_max, grad_factor).to(device)

                dout = torch.rand(X_curr.shape, dtype=torch.float).to(device)
                dX, dScale, dZeroPoint = _fake_quantize_learnable_per_channel_affine_grad_reference(
                    dout, X_curr, scale_curr, zero_point_curr, axis, quant_min, quant_max, device)
                Y_prime.backward(dout)

                dX_expected = dX.to(device).detach()
                dX_actual = X_curr.to(device).grad.detach()
                dScale_expected = dScale.to(device).detach()
                dScale_actual = scale_curr.to(device).grad.detach()
                dZeroPoint_expected = dZeroPoint.to(device).detach()
                dZeroPoint_actual = zero_point_curr.to(device).grad.detach()
                tolerance = 1e-4

                self.assertTrue(
                    torch.allclose(dX_expected, dX_actual, rtol=tolerance, atol=tolerance),
                    "Expected dX={} to match X.grad={}, X={}, s={}, z={}, dout={}, n_bits={}".format(
                        dX_expected, dX_actual, X_curr, scale_curr, zero_point_curr, dout, n_bits))
                self.assertTrue(
                    torch.allclose(dScale_expected * grad_factor, dScale_actual, rtol=tolerance, atol=tolerance),
                    "Expected dScale={} to match scale.grad={}, X={}, s={}, z={}, dout={}, n_bits={}".format(
                        dScale_expected * grad_factor, dScale_actual,
                        X_curr, scale_curr, zero_point_curr, dout, n_bits))
                self.assertTrue(
                    torch.allclose(dZeroPoint_expected * grad_factor, dZeroPoint_actual, rtol=tolerance, atol=tolerance),
                    "Expected dZeroPoint={} to match zero_point.grad={}, X={}, s={}, z={}, dout={}, n_bits={}".format(
                        dZeroPoint_expected * grad_factor, dZeroPoint_actual,
                        X_curr, scale_curr, zero_point_curr, dout, n_bits))
                X_curr.grad.data.zero_()
                scale_curr.grad.data.zero_()
                zero_point_curr.grad.data.zero_()

    @given(X=hu.per_channel_tensor(shapes=hu.array_shapes(2, 5,),
                                   qparams=hu.qparams(dtypes=torch.quint8)))
    def test_learnable_backward_per_channel_cpu(self, X):
        torch.random.manual_seed(NP_RANDOM_SEED)
        X, (_, _, axis, _) = X
        X_base = torch.tensor(X).to('cpu')
        channel_size = X_base.size(axis)
        scale_base = torch.normal(mean=0, std=1, size=(channel_size,)).clamp(1e-4, 100)
        zero_point_base = torch.normal(mean=0, std=128, size=(channel_size,))
        self._test_learnable_backward_per_channel(
            X_base, 'cpu', scale_base, zero_point_base, axis)

    @given(X=hu.per_channel_tensor(shapes=hu.array_shapes(2, 5,),
                                   qparams=hu.qparams(dtypes=torch.quint8)))
    @unittest.skipIf(not TEST_CUDA, "No gpu is not available.")
    def test_learnable_backward_per_channel_cuda(self, X):
        torch.random.manual_seed(NP_RANDOM_SEED)
        X, (scale, zero_point, axis, torch_type) = X
        X_base = torch.tensor(X).to('cuda')
        scale_base = to_tensor(scale, 'cuda')
        zero_point_base = to_tensor(zero_point, 'cuda')
        self._test_learnable_backward_per_channel(
            X_base, 'cuda', scale_base, zero_point_base, axis)

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
    def test_fq_module_per_channel(self, device, X):
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
        dout = torch.rand_like(X, dtype=torch.float, device=device)
        Y_prime.backward(dout)
        dX = _fake_quantize_per_channel_affine_grad_reference(dout, X, fq_module.scale,
                                                              fq_module.zero_point, axis, quant_min, quant_max)
        np.testing.assert_allclose(dX.cpu().numpy(), X.grad.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)

    def test_fq_serializable_per_channel(self):
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
            torch.quantization.PerChannelMinMaxObserver.with_args(dtype=torch.qint8),
            torch.quantization.MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8),
            torch.quantization.HistogramObserver.with_args(dtype=torch.qint8),
            torch.quantization.RecordingObserver.with_args(dtype=torch.qint8),
            torch.quantization.PlaceholderObserver.with_args(dtype=torch.float16),
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
                msg="{}: Buffers must be modified in place".format(str(observer)))

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
        model.apply(torch.quantization.enable_fake_quant)
        model.apply(torch.quantization.disable_fake_quant)
        model.apply(torch.quantization.enable_observer)
        model.apply(torch.quantization.disable_observer)
        buffer_ids_after = _get_buffer_ids(model)
        self.assertEqual(
            buffer_ids_before,
            buffer_ids_after,
            msg="FakeQuant: Buffers must be modified in place")

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

    def test_syncbn_preserves_qconfig(self):
        """
        Makes sure that if a BatchNorm is not fused and a qconfig exists,
        convering the module to SyncBatchNorm preserves the qconfig.
        """
        m = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
        )
        m[1].qconfig = torch.quantization.default_qconfig
        m = torch.nn.SyncBatchNorm.convert_sync_batchnorm(m)
        self.assertTrue(
            hasattr(m[1], "qconfig"),
            "missing qconfig after SyncBatchNorm conversion")

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @override_qengines
    def test_device_affinity(self):
        """
        Tests that converting a model to QAT respects device affinity
        """
        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.conv = nn.Conv2d(1, 1, 1)
                self.bn = nn.BatchNorm2d(1)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                return x

        model = Model()
        model.qconfig = torch.quantization.get_default_qat_qconfig(torch.backends.quantized.engine)
        device = torch.device('cuda:0')
        model.to(device)
        torch.quantization.prepare_qat(model, inplace=True)
        model_devices = {p.device for p in model.parameters()} | \
            {p.device for p in model.buffers()}
        self.assertEqual(len(model_devices), 1)
        model_device = next(iter(model_devices))
        self.assertEqual(model_device, device)

        # ensure that running an input on CUDA works without any needed changes
        input = torch.randn(4, 1, 4, 4, device=device)
        model(input)
