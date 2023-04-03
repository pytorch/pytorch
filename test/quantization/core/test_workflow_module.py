# Owner(s): ["oncall: quantization"]

# Torch
import torch
from torch.ao.quantization import (
    MinMaxObserver,
    PerChannelMinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    HistogramObserver,
    RecordingObserver,
    PlaceholderObserver,
    NoopObserver,
    FakeQuantize,
    FixedQParamsObserver,
    default_debug_qconfig,
    default_observer,
    default_histogram_observer,
    default_per_channel_weight_observer,
    prepare,
    prepare_qat,
    convert,
    QConfig,
    FusedMovingAvgObsFakeQuantize,
    get_embedding_qat_module_mappings,
    get_embedding_static_quant_module_mappings,
)
from torch.ao.quantization.quantize import _get_observer_dict

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
    _fake_quantize_per_channel_affine_reference,
    _fake_quantize_per_channel_affine_grad_reference,
    to_tensor,
)

from torch.testing._internal.common_quantization import (
    DeFusedEmbeddingBagLinear,
)

NP_RANDOM_SEED = 19
tolerance = 1e-6

class TestObserver(QuantizationTestCase):
    @given(qdtype=st.sampled_from((torch.qint8, torch.quint8, torch.qint32)),
           qscheme=st.sampled_from((torch.per_tensor_affine, torch.per_tensor_symmetric)),
           reduce_range=st.booleans())
    def test_per_tensor_observers(self, qdtype, qscheme, reduce_range):
        # reduce_range cannot be true for symmetric quantization with uint8
        if (qdtype == torch.quint8 and qscheme == torch.per_tensor_symmetric) or qdtype == torch.qint32:
            reduce_range = False
        ObserverList = [MinMaxObserver(dtype=qdtype, qscheme=qscheme, reduce_range=reduce_range),
                        MovingAverageMinMaxObserver(averaging_constant=0.5,
                                                    dtype=qdtype,
                                                    qscheme=qscheme,
                                                    reduce_range=reduce_range)]

        def _get_ref_params(reduce_range, qscheme, dtype, input_scale, min_val, max_val):
            eps = torch.tensor([tolerance])
            if dtype == torch.qint8:
                if reduce_range:
                    quant_min, quant_max = -64, 63
                else:
                    quant_min, quant_max = -128, 127
            elif dtype == torch.quint8:
                if reduce_range:
                    quant_min, quant_max = 0, 127
                else:
                    quant_min, quant_max = 0, 255
            elif dtype == torch.qint32:
                quant_min, quant_max = -1 * (2 ** 31), (2 ** 31) - 1

            min_val_neg = torch.tensor([0.])
            max_val_pos = torch.tensor([input_scale * max_val]) if qdtype is torch.qint32 else torch.tensor([max_val])

            scale, zero_point = 1.0, 0
            if qscheme == torch.per_tensor_symmetric or qscheme == torch.per_channel_symmetric:
                scale = torch.max(-min_val_neg, max_val_pos) / (float(quant_max - quant_min) / 2)
                scale = torch.max(scale, eps)
                if dtype == torch.quint8:
                    zero_point = 128
            else:
                scale = torch.max((max_val_pos - min_val_neg) / float(quant_max - quant_min), eps)
                zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
                zero_point = torch.clamp(zero_point, quant_min, quant_max)
            return scale, zero_point

        for myobs in ObserverList:
            # Calculate Qparams should return with a warning for observers with no data
            qparams = myobs.calculate_qparams()
            input_scale = 2**16 if qdtype is torch.qint32 else 1
            if type(myobs) == MinMaxObserver:
                x = torch.tensor([1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0]) * input_scale
                y = torch.tensor([4.0, 5.0, 5.0, 6.0, 7.0, 8.0]) * input_scale
            else:
                # Moving average of min/max for x and y matches that of
                # extreme values for x/y used for minmax observer
                x = torch.tensor([0.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0]) * input_scale
                y = torch.tensor([2.0, 5.0, 5.0, 6.0, 7.0, 10.0]) * input_scale

            result = myobs(x)
            result = myobs(y)
            self.assertEqual(result, y)
            self.assertEqual(myobs.min_val, 1.0 * input_scale)
            self.assertEqual(myobs.max_val, 8.0 * input_scale)
            qparams = myobs.calculate_qparams()
            ref_scale, ref_zero_point = _get_ref_params(reduce_range, qscheme, qdtype, input_scale, 1.0, 8.0)

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

            self.assertEqual(myobs.min_val, ref_min_vals[ch_axis])
            self.assertEqual(myobs.max_val, ref_max_vals[ch_axis])
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
            self.assertEqual(qparams[0], torch.tensor(ref_scales, dtype=qparams[0].dtype), rtol=1e-5, atol=0.0001)
            if qscheme == torch.per_channel_affine_float_qparams:
                self.assertEqual(qparams[1], torch.tensor(ref_zero_points, dtype=qparams[1].dtype), rtol=1e-5, atol=1)
            else:
                self.assertEqual(qparams[1], torch.tensor(ref_zero_points, dtype=qparams[1].dtype))


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
            self.assertEqual(myobs.min_val, loaded_obs.min_val)
            self.assertEqual(myobs.max_val, loaded_obs.max_val)
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
        obs_list = [MinMaxObserver, MovingAverageMinMaxObserver, HistogramObserver]

        for obs in obs_list:
            model = SingleLayerLinearModel().eval()
            qconfig = QConfig(activation=default_observer, weight=obs)
            qconfig_dict = {'' : qconfig}
            scripted = torch.jit.script(model)
            scripted = torch.ao.quantization.prepare_jit(scripted, qconfig_dict)
            x = torch.rand(5, 5)
            scripted(x)
            obs_dict = torch.ao.quantization.get_observer_state_dict(scripted)

            # Load stats
            scripted_2 = torch.jit.script(model)
            scripted_2 = torch.ao.quantization.prepare_jit(scripted_2, qconfig_dict)
            torch.ao.quantization.load_observer_state_dict(scripted_2, obs_dict)
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
                    FakeQuantize, FixedQParamsObserver]
        for obs_cls in obs_list:
            if obs_cls is FixedQParamsObserver:
                obs = obs_cls(0.1, 0)
            else:
                obs = obs_cls()
            x = torch.tensor([])
            # verify no crash
            x = obs(x)

    def _test_memoryless(self, obs_class):
        obs = obs_class(averaging_constant=1)
        x = torch.randn((3, 3))
        obs(x)
        params = obs.calculate_qparams()
        for _ in range(20):
            obs(10 * torch.randn((3, 3)))
            self.assertNotEqual(params, obs.calculate_qparams())
            obs(x)
            self.assertEqual(params, obs.calculate_qparams())

    def test_memoryless_minmaxobserver(self):
        self._test_memoryless(MovingAverageMinMaxObserver)

    def test_memoryless_perchannelminmaxobserver(self):
        self._test_memoryless(MovingAveragePerChannelMinMaxObserver)

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
        total = torch.sum(self.histogram).item()
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
                _get_observer_dict(model, observer_dict)

                self.assertTrue('fc1.module.activation_post_process' in observer_dict.keys(),
                                'observer is not recorded in the dict')
                self.assertEqual(len(observer_dict['fc1.module.activation_post_process'].get_tensor_value()),
                                 2 * len(self.calib_data))
                self.assertEqual(observer_dict['fc1.module.activation_post_process'].get_tensor_value()[0],
                                 model(self.calib_data[0][0]))

    @given(qdtype=st.sampled_from((torch.qint8, torch.quint8)))
    def test_observer_scriptable(self, qdtype):
        obs = RecordingObserver(dtype=qdtype)
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
            self.assertEqual(my_obs.histogram, ref_obs.histogram)
            self.assertEqual(my_obs.min_val, ref_obs.min_val)
            self.assertEqual(my_obs.max_val, ref_obs.max_val)

        ref_qparams = ref_obs.calculate_qparams()
        my_qparams = my_obs.calculate_qparams()

        for i in range(0, bins, 200):
            for j in range(i + 5, bins, 200):
                ref_qe = ref_obs._compute_quantization_error(i, j)
                qe = my_obs._compute_quantization_error(i, j)
                self.assertEqual(ref_qe, qe)

        self.assertEqual(ref_qparams, my_qparams)

    def test_histogram_observer_extreme_inputs(self):
        """
        Ensures that the HistogramObserver is able to work correctly in
        a rare case: extreme samll max values
        """
        obs = HistogramObserver()
        test_input = torch.tensor(
            [0.0, 0.0, 4.58e-41, 4.58e-41]
        )
        # Make sure it runs, two passes are required based on the behavior of forward func
        # The first pass initializes min_val&max_val, and second pass calls _adjust_min_max
        obs(test_input)
        obs(test_input)

    def test_histogram_observer_correct_numel(self):
        for i in range(1, 10):
            obs = HistogramObserver()
            obs(torch.randn(i, i))
            self.assertEqual(obs.histogram.sum().item(), i**2)


class TestFakeQuantize(TestCase):
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

    def test_quant_min_max_override(self):
        observer = default_per_channel_weight_observer
        # test no override
        fq_module = FakeQuantize(observer)
        self.assertEqual(fq_module.activation_post_process.quant_min, -128)
        self.assertEqual(fq_module.activation_post_process.quant_max, 127)
        # test quant_min/quant_max override
        fq_module = FakeQuantize(observer, quant_min=0, quant_max=127)
        self.assertEqual(fq_module.activation_post_process.quant_min, 0)
        self.assertEqual(fq_module.activation_post_process.quant_max, 127)

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
            torch.ao.quantization.MinMaxObserver.with_args(dtype=torch.qint8),
            torch.ao.quantization.MovingAverageMinMaxObserver.with_args(dtype=torch.qint8),
            torch.ao.quantization.PerChannelMinMaxObserver.with_args(dtype=torch.qint8),
            torch.ao.quantization.MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8),
            torch.ao.quantization.HistogramObserver.with_args(dtype=torch.qint8),
            torch.ao.quantization.RecordingObserver.with_args(dtype=torch.qint8),
            torch.ao.quantization.PlaceholderObserver.with_args(dtype=torch.float16),
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
        model = torch.ao.quantization.FakeQuantize()
        buffer_ids_before = _get_buffer_ids(model)
        for _i in range(5):
            inputs = torch.rand((4, 4, 4))
            model(inputs)
        model.apply(torch.ao.quantization.enable_fake_quant)
        model.apply(torch.ao.quantization.disable_fake_quant)
        model.apply(torch.ao.quantization.enable_observer)
        model.apply(torch.ao.quantization.disable_observer)
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
                torch.ao.quantization.QuantStub(),
                nn.Conv2d(3, 1, 1, bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(),
                nn.Conv2d(1, 2, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(2),
                nn.AvgPool2d(14),
                nn.Sigmoid(),
                torch.ao.quantization.DeQuantStub(),
            )

            torch.ao.quantization.fuse_modules_qat(model, [['1', '2', '3'], ['4', '5']], inplace=True)

            model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
            torch.ao.quantization.prepare_qat(model, inplace=True)
            model = nn.DataParallel(model, device_ids=[0, 1])
            model.to(device)
            model.train()

            for epoch in range(3):
                inputs = torch.rand(2, 3, 28, 28).to(device)
                model(inputs)
                if epoch >= 1:
                    model.apply(torch.ao.quantization.disable_observer)
                if epoch >= 2:
                    model.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
                quant_model = copy.deepcopy(model.module)
                quant_model = torch.ao.quantization.convert(quant_model.eval().cpu(), inplace=False)
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
                    super().__init__()
                    self.conv = nn.Conv2d(4, 1, 3, padding=1)
                    self.bn = nn.BatchNorm2d(1)

                def forward(self, x):
                    x = self.conv(x)
                    x = self.bn(x)
                    return x

            model = Model()
            # fuse it
            fused_model = torch.ao.quantization.fuse_modules_qat(
                model,
                [['conv', 'bn']],
            )
            # convert to QAT
            fused_model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
            torch.ao.quantization.prepare_qat(fused_model, inplace=True)
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
        m[1].qconfig = torch.ao.quantization.default_qconfig
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
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 1)
                self.bn = nn.BatchNorm2d(1)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                return x

        model = Model()
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig(torch.backends.quantized.engine)
        device = torch.device('cuda:0')
        model.to(device)
        torch.ao.quantization.prepare_qat(model, inplace=True)
        model_devices = {p.device for p in model.parameters()} | \
            {p.device for p in model.buffers()}
        self.assertEqual(len(model_devices), 1)
        model_device = next(iter(model_devices))
        self.assertEqual(model_device, device)

        # ensure that running an input on CUDA works without any needed changes
        input = torch.randn(4, 1, 4, 4, device=device)
        model(input)

class TestFusedObsFakeQuantModule(TestCase):
    @given(
        device=st.sampled_from(
            ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        )
    )
    @settings(deadline=None)
    def test_fused_obs_fq_module(self, device):
        # Set up the parameters
        x = torch.randn(5, 5, device=device)
        running_min_op = torch.tensor(float("inf"), device=device)
        running_max_op = torch.tensor(float("-inf"), device=device)
        avg_const = 0.01
        scale = torch.tensor([1.0], device=device)
        zero_point = torch.tensor([0], dtype=torch.int, device=device)

        # Run the forward on the Module
        mod = FusedMovingAvgObsFakeQuantize()
        torch.ao.quantization.enable_fake_quant(mod)
        torch.ao.quantization.enable_observer(mod)
        mod.to(device)
        out = mod(x)

        # Run the operator directly
        pt_op = torch.fused_moving_avg_obs_fake_quant

        out_ref = pt_op(
            x,
            mod.observer_enabled,
            mod.fake_quant_enabled,
            running_min_op,
            running_max_op,
            scale,
            zero_point,
            avg_const,
            0,
            255,
            0,
            False,
        )

        # Compare params with reference
        torch.testing.assert_close(out, out_ref)
        torch.testing.assert_close(
            running_min_op, mod.activation_post_process.min_val
        )
        torch.testing.assert_close(
            running_max_op, mod.activation_post_process.max_val
        )

    @given(
        device=st.sampled_from(
            ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        )
    )
    @settings(deadline=None)
    def test_fused_obs_fq_moving_avg_module(self, device):
        # Set up the parameters
        running_min_op = torch.tensor(float("inf"), device=device)
        running_max_op = torch.tensor(float("-inf"), device=device)
        avg_const = 0.001
        scale = torch.tensor([1.0], device=device)
        zero_point = torch.tensor([0], dtype=torch.int, device=device)

        mod = FusedMovingAvgObsFakeQuantize(averaging_constant=0.001)
        mod.to(device)
        mod.observer_enabled[0] = 0
        mod.fake_quant_enabled[0] = 0

        for i in range(10):
            x = torch.randn(5, 5, device=device)
            if i > 2:
                mod.observer_enabled[0] = 1
            if i > 4:
                mod.fake_quant_enabled[0] = 1
            # Run the forward on the Module
            out = mod(x)

            # Run the operator directly
            pt_op = torch.fused_moving_avg_obs_fake_quant

            out_ref = pt_op(
                x,
                mod.observer_enabled,
                mod.fake_quant_enabled,
                running_min_op,
                running_max_op,
                scale,
                zero_point,
                avg_const,
                0,
                255,
                0,
                False,
            )

            # Compare params with reference
            torch.testing.assert_close(out, out_ref)
            torch.testing.assert_close(
                running_min_op, mod.activation_post_process.min_val
            )
            torch.testing.assert_close(
                running_max_op, mod.activation_post_process.max_val
            )

    @given(
        device=st.sampled_from(
            ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        )
    )
    @settings(deadline=None)
    def test_compare_fused_obs_fq_oss_module(self, device):
        mod = FusedMovingAvgObsFakeQuantize()
        torch.ao.quantization.enable_fake_quant(mod)
        torch.ao.quantization.enable_observer(mod)
        mod.to(device)

        mod_ref = FakeQuantize()
        torch.ao.quantization.enable_fake_quant(mod_ref)
        torch.ao.quantization.enable_observer(mod_ref)
        mod_ref.to(device)

        for i in range(10):
            x = torch.randn(5, 5, device=device)
            out = mod(x)
            out_ref = mod_ref(x)
            torch.testing.assert_close(out, out_ref)
            torch.testing.assert_close(
                mod_ref.activation_post_process.min_val,
                mod.activation_post_process.min_val,
            )
            torch.testing.assert_close(
                mod_ref.activation_post_process.max_val,
                mod.activation_post_process.max_val,
            )

    def test_fused_mod_per_channel(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        m = 5
        n = 10
        for device in devices:
            running_min_op = torch.empty(m, device=device).fill_(float("inf"))
            running_max_op = torch.empty(m, device=device).fill_(float("-inf"))
            avg_const = 0.001
            scale = torch.empty(m, device=device).fill_(0.1)
            zero_point = torch.empty(m, dtype=torch.int, device=device).fill_(0)
            obs = FusedMovingAvgObsFakeQuantize.with_args(
                averaging_constant=avg_const,
                observer=MovingAveragePerChannelMinMaxObserver,
            )
            mod = obs()
            mod = torch.jit.script(mod)
            mod.to(device)

            for i in range(10):
                x = torch.randn(m, n, device=device)
                if i > 2:
                    mod.observer_enabled[0] = 1
                if i > 4:
                    mod.fake_quant_enabled[0] = 1
                # Run the forward on the Module
                out = mod(x)

                # Run the operator directly
                pt_op = torch.fused_moving_avg_obs_fake_quant

                out_ref = pt_op(
                    x,
                    mod.observer_enabled,
                    mod.fake_quant_enabled,
                    running_min_op,
                    running_max_op,
                    scale,
                    zero_point,
                    avg_const,
                    0,
                    255,
                    0,
                    True,
                    False,
                )
                # Compare params with reference
                torch.testing.assert_close(out, out_ref)
                if mod.observer_enabled[0]:
                    torch.testing.assert_close(
                        running_min_op, mod.activation_post_process.min_val
                    )
                    torch.testing.assert_close(
                        running_max_op, mod.activation_post_process.max_val
                    )
                if mod.fake_quant_enabled:
                    torch.testing.assert_close(scale, mod.scale)
                    torch.testing.assert_close(zero_point, mod.zero_point)

            torch.testing.assert_close(mod.state_dict()['activation_post_process.min_val'], running_min_op)
            torch.testing.assert_close(mod.state_dict()['activation_post_process.max_val'], running_max_op)

    def test_fused_mod_reduce_range(self):
        obs = FusedMovingAvgObsFakeQuantize(quant_min=0, quant_max=255, dtype=torch.quint8, reduce_range=True)
        self.assertEqual(obs.activation_post_process.quant_min, 0)
        self.assertEqual(obs.activation_post_process.quant_max, 127)

    def test_embedding_bag_qat_config(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb1 = torch.nn.EmbeddingBag(num_embeddings=10, embedding_dim=12,
                                                  include_last_offset=True, scale_grad_by_freq=False, mode='sum')
                self.emb2 = torch.nn.EmbeddingBag(num_embeddings=10, embedding_dim=12,
                                                  include_last_offset=True, scale_grad_by_freq=False, mode='sum')

            def forward(self, indices):
                return torch.cat((self.emb1(indices), self.emb2(indices)))


        qconfigs = [torch.ao.quantization.default_embedding_qat_qconfig,
                    torch.ao.quantization.default_embedding_qat_qconfig_4bit]
        for qconfig in qconfigs:
            model = Model().train()
            indices = torch.randint(0, 10, (5, 12))

            model.qconfig = qconfig

            quant_model = prepare_qat(model,
                                      mapping=get_embedding_qat_module_mappings())

            count_fake_quant = 0
            for name, mod in quant_model.named_modules():
                if name.endswith('weight_fake_quant'):
                    count_fake_quant += 1
                    self.assertEqual(type(mod), FakeQuantize)
            self.assertEqual(count_fake_quant, 2)

            quant_model(indices)

            # Ensure that EmbeddingBags have float zero_point values
            self.assertEqual(quant_model.emb1.weight_fake_quant.zero_point.dtype, torch.float32)
            self.assertEqual(quant_model.emb2.weight_fake_quant.zero_point.dtype, torch.float32)

            inference_gm = convert(quant_model.eval().cpu(),
                                   mapping=get_embedding_static_quant_module_mappings())

            # Ensure that EmbeddingBags are now quantized with the appropriate bitwidth.
            self.assertEqual(type(inference_gm.emb1), torch.ao.nn.quantized.EmbeddingBag)
            self.assertEqual(type(inference_gm.emb2), torch.ao.nn.quantized.EmbeddingBag)
            self.assertEqual(inference_gm.emb1.dtype, qconfig.weight().dtype)
            self.assertEqual(inference_gm.emb2.dtype, qconfig.weight().dtype)

    def test_embedding_qat_config(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = DeFusedEmbeddingBagLinear()
                indices = torch.randint(0, 10, (5, 12))
                quant_model = prepare_qat(model,
                                          mapping=get_embedding_qat_module_mappings())

                count_fake_quant = 0
                count_activation_postproc = 0
                for name, mod in quant_model.named_modules():
                    if name.endswith('weight_fake_quant'):
                        count_fake_quant += 1
                    if name.count('activation_post_process') == 1 and 'weight_fake_quant' not in name:
                        count_activation_postproc += 1
                # One for embeddings, one for linear layer.
                self.assertEqual(count_fake_quant, 2)
                # One for embeddings (but it is a NoOp), One for quantize, one for linear layer.
                self.assertEqual(count_activation_postproc, 3)

                self.assertEqual(type(quant_model.emb.weight_fake_quant), FakeQuantize)
                self.assertEqual(quant_model.emb.weight_fake_quant.zero_point.dtype, torch.float32)
                self.assertEqual(type(quant_model.emb.activation_post_process), NoopObserver)
                self.assertEqual(type(quant_model.linear.weight_fake_quant), FusedMovingAvgObsFakeQuantize)
                self.assertEqual(type(quant_model.linear.activation_post_process), FusedMovingAvgObsFakeQuantize)

                quant_model(indices)
                inference_gm = convert(quant_model,
                                       mapping=get_embedding_static_quant_module_mappings())
                # Ensure that Embedding is now quantized
                self.assertEqual(type(inference_gm.emb), torch.ao.nn.quantized.Embedding)
                # Ensure that Linear is now quantized
                self.assertEqual(type(inference_gm.linear), torch.ao.nn.quantized.Linear)

    def test_default_fused_qat_config(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.linear(x)
                x = self.relu(x)
                return x

        for qengine in ["fbgemm", "qnnpack"]:
            model = Model()
            model.linear.weight = torch.nn.Parameter(torch.randn(2, 2))
            sample_input = torch.randn(2, 2)
            model.qconfig = torch.ao.quantization.get_default_qat_qconfig(qengine, version=1)
            ref_model = torch.ao.quantization.QuantWrapper(model)
            ref_model = torch.ao.quantization.prepare_qat(ref_model)
            ref_model(sample_input)
            count_fake_quant = 0
            for name, mod in ref_model.named_modules():
                if name.endswith('weight_fake_quant'):
                    count_fake_quant += 1
                    self.assertEqual(type(mod), FusedMovingAvgObsFakeQuantize)

                if name.count('activation_post_process') == 1 and 'weight_fake_quant' not in name:
                    count_fake_quant += 1
                    self.assertEqual(type(mod), FusedMovingAvgObsFakeQuantize)

            self.assertEqual(count_fake_quant, 3)

            if qengine == "fbgemm":
                lower_bnd = 0
                upper_bnd = 127
                obs2match = MovingAveragePerChannelMinMaxObserver

            else:
                lower_bnd = 0
                upper_bnd = 255
                obs2match = MovingAverageMinMaxObserver

            self.assertEqual(ref_model.quant.activation_post_process.activation_post_process.quant_min, lower_bnd)
            self.assertEqual(ref_model.quant.activation_post_process.activation_post_process.quant_max, upper_bnd)
            self.assertEqual(type(ref_model.module.linear.weight_fake_quant.activation_post_process),
                             obs2match)

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_quantization.py TESTNAME\n\n"
                       "instead.")
