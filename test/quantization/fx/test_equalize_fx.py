import torch
from torch.testing._internal.common_quantization import QuantizationTestCase
from torch.quantization.fx._equalize import (
    _InputEqualizationObserver, _WeightEqualizationObserver, calculate_equalization_scale
)

# Standard Libraries
import numpy as np

# Testing utils
from hypothesis import given
from hypothesis import strategies as st


class TestEqualizeFx(QuantizationTestCase):
    @given(input_qdtype=st.sampled_from((torch.qint8, torch.quint8)),
           input_qscheme=st.sampled_from((torch.per_tensor_affine, torch.per_tensor_symmetric)),
           weight_qdtype=st.sampled_from((torch.qint8, torch.quint8)),
           weight_qscheme=st.sampled_from((torch.per_channel_affine, torch.per_channel_symmetric,
                                           torch.per_channel_affine_float_qparams)))
    def test_input_weight_observer(self, input_qdtype, input_qscheme, weight_qdtype, weight_qscheme):
        input_obs = _InputEqualizationObserver(dtype=input_qdtype, qscheme=input_qscheme)
        weight_obs = _WeightEqualizationObserver(dtype=weight_qdtype, qscheme=weight_qscheme)

        width = np.random.randint(1, 10)
        x_height = np.random.randint(2, 10)
        w_height = np.random.randint(2, 10)

        x = (np.random.random(size=(x_height, width)) * 10).round(decimals=2).astype(np.float32)
        w = (np.random.random(size=(w_height, width)) * 10).round(decimals=2).astype(np.float32)

        ret_x = input_obs(torch.tensor(x))
        ret_w = weight_obs(torch.tensor(w))
        self.assertEqual((ret_x, ret_w), (x, w))

        # Check the min/max input columns are correct
        ref_min_inputs = x.min(axis=0)
        ref_max_inputs = x.max(axis=0)
        self.assertEqual(input_obs.get_input_minmax(), (ref_min_inputs, ref_max_inputs))

        # Check the min/max weight columns are correct
        ref_min_weights_col = w.min(axis=0)
        ref_max_weights_col = w.max(axis=0)
        self.assertEqual(weight_obs.get_weight_col_minmax(), (ref_min_weights_col, ref_max_weights_col))

        # Check the min/max weight rows are correct
        ref_min_weights_row = w.min(axis=1)
        ref_max_weights_row = w.max(axis=1)
        self.assertEqual(weight_obs.get_weight_row_minmax(), (ref_min_weights_row, ref_max_weights_row))

        # Check the column indices of the min/max weight rows are correct
        ref_min_weights_ind = w.argmin(axis=1)
        ref_max_weights_ind = w.argmax(axis=1)
        self.assertEqual((weight_obs.min_weights_ind, weight_obs.max_weights_ind),
                         (ref_min_weights_ind, ref_max_weights_ind))

        # Check the equalization scale is correct
        equalization_scale = calculate_equalization_scale(input_obs, weight_obs)
        ref_equalization_scale = np.sqrt((ref_max_weights_col - ref_min_weights_col) /
                                         (ref_max_inputs - ref_min_inputs))
        self.assertEqual(equalization_scale, ref_equalization_scale)

        input_obs.set_equalization_scale(equalization_scale)
        weight_obs.set_equalization_scale(equalization_scale)

        # check the input scale/zero-point values
        input_qparams = input_obs.calculate_qparams()

        min_input_scaled = np.min(ref_min_inputs * ref_equalization_scale)
        min_input_scaled = min(0, min_input_scaled)
        max_input_scaled = np.max(ref_max_inputs * ref_equalization_scale)
        max_input_scaled = max(0, max_input_scaled)

        if input_qscheme == torch.per_tensor_symmetric:
            ref_scale = 2 * max(abs(min_input_scaled), max_input_scaled) / 255
            ref_zero_point = 0 if input_qdtype is torch.qint8 else 128
        else:
            ref_scale = (max_input_scaled - min_input_scaled) / 255
            ref_zero_point = -128 if input_qdtype is torch.qint8 else 0

        self.assertEqual(input_qparams[0].item(), ref_scale, atol=1e-5, rtol=0)
        self.assertEqual(input_qparams[1].item(), ref_zero_point)

        # check the weight scale/zero-point values
        weight_qparams = weight_obs.calculate_qparams()

        min_weights_scaled = ref_min_weights_row * (1 / ref_equalization_scale[ref_min_weights_ind])
        max_weights_scaled = ref_max_weights_row * (1 / ref_equalization_scale[ref_max_weights_ind])

        if weight_qscheme == torch.per_channel_symmetric:
            min_weights_scaled = np.minimum(np.zeros(min_weights_scaled.shape), min_weights_scaled)
            max_weights_scaled = np.maximum(np.zeros(max_weights_scaled.shape), max_weights_scaled)

            ref_scales = 2 * np.maximum(np.abs(min_weights_scaled), max_weights_scaled) / 255
            ref_zero_points = np.zeros_like(
                ref_scales) if weight_qdtype is torch.qint8 else np.ones_like(ref_scales) * 128
        elif weight_qscheme == torch.per_channel_affine_float_qparams:
            ref_scales = (max_weights_scaled - min_weights_scaled) / 255
            ref_scales = np.where(ref_scales > 1e-7, ref_scales, np.ones_like(ref_scales))
            ref_zero_points = -1 * min_weights_scaled / ref_scales
        else:
            min_weights_scaled = np.minimum(np.zeros_like(min_weights_scaled), min_weights_scaled)
            max_weights_scaled = np.maximum(np.zeros_like(max_weights_scaled), max_weights_scaled)

            ref_scales = (max_weights_scaled - min_weights_scaled) / 255
            ref_zero_points = -128 if weight_qdtype is torch.qint8 else 0
            ref_zero_points = ref_zero_points - np.round(min_weights_scaled / ref_scales)

        self.assertTrue(torch.allclose(weight_qparams[0], torch.tensor(
            ref_scales, dtype=weight_qparams[0].dtype), atol=0.0001))
        self.assertTrue(torch.allclose(weight_qparams[1], torch.tensor(
            ref_zero_points, dtype=weight_qparams[1].dtype), atol=1))
