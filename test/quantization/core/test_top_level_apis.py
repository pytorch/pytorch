# Owner(s): ["oncall: quantization"]

import torch
import torch.ao.quantization
from torch.testing._internal.common_utils import TestCase


class TestDefaultObservers(TestCase):
    observers = [
        "default_affine_fixed_qparams_observer",
        "default_debug_observer",
        "default_dynamic_quant_observer",
        "default_placeholder_observer",
        "default_fixed_qparams_range_0to1_observer",
        "default_fixed_qparams_range_neg1to1_observer",
        "default_float_qparams_observer",
        "default_float_qparams_observer_4bit",
        "default_histogram_observer",
        "default_observer",
        "default_per_channel_weight_observer",
        "default_reuse_input_observer",
        "default_symmetric_fixed_qparams_observer",
        "default_weight_observer",
        "per_channel_weight_observer_range_neg_127_to_127",
        "weight_observer_range_neg_127_to_127",
    ]

    fake_quants = [
        "default_affine_fixed_qparams_fake_quant",
        "default_dynamic_fake_quant",
        "default_embedding_fake_quant",
        "default_embedding_fake_quant_4bit",
        "default_fake_quant",
        "default_fixed_qparams_range_0to1_fake_quant",
        "default_fixed_qparams_range_neg1to1_fake_quant",
        "default_fused_act_fake_quant",
        "default_fused_per_channel_wt_fake_quant",
        "default_fused_wt_fake_quant",
        "default_histogram_fake_quant",
        "default_per_channel_weight_fake_quant",
        "default_symmetric_fixed_qparams_fake_quant",
        "default_weight_fake_quant",
        "fused_per_channel_wt_fake_quant_range_neg_127_to_127",
        "fused_wt_fake_quant_range_neg_127_to_127",
    ]

    def _get_observer_ins(self, observer):
        obs_func = getattr(torch.ao.quantization, observer)
        return obs_func()

    def test_observers(self) -> None:
        t = torch.rand(1, 2, 3, 4)
        for observer in self.observers:
            obs = self._get_observer_ins(observer)
            obs.forward(t)

    def test_fake_quants(self) -> None:
        t = torch.rand(1, 2, 3, 4)
        for observer in self.fake_quants:
            obs = self._get_observer_ins(observer)
            obs.forward(t)
