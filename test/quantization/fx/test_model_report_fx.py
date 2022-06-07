# -*- coding: utf-8 -*-
# Owner(s): ["oncall: quantization"]

import torch
import torch.ao.quantization.quantize_fx
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.fx._model_report._detector import _detect_per_channel

from torch.testing._internal.common_quantization import (
    ConvModel,
    QuantizationTestCase,
)


"""
Partition of input domain:

Model contains: conv or linear, both conv and linear
    Model contains: ConvTransposeNd (not supported for per_channel)

Model is: post training quantization model, quantization aware training model
Model is: composed with nn.Sequential, composed in class structure

Backend is one of three default backends (fbgemm, qnnpack, onednn), backend is different

QConfig utilizes per_channel weight observer, backend uses non per_channel weight observer
QConfig_dict uses only one default qconfig, Qconfig dict uses > 1 unique qconfigs

Partition on output domain:

There are possible changes / suggestions, there are no changes / suggestions

#TODO Add test to ensure that qconfig file name is in string
#TODO Test for xnnpack
"""

# Default output for string if no optimizations are possible
DEFAULT_NO_OPTIMS_ANSWER_STRING = "Further Optimizations for qconfig {}: \nNo further per_channel optimizations possible."


class TestModelReportFxDetector(QuantizationTestCase):

    """Prepares and callibrate the model"""

    def prepare_model_and_run_input(self, model, q_config_mapping, input):
        model_prep = torch.ao.quantization.quantize_fx.prepare_fx(
            model, q_config_mapping, input
        )  # prep model
        model_prep(input).sum()  # callibrate the model
        return model_prep

    """Case includes:
    one conv or linear
    post training quantiztion
    composed as module
    backend is one of defaults
    qconfig uses per_channel weight observer
    Only 1 qconfig in qconfig dict
    Output has no changes / suggestions
    """

    def test_simple_conv(self):
        torch.backends.quantized.engine = "onednn"

        q_config_mapping = QConfigMapping()
        q_config_mapping.set_global(
            torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine)
        )

        input = torch.randn(1, 3, 10, 10)
        prepared_model = self.prepare_model_and_run_input(
            ConvModel(), q_config_mapping, input
        )

        # run the detector
        optims_str, per_channel_info = _detect_per_channel(prepared_model)

        # no optims possible and there should be nothing in per_channel_status
        self.assertEqual(
            optims_str,
            DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine),
        )
        self.assertEqual(per_channel_info["qconfig"], torch.backends.quantized.engine)
        self.assertEqual(len(per_channel_info["per_channel_status"]), 1)
        self.assertEqual(list(per_channel_info["per_channel_status"])[0], ".conv")
        self.assertEqual(
            per_channel_info["per_channel_status"][".conv"]["per_channel_supported"],
            True,
        )
        self.assertEqual(
            per_channel_info["per_channel_status"][".conv"]["per_channel_used"], True
        )
