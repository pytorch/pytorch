# -*- coding: utf-8 -*-
# Owner(s): ["oncall: quantization"]

import torch
import torch.ao.quantization.quantize_fx
from torch.ao.quantization import QConfig, QConfigMapping
from torch.ao.quantization.fx._model_report._detector import _detect_per_channel
from torch.ao.quantization.observer import (
    default_per_channel_weight_observer,
    HistogramObserver,
)
from torch.nn.intrinsic.modules.fused import ConvReLU2d, LinearReLU
from torch.testing._internal.common_quantization import (
    ConvModel,
    QuantizationTestCase,
    skipIfNoFBGEMM,
    skipIfNoQNNPACK,
    TwoLayerLinearModel,
)

"""
Partition of input domain:

Model contains: conv or linear, both conv and linear
    Model contains: ConvTransposeNd (not supported for per_channel)

Model is: post training quantization model, quantization aware training model
Model is: composed with nn.Sequential, composed in class structure

QConfig utilizes per_channel weight observer, backend uses non per_channel weight observer
QConfig_dict uses only one default qconfig, Qconfig dict uses > 1 unique qconfigs

Partition on output domain:

There are possible changes / suggestions, there are no changes / suggestions
"""

# Default output for string if no optimizations are possible
DEFAULT_NO_OPTIMS_ANSWER_STRING = "Further Optimizations for backend {}: \nNo further per_channel optimizations possible."

# Example Sequential Model with multiple Conv and Linear with nesting involved
NESTED_CONV_LINEAR_EXAMPLE = torch.nn.Sequential(
    torch.nn.Conv2d(3, 3, 2, 1),
    torch.nn.Sequential(torch.nn.Linear(9, 27), torch.nn.ReLU()),
    torch.nn.Linear(27, 27),
    torch.nn.ReLU(),
    torch.nn.Conv2d(3, 3, 2, 1),
)

# Example Sequential Model with Conv sub-class example
LAZY_CONV_LINEAR_EXAMPLE = torch.nn.Sequential(
    torch.nn.LazyConv2d(3, 3, 2, 1),
    torch.nn.Sequential(torch.nn.Linear(5, 27), torch.nn.ReLU()),
    torch.nn.ReLU(),
    torch.nn.Linear(27, 27),
    torch.nn.ReLU(),
    torch.nn.LazyConv2d(3, 3, 2, 1),
)

# Example Sequential Model with Fusion directly built into model
FUSION_CONV_LINEAR_EXAMPLE = torch.nn.Sequential(
    ConvReLU2d(torch.nn.Conv2d(3, 3, 2, 1), torch.nn.ReLU()),
    torch.nn.Sequential(LinearReLU(torch.nn.Linear(9, 27), torch.nn.ReLU())),
    LinearReLU(torch.nn.Linear(27, 27), torch.nn.ReLU()),
    torch.nn.Conv2d(3, 3, 2, 1),
)


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
        self.assertEqual(per_channel_info["backend"], torch.backends.quantized.engine)
        self.assertEqual(len(per_channel_info["per_channel_status"]), 1)
        self.assertEqual(list(per_channel_info["per_channel_status"])[0], ".conv")
        self.assertEqual(
            per_channel_info["per_channel_status"][".conv"]["per_channel_supported"],
            True,
        )
        self.assertEqual(
            per_channel_info["per_channel_status"][".conv"]["per_channel_used"], True
        )

    """Case includes:
        Multiple conv or linear
        post training quantization
        composed as module
        qconfig doesn't use per_channel weight observer
        Only 1 qconfig in qconfig dict
        Output has possible changes / suggestions
    """

    @skipIfNoQNNPACK
    def test_multi_linear_model_without_per_channel(self):
        torch.backends.quantized.engine = "qnnpack"

        q_config_mapping = QConfigMapping()
        q_config_mapping.set_global(
            torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine)
        )

        prepared_model = self.prepare_model_and_run_input(
            TwoLayerLinearModel(),
            q_config_mapping,
            TwoLayerLinearModel().get_example_inputs()[0],
        )

        # run the detector
        optims_str, per_channel_info = _detect_per_channel(prepared_model)

        # there should be optims possible
        self.assertNotEqual(
            optims_str,
            DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine),
        )
        self.assertEqual(per_channel_info["backend"], torch.backends.quantized.engine)
        self.assertEqual(len(per_channel_info["per_channel_status"]), 2)

        # for each linear layer, should be supported but not used
        for linear_key in per_channel_info["per_channel_status"].keys():
            module_entry = per_channel_info["per_channel_status"][linear_key]

            self.assertEqual(module_entry["per_channel_supported"], True)
            self.assertEqual(module_entry["per_channel_used"], False)

    """Case includes:
        Multiple conv or linear
        post training quantization
        composed as Module
        qconfig doesn't use per_channel weight observer
        More than 1 qconfig in qconfig dict
        Output has possible changes / suggestions
    """

    @skipIfNoQNNPACK
    def test_multiple_q_config_options(self):
        torch.backends.quantized.engine = "qnnpack"

        # qconfig with support for per_channel quantization
        per_channel_qconfig = QConfig(
            activation=HistogramObserver.with_args(reduce_range=True),
            weight=default_per_channel_weight_observer,
        )

        # we need to design the model
        class ConvLinearModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 2, 1)
                self.fc1 = torch.nn.Linear(9, 27)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(27, 27)
                self.conv2 = torch.nn.Conv2d(3, 3, 2, 1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                x = self.conv2(x)
                return x

        q_config_mapping = QConfigMapping()
        q_config_mapping.set_global(
            torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine)
        ).set_object_type(torch.nn.Conv2d, per_channel_qconfig)

        prepared_model = self.prepare_model_and_run_input(
            ConvLinearModel(),
            q_config_mapping,
            torch.randn(1, 3, 10, 10),
        )

        # run the detector
        optims_str, per_channel_info = _detect_per_channel(prepared_model)

        # the only suggestions should be to linear layers

        # there should be optims possible
        self.assertNotEqual(
            optims_str,
            DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine),
        )

        # to ensure it got into the nested layer
        self.assertEqual(len(per_channel_info["per_channel_status"]), 4)

        # for each layer, should be supported but not used
        for key in per_channel_info["per_channel_status"].keys():
            module_entry = per_channel_info["per_channel_status"][key]
            self.assertEqual(module_entry["per_channel_supported"], True)

            # if linear False, if conv2d true cuz it uses different config
            if "fc" in key:
                self.assertEqual(module_entry["per_channel_used"], False)
            elif "conv" in key:
                self.assertEqual(module_entry["per_channel_used"], True)
            else:
                raise ValueError(
                    "Should only contain conv and linear layers as key values"
                )

    """Case includes:
        Multiple conv or linear
        post training quantization
        composed as sequential
        qconfig doesn't use per_channel weight observer
        Only 1 qconfig in qconfig dict
        Output has possible changes / suggestions
    """

    @skipIfNoQNNPACK
    def test_sequential_model_format(self):
        torch.backends.quantized.engine = "qnnpack"

        q_config_mapping = QConfigMapping()
        q_config_mapping.set_global(
            torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine)
        )

        prepared_model = self.prepare_model_and_run_input(
            NESTED_CONV_LINEAR_EXAMPLE,
            q_config_mapping,
            torch.randn(1, 3, 10, 10),
        )

        # run the detector
        optims_str, per_channel_info = _detect_per_channel(prepared_model)

        # there should be optims possible
        self.assertNotEqual(
            optims_str,
            DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine),
        )

        # to ensure it got into the nested layer
        self.assertEqual(len(per_channel_info["per_channel_status"]), 4)

        # for each layer, should be supported but not used
        for key in per_channel_info["per_channel_status"].keys():
            module_entry = per_channel_info["per_channel_status"][key]

            self.assertEqual(module_entry["per_channel_supported"], True)
            self.assertEqual(module_entry["per_channel_used"], False)

    """Case includes:
        Multiple conv or linear
        post training quantization
        composed as sequential
        qconfig doesn't use per_channel weight observer
        Only 1 qconfig in qconfig dict
        Output has possible changes / suggestions
    """

    @skipIfNoQNNPACK
    def test_conv_sub_class_considered(self):
        torch.backends.quantized.engine = "qnnpack"

        q_config_mapping = QConfigMapping()
        q_config_mapping.set_global(
            torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine)
        )

        prepared_model = self.prepare_model_and_run_input(
            LAZY_CONV_LINEAR_EXAMPLE,
            q_config_mapping,
            torch.randn(1, 3, 10, 10),
        )

        # run the detector
        optims_str, per_channel_info = _detect_per_channel(prepared_model)

        # there should be optims possible
        self.assertNotEqual(
            optims_str,
            DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine),
        )

        # to ensure it got into the nested layer and it considered the lazyConv2d
        self.assertEqual(len(per_channel_info["per_channel_status"]), 4)

        # for each layer, should be supported but not used
        for key in per_channel_info["per_channel_status"].keys():
            module_entry = per_channel_info["per_channel_status"][key]

            self.assertEqual(module_entry["per_channel_supported"], True)
            self.assertEqual(module_entry["per_channel_used"], False)

    """Case includes:
        Multiple conv or linear
        post training quantization
        composed as sequential
        qconfig uses per_channel weight observer
        Only 1 qconfig in qconfig dict
        Output has no possible changes / suggestions
    """

    @skipIfNoFBGEMM
    def test_fusion_layer_in_sequential(self):
        torch.backends.quantized.engine = "fbgemm"

        q_config_mapping = QConfigMapping()
        q_config_mapping.set_global(
            torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine)
        )

        prepared_model = self.prepare_model_and_run_input(
            FUSION_CONV_LINEAR_EXAMPLE,
            q_config_mapping,
            torch.randn(1, 3, 10, 10),
        )

        # run the detector
        optims_str, per_channel_info = _detect_per_channel(prepared_model)

        # no optims possible and there should be nothing in per_channel_status
        self.assertEqual(
            optims_str,
            DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine),
        )

        # to ensure it got into the nested layer and it considered all the nested fusion components
        self.assertEqual(len(per_channel_info["per_channel_status"]), 4)

        # for each layer, should be supported but not used
        for key in per_channel_info["per_channel_status"].keys():
            module_entry = per_channel_info["per_channel_status"][key]
            self.assertEqual(module_entry["per_channel_supported"], True)
            self.assertEqual(module_entry["per_channel_used"], True)

    """Case includes:
        Multiple conv or linear
        quantitative aware training
        composed as model
        qconfig does not use per_channel weight observer
        Only 1 qconfig in qconfig dict
        Output has possible changes / suggestions
    """

    @skipIfNoQNNPACK
    def test_qat_aware_model_example(self):

        # first we want a QAT model
        class QATConvLinearReluModel(torch.nn.Module):
            def __init__(self):
                super(QATConvLinearReluModel, self).__init__()
                # QuantStub converts tensors from floating point to quantized
                self.quant = torch.quantization.QuantStub()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.bn = torch.nn.BatchNorm2d(1)
                self.relu = torch.nn.ReLU()
                # DeQuantStub converts tensors from quantized to floating point
                self.dequant = torch.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                x = self.dequant(x)
                return x

        # create a model instance
        model_fp32 = QATConvLinearReluModel()

        model_fp32.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")

        # model must be in eval mode for fusion
        model_fp32.eval()
        model_fp32_fused = torch.quantization.fuse_modules(
            model_fp32, [["conv", "bn", "relu"]]
        )

        # model must be set to train mode for QAT logic to work
        model_fp32_fused.train()

        # prepare the model for QAT, different than for post training quantization
        model_fp32_prepared = torch.quantization.prepare_qat(model_fp32_fused)

        # run the detector
        optims_str, per_channel_info = _detect_per_channel(model_fp32_prepared)

        # there should be optims possible
        self.assertNotEqual(
            optims_str,
            DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine),
        )

        # make sure it was able to find the single conv in the fused model
        self.assertEqual(len(per_channel_info["per_channel_status"]), 1)

        # for the one conv, it should still give advice to use different qconfig
        for key in per_channel_info["per_channel_status"].keys():
            module_entry = per_channel_info["per_channel_status"][key]
            self.assertEqual(module_entry["per_channel_supported"], True)
            self.assertEqual(module_entry["per_channel_used"], False)
