# Owner(s): ["oncall: quantization"]
# ruff: noqa: F841
from typing import Set

import torch
import torch.nn as nn
import torch.ao.quantization.quantize_fx as quantize_fx
import torch.nn.functional as F
from torch.ao.quantization import QConfig, QConfigMapping
from torch.ao.quantization.fx._model_report.detector import (
    DynamicStaticDetector,
    InputWeightEqualizationDetector,
    PerChannelDetector,
    OutlierDetector,
)
from torch.ao.quantization.fx._model_report.model_report_observer import ModelReportObserver
from torch.ao.quantization.fx._model_report.model_report_visualizer import ModelReportVisualizer
from torch.ao.quantization.fx._model_report.model_report import ModelReport
from torch.ao.quantization.observer import (
    HistogramObserver,
    default_per_channel_weight_observer,
    default_observer
)
from torch.ao.nn.intrinsic.modules.fused import ConvReLU2d, LinearReLU
from torch.testing._internal.common_quantization import (
    ConvModel,
    QuantizationTestCase,
    SingleLayerLinearModel,
    TwoLayerLinearModel,
    skipIfNoFBGEMM,
    skipIfNoQNNPACK,
    override_quantized_engine,
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
DEFAULT_NO_OPTIMS_ANSWER_STRING = (
    "Further Optimizations for backend {}: \nNo further per_channel optimizations possible."
)

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

# Test class
# example model to use for tests
class ThreeOps(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def get_example_inputs(self):
        return (torch.randn(1, 3, 3, 3),)

class TwoThreeOps(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = ThreeOps()
        self.block2 = ThreeOps()

    def forward(self, x):
        x = self.block1(x)
        y = self.block2(x)
        z = x + y
        z = F.relu(z)
        return z

    def get_example_inputs(self):
        return (torch.randn(1, 3, 3, 3),)

class TestFxModelReportDetector(QuantizationTestCase):

    """Prepares and calibrate the model"""

    def _prepare_model_and_run_input(self, model, q_config_mapping, input):
        model_prep = torch.ao.quantization.quantize_fx.prepare_fx(model, q_config_mapping, input)  # prep model
        model_prep(input).sum()  # calibrate the model
        return model_prep

    """Case includes:
        one conv or linear
        post training quantization
        composed as module
        qconfig uses per_channel weight observer
        Only 1 qconfig in qconfig dict
        Output has no changes / suggestions
    """

    @skipIfNoFBGEMM
    def test_simple_conv(self):

        with override_quantized_engine('fbgemm'):
            torch.backends.quantized.engine = "fbgemm"

            q_config_mapping = QConfigMapping()
            q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))

            input = torch.randn(1, 3, 10, 10)
            prepared_model = self._prepare_model_and_run_input(ConvModel(), q_config_mapping, input)

            # run the detector
            per_channel_detector = PerChannelDetector(torch.backends.quantized.engine)
            optims_str, per_channel_info = per_channel_detector.generate_detector_report(prepared_model)

            # no optims possible and there should be nothing in per_channel_status
            self.assertEqual(
                optims_str,
                DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine),
            )

            # there should only be one conv there in this model
            self.assertEqual(per_channel_info["conv"]["backend"], torch.backends.quantized.engine)
            self.assertEqual(len(per_channel_info), 1)
            self.assertEqual(next(iter(per_channel_info)), "conv")
            self.assertEqual(
                per_channel_info["conv"]["per_channel_quantization_supported"],
                True,
            )
            self.assertEqual(per_channel_info["conv"]["per_channel_quantization_used"], True)

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

        with override_quantized_engine('qnnpack'):
            torch.backends.quantized.engine = "qnnpack"

            q_config_mapping = QConfigMapping()
            q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))

            prepared_model = self._prepare_model_and_run_input(
                TwoLayerLinearModel(),
                q_config_mapping,
                TwoLayerLinearModel().get_example_inputs()[0],
            )

            # run the detector
            per_channel_detector = PerChannelDetector(torch.backends.quantized.engine)
            optims_str, per_channel_info = per_channel_detector.generate_detector_report(prepared_model)

            # there should be optims possible
            self.assertNotEqual(
                optims_str,
                DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine),
            )
            # pick a random key to look at
            rand_key: str = next(iter(per_channel_info.keys()))
            self.assertEqual(per_channel_info[rand_key]["backend"], torch.backends.quantized.engine)
            self.assertEqual(len(per_channel_info), 2)

            # for each linear layer, should be supported but not used
            for linear_key in per_channel_info.keys():
                module_entry = per_channel_info[linear_key]

                self.assertEqual(module_entry["per_channel_quantization_supported"], True)
                self.assertEqual(module_entry["per_channel_quantization_used"], False)

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

        with override_quantized_engine('qnnpack'):
            torch.backends.quantized.engine = "qnnpack"

            # qconfig with support for per_channel quantization
            per_channel_qconfig = QConfig(
                activation=HistogramObserver.with_args(reduce_range=True),
                weight=default_per_channel_weight_observer,
            )

            # we need to design the model
            class ConvLinearModel(torch.nn.Module):
                def __init__(self) -> None:
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

            prepared_model = self._prepare_model_and_run_input(
                ConvLinearModel(),
                q_config_mapping,
                torch.randn(1, 3, 10, 10),
            )

            # run the detector
            per_channel_detector = PerChannelDetector(torch.backends.quantized.engine)
            optims_str, per_channel_info = per_channel_detector.generate_detector_report(prepared_model)

            # the only suggestions should be to linear layers

            # there should be optims possible
            self.assertNotEqual(
                optims_str,
                DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine),
            )

            # to ensure it got into the nested layer
            self.assertEqual(len(per_channel_info), 4)

            # for each layer, should be supported but not used
            for key in per_channel_info.keys():
                module_entry = per_channel_info[key]
                self.assertEqual(module_entry["per_channel_quantization_supported"], True)

                # if linear False, if conv2d true cuz it uses different config
                if "fc" in key:
                    self.assertEqual(module_entry["per_channel_quantization_used"], False)
                elif "conv" in key:
                    self.assertEqual(module_entry["per_channel_quantization_used"], True)
                else:
                    raise ValueError("Should only contain conv and linear layers as key values")

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

        with override_quantized_engine('qnnpack'):
            torch.backends.quantized.engine = "qnnpack"

            q_config_mapping = QConfigMapping()
            q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))

            prepared_model = self._prepare_model_and_run_input(
                NESTED_CONV_LINEAR_EXAMPLE,
                q_config_mapping,
                torch.randn(1, 3, 10, 10),
            )

            # run the detector
            per_channel_detector = PerChannelDetector(torch.backends.quantized.engine)
            optims_str, per_channel_info = per_channel_detector.generate_detector_report(prepared_model)

            # there should be optims possible
            self.assertNotEqual(
                optims_str,
                DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine),
            )

            # to ensure it got into the nested layer
            self.assertEqual(len(per_channel_info), 4)

            # for each layer, should be supported but not used
            for key in per_channel_info.keys():
                module_entry = per_channel_info[key]

                self.assertEqual(module_entry["per_channel_quantization_supported"], True)
                self.assertEqual(module_entry["per_channel_quantization_used"], False)

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

        with override_quantized_engine('qnnpack'):
            torch.backends.quantized.engine = "qnnpack"

            q_config_mapping = QConfigMapping()
            q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))

            prepared_model = self._prepare_model_and_run_input(
                LAZY_CONV_LINEAR_EXAMPLE,
                q_config_mapping,
                torch.randn(1, 3, 10, 10),
            )

            # run the detector
            per_channel_detector = PerChannelDetector(torch.backends.quantized.engine)
            optims_str, per_channel_info = per_channel_detector.generate_detector_report(prepared_model)

            # there should be optims possible
            self.assertNotEqual(
                optims_str,
                DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine),
            )

            # to ensure it got into the nested layer and it considered the lazyConv2d
            self.assertEqual(len(per_channel_info), 4)

            # for each layer, should be supported but not used
            for key in per_channel_info.keys():
                module_entry = per_channel_info[key]

                self.assertEqual(module_entry["per_channel_quantization_supported"], True)
                self.assertEqual(module_entry["per_channel_quantization_used"], False)

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

        with override_quantized_engine('fbgemm'):
            torch.backends.quantized.engine = "fbgemm"

            q_config_mapping = QConfigMapping()
            q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))

            prepared_model = self._prepare_model_and_run_input(
                FUSION_CONV_LINEAR_EXAMPLE,
                q_config_mapping,
                torch.randn(1, 3, 10, 10),
            )

            # run the detector
            per_channel_detector = PerChannelDetector(torch.backends.quantized.engine)
            optims_str, per_channel_info = per_channel_detector.generate_detector_report(prepared_model)

            # no optims possible and there should be nothing in per_channel_status
            self.assertEqual(
                optims_str,
                DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine),
            )

            # to ensure it got into the nested layer and it considered all the nested fusion components
            self.assertEqual(len(per_channel_info), 4)

            # for each layer, should be supported but not used
            for key in per_channel_info.keys():
                module_entry = per_channel_info[key]
                self.assertEqual(module_entry["per_channel_quantization_supported"], True)
                self.assertEqual(module_entry["per_channel_quantization_used"], True)

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
            def __init__(self) -> None:
                super().__init__()
                # QuantStub converts tensors from floating point to quantized
                self.quant = torch.ao.quantization.QuantStub()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.bn = torch.nn.BatchNorm2d(1)
                self.relu = torch.nn.ReLU()
                # DeQuantStub converts tensors from quantized to floating point
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                x = self.dequant(x)
                return x

        with override_quantized_engine('qnnpack'):
            # create a model instance
            model_fp32 = QATConvLinearReluModel()

            model_fp32.qconfig = torch.ao.quantization.get_default_qat_qconfig("qnnpack")

            # model must be in eval mode for fusion
            model_fp32.eval()
            model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, [["conv", "bn", "relu"]])

            # model must be set to train mode for QAT logic to work
            model_fp32_fused.train()

            # prepare the model for QAT, different than for post training quantization
            model_fp32_prepared = torch.ao.quantization.prepare_qat(model_fp32_fused)

            # run the detector
            per_channel_detector = PerChannelDetector(torch.backends.quantized.engine)
            optims_str, per_channel_info = per_channel_detector.generate_detector_report(model_fp32_prepared)

            # there should be optims possible
            self.assertNotEqual(
                optims_str,
                DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine),
            )

            # make sure it was able to find the single conv in the fused model
            self.assertEqual(len(per_channel_info), 1)

            # for the one conv, it should still give advice to use different qconfig
            for key in per_channel_info.keys():
                module_entry = per_channel_info[key]
                self.assertEqual(module_entry["per_channel_quantization_supported"], True)
                self.assertEqual(module_entry["per_channel_quantization_used"], False)


"""
Partition on Domain / Things to Test

- All zero tensor
- Multiple tensor dimensions
- All of the outward facing functions
- Epoch min max are correctly updating
- Batch range is correctly averaging as expected
- Reset for each epoch is correctly resetting the values

Partition on Output
- the calcuation of the ratio is occurring correctly

"""


class TestFxModelReportObserver(QuantizationTestCase):
    class NestedModifiedSingleLayerLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.obs1 = ModelReportObserver()
            self.mod1 = SingleLayerLinearModel()
            self.obs2 = ModelReportObserver()
            self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.obs1(x)
            x = self.mod1(x)
            x = self.obs2(x)
            x = self.fc1(x)
            x = self.relu(x)
            return x

    def run_model_and_common_checks(self, model, ex_input, num_epochs, batch_size):
        # split up data into batches
        split_up_data = torch.split(ex_input, batch_size)
        for epoch in range(num_epochs):
            # reset all model report obs
            model.apply(
                lambda module: module.reset_batch_and_epoch_values()
                if isinstance(module, ModelReportObserver)
                else None
            )

            # quick check that a reset occurred
            self.assertEqual(
                model.obs1.average_batch_activation_range,
                torch.tensor(float(0)),
            )
            self.assertEqual(model.obs1.epoch_activation_min, torch.tensor(float("inf")))
            self.assertEqual(model.obs1.epoch_activation_max, torch.tensor(float("-inf")))

            # loop through the batches and run through
            for index, batch in enumerate(split_up_data):

                num_tracked_so_far = model.obs1.num_batches_tracked
                self.assertEqual(num_tracked_so_far, index)

                # get general info about the batch and the model to use later
                batch_min, batch_max = torch.aminmax(batch)
                current_average_range = model.obs1.average_batch_activation_range
                current_epoch_min = model.obs1.epoch_activation_min
                current_epoch_max = model.obs1.epoch_activation_max

                # run input through
                model(ex_input)

                # check that average batch activation range updated correctly
                correct_updated_value = (current_average_range * num_tracked_so_far + (batch_max - batch_min)) / (
                    num_tracked_so_far + 1
                )
                self.assertEqual(
                    model.obs1.average_batch_activation_range,
                    correct_updated_value,
                )

                if current_epoch_max - current_epoch_min > 0:
                    self.assertEqual(
                        model.obs1.get_batch_to_epoch_ratio(),
                        correct_updated_value / (current_epoch_max - current_epoch_min),
                    )

    """Case includes:
        all zero tensor
        dim size = 2
        run for 1 epoch
        run for 10 batch
        tests input data observer
    """

    def test_zero_tensor_errors(self):
        # initialize the model
        model = self.NestedModifiedSingleLayerLinear()

        # generate the desired input
        ex_input = torch.zeros((10, 1, 5))

        # run it through the model and do general tests
        self.run_model_and_common_checks(model, ex_input, 1, 1)

        # make sure final values are all 0
        self.assertEqual(model.obs1.epoch_activation_min, 0)
        self.assertEqual(model.obs1.epoch_activation_max, 0)
        self.assertEqual(model.obs1.average_batch_activation_range, 0)

        # we should get an error if we try to calculate the ratio
        with self.assertRaises(ValueError):
            ratio_val = model.obs1.get_batch_to_epoch_ratio()

    """Case includes:
    non-zero tensor
    dim size = 2
    run for 1 epoch
    run for 1 batch
    tests input data observer
    """

    def test_single_batch_of_ones(self):
        # initialize the model
        model = self.NestedModifiedSingleLayerLinear()

        # generate the desired input
        ex_input = torch.ones((1, 1, 5))

        # run it through the model and do general tests
        self.run_model_and_common_checks(model, ex_input, 1, 1)

        # make sure final values are all 0 except for range
        self.assertEqual(model.obs1.epoch_activation_min, 1)
        self.assertEqual(model.obs1.epoch_activation_max, 1)
        self.assertEqual(model.obs1.average_batch_activation_range, 0)

        # we should get an error if we try to calculate the ratio
        with self.assertRaises(ValueError):
            ratio_val = model.obs1.get_batch_to_epoch_ratio()

    """Case includes:
    non-zero tensor
    dim size = 2
    run for 10 epoch
    run for 15 batch
    tests non input data observer
    """

    def test_observer_after_relu(self):

        # model specific to this test
        class NestedModifiedObserverAfterRelu(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.obs1 = ModelReportObserver()
                self.mod1 = SingleLayerLinearModel()
                self.obs2 = ModelReportObserver()
                self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.obs1(x)
                x = self.mod1(x)
                x = self.fc1(x)
                x = self.relu(x)
                x = self.obs2(x)
                return x

        # initialize the model
        model = NestedModifiedObserverAfterRelu()

        # generate the desired input
        ex_input = torch.randn((15, 1, 5))

        # run it through the model and do general tests
        self.run_model_and_common_checks(model, ex_input, 10, 15)

    """Case includes:
        non-zero tensor
        dim size = 2
        run for multiple epoch
        run for multiple batch
        tests input data observer
    """

    def test_random_epochs_and_batches(self):

        # set up a basic model
        class TinyNestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.obs1 = ModelReportObserver()
                self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)
                self.relu = torch.nn.ReLU()
                self.obs2 = ModelReportObserver()

            def forward(self, x):
                x = self.obs1(x)
                x = self.fc1(x)
                x = self.relu(x)
                x = self.obs2(x)
                return x

        class LargerIncludeNestModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.obs1 = ModelReportObserver()
                self.nested = TinyNestModule()
                self.fc1 = SingleLayerLinearModel()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.obs1(x)
                x = self.nested(x)
                x = self.fc1(x)
                x = self.relu(x)
                return x

        class ModifiedThreeOps(torch.nn.Module):
            def __init__(self, batch_norm_dim):
                super().__init__()
                self.obs1 = ModelReportObserver()
                self.linear = torch.nn.Linear(7, 3, 2)
                self.obs2 = ModelReportObserver()

                if batch_norm_dim == 2:
                    self.bn = torch.nn.BatchNorm2d(2)
                elif batch_norm_dim == 3:
                    self.bn = torch.nn.BatchNorm3d(4)
                else:
                    raise ValueError("Dim should only be 2 or 3")

                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.obs1(x)
                x = self.linear(x)
                x = self.obs2(x)
                x = self.bn(x)
                x = self.relu(x)
                return x

        class HighDimensionNet(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.obs1 = ModelReportObserver()
                self.fc1 = torch.nn.Linear(3, 7)
                self.block1 = ModifiedThreeOps(3)
                self.fc2 = torch.nn.Linear(3, 7)
                self.block2 = ModifiedThreeOps(3)
                self.fc3 = torch.nn.Linear(3, 7)

            def forward(self, x):
                x = self.obs1(x)
                x = self.fc1(x)
                x = self.block1(x)
                x = self.fc2(x)
                y = self.block2(x)
                y = self.fc3(y)
                z = x + y
                z = F.relu(z)
                return z

        # the purpose of this test is to give the observers a variety of data examples
        # initialize the model
        models = [
            self.NestedModifiedSingleLayerLinear(),
            LargerIncludeNestModel(),
            ModifiedThreeOps(2),
            HighDimensionNet(),
        ]

        # get some number of epochs and batches
        num_epochs = 10
        num_batches = 15

        input_shapes = [(1, 5), (1, 5), (2, 3, 7), (4, 1, 8, 3)]

        # generate the desired inputs
        inputs = []
        for shape in input_shapes:
            ex_input = torch.randn((num_batches, *shape))
            inputs.append(ex_input)

        # run it through the model and do general tests
        for index, model in enumerate(models):
            self.run_model_and_common_checks(model, inputs[index], num_epochs, num_batches)


"""
Partition on domain / things to test

There is only a single test case for now.

This will be more thoroughly tested with the implementation of the full end to end tool coming soon.
"""


class TestFxModelReportDetectDynamicStatic(QuantizationTestCase):
    @skipIfNoFBGEMM
    def test_nested_detection_case(self):
        class SingleLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                x = self.linear(x)
                return x

        class TwoBlockNet(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.block1 = SingleLinear()
                self.block2 = SingleLinear()

            def forward(self, x):
                x = self.block1(x)
                y = self.block2(x)
                z = x + y
                z = F.relu(z)
                return z


        with override_quantized_engine('fbgemm'):
            # create model, example input, and qconfig mapping
            torch.backends.quantized.engine = "fbgemm"
            model = TwoBlockNet()
            example_input = torch.randint(-10, 0, (1, 3, 3, 3))
            example_input = example_input.to(torch.float)
            q_config_mapping = QConfigMapping()
            q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig("fbgemm"))

            # prep model and select observer
            model_prep = quantize_fx.prepare_fx(model, q_config_mapping, example_input)
            obs_ctr = ModelReportObserver

            # find layer to attach to and store
            linear_fqn = "block2.linear"  # fqn of target linear

            target_linear = None
            for node in model_prep.graph.nodes:
                if node.target == linear_fqn:
                    target_linear = node
                    break

            # insert into both module and graph pre and post

            # set up to insert before target_linear (pre_observer)
            with model_prep.graph.inserting_before(target_linear):
                obs_to_insert = obs_ctr()
                pre_obs_fqn = linear_fqn + ".model_report_pre_observer"
                model_prep.add_submodule(pre_obs_fqn, obs_to_insert)
                model_prep.graph.create_node(op="call_module", target=pre_obs_fqn, args=target_linear.args)

            # set up and insert after the target_linear (post_observer)
            with model_prep.graph.inserting_after(target_linear):
                obs_to_insert = obs_ctr()
                post_obs_fqn = linear_fqn + ".model_report_post_observer"
                model_prep.add_submodule(post_obs_fqn, obs_to_insert)
                model_prep.graph.create_node(op="call_module", target=post_obs_fqn, args=(target_linear,))

            # need to recompile module after submodule added and pass input through
            model_prep.recompile()

            num_iterations = 10
            for i in range(num_iterations):
                if i % 2 == 0:
                    example_input = torch.randint(-10, 0, (1, 3, 3, 3)).to(torch.float)
                else:
                    example_input = torch.randint(0, 10, (1, 3, 3, 3)).to(torch.float)
                model_prep(example_input)

            # run it through the dynamic vs static detector
            dynamic_vs_static_detector = DynamicStaticDetector()
            dynam_vs_stat_str, dynam_vs_stat_dict = dynamic_vs_static_detector.generate_detector_report(model_prep)

            # one of the stats should be stationary, and the other non-stationary
            # as a result, dynamic should be recommended
            data_dist_info = [
                dynam_vs_stat_dict[linear_fqn][DynamicStaticDetector.PRE_OBS_DATA_DIST_KEY],
                dynam_vs_stat_dict[linear_fqn][DynamicStaticDetector.POST_OBS_DATA_DIST_KEY],
            ]

            self.assertTrue("stationary" in data_dist_info)
            self.assertTrue("non-stationary" in data_dist_info)
            self.assertTrue(dynam_vs_stat_dict[linear_fqn]["dynamic_recommended"])

class TestFxModelReportClass(QuantizationTestCase):

    @skipIfNoFBGEMM
    def test_constructor(self):
        """
        Tests the constructor of the ModelReport class.
        Specifically looks at:
        - The desired reports
        - Ensures that the observers of interest are properly initialized
        """

        with override_quantized_engine('fbgemm'):
            # set the backend for this test
            torch.backends.quantized.engine = "fbgemm"
            backend = torch.backends.quantized.engine

            # create a model
            model = ThreeOps()
            q_config_mapping = QConfigMapping()
            q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))
            model_prep = quantize_fx.prepare_fx(model, q_config_mapping, model.get_example_inputs()[0])

            # make an example set of detectors
            test_detector_set = {DynamicStaticDetector(), PerChannelDetector(backend)}
            # initialize with an empty detector
            model_report = ModelReport(model_prep, test_detector_set)

            # make sure internal valid reports matches
            detector_name_set = {detector.get_detector_name() for detector in test_detector_set}
            self.assertEqual(model_report.get_desired_reports_names(), detector_name_set)

            # now attempt with no valid reports, should raise error
            with self.assertRaises(ValueError):
                model_report = ModelReport(model, set())

            # number of expected obs of interest entries
            num_expected_entries = len(test_detector_set)
            self.assertEqual(len(model_report.get_observers_of_interest()), num_expected_entries)

            for value in model_report.get_observers_of_interest().values():
                self.assertEqual(len(value), 0)

    @skipIfNoFBGEMM
    def test_prepare_model_callibration(self):
        """
        Tests model_report.prepare_detailed_calibration that prepares the model for callibration
        Specifically looks at:
        - Whether observers are properly inserted into regular nn.Module
        - Whether the target and the arguments of the observers are proper
        - Whether the internal representation of observers of interest is updated
        """

        with override_quantized_engine('fbgemm'):
            # create model report object

            # create model
            model = TwoThreeOps()
            # make an example set of detectors
            torch.backends.quantized.engine = "fbgemm"
            backend = torch.backends.quantized.engine
            test_detector_set = {DynamicStaticDetector(), PerChannelDetector(backend)}
            # initialize with an empty detector

            # prepare the model
            example_input = model.get_example_inputs()[0]
            current_backend = torch.backends.quantized.engine
            q_config_mapping = QConfigMapping()
            q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))

            model_prep = quantize_fx.prepare_fx(model, q_config_mapping, example_input)

            model_report = ModelReport(model_prep, test_detector_set)

            # prepare the model for callibration
            prepared_for_callibrate_model = model_report.prepare_detailed_calibration()

            # see whether observers properly in regular nn.Module
            # there should be 4 observers present in this case
            modules_observer_cnt = 0
            for fqn, module in prepared_for_callibrate_model.named_modules():
                if isinstance(module, ModelReportObserver):
                    modules_observer_cnt += 1

            self.assertEqual(modules_observer_cnt, 4)

            model_report_str_check = "model_report"
            # also make sure arguments for observers in the graph are proper
            for node in prepared_for_callibrate_model.graph.nodes:
                # not all node targets are strings, so check
                if isinstance(node.target, str) and model_report_str_check in node.target:
                    # if pre-observer has same args as the linear (next node)
                    if "pre_observer" in node.target:
                        self.assertEqual(node.args, node.next.args)
                    # if post-observer, args are the target linear (previous node)
                    if "post_observer" in node.target:
                        self.assertEqual(node.args, (node.prev,))

            # ensure model_report observers of interest updated
            # there should be two entries
            self.assertEqual(len(model_report.get_observers_of_interest()), 2)
            for detector in test_detector_set:
                self.assertTrue(detector.get_detector_name() in model_report.get_observers_of_interest().keys())

                # get number of entries for this detector
                detector_obs_of_interest_fqns = model_report.get_observers_of_interest()[detector.get_detector_name()]

                # assert that the per channel detector has 0 and the dynamic static has 4
                if isinstance(detector, PerChannelDetector):
                    self.assertEqual(len(detector_obs_of_interest_fqns), 0)
                elif isinstance(detector, DynamicStaticDetector):
                    self.assertEqual(len(detector_obs_of_interest_fqns), 4)

            # ensure that we can prepare for callibration only once
            with self.assertRaises(ValueError):
                prepared_for_callibrate_model = model_report.prepare_detailed_calibration()


    def get_module_and_graph_cnts(self, callibrated_fx_module):
        r"""
        Calculates number of ModelReportObserver modules in the model as well as the graph structure.
        Returns a tuple of two elements:
        int: The number of ModelReportObservers found in the model
        int: The number of model_report nodes found in the graph
        """
        # get the number of observers stored as modules
        modules_observer_cnt = 0
        for fqn, module in callibrated_fx_module.named_modules():
            if isinstance(module, ModelReportObserver):
                modules_observer_cnt += 1

        # get number of observers in the graph
        model_report_str_check = "model_report"
        graph_observer_cnt = 0
        # also make sure arguments for observers in the graph are proper
        for node in callibrated_fx_module.graph.nodes:
            # not all node targets are strings, so check
            if isinstance(node.target, str) and model_report_str_check in node.target:
                # increment if we found a graph observer
                graph_observer_cnt += 1

        return (modules_observer_cnt, graph_observer_cnt)

    @skipIfNoFBGEMM
    def test_generate_report(self):
        """
            Tests model_report.generate_model_report to ensure report generation
            Specifically looks at:
            - Whether correct number of reports are being generated
            - Whether observers are being properly removed if specified
            - Whether correct blocking from generating report twice if obs removed
        """

        with override_quantized_engine('fbgemm'):
            # set the backend for this test
            torch.backends.quantized.engine = "fbgemm"

            # check whether the correct number of reports are being generated
            filled_detector_set = {DynamicStaticDetector(), PerChannelDetector(torch.backends.quantized.engine)}
            single_detector_set = {DynamicStaticDetector()}

            # create our models
            model_full = TwoThreeOps()
            model_single = TwoThreeOps()

            # prepare and callibrate two different instances of same model
            # prepare the model
            example_input = model_full.get_example_inputs()[0]
            current_backend = torch.backends.quantized.engine
            q_config_mapping = QConfigMapping()
            q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))

            model_prep_full = quantize_fx.prepare_fx(model_full, q_config_mapping, example_input)
            model_prep_single = quantize_fx.prepare_fx(model_single, q_config_mapping, example_input)

            # initialize one with filled detector
            model_report_full = ModelReport(model_prep_full, filled_detector_set)
            # initialize another with a single detector set
            model_report_single = ModelReport(model_prep_single, single_detector_set)

            # prepare the models for callibration
            prepared_for_callibrate_model_full = model_report_full.prepare_detailed_calibration()
            prepared_for_callibrate_model_single = model_report_single.prepare_detailed_calibration()

            # now callibrate the two models
            num_iterations = 10
            for i in range(num_iterations):
                example_input = torch.tensor(torch.randint(100, (1, 3, 3, 3)), dtype=torch.float)
                prepared_for_callibrate_model_full(example_input)
                prepared_for_callibrate_model_single(example_input)

            # now generate the reports
            model_full_report = model_report_full.generate_model_report(True)
            model_single_report = model_report_single.generate_model_report(False)

            # check that sizes are appropriate
            self.assertEqual(len(model_full_report), len(filled_detector_set))
            self.assertEqual(len(model_single_report), len(single_detector_set))

            # make sure observers are being properly removed for full report since we put flag in
            modules_observer_cnt, graph_observer_cnt = self.get_module_and_graph_cnts(prepared_for_callibrate_model_full)
            self.assertEqual(modules_observer_cnt, 0)  # assert no more observer modules
            self.assertEqual(graph_observer_cnt, 0)  # assert no more observer nodes in graph

            # make sure observers aren't being removed for single report since not specified
            modules_observer_cnt, graph_observer_cnt = self.get_module_and_graph_cnts(prepared_for_callibrate_model_single)
            self.assertNotEqual(modules_observer_cnt, 0)
            self.assertNotEqual(graph_observer_cnt, 0)

            # make sure error when try to rerun report generation for full report but not single report
            with self.assertRaises(Exception):
                model_full_report = model_report_full.generate_model_report(
                    prepared_for_callibrate_model_full, False
                )

            # make sure we don't run into error for single report
            model_single_report = model_report_single.generate_model_report(False)

    @skipIfNoFBGEMM
    def test_generate_visualizer(self):
        """
        Tests that the ModelReport class can properly create the ModelReportVisualizer instance
        Checks that:
            - Correct number of modules are represented
            - Modules are sorted
            - Correct number of features for each module
        """
        with override_quantized_engine('fbgemm'):
            # set the backend for this test
            torch.backends.quantized.engine = "fbgemm"
            # test with multiple detectors
            detector_set = set()
            detector_set.add(OutlierDetector(reference_percentile=0.95))
            detector_set.add(InputWeightEqualizationDetector(0.5))

            model = TwoThreeOps()

            # get tst model and callibrate
            prepared_for_callibrate_model, mod_report = _get_prepped_for_calibration_model_helper(
                model, detector_set, model.get_example_inputs()[0]
            )

            # now we actually callibrate the model
            example_input = model.get_example_inputs()[0]
            example_input = example_input.to(torch.float)

            prepared_for_callibrate_model(example_input)

            # try to visualize without generating report, should throw error
            with self.assertRaises(Exception):
                mod_rep_visualizaiton = mod_report.generate_visualizer()

            # now get the report by running it through ModelReport instance
            generated_report = mod_report.generate_model_report(remove_inserted_observers=False)

            # now we get the visualizer should not error
            mod_rep_visualizer: ModelReportVisualizer = mod_report.generate_visualizer()

            # since we tested with outlier detector, which looks at every base level module
            # should be six entries in the ordered dict
            mod_fqns_to_features = mod_rep_visualizer.generated_reports

            self.assertEqual(len(mod_fqns_to_features), 6)

            # outlier detector has 9 feature per module
            # input-weight has 12 features per module
            # there are 1 common data point, so should be 12 + 9 - 1 = 20 unique features per common modules
            # all linears will be common
            for module_fqn in mod_fqns_to_features:
                if ".linear" in module_fqn:
                    linear_info = mod_fqns_to_features[module_fqn]
                    self.assertEqual(len(linear_info), 20)

    @skipIfNoFBGEMM
    def test_qconfig_mapping_generation(self):
        """
        Tests for generation of qconfigs by ModelReport API
        - Tests that qconfigmapping is generated
        - Tests that mappings include information for for relavent modules
        """
        with override_quantized_engine('fbgemm'):
            # set the backend for this test
            torch.backends.quantized.engine = "fbgemm"
            # test with multiple detectors
            detector_set = set()
            detector_set.add(PerChannelDetector())
            detector_set.add(DynamicStaticDetector())

            model = TwoThreeOps()

            # get tst model and callibrate
            prepared_for_callibrate_model, mod_report = _get_prepped_for_calibration_model_helper(
                model, detector_set, model.get_example_inputs()[0]
            )

            # now we actually callibrate the models
            example_input = model.get_example_inputs()[0]
            example_input = example_input.to(torch.float)

            prepared_for_callibrate_model(example_input)


            # get the mapping without error
            qconfig_mapping = mod_report.generate_qconfig_mapping()

            # now get the report by running it through ModelReport instance
            generated_report = mod_report.generate_model_report(remove_inserted_observers=False)

            # get the visualizer so we can get access to reformatted reports by module fqn
            mod_reports_by_fqn = mod_report.generate_visualizer().generated_reports

            # compare the entries of the mapping to those of the report
            # we should have the same number of entries
            self.assertEqual(len(qconfig_mapping.module_name_qconfigs), len(mod_reports_by_fqn))

            # for the non_empty one, we should have 2 because we have only applicable linears
            # so should have suggestions for each module named
            self.assertEqual(len(qconfig_mapping.module_name_qconfigs), 2)

            # only two linears, make sure per channel min max for weight since fbgemm
            # also static distribution since a simple single callibration
            for key in qconfig_mapping.module_name_qconfigs:
                config = qconfig_mapping.module_name_qconfigs[key]
                self.assertEqual(config.weight, default_per_channel_weight_observer)
                self.assertEqual(config.activation, default_observer)

            # make sure these can actually be used to prepare the model
            prepared = quantize_fx.prepare_fx(TwoThreeOps(), qconfig_mapping, example_input)

            # now convert the model to ensure no errors in conversion
            converted = quantize_fx.convert_fx(prepared)

    @skipIfNoFBGEMM
    def test_equalization_mapping_generation(self):
        """
        Tests for generation of qconfigs by ModelReport API
        - Tests that equalization config generated when input-weight equalization detector used
        - Tests that mappings include information for for relavent modules
        """
        with override_quantized_engine('fbgemm'):
            # set the backend for this test
            torch.backends.quantized.engine = "fbgemm"
            # test with multiple detectors
            detector_set = set()
            detector_set.add(InputWeightEqualizationDetector(0.6))

            model = TwoThreeOps()

            # get tst model and callibrate
            prepared_for_callibrate_model, mod_report = _get_prepped_for_calibration_model_helper(
                model, detector_set, model.get_example_inputs()[0]
            )

            # now we actually callibrate the models
            example_input = model.get_example_inputs()[0]
            example_input = example_input.to(torch.float)

            prepared_for_callibrate_model(example_input)


            # get the mapping without error
            qconfig_mapping = mod_report.generate_qconfig_mapping()
            equalization_mapping = mod_report.generate_equalization_mapping()

            # tests a lot more simple for the equalization mapping

            # shouldn't have any equalization suggestions for this case
            self.assertEqual(len(qconfig_mapping.module_name_qconfigs), 2)


            # make sure these can actually be used to prepare the model
            prepared = quantize_fx.prepare_fx(
                TwoThreeOps(),
                qconfig_mapping,
                example_input,
                _equalization_config=equalization_mapping
            )

            # now convert the model to ensure no errors in conversion
            converted = quantize_fx.convert_fx(prepared)

class TestFxDetectInputWeightEqualization(QuantizationTestCase):

    class SimpleConv(torch.nn.Module):
        def __init__(self, con_dims):
            super().__init__()
            self.relu = torch.nn.ReLU()
            self.conv = torch.nn.Conv2d(con_dims[0], con_dims[1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            return x

    class TwoBlockComplexNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block1 = TestFxDetectInputWeightEqualization.SimpleConv((3, 32))
            self.block2 = TestFxDetectInputWeightEqualization.SimpleConv((3, 3))
            self.conv = torch.nn.Conv2d(32, 3, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False)
            self.linear = torch.nn.Linear(768, 10)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.block1(x)
            x = self.conv(x)
            y = self.block2(x)
            y = y.repeat(1, 1, 2, 2)
            z = x + y
            z = z.flatten(start_dim=1)
            z = self.linear(z)
            z = self.relu(z)
            return z

        def get_fusion_modules(self):
            return [['conv', 'relu']]

        def get_example_inputs(self):
            return (torch.randn((1, 3, 28, 28)),)

    class ReluOnly(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.relu(x)
            return x

        def get_example_inputs(self):
            return (torch.arange(27).reshape((1, 3, 3, 3)),)

    def _get_prepped_for_calibration_model(self, model, detector_set, fused=False):
        r"""Returns a model that has been prepared for callibration and corresponding model_report"""

        # pass in necessary inputs to helper
        example_input = model.get_example_inputs()[0]
        return _get_prepped_for_calibration_model_helper(model, detector_set, example_input, fused)

    @skipIfNoFBGEMM
    def test_input_weight_equalization_determine_points(self):
        # use fbgemm and create our model instance
        # then create model report instance with detector
        with override_quantized_engine('fbgemm'):

            detector_set = {InputWeightEqualizationDetector(0.5)}

            # get tst model and callibrate
            non_fused = self._get_prepped_for_calibration_model(self.TwoBlockComplexNet(), detector_set)
            fused = self._get_prepped_for_calibration_model(self.TwoBlockComplexNet(), detector_set, fused=True)

            # reporter should still give same counts even for fused model
            for prepared_for_callibrate_model, mod_report in [non_fused, fused]:

                # supported modules to check
                mods_to_check = {nn.Linear, nn.Conv2d}

                # get the set of all nodes in the graph their fqns
                node_fqns = {node.target for node in prepared_for_callibrate_model.graph.nodes}

                # there should be 4 node fqns that have the observer inserted
                correct_number_of_obs_inserted = 4
                number_of_obs_found = 0
                obs_name_to_find = InputWeightEqualizationDetector.DEFAULT_PRE_OBSERVER_NAME

                for node in prepared_for_callibrate_model.graph.nodes:
                    # if the obs name is inside the target, we found an observer
                    if obs_name_to_find in str(node.target):
                        number_of_obs_found += 1

                self.assertEqual(number_of_obs_found, correct_number_of_obs_inserted)

                # assert that each of the desired modules have the observers inserted
                for fqn, module in prepared_for_callibrate_model.named_modules():
                    # check if module is a supported module
                    is_in_include_list = sum(isinstance(module, x) for x in mods_to_check) > 0

                    if is_in_include_list:
                        # make sure it has the observer attribute
                        self.assertTrue(hasattr(module, InputWeightEqualizationDetector.DEFAULT_PRE_OBSERVER_NAME))
                    else:
                        # if it's not a supported type, it shouldn't have observer attached
                        self.assertTrue(not hasattr(module, InputWeightEqualizationDetector.DEFAULT_PRE_OBSERVER_NAME))

    @skipIfNoFBGEMM
    def test_input_weight_equalization_report_gen(self):
        # use fbgemm and create our model instance
        # then create model report instance with detector
        with override_quantized_engine('fbgemm'):

            test_input_weight_detector = InputWeightEqualizationDetector(0.4)
            detector_set = {test_input_weight_detector}
            model = self.TwoBlockComplexNet()
            # prepare the model for callibration
            prepared_for_callibrate_model, model_report = self._get_prepped_for_calibration_model(
                model, detector_set
            )

            # now we actually callibrate the model
            example_input = model.get_example_inputs()[0]
            example_input = example_input.to(torch.float)

            prepared_for_callibrate_model(example_input)

            # now get the report by running it through ModelReport instance
            generated_report = model_report.generate_model_report(True)

            # check that sizes are appropriate only 1 detector
            self.assertEqual(len(generated_report), 1)

            # get the specific report for input weight equalization
            input_weight_str, input_weight_dict = generated_report[test_input_weight_detector.get_detector_name()]

            # we should have 5 layers looked at since 4 conv / linear layers
            self.assertEqual(len(input_weight_dict), 4)

            # we can validate that the max and min values of the detector were recorded properly for the first one
            # this is because no data has been processed yet, so it should be values from original input

            example_input = example_input.reshape((3, 28, 28))  # reshape input
            for module_fqn in input_weight_dict:
                # look for the first linear
                if "block1.linear" in module_fqn:
                    block_1_lin_recs = input_weight_dict[module_fqn]
                    # get input range info and the channel axis
                    ch_axis = block_1_lin_recs[InputWeightEqualizationDetector.CHANNEL_KEY]

                    # ensure that the min and max values extracted match properly
                    example_min, example_max = torch.aminmax(example_input, dim=ch_axis)
                    dimension_min = torch.amin(example_min, dim=ch_axis)
                    dimension_max = torch.amax(example_max, dim=ch_axis)

                    # make sure per channel min and max are as expected
                    min_per_key = InputWeightEqualizationDetector.ACTIVATION_PREFIX
                    min_per_key += InputWeightEqualizationDetector.PER_CHANNEL_MIN_KEY

                    max_per_key = InputWeightEqualizationDetector.ACTIVATION_PREFIX
                    max_per_key += InputWeightEqualizationDetector.PER_CHANNEL_MAX_KEY

                    per_channel_min = block_1_lin_recs[min_per_key]
                    per_channel_max = block_1_lin_recs[max_per_key]
                    self.assertEqual(per_channel_min, dimension_min)
                    self.assertEqual(per_channel_max, dimension_max)

                    # make sure per channel min and max are as expected
                    min_key = InputWeightEqualizationDetector.ACTIVATION_PREFIX
                    min_key += InputWeightEqualizationDetector.GLOBAL_MIN_KEY

                    max_key = InputWeightEqualizationDetector.ACTIVATION_PREFIX
                    max_key += InputWeightEqualizationDetector.GLOBAL_MAX_KEY

                    # make sure the global min and max were correctly recorded and presented
                    global_min = block_1_lin_recs[min_key]
                    global_max = block_1_lin_recs[max_key]
                    self.assertEqual(global_min, min(dimension_min))
                    self.assertEqual(global_max, max(dimension_max))

                    input_ratio = torch.sqrt((per_channel_max - per_channel_min) / (global_max - global_min))
                    # ensure comparision stat passed back is sqrt of range ratios
                    # need to get the weight ratios first

                    # make sure per channel min and max are as expected
                    min_per_key = InputWeightEqualizationDetector.WEIGHT_PREFIX
                    min_per_key += InputWeightEqualizationDetector.PER_CHANNEL_MIN_KEY

                    max_per_key = InputWeightEqualizationDetector.WEIGHT_PREFIX
                    max_per_key += InputWeightEqualizationDetector.PER_CHANNEL_MAX_KEY

                    # get weight per channel and global info
                    per_channel_min = block_1_lin_recs[min_per_key]
                    per_channel_max = block_1_lin_recs[max_per_key]

                    # make sure per channel min and max are as expected
                    min_key = InputWeightEqualizationDetector.WEIGHT_PREFIX
                    min_key += InputWeightEqualizationDetector.GLOBAL_MIN_KEY

                    max_key = InputWeightEqualizationDetector.WEIGHT_PREFIX
                    max_key += InputWeightEqualizationDetector.GLOBAL_MAX_KEY

                    global_min = block_1_lin_recs[min_key]
                    global_max = block_1_lin_recs[max_key]

                    weight_ratio = torch.sqrt((per_channel_max - per_channel_min) / (global_max - global_min))

                    # also get comp stat for this specific layer
                    comp_stat = block_1_lin_recs[InputWeightEqualizationDetector.COMP_METRIC_KEY]

                    weight_to_input_ratio = weight_ratio / input_ratio

                    self.assertEqual(comp_stat, weight_to_input_ratio)
                    # only looking at the first example so can break
                    break

    @skipIfNoFBGEMM
    def test_input_weight_equalization_report_gen_empty(self):
        # tests report gen on a model that doesn't have any layers
        # use fbgemm and create our model instance
        # then create model report instance with detector
        with override_quantized_engine('fbgemm'):
            test_input_weight_detector = InputWeightEqualizationDetector(0.4)
            detector_set = {test_input_weight_detector}
            model = self.ReluOnly()
            # prepare the model for callibration
            prepared_for_callibrate_model, model_report = self._get_prepped_for_calibration_model(model, detector_set)

            # now we actually callibrate the model
            example_input = model.get_example_inputs()[0]
            example_input = example_input.to(torch.float)

            prepared_for_callibrate_model(example_input)

            # now get the report by running it through ModelReport instance
            generated_report = model_report.generate_model_report(True)

            # check that sizes are appropriate only 1 detector
            self.assertEqual(len(generated_report), 1)

            # get the specific report for input weight equalization
            input_weight_str, input_weight_dict = generated_report[test_input_weight_detector.get_detector_name()]

            # we should have 0 layers since there is only a Relu
            self.assertEqual(len(input_weight_dict), 0)

            # make sure that the string only has two lines, as should be if no suggestions
            self.assertEqual(input_weight_str.count("\n"), 2)


class TestFxDetectOutliers(QuantizationTestCase):

    class LargeBatchModel(torch.nn.Module):
        def __init__(self, param_size):
            super().__init__()
            self.param_size = param_size
            self.linear = torch.nn.Linear(param_size, param_size)
            self.relu_1 = torch.nn.ReLU()
            self.conv = torch.nn.Conv2d(param_size, param_size, 1)
            self.relu_2 = torch.nn.ReLU()

        def forward(self, x):
            x = self.linear(x)
            x = self.relu_1(x)
            x = self.conv(x)
            x = self.relu_2(x)
            return x

        def get_example_inputs(self):
            param_size = self.param_size
            return (torch.randn((1, param_size, param_size, param_size)),)

        def get_outlier_inputs(self):
            param_size = self.param_size
            random_vals = torch.randn((1, param_size, param_size, param_size))
            # change one in some of them to be a massive value
            random_vals[:, 0:param_size:2, 0, 3] = torch.tensor([3.28e8])
            return (random_vals,)


    def _get_prepped_for_calibration_model(self, model, detector_set, use_outlier_data=False):
        r"""Returns a model that has been prepared for callibration and corresponding model_report"""
        # call the general helper function to callibrate
        example_input = model.get_example_inputs()[0]

        # if we specifically want to test data with outliers replace input
        if use_outlier_data:
            example_input = model.get_outlier_inputs()[0]

        return _get_prepped_for_calibration_model_helper(model, detector_set, example_input)

    @skipIfNoFBGEMM
    def test_outlier_detection_determine_points(self):
        # use fbgemm and create our model instance
        # then create model report instance with detector
        # similar to test for InputWeightEqualization but key differences that made refactoring not viable
        # not explicitly testing fusion because fx workflow automatically
        with override_quantized_engine('fbgemm'):

            detector_set = {OutlierDetector(reference_percentile=0.95)}

            # get tst model and callibrate
            prepared_for_callibrate_model, mod_report = self._get_prepped_for_calibration_model(
                self.LargeBatchModel(param_size=128), detector_set
            )

            # supported modules to check
            mods_to_check = {nn.Linear, nn.Conv2d, nn.ReLU}

            # there should be 4 node fqns that have the observer inserted
            correct_number_of_obs_inserted = 4
            number_of_obs_found = 0
            obs_name_to_find = InputWeightEqualizationDetector.DEFAULT_PRE_OBSERVER_NAME

            number_of_obs_found = sum(
                1 if obs_name_to_find in str(node.target) else 0 for node in prepared_for_callibrate_model.graph.nodes
            )
            self.assertEqual(number_of_obs_found, correct_number_of_obs_inserted)

            # assert that each of the desired modules have the observers inserted
            for fqn, module in prepared_for_callibrate_model.named_modules():
                # check if module is a supported module
                is_in_include_list = isinstance(module, tuple(mods_to_check))

                if is_in_include_list:
                    # make sure it has the observer attribute
                    self.assertTrue(hasattr(module, InputWeightEqualizationDetector.DEFAULT_PRE_OBSERVER_NAME))
                else:
                    # if it's not a supported type, it shouldn't have observer attached
                    self.assertTrue(not hasattr(module, InputWeightEqualizationDetector.DEFAULT_PRE_OBSERVER_NAME))

    @skipIfNoFBGEMM
    def test_no_outlier_report_gen(self):
        # use fbgemm and create our model instance
        # then create model report instance with detector
        with override_quantized_engine('fbgemm'):

            # test with multiple detectors
            outlier_detector = OutlierDetector(reference_percentile=0.95)
            dynamic_static_detector = DynamicStaticDetector(tolerance=0.5)

            param_size: int = 4
            detector_set = {outlier_detector, dynamic_static_detector}
            model = self.LargeBatchModel(param_size=param_size)

            # get tst model and callibrate
            prepared_for_callibrate_model, mod_report = self._get_prepped_for_calibration_model(
                model, detector_set
            )

            # now we actually callibrate the model
            example_input = model.get_example_inputs()[0]
            example_input = example_input.to(torch.float)

            prepared_for_callibrate_model(example_input)

            # now get the report by running it through ModelReport instance
            generated_report = mod_report.generate_model_report(True)

            # check that sizes are appropriate only 2 detectors
            self.assertEqual(len(generated_report), 2)

            # get the specific report for input weight equalization
            outlier_str, outlier_dict = generated_report[outlier_detector.get_detector_name()]

            # we should have 5 layers looked at since 4 conv + linear + relu
            self.assertEqual(len(outlier_dict), 4)

            # assert the following are true for all the modules
            for module_fqn in outlier_dict:
                # get the info for the specific module
                module_dict = outlier_dict[module_fqn]

                # there really should not be any outliers since we used a normal distribution to perform this calculation
                outlier_info = module_dict[OutlierDetector.OUTLIER_KEY]
                self.assertEqual(sum(outlier_info), 0)

                # ensure that the number of ratios and batches counted is the same as the number of params
                self.assertEqual(len(module_dict[OutlierDetector.COMP_METRIC_KEY]), param_size)
                self.assertEqual(len(module_dict[OutlierDetector.NUM_BATCHES_KEY]), param_size)


    @skipIfNoFBGEMM
    def test_all_outlier_report_gen(self):
        # make the percentile 0 and the ratio 1, and then see that everything is outlier according to it
        # use fbgemm and create our model instance
        # then create model report instance with detector
        with override_quantized_engine('fbgemm'):
            # create detector of interest
            outlier_detector = OutlierDetector(ratio_threshold=1, reference_percentile=0)

            param_size: int = 16
            detector_set = {outlier_detector}
            model = self.LargeBatchModel(param_size=param_size)

            # get tst model and callibrate
            prepared_for_callibrate_model, mod_report = self._get_prepped_for_calibration_model(
                model, detector_set
            )

            # now we actually callibrate the model
            example_input = model.get_example_inputs()[0]
            example_input = example_input.to(torch.float)

            prepared_for_callibrate_model(example_input)

            # now get the report by running it through ModelReport instance
            generated_report = mod_report.generate_model_report(True)

            # check that sizes are appropriate only 1 detector
            self.assertEqual(len(generated_report), 1)

            # get the specific report for input weight equalization
            outlier_str, outlier_dict = generated_report[outlier_detector.get_detector_name()]

            # we should have 5 layers looked at since 4 conv + linear + relu
            self.assertEqual(len(outlier_dict), 4)

            # assert the following are true for all the modules
            for module_fqn in outlier_dict:
                # get the info for the specific module
                module_dict = outlier_dict[module_fqn]

                # everything should be an outlier because we said that the max should be equal to the min for all of them
                # however we will just test and say most should be in case we have several 0 channel values
                outlier_info = module_dict[OutlierDetector.OUTLIER_KEY]
                assert sum(outlier_info) >= len(outlier_info) / 2

                # ensure that the number of ratios and batches counted is the same as the number of params
                self.assertEqual(len(module_dict[OutlierDetector.COMP_METRIC_KEY]), param_size)
                self.assertEqual(len(module_dict[OutlierDetector.NUM_BATCHES_KEY]), param_size)

    @skipIfNoFBGEMM
    def test_multiple_run_consistent_spike_outlier_report_gen(self):
        # specifically make a row really high consistently in the number of batches that you are testing and try that
        # generate report after just 1 run, and after many runs (30) and make sure above minimum threshold is there
        with override_quantized_engine('fbgemm'):

            # detector of interest
            outlier_detector = OutlierDetector(reference_percentile=0.95)

            param_size: int = 8
            detector_set = {outlier_detector}
            model = self.LargeBatchModel(param_size=param_size)

            # get tst model and callibrate
            prepared_for_callibrate_model, mod_report = self._get_prepped_for_calibration_model(
                model, detector_set, use_outlier_data=True
            )

            # now we actually callibrate the model
            example_input = model.get_outlier_inputs()[0]
            example_input = example_input.to(torch.float)

            # now callibrate minimum 30 times to make it above minimum threshold
            for i in range(30):
                example_input = model.get_outlier_inputs()[0]
                example_input = example_input.to(torch.float)

                # make 2 of the batches to have zero channel
                if i % 14 == 0:
                    # make one channel constant
                    example_input[0][1] = torch.zeros_like(example_input[0][1])

                prepared_for_callibrate_model(example_input)

            # now get the report by running it through ModelReport instance
            generated_report = mod_report.generate_model_report(True)

            # check that sizes are appropriate only 1 detector
            self.assertEqual(len(generated_report), 1)

            # get the specific report for input weight equalization
            outlier_str, outlier_dict = generated_report[outlier_detector.get_detector_name()]

            # we should have 5 layers looked at since 4 conv + linear + relu
            self.assertEqual(len(outlier_dict), 4)

            # assert the following are true for all the modules
            for module_fqn in outlier_dict:
                # get the info for the specific module
                module_dict = outlier_dict[module_fqn]

                # because we ran 30 times, we should have at least a couple be significant
                # could be less because some channels could possibly be all 0
                sufficient_batches_info = module_dict[OutlierDetector.IS_SUFFICIENT_BATCHES_KEY]
                assert sum(sufficient_batches_info) >= len(sufficient_batches_info) / 2

                # half of them should be outliers, because we set a really high value every 2 channels
                outlier_info = module_dict[OutlierDetector.OUTLIER_KEY]
                self.assertEqual(sum(outlier_info), len(outlier_info) / 2)

                # ensure that the number of ratios and batches counted is the same as the number of params
                self.assertEqual(len(module_dict[OutlierDetector.COMP_METRIC_KEY]), param_size)
                self.assertEqual(len(module_dict[OutlierDetector.NUM_BATCHES_KEY]), param_size)

                # for the first one ensure the per channel max values are what we set
                if module_fqn == "linear.0":

                    # check that the non-zero channel count, at least 2 should be there
                    # for the first module
                    counts_info = module_dict[OutlierDetector.CONSTANT_COUNTS_KEY]
                    assert sum(counts_info) >= 2

                    # half of the recorded max values should be what we set
                    matched_max = sum(val == 3.28e8 for val in module_dict[OutlierDetector.MAX_VALS_KEY])
                    self.assertEqual(matched_max, param_size / 2)


class TestFxModelReportVisualizer(QuantizationTestCase):

    def _callibrate_and_generate_visualizer(self, model, prepared_for_callibrate_model, mod_report):
        r"""
        Callibrates the passed in model, generates report, and returns the visualizer
        """
        # now we actually callibrate the model
        example_input = model.get_example_inputs()[0]
        example_input = example_input.to(torch.float)

        prepared_for_callibrate_model(example_input)

        # now get the report by running it through ModelReport instance
        generated_report = mod_report.generate_model_report(remove_inserted_observers=False)

        # now we get the visualizer should not error
        mod_rep_visualizer: ModelReportVisualizer = mod_report.generate_visualizer()

        return mod_rep_visualizer

    @skipIfNoFBGEMM
    def test_get_modules_and_features(self):
        """
        Tests the get_all_unique_module_fqns and get_all_unique_feature_names methods of
        ModelReportVisualizer

        Checks whether returned sets are of proper size and filtered properly
        """
        with override_quantized_engine('fbgemm'):
            # set the backend for this test
            torch.backends.quantized.engine = "fbgemm"
            # test with multiple detectors
            detector_set = set()
            detector_set.add(OutlierDetector(reference_percentile=0.95))
            detector_set.add(InputWeightEqualizationDetector(0.5))

            model = TwoThreeOps()

            # get tst model and callibrate
            prepared_for_callibrate_model, mod_report = _get_prepped_for_calibration_model_helper(
                model, detector_set, model.get_example_inputs()[0]
            )

            mod_rep_visualizer: ModelReportVisualizer = self._callibrate_and_generate_visualizer(
                model, prepared_for_callibrate_model, mod_report
            )

            # ensure the module fqns match the ones given by the get_all_unique_feature_names method
            actual_model_fqns = set(mod_rep_visualizer.generated_reports.keys())
            returned_model_fqns = mod_rep_visualizer.get_all_unique_module_fqns()
            self.assertEqual(returned_model_fqns, actual_model_fqns)

            # now ensure that features are all properly returned
            # all the linears have all the features for two detectors
            # can use those as check that method is working reliably
            b_1_linear_features = mod_rep_visualizer.generated_reports["block1.linear"]

            # first test all features
            returned_all_feats = mod_rep_visualizer.get_all_unique_feature_names(False)
            self.assertEqual(returned_all_feats, set(b_1_linear_features.keys()))

            # now test plottable features
            plottable_set = set()

            for feature_name in b_1_linear_features:
                if type(b_1_linear_features[feature_name]) == torch.Tensor:
                    plottable_set.add(feature_name)

            returned_plottable_feats = mod_rep_visualizer.get_all_unique_feature_names()
            self.assertEqual(returned_plottable_feats, plottable_set)

    def _prep_visualizer_helper(self):
        r"""
        Returns a mod rep visualizer that we test in various ways
        """
        # set backend for test
        torch.backends.quantized.engine = "fbgemm"

        # test with multiple detectors
        detector_set = set()
        detector_set.add(OutlierDetector(reference_percentile=0.95))
        detector_set.add(InputWeightEqualizationDetector(0.5))

        model = TwoThreeOps()

        # get tst model and callibrate
        prepared_for_callibrate_model, mod_report = _get_prepped_for_calibration_model_helper(
            model, detector_set, model.get_example_inputs()[0]
        )

        mod_rep_visualizer: ModelReportVisualizer = self._callibrate_and_generate_visualizer(
            model, prepared_for_callibrate_model, mod_report
        )

        return mod_rep_visualizer

    @skipIfNoFBGEMM
    def test_generate_tables_match_with_report(self):
        """
        Tests the generate_table_view()
        ModelReportVisualizer

        Checks whether the generated dict has proper information
            Visual check that the tables look correct performed during testing
        """
        with override_quantized_engine('fbgemm'):

            # get the visualizer
            mod_rep_visualizer = self._prep_visualizer_helper()

            table_dict = mod_rep_visualizer.generate_filtered_tables()

            # test primarily the dict since it has same info as str
            tensor_headers, tensor_table = table_dict[ModelReportVisualizer.TABLE_TENSOR_KEY]
            channel_headers, channel_table = table_dict[ModelReportVisualizer.TABLE_CHANNEL_KEY]

            # these two together should be the same as the generated report info in terms of keys
            tensor_info_modules = {row[1] for row in tensor_table}
            channel_info_modules = {row[1] for row in channel_table}
            combined_modules: Set = tensor_info_modules.union(channel_info_modules)

            generated_report_keys: Set = set(mod_rep_visualizer.generated_reports.keys())
            self.assertEqual(combined_modules, generated_report_keys)

    @skipIfNoFBGEMM
    def test_generate_tables_no_match(self):
        """
        Tests the generate_table_view()
        ModelReportVisualizer

        Checks whether the generated dict has proper information
            Visual check that the tables look correct performed during testing
        """
        with override_quantized_engine('fbgemm'):
            # get the visualizer
            mod_rep_visualizer = self._prep_visualizer_helper()

            # try a random filter and make sure that there are no rows for either table
            empty_tables_dict = mod_rep_visualizer.generate_filtered_tables(module_fqn_filter="random not there module")

            # test primarily the dict since it has same info as str
            tensor_headers, tensor_table = empty_tables_dict[ModelReportVisualizer.TABLE_TENSOR_KEY]
            channel_headers, channel_table = empty_tables_dict[ModelReportVisualizer.TABLE_CHANNEL_KEY]

            tensor_info_modules = {row[1] for row in tensor_table}
            channel_info_modules = {row[1] for row in channel_table}
            combined_modules: Set = tensor_info_modules.union(channel_info_modules)
            self.assertEqual(len(combined_modules), 0)  # should be no matching modules

    @skipIfNoFBGEMM
    def test_generate_tables_single_feat_match(self):
        """
        Tests the generate_table_view()
        ModelReportVisualizer

        Checks whether the generated dict has proper information
            Visual check that the tables look correct performed during testing
        """
        with override_quantized_engine('fbgemm'):
            # get the visualizer
            mod_rep_visualizer = self._prep_visualizer_helper()

            # try a matching filter for feature and make sure only those features show up
            # if we filter to a very specific feature name, should only have 1 additional column in each table row
            single_feat_dict = mod_rep_visualizer.generate_filtered_tables(feature_filter=OutlierDetector.MAX_VALS_KEY)

            # test primarily the dict since it has same info as str
            tensor_headers, tensor_table = single_feat_dict[ModelReportVisualizer.TABLE_TENSOR_KEY]
            channel_headers, channel_table = single_feat_dict[ModelReportVisualizer.TABLE_CHANNEL_KEY]

            # get the number of features in each of these
            tensor_info_features = len(tensor_headers)
            channel_info_features = len(channel_headers) - ModelReportVisualizer.NUM_NON_FEATURE_CHANNEL_HEADERS

            # make sure that there are no tensor features, and that there is one channel level feature
            self.assertEqual(tensor_info_features, 0)
            self.assertEqual(channel_info_features, 1)

def _get_prepped_for_calibration_model_helper(model, detector_set, example_input, fused: bool = False):
    r"""Returns a model that has been prepared for callibration and corresponding model_report"""
    # set the backend for this test
    torch.backends.quantized.engine = "fbgemm"

    # create model instance and prepare it
    example_input = example_input.to(torch.float)
    q_config_mapping = torch.ao.quantization.get_default_qconfig_mapping()

    # if they passed in fusion paramter, make sure to test that
    if fused:
        model = torch.ao.quantization.fuse_modules(model, model.get_fusion_modules())

    model_prep = quantize_fx.prepare_fx(model, q_config_mapping, example_input)

    model_report = ModelReport(model_prep, detector_set)

    # prepare the model for callibration
    prepared_for_callibrate_model = model_report.prepare_detailed_calibration()

    return (prepared_for_callibrate_model, model_report)
