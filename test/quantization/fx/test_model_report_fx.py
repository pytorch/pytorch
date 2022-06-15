# -*- coding: utf-8 -*-
# Owner(s): ["oncall: quantization"]

import torch
import torch.ao.quantization.quantize_fx as quantize_fx
import torch.nn.functional as F
from torch.ao.quantization import QConfig, QConfigMapping
from torch.ao.quantization.fx._model_report._detector import _detect_dynamic_vs_static, _detect_per_channel
from torch.ao.quantization.fx._model_report.model_report_observer import ModelReportObserver
from torch.ao.quantization.observer import HistogramObserver, default_per_channel_weight_observer
from torch.nn.intrinsic.modules.fused import ConvReLU2d, LinearReLU
from torch.testing._internal.common_quantization import (
    ConvModel,
    QuantizationTestCase,
    SingleLayerLinearModel,
    TwoLayerLinearModel,
    skipIfNoFBGEMM,
    skipIfNoQNNPACK,
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


class TestFxModelReportDetector(QuantizationTestCase):

    """Prepares and callibrate the model"""

    def _prepare_model_and_run_input(self, model, q_config_mapping, input):
        model_prep = torch.ao.quantization.quantize_fx.prepare_fx(model, q_config_mapping, input)  # prep model
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
        q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))

        input = torch.randn(1, 3, 10, 10)
        prepared_model = self._prepare_model_and_run_input(ConvModel(), q_config_mapping, input)

        # run the detector
        optims_str, per_channel_info = _detect_per_channel(prepared_model)

        # no optims possible and there should be nothing in per_channel_status
        self.assertEqual(
            optims_str,
            DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine),
        )
        self.assertEqual(per_channel_info["backend"], torch.backends.quantized.engine)
        self.assertEqual(len(per_channel_info["per_channel_status"]), 1)
        self.assertEqual(list(per_channel_info["per_channel_status"])[0], "conv")
        self.assertEqual(
            per_channel_info["per_channel_status"]["conv"]["per_channel_supported"],
            True,
        )
        self.assertEqual(per_channel_info["per_channel_status"]["conv"]["per_channel_used"], True)

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
        q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))

        prepared_model = self._prepare_model_and_run_input(
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

        prepared_model = self._prepare_model_and_run_input(
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
        torch.backends.quantized.engine = "qnnpack"

        q_config_mapping = QConfigMapping()
        q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))

        prepared_model = self._prepare_model_and_run_input(
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
        q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))

        prepared_model = self._prepare_model_and_run_input(
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
        q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))

        prepared_model = self._prepare_model_and_run_input(
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
        model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [["conv", "bn", "relu"]])

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
        def __init__(self):
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
                getattr(model, "obs1").average_batch_activation_range,
                torch.tensor(float(0)),
            )
            self.assertEqual(getattr(model, "obs1").epoch_activation_min, torch.tensor(float("inf")))
            self.assertEqual(getattr(model, "obs1").epoch_activation_max, torch.tensor(float("-inf")))

            # loop through the batches and run through
            for index, batch in enumerate(split_up_data):

                num_tracked_so_far = getattr(model, "obs1").num_batches_tracked
                self.assertEqual(num_tracked_so_far, index)

                # get general info about the batch and the model to use later
                batch_min, batch_max = torch.aminmax(batch)
                current_average_range = getattr(model, "obs1").average_batch_activation_range
                current_epoch_min = getattr(model, "obs1").epoch_activation_min
                current_epoch_max = getattr(model, "obs1").epoch_activation_max

                # run input through
                model(ex_input)

                # check that average batch activation range updated correctly
                correct_updated_value = (current_average_range * num_tracked_so_far + (batch_max - batch_min)) / (
                    num_tracked_so_far + 1
                )
                self.assertEqual(
                    getattr(model, "obs1").average_batch_activation_range,
                    correct_updated_value,
                )

                if current_epoch_max - current_epoch_min > 0:
                    self.assertEqual(
                        getattr(model, "obs1").get_batch_to_epoch_ratio(),
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
        self.assertEqual(getattr(model, "obs1").epoch_activation_min, 0)
        self.assertEqual(getattr(model, "obs1").epoch_activation_max, 0)
        self.assertEqual(getattr(model, "obs1").average_batch_activation_range, 0)

        # we should get an error if we try to calculate the ratio
        with self.assertRaises(ValueError):
            ratio_val = getattr(model, "obs1").get_batch_to_epoch_ratio()

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
        self.assertEqual(getattr(model, "obs1").epoch_activation_min, 1)
        self.assertEqual(getattr(model, "obs1").epoch_activation_max, 1)
        self.assertEqual(getattr(model, "obs1").average_batch_activation_range, 0)

        # we should get an error if we try to calculate the ratio
        with self.assertRaises(ValueError):
            ratio_val = getattr(model, "obs1").get_batch_to_epoch_ratio()

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
            def __init__(self):
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
            def __init__(self):
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
            def __init__(self):
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
                super(ModifiedThreeOps, self).__init__()
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
            def __init__(self):
                super(HighDimensionNet, self).__init__()
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
            def __init__(self):
                super(SingleLinear, self).__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                x = self.linear(x)
                return x

        class TwoBlockNet(torch.nn.Module):
            def __init__(self):
                super(TwoBlockNet, self).__init__()
                self.block1 = SingleLinear()
                self.block2 = SingleLinear()

            def forward(self, x):
                x = self.block1(x)
                y = self.block2(x)
                z = x + y
                z = F.relu(z)
                return z

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
        dynam_vs_stat_str, dynam_vs_stat_dict = _detect_dynamic_vs_static(model_prep, tolerance=0.5)

        # one of the stats should be stationary, and the other non-stationary
        # as a result, dynamic should be recommended
        data_dist_info = [
            dynam_vs_stat_dict[linear_fqn]["pre_observer_data_dist"],
            dynam_vs_stat_dict[linear_fqn]["post_observer_data_dist"],
        ]

        self.assertTrue("stationary" in data_dist_info)
        self.assertTrue("non-stationary" in data_dist_info)
        self.assertTrue(dynam_vs_stat_dict[linear_fqn]["dynamic_recommended"])


class TestFxModelReportClass(QuantizationTestCase):

    def test_simple_pass_case(self):
        pass
