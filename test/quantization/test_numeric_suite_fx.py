import copy

import torch
import torch.nn as nn
import torch.nn.intrinsic as nni
from torch.quantization import get_default_qconfig, default_dynamic_qconfig
from torch.quantization._numeric_suite_fx import (
    remove_qconfig_observer_fx,
    compare_model_outputs_fx,
    compare_weights_fx,
    compare_model_stub_fx,
)
from torch.quantization.fx.quantize import is_activation_post_process
from torch.quantization.quantize_fx import convert_fx, fuse_fx, prepare_fx
from torch.testing._internal.common_quantization import (
    ConvBnModel,
    ConvBnReLUModel,
    ConvModel,
    QuantizationTestCase,
    SingleLayerLinearDynamicModel,
    SingleLayerLinearModel,
    LSTMwithHiddenDynamicModel,
    test_only_eval_fn,
)
from torch.testing._internal.common_quantized import override_qengines


class TestGraphModeNumericSuite(QuantizationTestCase):
    @override_qengines
    def test_remove_qconfig_observer_fx(self):
        r"""Remove activation_post_process node from fx prepred model"""
        float_model = SingleLayerLinearModel()
        float_model.eval()

        qengine = torch.backends.quantized.engine
        qconfig = get_default_qconfig(qengine)

        qconfig_dict = {"": qconfig}

        prepared_model = prepare_fx(float_model, qconfig_dict)

        prepared_float_model = copy.deepcopy(prepared_model)
        prepared_float_model.eval()

        model = remove_qconfig_observer_fx(prepared_float_model)

        modules = dict(model.named_modules())
        for node in model.graph.nodes:
            if node.op == "call_module":
                self.assertFalse(is_activation_post_process(modules[node.target]))

    def compare_and_validate_model_weights_results_fx(
        self, prepared_float_model, q_model, expected_weight_dict_keys
    ):

        weight_dict = compare_weights_fx(
            prepared_float_model.state_dict(), q_model.state_dict()
        )

        self.assertTrue(weight_dict.keys() == expected_weight_dict_keys)
        self.assertEqual(len(weight_dict), 1)

        for k, v in weight_dict.items():
            self.assertTrue(v["float"].shape == v["quantized"].shape)

    @override_qengines
    def test_compare_weights_conv_static_fx(self):
        r"""Compare the weights of float and static quantized conv layer"""

        qengine = torch.backends.quantized.engine
        qconfig = get_default_qconfig(qengine)
        qconfig_dict = {"": qconfig}

        model_list = [ConvModel(), ConvBnModel(), ConvBnReLUModel()]
        for float_model in model_list:
            float_model.eval()

            fused = fuse_fx(float_model)
            prepared_model = prepare_fx(float_model, qconfig_dict)

            # Run calibration
            test_only_eval_fn(prepared_model, self.img_data_2d)
            q_model = convert_fx(prepared_model)

            expected_weight_dict_keys = {"conv.weight"}
            self.compare_and_validate_model_weights_results_fx(
                fused, q_model, expected_weight_dict_keys
            )

    @override_qengines
    def test_compare_weights_linear_static_fx(self):
        r"""Compare the weights of float and static quantized linear layer"""

        qengine = torch.backends.quantized.engine
        qconfig = get_default_qconfig(qengine)
        qconfig_dict = {"": qconfig}

        float_model = SingleLayerLinearModel()
        float_model.eval()

        prepared_model = prepare_fx(float_model, qconfig_dict)

        prepared_float_model = copy.deepcopy(prepared_model)
        prepared_float_model.eval()

        # Run calibration
        test_only_eval_fn(prepared_model, self.calib_data)
        q_model = convert_fx(prepared_model)

        expected_weight_dict_keys = {"fc1._packed_params._packed_params"}
        self.compare_and_validate_model_weights_results_fx(
            prepared_float_model, q_model, expected_weight_dict_keys
        )

    @override_qengines
    def test_compare_weights_linear_dynamic_fx(self):
        r"""Compare the weights of float and dynamic quantized linear layer"""

        qconfig_dict = {"object_type": [(nn.Linear, default_dynamic_qconfig)]}

        float_model = SingleLayerLinearDynamicModel()
        float_model.eval()

        prepared_model = prepare_fx(float_model, qconfig_dict)

        prepared_float_model = copy.deepcopy(prepared_model)
        prepared_float_model.eval()

        q_model = convert_fx(prepared_model)

        expected_weight_dict_keys = {"fc1._packed_params._packed_params"}
        self.compare_and_validate_model_weights_results_fx(
            prepared_float_model, q_model, expected_weight_dict_keys
        )

    @override_qengines
    def test_compare_weights_lstm_dynamic_fx(self):
        r"""Compare the weights of float and dynamic quantized lstm layer"""

        qconfig_dict = {"object_type": [(nn.LSTM, default_dynamic_qconfig)]}

        float_model = LSTMwithHiddenDynamicModel()
        float_model.eval()

        prepared_model = prepare_fx(float_model, qconfig_dict)

        prepared_float_model = copy.deepcopy(prepared_model)
        prepared_float_model.eval()

        q_model = convert_fx(prepared_model)

        expected_weight_dict_keys = {"lstm._all_weight_values.0.param"}
        self.compare_and_validate_model_weights_results_fx(
            prepared_float_model, q_model, expected_weight_dict_keys
        )

    # TODO: Add submodule and functional test cases for compare_model_stub_fx
    def compare_and_validate_model_stub_results_fx(
        self,
        prepared_float_model,
        q_model,
        module_swap_list,
        expected_ob_dict_keys,
        *data,
    ):
        ob_dict = compare_model_stub_fx(
            prepared_float_model, q_model, module_swap_list, *data
        )

        self.assertTrue(expected_ob_dict_keys == ob_dict.keys())
        self.assertEqual(len(ob_dict), 1)

        for k, v in ob_dict.items():
            self.assertTrue(len(v["float"]) == len(v["quantized"]))
            for i, val in enumerate(v["quantized"]):
                self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

    @override_qengines
    def test_compare_model_stub_conv_static_fx(self):
        r"""Compare the output of static quantized conv layer and its float shadow module"""

        qengine = torch.backends.quantized.engine
        qconfig = get_default_qconfig(qengine)
        qconfig_dict = {"": qconfig}

        model_list = [ConvModel(), ConvBnReLUModel()]

        for float_model in model_list:
            float_model.eval()

            prepared_model = prepare_fx(float_model, qconfig_dict)

            prepared_float_model = copy.deepcopy(prepared_model)

            # Run calibration
            test_only_eval_fn(prepared_model, self.img_data_2d)
            q_model = convert_fx(prepared_model)

            module_swap_list = [nn.Conv2d, nni.modules.fused.ConvReLU2d]

            expected_ob_dict_keys = {"conv.stats"}
            self.compare_and_validate_model_stub_results_fx(
                prepared_float_model,
                q_model,
                module_swap_list,
                expected_ob_dict_keys,
                self.img_data_2d[0][0],
            )

    @override_qengines
    def test_compare_model_stub_linear_static_fx(self):
        r"""Compare the output of static quantized linear layer and its float shadow module"""

        qengine = torch.backends.quantized.engine
        qconfig = get_default_qconfig(qengine)
        qconfig_dict = {"": qconfig}

        float_model = SingleLayerLinearModel()
        float_model.eval()

        prepared_model = prepare_fx(float_model, qconfig_dict)

        prepared_float_model = copy.deepcopy(prepared_model)

        # Run calibration
        test_only_eval_fn(prepared_model, self.calib_data)
        q_model = convert_fx(prepared_model)

        linear_data = self.calib_data[0][0]
        module_swap_list = [nn.Linear]

        expected_ob_dict_keys = {"fc1.stats"}

        self.compare_and_validate_model_stub_results_fx(
            prepared_float_model,
            q_model,
            module_swap_list,
            expected_ob_dict_keys,
            linear_data,
        )

    @override_qengines
    def test_compare_model_stub_linear_dynamic_fx(self):
        r"""Compare the output of dynamic quantized linear layer and its float shadow module"""

        qconfig_dict = {"object_type": [(nn.Linear, default_dynamic_qconfig)]}

        float_model = SingleLayerLinearDynamicModel()
        float_model.eval()

        prepared_model = prepare_fx(float_model, qconfig_dict)

        prepared_float_model = copy.deepcopy(prepared_model)
        prepared_float_model.eval()

        q_model = convert_fx(prepared_model)

        linear_data = self.calib_data[0][0]
        module_swap_list = [nn.Linear]

        expected_ob_dict_keys = {"fc1.stats"}
        self.compare_and_validate_model_stub_results_fx(
            prepared_float_model,
            q_model,
            module_swap_list,
            expected_ob_dict_keys,
            linear_data,
        )

    @override_qengines
    def test_compare_model_stub_lstm_dynamic_fx(self):
        r"""Compare the output of dynamic quantized linear layer and its float shadow module"""

        qconfig_dict = {"object_type": [(nn.LSTM, default_dynamic_qconfig)]}

        float_model = LSTMwithHiddenDynamicModel()
        float_model.eval()

        prepared_model = prepare_fx(float_model, qconfig_dict)

        prepared_float_model = copy.deepcopy(prepared_model)
        prepared_float_model.eval()

        q_model = convert_fx(prepared_model)

        module_swap_list = [nn.LSTM]

        lstm_input = torch.rand((1, 1, 2))
        lstm_hidden = (torch.rand(1, 1, 2), torch.rand(1, 1, 2))

        expected_ob_dict_keys = {"lstm.stats"}
        self.compare_and_validate_model_stub_results_fx(
            prepared_float_model,
            q_model,
            module_swap_list,
            expected_ob_dict_keys,
            lstm_input,
            lstm_hidden,
        )

    def compare_and_validate_model_outputs_results_fx(
        self, prepared_float_model, q_model, expected_act_compare_dict_keys, *data
    ):
        act_compare_dict = compare_model_outputs_fx(
            prepared_float_model, q_model, *data
        )

        self.assertTrue(act_compare_dict.keys() == expected_act_compare_dict_keys)
        for k, v in act_compare_dict.items():
            self.assertTrue(len(v["float"]) == 1)
            self.assertTrue(len(v["float"]) == len(v["quantized"]))

            for i, val in enumerate(v["quantized"]):
                if "lstm_1.stats" not in act_compare_dict:
                    self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)
                else:
                    self.assertTrue(
                        v["float"][i][0].shape == v["quantized"][i][0].shape
                    )
                    if i == 1:
                        self.assertTrue(
                            v["float"][i][1].shape == v["quantized"][i][1].shape
                        )

    @override_qengines
    def test_compare_model_outputs_conv_static_fx(self):
        r"""Compare the output of conv layer in static quantized model and corresponding
        output of conv layer in float model
        """

        qengine = torch.backends.quantized.engine
        qconfig = get_default_qconfig(qengine)
        qconfig_dict = {"": qconfig}

        model_list = [ConvModel(), ConvBnReLUModel()]

        for float_model in model_list:
            float_model.eval()
            prepared_model = prepare_fx(float_model, qconfig_dict)
            prepared_float_model = copy.deepcopy(prepared_model)

            # Run calibration
            test_only_eval_fn(prepared_model, self.img_data_2d)
            q_model = convert_fx(prepared_model)

            expected_act_compare_dict_keys = {"x.stats", "conv.stats"}
            self.compare_and_validate_model_outputs_results_fx(
                prepared_float_model,
                q_model,
                expected_act_compare_dict_keys,
                self.img_data_2d[0][0],
            )

    @override_qengines
    def test_compare_model_outputs_linear_static_fx(self):
        r"""Compare the output of linear layer in static quantized model and corresponding
        output of linear layer in float model
        """

        qengine = torch.backends.quantized.engine
        qconfig = get_default_qconfig(qengine)
        qconfig_dict = {"": qconfig}

        float_model = SingleLayerLinearModel()
        float_model.eval()

        prepared_model = prepare_fx(float_model, qconfig_dict)

        prepared_float_model = copy.deepcopy(prepared_model)

        # Run calibration
        test_only_eval_fn(prepared_model, self.calib_data)
        q_model = convert_fx(prepared_model)

        linear_data = self.calib_data[0][0]

        expected_act_compare_dict_keys = {"x.stats", "fc1.stats"}
        self.compare_and_validate_model_outputs_results_fx(
            prepared_float_model, q_model, expected_act_compare_dict_keys, linear_data
        )

    @override_qengines
    def test_compare_model_outputs_linear_dynamic_fx(self):
        r"""Compare the output of linear layer in dynamic quantized model and corresponding
        output of linear layer in float model
        """

        qconfig_dict = {"object_type": [(nn.Linear, default_dynamic_qconfig)]}

        float_model = SingleLayerLinearDynamicModel()
        float_model.eval()

        prepared_model = prepare_fx(float_model, qconfig_dict)
        prepared_float_model = copy.deepcopy(prepared_model)

        q_model = convert_fx(prepared_model)

        linear_data = self.calib_data[0][0]

        expected_act_compare_dict_keys = {"x.stats", "fc1.stats"}
        self.compare_and_validate_model_outputs_results_fx(
            prepared_float_model, q_model, expected_act_compare_dict_keys, linear_data
        )

    @override_qengines
    def test_compare_model_outputs_lstm_dynamic_fx(self):
        r"""Compare the output of LSTM layer in dynamic quantized model and corresponding
        output of linear layer in float model
        """

        qconfig_dict = {"object_type": [(nn.LSTM, default_dynamic_qconfig)]}

        float_model = LSTMwithHiddenDynamicModel()
        float_model.eval()

        prepared_model = prepare_fx(float_model, qconfig_dict)
        prepared_float_model = copy.deepcopy(prepared_model)

        q_model = convert_fx(prepared_model)

        lstm_input = torch.rand((1, 1, 2))
        lstm_hidden = (torch.rand(1, 1, 2), torch.rand(1, 1, 2))

        expected_act_compare_dict_keys = {"x.stats", "hid.stats", "lstm_1.stats"}
        self.compare_and_validate_model_outputs_results_fx(
            prepared_float_model,
            q_model,
            expected_act_compare_dict_keys,
            lstm_input,
            lstm_hidden,
        )
