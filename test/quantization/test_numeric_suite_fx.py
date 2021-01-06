import copy

import torch
import torch.nn as nn
from torch.quantization import get_default_qconfig
from torch.quantization._numeric_suite_fx import (
    compare_weights_fx,
    remove_qconfig_observer_fx,
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
    skipIfNoFBGEMM,
)


@skipIfNoFBGEMM
class TestGraphModeNumericSuite(QuantizationTestCase):
    def test_remove_qconfig_observer_fx(self):
        r"""Remove activation_post_process node from fx prepred model"""
        float_model = SingleLayerLinearModel()
        float_model.eval()

        qengine = torch.backends.quantized.engine
        qconfig = get_default_qconfig(qengine)

        qconfig_dict = {"": qconfig}

        prepared_model = prepare_fx(float_model, qconfig_dict)

        backup_prepared_model = copy.deepcopy(prepared_model)
        backup_prepared_model.eval()

        model = remove_qconfig_observer_fx(backup_prepared_model)

        modules = dict(model.named_modules())
        for node in model.graph.nodes:
            if node.op == "call_module":
                self.assertFalse(is_activation_post_process(modules[node.target]))

    @skipIfNoFBGEMM
    def test_compare_weights_conv_static_fx(self):
        r"""Compare the weights of float and static quantized conv layer"""

        def calibrate(model, calib_data):
            model.eval()
            with torch.no_grad():
                for inp in calib_data:
                    model(*inp)

        def compare_and_validate_results(float_model, q_model):
            weight_dict = compare_weights_fx(
                float_model.state_dict(), q_model.state_dict()
            )
            self.assertEqual(len(weight_dict), 1)
            for k, v in weight_dict.items():
                self.assertTrue(v["float"].shape == v["quantized"].shape)

        qengine = torch.backends.quantized.engine
        qconfig = get_default_qconfig(qengine)
        qconfig_dict = {"": qconfig}

        model_list = [ConvModel(), ConvBnModel(), ConvBNReLU()]
        for float_model in model_list:
            float_model.eval()

            fused = fuse_fx(float_model)
            prepared_model = prepare_fx(float_model, qconfig_dict)

            # Run calibration
            calibrate(prepared_model, self.img_data_2d)
            q_model = convert_fx(prepared_model)

            compare_and_validate_results(fused, q_model)

    @skipIfNoFBGEMM
    def test_compare_weights_linear_static_fx(self):
        r"""Compare the weights of float and static quantized linear layer"""

        def calibrate(model, calib_data):
            model.eval()
            with torch.no_grad():
                for inp in calib_data:
                    model(*inp)

        def compare_and_validate_results(float_model, q_model):
            weight_dict = compare_weights_fx(
                float_model.state_dict(), q_model.state_dict()
            )
            self.assertEqual(len(weight_dict), 1)
            for k, v in weight_dict.items():
                self.assertTrue(v["float"].shape == v["quantized"].shape)

        float_model = SingleLayerLinearModel()
        float_model.eval()

        qengine = torch.backends.quantized.engine
        qconfig = get_default_qconfig(qengine)
        qconfig_dict = {"": qconfig}

        prepared_model = prepare_fx(float_model, qconfig_dict)

        backup_prepared_model = copy.deepcopy(prepared_model)
        backup_prepared_model.eval()

        # Run calibration
        calibrate(prepared_model, self.calib_data)
        q_model = convert_fx(prepared_model)

        compare_and_validate_results(backup_prepared_model, q_model)

    @skipIfNoFBGEMM
    def test_compare_weights_linear_dynamic_fx(self):
        r"""Compare the weights of float and dynamic quantized linear layer"""

        def compare_and_validate_results(float_model, q_model):
            weight_dict = compare_weights_fx(
                float_model.state_dict(), q_model.state_dict()
            )
            self.assertEqual(len(weight_dict), 1)
            for k, v in weight_dict.items():
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

        float_model = SingleLayerLinearDynamicModel()
        float_model.eval()

        qconfig = torch.quantization.qconfig.default_dynamic_qconfig
        qconfig_dict = {"": qconfig}

        prepared_model = prepare_fx(float_model, qconfig_dict)

        backup_prepared_model = copy.deepcopy(prepared_model)
        backup_prepared_model.eval()

        q_model = convert_fx(prepared_model)

        compare_and_validate_results(backup_prepared_model, q_model)

    @skipIfNoFBGEMM
    def test_compare_model_stub_conv_static_fx(self):
        r"""Compare the output of static quantized conv layer and its float shadow module"""

        def calibrate(model, calib_data):
            model.eval()
            with torch.no_grad():
                for inp in calib_data:
                    model(*inp)

        def compare_and_validate_results(float_model, q_model, module_swap_list, data):
            ob_dict = compare_model_stub_fx(
                float_model, q_model, module_swap_list, data
            )
            expected_ob_dict_keys = {"conv.stats"}
            print(ob_dict.keys())
            self.assertTrue(ob_dict.keys() == expected_ob_dict_keys)
            self.assertEqual(len(ob_dict), 1)
            for k, v in ob_dict.items():
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

        qengine = torch.backends.quantized.engine
        qconfig = get_default_qconfig(qengine)
        qconfig_dict = {"": qconfig}

        model_list = [ConvModel(), ConvBnReLUModel()]

        for float_model in model_list:
            float_model.eval()

            prepared_model = prepare_fx(float_model, qconfig_dict)

            backup_prepared_model = copy.deepcopy(prepared_model)

            # Run calibration
            calibrate(prepared_model, self.img_data_2d)
            q_model = convert_fx(prepared_model)

            module_swap_list = [nn.Conv2d, nn.intrinsic.modules.fused.ConvReLU2d]
            compare_and_validate_results(
                backup_prepared_model,
                q_model,
                module_swap_list,
                self.img_data_2d[0][0],
            )

    @skipIfNoFBGEMM
    def test_compare_model_stub_linear_static_fx(self):
        r"""Compare the output of static quantized linear layer and its float shadow module"""

        def calibrate(model, calib_data):
            model.eval()
            with torch.no_grad():
                for inp in calib_data:
                    model(*inp)

        def compare_and_validate_results(float_model, q_model, module_swap_list, data):
            ob_dict = compare_model_stub_fx(
                float_model, q_model, module_swap_list, data
            )
            expected_ob_dict_keys = {"fc1.stats"}

            self.assertTrue(ob_dict.keys() == expected_ob_dict_keys)
            self.assertEqual(len(ob_dict), 1)
            for k, v in ob_dict.items():
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

        float_model = SingleLayerLinearModel()
        float_model.eval()

        qengine = torch.backends.quantized.engine
        qconfig = get_default_qconfig(qengine)
        qconfig_dict = {"": qconfig}

        prepared_model = prepare_fx(float_model, qconfig_dict)

        backup_prepared_model = copy.deepcopy(prepared_model)

        # Run calibration
        calibrate(prepared_model, self.calib_data)
        q_model = convert_fx(prepared_model)

        linear_data = self.calib_data[0][0]
        module_swap_list = [nn.Linear]

        compare_and_validate_results(
            backup_prepared_model, q_model, module_swap_list, linear_data
        )
