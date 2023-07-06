# -*- coding: utf-8 -*-
# Owner(s): ["oncall: quantization"]

import torch
import torch._C

from torch.ao.quantization import (
    default_dynamic_qconfig,
    per_channel_dynamic_qconfig,
)

from torch.ao.quantization.quantize_jit import (
    prepare_dynamic_jit,
    convert_dynamic_jit,
    _prepare_ondevice_dynamic_jit,
    _quantize_ondevice_dynamic_jit,
)

from torch.testing._internal.common_utils import TestCase

from torch.testing._internal.common_quantization import (
    get_script_module,
    LinearAddModel,
)

from torch.jit.mobile import _load_for_lite_interpreter, LiteScriptModule

from torch.testing import FileCheck
from torch.utils import bundled_inputs as bundled_inputs

import io
from typing import Dict

class myMod(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 5).float()
        self.fc1.weight = weight
        self.fc2 = torch.nn.Linear(5, 5).float()

    def forward(self, x):
        return self.fc2(self.fc1(x))


class MyConvLinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3)
        weight = torch.nn.Parameter(torch.ones(5, 5))
        self.weight1 = torch.nn.Parameter(torch.ones(5, 5))
        self.mymod = myMod(weight)

    def forward(self, x):
        conv_output = self.conv(x)
        y = self.mymod(conv_output)
        z = torch.nn.functional.linear(y, self.weight1)
        return z

    def get_example_inputs(self):
        return (torch.rand(1, 3, 12, 7),)


class OnDevicePTQUtils:
    observer_module_name = ['MinMaxObserver', 'PerChannelMinMaxObserver']

    @staticmethod
    def insert_observers(model, qconfig_dict):
        inputs = model.get_example_inputs()
        scripted_model = get_script_module(model, False, inputs)
        scripted_model = _prepare_ondevice_dynamic_jit(scripted_model, qconfig_dict)
        return scripted_model

    @staticmethod
    def ptq_dynamic_quantize(model, qconfig_dict):
        inputs = model.get_example_inputs()
        m = get_script_module(model, False, inputs)
        m = _quantize_ondevice_dynamic_jit(m, qconfig_dict, 'forward', True)
        return m

    @staticmethod
    def find_observer_modules(m):
        observer_modules = []
        for child_module in m.children():
            if child_module.original_name in OnDevicePTQUtils.observer_module_name:
                observer_modules.append(child_module)
        return observer_modules

    @staticmethod
    def is_value_type_observer(value):
        type_name = value.type()
        for observer_type in OnDevicePTQUtils.observer_module_name:
            if observer_type in type_name.str():
                return True
        return False

    @staticmethod
    def is_calculate_qparam(node):
        if node.kind() == "prim::CallMethod":
            if node.s('name') == "calculate_qparams":
                return True
        return False

    @staticmethod
    def get_linear_packed_param_fp_weight(node):
        weight = node.inputsAt(0).node()
        if weight.kind() != "aten::quantize_per_tensor" and weight.kind() != "aten::quantize_per_channel":
            raise ValueError("Quantized weight must be produced.")
        fp_weight = weight.inputsAt(0).node()
        assert fp_weight.kind() == "prim::GetAttr", "Weight must be an attribute of the module."
        fp_weight_name = fp_weight.s('name')
        return fp_weight_name

    @staticmethod
    def is_per_channel_quantized_packed_param(node):
        assert node.kind() == 'quantized::linear_prepack', "Node must corresponds to linear_prepack."
        weight = node.inputsAt(0).node()
        assert weight.kind() != "aten::quantize_per_tensor" or weight.kind() != "aten::quantize_per_channel"
        return weight.kind() != "aten::quantize_per_tensor"


class TestOnDeviceDynamicPTQInsertObservers(TestCase):
    def _check_num_and_type_of_observers(self, model, num_observers):
        qconfig_dict = {"": default_dynamic_qconfig}
        scripted_model = OnDevicePTQUtils.insert_observers(model, qconfig_dict)
        observer_modules = OnDevicePTQUtils.find_observer_modules(scripted_model)
        self.assertTrue(len(observer_modules) == num_observers)
        for observer in observer_modules:
            self.assertTrue(observer.original_name == 'MinMaxObserver')

        qconfig_dict = {"": per_channel_dynamic_qconfig}
        scripted_model = OnDevicePTQUtils.insert_observers(model, qconfig_dict)
        observer_modules = OnDevicePTQUtils.find_observer_modules(scripted_model)
        self.assertTrue(len(observer_modules) == num_observers)
        for observer in observer_modules:
            self.assertTrue(observer.original_name == 'PerChannelMinMaxObserver')

    def _check_observer_method(self, model, num_observers):
        qconfig_dict = {"": default_dynamic_qconfig}
        inputs = model.get_example_inputs()
        orig_scripted_model = get_script_module(model, False, inputs)
        torch._C._jit_pass_inline(orig_scripted_model.graph)
        orig_forward_graph = orig_scripted_model.graph.str()
        scripted_model = OnDevicePTQUtils.insert_observers(model, qconfig_dict)
        quant_forward_graph = scripted_model.graph.str()
        # exact graph matching is difficult so just resorting to # of lines
        # instead of implementing graph matching
        self.assertEqual(len(orig_forward_graph.splitlines()), len(quant_forward_graph.splitlines()))
        observe_method = scripted_model.observe_forward.graph
        FileCheck().check_count("prim::CallMethod[name=\"forward\"](%_observer",
                                num_observers, exactly=True).run(observe_method)
        reset_observers_method = scripted_model.reset_observers_forward.graph
        FileCheck().check_count(
            "prim::CallMethod[name=\"reset_min_max_vals\"](%_observer", num_observers, exactly=True).run(reset_observers_method)

    def _observer_is_weight_only(self, node):
        if (node.kind() == "prim::CallMethod") and node.s("name") == "forward":
            if (OnDevicePTQUtils.is_value_type_observer(node.inputsAt(0))):
                return (node.inputsAt(1).node().kind() == "prim::GetAttr")
        return False

    def test_num_observers(self):
        model = LinearAddModel()
        self._check_num_and_type_of_observers(model, 2)
        model = MyConvLinearModule()
        self._check_num_and_type_of_observers(model, 3)

    def test_observe_method(self):
        model = MyConvLinearModule()
        self._check_observer_method(model, 3)

    def test_weight_only_observers(self):
        model = MyConvLinearModule()
        qconfig_dict = {"": default_dynamic_qconfig}
        inputs = model.get_example_inputs()
        scripted_model = OnDevicePTQUtils.insert_observers(model, qconfig_dict)
        observe_forward_graph = scripted_model.observe_forward.graph
        num_weight_only_observers = 0
        for node in observe_forward_graph.nodes():
            if (self._observer_is_weight_only(node)):
                num_weight_only_observers += 1
        self.assertEqual(num_weight_only_observers, 3)


class TestOnDeviceDynamicPTQInsertQuantDequant(TestCase):
    def _validate_quant_dequant_nodes(self, model, num_nodes, per_channel=0):
        quantize_forward_graph = model.quantize_forward.graph
        quantize_per_tensor = quantize_per_channel = 0
        for n in quantize_forward_graph.nodes():
            if "aten::quantize_per_tensor" in n.kind():
                quantize_per_tensor += 1
            if "aten::quantize_per_channel" in n.kind():
                quantize_per_channel += 1
        self.assertEqual(quantize_per_tensor + quantize_per_channel, num_nodes)

    def _validate_calculate_qparams(self, model, num_nodes):
        quantize_forward_graph = model.quantize_forward.graph
        num_calculate_qparams = 0
        for n in quantize_forward_graph.nodes():
            if OnDevicePTQUtils.is_calculate_qparam(n):
                num_calculate_qparams += 1
        self.assertEqual(num_calculate_qparams, num_nodes)

    def _validate_no_observer_forward(self, model):
        quantize_forward_graph = model.quantize_forward.graph
        for n in quantize_forward_graph.nodes():
            if (n.kind() == "prim::CallMethod") and n.s("name") == "forward":
                if (OnDevicePTQUtils.is_value_type_observer(n.inputsAt(0))):
                    return False
        return True

    def _check_quant_dequant_and_calc_qparams(self, model, num_nodes):
        qconfig_dict = {"" : default_dynamic_qconfig}
        m = OnDevicePTQUtils.ptq_dynamic_quantize(model, qconfig_dict)
        self._validate_quant_dequant_nodes(m, num_nodes)
        self._validate_calculate_qparams(m, num_nodes)
        self._validate_no_observer_forward(m)

        qconfig_dict = {"" : per_channel_dynamic_qconfig}
        m = OnDevicePTQUtils.ptq_dynamic_quantize(model, qconfig_dict)
        self._validate_quant_dequant_nodes(m, num_nodes, num_nodes)
        self._validate_calculate_qparams(m, num_nodes)
        self._validate_no_observer_forward(m)

    def _check_quantize_forward_runs(self, model):
        inputs = model.get_example_inputs()
        qconfig_dict = {"" : default_dynamic_qconfig}
        m = OnDevicePTQUtils.ptq_dynamic_quantize(model, qconfig_dict)
        m.observe_forward(*inputs)
        m.quantize_forward(*inputs)

        qconfig_dict = {"" : per_channel_dynamic_qconfig}
        m = OnDevicePTQUtils.ptq_dynamic_quantize(model, qconfig_dict)
        # First must run observe forward to record the stats to produce
        # correct scales and zero points
        m.observe_forward(*inputs)
        m.quantize_forward(*inputs)

    def test_num_quant_dequant_nodes(self):
        model = LinearAddModel()
        self._check_quant_dequant_and_calc_qparams(model, 2)
        model = MyConvLinearModule()
        self._check_quant_dequant_and_calc_qparams(model, 3)

    def test_quantize_forward_runs(self):
        model = LinearAddModel()
        self._check_quantize_forward_runs(model)
        model = MyConvLinearModule()
        self._check_quantize_forward_runs(model)


class TestOnDeviceDynamicPTQFinalize(TestCase):
    def _validate_packed_params(self, model, num_nodes, per_channel=0):
        quantize_forward_graph = model.quantize_forward.graph
        quantize_per_tensor = quantize_per_channel = 0
        linear_prepack = 0
        linear_prepack_uses = 0
        for n in quantize_forward_graph.nodes():
            if n.kind() == 'prim::SetAttr':
                maybe_packed_param_value = n.inputsAt(1)
                maybe_packed_param = maybe_packed_param_value.node()
                if maybe_packed_param.kind() == 'quantized::linear_prepack':
                    linear_prepack += 1
                    linear_prepack_uses += len(maybe_packed_param_value.uses())
                    if OnDevicePTQUtils.is_per_channel_quantized_packed_param(maybe_packed_param):
                        quantize_per_channel += 1
                    else:
                        quantize_per_tensor += 1
        self.assertEqual(quantize_per_tensor + quantize_per_channel, num_nodes)
        self.assertEqual(quantize_per_channel, per_channel)
        self.assertEqual(linear_prepack, num_nodes)
        self.assertEqual(linear_prepack_uses, num_nodes)


    def _validate_no_linear_unpack(self, model):
        quantize_forward_graph = model.quantize_forward.graph
        for n in quantize_forward_graph.nodes():
            if n.kind() == 'quantized::linear_unpack':
                return False
        return True


    def _validate_setattr_fp_weights(self, model, num_nodes):
        quantize_forward_graph = model.quantize_forward.graph
        fp_weights_setattr = 0
        fp_weight_names = []
        for n in quantize_forward_graph.nodes():
            if n.kind() == 'prim::SetAttr':
                maybe_packed_param = n.inputsAt(1).node()
                if maybe_packed_param.kind() == 'quantized::linear_prepack':
                    weight_name = OnDevicePTQUtils.get_linear_packed_param_fp_weight(maybe_packed_param)
                    fp_weight_names.append(weight_name)

        for n in quantize_forward_graph.nodes():
            # This is basically detecting
            # %x = prim::Constant
            # = prim::SetAttr(<weight_name>)(module_value, x)
            # Thus making sure that the original fp weights are
            # reset
            if n.kind() == 'prim::SetAttr':
                weight_name = n.s('name')
                if weight_name in fp_weight_names:
                    maybe_constant = n.inputsAt(1).node()
                    if maybe_constant.kind() == 'prim::Constant':
                        fp_weights_setattr += 1
        self.assertEqual(fp_weights_setattr, num_nodes)


    def _validate_quantized_forward(self, model, num_nodes):
        quantized_forward_graph = model.quantized_forward.graph
        quantize_per_tensor = quantize_per_channel = 0
        quantized_linear_dynamic = 0
        linear_packed_params = 0
        num_setattr = 0
        for n in quantized_forward_graph.nodes():
            if "aten::quantize_per_tensor" in n.kind():
                quantize_per_tensor += 1
            if "aten::quantize_per_channel" in n.kind():
                quantize_per_channel += 1
            if "quantized::linear_dynamic" in n.kind():
                quantized_linear_dynamic += 1
            if n.kind() == 'prim::GetAttr':
                output = n.outputsAt(0)
                output_type = output.type()
                if "LinearPackedParamsBase" in output_type.str():
                    linear_packed_params += 1
            if n.kind() == 'prim::SetAttr':
                num_setattr += 1
        self.assertEqual(quantize_per_tensor, 0)
        self.assertEqual(quantize_per_channel, 0)
        self.assertEqual(quantized_linear_dynamic, num_nodes)
        self.assertEqual(linear_packed_params, num_nodes)
        # self.assertEqual(num_setattr, 0)


    def _check_quantize_forward(self, model, num_nodes):
        qconfig_dict = {"" : default_dynamic_qconfig}
        m = OnDevicePTQUtils.ptq_dynamic_quantize(model, qconfig_dict)
        self._validate_packed_params(m, num_nodes)
        self._validate_no_linear_unpack(m)
        self._validate_setattr_fp_weights(m, num_nodes)

        qconfig_dict = {"" : per_channel_dynamic_qconfig}
        m = OnDevicePTQUtils.ptq_dynamic_quantize(model, qconfig_dict)
        self._validate_packed_params(m, num_nodes, num_nodes)
        self._validate_no_linear_unpack(m)
        self._validate_setattr_fp_weights(m, num_nodes)


    def _check_quantized_forward(self, model, num_nodes):
        qconfig_dict = {"" : default_dynamic_qconfig}
        m = OnDevicePTQUtils.ptq_dynamic_quantize(model, qconfig_dict)
        self._validate_quantized_forward(m, num_nodes)

        qconfig_dict = {"" : per_channel_dynamic_qconfig}
        m = OnDevicePTQUtils.ptq_dynamic_quantize(model, qconfig_dict)
        self._validate_quantized_forward(m, num_nodes)


    def _check_against_ref_dynamic_ptq(self, model):
        model.eval()
        inputs = model.get_example_inputs()
        ref_m = torch.jit.script(model)
        torch._C._jit_pass_inline(ref_m.graph)
        qconfig_dict = {"" : default_dynamic_qconfig}
        ref_m = prepare_dynamic_jit(ref_m, qconfig_dict)
        ref_m = convert_dynamic_jit(ref_m)
        ref_output = ref_m(*inputs)

        m = OnDevicePTQUtils.ptq_dynamic_quantize(model, qconfig_dict)
        m.observe_forward(*inputs)
        m.quantize_forward(*inputs)
        output = m.quantized_forward(*inputs)
        self.assertTrue(torch.allclose(ref_output, output))
        thrown = False
        try:
            m(*inputs)
        except Exception as e:
            thrown = True
        self.assertTrue(thrown)

        # test with per channel quant
        ref_m = torch.jit.script(model)
        torch._C._jit_pass_inline(ref_m.graph)
        qconfig_dict = {"" : per_channel_dynamic_qconfig}
        ref_m = prepare_dynamic_jit(ref_m, qconfig_dict)
        ref_m = convert_dynamic_jit(ref_m)
        ref_output = ref_m(*inputs)

        m = OnDevicePTQUtils.ptq_dynamic_quantize(model, qconfig_dict)
        m.observe_forward(*inputs)
        m.quantize_forward(*inputs)
        output = m.quantized_forward(*inputs)
        self.assertTrue(torch.allclose(ref_output, output))
        thrown = False
        try:
            m(*inputs)
        except Exception as e:
            thrown = True
        self.assertTrue(thrown)


    def _check_serdes_and_device_side_api_helper(self, model, check_device_side_api=False):
        model.eval()
        inputs = model.get_example_inputs()
        ref_m = torch.jit.script(model)
        torch._C._jit_pass_inline(ref_m.graph)
        qconfig_dict = {"" : default_dynamic_qconfig}
        ref_m = prepare_dynamic_jit(ref_m, qconfig_dict)
        ref_m = convert_dynamic_jit(ref_m)
        buffer = io.BytesIO()
        torch.jit.save(ref_m, buffer)
        buffer.seek(0)
        ref_m = torch.jit.load(buffer)
        ref_output = ref_m(*inputs)

        if not check_device_side_api:
            m = OnDevicePTQUtils.ptq_dynamic_quantize(model, qconfig_dict)
            buffer = io.BytesIO()
            torch.jit.save(m, buffer)
            buffer.seek(0)
            m = torch.jit.load(buffer)
            m.reset_observers_forward()
            m.observe_forward(*inputs)
            m.quantize_forward(*inputs)
            output = m.quantized_forward(*inputs)
            self.assertTrue(torch.allclose(ref_output, output))
        else:
            # check for lite interpreter
            m = OnDevicePTQUtils.ptq_dynamic_quantize(model, qconfig_dict)
            first_input, = inputs
            rand_input = bundled_inputs.bundle_randn(first_input.size(), dtype=first_input.dtype)
            m = bundled_inputs.bundle_inputs(m, inputs=[(rand_input, )])
            buffer = io.BytesIO(m._save_to_buffer_for_lite_interpreter())
            buffer.seek(0)
            m = _load_for_lite_interpreter(buffer)  # Error here
            torch._C._quantize_ondevice_ptq_dynamic(m._c, "forward")
            self.assertFalse(m.find_method("quantized_forward"))
            self.assertFalse(m.find_method("quantize_forward"))
            self.assertFalse(m.find_method("observe_forward"))
            self.assertFalse(m.find_method("reset_observers_forward"))
            output = m(*inputs)
            self.assertTrue(torch.allclose(ref_output, output))

            # Now serialize to flabuffer and load from fb and check
            dict: Dict[str, str] = {}
            bytes = torch._C._save_mobile_module_to_bytes(m._c, dict)
            m = LiteScriptModule(torch._C._load_mobile_module_from_bytes(bytes))
            fb_output = m(*inputs)
            self.assertTrue(torch.allclose(ref_output, fb_output))

        model.eval()
        inputs = model.get_example_inputs()
        ref_m = torch.jit.script(model)
        torch._C._jit_pass_inline(ref_m.graph)
        qconfig_dict = {"" : per_channel_dynamic_qconfig}
        ref_m = prepare_dynamic_jit(ref_m, qconfig_dict)
        ref_m = convert_dynamic_jit(ref_m)
        buffer = io.BytesIO()
        torch.jit.save(ref_m, buffer)
        buffer.seek(0)
        ref_m = torch.jit.load(buffer)
        ref_output = ref_m(*inputs)

        if not check_device_side_api:
            m = OnDevicePTQUtils.ptq_dynamic_quantize(model, qconfig_dict)
            buffer = io.BytesIO()
            torch.jit.save(m, buffer)
            buffer.seek(0)
            m = torch.jit.load(buffer)
            m.reset_observers_forward()
            m.observe_forward(*inputs)
            m.quantize_forward(*inputs)
            output = m.quantized_forward(*inputs)
            self.assertTrue(torch.allclose(ref_output, output))
        else:
            # check for lite interpreter
            m = OnDevicePTQUtils.ptq_dynamic_quantize(model, qconfig_dict)
            first_input, = inputs
            rand_input = bundled_inputs.bundle_randn(first_input.size(), dtype=first_input.dtype)
            m = bundled_inputs.bundle_inputs(m, inputs=[(rand_input, )])
            buffer = io.BytesIO(m._save_to_buffer_for_lite_interpreter())
            buffer.seek(0)
            m = _load_for_lite_interpreter(buffer)  # Error here
            torch._C._quantize_ondevice_ptq_dynamic(m._c, "forward")
            self.assertFalse(m.find_method("quantized_forward"))
            self.assertFalse(m.find_method("quantize_forward"))
            self.assertFalse(m.find_method("observe_forward"))
            self.assertFalse(m.find_method("reset_observers_forward"))
            output = m(*inputs)
            self.assertTrue(torch.allclose(ref_output, output))


    def _check_serialization_deserialization(self, model):
        self._check_serdes_and_device_side_api_helper(model, False)


    def _check_device_side_api(self, model):
        self._check_serdes_and_device_side_api_helper(model, True)


    def test_quantize_forward(self):
        model = LinearAddModel()
        self._check_quantize_forward(model, 2)
        model = MyConvLinearModule()
        self._check_quantize_forward(model, 3)


    def test_quantized_forward(self):
        model = LinearAddModel()
        self._check_quantized_forward(model, 2)
        model = MyConvLinearModule()
        self._check_quantized_forward(model, 3)


    def test_against_offdevice_dynamic_ptq(self):
        model = LinearAddModel()
        self._check_against_ref_dynamic_ptq(model)
        model = MyConvLinearModule()
        self._check_against_ref_dynamic_ptq(model)


    def test_serialization_deserialization(self):
        model = MyConvLinearModule()
        self._check_serialization_deserialization(model)


    def test_device_side_api(self):
        model = MyConvLinearModule()
        self._check_device_side_api(model)
