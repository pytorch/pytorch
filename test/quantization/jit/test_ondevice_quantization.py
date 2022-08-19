# -*- coding: utf-8 -*-
# Owner(s): ["oncall: quantization"]

import torch

from torch.ao.quantization import (
    default_dynamic_qconfig,
    per_channel_dynamic_qconfig,
)

from torch.ao.quantization.quantize_jit import (
    _prepare_ondevice_dynamic_jit,
    _convert_ondevice_dynamic_jit,
)

from torch.testing._internal.common_utils import TestCase

from torch.testing._internal.common_quantization import (
    get_script_module,
    LinearAddModel,
)

from torch.testing import FileCheck


class myMod(torch.nn.Module):
    def __init__(self, weight):
        super(myMod, self).__init__()
        self.fc1 = torch.nn.Linear(5, 5).float()
        self.fc1.weight = weight
        self.fc2 = torch.nn.Linear(5, 5).float()

    def forward(self, x):
        return self.fc2(self.fc1(x))


class MyConvLinearModule(torch.nn.Module):
    def __init__(self):
        super(MyConvLinearModule, self).__init__()
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


class OnDevicePTQUtils(object):
    observer_module_name = ['MinMaxObserver', 'PerChannelMinMaxObserver']

    @staticmethod
    def insert_observers(model, qconfig_dict):
        inputs = model.get_example_inputs()
        scripted_model = get_script_module(model, False, inputs)
        scripted_model = _prepare_ondevice_dynamic_jit(scripted_model, qconfig_dict)
        return scripted_model

    @staticmethod
    def insert_observers_quant_dequant(model, qconfig_dict):
        inputs = model.get_example_inputs()
        m = get_script_module(model, False, inputs)
        m = _prepare_ondevice_dynamic_jit(m, qconfig_dict)
        m = _convert_ondevice_dynamic_jit(m, 'forward', True, False)
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
        quantize_per_tensor = quantize_per_channel = dequantize = 0
        for n in quantize_forward_graph.nodes():
            if "aten::quantize_per_tensor" in n.kind():
                quantize_per_tensor += 1
            if "aten::quantize_per_channel" in n.kind():
                quantize_per_channel += 1
            if "aten::dequantize" in n.kind():
                dequantize += 1
        self.assertEqual(quantize_per_tensor + quantize_per_channel, dequantize)
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
        qconfig_dict = {"": default_dynamic_qconfig}
        m = OnDevicePTQUtils.insert_observers_quant_dequant(model, qconfig_dict)
        self._validate_quant_dequant_nodes(m, num_nodes)
        self._validate_calculate_qparams(m, num_nodes)
        self._validate_no_observer_forward(m)

        qconfig_dict = {"": per_channel_dynamic_qconfig}
        m = OnDevicePTQUtils.insert_observers_quant_dequant(model, qconfig_dict)
        self._validate_quant_dequant_nodes(m, num_nodes, num_nodes)
        self._validate_calculate_qparams(m, num_nodes)
        self._validate_no_observer_forward(m)

    def _check_quantize_forward_runs(self, model):
        inputs = model.get_example_inputs()
        qconfig_dict = {"": default_dynamic_qconfig}
        m = OnDevicePTQUtils.insert_observers_quant_dequant(model, qconfig_dict)
        m.observe_forward(*inputs)
        m.quantize_forward(*inputs)

        qconfig_dict = {"": per_channel_dynamic_qconfig}
        m = OnDevicePTQUtils.insert_observers_quant_dequant(model, qconfig_dict)
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
