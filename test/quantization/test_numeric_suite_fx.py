import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.intrinsic as nni
from torch.quantization import get_default_qconfig, default_dynamic_qconfig
import torch.nn.quantized as nnq
toq = torch.ops.quantized
from torch.quantization._numeric_suite_fx import (
    remove_qconfig_observer_fx,
    compare_model_outputs_fx,
    compare_weights_fx,
    compare_model_stub_fx,
)
from torch.quantization.fx.quantize import is_activation_post_process
from torch.quantization.quantize_fx import (
    convert_fx,
    fuse_fx,
    prepare_fx,
    prepare_qat_fx,
)
from torch.testing._internal.common_quantization import (
    ConvBnModel,
    ConvBnReLUModel,
    ConvModel,
    QuantizationTestCase,
    SingleLayerLinearDynamicModel,
    SingleLayerLinearModel,
    LSTMwithHiddenDynamicModel,
    SparseNNModel,
    skip_if_no_torchvision,
    test_only_eval_fn,
)
from torch.testing._internal.common_quantization import NodeSpec as ns
from torch.testing._internal.common_quantized import override_qengines
from torch.quantization.ns.graph_matcher import (
    get_matching_subgraph_pairs,
    GraphMatchingException,
)
from torch.quantization.ns.numeric_suite_core_apis_fx import (
    compare_weights,
    prepare_model_outputs,
    OutputLogger,
    prepare_model_with_stubs,
    get_matching_activations,
    get_matching_activations_a_shadows_b,
)


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
                if "lstm.stats" not in act_compare_dict:
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

        expected_act_compare_dict_keys = {"x.stats", "hid.stats", "lstm.stats"}
        self.compare_and_validate_model_outputs_results_fx(
            prepared_float_model,
            q_model,
            expected_act_compare_dict_keys,
            lstm_input,
            lstm_hidden,
        )

class TestFXGraphMatcher(QuantizationTestCase):

    @override_qengines
    def test_simple_mod(self):
        m = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()
        mp = prepare_fx(m, {'': torch.quantization.default_qconfig})
        # TODO(future PR): prevent the need for copying here, we can copy the
        # modules but should reuse the underlying tensors
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = get_matching_subgraph_pairs(mp, mq)

        expected_types = {
            'base_op_torch.nn.Conv2d_0':
                ((nn.Conv2d, nn.Conv2d), (nnq.Conv2d, nnq.Conv2d)),
        }
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)

    @override_qengines
    def test_simple_fun(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.Tensor(1, 4))
                self.b = nn.Parameter(torch.zeros(1))
                torch.nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

            def forward(self, x):
                return F.linear(x, self.w, self.b)

        m = M().eval()
        mp = prepare_fx(m, {'': torch.quantization.default_qconfig})
        # TODO(future PR): prevent the need for copying here, we can copy the
        # modules but should reuse the underlying tensors
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = get_matching_subgraph_pairs(mp, mq)

        expected_types = {
            'base_op_torch.nn.functional.linear_0':
                ((F.linear, F.linear), (toq.linear, toq.linear))
        }
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)

    @override_qengines
    def test_simple_fusion(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.Tensor(4, 1))
                self.b = nn.Parameter(torch.zeros(4))
                torch.nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

            def forward(self, x):
                x = F.linear(x, self.w, self.b)
                x = F.relu(x)
                return x

        m = M().eval()
        mp = prepare_fx(m, {'': torch.quantization.default_qconfig})
        # TODO(future PR): prevent the need for copying here, we can copy the
        # modules but should reuse the underlying tensors
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = get_matching_subgraph_pairs(mp, mq)

        expected_types = {
            'base_op_torch.nn.functional.linear_0':
                ((F.linear, F.relu), (toq.linear_relu, toq.linear_relu)),
        }
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)

    @override_qengines
    def test_simple_mod_multi(self):
        m = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 1, 1),
            ),
            nn.Conv2d(1, 1, 1),
        ).eval()
        mp = prepare_fx(m, {'': torch.quantization.default_qconfig})
        # TODO(future PR): prevent the need for copying here, we can copy the
        # modules but should reuse the underlying tensors
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        # assume success if no exceptions
        results = get_matching_subgraph_pairs(mp, mq)

    @override_qengines
    def test_simple_tensor_ops(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = x + y
                return z

        m = M().eval()
        mp = prepare_fx(m, {'': torch.quantization.default_qconfig})
        # TODO(future PR): prevent the need for copying here, we can copy the
        # modules but should reuse the underlying tensors
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        # assume success if no exceptions
        results = get_matching_subgraph_pairs(mp, mq)

    @override_qengines
    def test_matching_failure_node_count(self):
        # verify that matching graphs with matching node types but
        # different counts of matchable nodes fails
        m1 = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()
        m2 = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Conv2d(1, 1, 1)).eval()
        mp1 = prepare_fx(m1, {'': torch.quantization.default_qconfig})
        mp2 = prepare_fx(m2, {'': torch.quantization.default_qconfig})
        with self.assertRaises(GraphMatchingException) as ex:
            results = get_matching_subgraph_pairs(mp1, mp2)

    @override_qengines
    def test_matching_failure_node_type(self):
        # verify that matching graphs with non-matching node types fails
        m1 = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()
        m2 = nn.Sequential(nn.Linear(1, 1)).eval()
        mp1 = prepare_fx(m1, {'': torch.quantization.default_qconfig})
        mp2 = prepare_fx(m2, {'': torch.quantization.default_qconfig})
        with self.assertRaises(GraphMatchingException) as ex:
            results = get_matching_subgraph_pairs(mp1, mp2)

    @override_qengines
    def test_nodes_before_cat(self):
        # verify that nodes before cat get matched
        class M(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x0):
                x1 = torch.add(x0, 1.0)
                y1 = torch.add(x0, 1.0)
                x2 = torch.cat([x1, y1])
                return x2

        m = M().eval()
        mp = prepare_fx(m, {'': torch.quantization.default_qconfig})
        # TODO(future PR): prevent the need for copying here, we can copy the
        # modules but should reuse the underlying tensors
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = get_matching_subgraph_pairs(mp, mq)

        expected_types = {
            'base_op_torch.cat_0': ((torch.cat, torch.cat), (toq.cat, toq.cat)),
            'base_op_torch.add_0': ((torch.add, torch.add), (toq.add, toq.add)),
            'base_op_torch.add_1': ((torch.add, torch.add), (toq.add, toq.add)),
        }
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)

    @override_qengines
    def test_dict_return_type(self):
        # verify that we can traverse up nodes which return dictionaries
        class M(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x0):
                x1 = torch.add(x0, 1.0)
                y1 = torch.add(x0, 1.0)
                z1 = torch.add(x0, 1.0)
                a1 = {'x1': x1, 'y1': (y1,), 'z1': [{'key': (z1,)}]}
                return a1

        m = M().eval()
        mp = prepare_fx(m, {'': torch.quantization.default_qconfig})
        # TODO(future PR): prevent the need for copying here, we can copy the
        # modules but should reuse the underlying tensors
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = get_matching_subgraph_pairs(mp, mq)

        expected_types = {
            'base_op_torch.add_0': ((torch.add, torch.add), (toq.add, toq.add)),
            'base_op_torch.add_1': ((torch.add, torch.add), (toq.add, toq.add)),
            'base_op_torch.add_2': ((torch.add, torch.add), (toq.add, toq.add)),
        }
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)

    @override_qengines
    def test_nodes_with_equal_types_do_not_get_matched(self):
        # verifies that by default, nodes with equivalent types do not get matched.
        # This is important for user defined types, for which we do not know
        # the weight extraction functions or input type. In the future, this can
        # be made configurable.
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.conv2 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = torch.mul(x, x)
                x = torch.sigmoid(x)
                x = F.relu(x)
                return x

        m = M().eval()
        # prevent conv2 from getting quantized, so we can test
        # modules with equal types
        qconfig_dict = {
            '': torch.quantization.default_qconfig,
            'module_name': [('conv2', None)],
        }
        mp = prepare_fx(m, qconfig_dict)
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = get_matching_subgraph_pairs(mp, mq)

        # Conv2 should not be matched because we disabled quantization for it,
        # so its type is the same in mp and mq. sigmoid and relu should not be
        # matched because they use the same function in mp and mq.
        expected_types = {
            'base_op_torch.nn.Conv2d_0':
                ((nn.Conv2d, nn.Conv2d), (nnq.Conv2d, nnq.Conv2d)),
            'base_op_torch.mul_0': ((torch.mul, torch.mul), (toq.mul, toq.mul)),
        }
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)


class TestFXGraphMatcherModels(QuantizationTestCase):

    @override_qengines
    @skip_if_no_torchvision
    def test_mobilenet_v2(self):
        # verify that mobilenetv2 graph is able to be matched
        import torchvision
        m = torchvision.models.__dict__['mobilenet_v2'](pretrained=False).eval().float()
        mp = prepare_fx(m, {'': torch.quantization.default_qconfig})
        # TODO(future PR): prevent the need for copying here, we can copy the
        # modules but should reuse the underlying tensors
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        # assume success if no exceptions
        results = get_matching_subgraph_pairs(mp, mq)

    @override_qengines
    @skip_if_no_torchvision
    def test_mobilenet_v2_qat(self):
        # verify that mobilenetv2 graph is able to be matched
        import torchvision
        m = torchvision.models.__dict__['mobilenet_v2'](pretrained=False).float()
        mp = prepare_qat_fx(m, {'': torch.quantization.get_default_qat_qconfig('fbgemm')})
        # TODO(future PR): prevent the need for copying here, we can copy the
        # modules but should reuse the underlying tensors
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        # assume success if no exceptions
        results = get_matching_subgraph_pairs(mp, mq)

class TestFXNumericSuiteCoreAPIs(QuantizationTestCase):

    @override_qengines
    def test_compare_weights_mod(self):
        m = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Conv2d(1, 1, 1)).eval()
        mp = prepare_fx(m, {'': torch.quantization.default_qconfig})
        # TODO(future PR): prevent the need for copying here, we can copy the
        # modules but should reuse the underlying tensors
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = compare_weights('fp32_prepared', mp, 'int8', mq)
        self.assertTrue(len(results) == 2)
        self.assert_ns_compare_dict_valid(results)

    @override_qengines
    def test_compare_weights_fun(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.Tensor(4, 4))
                self.b = nn.Parameter(torch.zeros(4))
                torch.nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

            def forward(self, x):
                x = F.linear(x, self.w, self.b)
                x = F.relu(x)
                x = F.linear(x, self.w, self.b)
                return x

        m = M().eval()
        mp = prepare_fx(m, {'': torch.quantization.default_qconfig})
        mp(torch.randn(1, 4))
        # TODO(future PR): prevent the need for copying here, we can copy the
        # modules but should reuse the underlying tensors
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = compare_weights('fp32_prepared', mp, 'int8', mq)
        self.assertTrue(len(results) == 2)
        self.assert_ns_compare_dict_valid(results)

    @override_qengines
    def test_match_activations_mod(self):
        m = nn.Sequential(
            torch.quantization.QuantStub(),
            nn.Conv2d(1, 1, 1),
            nn.Conv2d(1, 1, 1),
        ).eval()
        mp = prepare_fx(m, {'': torch.quantization.default_qconfig})
        mp(torch.randn(2, 1, 2, 2))
        # TODO(future PR): prevent the need for copying here, we can copy the
        # modules but should reuse the underlying tensors
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)

        mp_ns, mq_ns = prepare_model_outputs(
            'fp32_prepared', mp, 'int8', mq, OutputLogger)

        expected_occurrence = {
            ns.call_module(OutputLogger): 2,
        }
        self.checkGraphModuleNodes(
            mp_ns, expected_node_occurrence=expected_occurrence)
        self.checkGraphModuleNodes(
            mq_ns, expected_node_occurrence=expected_occurrence)

        # TODO(before land): test both scripted and non-scripted
        mp_ns = torch.jit.script(mp_ns)
        mq_ns = torch.jit.script(mq_ns)

        # calibrate
        input_fp32 = torch.randn(2, 1, 2, 2)
        mp_ns(input_fp32)
        mq_ns(input_fp32)

        # check activation result correctness
        act_compare_dict = get_matching_activations(mp_ns, mq_ns, OutputLogger)
        self.assertTrue(len(act_compare_dict) == 2)
        self.assert_ns_compare_dict_valid(act_compare_dict)

    @override_qengines
    def test_match_activations_fun(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = nn.Parameter(torch.Tensor(4, 4))
                self.b1 = nn.Parameter(torch.zeros(4))
                self.w2 = nn.Parameter(torch.Tensor(4, 4))
                self.b2 = nn.Parameter(torch.zeros(4))
                torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
                torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

            def forward(self, x):
                x = F.linear(x, self.w1, self.b1)
                x = F.linear(x, self.w2, self.b2)
                x = F.relu(x)
                return x

        m = M().eval()
        mp = prepare_fx(m, {'': torch.quantization.default_qconfig})
        mp(torch.randn(4, 4))
        # TODO(future PR): prevent the need for copying here, we can copy the
        # modules but should reuse the underlying tensors
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)

        mp_ns, mq_ns = prepare_model_outputs(
            'fp32_prepared', mp, 'int8', mq, OutputLogger)

        expected_occurrence = {
            ns.call_module(OutputLogger): 2,
        }
        self.checkGraphModuleNodes(
            mp_ns, expected_node_occurrence=expected_occurrence)
        self.checkGraphModuleNodes(
            mq_ns, expected_node_occurrence=expected_occurrence)

        # TODO(before land): test both scripted and non-scripted
        mp_ns = torch.jit.script(mp_ns)
        mq_ns = torch.jit.script(mq_ns)

        # calibrate
        input_fp32 = torch.randn(4, 4)
        mp_ns(input_fp32)
        mq_ns(input_fp32)

        # check activation result correctness
        act_compare_dict = get_matching_activations(mp_ns, mq_ns, OutputLogger)
        self.assertTrue(len(act_compare_dict) == 2)
        self.assert_ns_compare_dict_valid(act_compare_dict)

    @override_qengines
    def test_prepare_model_with_stubs_mod(self):
        m = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.Conv2d(1, 1, 1),
        ).eval()
        mp = prepare_fx(m, {'': torch.quantization.default_qconfig})
        mp(torch.randn(1, 1, 4, 4))
        # TODO(future PR): prevent the need for copying here, we can copy the
        # modules but should reuse the underlying tensors
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)

        mp_shadows_mq = prepare_model_with_stubs('fp32_prepared', mp, 'int8', mq, OutputLogger)

        # TODO(before land): test both scripted and non-scripted
        mp_shadows_mq = torch.jit.script(mp_shadows_mq)

        # calibrate
        input_fp32 = torch.randn(1, 1, 4, 4)
        mp_shadows_mq(input_fp32)

        # check activation result correctness
        act_compare_dict = get_matching_activations_a_shadows_b(
            mp_shadows_mq, OutputLogger)
        self.assertTrue(len(act_compare_dict) == 2)
        self.assert_ns_compare_dict_valid(act_compare_dict)

    @override_qengines
    def test_prepare_model_with_stubs_fun(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = nn.Parameter(torch.Tensor(4, 4))
                self.b1 = nn.Parameter(torch.zeros(4))
                self.w2 = nn.Parameter(torch.Tensor(4, 4))
                self.b2 = nn.Parameter(torch.zeros(4))
                torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
                torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

            def forward(self, x):
                x = F.linear(x, self.w1, self.b1)
                x = F.linear(x, self.w2, self.b2)
                x = F.relu(x)
                return x

        m = M().eval()
        mp = prepare_fx(m, {'': torch.quantization.default_qconfig})
        mp(torch.randn(4, 4))
        # TODO(future PR): prevent the need for copying here, we can copy the
        # modules but should reuse the underlying tensors
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)

        mp_shadows_mq = prepare_model_with_stubs('fp32_prepared', mp, 'int8', mq, OutputLogger)

        # TODO(before land): test both scripted and non-scripted
        mp_shadows_mq = torch.jit.script(mp_shadows_mq)

        # calibrate
        input_fp32 = torch.randn(4, 4)
        mp_shadows_mq(input_fp32)

        # check activation result correctness
        act_compare_dict = get_matching_activations_a_shadows_b(
            mp_shadows_mq, OutputLogger)
        self.assertTrue(len(act_compare_dict) == 2)
        self.assert_ns_compare_dict_valid(act_compare_dict)

    def test_prepare_model_with_stubs_multiple_dtype_casts(self):
        """
        Verifies that for nodes where the first input arg is a list,
        such as `cat`, we insert an individual dtype cast for each
        arg of the list.
        """
        class M(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = torch.cat([x, x, x], dim=0)
                return x

        m = M().eval()
        mp = prepare_fx(m, {'': torch.quantization.default_qconfig})
        mp(torch.randn(4, 4))
        # TODO(future PR): prevent the need for copying here, we can copy the
        # modules but should reuse the underlying tensors
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)

        mp_shadows_mq = prepare_model_with_stubs('fp32_prepared', mp, 'int8', mq, OutputLogger)

        expected_occurrence = {
            # 3 dequantize function calls from the 3 dtype casts for [x, x, x]
            ns.call_function(torch.dequantize): 3,
            # 1 dequantize method call for module output
            ns.call_method("dequantize"): 1,
        }
        self.checkGraphModuleNodes(
            mp_shadows_mq, expected_node_occurrence=expected_occurrence)

class TestFXNumericSuiteCoreAPIsModels(QuantizationTestCase):
    """
    Tests numeric suite core APIs on non-toy models.
    """

    @override_qengines
    def test_sparsenn_compare_activations(self):
        for should_log_inputs in (True, False):
            sparse_nn = SparseNNModel().eval()

            # quantize the embeddings and the dense part separately, using FX graph mode
            sparse_nn.dense_top = prepare_fx(
                sparse_nn.dense_top,
                {'': torch.quantization.default_qconfig},
            )

            # calibrate
            idx = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
            offsets = torch.LongTensor([0, 4])
            x = torch.randn(2, 4)
            sparse_nn(idx, offsets, x)

            # convert
            sparse_nn_q = copy.deepcopy(sparse_nn)
            sparse_nn_q.dense_top = convert_fx(sparse_nn_q.dense_top)

            # test out compare activations API
            sparse_nn.dense_top, sparse_nn_q.dense_top = prepare_model_outputs(
                'fp32_prepared', sparse_nn.dense_top, 'int8', sparse_nn_q.dense_top, OutputLogger,
                should_log_inputs=should_log_inputs)

            # calibrate
            sparse_nn(idx, offsets, x)
            sparse_nn_q(idx, offsets, x)

            # inspect results
            act_compare_dict = get_matching_activations(
                sparse_nn, sparse_nn_q, OutputLogger)
            if should_log_inputs:
                self.assertTrue(len(act_compare_dict) == 9)
            else:
                self.assertTrue(len(act_compare_dict) == 4)
            self.assert_ns_compare_dict_valid(act_compare_dict)

    @override_qengines
    def test_sparsenn_shadow(self):
        for should_log_inputs in (True, False):
            sparse_nn = SparseNNModel().eval()

            # quantize the embeddings and the dense part separately, using FX graph mode
            sparse_nn.dense_top = prepare_fx(
                sparse_nn.dense_top,
                {'': torch.quantization.default_qconfig},
            )

            # calibrate
            idx = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
            offsets = torch.LongTensor([0, 4])
            x = torch.randn(2, 4)
            sparse_nn(idx, offsets, x)

            # convert
            sparse_nn_q = copy.deepcopy(sparse_nn)
            sparse_nn_q.dense_top = convert_fx(sparse_nn_q.dense_top)

            # test out compare shadow activations API
            sparse_nn_q.dense_top = prepare_model_with_stubs(
                'fp32_prepared', sparse_nn.dense_top,
                'int8', sparse_nn_q.dense_top, OutputLogger,
                should_log_inputs=should_log_inputs)

            # calibrate
            sparse_nn_q(idx, offsets, x)

            # check activation result correctness
            act_compare_dict = get_matching_activations_a_shadows_b(
                sparse_nn_q, OutputLogger)
            # TODO(before land): align ref_name for input nodes
            for layer_name, models in act_compare_dict.items():
                print(layer_name)
                for model_name, data in models.items():
                    print('  ', model_name)
                    for data_k, data_v in data.items():
                        print('    ', data_k, data_v)

            if should_log_inputs:
                self.assertTrue(len(act_compare_dict) == 10)
            else:
                self.assertTrue(len(act_compare_dict) == 4)
            self.assert_ns_compare_dict_valid(act_compare_dict)
