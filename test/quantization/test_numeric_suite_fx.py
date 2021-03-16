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
    compare_model_outputs_fx,
    compare_model_stub_fx,
)
from torch.quantization.quantize_fx import (
    convert_fx,
    prepare_fx,
    prepare_qat_fx,
)
from torch.testing._internal.common_quantization import (
    ConvBnModel,
    ConvBnReLUModel,
    ConvModel,
    QuantizationTestCase,
    skipIfNoFBGEMM,
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
    extract_weights,
    add_loggers,
    OutputLogger,
    add_shadow_loggers,
    extract_logger_info,
    extract_shadow_logger_info,
)


class TestGraphModeNumericSuite(QuantizationTestCase):

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


class FXNumericSuiteQuantizationTestCase(QuantizationTestCase):
    def _test_extract_weights(self, m, results_len=0, qconfig_dict=None):
        if qconfig_dict is None:
            qconfig_dict = {'': torch.quantization.default_qconfig}
        mp = prepare_fx(m, qconfig_dict)
        # TODO(future PR): prevent the need for copying here, we can copy the
        # modules but should reuse the underlying tensors
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = extract_weights('fp32_prepared', mp, 'int8', mq)
        self.assertTrue(
            len(results) == results_len,
            f"expected len {results_len}, got len {len(results)}")
        self.assert_ns_compare_dict_valid(results)
        return results

    def _test_match_activations(
        self, m, data, prepared_expected_node_occurrence=None, results_len=0,
        should_log_inputs=False,
        qconfig_dict=None,
    ):
        if qconfig_dict is None:
            qconfig_dict = {'': torch.quantization.default_qconfig}
        mp = prepare_fx(m, qconfig_dict)
        mp(*data)
        # TODO(future PR): prevent the need for copying here, we can copy the
        # modules but should reuse the underlying tensors
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)

        mp_ns, mq_ns = add_loggers(
            'fp32_prepared', mp, 'int8', mq, OutputLogger,
            should_log_inputs=should_log_inputs)

        if prepared_expected_node_occurrence:
            self.checkGraphModuleNodes(
                mp_ns, expected_node_occurrence=prepared_expected_node_occurrence)
            self.checkGraphModuleNodes(
                mq_ns, expected_node_occurrence=prepared_expected_node_occurrence)

        mp_ns = torch.jit.script(mp_ns)
        mq_ns = torch.jit.script(mq_ns)

        # calibrate
        mp_ns(*data)
        mq_ns(*data)

        # check activation result correctness
        act_compare_dict = extract_logger_info(mp_ns, mq_ns, OutputLogger)
        self.assertTrue(
            len(act_compare_dict) == results_len,
            f"expected len {results_len}, got len {len(act_compare_dict)}")
        self.assert_ns_compare_dict_valid(act_compare_dict)
        return act_compare_dict

    def _test_match_shadow_activations(
        self, m, data, prepared_expected_node_occurrence=None, results_len=0,
        should_log_inputs=False,
    ):
        mp = prepare_fx(m, {'': torch.quantization.default_qconfig})
        mp(*data)
        # TODO(future PR): prevent the need for copying here, we can copy the
        # modules but should reuse the underlying tensors
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)

        mp_shadows_mq = add_shadow_loggers(
            'fp32_prepared', mp, 'int8', mq, OutputLogger,
            should_log_inputs=should_log_inputs)

        if prepared_expected_node_occurrence:
            self.checkGraphModuleNodes(
                mp_shadows_mq, expected_node_occurrence=prepared_expected_node_occurrence)

        # TODO(before land): test both scripted and non-scripted
        mp_shadows_mq = torch.jit.script(mp_shadows_mq)

        # calibrate
        mp_shadows_mq(*data)

        # check activation result correctness
        act_compare_dict = extract_shadow_logger_info(
            mp_shadows_mq, OutputLogger)
        self.assertTrue(
            len(act_compare_dict) == results_len,
            f"expected len {results_len}, got len {len(act_compare_dict)}")
        self.assert_ns_compare_dict_valid(act_compare_dict)


class TestFXNumericSuiteCoreAPIs(FXNumericSuiteQuantizationTestCase):

    @skipIfNoFBGEMM
    def test_extract_weights_mod(self):
        m = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Conv2d(1, 1, 1)).eval()
        self._test_extract_weights(m, results_len=2)

    @skipIfNoFBGEMM
    def test_extract_weights_fun(self):
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
        self._test_extract_weights(m, results_len=2)

    @skipIfNoFBGEMM
    def test_match_activations_mod(self):
        m = nn.Sequential(
            torch.quantization.QuantStub(),
            nn.Conv2d(1, 1, 1),
            nn.Conv2d(1, 1, 1),
        ).eval()
        expected_occurrence = {
            ns.call_module(OutputLogger): 2,
        }
        self._test_match_activations(
            m, (torch.randn(2, 1, 2, 2),),
            prepared_expected_node_occurrence=expected_occurrence,
            results_len=2)

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
        expected_occurrence = {
            ns.call_module(OutputLogger): 2,
        }
        self._test_match_activations(
            m, (torch.randn(4, 4),),
            prepared_expected_node_occurrence=expected_occurrence,
            results_len=2)

    @skipIfNoFBGEMM
    def test_add_shadow_loggers_mod(self):
        m = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.Conv2d(1, 1, 1),
        ).eval()
        self._test_match_shadow_activations(
            m, (torch.randn(1, 1, 4, 4),), results_len=2)

    @skipIfNoFBGEMM
    def test_add_shadow_loggers_fun(self):
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
        self._test_match_shadow_activations(
            m, (torch.randn(4, 4),), results_len=2)

    @skipIfNoFBGEMM
    def test_add_shadow_loggers_multiple_dtype_casts(self):
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
        expected_occurrence = {
            # 3 dequantize function calls from the 3 dtype casts for [x, x, x]
            ns.call_function(torch.dequantize): 3,
            # 1 dequantize method call for module output
            ns.call_method("dequantize"): 1,
        }
        self._test_match_shadow_activations(
            m, (torch.randn(4, 4),),
            prepared_expected_node_occurrence=expected_occurrence,
            results_len=1)

    @skipIfNoFBGEMM
    def test_logging_inputs(self):
        """
        Verifies that logging inputs works correctly
        """
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv(x)
                x = torch.cat([x, x], dim=0)
                return x

        m = M().eval()
        self._test_match_shadow_activations(
            m, (torch.randn(1, 1, 4, 4),),
            results_len=2,
            should_log_inputs=True)


class TestFXNumericSuiteCoreAPIsModels(FXNumericSuiteQuantizationTestCase):
    """
    Tests numeric suite core APIs on non-toy models.
    """

    @skipIfNoFBGEMM
    def test_compare_weights_conv(self):
        test_cases = (
            (ConvModel(),),
            (ConvBnModel(),),
            (ConvBnReLUModel(),),
        )
        for m, in test_cases:
            m.eval()
            self._test_extract_weights(m, results_len=1)

    @skipIfNoFBGEMM
    def test_compare_weights_linear(self):
        test_cases = (
            (SingleLayerLinearModel(), None),
            (
                SingleLayerLinearDynamicModel(),
                {"object_type": [(nn.Linear, default_dynamic_qconfig)]},
            ),
        )
        for m, qconfig_dict in test_cases:
            m.eval()
            res = self._test_extract_weights(
                m, results_len=1, qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_compare_weights_lstm_dynamic(self):
        qconfig_dict = {"object_type": [(nn.LSTM, default_dynamic_qconfig)]}
        m = LSTMwithHiddenDynamicModel().eval()
        res = self._test_extract_weights(
            m, results_len=1, qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_compare_activations_conv(self):
        test_cases = (
            (ConvModel(),),
            (ConvBnModel(),),
            (ConvBnReLUModel(),),
        )
        for m, in test_cases:
            m.eval()
            res = self._test_match_activations(
                m, (torch.randn(1, 3, 4, 4),), results_len=1)

    @skipIfNoFBGEMM
    def test_compare_activations_linear(self):
        test_cases = (
            (SingleLayerLinearModel(), None),
            (
                SingleLayerLinearDynamicModel(),
                {"object_type": [(nn.Linear, default_dynamic_qconfig)]},
            ),
        )
        for m, qconfig_dict in test_cases:
            m.eval()
            res = self._test_match_activations(
                m, (torch.randn(5, 5),), results_len=1, qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_sparsenn_compare_activations(self):
        for should_log_inputs in (True, False):
            sparse_nn = SparseNNModel().eval()
            idx = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
            offsets = torch.LongTensor([0, 4])
            x = torch.randn(2, 4)
            self._test_match_activations(
                sparse_nn, (idx, offsets, x),
                results_len=4,
                should_log_inputs=should_log_inputs)

    @skipIfNoFBGEMM
    def test_sparsenn_shadow(self):
        for should_log_inputs in (True, False):
            sparse_nn = SparseNNModel().eval()
            idx = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
            offsets = torch.LongTensor([0, 4])
            x = torch.randn(2, 4)
            self._test_match_activations(
                sparse_nn, (idx, offsets, x),
                results_len=4,
                should_log_inputs=should_log_inputs)
