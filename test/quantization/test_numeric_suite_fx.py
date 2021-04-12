import copy
import math
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import default_dynamic_qconfig
import torch.nn.quantized as nnq
toq = torch.ops.quantized
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
)
from torch.quantization.quantization_mappings import (
    get_default_static_quant_module_mappings,
    get_default_dynamic_quant_module_mappings,
    get_default_float_to_quantized_operator_mappings,
)
from torch.testing._internal.common_quantization import NodeSpec as ns
from torch.testing._internal.common_quantized import override_qengines
from torch.quantization.fx.pattern_utils import get_default_quant_patterns
import torch.quantization.fx.quantization_patterns as qp
from torch.quantization.ns.pattern_utils import (
    get_base_name_to_sets_of_related_ops,
    get_type_a_related_to_b,
)
from torch.quantization.ns.graph_matcher import (
    get_matching_subgraph_pairs,
    GraphMatchingException,
)
from torch.quantization._numeric_suite_fx import (
    extract_weights,
    _extract_weights_impl,
    add_loggers,
    _add_loggers_impl,
    OutputLogger,
    add_shadow_loggers,
    _add_shadow_loggers_impl,
    extract_logger_info,
    extract_shadow_logger_info,
)


# Note: these models are not for use outside of this file. While it's good
# to reuse code, we also need to be able to iterate on tests
# quickly when debugging. If a test model has a large number of callsites
# across various different files, speed of debugging on individual test cases
# decreases.
class LinearReluFunctional(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(4, 4))
        self.b1 = nn.Parameter(torch.zeros(4))
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

    def forward(self, x):
        x = F.linear(x, self.w1, self.b1)
        x = F.relu(x)
        return x


class LinearReluLinearFunctional(nn.Module):
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


class AllConvAndLinearFusionModules(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # conv1d
        self.conv1d_0 = nn.Conv1d(1, 1, 1)
        # conv1d - relu
        self.conv1d_1 = nn.Conv1d(1, 1, 1)
        self.relu_0 = nn.ReLU()
        # conv1d - bn (qat only)
        self.conv1d_2 = nn.Conv1d(1, 1, 1)
        self.bn1d_0 = nn.BatchNorm1d(1)
        # conv1d - bn - relu (qat only)
        self.conv1d_3 = nn.Conv1d(1, 1, 1)
        self.bn1d_1 = nn.BatchNorm1d(1)
        self.relu_4 = nn.ReLU()
        # conv2d
        self.conv2d_0 = nn.Conv2d(1, 1, 1)
        # conv2d - relu
        self.conv2d_1 = nn.Conv2d(1, 1, 1)
        self.relu_1 = nn.ReLU()
        # conv2d - bn (qat only)
        self.conv2d_2 = nn.Conv2d(1, 1, 1)
        self.bn2d_0 = nn.BatchNorm2d(1)
        # conv2d - bn - relu (qat only)
        self.conv2d_3 = nn.Conv2d(1, 1, 1)
        self.bn2d_1 = nn.BatchNorm2d(1)
        self.relu_5 = nn.ReLU()
        # conv3d
        self.conv3d_0 = nn.Conv3d(1, 1, 1)
        # conv3d - relu
        self.conv3d_1 = nn.Conv3d(1, 1, 1)
        self.relu_2 = nn.ReLU()
        # conv3d - bn (qat only)
        self.conv3d_2 = nn.Conv3d(1, 1, 1)
        self.bn3d_0 = nn.BatchNorm3d(1)
        # conv3d - bn - relu (qat only)
        self.conv3d_3 = nn.Conv3d(1, 1, 1)
        self.bn3d_1 = nn.BatchNorm3d(1)
        self.relu_6 = nn.ReLU()
        # linear
        self.linear_0 = nn.Linear(1, 1)
        # linear - relu
        self.linear_1 = nn.Linear(1, 1)
        self.relu_3 = nn.ReLU()

    def forward(self, x):
        # conv1d
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        x = self.relu_0(x)
        x = self.conv1d_2(x)
        x = self.bn1d_0(x)
        x = self.conv1d_3(x)
        x = self.bn1d_1(x)
        x = self.relu_4(x)
        # conv2d
        x = x.reshape(1, 1, 1, 1)
        x = self.conv2d_0(x)
        x = self.conv2d_1(x)
        x = self.relu_1(x)
        x = self.conv2d_2(x)
        x = self.bn2d_0(x)
        x = self.conv2d_3(x)
        x = self.bn2d_1(x)
        x = self.relu_5(x)
        # conv3d
        x = x.reshape(1, 1, 1, 1, 1)
        x = self.conv3d_0(x)
        x = self.conv3d_1(x)
        x = self.relu_2(x)
        x = self.conv3d_2(x)
        x = self.bn3d_0(x)
        x = self.conv3d_3(x)
        x = self.bn3d_1(x)
        x = self.relu_6(x)
        # linear
        x = x.reshape(1, 1)
        x = self.linear_0(x)
        x = self.linear_1(x)
        x = self.relu_3(x)
        return x


class AllConvFunctional(torch.nn.Module):
    def __init__(self, weight1d, weight2d, weight3d, bias1d, bias2d, bias3d):
        super().__init__()
        self.weight1d = torch.nn.Parameter(weight1d)
        self.weight2d = torch.nn.Parameter(weight2d)
        self.weight3d = torch.nn.Parameter(weight3d)
        self.bias1d = torch.nn.Parameter(bias1d)
        self.bias2d = torch.nn.Parameter(bias2d)
        self.bias3d = torch.nn.Parameter(bias3d)
        self.stride1d = 1
        self.padding1d = 0
        self.dilation1d = 1
        self.stride2d = (1, 1)
        self.padding2d = (0, 0)
        self.dilation2d = (1, 1)
        self.groups = 1
        self.stride3d = (1, 1, 1)
        self.padding3d = (0, 0, 0)
        self.dilation3d = (1, 1, 1)

    def forward(self, x):
        x = F.conv1d(
            x, self.weight1d, self.bias1d, self.stride1d, self.padding1d,
            self.dilation1d, self.groups)
        x = F.conv1d(
            x, self.weight1d, self.bias1d, self.stride1d, self.padding1d,
            self.dilation1d, self.groups)
        x = F.relu(x)
        x = F.conv2d(
            x, self.weight2d, self.bias2d, self.stride2d, self.padding2d,
            self.dilation2d, self.groups)
        x = F.conv2d(
            x, self.weight2d, self.bias2d, self.stride2d, self.padding2d,
            self.dilation2d, self.groups)
        x = F.relu(x)
        x = F.conv3d(
            x, self.weight3d, self.bias3d, self.stride3d, self.padding3d,
            self.dilation3d, self.groups)
        x = F.conv3d(
            x, self.weight3d, self.bias3d, self.stride3d, self.padding3d,
            self.dilation3d, self.groups)
        x = F.relu(x)
        return x


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
                self.w = nn.Parameter(torch.empty(1, 4))
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
        m = LinearReluFunctional().eval()
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

    @skipIfNoFBGEMM
    def test_nodes_with_equal_types_get_matched(self):
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

        # all of these should be matched
        expected_types = {
            'base_op_torch.nn.Conv2d_1':
                ((nn.Conv2d, nn.Conv2d), (nnq.Conv2d, nnq.Conv2d)),
            'base_op_torch.nn.Conv2d_0':
                ((nn.Conv2d, nn.Conv2d), (nn.Conv2d, nn.Conv2d)),
            'base_op_torch.mul_0': ((torch.mul, torch.mul), (toq.mul, toq.mul)),
            'base_op_torch.relu_0': ((F.relu, F.relu), (F.relu, F.relu)),
            'base_op_torch.sigmoid_0':
                ((torch.sigmoid, torch.sigmoid), (torch.sigmoid, torch.sigmoid)),
        }
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)

    def test_op_relationship_mapping(self):
        """
        Tests that the mapping of op relationships is complete.
        """
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        type_a_related_to_b = \
            get_type_a_related_to_b(base_name_to_sets_of_related_ops)

        # 1. check static quant module mappings
        static_quant_mod_mappings = get_default_static_quant_module_mappings()
        for fp32_type, int8_type in static_quant_mod_mappings.items():
            # skip quants and dequants, for the purposes of Numerical Suite
            types_to_skip = (
                torch.quantization.QuantStub,
                torch.quantization.DeQuantStub,
                nnq.FloatFunctional,
            )
            if fp32_type in types_to_skip:
                continue

            # verify relatedness
            in_type_a_related_to_b = \
                (fp32_type, int8_type) in type_a_related_to_b
            self.assertTrue(
                in_type_a_related_to_b,
                f"{fp32_type} and {int8_type} need a relationship mapping")

        # 2. check static quant op mappings
        static_quant_fun_mappings = get_default_float_to_quantized_operator_mappings()
        for fp32_type, int8_type in static_quant_fun_mappings.items():
            # verify relatedness
            in_type_a_related_to_b = \
                (fp32_type, int8_type) in type_a_related_to_b
            self.assertTrue(
                in_type_a_related_to_b,
                f"{fp32_type} and {int8_type} need a relationship mapping")

        # 3. check dynamic quant mappings
        dynamic_quant_mappings = get_default_dynamic_quant_module_mappings()
        for fp32_type, int8_type in dynamic_quant_mappings.items():
            # TODO(future PR): enable correct weight extraction for these
            # and remove from this list.
            types_to_skip = (
                nn.GRUCell,
                nn.GRU,
                nn.LSTMCell,
                nn.RNNCell,
            )
            if fp32_type in types_to_skip:
                continue
            # verify relatedness
            in_type_a_related_to_b = \
                (fp32_type, int8_type) in type_a_related_to_b
            self.assertTrue(
                in_type_a_related_to_b,
                f"{fp32_type} and {int8_type} need a relationship mapping")

        # 4. go through the ops mapped to each QuantizeHandler type, and verify
        # correctness.
        def _op_in_base_sets_of_related_ops(op):
            for name, ops in base_name_to_sets_of_related_ops.items():
                if op in ops:
                    return True
            return False

        default_quant_patterns = get_default_quant_patterns()
        for pattern, qhandler_cls in default_quant_patterns.items():
            base_op = None
            if isinstance(pattern, tuple):
                base_op = pattern[-1]
            elif isinstance(pattern, str):
                # TODO(future PR): add handling for these
                continue
            else:
                base_op = pattern

            qhandler_cls_all_ops_quantizeable = [
                qp.CatQuantizeHandler,
                qp.ConvReluQuantizeHandler,
                qp.LinearReLUQuantizeHandler,
                qp.BatchNormQuantizeHandler,
                qp.EmbeddingQuantizeHandler,
                qp.RNNDynamicQuantizeHandler,
                qp.ELUQuantizeHandler,
            ]

            qhandler_cls_quant_op_same_signature = [
                qp.FixedQParamsOpQuantizeHandler,
                qp.CopyNodeQuantizeHandler,
            ]

            if qhandler_cls == qp.BinaryOpQuantizeHandler:
                # these ops do not have quantized equivalents
                ops_to_skip = [
                    torch.bmm,
                    torch.sum,
                    torch.div,
                    torch.sub,
                    operator.truediv,
                    operator.sub
                ]
                if base_op in ops_to_skip:
                    continue
                self.assertTrue(
                    _op_in_base_sets_of_related_ops(base_op),
                    f"{base_op} not in sets of related ops")
            elif qhandler_cls == qp.RNNDynamicQuantizeHandler:
                # TODO(future PR): add support for all classes in
                # RNNDynamicQuantizeHandler
                pass
            elif qhandler_cls == qp.DefaultNodeQuantizeHandler:
                ops_to_skip = [
                    torch.nn.SiLU,
                    torch.nn.functional.silu,
                ]
                if base_op in ops_to_skip:
                    continue
                self.assertTrue(
                    _op_in_base_sets_of_related_ops(base_op),
                    f"{base_op} not in sets of related ops")
            elif qhandler_cls in qhandler_cls_quant_op_same_signature:
                # these ops use the same op signature for fp32 and quantized
                # tensors
                pass
            elif qhandler_cls in qhandler_cls_all_ops_quantizeable:
                self.assertTrue(
                    _op_in_base_sets_of_related_ops(base_op),
                    f"{base_op} not in sets of related ops")
            else:
                raise AssertionError(
                    f"handing for {qhandler_cls} not implemented")


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
    def _test_extract_weights(
        self, m, results_len=0, qconfig_dict=None, prepare_fn=prepare_fx
    ):
        if qconfig_dict is None:
            qconfig_dict = {'': torch.quantization.default_qconfig}
        mp = prepare_fn(m, qconfig_dict)
        # TODO(future PR): prevent the need for copying here, we can copy the
        # modules but should reuse the underlying tensors
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)

        # test both the public API as well as the internal GraphModule API
        for extract_weights_fun in (extract_weights, _extract_weights_impl):
            results = extract_weights_fun('fp32_prepared', mp, 'int8', mq)
            self.assertTrue(
                len(results) == results_len,
                f"expected len {results_len}, got len {len(results)}")
            self.assert_ns_compare_dict_valid(results)

    def _test_match_activations(
        self, m, data, prepared_expected_node_occurrence=None, results_len=0,
        should_log_inputs=False,
        qconfig_dict=None,
        skip_scripting=False,
        prepare_fn=prepare_fx,
    ):
        if qconfig_dict is None:
            qconfig_dict = {'': torch.quantization.default_qconfig}
        if prepare_fn == prepare_fx:
            m.eval()
        else:
            m.train()
        mp = prepare_fn(m, qconfig_dict)
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

        if not skip_scripting:
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
        should_log_inputs=False, qconfig_dict=None, skip_scripting=False,
        prepare_fn=prepare_fx,
    ):
        if qconfig_dict is None:
            qconfig_dict = {'': torch.quantization.default_qconfig}
        if prepare_fn == prepare_fx:
            m.eval()
        else:
            m.train()
        mp = prepare_fn(m, qconfig_dict)
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

        if not skip_scripting:
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
        return act_compare_dict


class TestFXNumericSuiteCoreAPIs(FXNumericSuiteQuantizationTestCase):

    @skipIfNoFBGEMM
    def test_extract_weights_mod_ptq(self):
        m = AllConvAndLinearFusionModules().eval()
        self._test_extract_weights(m, results_len=14)

    @skipIfNoFBGEMM
    def test_extract_weights_mod_qat(self):
        m = AllConvAndLinearFusionModules().train()
        qconfig_dict = {'': torch.quantization.get_default_qat_qconfig('fbgemm')}
        self._test_extract_weights(
            m, results_len=14, qconfig_dict=qconfig_dict, prepare_fn=prepare_qat_fx)

    @skipIfNoFBGEMM
    def test_extract_weights_linear_fun_ptq(self):
        m = LinearReluLinearFunctional().eval()
        self._test_extract_weights(m, results_len=2)

    @skipIfNoFBGEMM
    def test_extract_weights_linear_fun_qat(self):
        m = LinearReluLinearFunctional().train()
        qconfig_dict = {'': torch.quantization.get_default_qat_qconfig('fbgemm')}
        self._test_extract_weights(
            m, results_len=2, qconfig_dict=qconfig_dict, prepare_fn=prepare_qat_fx)

    @skipIfNoFBGEMM
    def test_extract_weights_conv_fun_ptq(self):
        w1d = torch.randn(1, 1, 1)
        w2d = torch.randn(1, 1, 1, 1)
        w3d = torch.randn(1, 1, 1, 1, 1)
        b1d = torch.randn(1)
        b2d = torch.randn(1)
        b3d = torch.randn(1)
        m = AllConvFunctional(w1d, w2d, w3d, b1d, b2d, b3d).eval()
        self._test_extract_weights(m, results_len=6)

    @skipIfNoFBGEMM
    def test_extract_weights_conv_fun_qat(self):
        w1d = torch.randn(1, 1, 1)
        w2d = torch.randn(1, 1, 1, 1)
        w3d = torch.randn(1, 1, 1, 1, 1)
        b1d = torch.randn(1)
        b2d = torch.randn(1)
        b3d = torch.randn(1)
        m = AllConvFunctional(w1d, w2d, w3d, b1d, b2d, b3d).train()
        qconfig_dict = {'': torch.quantization.get_default_qat_qconfig('fbgemm')}
        self._test_extract_weights(
            m, results_len=6, qconfig_dict=qconfig_dict, prepare_fn=prepare_qat_fx)

    @skipIfNoFBGEMM
    def test_extract_weights_dynamic(self):
        # TODO(future PR): add Linear-ReLU, after #55393 is fixed.
        m = nn.Sequential(nn.Linear(1, 1)).eval()
        qconfig_dict = {
            'object_type': [
                (nn.Linear, default_dynamic_qconfig),
            ],
        }
        self._test_extract_weights(m, results_len=1, qconfig_dict=qconfig_dict)

    def _test_match_activations_mod_impl(self, prepare_fn=prepare_fx):
        m = nn.Sequential(
            torch.quantization.QuantStub(),
            nn.Conv2d(1, 1, 1),
            nn.Conv2d(1, 1, 1),
        ).eval()
        qconfig_dict = None
        if prepare_fn == prepare_qat_fx:
            qconfig_dict = {'': torch.quantization.get_default_qat_qconfig('fbgemm')}
        expected_occurrence = {
            ns.call_module(OutputLogger): 2,
        }
        self._test_match_activations(
            m, (torch.randn(2, 1, 2, 2),),
            prepared_expected_node_occurrence=expected_occurrence,
            results_len=2, qconfig_dict=qconfig_dict, prepare_fn=prepare_fn)

    @skipIfNoFBGEMM
    def test_match_activations_mod_ptq(self):
        self._test_match_activations_mod_impl(prepare_fn=prepare_fx)

    @skipIfNoFBGEMM
    def test_match_activations_mod_qat(self):
        self._test_match_activations_mod_impl(prepare_fn=prepare_qat_fx)

    def _test_match_activations_fun_impl(self, prepare_fn=prepare_fx):
        m = LinearReluLinearFunctional().eval()
        qconfig_dict = None
        if prepare_fn == prepare_qat_fx:
            qconfig_dict = {'': torch.quantization.get_default_qat_qconfig('fbgemm')}
        expected_occurrence = {
            ns.call_module(OutputLogger): 2,
        }
        self._test_match_activations(
            m, (torch.randn(4, 4),),
            prepared_expected_node_occurrence=expected_occurrence,
            results_len=2, prepare_fn=prepare_fn, qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_match_activations_fun_ptq(self):
        self._test_match_activations_fun_impl(prepare_fn=prepare_fx)

    @skipIfNoFBGEMM
    def test_match_activations_fun_qat(self):
        self._test_match_activations_fun_impl(prepare_fn=prepare_qat_fx)

    def _test_add_shadow_loggers_mod_impl(self, prepare_fn=prepare_fx):
        m = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.Conv2d(1, 1, 1),
        ).eval()
        qconfig_dict = None
        if prepare_fn == prepare_qat_fx:
            qconfig_dict = {'': torch.quantization.get_default_qat_qconfig('fbgemm')}
        res = self._test_match_shadow_activations(
            m, (torch.randn(1, 1, 4, 4),), results_len=2,
            prepare_fn=prepare_fn, qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_add_shadow_loggers_mod_ptq(self):
        self._test_add_shadow_loggers_mod_impl(prepare_fn=prepare_fx)

    @skipIfNoFBGEMM
    def test_add_shadow_loggers_mod_qat(self):
        self._test_add_shadow_loggers_mod_impl(prepare_fn=prepare_qat_fx)

    def _test_add_shadow_loggers_fun_impl(self, prepare_fn=prepare_fx):
        m = LinearReluLinearFunctional()
        qconfig_dict = None
        if prepare_fn == prepare_qat_fx:
            qconfig_dict = {'': torch.quantization.get_default_qat_qconfig('fbgemm')}
        res = self._test_match_shadow_activations(
            m, (torch.randn(4, 4),), results_len=2, prepare_fn=prepare_fn,
            qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_add_shadow_loggers_fun_ptq(self):
        self._test_add_shadow_loggers_fun_impl(prepare_fn=prepare_fx)

    @skipIfNoFBGEMM
    def test_add_shadow_loggers_fun_qat(self):
        self._test_add_shadow_loggers_fun_impl(prepare_fn=prepare_qat_fx)

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

    @skipIfNoFBGEMM
    def test_ops_with_same_fp32_and_int8_signature(self):
        """
        Verifies that we can match pairs of ops which have the same aten
        signature for fp32 and int8 tensors.
        """
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.max_pool_2d = nn.MaxPool2d(2)

            def forward(self, x):
                x = self.max_pool_2d(x)
                x = F.relu(x)
                return x

        m = M().eval()
        self._test_match_activations(
            m, (torch.randn(1, 1, 2, 2),),
            results_len=2)

    @skipIfNoFBGEMM
    def test_linear_fp16_weights(self):
        qconfig_dict = {'': torch.quantization.float16_static_qconfig}
        m = LinearReluFunctional().eval()
        self._test_extract_weights(m, results_len=1, qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_linear_fp16_activations(self):
        for should_log_inputs in (True, False):
            qconfig_dict = {'': torch.quantization.float16_static_qconfig}
            m = LinearReluFunctional().eval()
            num_loggers = 2 if should_log_inputs else 1
            expected_occurrence = {
                ns.call_module(OutputLogger): num_loggers,
            }
            res = self._test_match_activations(
                m, (torch.randn(4, 4),),
                prepared_expected_node_occurrence=expected_occurrence,
                results_len=1,
                qconfig_dict=qconfig_dict,
                should_log_inputs=should_log_inputs)

    @skipIfNoFBGEMM
    def test_linear_fp16_shadow_activations(self):
        for should_log_inputs in (True, False):
            qconfig_dict = {'': torch.quantization.float16_static_qconfig}
            m = LinearReluFunctional().eval()
            num_loggers = 4 if should_log_inputs else 2
            expected_occurrence = {
                ns.call_module(OutputLogger): num_loggers,
            }
            res2 = self._test_match_shadow_activations(
                m, (torch.randn(4, 4),),
                prepared_expected_node_occurrence=expected_occurrence,
                results_len=1,
                qconfig_dict=qconfig_dict,
                should_log_inputs=should_log_inputs)

    @skipIfNoFBGEMM
    def test_user_module(self):
        """
        For user defined modules,
        1. weight extraction should not crash
        2. unshadowed activations should have loggers, loggers will only log if
             the output dtype is in the allowlist
        3. shadowed activations should not have loggers
             (since I/O dtype is unknown)
        """
        class UserModule(nn.Module):
            def forward(self, x):
                return x

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)
                self.user_module = UserModule()

            def forward(self, x):
                x = self.linear(x)
                x = self.user_module(x)
                return x

        m = M().eval()

        # quantize without tracing through UserModule
        qconfig_dict = {'': torch.quantization.default_qconfig}
        prepare_custom_config_dict = {'non_traceable_module_name': ['user_module']}
        mp = prepare_fx(m, qconfig_dict, prepare_custom_config_dict)
        mp(torch.randn(1, 1, 1))
        mq = convert_fx(copy.deepcopy(mp))

        # weight extraction should not crash
        weights = _extract_weights_impl('fp32_prepared', mp, 'int8', mq)

        # unshadowed activations should have loggers

        # add loggers, without retracing
        # note: converting again because we cannot copy a quantized linear
        mp_ns, mq_ns = _add_loggers_impl(
            'fp32_prepared', copy.deepcopy(mp), 'int8',
            convert_fx(copy.deepcopy(mp)), OutputLogger,
            should_log_inputs=True)
        # both fp32 and int8 models should have 4 loggers each, 2 for I/O
        # of linear, and 2 for I/O of user_module
        unshadowed_expected_occurrence = {
            ns.call_module(OutputLogger): 4,
        }
        self.checkGraphModuleNodes(
            mp_ns, expected_node_occurrence=unshadowed_expected_occurrence)
        self.checkGraphModuleNodes(
            mq_ns, expected_node_occurrence=unshadowed_expected_occurrence)

        # shadowed activations should only have loggers for nodes where
        # the types are known and we can do a dtype cast

        # add shadow loggers, without retracing
        mp_shadows_mq_ns = _add_shadow_loggers_impl(
            'fp32_prepared', mp, 'int8', mq, OutputLogger,
            should_log_inputs=True)
        # 2 loggers for I/O of linear, 0 loggers for I/O of user_module
        shadowed_expected_occurrence = {
            ns.call_module(OutputLogger): 2,
        }
        self.checkGraphModuleNodes(
            mp_shadows_mq_ns, expected_node_occurrence=unshadowed_expected_occurrence)


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
    def test_compare_activations_lstm_dynamic(self):
        qconfig_dict = {"object_type": [(nn.LSTM, default_dynamic_qconfig)]}
        m = LSTMwithHiddenDynamicModel().eval()
        lstm_input = torch.rand((1, 1, 2))
        lstm_hidden = (torch.rand(1, 1, 2), torch.rand(1, 1, 2))
        # TODO(future PR): enable scripting (quant prepared LSTM not scriptable)
        res = self._test_match_activations(
            m, (lstm_input, lstm_hidden), results_len=1, qconfig_dict=qconfig_dict,
            skip_scripting=True)

    @skipIfNoFBGEMM
    def test_compare_shadow_activations_conv(self):
        test_cases = (
            (ConvModel(),),
            (ConvBnModel(),),
            (ConvBnReLUModel(),),
        )
        for m, in test_cases:
            m.eval()
            res = self._test_match_shadow_activations(
                m, (torch.randn(1, 3, 4, 4),), results_len=1)

    @skipIfNoFBGEMM
    def test_compare_shadow_activations_linear(self):
        test_cases = (
            (SingleLayerLinearModel(), None),
            (
                SingleLayerLinearDynamicModel(),
                {"object_type": [(nn.Linear, default_dynamic_qconfig)]},
            ),
        )
        for m, qconfig_dict in test_cases:
            m.eval()
            res = self._test_match_shadow_activations(
                m, (torch.randn(5, 5),), results_len=1, qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_compare_shadow_activations_lstm_dynamic(self):
        qconfig_dict = {"object_type": [(nn.LSTM, default_dynamic_qconfig)]}
        m = LSTMwithHiddenDynamicModel().eval()
        lstm_input = torch.rand((1, 1, 2))
        lstm_hidden = (torch.rand(1, 1, 2), torch.rand(1, 1, 2))
        # TODO(future PR): enable scripting (quant prepared LSTM not scriptable)
        res = self._test_match_shadow_activations(
            m, (lstm_input, lstm_hidden), results_len=1, qconfig_dict=qconfig_dict,
            skip_scripting=True)

    @skipIfNoFBGEMM
    def test_sparsenn_compare_activations(self):
        for should_log_inputs in (True, False):
            sparse_nn = SparseNNModel().eval()
            idx = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
            offsets = torch.LongTensor([0, 4])
            x = torch.randn(2, 4)
            self._test_match_activations(
                sparse_nn, (idx, offsets, x),
                results_len=5,
                should_log_inputs=should_log_inputs)

    @skipIfNoFBGEMM
    def test_sparsenn_shadow(self):
        for should_log_inputs in (True, False):
            sparse_nn = SparseNNModel().eval()
            idx = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
            offsets = torch.LongTensor([0, 4])
            x = torch.randn(2, 4)
            self._test_match_shadow_activations(
                sparse_nn, (idx, offsets, x),
                results_len=4,
                should_log_inputs=should_log_inputs)
