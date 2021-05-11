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
    get_type_a_related_to_b,
)
from torch.quantization.ns.graph_matcher import (
    get_matching_subgraph_pairs,
    GraphMatchingException,
)
from torch.quantization.ns.mappings import (
    get_node_type_to_io_type_map,
    get_unmatchable_types_map,
    get_base_name_to_sets_of_related_ops,
    get_base_name_for_op,
    add_op_to_sets_of_related_ops,
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


class LinearFunctional(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(4, 4))
        self.b1 = nn.Parameter(torch.zeros(4))
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

    def forward(self, x):
        x = F.linear(x, self.w1, self.b1)
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


class AddMulFunctional(nn.Module):
    def forward(self, x, y):
        x = x + 1.0
        x = x * 1.0
        x = 1.0 + x
        x = 1.0 * x
        x = x + y
        x = x * y
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

@torch.fx.wrap
def _wrapped_hardswish(x):
    return F.hardswish(x)

@torch.fx.wrap
def _wrapped_hardswish_fp16(x):
    x = x.dequantize()
    x = F.hardswish(x)
    x = x.to(torch.float16)
    return x

@torch.fx.wrap
def _wrapped_sigmoid(x):
    return F.sigmoid(x)



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

        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        conv_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, nn.Conv2d) + '_0'

        expected_types = {
            conv_name_0: ((nn.Conv2d, nn.Conv2d), (nnq.Conv2d, nnq.Conv2d)),
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

        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        linear_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, F.linear) + '_0'

        expected_types = {
            linear_name_0:
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

        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        linear_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, F.linear) + '_0'

        expected_types = {
            linear_name_0:
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

        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        cat_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.cat) + '_0'
        add_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.add) + '_0'
        add_name_1 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.add) + '_1'

        expected_types = {
            cat_name_0: ((torch.cat, torch.cat), (torch.cat, torch.cat)),
            add_name_0: ((torch.add, torch.add), (toq.add, toq.add)),
            add_name_1: ((torch.add, torch.add), (toq.add, toq.add)),
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

        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        add_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.add) + '_0'
        add_name_1 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.add) + '_1'
        add_name_2 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.add) + '_2'

        expected_types = {
            add_name_0: ((torch.add, torch.add), (toq.add, toq.add)),
            add_name_1: ((torch.add, torch.add), (toq.add, toq.add)),
            add_name_2: ((torch.add, torch.add), (toq.add, toq.add)),
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

        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        conv_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, nn.Conv2d) + '_0'
        conv_name_1 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, nn.Conv2d) + '_1'
        mul_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.mul) + '_0'
        relu_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.relu) + '_0'
        sigmoid_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.sigmoid) + '_0'

        # all of these should be matched
        expected_types = {
            conv_name_1:
                ((nn.Conv2d, nn.Conv2d), (nnq.Conv2d, nnq.Conv2d)),
            conv_name_0:
                ((nn.Conv2d, nn.Conv2d), (nn.Conv2d, nn.Conv2d)),
            mul_name_0: ((torch.mul, torch.mul), (toq.mul, toq.mul)),
            relu_name_0: ((F.relu, F.relu), (F.relu, F.relu)),
            sigmoid_name_0:
                ((torch.sigmoid, torch.sigmoid), (torch.sigmoid, torch.sigmoid)),
        }
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)

    def test_methods(self):
        """
        Verify that graph matching works on methods
        """
        class M(nn.Module):
            def forward(self, x):
                x = x.sigmoid()
                return x

        m1 = M().eval()
        m2 = M().eval()
        qconfig_dict = {'': torch.quantization.default_qconfig}
        m1p = prepare_fx(m1, qconfig_dict)
        m2p = prepare_fx(m2, qconfig_dict)
        results = get_matching_subgraph_pairs(m1p, m2p)
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        sigmoid_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.sigmoid) + '_0'
        expected_types = {
            sigmoid_name_0:
                (('sigmoid', 'sigmoid'), ('sigmoid', 'sigmoid')),
        }
        self.assert_types_for_matched_subgraph_pairs(
            results, expected_types, m1p, m2p)


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

        unmatchable_types_map = get_unmatchable_types_map()
        FUNS_UNMATCHABLE = unmatchable_types_map['funs_unmatchable']
        MODS_UNMATCHABLE = unmatchable_types_map['mods_unmatchable']
        METHS_UNMATCHABLE = unmatchable_types_map['meths_unmatchable']

        def _op_is_unmatchable(op):
            return (
                op in FUNS_UNMATCHABLE or
                op in MODS_UNMATCHABLE or
                op in METHS_UNMATCHABLE
            )

        default_quant_patterns = get_default_quant_patterns()
        for pattern, qhandler_cls in default_quant_patterns.items():
            base_op = None
            if isinstance(pattern, tuple):
                base_op = pattern[-1]
            elif isinstance(pattern, str):
                base_op = pattern
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
                self.assertTrue(
                    _op_in_base_sets_of_related_ops(base_op),
                    f"{base_op} not in sets of related ops")
            elif qhandler_cls in qhandler_cls_quant_op_same_signature:
                # these ops use the same op signature for fp32 and quantized
                # tensors
                self.assertTrue(
                    _op_in_base_sets_of_related_ops(base_op) or
                    _op_is_unmatchable(base_op),
                    f"{base_op} not in sets of related ops or unmatchable")
            elif qhandler_cls in qhandler_cls_all_ops_quantizeable:
                self.assertTrue(
                    _op_in_base_sets_of_related_ops(base_op),
                    f"{base_op} not in sets of related ops")
            else:
                raise AssertionError(
                    f"handing for {qhandler_cls} not implemented")

    @skipIfNoFBGEMM
    def test_user_defined_function(self):
        """
        Verify that graph matching works on user defined functions
        """
        class M1(nn.Module):
            def forward(self, x):
                x = F.hardswish(x)
                return x

        class M2(nn.Module):
            def forward(self, x):
                x = _wrapped_hardswish(x)
                return x

        qconfig_dict = {'': torch.quantization.default_qconfig}
        m1 = prepare_fx(M1().eval(), qconfig_dict)
        m2 = prepare_fx(M2().eval(), qconfig_dict)

        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        add_op_to_sets_of_related_ops(
            base_name_to_sets_of_related_ops, _wrapped_hardswish, F.hardswish)

        results = get_matching_subgraph_pairs(
            m1, m2,
            base_name_to_sets_of_related_ops=base_name_to_sets_of_related_ops)

        hardswish_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, F.hardswish) + '_0'

        expected_types = {
            hardswish_name_0:
                ((F.hardswish, F.hardswish), (_wrapped_hardswish, _wrapped_hardswish)),
        }
        self.assert_types_for_matched_subgraph_pairs(
            results, expected_types, m1, m2)


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

    @skipIfNoFBGEMM
    def test_match_activations_meth_ptq(self):
        """
        Verify that add_loggers works on methods
        """
        class M(nn.Module):
            def forward(self, x):
                x = x.sigmoid()
                return x

        m = M().eval()
        res = self._test_match_activations(
            m, (torch.randn(4, 4),),
            results_len=1)

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
    def test_add_shadow_loggers_meth_ptq(self):
        """
        Verify that add_loggers works on methods
        """
        class M(nn.Module):
            def forward(self, x):
                x = x.sigmoid()
                return x

        m = M().eval()
        res = self._test_match_shadow_activations(
            m, (torch.randn(4, 4),),
            results_len=1)

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
            ns.call_module(torch.nn.Identity): 3,
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
    def test_add_mul_inputs_activations(self):
        m = AddMulFunctional().eval()
        res = self._test_match_activations(
            m, (torch.randn(2, 2), torch.randn(2, 2)),
            results_len=6, should_log_inputs=True)

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
    def test_linear_fp16_vs_linear_fp16_shadow_activations(self):
        m = LinearFunctional().eval()
        qconfig_dict = {'': torch.quantization.float16_static_qconfig}
        mp = prepare_fx(m, qconfig_dict)
        mq1 = convert_fx(copy.deepcopy(mp))
        mq2 = convert_fx(copy.deepcopy(mp))
        mq1_shadows_mq2 = _add_shadow_loggers_impl(
            'a', mq1, 'b', mq2, OutputLogger, should_log_inputs=False)
        mq1_shadows_mq2(torch.randn(4, 4))
        act_compare_dict = extract_shadow_logger_info(
            mq1_shadows_mq2, OutputLogger)
        self.assertTrue(len(act_compare_dict) == 1)
        self.assert_ns_compare_dict_valid(act_compare_dict)


    @skipIfNoFBGEMM
    def test_op_with_either_fp32_or_int8_input(self):
        """
        Verify that shadowing works with ops which accept either fp32 or
        int8 inputs.
        """
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(x)
                x = F.relu(x)
                return x

        m = M()
        res = self._test_match_shadow_activations(
            m, (torch.randn(4, 4),),
            results_len=2)

    def _test_int8_shadows_int8_impl(self, m):
        """
        Verify that shadowing works where both modules are int8
        """
        qconfig_dict = {'': torch.quantization.default_qconfig}
        mp = prepare_fx(m, qconfig_dict)
        mp(torch.randn(4, 1, 4, 4))
        mq1 = convert_fx(copy.deepcopy(mp))
        mq2 = convert_fx(mp)
        mq1_shadows_mq2 = add_shadow_loggers('a', mq1, 'b', mq2, OutputLogger)
        mq1_shadows_mq2(torch.randn(4, 1, 4, 4))
        act_compare_dict = extract_shadow_logger_info(
            mq1_shadows_mq2, OutputLogger)
        self.assertTrue(len(act_compare_dict) == 1)
        self.assert_ns_compare_dict_valid(act_compare_dict)

    @skipIfNoFBGEMM
    def test_int8_shadows_int8_mod(self):
        m = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()
        self._test_int8_shadows_int8_impl(m)

    @skipIfNoFBGEMM
    def test_int8_shadows_int8_fun(self):
        m = LinearFunctional().eval()
        self._test_int8_shadows_int8_impl(m)

    @skipIfNoFBGEMM
    def test_user_module_scriptable(self):
        # Logging of the output of this class is not supported, because it is
        # neither a tensor or an RNN return type.
        class M1(nn.Module):
            def forward(self, x):
                x1 = x * 2
                x2 = x * 4
                return (x1, x2)

        class M2(nn.Module):
            def __init__(self):
                super().__init__()
                self.m1 = M1()

            def forward(self, x):
                x1, x2 = self.m1(x)
                return x1, x2

        m = M2().eval()
        qconfig_dict = {'': torch.quantization.default_qconfig}
        prepare_custom_config_dict = {
            'non_traceable_module_class': [M1],
        }
        mp1 = prepare_fx(m, qconfig_dict, prepare_custom_config_dict)
        mp2 = copy.deepcopy(mp1)
        unmatchable_types_map = get_unmatchable_types_map()
        unmatchable_types_map['mods_unmatchable'].add(M1)
        mp1_ns, mp2_ns = _add_loggers_impl(
            'a', mp1, 'b', mp2, OutputLogger, should_log_inputs=False,
            unmatchable_types_map=unmatchable_types_map)

        # Scripting a model with loggers should succeed. If it fails because of
        # incorrect dtypes, we can blocklist the associated types from being instrumented.
        mp1_ns_scripted = torch.jit.script(mp1_ns)
        mp2_ns_scripted = torch.jit.script(mp2_ns)

    @skipIfNoFBGEMM
    def test_user_module(self):
        """
        For user defined modules,
        1. weight extraction should not crash
        2. unshadowed activations should only have loggers for known types
        3. shadowed activations should only have loggers for known types with
             known dtypes
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
        # both fp32 and int8 models should have 2 loggers each, 2 for I/O
        # of linear, and 0 for I/O of user_module
        unshadowed_expected_occurrence = {
            ns.call_module(OutputLogger): 2,
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
        # 4 loggers for I/O of linear, 0 loggers for I/O of user_module
        shadowed_expected_occurrence = {
            ns.call_module(OutputLogger): 4,
        }
        self.checkGraphModuleNodes(
            mp_shadows_mq_ns, expected_node_occurrence=shadowed_expected_occurrence)

    def test_op_io_dtype_coverage(self):
        """
        Tests that all the ops quantization cares about have input and output
        dtypes defined.
        """
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        type_a_related_to_b = \
            get_type_a_related_to_b(base_name_to_sets_of_related_ops)

        # TODO(future PR): clean this up
        node_type_to_io_type_map = get_node_type_to_io_type_map()
        FUNS_IO_TYPE_FP32 = node_type_to_io_type_map['funs_io_type_fp32']
        FUNS_IO_TYPE_INT8 = node_type_to_io_type_map['funs_io_type_int8']
        FUNS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map['funs_io_type_fp32_or_int8']
        MODS_IO_TYPE_FP32 = node_type_to_io_type_map['mods_io_type_fp32']
        MODS_IO_TYPE_INT8 = node_type_to_io_type_map['mods_io_type_int8']
        MODS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map['mods_io_type_fp32_or_int8']
        METHS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map['meths_io_type_fp32_or_int8']

        unmatchable_types_map = get_unmatchable_types_map()
        FUNS_UNMATCHABLE = unmatchable_types_map['funs_unmatchable']
        MODS_UNMATCHABLE = unmatchable_types_map['mods_unmatchable']
        METHS_UNMATCHABLE = unmatchable_types_map['meths_unmatchable']

        # 1. check static quant module mappings
        static_quant_mod_mappings = get_default_static_quant_module_mappings()
        for fp32_type, int8_type in static_quant_mod_mappings.items():
            types_to_skip = (
                torch.quantization.QuantStub,
                torch.quantization.DeQuantStub,
                nnq.FloatFunctional,
                # TODO(future PR): look into whether shadowing embeddings
                # makes sense
                nn.Embedding,
                nn.EmbeddingBag,
            )
            if fp32_type in types_to_skip:
                continue
            self.assertTrue(
                fp32_type in MODS_IO_TYPE_FP32,
                f"missing IO type handling for f{fp32_type}")
            self.assertTrue(
                int8_type in MODS_IO_TYPE_INT8,
                f"missing IO type handling for f{int8_type}")

        # 2. check static quant op mappings
        static_quant_fun_mappings = get_default_float_to_quantized_operator_mappings()
        for fp32_type, int8_type in static_quant_fun_mappings.items():
            self.assertTrue(
                fp32_type in FUNS_IO_TYPE_FP32,
                f"missing IO type handling for f{fp32_type}")
            self.assertTrue(
                int8_type in FUNS_IO_TYPE_INT8,
                f"missing IO type handling for f{int8_type}")

        # 3. check dynamic quant mappings
        dynamic_quant_mappings = get_default_dynamic_quant_module_mappings()
        for fp32_type1, fp32_type2 in dynamic_quant_mappings.items():
            # TODO(future PR): verify correct I/O for these and remove from
            # this list.
            types_to_skip = (
                nn.GRUCell,
                nn.GRU,
                nn.LSTMCell,
                nn.RNNCell,
            )
            if fp32_type1 in types_to_skip:
                continue
            self.assertTrue(
                fp32_type1 in MODS_IO_TYPE_FP32,
                f"missing IO type handling for f{fp32_type1}")
            self.assertTrue(
                fp32_type2 in MODS_IO_TYPE_FP32,
                f"missing IO type handling for f{fp32_type2}")

        # 4. go through the ops mapped to each QuantizeHandler type, and verify
        # correctness.
        default_quant_patterns = get_default_quant_patterns()
        for pattern, qhandler_cls in default_quant_patterns.items():
            base_op = None
            if isinstance(pattern, tuple):
                base_op = pattern[-1]
            elif isinstance(pattern, str):
                base_op = pattern
            else:
                base_op = pattern

            if (
                qhandler_cls in (
                    qp.BinaryOpQuantizeHandler,
                    qp.RNNDynamicQuantizeHandler,
                )
            ):
                # TODO(future PR): implement shadowing for binary ops
                # TODO(future PR): implement shadowing for RNN ops
                continue
            elif qhandler_cls == qp.CatQuantizeHandler:
                self.assertTrue(
                    base_op in FUNS_IO_TYPE_FP32_OR_INT8,
                    f"missing IO type handling for {base_op}")
            elif (
                qhandler_cls in (
                    qp.ConvReluQuantizeHandler,
                    qp.LinearReLUQuantizeHandler,
                    qp.BatchNormQuantizeHandler,
                    qp.DefaultNodeQuantizeHandler,
                    qp.ELUQuantizeHandler,
                )
            ):
                self.assertTrue(
                    (base_op in FUNS_IO_TYPE_FP32) or (base_op in MODS_IO_TYPE_FP32),
                    f"missing IO type handling for {base_op}")
            elif (
                qhandler_cls in (
                    qp.FixedQParamsOpQuantizeHandler,
                    qp.CopyNodeQuantizeHandler,
                )
            ):
                if (
                    base_op in FUNS_UNMATCHABLE or
                    base_op in MODS_UNMATCHABLE or
                    base_op in METHS_UNMATCHABLE
                ):
                    continue

                self.assertTrue(
                    (base_op in FUNS_IO_TYPE_FP32_OR_INT8) or
                    (base_op in MODS_IO_TYPE_FP32_OR_INT8) or
                    (base_op in METHS_IO_TYPE_FP32_OR_INT8),
                    f"missing IO type handling for {base_op}")
            elif qhandler_cls == qp.EmbeddingQuantizeHandler:
                # embedding shadowing is not implemented, for now
                continue
            else:
                raise AssertionError(
                    f"handing for {qhandler_cls} not implemented")

    @skipIfNoFBGEMM
    def test_user_defined_function(self):
        """
        Verify that NS APIs work on user defined functions
        """
        class M1(nn.Module):
            def forward(self, x):
                x = F.hardswish(x)
                x = x.sigmoid()
                return x

        class M2(nn.Module):
            def forward(self, x):
                x = _wrapped_hardswish(x)
                x = _wrapped_sigmoid(x)
                return x

        qconfig_dict = {'': torch.quantization.default_qconfig}
        m1 = prepare_fx(M1().eval(), qconfig_dict)
        m2 = prepare_fx(M2().eval(), qconfig_dict)
        data = torch.randn(4, 4)

        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        add_op_to_sets_of_related_ops(
            base_name_to_sets_of_related_ops, _wrapped_hardswish, F.hardswish)
        add_op_to_sets_of_related_ops(
            base_name_to_sets_of_related_ops, _wrapped_sigmoid, F.sigmoid)

        # test compare weights
        results = _extract_weights_impl(
            'a', m1, 'b', m2,
            base_name_to_sets_of_related_ops=base_name_to_sets_of_related_ops)
        self.assertTrue(len(results) == 2)
        # TODO(future PR): don't store empty dictionaries for nodes
        #   without weights.

        # test unshadowed activations

        m1_ns, m2_ns = _add_loggers_impl(
            'a', copy.deepcopy(m1), 'b', copy.deepcopy(m2), OutputLogger,
            should_log_inputs=False,
            base_name_to_sets_of_related_ops=base_name_to_sets_of_related_ops)

        # calibrate
        m1_ns(data)
        m2_ns(data)

        # check activation result correctness
        act_compare_dict = extract_logger_info(m1_ns, m2_ns, OutputLogger)
        self.assertTrue(len(act_compare_dict) == 2)
        self.assert_ns_compare_dict_valid(act_compare_dict)

        # test shadowed activations

        node_type_to_io_type_map = get_node_type_to_io_type_map()
        node_type_to_io_type_map['funs_io_type_fp32'].add(_wrapped_hardswish)
        node_type_to_io_type_map['funs_io_type_fp32'].add(_wrapped_sigmoid)

        m2_shadows_m1_ns = _add_shadow_loggers_impl(
            'a', m2, 'b', m1, OutputLogger,
            should_log_inputs=False,
            base_name_to_sets_of_related_ops=base_name_to_sets_of_related_ops,
            node_type_to_io_type_map=node_type_to_io_type_map)

        # calibrate
        m2_shadows_m1_ns(data)

        # check activation result correctness
        act_compare_dict = extract_shadow_logger_info(m2_shadows_m1_ns, OutputLogger)
        self.assertTrue(len(act_compare_dict) == 2)
        self.assert_ns_compare_dict_valid(act_compare_dict)


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
