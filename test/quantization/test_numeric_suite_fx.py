import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.intrinsic as nni
import torch.nn.quantized as nnq
toq = torch.ops.quantized
from torch.quantization import get_default_qconfig
from torch.quantization._numeric_suite_fx import (
    compare_weights_fx,
    remove_qconfig_observer_fx,
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
    skip_if_no_torchvision,
    test_only_eval_fn,
)
from torch.testing._internal.common_quantized import override_qengines
from torch.quantization.ns.graph_matcher import (
    get_matching_node_pairs,
    GraphMatchingException,
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

        backup_prepared_model = copy.deepcopy(prepared_model)
        backup_prepared_model.eval()

        model = remove_qconfig_observer_fx(backup_prepared_model)

        modules = dict(model.named_modules())
        for node in model.graph.nodes:
            if node.op == "call_module":
                self.assertFalse(is_activation_post_process(modules[node.target]))



    def compare_and_validate_model_weights_results_fx(
        self, float_model, q_model, expected_weight_dict_keys
    ):
        weight_dict = compare_weights_fx(float_model.state_dict(), q_model.state_dict())

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

        backup_prepared_model = copy.deepcopy(prepared_model)
        backup_prepared_model.eval()

        # Run calibration
        test_only_eval_fn(prepared_model, self.calib_data)
        q_model = convert_fx(prepared_model)

        expected_weight_dict_keys = {"fc1._packed_params._packed_params"}
        self.compare_and_validate_model_weights_results_fx(
            backup_prepared_model, q_model, expected_weight_dict_keys
        )

    @override_qengines
    def test_compare_weights_linear_dynamic_fx(self):
        r"""Compare the weights of float and dynamic quantized linear layer"""

        qconfig = torch.quantization.qconfig.default_dynamic_qconfig
        qconfig_dict = {"": qconfig}

        float_model = SingleLayerLinearDynamicModel()
        float_model.eval()

        prepared_model = prepare_fx(float_model, qconfig_dict)

        backup_prepared_model = copy.deepcopy(prepared_model)
        backup_prepared_model.eval()

        q_model = convert_fx(prepared_model)

        expected_weight_dict_keys = {"fc1._packed_params._packed_params"}
        self.compare_and_validate_model_weights_results_fx(
            backup_prepared_model, q_model, expected_weight_dict_keys
        )

    def compare_and_validate_model_stub_results_fx(
        self, float_model, q_model, module_swap_list, data, expected_ob_dict_keys
    ):
        ob_dict = compare_model_stub_fx(float_model, q_model, module_swap_list, data)

        self.assertTrue(ob_dict.keys() == expected_ob_dict_keys)
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

            backup_prepared_model = copy.deepcopy(prepared_model)

            # Run calibration
            test_only_eval_fn(prepared_model, self.img_data_2d)
            q_model = convert_fx(prepared_model)

            module_swap_list = [nn.Conv2d, nni.modules.fused.ConvReLU2d]

            expected_ob_dict_keys = {"conv.stats"}
            self.compare_and_validate_model_stub_results_fx(
                backup_prepared_model,
                q_model,
                module_swap_list,
                self.img_data_2d[0][0],
                expected_ob_dict_keys,
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

        backup_prepared_model = copy.deepcopy(prepared_model)

        # Run calibration
        test_only_eval_fn(prepared_model, self.calib_data)
        q_model = convert_fx(prepared_model)

        linear_data = self.calib_data[0][0]
        module_swap_list = [nn.Linear]

        expected_ob_dict_keys = {"fc1.stats"}
        self.compare_and_validate_model_stub_results_fx(
            backup_prepared_model,
            q_model,
            module_swap_list,
            linear_data,
            expected_ob_dict_keys,
        )

class TestFXGraphMatcher(QuantizationTestCase):
    # TODO(future PR): more tests

    def test_conv_mod_fp32_prepared_vs_int8(self):
        m = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()
        mp = prepare_fx(m, {'': torch.quantization.default_qconfig})
        # TODO(future PR): prevent the need for copying here, we can copy the
        # modules but should reuse the underlying tensors
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = get_matching_node_pairs(mp, mq)

        expected_types = {'0': (nn.Conv2d, nnq.Conv2d)}
        self.assert_types_for_matched_node_pairs(results, expected_types, mp, mq)

    def test_linear_func_fp32_prepared_vs_int8(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.Tensor(1, 1))
                self.b = nn.Parameter(torch.Tensor(1))

            def forward(self, x):
                return F.linear(x, self.w, self.b)

        m = M().eval()
        mp = prepare_fx(m, {'': torch.quantization.default_qconfig})
        # TODO(future PR): prevent the need for copying here, we can copy the
        # modules but should reuse the underlying tensors
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = get_matching_node_pairs(mp, mq)

        expected_types = {'linear_1': (F.linear, toq.linear)}
        self.assert_types_for_matched_node_pairs(results, expected_types, mp, mq)

    def test_matching_failure_node_count(self):
        # verify that matching graphs with matching node types but
        # different counts of matchable nodes fails
        m1 = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()
        m2 = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Conv2d(1, 1, 1)).eval()
        mp1 = prepare_fx(m1, {'': torch.quantization.default_qconfig})
        mp2 = prepare_fx(m2, {'': torch.quantization.default_qconfig})
        with self.assertRaises(GraphMatchingException) as ex:
            results = get_matching_node_pairs(mp1, mp2)

    def test_matching_failure_node_type(self):
        # verify that matching graphs with non-matching node types fails
        m1 = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()
        m2 = nn.Sequential(nn.Linear(1, 1)).eval()
        mp1 = prepare_fx(m1, {'': torch.quantization.default_qconfig})
        mp2 = prepare_fx(m2, {'': torch.quantization.default_qconfig})
        with self.assertRaises(GraphMatchingException) as ex:
            results = get_matching_node_pairs(mp1, mp2)

    def test_conv_multilayer_mod_fp32_prepared_vs_int8(self):
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
        results = get_matching_node_pairs(mp, mq)

    def test_tensor_ops_fp32_prepared_vs_int8(self):
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
        results = get_matching_node_pairs(mp, mq)

    @skip_if_no_torchvision
    def test_mobilenet_v2_fp32_prepared_vs_int8(self):
        # verify that mobilenetv2 graph is able to be matched
        import torchvision
        m = torchvision.models.__dict__['mobilenet_v2'](pretrained=False).eval().float()
        mp = prepare_fx(m, {'': torch.quantization.default_qconfig})
        # TODO(future PR): prevent the need for copying here, we can copy the
        # modules but should reuse the underlying tensors
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        # assume success if no exceptions
        results = get_matching_node_pairs(mp, mq)

    @skip_if_no_torchvision
    def test_mobilenet_v2_fp32_qat_prepared_vs_int8(self):
        # verify that mobilenetv2 graph is able to be matched
        import torchvision
        m = torchvision.models.__dict__['mobilenet_v2'](pretrained=False).float()
        mp = prepare_qat_fx(m, {'': torch.quantization.get_default_qat_qconfig('fbgemm')})
        # TODO(future PR): prevent the need for copying here, we can copy the
        # modules but should reuse the underlying tensors
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        # assume success if no exceptions
        results = get_matching_node_pairs(mp, mq)
