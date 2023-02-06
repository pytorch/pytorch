# Owner(s): ["oncall: quantization"]

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat
import torch.ao.nn.quantized.reference as nnqr
from torch.testing._internal.common_quantization import QuantizationTestCase

from torch.ao.quantization.backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    DTypeWithConstraints,
    ObservationType,
)
from torch.ao.quantization.fuser_method_mappings import _sequential_wrapper2
from torch.ao.quantization.fx.quantize_handler import _default_root_node_getter


class TestBackendConfig(QuantizationTestCase):

    # =============
    #  DTypeConfig
    # =============

    dtype_config1 = DTypeConfig(
        input_dtype=torch.quint8,
        output_dtype=torch.quint8,
        weight_dtype=torch.qint8,
        bias_dtype=torch.float
    )

    dtype_config2 = DTypeConfig(
        input_dtype=torch.float16,
        output_dtype=torch.float,
        is_dynamic=True
    )

    activation_dtype_with_constraints = DTypeWithConstraints(
        dtype=torch.quint8,
        quant_min_lower_bound=0,
        quant_max_upper_bound=127,
        scale_min_lower_bound=2 ** -12,
    )

    weight_dtype_with_constraints = DTypeWithConstraints(
        dtype=torch.qint8,
        quant_min_lower_bound=-128,
        quant_max_upper_bound=127,
        scale_min_lower_bound=2 ** -12,
    )

    dtype_config3 = DTypeConfig(
        input_dtype=activation_dtype_with_constraints,
        output_dtype=activation_dtype_with_constraints,
        weight_dtype=weight_dtype_with_constraints,
    )

    dtype_config_dict1_legacy = {
        "input_dtype": torch.quint8,
        "output_dtype": torch.quint8,
        "weight_dtype": torch.qint8,
        "bias_dtype": torch.float,
    }

    dtype_config_dict2_legacy = {
        "input_dtype": torch.float16,
        "output_dtype": torch.float,
        "is_dynamic": True,
    }

    dtype_config_dict1 = {
        "input_dtype": DTypeWithConstraints(dtype=torch.quint8),
        "output_dtype": DTypeWithConstraints(torch.quint8),
        "weight_dtype": DTypeWithConstraints(torch.qint8),
        "bias_dtype": torch.float,
    }

    dtype_config_dict2 = {
        "input_dtype": DTypeWithConstraints(dtype=torch.float16),
        "output_dtype": DTypeWithConstraints(dtype=torch.float),
        "is_dynamic": True,
    }

    dtype_config_dict3 = {
        "input_dtype": activation_dtype_with_constraints,
        "output_dtype": activation_dtype_with_constraints,
        "weight_dtype": weight_dtype_with_constraints,
    }

    def test_dtype_config_from_dict(self):
        self.assertEqual(DTypeConfig.from_dict(self.dtype_config_dict1_legacy), self.dtype_config1)
        self.assertEqual(DTypeConfig.from_dict(self.dtype_config_dict2_legacy), self.dtype_config2)
        self.assertEqual(DTypeConfig.from_dict(self.dtype_config_dict1), self.dtype_config1)
        self.assertEqual(DTypeConfig.from_dict(self.dtype_config_dict2), self.dtype_config2)
        self.assertEqual(DTypeConfig.from_dict(self.dtype_config_dict3), self.dtype_config3)

    def test_dtype_config_to_dict(self):
        self.assertEqual(self.dtype_config1.to_dict(), self.dtype_config_dict1)
        self.assertEqual(self.dtype_config2.to_dict(), self.dtype_config_dict2)
        self.assertEqual(self.dtype_config3.to_dict(), self.dtype_config_dict3)

    # ======================
    #  BackendPatternConfig
    # ======================

    _fuser_method = _sequential_wrapper2(nni.LinearReLU)

    _num_tensor_args_to_observation_type = {
        0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
        2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    }
    _input_type_to_index = {
        "bias": 0,
        "input": 1,
        "weight": 2,
    }

    def _extra_inputs_getter(self, p):
        return (torch.rand(3, 3),)

    def _get_backend_op_config1(self):
        return BackendPatternConfig((torch.nn.Linear, torch.nn.ReLU)) \
            .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
            .add_dtype_config(self.dtype_config1) \
            .add_dtype_config(self.dtype_config2) \
            .set_root_module(torch.nn.Linear) \
            .set_qat_module(nnqat.Linear) \
            .set_reference_quantized_module(nnqr.Linear) \
            .set_fused_module(nni.LinearReLU) \
            .set_fuser_method(self._fuser_method)

    def _get_backend_op_config2(self):
        return BackendPatternConfig(torch.add) \
            .add_dtype_config(self.dtype_config2) \
            ._set_root_node_getter(_default_root_node_getter) \
            ._set_extra_inputs_getter(self._extra_inputs_getter) \
            ._set_num_tensor_args_to_observation_type(self._num_tensor_args_to_observation_type) \
            ._set_input_type_to_index(self._input_type_to_index)

    def _get_backend_pattern_config_dict1(self):
        return {
            "pattern": (torch.nn.Linear, torch.nn.ReLU),
            "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
            "dtype_configs": [self.dtype_config_dict1, self.dtype_config_dict2],
            "root_module": torch.nn.Linear,
            "qat_module": nnqat.Linear,
            "reference_quantized_module_for_root": nnqr.Linear,
            "fused_module": nni.LinearReLU,
            "fuser_method": self._fuser_method,
        }

    def _get_backend_pattern_config_dict2(self):
        return {
            "pattern": torch.add,
            "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
            "dtype_configs": [self.dtype_config_dict2],
            "root_node_getter": _default_root_node_getter,
            "extra_inputs_getter": self._extra_inputs_getter,
            "num_tensor_args_to_observation_type": self._num_tensor_args_to_observation_type,
            "input_type_to_index": self._input_type_to_index,
        }

    def test_backend_op_config_set_observation_type(self):
        conf = BackendPatternConfig(torch.nn.Linear)
        self.assertEqual(conf.observation_type, ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        conf.set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
        self.assertEqual(conf.observation_type, ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)

    def test_backend_op_config_add_dtype_config(self):
        conf = BackendPatternConfig(torch.nn.Linear)
        self.assertEqual(len(conf.dtype_configs), 0)
        conf.add_dtype_config(self.dtype_config1)
        conf.add_dtype_config(self.dtype_config2)
        self.assertEqual(len(conf.dtype_configs), 2)
        self.assertEqual(conf.dtype_configs[0], self.dtype_config1)
        self.assertEqual(conf.dtype_configs[1], self.dtype_config2)

    def test_backend_op_config_set_root_module(self):
        conf = BackendPatternConfig(nni.LinearReLU)
        self.assertTrue(conf.root_module is None)
        conf.set_root_module(torch.nn.Linear)
        self.assertEqual(conf.root_module, torch.nn.Linear)

    def test_backend_op_config_set_qat_module(self):
        conf = BackendPatternConfig(torch.nn.Linear)
        self.assertTrue(conf.qat_module is None)
        conf.set_qat_module(nnqat.Linear)
        self.assertEqual(conf.qat_module, nnqat.Linear)

    def test_backend_op_config_set_reference_quantized_module(self):
        conf = BackendPatternConfig(torch.nn.Linear)
        self.assertTrue(conf.reference_quantized_module is None)
        conf.set_reference_quantized_module(nnqr.Linear)
        self.assertEqual(conf.reference_quantized_module, nnqr.Linear)

    def test_backend_op_config_set_fused_module(self):
        conf = BackendPatternConfig((torch.nn.Linear, torch.nn.ReLU))
        self.assertTrue(conf.fused_module is None)
        conf.set_fused_module(nni.LinearReLU)
        self.assertEqual(conf.fused_module, nni.LinearReLU)

    def test_backend_op_config_set_fuser_method(self):
        conf = BackendPatternConfig((torch.nn.Linear, torch.nn.ReLU))
        self.assertTrue(conf.fuser_method is None)
        conf.set_fuser_method(self._fuser_method)
        self.assertEqual(conf.fuser_method, self._fuser_method)

    def test_backend_op_config_set_root_node_getter(self):
        conf = BackendPatternConfig((torch.nn.Linear, torch.nn.ReLU))
        self.assertTrue(conf._root_node_getter is None)
        conf._set_root_node_getter(_default_root_node_getter)
        self.assertEqual(conf._root_node_getter, _default_root_node_getter)

    def test_backend_op_config_set_extra_inputs_getter(self):
        conf = BackendPatternConfig(torch.nn.Linear)
        self.assertTrue(conf._extra_inputs_getter is None)
        conf._set_extra_inputs_getter(self._extra_inputs_getter)
        self.assertEqual(conf._extra_inputs_getter, self._extra_inputs_getter)

    def test_backend_op_config_set_num_tensor_args_to_observation_type(self):
        conf = BackendPatternConfig(torch.add)
        self.assertEqual(len(conf._num_tensor_args_to_observation_type), 0)
        conf._set_num_tensor_args_to_observation_type(self._num_tensor_args_to_observation_type)
        self.assertEqual(conf._num_tensor_args_to_observation_type, self._num_tensor_args_to_observation_type)

    def test_backend_op_config_set_input_type_to_index(self):
        conf = BackendPatternConfig(torch.addmm)
        self.assertEqual(len(conf._input_type_to_index), 0)
        conf._set_input_type_to_index(self._input_type_to_index)
        self.assertEqual(conf._input_type_to_index, self._input_type_to_index)

    def test_backend_op_config_from_dict(self):
        conf_dict1 = self._get_backend_pattern_config_dict1()
        conf1 = BackendPatternConfig.from_dict(conf_dict1)
        self.assertEqual(conf1.pattern, (torch.nn.Linear, torch.nn.ReLU))
        self.assertEqual(conf1.observation_type, ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        self.assertEqual(conf1.root_module, torch.nn.Linear)
        self.assertEqual(conf1.qat_module, nnqat.Linear)
        self.assertEqual(conf1.reference_quantized_module, nnqr.Linear)
        self.assertEqual(conf1.fused_module, nni.LinearReLU)
        self.assertEqual(conf1.fuser_method, self._fuser_method)
        self.assertTrue(conf1._root_node_getter is None)
        self.assertTrue(conf1._extra_inputs_getter is None)
        self.assertEqual(len(conf1._num_tensor_args_to_observation_type), 0)
        self.assertEqual(len(conf1._input_type_to_index), 0)
        # Test temporary/internal keys
        conf_dict2 = self._get_backend_pattern_config_dict2()
        conf2 = BackendPatternConfig.from_dict(conf_dict2)
        self.assertEqual(conf2.pattern, torch.add)
        self.assertEqual(conf2.observation_type, ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        self.assertTrue(conf2.root_module is None)
        self.assertTrue(conf2.qat_module is None)
        self.assertTrue(conf2.reference_quantized_module is None)
        self.assertTrue(conf2.fused_module is None)
        self.assertTrue(conf2.fuser_method is None)
        self.assertEqual(conf2._root_node_getter, _default_root_node_getter)
        self.assertEqual(conf2._extra_inputs_getter, self._extra_inputs_getter)
        self.assertEqual(conf2._num_tensor_args_to_observation_type, self._num_tensor_args_to_observation_type)
        self.assertEqual(conf2._input_type_to_index, self._input_type_to_index)

    def test_backend_op_config_to_dict(self):
        conf1 = self._get_backend_op_config1()
        conf2 = self._get_backend_op_config2()
        conf_dict1 = self._get_backend_pattern_config_dict1()
        conf_dict2 = self._get_backend_pattern_config_dict2()
        self.assertEqual(conf1.to_dict(), conf_dict1)
        self.assertEqual(conf2.to_dict(), conf_dict2)

    # ===============
    #  BackendConfig
    # ===============

    def test_backend_config_set_name(self):
        conf = BackendConfig("name1")
        self.assertEqual(conf.name, "name1")
        conf.set_name("name2")
        self.assertEqual(conf.name, "name2")

    def test_backend_config_set_backend_pattern_config(self):
        conf = BackendConfig("name1")
        self.assertEqual(len(conf.configs), 0)
        backend_op_config1 = self._get_backend_op_config1()
        backend_op_config2 = self._get_backend_op_config2()
        conf.set_backend_pattern_config(backend_op_config1)
        self.assertEqual(conf._pattern_complex_format_to_config, {
            (torch.nn.ReLU, torch.nn.Linear): backend_op_config1,
        })
        conf.set_backend_pattern_config(backend_op_config2)
        self.assertEqual(conf._pattern_complex_format_to_config, {
            (torch.nn.ReLU, torch.nn.Linear): backend_op_config1,
            torch.add: backend_op_config2
        })

    def test_backend_config_from_dict(self):
        op1 = self._get_backend_op_config1()
        op2 = self._get_backend_op_config2()
        op_dict1 = self._get_backend_pattern_config_dict1()
        op_dict2 = self._get_backend_pattern_config_dict2()
        conf_dict = {
            "name": "name1",
            "configs": [op_dict1, op_dict2],
        }
        conf = BackendConfig.from_dict(conf_dict)
        self.assertEqual(conf.name, "name1")
        self.assertEqual(len(conf.configs), 2)
        key1 = (torch.nn.ReLU, torch.nn.Linear)
        key2 = torch.add
        self.assertTrue(key1 in conf._pattern_complex_format_to_config)
        self.assertTrue(key2 in conf._pattern_complex_format_to_config)
        self.assertEqual(conf._pattern_complex_format_to_config[key1].to_dict(), op_dict1)
        self.assertEqual(conf._pattern_complex_format_to_config[key2].to_dict(), op_dict2)

    def test_backend_config_to_dict(self):
        op1 = self._get_backend_op_config1()
        op2 = self._get_backend_op_config2()
        op_dict1 = self._get_backend_pattern_config_dict1()
        op_dict2 = self._get_backend_pattern_config_dict2()
        conf = BackendConfig("name1").set_backend_pattern_config(op1).set_backend_pattern_config(op2)
        conf_dict = {
            "name": "name1",
            "configs": [op_dict1, op_dict2],
        }
        self.assertEqual(conf.to_dict(), conf_dict)

if __name__ == '__main__':
    raise RuntimeError("This _test file is not meant to be run directly, use:\n\n"
                       "\tpython _test/_test_quantization.py TESTNAME\n\n"
                       "instead.")
