from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.ao.quantization.backend_config.observation_type import ObservationType
from torch.ao.quantization.observer import _PartialWrapper
from torch.ao.quantization.utils import Pattern


__all__ = [
    "BackendConfig",
    "BackendOpConfig",
    "DtypeConfig",
]


# DtypeConfig dict keys
INPUT_DTYPE_DICT_KEY = "input_dtype"
OUTPUT_DTYPE_DICT_KEY = "output_dtype"
WEIGHT_DTYPE_DICT_KEY = "weight_dtype"
BIAS_DTYPE_DICT_KEY = "bias_dtype"
IS_DYNAMIC_DICT_KEY = "is_dynamic"

# BackendConfig dict keys
NAME_DICT_KEY = "name"
CONFIGS_DICT_KEY = "configs"

# BackendOpConfig dict keys
PATTERN_DICT_KEY = "pattern"
OBSERVATION_TYPE_DICT_KEY = "observation_type"
DTYPE_CONFIGS_DICT_KEY = "dtype_configs"
ROOT_MODULE_DICT_KEY = "root_module"
QAT_MODULE_DICT_KEY = "qat_module"
REFERENCE_QUANTIZED_MODULE_DICT_KEY = "reference_quantized_module_for_root"
FUSER_MODULE_DICT_KEY = "fuser_module"
FUSER_METHOD_DICT_KEY = "fuser_method"
ROOT_NODE_GETTER_DICT_KEY = "root_node_getter"
EXTRA_INPUTS_GETTER_DICT_KEY = "extra_inputs_getter"
NUM_TENSOR_ARGS_TO_OBSERVATION_TYPE_DICT_KEY = "num_tensor_args_to_observation_type"
INPUT_TYPE_TO_INDEX_DICT_KEY = "input_type_to_index"
INPUT_OUTPUT_OBSERVED_DICT_KEY = "input_output_observed"
OVERWRITE_OUTPUT_FAKE_QUANTIZE_DICT_KEY = "overwrite_output_fake_quantize"
OVERWRITE_OUTPUT_OBSERVER_DICT_KEY = "overwrite_output_observer"


@dataclass
class DtypeConfig:
    """
    Data type config for ops defined in :class:`~torch.ao.quantization.backend_config.BackendConfig`.
    """
    input_dtype: Optional[torch.dtype] = None
    output_dtype: Optional[torch.dtype] = None
    weight_dtype: Optional[torch.dtype] = None
    bias_dtype: Optional[torch.dtype] = None
    is_dynamic: Optional[bool] = None

    @classmethod
    def from_dict(cls, dtype_config_dict: Dict[str, Any]) -> DtypeConfig:
        """
        Create a `DtypeConfig` from a dictionary with the following items (all optional):

            "input_dtype": torch.dtype
            "output_dtype": torch.dtype
            "weight_dtype": torch.dtype
            "bias_type": torch.dtype
            "is_dynamic": bool
        """
        input_dtype = dtype_config_dict.get(INPUT_DTYPE_DICT_KEY, None)
        output_dtype = dtype_config_dict.get(OUTPUT_DTYPE_DICT_KEY, None)
        weight_dtype = dtype_config_dict.get(WEIGHT_DTYPE_DICT_KEY, None)
        bias_dtype = dtype_config_dict.get(BIAS_DTYPE_DICT_KEY, None)
        is_dynamic = dtype_config_dict.get(IS_DYNAMIC_DICT_KEY, None)
        return cls(input_dtype, output_dtype, weight_dtype, bias_dtype, is_dynamic)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this `DtypeConfig` to a dictionary with the items described in
        :func:`~torch.ao.quantization.backend_config.DtypeConfig.from_dict`.
        """
        dtype_config_dict: Dict[str, Any] = {}
        if self.input_dtype is not None:
            dtype_config_dict[INPUT_DTYPE_DICT_KEY] = self.input_dtype
        if self.output_dtype is not None:
            dtype_config_dict[OUTPUT_DTYPE_DICT_KEY] = self.output_dtype
        if self.weight_dtype is not None:
            dtype_config_dict[WEIGHT_DTYPE_DICT_KEY] = self.weight_dtype
        if self.bias_dtype is not None:
            dtype_config_dict[BIAS_DTYPE_DICT_KEY] = self.bias_dtype
        if self.is_dynamic is not None:
            dtype_config_dict[IS_DYNAMIC_DICT_KEY] = self.is_dynamic
        return dtype_config_dict


class BackendConfig:
    # TODO: refer to native fbgemm BackendConfig once that is implemented
    """
    Config that defines which ops are supported and how they are quantized on a custom backend.

    This config specifies how reference quantized models are produced (during the prepare phase) and how they
    are lowered to implementations specific to the target backend (during the convert phase). Each op supported
    on the target backend can be individually configured through :class:`~torch.ao.quantization.backend_config.BackendOpConfig`.

    Example usage::

        import torch
        from torch.ao.quantization.backend_config import BackendConfig, BackendOpConfig, DtypeConfig, ObservationType
        from torch.ao.quantization.fuser_method_mappings import reverse_sequential_wrapper2

        weighted_int8_dtype_config = DtypeConfig(
            input_dtype=torch.quint8,
            output_dtype=torch.quint8,
            weight_dtype=torch.qint8,
            bias_type=torch.float)

        linear_config = BackendOpConfig(torch.nn.Linear) \
            .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
            .set_dtype_configs([weighted_int8_dtype_config]) \
            .set_root_module(torch.nn.Linear) \
            .set_qat_module(torch.nn.qat.Linear) \
            .set_reference_quantized_module(torch.nn.quantized._reference.Linear)

        conv_relu_config = BackendOpConfig((torch.nn.ReLU, torch.nn.Conv2d)) \
            .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
            .set_dtype_configs([weighted_int8_dtype_config]) \
            .set_fuser_module(torch.nn.intrinsic.ConvReLU2d) \
            .set_fuser_method(reverse_sequential_wrapper2(torch.nn.intrinsic.ConvReLU2d))

        backend_config = BackendConfig("my_backend") \
            .set_config(linear_config) \
            .set_config(conv_relu_config)
    """
    def __init__(self, name: str):
        self.name = name
        self.configs: Dict[Pattern, BackendOpConfig] = {}

    def set_name(self, name: str) -> BackendConfig:
        """
        Set the name of the target backend.
        """
        self.name = name
        return self

    def set_config(self, config: BackendOpConfig) -> BackendConfig:
        """
        Set the config for an op that can be run on the target backend.
        This overrides any existing config for the given op.
        """
        self.configs[config.pattern] = config
        return self

    @classmethod
    def from_dict(cls, backend_config_dict: Dict[str, Any]) -> BackendConfig:
        """
        Create a `BackendConfig` from a dictionary with the following items:

            "name": the name of the target backend
            "configs": a list of dictionaries that each represents a `BackendOpConfig`
        """
        for dict_key in [NAME_DICT_KEY, CONFIGS_DICT_KEY]:
            if dict_key not in backend_config_dict:
                raise ValueError("backend_config_dict must contain '%s'" % dict_key)
        conf = cls(backend_config_dict[NAME_DICT_KEY])
        for d in backend_config_dict[CONFIGS_DICT_KEY]:
            if isinstance(d, BackendOpConfig):
                conf.set_config(d)
            elif isinstance(d, Dict):
                conf.set_config(BackendOpConfig.from_dict(d))
            else:
                raise ValueError("Expected backend_config_dict['%s'] to be a dictionary" % CONFIGS_DICT_KEY)
        return conf

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this `BackendConfig` to a dictionary with the items described in
        :func:`~torch.ao.quantization.backend_config.BackendConfig.from_dict`.
        """
        return {
            NAME_DICT_KEY: self.name,
            CONFIGS_DICT_KEY: [c.to_dict() for c in self.configs.values()],
        }


class BackendOpConfig:
    """
    Config for ops defined in :class:`~torch.ao.quantization.backend_config.BackendConfig`.

    The user can configure how an op is handled on a given backend using the following methods:

        `set_pattern`: sets the pattern that identifies this op. The format is described in
            https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/README.md
        `set_observation_type`: sets how observers should be inserted for this op.
            See :class:`~torch.ao.quantization.backend_config.ObservationType`
        `set_dtype_configs`: sets the supported data types for this op
        `set_root_module`: sets the module that represents the root for this op
        `set_qat_module`: sets the module that represents the QAT implementation for this op
        `set_reference_quantized_module`: sets the module that represents the reference quantized
            implementation for this op's root module.
        `set_fuser_module`: sets the module that represents the fused implementation for this op
        `set_fuser_method`: sets the function that specifies how to fuse the pattern for this op

    For a detailed example usage, see :class:`~torch.ao.quantization.backend_config.BackendConfig`.
    """
    def __init__(self, pattern: Pattern):
        self.pattern = pattern
        self.observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
        self.dtype_configs: List[DtypeConfig] = []
        self.root_module: Optional[torch.nn.Module] = None
        self.qat_module: Optional[torch.nn.Module] = None
        self.reference_quantized_module: Optional[torch.nn.Module] = None
        self.fuser_module: Optional[torch.nn.Module] = None
        self.fuser_method: Optional[Callable] = None

        # Temporary/internal configs
        self._root_node_getter: Optional[Callable] = None
        self._extra_inputs_getter: Optional[Callable] = None
        self._num_tensor_args_to_observation_type: Dict[int, ObservationType] = {}
        self._input_type_to_index: Dict[str, int] = {}
        self._input_output_observed: Optional[bool] = None
        self._overwrite_output_fake_quantize: Optional[_PartialWrapper] = None
        self._overwrite_output_observer: Optional[_PartialWrapper] = None

    def set_pattern(self, pattern: Pattern) -> BackendOpConfig:
        """
        Set the pattern that identifies the op.
        """
        self.pattern = pattern
        return self

    def set_observation_type(self, observation_type: ObservationType) -> BackendOpConfig:
        """
        Set how observers should be inserted for this op.
        """
        self.observation_type = observation_type
        return self

    def set_dtype_configs(self, dtype_configs: List[DtypeConfig]) -> BackendOpConfig:
        """
        Set the supported data types for this op.
        """
        self.dtype_configs = dtype_configs
        return self

    def set_root_module(self, root_module: torch.nn.Module) -> BackendOpConfig:
        """
        Set the module that represents the root for this op.
        For example, the root module for :class:`torch.nn.intrinsic.LinearReLU` should be :class:`torch.nn.Linear`.
        """
        self.root_module = root_module
        return self

    def set_qat_module(self, qat_module: torch.nn.Module) -> BackendOpConfig:
        """
        Set the module that represents the QAT implementation for this op.
        """
        self.qat_module = qat_module
        return self

    def set_reference_quantized_module(self, reference_quantized_module: torch.nn.Module) -> BackendOpConfig:
        """
        Set the module that represents the reference quantized implementation for this op's root module.
        """
        self.reference_quantized_module = reference_quantized_module
        return self

    def set_fuser_module(self, fuser_module: torch.nn.Module) -> BackendOpConfig:
        """
        Set the module that represents the fused implementation for this op.
        """
        self.fuser_module = fuser_module
        return self

    def set_fuser_method(self, fuser_method: Callable) -> BackendOpConfig:
        """
        Set the function that specifies how to fuse the pattern for this op.
        """
        self.fuser_method = fuser_method
        return self

    def _set_root_node_getter(self, root_node_getter: Callable) -> BackendOpConfig:
        self._root_node_getter = root_node_getter
        return self

    def _set_extra_inputs_getter(self, extra_inputs_getter: Callable) -> BackendOpConfig:
        self._extra_inputs_getter = extra_inputs_getter
        return self

    def _set_num_tensor_args_to_observation_type(
            self, num_tensor_args_to_observation_type: Dict[int, ObservationType]) -> BackendOpConfig:
        self._num_tensor_args_to_observation_type = num_tensor_args_to_observation_type
        return self

    def _set_input_type_to_index(self, input_type_to_index: Dict[str, int]) -> BackendOpConfig:
        self._input_type_to_index = input_type_to_index
        return self

    def _set_input_output_observed(self, input_output_observed: bool) -> BackendOpConfig:
        self._input_output_observed = input_output_observed
        return self

    def _set_overwrite_output_fake_quantize(self, overwrite_output_fake_quantize: _PartialWrapper) -> BackendOpConfig:
        self._overwrite_output_fake_quantize = overwrite_output_fake_quantize
        return self

    def _set_overwrite_output_observer(self, overwrite_output_observer: _PartialWrapper) -> BackendOpConfig:
        self._overwrite_output_observer = overwrite_output_observer
        return self

    @classmethod
    def from_dict(cls, backend_op_config_dict: Dict[str, Any]) -> BackendOpConfig:
        """
        Create a `BackendOpConfig` from a dictionary with the following items:

            "pattern": the pattern that identifies this op
            "observation_type": the :class:`~torch.ao.quantization.backend_config.ObservationType` that specifies how
                observers should be inserted for this op
            "dtype_configs": a list of dictionaries that represents :class:`~torch.ao.quantization.backend_config.DtypeConfig`s
            "root_module": a :class:`torch.nn.Module` that represents the root for this op
            "qat_module": a :class:`torch.nn.Module` that represents the QAT implementation for this op
            "reference_quantized_module": a :class:`torch.nn.Module` that represents the reference quantized
                implementation for this op's root module.
            "fuser_module": a :class:`torch.nn.Module` that represents the fused implementation for this op
            "fuser_method": a function that specifies how to fuse the pattern for this op
        """
        def _get_dtype_config(obj: Any) -> DtypeConfig:
            """
            Convert the given object into a `DtypeConfig` if possible, else throw an exception.
            """
            if isinstance(obj, DtypeConfig):
                return obj
            if isinstance(obj, Dict):
                return DtypeConfig.from_dict(obj)
            raise ValueError("Expected a list of DtypeConfigs in backend_op_config_dict[\"%s\"], got '%s'" %
                             (DTYPE_CONFIGS_DICT_KEY, type(obj)))

        if PATTERN_DICT_KEY not in backend_op_config_dict:
            raise ValueError("backend_op_config_dict must contain '%s'" % PATTERN_DICT_KEY)
        conf = cls(backend_op_config_dict[PATTERN_DICT_KEY])
        if OBSERVATION_TYPE_DICT_KEY in backend_op_config_dict:
            conf.set_observation_type(backend_op_config_dict[OBSERVATION_TYPE_DICT_KEY])
        conf.set_dtype_configs([
            _get_dtype_config(d) for d in backend_op_config_dict.get(DTYPE_CONFIGS_DICT_KEY, [])
        ])
        conf.set_root_module(backend_op_config_dict.get(ROOT_MODULE_DICT_KEY, None))
        conf.set_qat_module(backend_op_config_dict.get(QAT_MODULE_DICT_KEY, None))
        conf.set_reference_quantized_module(backend_op_config_dict.get(REFERENCE_QUANTIZED_MODULE_DICT_KEY, None))
        conf.set_fuser_module(backend_op_config_dict.get(FUSER_MODULE_DICT_KEY, None))
        conf.set_fuser_method(backend_op_config_dict.get(FUSER_METHOD_DICT_KEY, None))
        conf._set_root_node_getter(backend_op_config_dict.get(ROOT_NODE_GETTER_DICT_KEY, None))
        conf._set_extra_inputs_getter(backend_op_config_dict.get(EXTRA_INPUTS_GETTER_DICT_KEY, None))
        conf._set_num_tensor_args_to_observation_type(backend_op_config_dict.get(NUM_TENSOR_ARGS_TO_OBSERVATION_TYPE_DICT_KEY, {}))
        conf._set_input_type_to_index(backend_op_config_dict.get(INPUT_TYPE_TO_INDEX_DICT_KEY, {}))
        conf._set_input_output_observed(backend_op_config_dict.get(INPUT_OUTPUT_OBSERVED_DICT_KEY, None))
        conf._set_overwrite_output_fake_quantize(backend_op_config_dict.get(OVERWRITE_OUTPUT_FAKE_QUANTIZE_DICT_KEY, None))
        conf._set_overwrite_output_observer(backend_op_config_dict.get(OVERWRITE_OUTPUT_OBSERVER_DICT_KEY, None))
        return conf

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this `BackendOpConfig` to a dictionary with the items described in
        :func:`~torch.ao.quantization.backend_config.BackendOpConfig.from_dict`.
        """
        backend_op_config_dict: Dict[str, Any] = {
            PATTERN_DICT_KEY: self.pattern,
            OBSERVATION_TYPE_DICT_KEY: self.observation_type,
            DTYPE_CONFIGS_DICT_KEY: [c.to_dict() for c in self.dtype_configs],
        }
        if self.root_module is not None:
            backend_op_config_dict[ROOT_MODULE_DICT_KEY] = self.root_module
        if self.qat_module is not None:
            backend_op_config_dict[QAT_MODULE_DICT_KEY] = self.qat_module
        if self.reference_quantized_module is not None:
            backend_op_config_dict[REFERENCE_QUANTIZED_MODULE_DICT_KEY] = self.reference_quantized_module
        if self.fuser_module is not None:
            backend_op_config_dict[FUSER_MODULE_DICT_KEY] = self.fuser_module
        if self.fuser_method is not None:
            backend_op_config_dict[FUSER_METHOD_DICT_KEY] = self.fuser_method
        if self._root_node_getter is not None:
            backend_op_config_dict[ROOT_NODE_GETTER_DICT_KEY] = self._root_node_getter
        if self._extra_inputs_getter is not None:
            backend_op_config_dict[EXTRA_INPUTS_GETTER_DICT_KEY] = self._extra_inputs_getter
        if len(self._num_tensor_args_to_observation_type) > 0:
            backend_op_config_dict[NUM_TENSOR_ARGS_TO_OBSERVATION_TYPE_DICT_KEY] = self._num_tensor_args_to_observation_type
        if len(self._input_type_to_index) > 0:
            backend_op_config_dict[INPUT_TYPE_TO_INDEX_DICT_KEY] = self._input_type_to_index
        if self._input_output_observed is not None:
            backend_op_config_dict[INPUT_OUTPUT_OBSERVED_DICT_KEY] = self._input_output_observed
        if self._overwrite_output_fake_quantize is not None:
            backend_op_config_dict[OVERWRITE_OUTPUT_FAKE_QUANTIZE_DICT_KEY] = self._overwrite_output_fake_quantize
        if self._overwrite_output_observer is not None:
            backend_op_config_dict[OVERWRITE_OUTPUT_OBSERVER_DICT_KEY] = self._overwrite_output_observer
        return backend_op_config_dict
