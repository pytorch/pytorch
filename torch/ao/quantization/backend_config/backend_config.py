from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch
from torch.ao.quantization.utils import Pattern
from enum import Enum


__all__ = [
    "BackendConfig",
    "BackendPatternConfig",
    "DTypeConfig",
    "DTypeWithConstraints",
    "ObservationType",
]


# DTypeConfig dict keys
INPUT_DTYPE_DICT_KEY = "input_dtype"
OUTPUT_DTYPE_DICT_KEY = "output_dtype"
WEIGHT_DTYPE_DICT_KEY = "weight_dtype"
BIAS_DTYPE_DICT_KEY = "bias_dtype"
IS_DYNAMIC_DICT_KEY = "is_dynamic"

# BackendConfig dict keys
NAME_DICT_KEY = "name"
CONFIGS_DICT_KEY = "configs"

# BackendPatternConfig dict keys
PATTERN_DICT_KEY = "pattern"
OBSERVATION_TYPE_DICT_KEY = "observation_type"
DTYPE_CONFIGS_DICT_KEY = "dtype_configs"
ROOT_MODULE_DICT_KEY = "root_module"
QAT_MODULE_DICT_KEY = "qat_module"
REFERENCE_QUANTIZED_MODULE_DICT_KEY = "reference_quantized_module_for_root"
FUSED_MODULE_DICT_KEY = "fused_module"
FUSER_METHOD_DICT_KEY = "fuser_method"
ROOT_NODE_GETTER_DICT_KEY = "root_node_getter"
EXTRA_INPUTS_GETTER_DICT_KEY = "extra_inputs_getter"
NUM_TENSOR_ARGS_TO_OBSERVATION_TYPE_DICT_KEY = "num_tensor_args_to_observation_type"
INPUT_TYPE_TO_INDEX_DICT_KEY = "input_type_to_index"
INPUT_OUTPUT_OBSERVED_DICT_KEY = "input_output_observed"


# TODO: maybe rename this to something that's not related to observer
# e.g. QParamsType
class ObservationType(Enum):
    """ An enum that represents different ways of how an operator/operator pattern
    should be observed
    """

    OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT = 0
    """this means input and output are observed with different observers, based
    on qconfig.activation
    example: conv, linear, softmax
    """

    OUTPUT_SHARE_OBSERVER_WITH_INPUT = 1
    """this means the output will use the same observer instance as input, based
    on qconfig.activation
    example: torch.cat, maxpool
    """


@dataclass
class DTypeWithConstraints:
    """
    Config for specifying additional constraints for a given dtype, such as quantization
    value ranges, scale value ranges, and fixed quantization params, to be used in
    :class:`~torch.ao.quantization.backend_config.DTypeConfig`.
    """
    dtype: Optional[torch.dtype] = None
    quant_min_lower_bound: Union[int, float, None] = None
    quant_max_upper_bound: Union[int, float, None] = None
    scale_min_lower_bound: Union[int, float, None] = None
    scale_max_upper_bound: Union[int, float, None] = None
    fixed_scale: Optional[float] = None
    fixed_zero_point: Optional[int] = None


@dataclass
class DTypeConfig:
    """
    Config for the set of supported input/output activation, weight, and bias data types for the
    patterns defined in :class:`~torch.ao.quantization.backend_config.BackendConfig`.

    Example usage::

        >>> dtype_config1 = DTypeConfig(
        ...     input_dtype=torch.quint8,
        ...     output_dtype=torch.quint8,
        ...     weight_dtype=torch.qint8,
        ...     bias_dtype=torch.float)

        >>> dtype_config2 = DTypeConfig(
        ...     input_dtype=DTypeWithConstraints(
        ...         dtype=torch.quint8,
        ...         quant_min_lower_bound=0,
        ...         quant_max_upper_bound=255,
        ...     ),
        ...     output_dtype=DTypeWithConstraints(
        ...         dtype=torch.quint8,
        ...         quant_min_lower_bound=0,
        ...         quant_max_upper_bound=255,
        ...     ),
        ...     weight_dtype=DTypeWithConstraints(
        ...         dtype=torch.qint8,
        ...         quant_min_lower_bound=-128,
        ...         quant_max_upper_bound=127,
        ...     ),
        ...     bias_dtype=torch.float)

        >>> dtype_config1.input_dtype
        torch.quint8

        >>> dtype_config2.input_dtype
        torch.quint8

        >>> dtype_config2.input_dtype_with_constraints
        DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, \
scale_min_lower_bound=None, scale_max_upper_bound=None)
    """
    input_dtype_with_constraints: DTypeWithConstraints
    output_dtype_with_constraints: DTypeWithConstraints
    weight_dtype_with_constraints: DTypeWithConstraints
    bias_dtype: Optional[torch.dtype]
    is_dynamic: Optional[bool]

    def __init__(
        self,
        input_dtype: Union[torch.dtype, DTypeWithConstraints, None] = None,
        output_dtype: Union[torch.dtype, DTypeWithConstraints, None] = None,
        weight_dtype: Union[torch.dtype, DTypeWithConstraints, None] = None,
        bias_dtype: Optional[torch.dtype] = None,
        is_dynamic: Optional[bool] = None,
    ):
        if isinstance(input_dtype, DTypeWithConstraints):
            self.input_dtype_with_constraints = input_dtype
        else:
            self.input_dtype_with_constraints = DTypeWithConstraints(dtype=input_dtype)

        if isinstance(output_dtype, DTypeWithConstraints):
            self.output_dtype_with_constraints = output_dtype
        else:
            self.output_dtype_with_constraints = DTypeWithConstraints(dtype=output_dtype)

        if isinstance(weight_dtype, DTypeWithConstraints):
            self.weight_dtype_with_constraints = weight_dtype
        else:
            self.weight_dtype_with_constraints = DTypeWithConstraints(dtype=weight_dtype)

        self.bias_dtype = bias_dtype
        self.is_dynamic = is_dynamic

    @property
    def input_dtype(self) -> Optional[torch.dtype]:
        return self.input_dtype_with_constraints.dtype

    @property
    def output_dtype(self) -> Optional[torch.dtype]:
        return self.output_dtype_with_constraints.dtype

    @property
    def weight_dtype(self) -> Optional[torch.dtype]:
        return self.weight_dtype_with_constraints.dtype

    @classmethod
    def from_dict(cls, dtype_config_dict: Dict[str, Any]) -> DTypeConfig:
        """
        Create a ``DTypeConfig`` from a dictionary with the following items (all optional):
            "input_dtype": torch.dtype or ``DTypeWithConstraints``
            "output_dtype": torch.dtype or ``DTypeWithConstraints``
            "weight_dtype": torch.dtype or ``DTypeWithConstraints``
            "bias_type": torch.dtype
            "is_dynamic": bool
        """
        input_dtype = dtype_config_dict.get(INPUT_DTYPE_DICT_KEY, None)
        if input_dtype is not None and not isinstance(input_dtype, (torch.dtype, DTypeWithConstraints)):
            raise ValueError("Expected input_dtype to be a torch.dtype or DTypeWithConstraints")
        output_dtype = dtype_config_dict.get(OUTPUT_DTYPE_DICT_KEY, None)
        if output_dtype is not None and not isinstance(output_dtype, (torch.dtype, DTypeWithConstraints)):
            raise ValueError("Expected output_dtype to be a torch.dtype or DTypeWithConstraints")
        weight_dtype = dtype_config_dict.get(WEIGHT_DTYPE_DICT_KEY, None)
        if weight_dtype is not None and not isinstance(weight_dtype, (torch.dtype, DTypeWithConstraints)):
            raise ValueError("Expected weight_dtype to be a torch.dtype or DTypeWithConstraints")
        bias_dtype = dtype_config_dict.get(BIAS_DTYPE_DICT_KEY, None)
        is_dynamic = dtype_config_dict.get(IS_DYNAMIC_DICT_KEY, None)
        return cls(input_dtype, output_dtype, weight_dtype, bias_dtype, is_dynamic)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this ``DTypeConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.backend_config.DTypeConfig.from_dict`.
        """
        dtype_config_dict: Dict[str, Any] = {}
        if self.input_dtype is not None:
            dtype_config_dict[INPUT_DTYPE_DICT_KEY] = self.input_dtype_with_constraints
        if self.output_dtype is not None:
            dtype_config_dict[OUTPUT_DTYPE_DICT_KEY] = self.output_dtype_with_constraints
        if self.weight_dtype is not None:
            dtype_config_dict[WEIGHT_DTYPE_DICT_KEY] = self.weight_dtype_with_constraints
        if self.bias_dtype is not None:
            dtype_config_dict[BIAS_DTYPE_DICT_KEY] = self.bias_dtype
        if self.is_dynamic is not None:
            dtype_config_dict[IS_DYNAMIC_DICT_KEY] = self.is_dynamic
        return dtype_config_dict


class BackendConfig:
    # TODO: refer to NativeBackendConfig once that is implemented
    """Config that defines the set of patterns that can be quantized on a given backend, and how reference
    quantized models can be produced from these patterns.

    A pattern in this context refers to a module, a functional, an operator, or a directed acyclic graph
    of the above. Each pattern supported on the target backend can be individually configured through
    :class:`~torch.ao.quantization.backend_config.BackendPatternConfig` in terms of:

    (1) The supported input/output activation, weight, and bias data types

    (2) How observers and quant/dequant ops are inserted in order to construct the reference pattern, and

    (3) (Optionally) Fusion, QAT, and reference module mappings.

    The format of the patterns is described in:
    https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/README.md

    Example usage::

        import torch
        from torch.ao.quantization.backend_config import BackendConfig, BackendPatternConfig, DTypeConfig, ObservationType
        from torch.ao.quantization.fuser_method_mappings import reverse_sequential_wrapper2

        weighted_int8_dtype_config = DTypeConfig(
            input_dtype=torch.quint8,
            output_dtype=torch.quint8,
            weight_dtype=torch.qint8,
            bias_type=torch.float)

        linear_config = BackendPatternConfig(torch.nn.Linear) \
            .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
            .add_dtype_config(weighted_int8_dtype_config) \
            .set_root_module(torch.nn.Linear) \
            .set_qat_module(torch.nn.qat.Linear) \
            .set_reference_quantized_module(torch.nn.quantized._reference.Linear)

        conv_relu_config = BackendPatternConfig((torch.nn.ReLU, torch.nn.Conv2d)) \
            .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
            .add_dtype_config(weighted_int8_dtype_config) \
            .set_fused_module(torch.nn.intrinsic.ConvReLU2d) \
            .set_fuser_method(reverse_sequential_wrapper2(torch.nn.intrinsic.ConvReLU2d))

        backend_config = BackendConfig("my_backend") \
            .set_backend_pattern_config(linear_config) \
            .set_backend_pattern_config(conv_relu_config)

    """
    def __init__(self, name: str = ""):
        self.name = name
        self.configs: Dict[Pattern, BackendPatternConfig] = {}

    def set_name(self, name: str) -> BackendConfig:
        """
        Set the name of the target backend.
        """
        self.name = name
        return self

    def set_backend_pattern_config(self, config: BackendPatternConfig) -> BackendConfig:
        """
        Set the config for an pattern that can be run on the target backend.
        This overrides any existing config for the given pattern.
        """
        self.configs[config.pattern] = config
        return self

    def set_backend_pattern_configs(self, configs: List[BackendPatternConfig]) -> BackendConfig:
        """
        Set the configs for patterns that can be run on the target backend.
        This overrides any existing config for a given pattern if it was previously registered already.
        """
        for conf in configs:
            self.set_backend_pattern_config(conf)
        return self

    @classmethod
    def from_dict(cls, backend_config_dict: Dict[str, Any]) -> BackendConfig:
        """
        Create a ``BackendConfig`` from a dictionary with the following items:

            "name": the name of the target backend

            "configs": a list of dictionaries that each represents a `BackendPatternConfig`

        """
        conf = cls(backend_config_dict.get(NAME_DICT_KEY, ""))
        for d in backend_config_dict.get(CONFIGS_DICT_KEY, []):
            if isinstance(d, BackendPatternConfig):
                conf.set_backend_pattern_config(d)
            elif isinstance(d, Dict):
                conf.set_backend_pattern_config(BackendPatternConfig.from_dict(d))
            else:
                raise ValueError("Expected backend_config_dict['%s'] to be a dictionary" % CONFIGS_DICT_KEY)
        return conf

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this ``BackendConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.backend_config.BackendConfig.from_dict`.
        """
        return {
            NAME_DICT_KEY: self.name,
            CONFIGS_DICT_KEY: [c.to_dict() for c in self.configs.values()],
        }


class BackendPatternConfig:
    """
    Config for ops defined in :class:`~torch.ao.quantization.backend_config.BackendConfig`.
    For a detailed example usage, see :class:`~torch.ao.quantization.backend_config.BackendConfig`.

    """
    def __init__(self, pattern: Pattern):
        self.pattern = pattern
        self.observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
        self.dtype_configs: List[DTypeConfig] = []
        self.root_module: Optional[Type[torch.nn.Module]] = None
        self.qat_module: Optional[Type[torch.nn.Module]] = None
        self.reference_quantized_module: Optional[Type[torch.nn.Module]] = None
        self.fused_module: Optional[Type[torch.nn.Module]] = None
        self.fuser_method: Optional[Callable] = None

        # Temporary/internal configs
        self._root_node_getter: Optional[Callable] = None
        self._extra_inputs_getter: Optional[Callable] = None
        self._num_tensor_args_to_observation_type: Dict[int, ObservationType] = {}
        self._input_type_to_index: Dict[str, int] = {}
        self._input_output_observed: Optional[bool] = None

    def set_observation_type(self, observation_type: ObservationType) -> BackendPatternConfig:
        """
        Set how observers should be inserted in the graph for this pattern.
        There are two observation types:

            `OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT` (default): the output observer instance will be
            different from the input. This is the most common observation type.

            `OUTPUT_SHARE_OBSERVER_WITH_INPUT`: the output observer instance will be the same as the input.
            This is useful for operators like `cat`.

        Note: This will be renamed in the near future, since we will soon insert QuantDeQuantStubs with
        observers (and fake quantizes) attached instead of observers themselves.
        """
        self.observation_type = observation_type
        return self

    def add_dtype_config(self, dtype_config: DTypeConfig) -> BackendPatternConfig:
        """
        Add a set of supported input/output activation, weight, and bias data types for this pattern.
        """
        self.dtype_configs.append(dtype_config)
        return self

    def set_dtype_configs(self, dtype_configs: List[DTypeConfig]) -> BackendPatternConfig:
        """
        Set the supported input/output activation, weight, and bias data types for this pattern,
        overriding all previously registered data types.
        """
        self.dtype_configs = dtype_configs
        return self

    def set_root_module(self, root_module: Type[torch.nn.Module]) -> BackendPatternConfig:
        """
        Set the module that represents the root for this pattern.
        For example, the root module for :class:`torch.nn.intrinsic.LinearReLU` should be :class:`torch.nn.Linear`.
        """
        self.root_module = root_module
        return self

    def set_qat_module(self, qat_module: Type[torch.nn.Module]) -> BackendPatternConfig:
        """
        Set the module that represents the QAT implementation for this pattern.
        """
        self.qat_module = qat_module
        return self

    def set_reference_quantized_module(self, reference_quantized_module: Type[torch.nn.Module]) -> BackendPatternConfig:
        """
        Set the module that represents the reference quantized implementation for this pattern's root module.
        """
        self.reference_quantized_module = reference_quantized_module
        return self

    def set_fused_module(self, fused_module: Type[torch.nn.Module]) -> BackendPatternConfig:
        """
        Set the module that represents the fused implementation for this pattern.
        """
        self.fused_module = fused_module
        return self

    def set_fuser_method(self, fuser_method: Callable) -> BackendPatternConfig:
        """
        Set the function that specifies how to fuse the pattern for this pattern.

        The first argument of this function should be `is_qat`, and the rest of the arguments
        should be the items in the tuple pattern, e.g. (`torch.nn.ReLU`, `torch.nn.Linear`)
        will have a function with three arguments, `is_qat`, `relu`, and `linear`.
        The return value of this function should be the resulting fused module.
        """
        self.fuser_method = fuser_method
        return self

    def _set_root_node_getter(self, root_node_getter: Callable) -> BackendPatternConfig:
        self._root_node_getter = root_node_getter
        return self

    def _set_extra_inputs_getter(self, extra_inputs_getter: Callable) -> BackendPatternConfig:
        self._extra_inputs_getter = extra_inputs_getter
        return self

    def _set_num_tensor_args_to_observation_type(
            self, num_tensor_args_to_observation_type: Dict[int, ObservationType]) -> BackendPatternConfig:
        self._num_tensor_args_to_observation_type = num_tensor_args_to_observation_type
        return self

    def _set_input_type_to_index(self, input_type_to_index: Dict[str, int]) -> BackendPatternConfig:
        self._input_type_to_index = input_type_to_index
        return self

    def _set_input_output_observed(self, input_output_observed: bool) -> BackendPatternConfig:
        self._input_output_observed = input_output_observed
        return self

    @classmethod
    def from_dict(cls, backend_pattern_config_dict: Dict[str, Any]) -> BackendPatternConfig:
        """
        Create a ``BackendPatternConfig`` from a dictionary with the following items:

            "pattern": the pattern being configured
            "observation_type": the :class:`~torch.ao.quantization.backend_config.ObservationType` that specifies how
            observers should be inserted for this pattern
            "dtype_configs": a list of dictionaries that represents :class:`~torch.ao.quantization.backend_config.DTypeConfig` s
            "root_module": a :class:`torch.nn.Module` that represents the root for this pattern
            "qat_module": a :class:`torch.nn.Module` that represents the QAT implementation for this pattern
            "reference_quantized_module": a :class:`torch.nn.Module` that represents the reference quantized
            implementation for this pattern's root module.
            "fused_module": a :class:`torch.nn.Module` that represents the fused implementation for this pattern
            "fuser_method": a function that specifies how to fuse the pattern for this pattern

        """
        def _get_dtype_config(obj: Any) -> DTypeConfig:
            """
            Convert the given object into a ``DTypeConfig`` if possible, else throw an exception.
            """
            if isinstance(obj, DTypeConfig):
                return obj
            if isinstance(obj, Dict):
                return DTypeConfig.from_dict(obj)
            raise ValueError("Expected a list of DTypeConfigs in backend_pattern_config_dict[\"%s\"], got '%s'" %
                             (DTYPE_CONFIGS_DICT_KEY, type(obj)))

        if PATTERN_DICT_KEY not in backend_pattern_config_dict:
            raise ValueError("backend_pattern_config_dict must contain '%s'" % PATTERN_DICT_KEY)
        conf = cls(backend_pattern_config_dict[PATTERN_DICT_KEY])
        if OBSERVATION_TYPE_DICT_KEY in backend_pattern_config_dict:
            conf.set_observation_type(backend_pattern_config_dict[OBSERVATION_TYPE_DICT_KEY])
        for d in backend_pattern_config_dict.get(DTYPE_CONFIGS_DICT_KEY, []):
            conf.add_dtype_config(_get_dtype_config(d))
        conf.set_root_module(backend_pattern_config_dict.get(ROOT_MODULE_DICT_KEY, None))
        conf.set_qat_module(backend_pattern_config_dict.get(QAT_MODULE_DICT_KEY, None))
        conf.set_reference_quantized_module(backend_pattern_config_dict.get(REFERENCE_QUANTIZED_MODULE_DICT_KEY, None))
        conf.set_fused_module(backend_pattern_config_dict.get(FUSED_MODULE_DICT_KEY, None))
        conf.set_fuser_method(backend_pattern_config_dict.get(FUSER_METHOD_DICT_KEY, None))
        conf._set_root_node_getter(backend_pattern_config_dict.get(ROOT_NODE_GETTER_DICT_KEY, None))
        conf._set_extra_inputs_getter(backend_pattern_config_dict.get(EXTRA_INPUTS_GETTER_DICT_KEY, None))
        conf._set_num_tensor_args_to_observation_type(
            backend_pattern_config_dict.get(NUM_TENSOR_ARGS_TO_OBSERVATION_TYPE_DICT_KEY, {}))
        conf._set_input_type_to_index(backend_pattern_config_dict.get(INPUT_TYPE_TO_INDEX_DICT_KEY, {}))
        conf._set_input_output_observed(backend_pattern_config_dict.get(INPUT_OUTPUT_OBSERVED_DICT_KEY, None))
        return conf

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this ``BackendPatternConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.backend_config.BackendPatternConfig.from_dict`.
        """
        backend_pattern_config_dict: Dict[str, Any] = {
            PATTERN_DICT_KEY: self.pattern,
            OBSERVATION_TYPE_DICT_KEY: self.observation_type,
            DTYPE_CONFIGS_DICT_KEY: [c.to_dict() for c in self.dtype_configs],
        }
        if self.root_module is not None:
            backend_pattern_config_dict[ROOT_MODULE_DICT_KEY] = self.root_module
        if self.qat_module is not None:
            backend_pattern_config_dict[QAT_MODULE_DICT_KEY] = self.qat_module
        if self.reference_quantized_module is not None:
            backend_pattern_config_dict[REFERENCE_QUANTIZED_MODULE_DICT_KEY] = self.reference_quantized_module
        if self.fused_module is not None:
            backend_pattern_config_dict[FUSED_MODULE_DICT_KEY] = self.fused_module
        if self.fuser_method is not None:
            backend_pattern_config_dict[FUSER_METHOD_DICT_KEY] = self.fuser_method
        if self._root_node_getter is not None:
            backend_pattern_config_dict[ROOT_NODE_GETTER_DICT_KEY] = self._root_node_getter
        if self._extra_inputs_getter is not None:
            backend_pattern_config_dict[EXTRA_INPUTS_GETTER_DICT_KEY] = self._extra_inputs_getter
        if len(self._num_tensor_args_to_observation_type) > 0:
            backend_pattern_config_dict[NUM_TENSOR_ARGS_TO_OBSERVATION_TYPE_DICT_KEY] = self._num_tensor_args_to_observation_type
        if len(self._input_type_to_index) > 0:
            backend_pattern_config_dict[INPUT_TYPE_TO_INDEX_DICT_KEY] = self._input_type_to_index
        if self._input_output_observed is not None:
            backend_pattern_config_dict[INPUT_OUTPUT_OBSERVED_DICT_KEY] = self._input_output_observed
        return backend_pattern_config_dict
