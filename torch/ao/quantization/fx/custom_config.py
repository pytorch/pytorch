from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.quant_type import QuantType, _quant_type_from_str, quant_type_to_str


__all__ = [
    "ConvertCustomConfig",
    "FuseCustomConfig",
    "PrepareCustomConfig",
    "StandaloneModuleConfigEntry",
]


# TODO: replace all usages with these constants
STANDALONE_MODULE_NAME_DICT_KEY = "standalone_module_name"
STANDALONE_MODULE_CLASS_DICT_KEY = "standalone_module_class"
FLOAT_TO_OBSERVED_DICT_KEY = "float_to_observed_custom_module_class"
OBSERVED_TO_QUANTIZED_DICT_KEY = "observed_to_quantized_custom_module_class"
NON_TRACEABLE_MODULE_NAME_DICT_KEY = "non_traceable_module_name"
NON_TRACEABLE_MODULE_CLASS_DICT_KEY = "non_traceable_module_class"
INPUT_QUANTIZED_INDEXES_DICT_KEY = "input_quantized_idxs"
OUTPUT_QUANTIZED_INDEXES_DICT_KEY = "output_quantized_idxs"
PRESERVED_ATTRIBUTES_DICT_KEY = "preserved_attributes"


@dataclass
class StandaloneModuleConfigEntry:
    # qconfig_mapping for the prepare function called in the submodule,
    # None means use qconfig from parent qconfig_mapping
    qconfig_mapping: Optional[QConfigMapping]
    example_inputs: Tuple[Any, ...]
    prepare_custom_config: Optional[PrepareCustomConfig]
    # TODO: replace this with BackendConfig
    backend_config_dict: Optional[Dict[str, Any]]


class PrepareCustomConfig:
    """
    TODO: write this
    """

    def __init__(self):
        self.standalone_module_names: Dict[str, StandaloneModuleConfigEntry] = {}
        self.standalone_module_classes: Dict[Type, StandaloneModuleConfigEntry] = {}
        self.float_to_observed_mapping: Dict[QuantType, Dict[Type, Type]] = {}
        self.non_traceable_module_names: List[str] = []
        self.non_traceable_module_classes: List[Type] = []
        self.input_quantized_indexes: List[int] = []
        self.output_quantized_indexes: List[int] = []
        self.preserved_attributes: List[str] = []

    def set_standalone_module_name(
            self,
            module_name: str,
            qconfig_mapping: Optional[QConfigMapping],
            example_inputs: Tuple[Any, ...],
            prepare_custom_config: Optional[PrepareCustomConfig],
            backend_config_dict: Optional[Dict[str, Any]]) -> PrepareCustomConfig:
        """
        TODO: write this
        """
        self.standalone_module_names[module_name] = \
            StandaloneModuleConfigEntry(qconfig_mapping, example_inputs, prepare_custom_config, backend_config_dict)
        return self

    def set_standalone_module_class(
            self,
            module_class: Type,
            qconfig_mapping: Optional[QConfigMapping],
            example_inputs: Tuple[Any, ...],
            prepare_custom_config: Optional[PrepareCustomConfig],
            backend_config_dict: Optional[Dict[str, Any]]) -> PrepareCustomConfig:
        """
        TODO: write this
        """
        self.standalone_module_classes[module_class] = \
            StandaloneModuleConfigEntry(qconfig_mapping, example_inputs, prepare_custom_config, backend_config_dict)
        return self

    def set_float_to_observed_mapping(
            self,
            float_class: Type,
            observed_class: Type,
            quant_type: QuantType = QuantType.STATIC) -> PrepareCustomConfig:
        """
        TODO: write this
        """
        if quant_type not in self.float_to_observed_mapping:
            self.float_to_observed_mapping[quant_type] = {}
        self.float_to_observed_mapping[quant_type][float_class] = observed_class
        return self

    def set_non_traceable_module_names(self, module_names: List[str]) -> PrepareCustomConfig:
        """
        TODO: write this
        """
        self.non_traceable_module_names = module_names
        return self

    def set_non_traceable_module_classes(self, module_classes: List[Type]) -> PrepareCustomConfig:
        """
        TODO: write this
        """
        self.non_traceable_module_classes = module_classes
        return self

    def set_input_quantized_indexes(self, indexes: List[int]) -> PrepareCustomConfig:
        """
        TODO: write this
        """
        self.input_quantized_indexes = indexes
        return self

    def set_output_quantized_indexes(self, indexes: List[int]) -> PrepareCustomConfig:
        """
        TODO: write this
        """
        self.output_quantized_indexes = indexes
        return self

    def set_preserved_attributes(self, attributes: List[str]) -> PrepareCustomConfig:
        """
        TODO: write this
        """
        self.preserved_attributes = attributes
        return self

    # TODO: remove this
    @classmethod
    def from_dict(cls, prepare_custom_config_dict: Dict[str, Any]) -> PrepareCustomConfig:
        """
        TODO: write this
        """
        def _get_qconfig_mapping(obj: Any, dict_key: str) -> Optional[QConfigMapping]:
            """
            Convert the given object into a QConfigMapping if possible, else throw an exception.
            """
            if isinstance(obj, QConfigMapping) or obj is None:
                return obj
            if isinstance(obj, Dict):
                return QConfigMapping.from_dict(obj)
            raise ValueError("Expected QConfigMapping in prepare_custom_config_dict[\"%s\"], got '%s'" %
                             (dict_key, type(obj)))

        def _get_prepare_custom_config(obj: Any, dict_key: str) -> Optional[PrepareCustomConfig]:
            """
            Convert the given object into a PrepareCustomConfig if possible, else throw an exception.
            """
            if isinstance(obj, PrepareCustomConfig) or obj is None:
                return obj
            if isinstance(obj, Dict):
                return PrepareCustomConfig.from_dict(obj)
            raise ValueError("Expected PrepareCustomConfig in prepare_custom_config_dict[\"%s\"], got '%s'" %
                             (dict_key, type(obj)))

        conf = cls()
        for (module_name, qconfig_dict, example_inputs, _prepare_custom_config_dict, backend_config_dict) in\
                prepare_custom_config_dict.get(STANDALONE_MODULE_NAME_DICT_KEY, []):
            qconfig_mapping = _get_qconfig_mapping(qconfig_dict, STANDALONE_MODULE_NAME_DICT_KEY)
            prepare_custom_config = _get_prepare_custom_config(_prepare_custom_config_dict, STANDALONE_MODULE_NAME_DICT_KEY)
            conf.set_standalone_module_name(
                module_name, qconfig_mapping, example_inputs, prepare_custom_config, backend_config_dict)
        for (module_class, qconfig_dict, example_inputs, _prepare_custom_config_dict, backend_config_dict) in\
                prepare_custom_config_dict.get(STANDALONE_MODULE_CLASS_DICT_KEY, []):
            qconfig_mapping = _get_qconfig_mapping(qconfig_dict, STANDALONE_MODULE_CLASS_DICT_KEY)
            prepare_custom_config = _get_prepare_custom_config(_prepare_custom_config_dict, STANDALONE_MODULE_CLASS_DICT_KEY)
            conf.set_standalone_module_class(
                module_class, qconfig_mapping, example_inputs, prepare_custom_config, backend_config_dict)
        for quant_type_name, custom_module_mapping in prepare_custom_config_dict.get(FLOAT_TO_OBSERVED_DICT_KEY, {}).items():
            quant_type = _quant_type_from_str(quant_type_name)
            for float_class, observed_class in custom_module_mapping.items():
                conf.set_float_to_observed_mapping(float_class, observed_class, quant_type)
        conf.set_non_traceable_module_names(prepare_custom_config_dict.get(NON_TRACEABLE_MODULE_NAME_DICT_KEY, []))
        conf.set_non_traceable_module_classes(prepare_custom_config_dict.get(NON_TRACEABLE_MODULE_CLASS_DICT_KEY, []))
        conf.set_input_quantized_indexes(prepare_custom_config_dict.get(INPUT_QUANTIZED_INDEXES_DICT_KEY, []))
        conf.set_output_quantized_indexes(prepare_custom_config_dict.get(OUTPUT_QUANTIZED_INDEXES_DICT_KEY, []))
        conf.set_preserved_attributes(prepare_custom_config_dict.get(PRESERVED_ATTRIBUTES_DICT_KEY, []))
        return conf

    # TODO: remove this
    def to_dict(self) -> Dict[str, Any]:
        """
        TODO: write this
        """
        def _make_tuple(key: Any, e: StandaloneModuleConfigEntry):
            qconfig_dict = e.qconfig_mapping.to_dict() if e.qconfig_mapping else None
            prepare_custom_config_dict = e.prepare_custom_config.to_dict() if e.prepare_custom_config else None
            return (key, qconfig_dict, e.example_inputs, prepare_custom_config_dict, e.backend_config_dict)

        d: Dict[str, Any] = {}
        for module_name, sm_config_entry in self.standalone_module_names.items():
            if STANDALONE_MODULE_NAME_DICT_KEY not in d:
                d[STANDALONE_MODULE_NAME_DICT_KEY] = []
            d[STANDALONE_MODULE_NAME_DICT_KEY].append(_make_tuple(module_name, sm_config_entry))
        for module_class, sm_config_entry in self.standalone_module_classes.items():
            if STANDALONE_MODULE_CLASS_DICT_KEY not in d:
                d[STANDALONE_MODULE_CLASS_DICT_KEY] = []
            d[STANDALONE_MODULE_CLASS_DICT_KEY].append(_make_tuple(module_class, sm_config_entry))
        for quant_type, float_to_observed_mapping in self.float_to_observed_mapping.items():
            if FLOAT_TO_OBSERVED_DICT_KEY not in d:
                d[FLOAT_TO_OBSERVED_DICT_KEY] = {}
            d[FLOAT_TO_OBSERVED_DICT_KEY][quant_type_to_str(quant_type)] = float_to_observed_mapping
        if len(self.non_traceable_module_names) > 0:
            d[NON_TRACEABLE_MODULE_NAME_DICT_KEY] = self.non_traceable_module_names
        if len(self.non_traceable_module_classes) > 0:
            d[NON_TRACEABLE_MODULE_CLASS_DICT_KEY] = self.non_traceable_module_classes
        if len(self.input_quantized_indexes) > 0:
            d[INPUT_QUANTIZED_INDEXES_DICT_KEY] = self.input_quantized_indexes
        if len(self.output_quantized_indexes) > 0:
            d[OUTPUT_QUANTIZED_INDEXES_DICT_KEY] = self.output_quantized_indexes
        if len(self.preserved_attributes) > 0:
            d[PRESERVED_ATTRIBUTES_DICT_KEY] = self.preserved_attributes
        return d


class ConvertCustomConfig:
    """
    TODO: write this
    """

    def __init__(self):
        self.observed_to_quantized_mapping: Dict[QuantType, Dict[Type, Type]] = {}
        self.preserved_attributes: List[str] = []

    def set_observed_to_quantized_mapping(
            self,
            observed_class: Type,
            quantized_class: Type,
            quant_type: QuantType = QuantType.STATIC) -> ConvertCustomConfig:
        """
        TODO: write this
        """
        if quant_type not in self.observed_to_quantized_mapping:
            self.observed_to_quantized_mapping[quant_type] = {}
        self.observed_to_quantized_mapping[quant_type][observed_class] = quantized_class
        return self

    def set_preserved_attributes(self, attributes: List[str]) -> ConvertCustomConfig:
        """
        TODO: write this
        """
        self.preserved_attributes = attributes
        return self

    # TODO: remove this
    @classmethod
    def from_dict(cls, convert_custom_config_dict: Dict[str, Any]) -> ConvertCustomConfig:
        """
        TODO: write this
        """
        conf = cls()
        for quant_type_name, custom_module_mapping in convert_custom_config_dict.get(OBSERVED_TO_QUANTIZED_DICT_KEY, {}).items():
            quant_type = _quant_type_from_str(quant_type_name)
            for observed_class, quantized_class in custom_module_mapping.items():
                conf.set_observed_to_quantized_mapping(observed_class, quantized_class, quant_type)
        conf.set_preserved_attributes(convert_custom_config_dict.get(PRESERVED_ATTRIBUTES_DICT_KEY, []))
        return conf


class FuseCustomConfig:
    """
    TODO: write this
    """

    def __init__(self):
        self.preserved_attributes: List[str] = []

    def set_preserved_attributes(self, attributes: List[str]) -> FuseCustomConfig:
        """
        TODO: write this
        """
        self.preserved_attributes = attributes
        return self

    # TODO: remove this
    @classmethod
    def from_dict(cls, fuse_custom_config_dict: Dict[str, Any]) -> FuseCustomConfig:
        """
        TODO: write this
        """
        conf = cls()
        conf.set_preserved_attributes(fuse_custom_config_dict.get(PRESERVED_ATTRIBUTES_DICT_KEY, []))
        return conf
