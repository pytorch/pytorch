from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Type

from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.quant_type import QuantType, quant_type_from_str


__all__ = [
    "ConvertCustomConfig",
    "PrepareCustomConfig",
    "StandaloneModuleNameConfigEntry",
    "StandaloneModuleClassConfigEntry",
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
class StandaloneModuleNameConfigEntry:
    module_name: str
    # qconfig_dict for the prepare function called in the submodule,
    # None means use qconfig from parent qconfig_dict
    qconfig_mapping: QConfigMapping
    prepare_custom_config: PrepareCustomConfig
    # TODO: replace this with BackendConfig
    backend_config_dict: Dict[str, Any]


@dataclass
class StandaloneModuleClassConfigEntry:
    module_class: Type
    # qconfig_dict for the prepare function called in the submodule,
    # None means use qconfig from parent qconfig_dict
    # TODO: replace this with QConfigMapping
    qconfig_mapping: QConfigMapping
    prepare_custom_config: PrepareCustomConfig
    # TODO: replace this with BackendConfig
    backend_config_dict: Dict[str, Any]


class PrepareCustomConfig:
    """
    TODO: write this
    """

    def __init__(self):
        self.standalone_module_name_configs: List[StandaloneModuleNameConfigEntry] = []
        self.standalone_module_class_configs: List[StandaloneModuleClassConfigEntry] = []
        self.float_to_observed_mapping: Dict[QuantType, Dict[Type, Type]] = {}
        self.non_traceable_module_names: List[str] = []
        self.non_traceable_module_classes: List[Type] = []
        self.input_quantized_indexes: List[int] = []
        self.output_quantized_indexes: List[int] = []
        self.preserved_attributes: List[str] = []

    def set_standalone_module_name(
            self,
            module_name: str,
            qconfig_mapping: QConfigMapping,
            prepare_custom_config: PrepareCustomConfig,
            backend_config: Dict[str, Any]) -> PrepareCustomConfig:
        """
        TODO: write this
        """
        self.standalone_module_name_configs.append(
            StandaloneModuleNameConfigEntry(
                module_name, qconfig_mapping, prepare_custom_config, backend_config_dict))
        return self

    def set_standalone_module_class(
            self,
            module_class: str,
            qconfig_mapping: QConfigMapping,
            prepare_custom_config: PrepareCustomConfig,
            backend_config: Dict[str, Any]) -> PrepareCustomConfig:
        """
        TODO: write this
        """
        self.standalone_module_class_configs.append(
            StandaloneModuleClassConfigEntry(
                module_class, qconfig_mapping, prepare_custom_config, backend_config_dict))
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
        def _get_qconfig_mapping(obj: Any, dict_key: str) -> QConfigMapping:
            """
            Convert the given object into a QConfigMapping if possible, else throw an exception.
            """
            if isinstance(obj, QConfigMapping):
                return obj
            if isinstance(obj, Dict[str, Any]):
                return QConfigMapping.from_dict(obj)
            raise ValueError("Expected QConfigMapping in prepare_custom_config_dict[\"%s\"], got '%s'" %
                             (dict_key, type(obj)))

        def _get_prepare_custom_config(obj: Any, dict_key: str) -> PrepareCustomConfig:
            """
            Convert the given object into a PrepareCustomConfig if possible, else throw an exception.
            """
            if isinstance(obj, PrepareCustomConfig):
                return obj
            if isinstance(obj, Dict[str, Any]):
                return PrepareCustomConfig.from_dict(obj)
            raise ValueError("Expected PrepareCustomConfig in prepare_custom_config_dict[\"%s\"], got '%s'" %
                             (dict_key, type(obj)))

        conf = cls()
        for (module_name, qconfig_dict, prepare_custom_config_dict, backend_config_dict) in\
                prepare_custom_config_dict.get(STANDALONE_MODULE_NAME_DICT_KEY, []):
            qconfig_mapping = _get_qconfig_mapping(qconfig_dict, STANDALONE_MODULE_NAME_DICT_KEY)
            prepare_custom_config = _get_prepare_custom_config(prepare_custom_config_dict, STANDALONE_MODULE_NAME_DICT_KEY)
            conf.set_standalone_module_name(module_name, qconfig_mapping, prepare_custom_config, backend_config_dict)
        for (module_class, qconfig_dict, prepare_custom_config_dict, backend_config_dict) in\
                prepare_custom_config_dict.get(STANDALONE_MODULE_CLASS_DICT_KEY, []):
            qconfig_mapping = _get_qconfig_mapping(qconfig_dict, STANDALONE_MODULE_CLASS_DICT_KEY)
            prepare_custom_config = _get_prepare_custom_config(prepare_custom_config_dict, STANDALONE_MODULE_CLASS_DICT_KEY)
            conf.set_standalone_module_class(module_class, qconfig_mapping, prepare_custom_config, backend_config_dict)
        for quant_type_name, custom_module_mapping in prepare_custom_config_dict.get(FLOAT_TO_OBSERVED_DICT_KEY, {}):
            quant_type = quant_type_from_str(quant_type_name)
            for float_class, observed_class in custom_module_mapping.items():
                conf.set_float_to_observed_mapping(float_class, observed_class, quant_type)
        conf.set_non_traceable_module_names(prepare_custom_config_dict.get(NON_TRACEABLE_MODULE_NAME_DICT_KEY, []))
        conf.set_non_traceable_module_classes(prepare_custom_config_dict.get(NON_TRACEABLE_MODULE_CLASS_DICT_KEY, []))
        conf.set_input_quantized_indexes(prepare_custom_config_dict.get(INPUT_QUANTIZED_INDEXES_DICT_KEY, []))
        conf.set_output_quantized_indexes(prepare_custom_config_dict.get(OUTPUT_QUANTIZED_INDEXES_DICT_KEY, []))
        conf.set_preserved_attributes(prepare_custom_config_dict.get(PRESERVED_ATTRIBUTES_DICT_KEY, []))
        return conf


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
        for quant_type_name, custom_module_mapping in convert_custom_config_dict.get(OBSERVED_TO_QUANTIZED_DICT_KEY, {}):
            quant_type = quant_type_from_str(quant_type_name)
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
