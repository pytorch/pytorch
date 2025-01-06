# mypy: allow-untyped-defs
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.quant_type import (
    _get_quant_type_to_str,
    _quant_type_from_str,
    QuantType,
)


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
    example_inputs: tuple[Any, ...]
    prepare_custom_config: Optional[PrepareCustomConfig]
    backend_config: Optional[BackendConfig]


class PrepareCustomConfig:
    """
    Custom configuration for :func:`~torch.ao.quantization.quantize_fx.prepare_fx` and
    :func:`~torch.ao.quantization.quantize_fx.prepare_qat_fx`.

    Example usage::

        prepare_custom_config = PrepareCustomConfig() \
            .set_standalone_module_name("module1", qconfig_mapping, example_inputs, \
                child_prepare_custom_config, backend_config) \
            .set_standalone_module_class(MyStandaloneModule, qconfig_mapping, example_inputs, \
                child_prepare_custom_config, backend_config) \
            .set_float_to_observed_mapping(FloatCustomModule, ObservedCustomModule) \
            .set_non_traceable_module_names(["module2", "module3"]) \
            .set_non_traceable_module_classes([NonTraceableModule1, NonTraceableModule2]) \
            .set_input_quantized_indexes([0]) \
            .set_output_quantized_indexes([0]) \
            .set_preserved_attributes(["attr1", "attr2"])
    """

    def __init__(self) -> None:
        self.standalone_module_names: dict[str, StandaloneModuleConfigEntry] = {}
        self.standalone_module_classes: dict[type, StandaloneModuleConfigEntry] = {}
        self.float_to_observed_mapping: dict[QuantType, dict[type, type]] = {}
        self.non_traceable_module_names: list[str] = []
        self.non_traceable_module_classes: list[type] = []
        self.input_quantized_indexes: list[int] = []
        self.output_quantized_indexes: list[int] = []
        self.preserved_attributes: list[str] = []

    def __repr__(self):
        dict_nonempty = {k: v for k, v in self.__dict__.items() if len(v) > 0}
        return f"PrepareCustomConfig({dict_nonempty})"

    def set_standalone_module_name(
        self,
        module_name: str,
        qconfig_mapping: Optional[QConfigMapping],
        example_inputs: tuple[Any, ...],
        prepare_custom_config: Optional[PrepareCustomConfig],
        backend_config: Optional[BackendConfig],
    ) -> PrepareCustomConfig:
        """
        Set the configuration for running a standalone module identified by ``module_name``.

        If ``qconfig_mapping`` is None, the parent ``qconfig_mapping`` will be used instead.
        If ``prepare_custom_config`` is None, an empty ``PrepareCustomConfig`` will be used.
        If ``backend_config`` is None, the parent ``backend_config`` will be used instead.
        """
        self.standalone_module_names[module_name] = StandaloneModuleConfigEntry(
            qconfig_mapping, example_inputs, prepare_custom_config, backend_config
        )
        return self

    def set_standalone_module_class(
        self,
        module_class: type,
        qconfig_mapping: Optional[QConfigMapping],
        example_inputs: tuple[Any, ...],
        prepare_custom_config: Optional[PrepareCustomConfig],
        backend_config: Optional[BackendConfig],
    ) -> PrepareCustomConfig:
        """
        Set the configuration for running a standalone module identified by ``module_class``.

        If ``qconfig_mapping`` is None, the parent ``qconfig_mapping`` will be used instead.
        If ``prepare_custom_config`` is None, an empty ``PrepareCustomConfig`` will be used.
        If ``backend_config`` is None, the parent ``backend_config`` will be used instead.
        """
        self.standalone_module_classes[module_class] = StandaloneModuleConfigEntry(
            qconfig_mapping, example_inputs, prepare_custom_config, backend_config
        )
        return self

    def set_float_to_observed_mapping(
        self,
        float_class: type,
        observed_class: type,
        quant_type: QuantType = QuantType.STATIC,
    ) -> PrepareCustomConfig:
        """
        Set the mapping from a custom float module class to a custom observed module class.

        The observed module class must have a ``from_float`` class method that converts the float module class
        to the observed module class. This is currently only supported for static quantization.
        """
        if quant_type != QuantType.STATIC:
            raise ValueError(
                "set_float_to_observed_mapping is currently only supported for static quantization"
            )
        if quant_type not in self.float_to_observed_mapping:
            self.float_to_observed_mapping[quant_type] = {}
        self.float_to_observed_mapping[quant_type][float_class] = observed_class
        return self

    def set_non_traceable_module_names(
        self, module_names: list[str]
    ) -> PrepareCustomConfig:
        """
        Set the modules that are not symbolically traceable, identified by name.
        """
        self.non_traceable_module_names = module_names
        return self

    def set_non_traceable_module_classes(
        self, module_classes: list[type]
    ) -> PrepareCustomConfig:
        """
        Set the modules that are not symbolically traceable, identified by class.
        """
        self.non_traceable_module_classes = module_classes
        return self

    def set_input_quantized_indexes(self, indexes: list[int]) -> PrepareCustomConfig:
        """
        Set the indexes of the inputs of the graph that should be quantized.
        Inputs are otherwise assumed to be in fp32 by default instead.
        """
        self.input_quantized_indexes = indexes
        return self

    def set_output_quantized_indexes(self, indexes: list[int]) -> PrepareCustomConfig:
        """
        Set the indexes of the outputs of the graph that should be quantized.
        Outputs are otherwise assumed to be in fp32 by default instead.
        """
        self.output_quantized_indexes = indexes
        return self

    def set_preserved_attributes(self, attributes: list[str]) -> PrepareCustomConfig:
        """
        Set the names of the attributes that will persist in the graph module even if they are not used in
        the model's ``forward`` method.
        """
        self.preserved_attributes = attributes
        return self

    # TODO: remove this
    @classmethod
    def from_dict(
        cls, prepare_custom_config_dict: dict[str, Any]
    ) -> PrepareCustomConfig:
        """
        Create a ``PrepareCustomConfig`` from a dictionary with the following items:

            "standalone_module_name": a list of (module_name, qconfig_mapping, example_inputs,
            child_prepare_custom_config, backend_config) tuples

            "standalone_module_class" a list of (module_class, qconfig_mapping, example_inputs,
            child_prepare_custom_config, backend_config) tuples

            "float_to_observed_custom_module_class": a nested dictionary mapping from quantization
            mode to an inner mapping from float module classes to observed module classes, e.g.
            {"static": {FloatCustomModule: ObservedCustomModule}}

            "non_traceable_module_name": a list of modules names that are not symbolically traceable
            "non_traceable_module_class": a list of module classes that are not symbolically traceable
            "input_quantized_idxs": a list of indexes of graph inputs that should be quantized
            "output_quantized_idxs": a list of indexes of graph outputs that should be quantized
            "preserved_attributes": a list of attributes that persist even if they are not used in ``forward``

        This function is primarily for backward compatibility and may be removed in the future.
        """

        def _get_qconfig_mapping(obj: Any, dict_key: str) -> Optional[QConfigMapping]:
            """
            Convert the given object into a QConfigMapping if possible, else throw an exception.
            """
            if isinstance(obj, QConfigMapping) or obj is None:
                return obj
            if isinstance(obj, dict):
                return QConfigMapping.from_dict(obj)
            raise ValueError(
                f"Expected QConfigMapping in prepare_custom_config_dict[\"{dict_key}\"], got '{type(obj)}'"
            )

        def _get_prepare_custom_config(
            obj: Any, dict_key: str
        ) -> Optional[PrepareCustomConfig]:
            """
            Convert the given object into a PrepareCustomConfig if possible, else throw an exception.
            """
            if isinstance(obj, PrepareCustomConfig) or obj is None:
                return obj
            if isinstance(obj, dict):
                return PrepareCustomConfig.from_dict(obj)
            raise ValueError(
                f"Expected PrepareCustomConfig in prepare_custom_config_dict[\"{dict_key}\"], got '{type(obj)}'"
            )

        def _get_backend_config(obj: Any, dict_key: str) -> Optional[BackendConfig]:
            """
            Convert the given object into a BackendConfig if possible, else throw an exception.
            """
            if isinstance(obj, BackendConfig) or obj is None:
                return obj
            if isinstance(obj, dict):
                return BackendConfig.from_dict(obj)
            raise ValueError(
                f"Expected BackendConfig in prepare_custom_config_dict[\"{dict_key}\"], got '{type(obj)}'"
            )

        conf = cls()
        for (
            module_name,
            qconfig_dict,
            example_inputs,
            _prepare_custom_config_dict,
            backend_config_dict,
        ) in prepare_custom_config_dict.get(STANDALONE_MODULE_NAME_DICT_KEY, []):
            qconfig_mapping = _get_qconfig_mapping(
                qconfig_dict, STANDALONE_MODULE_NAME_DICT_KEY
            )
            prepare_custom_config = _get_prepare_custom_config(
                _prepare_custom_config_dict, STANDALONE_MODULE_NAME_DICT_KEY
            )
            backend_config = _get_backend_config(
                backend_config_dict, STANDALONE_MODULE_NAME_DICT_KEY
            )
            conf.set_standalone_module_name(
                module_name,
                qconfig_mapping,
                example_inputs,
                prepare_custom_config,
                backend_config,
            )
        for (
            module_class,
            qconfig_dict,
            example_inputs,
            _prepare_custom_config_dict,
            backend_config_dict,
        ) in prepare_custom_config_dict.get(STANDALONE_MODULE_CLASS_DICT_KEY, []):
            qconfig_mapping = _get_qconfig_mapping(
                qconfig_dict, STANDALONE_MODULE_CLASS_DICT_KEY
            )
            prepare_custom_config = _get_prepare_custom_config(
                _prepare_custom_config_dict, STANDALONE_MODULE_CLASS_DICT_KEY
            )
            backend_config = _get_backend_config(
                backend_config_dict, STANDALONE_MODULE_CLASS_DICT_KEY
            )
            conf.set_standalone_module_class(
                module_class,
                qconfig_mapping,
                example_inputs,
                prepare_custom_config,
                backend_config,
            )
        for quant_type_name, custom_module_mapping in prepare_custom_config_dict.get(
            FLOAT_TO_OBSERVED_DICT_KEY, {}
        ).items():
            quant_type = _quant_type_from_str(quant_type_name)
            for float_class, observed_class in custom_module_mapping.items():
                conf.set_float_to_observed_mapping(
                    float_class, observed_class, quant_type
                )
        conf.set_non_traceable_module_names(
            prepare_custom_config_dict.get(NON_TRACEABLE_MODULE_NAME_DICT_KEY, [])
        )
        conf.set_non_traceable_module_classes(
            prepare_custom_config_dict.get(NON_TRACEABLE_MODULE_CLASS_DICT_KEY, [])
        )
        conf.set_input_quantized_indexes(
            prepare_custom_config_dict.get(INPUT_QUANTIZED_INDEXES_DICT_KEY, [])
        )
        conf.set_output_quantized_indexes(
            prepare_custom_config_dict.get(OUTPUT_QUANTIZED_INDEXES_DICT_KEY, [])
        )
        conf.set_preserved_attributes(
            prepare_custom_config_dict.get(PRESERVED_ATTRIBUTES_DICT_KEY, [])
        )
        return conf

    def to_dict(self) -> dict[str, Any]:
        """
        Convert this ``PrepareCustomConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.fx.custom_config.PrepareCustomConfig.from_dict`.
        """

        def _make_tuple(key: Any, e: StandaloneModuleConfigEntry):
            qconfig_dict = e.qconfig_mapping.to_dict() if e.qconfig_mapping else None
            prepare_custom_config_dict = (
                e.prepare_custom_config.to_dict() if e.prepare_custom_config else None
            )
            return (
                key,
                qconfig_dict,
                e.example_inputs,
                prepare_custom_config_dict,
                e.backend_config,
            )

        d: dict[str, Any] = {}
        for module_name, sm_config_entry in self.standalone_module_names.items():
            if STANDALONE_MODULE_NAME_DICT_KEY not in d:
                d[STANDALONE_MODULE_NAME_DICT_KEY] = []
            d[STANDALONE_MODULE_NAME_DICT_KEY].append(
                _make_tuple(module_name, sm_config_entry)
            )
        for module_class, sm_config_entry in self.standalone_module_classes.items():
            if STANDALONE_MODULE_CLASS_DICT_KEY not in d:
                d[STANDALONE_MODULE_CLASS_DICT_KEY] = []
            d[STANDALONE_MODULE_CLASS_DICT_KEY].append(
                _make_tuple(module_class, sm_config_entry)
            )
        for (
            quant_type,
            float_to_observed_mapping,
        ) in self.float_to_observed_mapping.items():
            if FLOAT_TO_OBSERVED_DICT_KEY not in d:
                d[FLOAT_TO_OBSERVED_DICT_KEY] = {}
            d[FLOAT_TO_OBSERVED_DICT_KEY][
                _get_quant_type_to_str(quant_type)
            ] = float_to_observed_mapping
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
    Custom configuration for :func:`~torch.ao.quantization.quantize_fx.convert_fx`.

    Example usage::

        convert_custom_config = ConvertCustomConfig() \
            .set_observed_to_quantized_mapping(ObservedCustomModule, QuantizedCustomModule) \
            .set_preserved_attributes(["attr1", "attr2"])
    """

    def __init__(self) -> None:
        self.observed_to_quantized_mapping: dict[QuantType, dict[type, type]] = {}
        self.preserved_attributes: list[str] = []

    def __repr__(self):
        dict_nonempty = {k: v for k, v in self.__dict__.items() if len(v) > 0}
        return f"ConvertCustomConfig({dict_nonempty})"

    def set_observed_to_quantized_mapping(
        self,
        observed_class: type,
        quantized_class: type,
        quant_type: QuantType = QuantType.STATIC,
    ) -> ConvertCustomConfig:
        """
        Set the mapping from a custom observed module class to a custom quantized module class.

        The quantized module class must have a ``from_observed`` class method that converts the observed module class
        to the quantized module class.
        """
        if quant_type not in self.observed_to_quantized_mapping:
            self.observed_to_quantized_mapping[quant_type] = {}
        self.observed_to_quantized_mapping[quant_type][observed_class] = quantized_class
        return self

    def set_preserved_attributes(self, attributes: list[str]) -> ConvertCustomConfig:
        """
        Set the names of the attributes that will persist in the graph module even if they are not used in
        the model's ``forward`` method.
        """
        self.preserved_attributes = attributes
        return self

    # TODO: remove this
    @classmethod
    def from_dict(
        cls, convert_custom_config_dict: dict[str, Any]
    ) -> ConvertCustomConfig:
        """
        Create a ``ConvertCustomConfig`` from a dictionary with the following items:

            "observed_to_quantized_custom_module_class": a nested dictionary mapping from quantization
            mode to an inner mapping from observed module classes to quantized module classes, e.g.::
            {
            "static": {FloatCustomModule: ObservedCustomModule},
            "dynamic": {FloatCustomModule: ObservedCustomModule},
            "weight_only": {FloatCustomModule: ObservedCustomModule}
            }
            "preserved_attributes": a list of attributes that persist even if they are not used in ``forward``

        This function is primarily for backward compatibility and may be removed in the future.
        """
        conf = cls()
        for quant_type_name, custom_module_mapping in convert_custom_config_dict.get(
            OBSERVED_TO_QUANTIZED_DICT_KEY, {}
        ).items():
            quant_type = _quant_type_from_str(quant_type_name)
            for observed_class, quantized_class in custom_module_mapping.items():
                conf.set_observed_to_quantized_mapping(
                    observed_class, quantized_class, quant_type
                )
        conf.set_preserved_attributes(
            convert_custom_config_dict.get(PRESERVED_ATTRIBUTES_DICT_KEY, [])
        )
        return conf

    def to_dict(self) -> dict[str, Any]:
        """
        Convert this ``ConvertCustomConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict`.
        """
        d: dict[str, Any] = {}
        for (
            quant_type,
            observed_to_quantized_mapping,
        ) in self.observed_to_quantized_mapping.items():
            if OBSERVED_TO_QUANTIZED_DICT_KEY not in d:
                d[OBSERVED_TO_QUANTIZED_DICT_KEY] = {}
            d[OBSERVED_TO_QUANTIZED_DICT_KEY][
                _get_quant_type_to_str(quant_type)
            ] = observed_to_quantized_mapping
        if len(self.preserved_attributes) > 0:
            d[PRESERVED_ATTRIBUTES_DICT_KEY] = self.preserved_attributes
        return d


class FuseCustomConfig:
    """
    Custom configuration for :func:`~torch.ao.quantization.quantize_fx.fuse_fx`.

    Example usage::

        fuse_custom_config = FuseCustomConfig().set_preserved_attributes(["attr1", "attr2"])
    """

    def __init__(self) -> None:
        self.preserved_attributes: list[str] = []

    def __repr__(self):
        dict_nonempty = {k: v for k, v in self.__dict__.items() if len(v) > 0}
        return f"FuseCustomConfig({dict_nonempty})"

    def set_preserved_attributes(self, attributes: list[str]) -> FuseCustomConfig:
        """
        Set the names of the attributes that will persist in the graph module even if they are not used in
        the model's ``forward`` method.
        """
        self.preserved_attributes = attributes
        return self

    # TODO: remove this
    @classmethod
    def from_dict(cls, fuse_custom_config_dict: dict[str, Any]) -> FuseCustomConfig:
        """
        Create a ``ConvertCustomConfig`` from a dictionary with the following items:

            "preserved_attributes": a list of attributes that persist even if they are not used in ``forward``

        This function is primarily for backward compatibility and may be removed in the future.
        """
        conf = cls()
        conf.set_preserved_attributes(
            fuse_custom_config_dict.get(PRESERVED_ATTRIBUTES_DICT_KEY, [])
        )
        return conf

    def to_dict(self) -> dict[str, Any]:
        """
        Convert this ``FuseCustomConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict`.
        """
        d: dict[str, Any] = {}
        if len(self.preserved_attributes) > 0:
            d[PRESERVED_ATTRIBUTES_DICT_KEY] = self.preserved_attributes
        return d
