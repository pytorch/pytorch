from __future__ import annotations
from collections import OrderedDict
from dataclasses import astuple, dataclass
from typing import Any, Callable, Dict, List, Type, TypeVar, Union

from .qconfig import QConfigAny


__all__ = [
    "ConvertQuantizationConfig",
    "EqualizationConfig",
    "PrepareQuantizationConfig",
    "QConfigModuleNameEntry",
    "QConfigModuleNameRegexEntry",
    "QConfigModuleNameObjectTypeOrderEntry",
    "QConfigObjectTypeEntry",
    "QuantizationConfigBase",
]


# TODO: replace all usages with these constants
GLOBAL_DICT_KEY = ""
OBJECT_TYPE_DICT_KEY = "object_type"
MODULE_NAME_REGEX_DICT_KEY = "module_name_regex"
MODULE_NAME_DICT_KEY = "module_name"
MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY = "module_name_object_type_order"


@dataclass
class QConfigObjectTypeEntry:
    # object_type can be
    # 1. module type (call_module)
    # 2. function (call_function)
    # 3. string (call_method)
    object_type: Union[Callable, str]
    qconfig: QConfigAny


@dataclass
class QConfigModuleNameRegexEntry:
    module_name_regex: str
    qconfig: QConfigAny


@dataclass
class QConfigModuleNameEntry:
    module_name: str
    qconfig: QConfigAny


@dataclass
class QConfigModuleNameObjectTypeOrderEntry:
    module_name: str
    object_type: Callable
    index: int
    qconfig: QConfigAny


_T = TypeVar("_T", bound="QuantizationConfigBase")

class QuantizationConfigBase:
    """
    Config for specifying how to quantize a given model.

    The user can specify QConfigs using the following methods (in increasing match priority):

        `set_global`: sets the global (default) qconfig
        `set_object_type`: sets the qconfig for a given module type, function, or method name
        `set_module_name_regex`: sets the qconfig for modules matching the given regex string
        `set_module_name`: sets the qconfig for modules matching the given module name
        `set_module_name_object_type_order`: sets the qconfig for modules matching a combination
            of the given module name, object type, and the index at which the module appears

    Example usage::

        quantization_config = PrepareQuantizationConfig()
            .set_global(global_qconfig)
            .set_object_type(torch.nn.Linear, qconfig1)
            .set_object_type(torch.nn.ReLU, qconfig1)
            .set_module_name_regex("foo.*bar.*conv[0-9]+", qconfig1)
            .set_module_name_regex("foo.*", qconfig2)
            .set_module_name("module1", qconfig1)
            .set_module_name("module2", qconfig2)
            .set_module_name_object_type_order("foo.bar", torch.nn.functional.linear, 0, qconfig3)
    """

    def __init__(self):
        # In increasing match priority:
        self.global_qconfig: QConfigAny = None
        self.object_type_qconfigs: List[QConfigObjectTypeEntry] = []
        self.module_name_regex_qconfigs: List[QConfigModuleNameRegexEntry] = []
        self.module_name_qconfigs: List[QConfigModuleNameEntry] = []
        self.module_name_object_type_order_qconfigs: List[QConfigModuleNameObjectTypeOrderEntry] = []

        # For mypy warnings
        self._object_type_qconfig_dict: OrderedDict[Union[Callable, str], QConfigAny] = OrderedDict()
        self._module_name_regex_qconfig_dict: OrderedDict[str, QConfigAny] = OrderedDict()
        self._module_name_qconfig_dict: OrderedDict[str, QConfigAny] = OrderedDict()

    def set_global(self: _T, global_qconfig: QConfigAny) -> _T:
        """
        Set the global (default) qconfig.
        """
        self.global_qconfig = global_qconfig
        return self

    def set_object_type(self: _T, object_type: Union[Callable, str], qconfig: QConfigAny) -> _T:
        """
        Set the qconfig for a given module type, function, or method name.
        If the qconfig for an existing object type was already set, the new qconfig will override the old one.
        """
        self.object_type_qconfigs.append(QConfigObjectTypeEntry(object_type, qconfig))
        return self

    def set_module_name_regex(self: _T, module_name_regex: str, qconfig: QConfigAny) -> _T:
        """
        Set the qconfig for modules matching the given regex string.

        Regexes will be matched in the order in which they are registered through this method.
        Thus, the caller should register more specific patterns first, e.g.::

            quantization_config = PrepareQuantizationConfig()
                .set_module_name_regex("foo.*bar.*conv[0-9]+", qconfig1)
                .set_module_name_regex("foo.*bar.*", qconfig2)
                .set_module_name_regex("foo.*", qconfig3)

        In this example, "foo.bar.conv0" would match qconfig1, "foo.bar.linear" would match qconfig2,
        and "foo.baz.relu" would match qconfig3.

        If the qconfig for an existing module name regex was already set, the new qconfig will override the
        old one while preserving the order in which the regexes were originally registered.
        """
        self.module_name_regex_qconfigs.append(QConfigModuleNameRegexEntry(module_name_regex, qconfig))
        return self

    def set_module_name(self: _T, module_name: str, qconfig: QConfigAny) -> _T:
        """
        Set the qconfig for modules matching the given module name.
        If the qconfig for an existing module name was already set, the new qconfig will override the old one.
        """
        self.module_name_qconfigs.append(QConfigModuleNameEntry(module_name, qconfig))
        return self

    def set_module_name_object_type_order(
            self: _T,
            module_name: str,
            object_type: Callable,
            index: int,
            qconfig: QConfigAny) -> _T:
        """
        Set the qconfig for modules matching a combination of the given module name, object type,
        and the index at which the module appears.

        If the qconfig for an existing (module name, object type, index) combination was already set,
        the new qconfig will override the old one.
        """
        self.module_name_object_type_order_qconfigs.append(
            QConfigModuleNameObjectTypeOrderEntry(module_name, object_type, index, qconfig))
        return self

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this `QuantizationConfigBase` to a qconfig dictionary with the following keys:

            "" (for global qconfig)
            "object_type"
            "module_name_regex"
            "module_name"
            "module_name_object_type_order"

        The values of this dictionary are lists of tuples.
        """

        def to_tuple_list(l):
            return [astuple(e) for e in l]
        return {
            GLOBAL_DICT_KEY: self.global_qconfig,
            OBJECT_TYPE_DICT_KEY: to_tuple_list(self.object_type_qconfigs),
            MODULE_NAME_REGEX_DICT_KEY: to_tuple_list(self.module_name_regex_qconfigs),
            MODULE_NAME_DICT_KEY: to_tuple_list(self.module_name_qconfigs),
            MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY: to_tuple_list(self.module_name_object_type_order_qconfigs),
        }

    @classmethod
    def from_dict(cls: Type[_T], qconfig_dict: Dict[str, Any]) -> _T:
        """
        Create a `QuantizationConfigBase` from a qconfig dictionary with the following keys (all optional):

            "" (for global qconfig)
            "object_type"
            "module_name_regex"
            "module_name"
            "module_name_object_type_order"

        The values of this dictionary are expected to be lists of tuples.
        """
        conf = cls()
        if GLOBAL_DICT_KEY in qconfig_dict:
            conf.set_global(qconfig_dict[GLOBAL_DICT_KEY])
        for object_type, qconfig in qconfig_dict.get(OBJECT_TYPE_DICT_KEY, []):
            conf.set_object_type(object_type, qconfig)
        for module_name_regex, qconfig in qconfig_dict.get(MODULE_NAME_REGEX_DICT_KEY, []):
            conf.set_module_name_regex(module_name_regex, qconfig)
        for module_name, qconfig in qconfig_dict.get(MODULE_NAME_DICT_KEY, []):
            conf.set_module_name(module_name, qconfig)
        for module_name, object_type, index, qconfig in qconfig_dict.get(MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY, []):
            conf.set_module_name_object_type_order(module_name, object_type, index, qconfig)
        return conf


class EqualizationConfig(QuantizationConfigBase):
    """
    Config for specifying how to perform equalization on a model.
    For full details, please refer to the documentation for `QuantizationConfigBase`.
    """
    pass


class PrepareQuantizationConfig(QuantizationConfigBase):
    """
    Config for specifying how to prepare a model for quantization.
    For full details, please refer to the documentation for `QuantizationConfigBase`.
    """
    pass


class ConvertQuantizationConfig(QuantizationConfigBase):
    """
    Config for specifying how to convert a model for quantization.
    For full details, please refer to the documentation for `QuantizationConfigBase`.
    """
    pass
