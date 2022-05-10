from __future__ import annotations
from dataclasses import astuple, dataclass
import pprint
from typing import Any, Callable, Dict, List

import torch
from .qconfig import QConfigAny


# TODO: replace all usages with these constants
GLOBAL_DICT_KEY = ""
OBJECT_TYPE_DICT_KEY = "object_type"
MODULE_NAME_REGEX_DICT_KEY = "module_name_regex"
MODULE_NAME_DICT_KEY = "module_name"
MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY = "module_name_object_type_order"


@dataclass
class QConfigObjectTypeEntry:
    object_type: Callable
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


class QuantizationConfigBase:
    """
    TODO: write this
    """

    def __init__(self):
        # In increasing match priority:
        self.global_qconfig: QConfigAny = None
        self.object_type_qconfigs: List[QConfigObjectTypeEntry] = []
        self.module_name_regex_qconfigs: List[QConfigModuleNameRegexEntry] = []
        self.module_name_qconfigs: List[QConfigModuleNameEntry] = []
        self.module_name_object_type_order_qconfigs: List[QConfigModuleNameObjectTypeOrderEntry] = []

    def set_global(self, global_qconfig: QConfigAny) -> QuantizationConfigBase:
        """
        TODO: write this
        """
        self.global_qconfig = global_qconfig
        return self

    def set_object_type(self, object_type: Callable, qconfig: QConfigAny) -> QuantizationConfigBase:
        """
        TODO: write this
        """
        self.object_type_qconfigs.append(QConfigObjectTypeEntry(object_type, qconfig))
        return self

    def set_module_name_regex(self, module_name_regex: str, qconfig: QConfigAny) -> QuantizationConfigBase:
        """
        TODO: write this
        """
        self.module_name_regex_qconfigs.append(QConfigModuleNameRegexEntry(module_name_regex, qconfig))
        return self

    def set_module_name(self, module_name: str, qconfig: QConfigAny) -> QuantizationConfigBase:
        """
        TODO: write this
        """
        self.module_name_qconfigs.append(QConfigModuleNameEntry(module_name, qconfig))
        return self

    def set_module_name_object_type_order(
            self,
            module_name: str,
            object_type: Callable,
            index: int,
            qconfig: QConfigAny,
        ) -> QuantizationConfigBase:
        """
        TODO: write this
        """
        self.module_name_object_type_order_qconfigs.append(
            QConfigModuleNameObjectTypeOrderEntry(module_name, object_type, index, qconfig))
        return self

    def to_dict(self) -> Dict[str, Any]:
        """
        TODO: write this
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
    def from_dict(cls, qconfig_dict: Dict[str, Any]) -> QuantizationConfigBase:
        """
        TODO: write this
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


class PrepareQuantizationConfig(QuantizationConfigBase):
    """
    TODO: write this
    """
    pass

class ConvertQuantizationConfig(QuantizationConfigBase):
    """
    TODO: write this
    """
    pass

