from collections import OrderedDict
import dataclasses
import re
from typing import Any, Dict, Callable, Union

from .utils import (
    get_combined_dict,
    _parent_name,
)
from .quantization_mappings import (
    get_default_qat_module_mappings,
)
from .qconfig import QConfigAny
from .quantization_config import QuantizationConfigBase


def get_object_type_qconfig(
        quantization_config: QuantizationConfigBase,
        object_type: Union[Callable, str],
        fallback_qconfig: QConfigAny) -> QConfigAny:
    return quantization_config._object_type_qconfig_dict.get(object_type, fallback_qconfig)


def get_module_name_regex_qconfig(quantization_config, module_name, fallback_qconfig):
    for regex_pattern, qconfig in quantization_config._module_name_regex_qconfig_dict.items():
        if re.match(regex_pattern, module_name):
            # first match wins
            return qconfig
    return fallback_qconfig


def get_module_name_qconfig(quantization_config, module_name, fallback_qconfig):
    if module_name == '':
        # module name qconfig not found
        return fallback_qconfig
    if module_name in quantization_config._module_name_qconfig_dict:
        return quantization_config._module_name_qconfig_dict[module_name]
    else:
        parent, _ = _parent_name(module_name)
        return get_module_name_qconfig(quantization_config, parent, fallback_qconfig)


def maybe_adjust_qconfig_for_module_type_or_name(quantization_config, module_type, module_name, global_qconfig):
    # get qconfig for module_name,
    # fallback to module_name_regex_qconfig, module_type_qconfig,
    # global_qconfig if necessary
    module_type_qconfig = get_object_type_qconfig(
        quantization_config, module_type, global_qconfig)
    module_name_regex_qconfig = get_module_name_regex_qconfig(
        quantization_config, module_name, module_type_qconfig)
    module_name_qconfig = get_module_name_qconfig(
        quantization_config, module_name, module_name_regex_qconfig)
    return module_name_qconfig


def get_flattened_qconfig_dict(quantization_config: QuantizationConfigBase):
    """ flatten the global, object_type and module_name qconfig
    to the same qconfig_dict so that it can be used by
    propagate_qconfig_ function.
    "module_name_regex" is ignored for now since it's not supported
    in propagate_qconfig_, but it can be fixed later.

    For example:
    Input: {
      "": qconfig,
      "object_type": [
        (torch.add, qconfig)
      ],
      "module_name": [
        ("conv", qconfig)
      ]
    }

    Output: {
      "": qconfig,
      torch.add: qconfig,
      "conv": qconfig
    }
    """
    flattened: Dict[Union[Callable, str], QConfigAny] = {"": quantization_config.global_qconfig}
    for obj, qconfig in quantization_config._object_type_qconfig_dict.items():
        flattened[obj] = qconfig
    for obj, qconfig in quantization_config._module_name_qconfig_dict.items():
        flattened[obj] = qconfig
    return flattened


def convert_lists_to_ordered_dicts(quantization_config: QuantizationConfigBase):
    """
    Convert lists in a QuantizationConfigBase to OrderedDicts.
    """
    def to_tuple_list(dataclass_list):
        return [dataclasses.astuple(d) for d in dataclass_list]
    quantization_config._object_type_qconfig_dict = \
        OrderedDict(to_tuple_list(quantization_config.object_type_qconfigs))
    quantization_config._module_name_qconfig_dict = \
        OrderedDict(to_tuple_list(quantization_config.module_name_qconfigs))
    quantization_config._module_name_regex_qconfig_dict = \
        OrderedDict(to_tuple_list(quantization_config.module_name_regex_qconfigs))


def update_qconfig_for_qat(
    quantization_config: QuantizationConfigBase,
    additional_qat_module_mapping: Dict[Callable, Callable]
) -> Any:
    """
    Update the qconfig_dict to account for module swaps during QAT.
    During QAT we perform a module swap on the nn.Module types to the corresponding nn.qat.modules types.
    """
    all_qat_mappings = get_combined_dict(
        get_default_qat_module_mappings(), additional_qat_module_mapping)
    object_type_dict = quantization_config._object_type_qconfig_dict
    new_object_type_dict = object_type_dict.copy()
    for k, v in new_object_type_dict.items():
        if k in all_qat_mappings:
            object_type_dict[all_qat_mappings[k]] = v
