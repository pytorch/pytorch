import re
from typing import Dict, Callable, Union

from .utils import (
    _get_combined_dict,
    _parent_name,
)
from .quantization_mappings import (
    get_default_qat_module_mappings,
)
from .qconfig import QConfigAny
from .qconfig_mapping import QConfigMapping


__all__ = [
]


def _get_object_type_qconfig(
        qconfig_mapping: QConfigMapping,
        object_type: Union[Callable, str],
        fallback_qconfig: QConfigAny) -> QConfigAny:
    return qconfig_mapping.object_type_qconfigs.get(object_type, fallback_qconfig)


def _get_module_name_regex_qconfig(qconfig_mapping, module_name, fallback_qconfig):
    for regex_pattern, qconfig in qconfig_mapping.module_name_regex_qconfigs.items():
        if re.match(regex_pattern, module_name):
            # first match wins
            return qconfig
    return fallback_qconfig


def _get_module_name_qconfig(qconfig_mapping, module_name, fallback_qconfig):
    if module_name == '':
        # module name qconfig not found
        return fallback_qconfig
    if module_name in qconfig_mapping.module_name_qconfigs:
        return qconfig_mapping.module_name_qconfigs[module_name]
    else:
        parent, _ = _parent_name(module_name)
        return _get_module_name_qconfig(qconfig_mapping, parent, fallback_qconfig)


def _maybe_adjust_qconfig_for_module_type_or_name(qconfig_mapping, module_type, module_name, global_qconfig):
    # get qconfig for module_name,
    # fallback to module_name_regex_qconfig, module_type_qconfig,
    # global_qconfig if necessary
    module_type_qconfig = _get_object_type_qconfig(
        qconfig_mapping, module_type, global_qconfig)
    module_name_regex_qconfig = _get_module_name_regex_qconfig(
        qconfig_mapping, module_name, module_type_qconfig)
    module_name_qconfig = _get_module_name_qconfig(
        qconfig_mapping, module_name, module_name_regex_qconfig)
    return module_name_qconfig


def _get_flattened_qconfig_dict(qconfig_mapping: QConfigMapping) -> Dict[Union[Callable, str], QConfigAny]:
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
    flattened: Dict[Union[Callable, str], QConfigAny] = {"": qconfig_mapping.global_qconfig}
    for obj, qconfig in qconfig_mapping.object_type_qconfigs.items():
        flattened[obj] = qconfig
    for obj, qconfig in qconfig_mapping.module_name_qconfigs.items():
        flattened[obj] = qconfig
    return flattened


def _update_qconfig_for_qat(
        qconfig_mapping: QConfigMapping,
        additional_qat_module_mapping: Dict[Callable, Callable]):
    """
    Update the qconfig_dict to account for module swaps during QAT.
    During QAT we perform a module swap on the nn.Module types to the corresponding nn.qat.modules types.
    """
    all_qat_mappings = _get_combined_dict(
        get_default_qat_module_mappings(), additional_qat_module_mapping)
    object_type_dict = qconfig_mapping.object_type_qconfigs
    new_object_type_dict = object_type_dict.copy()
    for k, v in new_object_type_dict.items():
        if k in all_qat_mappings:
            object_type_dict[all_qat_mappings[k]] = v
