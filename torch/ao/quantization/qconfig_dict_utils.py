from collections import OrderedDict
from typing import Any, Dict, Callable

from .utils import (
    get_combined_dict,
)
from .quantization_mappings import (
    get_default_qat_module_mappings,
)

def get_flattened_qconfig_dict(qconfig_dict):
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
    flattened = dict()
    if '' in qconfig_dict:
        flattened[''] = qconfig_dict['']

    def flatten_key(key):
        if key in qconfig_dict:
            for (obj, qconfig) in qconfig_dict[key].items():
                flattened[obj] = qconfig

    flatten_key('object_type')
    flatten_key('module_name')
    return flattened


def convert_dict_to_ordered_dict(qconfig_dict: Any) -> Dict[str, Dict[Any, Any]]:
    """ Convert dict in qconfig_dict to ordered dict
    """
    # convert a qconfig list for a type to OrderedDict
    def _convert_to_ordered_dict(key, qconfig_dict):
        qconfig_dict[key] = OrderedDict(qconfig_dict.get(key, []))

    _convert_to_ordered_dict('object_type', qconfig_dict)
    _convert_to_ordered_dict('module_name_regex', qconfig_dict)
    _convert_to_ordered_dict('module_name', qconfig_dict)
    return qconfig_dict


def update_qconfig_for_qat(
    qconfig_dict: Any,
    additional_qat_module_mapping: Dict[Callable, Callable]
) -> Any:
    """
    Update the qconfig_dict to account for module swaps during QAT.
    During QAT we perform a module swap on the nn.Module types to the corresponding nn.qat.modules types.
    """
    all_qat_mappings = get_combined_dict(
        get_default_qat_module_mappings(), additional_qat_module_mapping)
    object_type_dict = qconfig_dict.get("object_type", None)
    new_object_type_dict = object_type_dict.copy()
    for k, v in new_object_type_dict.items():
        if k in all_qat_mappings:
            object_type_dict[all_qat_mappings[k]] = v
    return qconfig_dict
