import torch
from collections import OrderedDict, defaultdict
from typing import Union, Callable, Any, Dict, Tuple, Set
from torch.quantization.qconfig import add_module_to_qconfig_obs_ctr, QConfigAny

import re

from torch.fx.graph import (
    Graph,
)

from .utils import _parent_name


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


def get_object_type_qconfig(
        qconfig_dict: Any,
        object_type: Union[Callable, str],
        fallback_qconfig: QConfigAny) -> QConfigAny:
    # object_type can be
    # 1. module type (call_module)
    # 2. function (call_function)
    # 3. string (call_method)
    return qconfig_dict['object_type'].get(
        object_type, fallback_qconfig)


def get_module_name_regex_qconfig(qconfig_dict, module_name, fallback_qconfig):
    for regex_pattern, qconfig in \
            qconfig_dict['module_name_regex'].items():
        if re.match(regex_pattern, module_name):
            # first match wins
            return qconfig
    return fallback_qconfig


def maybe_adjust_qconfig_for_module_name_object_type_order(
    qconfig_dict: Any,
    cur_module_path: str,
    cur_object_type: Callable,
    cur_object_type_idx: int,
    fallback_qconfig: QConfigAny,
) -> QConfigAny:
    qconfig_module_name_object_type_order = \
        qconfig_dict.get('module_name_object_type_order', {})
    for module_path, object_type, object_type_idx, qconfig in \
            qconfig_module_name_object_type_order:
        if (
            (module_path == cur_module_path) and
            (object_type == cur_object_type) and
            (object_type_idx == cur_object_type_idx)
        ):
            return qconfig

    return fallback_qconfig


def get_module_name_qconfig(qconfig_dict, module_name, fallback_qconfig):
    if module_name == '':
        # module name qconfig not found
        return fallback_qconfig
    if module_name in qconfig_dict['module_name']:
        return qconfig_dict['module_name'][module_name]
    else:
        parent, _ = _parent_name(module_name)
        return get_module_name_qconfig(qconfig_dict, parent, fallback_qconfig)

# get qconfig for module_name,
# fallback to module_name_regex_qconfig, module_type_qconfig,
# global_qconfig if necessary


def maybe_adjust_qconfig_for_module_type_or_name(qconfig_dict, module_type, module_name, global_qconfig):
    module_type_qconfig = get_object_type_qconfig(
        qconfig_dict, module_type, global_qconfig)
    module_name_regex_qconfig = get_module_name_regex_qconfig(
        qconfig_dict, module_name, module_type_qconfig)
    module_name_qconfig = get_module_name_qconfig(
        qconfig_dict, module_name, module_name_regex_qconfig)
    return module_name_qconfig


def generate_qconfig_map(
        root: torch.nn.Module,
        modules: Dict[str, torch.nn.Module],
        input_graph: Graph,
        qconfig_dict: Any,
        node_name_to_scope: Dict[str, Tuple[str, type]]) -> Dict[str, QConfigAny]:
    global_qconfig = qconfig_dict.get("", None)
    qconfig_map = dict()

    # example:
    #
    #   {'foo.bar': {F.linear: 0, F.conv2d: 1, ...}, ...}
    #
    # meaning in submodule 'foo.bar', we have seen 0 F.linear and
    # 1 F.conv2d invocations so far.
    submodule_to_object_type_to_cur_idx: Dict[str, Dict[Callable, int]] = \
        defaultdict(lambda: defaultdict(int))

    for node in input_graph.nodes:
        qconfig = None
        if node.op == "get_attr":
            module_name, _ = _parent_name(node.target)
            qconfig = maybe_adjust_qconfig_for_module_type_or_name(
                qconfig_dict, type(modules[module_name]), module_name, global_qconfig)
            qconfig_with_device_check = add_module_to_qconfig_obs_ctr(qconfig, modules.get(node.target, None))
        elif node.op == "call_function":
            # precedence: module_name_qconfig
            # > function_qconfig > global_qconfig
            # module_name takes precedence over function qconfig
            function_qconfig = get_object_type_qconfig(
                qconfig_dict, node.target, global_qconfig)
            module_path, module_type = node_name_to_scope[node.name]
            qconfig = maybe_adjust_qconfig_for_module_type_or_name(
                qconfig_dict, module_type, module_path, function_qconfig)

            cur_object_type_idx = \
                submodule_to_object_type_to_cur_idx[module_path][node.target]
            submodule_to_object_type_to_cur_idx[module_path][node.target] += 1
            qconfig = maybe_adjust_qconfig_for_module_name_object_type_order(
                qconfig_dict, module_path, node.target, cur_object_type_idx,
                qconfig)
            qconfig_with_device_check = add_module_to_qconfig_obs_ctr(qconfig, modules.get(node.target, None))

        elif node.op == "call_method":
            module_path, module_type = node_name_to_scope[node.name]
            # use the qconfig of the module that the node belongs to
            qconfig = maybe_adjust_qconfig_for_module_type_or_name(
                qconfig_dict, module_type, module_path, global_qconfig)
            # Currently call_method does not support modifying qconfig
            # by order, we can add this later if it is needed.
            qconfig_with_device_check = add_module_to_qconfig_obs_ctr(qconfig, modules.get(node.target, None))

        elif node.op == 'call_module':
            qconfig = maybe_adjust_qconfig_for_module_type_or_name(
                qconfig_dict, type(modules[node.target]), node.target, global_qconfig)

            module_path, module_type = node_name_to_scope[node.name]
            # Note: for call_module, the module_path is the current module's name.
            # to meaningfully count invocations, we need to count them in the parent
            # module.
            parent_name, _ = _parent_name(module_path)
            cur_object_type_idx = \
                submodule_to_object_type_to_cur_idx[parent_name][module_type]
            submodule_to_object_type_to_cur_idx[parent_name][module_type] += 1
            qconfig = maybe_adjust_qconfig_for_module_name_object_type_order(
                qconfig_dict, parent_name, module_type, cur_object_type_idx,
                qconfig)
            qconfig_with_device_check = add_module_to_qconfig_obs_ctr(qconfig, modules.get(node.target, None))

            # regex is not supported eager mode propagate_qconfig_, we'll
            # need to set the qconfig explicitly here in case regex
            # is used
            modules[node.target].qconfig = qconfig_with_device_check
        else:
            qconfig_with_device_check = None

        qconfig_map[node.name] = qconfig_with_device_check
    return qconfig_map


def check_is_valid_config_dict(config_dict: Any, allowed_keys: Set[str], dict_name: str) -> None:
    r""" Checks if the given config_dict has the correct keys

    Args:
      `config_dict`: dictionary whose keys we want to check
    """

    for k in config_dict.keys():
        if k not in allowed_keys:
            raise ValueError(
                'Expected ' + dict_name + ' to have the following keys: ' +
                str(allowed_keys) + '. But found \'' + k +
                '\' instead.')


def check_is_valid_qconfig_dict(qconfig_dict: Any) -> None:
    r""" Checks if the given qconfig_dict has the correct keys

    Args:
      `qconfig_dict`: dictionary whose keys we want to check
    """

    qconfig_dict_allowed_keys = {
        "", "object_type", "module_name_regex", "module_name",
        "module_name_object_type_order"}
    check_is_valid_config_dict(qconfig_dict, qconfig_dict_allowed_keys, "qconfig_dict")


def check_is_valid_prepare_custom_config_dict(prepare_custom_config_dict: Dict[str, Any] = None) -> None:
    r""" Checks if the given prepare_custom_config_dict has the correct keys

    Args:
      `prepare_custom_config_dict`: customization configuration dictionary for
      quantization tool
    """
    if not prepare_custom_config_dict:
        return

    prepare_custom_config_dict_allowed_keys = {"standalone_module_name",
                                               "standalone_module_class",
                                               "float_to_observed_custom_module_class",
                                               "non_traceable_module_name",
                                               "non_traceable_module_class",
                                               "additional_fuser_method_mapping",
                                               "additional_qat__module_mapping",
                                               "additional_fusion_pattern",
                                               "additional_quant_pattern",
                                               "input_quantized_idxs",
                                               "output_quantized_idxs",
                                               "preserved_attributes"}
    check_is_valid_config_dict(prepare_custom_config_dict,
                               prepare_custom_config_dict_allowed_keys, "prepare_custom_config_dict")


def check_is_valid_convert_custom_config_dict(convert_custom_config_dict: Dict[str, Any] = None) -> None:
    r""" Checks if the given convert_custom_config_dict has the correct keys

    Args:
      `convert_custom_config_dict`: dictionary for custom configurations for
      convert function
    """
    if not convert_custom_config_dict:
        return

    convert_custom_config_dict_allowed_keys = {"additional_object_mapping",
                                               "observed_to_quantized_custom_module_class",
                                               "preserved_attributes"}
    check_is_valid_config_dict(convert_custom_config_dict,
                               convert_custom_config_dict_allowed_keys, "convert_custom_config_dict")


def check_is_valid_fuse_custom_config_dict(fuse_custom_config_dict: Dict[str, Any] = None) -> None:
    r""" Checks if the given fuse_custom_config_dict has the correct keys

    Args:
      `fuse_custom_config_dict`: dictionary for custom configurations for fuse_fx
    """
    if not fuse_custom_config_dict:
        return

    fuse_custom_config_dict_allowed_keys = {"additional_fuser_method_mapping",
                                            "preserved_attributes"}
    check_is_valid_config_dict(fuse_custom_config_dict, fuse_custom_config_dict_allowed_keys, "fuse_custom_config_dict")
