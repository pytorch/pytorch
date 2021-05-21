import torch
from collections import OrderedDict
from typing import Union, Callable, Any, Dict, Tuple
import re

from torch.fx.graph import (
    Graph,
)

from .utils import _parent_name

QConfigAny = Union[torch.quantization.QConfig,
                   torch.quantization.QConfigDynamic, None]

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
def get_qconfig(qconfig_dict, module_type, module_name, global_qconfig):
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
    for node in input_graph.nodes:
        qconfig = None
        if node.op == "get_attr":
            module_name, _ = _parent_name(node.target)
            qconfig = get_qconfig(
                qconfig_dict, type(modules[module_name]), module_name, global_qconfig)
        elif node.op == "call_function":
            # precedence: module_name_qconfig
            # > function_qconfig > global_qconfig
            # module_name takes precedence over function qconfig
            function_qconfig = get_object_type_qconfig(
                qconfig_dict, node.target, global_qconfig)
            module_path, module_type = node_name_to_scope[node.name]
            qconfig = get_qconfig(
                qconfig_dict, module_type, module_path, function_qconfig)
        elif node.op == "call_method":
            module_path, module_type = node_name_to_scope[node.name]
            # use the qconfig of the module that the node belongs to
            qconfig = get_qconfig(
                qconfig_dict, module_type, module_path, global_qconfig)
        elif node.op == 'call_module':
            qconfig = get_qconfig(
                qconfig_dict, type(modules[node.target]), node.target, global_qconfig)
            # regex is not supported eager mode propagate_qconfig_, we'll
            # need to set the qconfig explicitly here in case regex
            # is used
            modules[node.target].qconfig = qconfig
        qconfig_map[node.name] = qconfig
    return qconfig_map
