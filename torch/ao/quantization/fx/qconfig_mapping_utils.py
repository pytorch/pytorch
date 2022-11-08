import torch
from collections import defaultdict, OrderedDict
from typing import Callable, Any, Dict, Tuple, Set, List
from torch.ao.quantization import QConfig
from torch.ao.quantization.qconfig import _add_module_to_qconfig_obs_ctr, QConfigAny, qconfig_equals
from torch.ao.quantization.quantize import (
    is_activation_post_process,
)
from torch.ao.quantization.backend_config import (
    DTypeConfig,
)

from torch.fx import (
    GraphModule,
)
from torch.fx.graph import (
    Graph,
)
from torch.nn.intrinsic import _FusedModule

from ..utils import (
    _parent_name,
    get_qconfig_dtypes,
)
from ..qconfig_mapping import (
    OBJECT_TYPE_DICT_KEY,
    MODULE_NAME_DICT_KEY,
    MODULE_NAME_REGEX_DICT_KEY,
    QConfigMapping,
)
from ..qconfig_mapping_utils import (
    get_object_type_qconfig,
    maybe_adjust_qconfig_for_module_type_or_name,
)


# TODO: revisit this list. Many helper methods shouldn't be public
__all__ = [
    "check_is_valid_config_dict",
    "compare_prepare_convert_qconfig_mappings",
    "generate_node_name_to_qconfig",
    "is_qconfig_supported_by_dtype_configs",
    "maybe_adjust_qconfig_for_module_name_object_type_order",
    "update_qconfig_for_fusion",
]



def maybe_adjust_qconfig_for_module_name_object_type_order(
    qconfig_mapping: QConfigMapping,
    cur_module_path: str,
    cur_object_type: Callable,
    cur_object_type_idx: int,
    fallback_qconfig: QConfigAny,
) -> QConfigAny:
    for (module_name, object_type, index), qconfig in qconfig_mapping.module_name_object_type_order_qconfigs.items():
        if (
            (module_name == cur_module_path) and
            (object_type == cur_object_type) and
            (index == cur_object_type_idx)
        ):
            return qconfig
    return fallback_qconfig


def update_qconfig_for_fusion(model: GraphModule, qconfig_mapping: QConfigMapping):
    """
    Update the QConfigMapping to account for fused modules such as LinearReLU.
    This assumes the QConfigMapping's attributes have already been converted to OrderedDicts.
    """
    object_type_dict = qconfig_mapping.object_type_qconfigs
    if len(object_type_dict) == 0:
        return qconfig_mapping

    modules = dict(model.named_modules())

    for node in model.graph.nodes:
        if node.op == 'call_module' and node.target in modules:
            maybe_fused_module = modules[str(node.target)]
            if not isinstance(maybe_fused_module, _FusedModule):
                continue

            ops = list(maybe_fused_module._modules.values())
            fused_qconfig = object_type_dict.get(type(ops[0]), None)

            # Raise an error if the modules in the fused module have
            # different qconfigs specified in the qconfig_dict
            # TODO: currently it only works for modules,
            # need to make this work for torch.nn.functional.relu
            # TODO: currently it only works for object_type configurations,
            # ideally it should work for different types of configurations,
            # maybe we want to redesign this part
            for op in ops[1:]:
                if not qconfig_equals(object_type_dict.get(type(op), None), fused_qconfig):
                    raise LookupError(
                        "During fusion, we need to specify the same " +
                        f"qconfigs for all module types in {type(maybe_fused_module)} " +
                        f"offending type: {type(op)}")

            if fused_qconfig is not None:
                object_type_dict[type(maybe_fused_module)] = fused_qconfig

def generate_node_name_to_qconfig(
        root: torch.nn.Module,
        modules: Dict[str, torch.nn.Module],
        input_graph: Graph,
        qconfig_mapping: QConfigMapping,
        node_name_to_scope: Dict[str, Tuple[str, type]]) -> Dict[str, QConfigAny]:
    global_qconfig = qconfig_mapping.global_qconfig
    node_name_to_qconfig = {}

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
                qconfig_mapping, type(modules[module_name]), module_name, global_qconfig)
            qconfig_with_device_check = _add_module_to_qconfig_obs_ctr(qconfig, modules.get(node.target, None))
        elif node.op == "call_function":
            # precedence: module_name_qconfig
            # > function_qconfig > global_qconfig
            # module_name takes precedence over function qconfig
            function_qconfig = get_object_type_qconfig(
                qconfig_mapping, node.target, global_qconfig)
            module_path, module_type = node_name_to_scope[node.name]
            qconfig = maybe_adjust_qconfig_for_module_type_or_name(
                qconfig_mapping, module_type, module_path, function_qconfig)

            cur_object_type_idx = \
                submodule_to_object_type_to_cur_idx[module_path][node.target]
            submodule_to_object_type_to_cur_idx[module_path][node.target] += 1
            qconfig = maybe_adjust_qconfig_for_module_name_object_type_order(
                qconfig_mapping, module_path, node.target, cur_object_type_idx, qconfig)
            qconfig_with_device_check = _add_module_to_qconfig_obs_ctr(qconfig, modules.get(node.target, None))

        elif node.op == "call_method":
            module_path, module_type = node_name_to_scope[node.name]
            # first use node.target (string) to get the qconfig
            # this is to support configs like
            # "object_type": [("reshpe", qconfig)]
            qconfig = maybe_adjust_qconfig_for_module_type_or_name(
                qconfig_mapping, node.target, module_path, global_qconfig)
            # if there is no special config for the method, we'll fall back to the
            # config for the module that contains the call_method node
            qconfig = maybe_adjust_qconfig_for_module_type_or_name(
                qconfig_mapping, module_type, module_path, qconfig)
            # currently call_method does not support modifying qconfig
            # by order, we can add this later if it is needed.
            qconfig_with_device_check = _add_module_to_qconfig_obs_ctr(qconfig, modules.get(node.target, None))

        elif node.op == 'call_module':
            # if the node is an observer, just continue - don't add it to the qconfig_map
            if is_activation_post_process(modules[node.target]):
                continue
            qconfig = maybe_adjust_qconfig_for_module_type_or_name(
                qconfig_mapping, type(modules[node.target]), node.target, global_qconfig)

            module_path, module_type = node_name_to_scope[node.name]
            # Note: for call_module, the module_path is the current module's name.
            # to meaningfully count invocations, we need to count them in the parent
            # module.
            parent_name, _ = _parent_name(module_path)
            cur_object_type_idx = \
                submodule_to_object_type_to_cur_idx[parent_name][module_type]
            submodule_to_object_type_to_cur_idx[parent_name][module_type] += 1
            qconfig = maybe_adjust_qconfig_for_module_name_object_type_order(
                qconfig_mapping, parent_name, module_type, cur_object_type_idx,
                qconfig)
            qconfig_with_device_check = _add_module_to_qconfig_obs_ctr(qconfig, modules.get(node.target, None))

            # regex is not supported eager mode propagate_qconfig_, we'll
            # need to set the qconfig explicitly here in case regex
            # is used
            modules[node.target].qconfig = qconfig_with_device_check
        else:
            qconfig_with_device_check = None

        node_name_to_qconfig[node.name] = qconfig_with_device_check
    return node_name_to_qconfig


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


def compare_prepare_convert_qconfig_mappings(
        prepare_qconfig_mapping: QConfigMapping,
        convert_qconfig_mapping: QConfigMapping):
    r""" Compare the qconfig_mapping passed in convert to the one from prepare and check the values

    Args:
      `prepare_qconfig_mapping`: configuration for prepare quantization step
      `convert_qconfig_mapping`: configuration for convert quantization step
    """
    assert qconfig_equals(prepare_qconfig_mapping.global_qconfig, convert_qconfig_mapping.global_qconfig), \
        "Expected global qconfigs to be the same in the prepare and convert quantization configs"
    prepare_dicts: List[OrderedDict] = [
        prepare_qconfig_mapping.object_type_qconfigs,
        prepare_qconfig_mapping.module_name_qconfigs,
        prepare_qconfig_mapping.module_name_regex_qconfigs,
    ]
    convert_dicts: List[OrderedDict] = [
        convert_qconfig_mapping.object_type_qconfigs,
        convert_qconfig_mapping.module_name_qconfigs,
        convert_qconfig_mapping.module_name_regex_qconfigs,
    ]
    dict_names = [OBJECT_TYPE_DICT_KEY, MODULE_NAME_DICT_KEY, MODULE_NAME_REGEX_DICT_KEY]
    for i in range(len(prepare_dicts)):
        for name, qconfig in prepare_dicts[i].items():
            assert name in convert_dicts[i], "Missing key {} {} in convert QConfigMapping \
                when it was present in prepare".format(dict_names[i], name)
            assert convert_dicts[i][name] is None \
                or qconfig_equals(prepare_dicts[i][name], convert_dicts[i][name]), \
                "Expected convert QConfigMapping to have the same qconfig as prepare for key {} {}; \
                prepare: {}; convert: {}".format(dict_names[i], name, prepare_dicts[i][name], convert_dicts[i][name])

def is_qconfig_supported_by_dtype_configs(qconfig: QConfig, dtype_configs: List[DTypeConfig]):
    for dtype_config in dtype_configs:
        is_dynamic = dtype_config.is_dynamic
        if is_dynamic is None:
            is_dynamic = False
        input_dtype = dtype_config.input_dtype or torch.float
        weight_dtype = dtype_config.weight_dtype or torch.float
        bias_dtype = dtype_config.bias_dtype or torch.float
        output_dtype = dtype_config.output_dtype or torch.float
        qconfig_activation_dtype, qconfig_weight_dtype, qconfig_compute_dtype = \
            get_qconfig_dtypes(qconfig)
        qconfig_bias_dtype = torch.float16 \
            if (
                qconfig_activation_dtype == torch.float16
                and qconfig_weight_dtype == torch.float16
                and not is_dynamic
            ) else torch.float

        if is_dynamic:
            is_match = input_dtype == qconfig_compute_dtype and \
                output_dtype == torch.float and \
                weight_dtype == qconfig_weight_dtype
        else:
            is_match = input_dtype == qconfig_activation_dtype and \
                output_dtype == qconfig_activation_dtype and \
                weight_dtype == qconfig_weight_dtype and \
                bias_dtype == qconfig_bias_dtype
        if is_match:
            return True
    return False
