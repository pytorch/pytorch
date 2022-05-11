import torch
from collections import defaultdict, OrderedDict
import dataclasses
from typing import Callable, Any, Dict, Tuple, Set, Optional, List
from torch.ao.quantization import QConfig
from torch.ao.quantization.qconfig import add_module_to_qconfig_obs_ctr, QConfigAny, qconfig_equals
from torch.ao.quantization.quantize import (
    is_activation_post_process,
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
from ..quantization_config import (
    OBJECT_TYPE_DICT_KEY,
    MODULE_NAME_DICT_KEY,
    MODULE_NAME_REGEX_DICT_KEY,
    ConvertQuantizationConfig,
    PrepareQuantizationConfig,
    QuantizationConfigBase,
)
from ..qconfig_dict_utils import (
    get_object_type_qconfig,
    maybe_adjust_qconfig_for_module_type_or_name,
)


def maybe_adjust_qconfig_for_module_name_object_type_order(
    quantization_config: QuantizationConfigBase,
    cur_module_path: str,
    cur_object_type: Callable,
    cur_object_type_idx: int,
    fallback_qconfig: QConfigAny,
) -> QConfigAny:
    for qconfig_entry in quantization_config.module_name_object_type_order_qconfigs:
        (module_path, object_type, object_type_idx, qconfig) = dataclasses.astuple(qconfig_entry)
        if (
            (module_path == cur_module_path) and
            (object_type == cur_object_type) and
            (object_type_idx == cur_object_type_idx)
        ):
            return qconfig

    return fallback_qconfig


def update_qconfig_for_fusion(model: GraphModule, quantization_config: QuantizationConfigBase):
    """
    Update the QuantizationConfigBase to account for fused modules such as LinearReLU.
    This assumes the QuantizationConfigBase's attributes have already been converted to OrderedDicts.
    """
    object_type_dict = quantization_config._object_type_qconfig_dict
    if len(object_type_dict) == 0:
        return quantization_config

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

def generate_qconfig_map(
        root: torch.nn.Module,
        modules: Dict[str, torch.nn.Module],
        input_graph: Graph,
        quantization_config: QuantizationConfigBase,
        node_name_to_scope: Dict[str, Tuple[str, type]]) -> Dict[str, QConfigAny]:
    global_qconfig = quantization_config.global_qconfig
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
                quantization_config, type(modules[module_name]), module_name, global_qconfig)
            qconfig_with_device_check = add_module_to_qconfig_obs_ctr(qconfig, modules.get(node.target, None))
        elif node.op == "call_function":
            # precedence: module_name_qconfig
            # > function_qconfig > global_qconfig
            # module_name takes precedence over function qconfig
            function_qconfig = get_object_type_qconfig(
                quantization_config, node.target, global_qconfig)
            module_path, module_type = node_name_to_scope[node.name]
            qconfig = maybe_adjust_qconfig_for_module_type_or_name(
                quantization_config, module_type, module_path, function_qconfig)

            cur_object_type_idx = \
                submodule_to_object_type_to_cur_idx[module_path][node.target]
            submodule_to_object_type_to_cur_idx[module_path][node.target] += 1
            qconfig = maybe_adjust_qconfig_for_module_name_object_type_order(
                quantization_config, module_path, node.target, cur_object_type_idx, qconfig)
            qconfig_with_device_check = add_module_to_qconfig_obs_ctr(qconfig, modules.get(node.target, None))

        elif node.op == "call_method":
            module_path, module_type = node_name_to_scope[node.name]
            # first use node.target (string) to get the qconfig
            # this is to support configs like
            # "object_type": [("reshpe", qconfig)]
            qconfig = maybe_adjust_qconfig_for_module_type_or_name(
                quantization_config, node.target, module_path, global_qconfig)
            # if there is no special config for the method, we'll fall back to the
            # config for the module that contains the call_method node
            qconfig = maybe_adjust_qconfig_for_module_type_or_name(
                quantization_config, module_type, module_path, qconfig)
            # currently call_method does not support modifying qconfig
            # by order, we can add this later if it is needed.
            qconfig_with_device_check = add_module_to_qconfig_obs_ctr(qconfig, modules.get(node.target, None))

        elif node.op == 'call_module':
            # if the node is an observer, just continue - don't add it to the qconfig_map
            if is_activation_post_process(modules[node.target]):
                continue
            qconfig = maybe_adjust_qconfig_for_module_type_or_name(
                quantization_config, type(modules[node.target]), node.target, global_qconfig)

            module_path, module_type = node_name_to_scope[node.name]
            # Note: for call_module, the module_path is the current module's name.
            # to meaningfully count invocations, we need to count them in the parent
            # module.
            parent_name, _ = _parent_name(module_path)
            cur_object_type_idx = \
                submodule_to_object_type_to_cur_idx[parent_name][module_type]
            submodule_to_object_type_to_cur_idx[parent_name][module_type] += 1
            qconfig = maybe_adjust_qconfig_for_module_name_object_type_order(
                quantization_config, parent_name, module_type, cur_object_type_idx,
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


def check_is_valid_prepare_custom_config_dict(prepare_custom_config_dict: Optional[Dict[str, Any]] = None) -> None:
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
                                               "input_quantized_idxs",
                                               "output_quantized_idxs",
                                               "preserved_attributes"}
    check_is_valid_config_dict(prepare_custom_config_dict,
                               prepare_custom_config_dict_allowed_keys, "prepare_custom_config_dict")


def check_is_valid_convert_custom_config_dict(convert_custom_config_dict: Optional[Dict[str, Any]] = None) -> None:
    r""" Checks if the given convert_custom_config_dict has the correct keys

    Args:
      `convert_custom_config_dict`: dictionary for custom configurations for
      convert function
    """
    if not convert_custom_config_dict:
        return

    convert_custom_config_dict_allowed_keys = {"observed_to_quantized_custom_module_class",
                                               "preserved_attributes"}
    check_is_valid_config_dict(convert_custom_config_dict,
                               convert_custom_config_dict_allowed_keys, "convert_custom_config_dict")


def check_is_valid_fuse_custom_config_dict(fuse_custom_config_dict: Optional[Dict[str, Any]] = None) -> None:
    r""" Checks if the given fuse_custom_config_dict has the correct keys

    Args:
      `fuse_custom_config_dict`: dictionary for custom configurations for fuse_fx
    """
    if not fuse_custom_config_dict:
        return

    fuse_custom_config_dict_allowed_keys = {"preserved_attributes"}
    check_is_valid_config_dict(fuse_custom_config_dict, fuse_custom_config_dict_allowed_keys, "fuse_custom_config_dict")


def compare_prepare_convert_quantization_configs(
        prepare_quantization_config: PrepareQuantizationConfig,
        convert_quantization_config: ConvertQuantizationConfig):
    r""" Compare the quantization_config passed in convert to the one from prepare and check the values

    Args:
      `prepare_quantization_config`: configuration for prepare quantization step
      `convert_quantization_config`: configuration for convert quantization step
    """
    assert qconfig_equals(prepare_quantization_config.global_qconfig, convert_quantization_config.global_qconfig), \
        "Expected global qconfigs to be the same in the prepare and convert quantization configs"
    prepare_dicts: List[OrderedDict] = [
        prepare_quantization_config._object_type_qconfig_dict,
        prepare_quantization_config._module_name_qconfig_dict,
        prepare_quantization_config._module_name_regex_qconfig_dict,
    ]
    convert_dicts: List[OrderedDict] = [
        convert_quantization_config._object_type_qconfig_dict,
        convert_quantization_config._module_name_qconfig_dict,
        convert_quantization_config._module_name_regex_qconfig_dict,
    ]
    dict_names = [OBJECT_TYPE_DICT_KEY, MODULE_NAME_DICT_KEY, MODULE_NAME_REGEX_DICT_KEY]
    for i in range(len(prepare_dicts)):
        for name, qconfig in prepare_dicts[i].items():
            assert name in convert_dicts[i], "Missing key {} {} in ConvertQuantizationConfig \
                when it was present in prepare".format(dict_names[i], name)
            assert convert_dicts[i][name] is None \
                or qconfig_equals(prepare_dicts[i][name], convert_dicts[i][name]), \
                "Expected ConvertQuantizationConfig to have the same qconfig as prepare for key {} {}; \
                prepare: {}; convert: {}".format(dict_names[i], name, prepare_dicts[i][name], convert_dicts[i][name])

def is_qconfig_supported_by_dtype_configs(qconfig: QConfig, dtype_configs: List[Dict[str, Any]]):
    for dtype_config in dtype_configs:
        is_dynamic = dtype_config.get("is_dynamic", False)
        input_dtype = dtype_config.get("input_dtype", torch.float)
        weight_dtype = dtype_config.get("weight_dtype", torch.float)
        bias_dtype = dtype_config.get("bias_dtype", torch.float)
        output_dtype = dtype_config.get("output_dtype", torch.float)
        qconfig_activation_dtype, qconfig_weight_dtype, qconfig_compute_dtype = \
            get_qconfig_dtypes(qconfig)
        qconfig_bias_dtype = torch.float16 \
            if qconfig_activation_dtype == torch.float16 and \
            qconfig_weight_dtype == torch.float16 \
            else torch.float

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

# TODO: rename this file to config_utils
def get_standalone_module_configs(
        module_name: str,
        module_type: Callable,
        custom_config_dict: Dict[str, Any]):
    standalone_module_name_configs = \
        custom_config_dict.get("standalone_module_name", [])
    standalone_module_class_configs = \
        custom_config_dict.get("standalone_module_class", [])
    class_config_map = {x[0]: (x[1], x[2], x[3]) for x in standalone_module_class_configs}
    name_config_map = {x[0]: (x[1], x[2], x[3]) for x in standalone_module_name_configs}
    config = class_config_map.get(module_type, (None, None, None))
    # name config has precedence over type config
    config = name_config_map.get(module_name, config)
    return config
