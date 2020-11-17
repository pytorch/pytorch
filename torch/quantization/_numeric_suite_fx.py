from typing import Dict

import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
from torch.fx import GraphModule  # type: ignore
from torch.fx import map_arg  # type: ignore
from torch.fx.graph import Graph
from torch.quantization.fx.quantize import _remove_qconfig, is_activation_post_process


NON_LEAF_MODULE_TO_ADD_OBSERVER_ALLOW_LIST = {
    nnqd.Linear,
    nnq.Linear,
    nnqd.LSTM,
    nn.LSTM,
}


def remove_qconfig_observer_fx(model):
    # remove activation post process
    act_post_process_removed_graph = Graph()
    env = {}  # type: Dict[str, str]

    modules = dict(model.named_modules())

    def load_arg(a):
        return map_arg(a, lambda node: env[node.name])

    for node in model.graph.nodes:
        if node.op == "output":
            act_post_process_removed_graph.output(map_arg(node.args[0], load_arg))
            continue
        if node.op == "call_module" and is_activation_post_process(
            modules[node.target]
        ):
            # remove activation post process node
            env[node.name] = env[node.args[0].name]
        else:
            env[node.name] = act_post_process_removed_graph.node_copy(node, load_arg)

        # removes qconfig and activation_post_process modules
    _remove_qconfig(model)
    model = GraphModule(model, act_post_process_removed_graph)
    return model


def _find_match(str_list, key_str, postfix):
    split_str = key_str.split(".")
    if split_str[-1] == postfix:
        match_string = "".join(key_str.split(".")[0:-1])
        for s2 in str_list:
            pattern1 = "".join(s2.split(".")[0:-1])
            pattern2 = "".join(s2.split(".")[0:-2])
            if match_string == pattern1:
                return s2
            if match_string == pattern2:
                return s2

        # For matching "fc.weight" and "fc._packed_params._packed_params"
        if postfix == "_packed_params":
            match_string = "".join(key_str.split(".")[0:-2])
            if len(match_string) == 0:
                return None
            for s2 in str_list:
                pattern1 = "".join(s2.split(".")[0:-1])
                pattern2 = "".join(s2.split(".")[0:-2])
                if match_string == pattern1:
                    return s2
                if match_string == pattern2:
                    return s2
    else:
        return None


def compare_weights_fx(float_dict, quantized_dict):
    r"""Compare the weights of the float module with its corresponding quantized
    module. Return a dict with key corresponding to module names and each entry being
    a dictionary with two keys 'float' and 'quantized', containing the float and
    quantized weights. This dict can be used to compare and compute the quantization
    error of the weights of float and quantized models.

    Example usage:
        prepared_model = prepare_fx(float_model, qconfig_dict)
        backup_prepared_model = copy.deepcopy(prepared_model)
        quantized_model = convert_fx(prepared_model)

        qmodel = quantized_model
        wt_compare_dict = compare_weights(backup_prepared_model.state_dict(), qmodel.state_dict())
        for key in wt_compare_dict:
            print(key, compute_error(wt_compare_dict[key]['float'], wt_compare_dict[key]['quantized'].dequantize()))

    Args:
        float_dict: state dict of the float model (prepared model)
        quantized_dict: state dict of the quantized model

    Return:
        weight_dict: dict with key corresponding to module names and each entry being
        a dictionary with two keys 'float' and 'quantized', containing the float and
        quantized weights
    """
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite_fx.compare_weights_fx"
    )
    weight_dict: Dict[str, Dict] = {}
    for key in quantized_dict:
        match_key = _find_match(float_dict, key, "weight")
        if match_key is not None:
            weight_dict[key] = {}
            weight_dict[key]["float"] = float_dict[match_key]
            weight_dict[key]["quantized"] = quantized_dict[key]
            continue

        # For matching "fc.weight" and "fc._packed_params._packed_params"
        match_key = _find_match(float_dict, key, "_packed_params")
        if match_key is not None:
            weight_dict[key] = {}
            weight_dict[key]["float"] = float_dict[match_key]
            weight_dict[key]["quantized"] = quantized_dict[key][0]

        # For LSTM
        split_str = key.split(".")
        if split_str[-1] == "param" and split_str[-3] == "_all_weight_values":
            layer = split_str[-2]
            module_name = ".".join(split_str[:-3])
            float_weight_ih_key = module_name + ".weight_ih_l" + layer
            float_weight_hh_key = module_name + ".weight_hh_l" + layer
            if float_weight_ih_key in float_dict and float_weight_hh_key in float_dict:
                weight_dict[key] = {}
                weight_dict[key]["float"] = float_dict[float_weight_ih_key]
                weight_dict[key]["quantized"] = (
                    quantized_dict[key].__getstate__()[0][4][0].__getstate__()[0][0]
                )
                weight_dict[key]["float"] = float_dict[float_weight_hh_key]
                weight_dict[key]["quantized"] = (
                    quantized_dict[key].__getstate__()[0][4][1].__getstate__()[0][0]
                )

    return weight_dict
