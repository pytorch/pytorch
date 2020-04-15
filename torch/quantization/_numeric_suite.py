from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
import torch.nn.quantized as nnq

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
    else:
        return None


def compare_weights(float_dict, quantized_dict):
    r"""Returns a dict with key corresponding to module names and each entry being
    a dictionary with two keys 'float' and 'quantized', containing the float and
    quantized weights. This dict can be used to compare and compute the quantization
    error of the weights of float and quantized models .

    Args:
        float_dict: state dict of the float model
        quantized_dict: state dict of the quantized model

    Return:
        weight_dict: dict with key corresponding to module names and each entry being
        a dictionary with two keys 'float' and 'quantized', containing the float and
        quantized weights
    """
    weight_dict = {}
    for key in quantized_dict:
        match_key = _find_match(float_dict, key, "weight")
        if match_key is not None:
            weight_dict[key] = {}
            weight_dict[key]["float"] = float_dict[match_key]
            weight_dict[key]["quantized"] = quantized_dict[key]
    return weight_dict


def _get_observer_dict(mod, target_dict, observer_name, prefix=""):
    r"""Traverse the modules and save all observers into dict.
    This is mainly used for quantization accuracy debug
    Args:
        mod: the top module we want to save all observers
        prefix: the prefix for the current module
        observer_name: the name of observer we want to get, "logger" is used
            to do the module level comparison between quantized module and its
            matching float shadow module, and "activation_post_process" is
            used to compare the module outputs between float and quantized
            models
        target_dict: the dictionary used to save all the observers
    """

    def get_prefix(prefix):
        return prefix if prefix == "" else prefix + "."

    if hasattr(mod, observer_name):
        if observer_name == "logger":
            target_dict[get_prefix(prefix) + "stats"] = mod.logger.stats
        elif observer_name == "activation_post_process":
            target_dict[
                get_prefix(prefix) + "stats"
            ] = mod.activation_post_process.tensor_val

    for name, child in mod.named_children():
        module_prefix = get_prefix(prefix) + name if prefix else name
        _get_observer_dict(child, target_dict, observer_name, module_prefix)


class Logger(nn.Module):
    r"""Class used in Shadow module to process the outputs of the original and
    shadow modules
    """

    def __init__(self):
        super(Logger, self).__init__()
        self.stats = {}
        self.stats["float"] = None
        self.stats["quantized"] = None

    def forward(self, x, y):
        if self.stats["float"] is None:
            if x.is_quantized:
                self.stats["quantized"] = x.dequantize().detach()
            else:  # Output is in float for dynamic quantization
                self.stats["quantized"] = x.detach()

            self.stats["float"] = y.detach()
        else:
            if x.is_quantized:
                self.stats["quantized"] = torch.cat(
                    (self.stats["quantized"], x.dequantize().detach())
                )
            else:  # Output is in float for dynamic quantization
                self.stats["quantized"] = torch.cat(
                    (self.stats["quantized"], x.detach())
                )

            self.stats["float"] = torch.cat((self.stats["float"], y.detach()))


class Shadow(nn.Module):
    r"""Shadow module attaches the float module to its matching quantized module
    as the shadow. Then it uses Logger module to process the outputs of both
    modules to do the comparison.
    """

    def __init__(self, q_module, float_module, Logger):
        super(Shadow, self).__init__()
        self.orig_module = q_module
        self.shadow_module = float_module
        self.dequant = nnq.DeQuantize()
        self.logger = Logger()

    def forward(self, x):
        output = self.orig_module(x)
        x = x.dequantize()
        shadow_output = self.shadow_module(x)
        self.logger(output, shadow_output)
        return output


def _prepare_model_with_stubs(float_module, q_module, module_swap_list, Logger):
    r"""Prepare the model by attaching the float module to its matching quantized
    module as the shadow if the float module type is in module_swap_list.

    Args:
        float_module: the float module used to generate the q_module
        q_module: the quantized module
        module_swap_list: list of float module types to attach the shadow
        Logger: the class to be used in shadow module to process the outputs of
            quantized module and its float shadow module
    """

    float_module_children = {}
    for name, mod in float_module.named_children():
        float_module_children[name] = mod

    reassign = {}
    for name, mod in q_module.named_children():
        if name not in float_module_children:
            continue

        float_mod = float_module_children[name]

        if type(float_mod) not in module_swap_list:
            _prepare_model_with_stubs(float_mod, mod, module_swap_list, Logger)

        if type(float_mod) in module_swap_list:
            reassign[name] = Shadow(mod, float_mod, Logger)

    for key, value in reassign.items():
        q_module._modules[key] = value


def compare_model_stub(float_model, q_model, module_swap_list, data, Logger=Logger):
    r"""Returns a dict with key corresponding to module names and each entry being
    a dictionary with two keys 'float' and 'quantized', containing the output
    tensors of quantized and its matching float shadow module. This dict can be
    used to compare and compute the module level quantization error.

    Args:
        float_module: the float module used to generate the q_module
        q_module: the quantized module
        module_swap_list: list of float module types to attach the shadow
        Logger: the class to be used in shadow module to process the outputs of
            quantized module and its float shadow module
    """
    _prepare_model_with_stubs(float_model, q_model, module_swap_list, Logger)
    q_model(data)
    ob_dict = {}
    _get_observer_dict(q_model, ob_dict, "logger")
    return ob_dict
