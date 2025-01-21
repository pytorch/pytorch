# mypy: allow-untyped-defs
from typing import Any, Callable, Dict, List, Optional, Set, Union

import torch
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.nn as nn
from torch.ao.quantization import prepare
from torch.ao.quantization.quantization_mappings import (
    get_default_compare_output_module_list,
)


NON_LEAF_MODULE_TO_ADD_OBSERVER_ALLOW_LIST = {
    nnqd.Linear,
    nnq.Linear,
    nnqd.LSTM,
    nn.LSTM,
}


def _find_match(
    str_list: Union[Dict[str, Any], List[str]],
    key_str: str,
    postfix: str,
) -> Optional[str]:
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
        return None
    else:
        return None


def compare_weights(
    float_dict: Dict[str, Any], quantized_dict: Dict[str, Any]
) -> Dict[str, Dict[str, torch.Tensor]]:
    r"""Compare the weights of the float module with its corresponding quantized
    module. Return a dict with key corresponding to module names and each entry being
    a dictionary with two keys 'float' and 'quantized', containing the float and
    quantized weights. This dict can be used to compare and compute the quantization
    error of the weights of float and quantized models.

    Example usage::

        wt_compare_dict = compare_weights(
            float_model.state_dict(), qmodel.state_dict())
        for key in wt_compare_dict:
            print(
                key,
                compute_error(
                    wt_compare_dict[key]['float'],
                    wt_compare_dict[key]['quantized'].dequantize()
                )
            )

    Args:
        float_dict: state dict of the float model
        quantized_dict: state dict of the quantized model

    Return:
        weight_dict: dict with key corresponding to module names and each entry being
        a dictionary with two keys 'float' and 'quantized', containing the float and
        quantized weights
    """
    torch._C._log_api_usage_once("quantization_api._numeric_suite.compare_weights")
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


def _get_logger_dict_helper(
    mod: nn.Module,
    target_dict: Dict[str, Any],
    prefix: str = "",
) -> None:
    r"""This is the helper function for get_logger_dict

    Args:
        mod: module we want to save all logger stats
        prefix: prefix for the current module
        target_dict: the dictionary used to save all logger stats
    """

    def get_prefix(prefix):
        return prefix if prefix == "" else prefix + "."

    for name, child in mod.named_children():
        if isinstance(child, Logger):
            target_dict[get_prefix(prefix) + "stats"] = child.stats
            break

    for name, child in mod.named_children():
        module_prefix = get_prefix(prefix) + name if prefix else name
        _get_logger_dict_helper(child, target_dict, module_prefix)


def get_logger_dict(mod: nn.Module, prefix: str = "") -> Dict[str, Dict]:
    r"""Traverse the modules and save all logger stats into target dict.
    This is mainly used for quantization accuracy debug.

    Type of loggers supported:
        ShadowLogger: used to log the outputs of the quantized module and its matching float shadow module,
        OutputLogger: used to log the outputs of the modules

    Args:
        mod: module we want to save all logger stats
        prefix: prefix for the current module

    Return:
        target_dict: the dictionary used to save all logger stats

    """
    torch._C._log_api_usage_once("quantization_api._numeric_suite.get_logger_dict")

    target_dict: Dict[str, Dict] = {}
    _get_logger_dict_helper(mod, target_dict, prefix)
    return target_dict


class Logger(nn.Module):
    r"""Base class for stats logging"""

    def __init__(self):
        super().__init__()
        self.stats = {}
        # We only insert observer if the op is quantized with static quantization,
        # which is identified by activation_observer.dtype == quint8.  This is needed
        # when attaching Logger as observer for FX mode
        self.dtype = torch.quint8

    def forward(self, x):
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on


class ShadowLogger(Logger):
    r"""Class used in Shadow module to record the outputs of the original and
    shadow modules.
    """

    def __init__(self):
        super().__init__()
        self.stats["float"] = []
        self.stats["quantized"] = []

    def forward(self, x, y):  # type: ignore[override]
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        if len(x) > 1:
            x = x[0]
        if len(y) > 1:
            y = y[0]
        self.stats["quantized"].append(x.detach())
        self.stats["float"].append(y.detach())


class OutputLogger(Logger):
    r"""Class used to log the outputs of the module"""

    def __init__(self):
        super().__init__()
        self.stats["tensor_val"] = []

    def forward(self, x):
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        self.stats["tensor_val"].append(x)
        return x


def _convert_tuple_to_list(t: Any) -> Any:
    return [_convert_tuple_to_list(x) for x in t] if type(t) is tuple else t


def _dequantize_tensor_list(t: Any) -> Any:
    return (
        [_dequantize_tensor_list(x) for x in t]
        if type(t) is list
        else t.dequantize()
        if t.is_quantized
        else t
    )


class Shadow(nn.Module):
    r"""Shadow module attaches the float module to its matching quantized module
    as the shadow. Then it uses Logger module to process the outputs of both
    modules.

    Args:
        q_module: module quantized from float_module that we want to shadow
        float_module: float module used to shadow q_module
        logger_cls: type of logger used to process the outputs of q_module and
            float_module. ShadowLogger or custom loggers can be used.
    """

    def __init__(self, q_module, float_module, logger_cls):
        super().__init__()
        self.orig_module = q_module
        self.shadow_module = float_module
        self.dequant = nnq.DeQuantize()
        self.logger = logger_cls()

    def forward(self, *x) -> torch.Tensor:
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        xl = _convert_tuple_to_list(x)
        output = self.orig_module(*xl)
        xl_float = _dequantize_tensor_list(xl)
        shadow_output = self.shadow_module(*xl_float)
        self.logger(output, shadow_output)
        return output

    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        output = self.orig_module.add(x, y)
        x = x.dequantize()
        y = y.dequantize()
        shadow_output = self.shadow_module.add(x, y)
        self.logger(output, shadow_output)
        return output

    def add_scalar(self, x: torch.Tensor, y: float) -> torch.Tensor:
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        output = self.orig_module.add_scalar(x, y)
        x = x.dequantize()
        shadow_output = self.shadow_module.add_scalar(x, y)
        self.logger(output, shadow_output)
        return output

    def mul(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        output = self.orig_module.mul(x, y)
        x = x.dequantize()
        y = y.dequantize()
        shadow_output = self.shadow_module.mul(x, y)
        self.logger(output, shadow_output)
        return output

    def mul_scalar(self, x: torch.Tensor, y: float) -> torch.Tensor:
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        output = self.orig_module.mul_scalar(x, y)
        x = x.dequantize()
        shadow_output = self.shadow_module.mul_scalar(x, y)
        self.logger(output, shadow_output)
        return output

    def cat(self, x: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        output = self.orig_module.cat(x, dim)
        x = [y.dequantize() for y in x]
        shadow_output = self.shadow_module.cat(x, dim)
        self.logger(output, shadow_output)
        return output

    def add_relu(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        output = self.orig_module.add_relu(x, y)
        x = x.dequantize()
        y = y.dequantize()
        shadow_output = self.shadow_module.add_relu(x, y)
        self.logger(output, shadow_output)
        return output


def prepare_model_with_stubs(
    float_module: nn.Module,
    q_module: nn.Module,
    module_swap_list: Set[type],
    logger_cls: Callable,
) -> None:
    r"""Prepare the model by attaching the float module to its matching quantized
    module as the shadow if the float module type is in module_swap_list.

    Example usage::

        prepare_model_with_stubs(float_model, q_model, module_swap_list, Logger)
        q_model(data)
        ob_dict = get_logger_dict(q_model)

    Args:
        float_module: float module used to generate the q_module
        q_module: module quantized from float_module
        module_swap_list: list of float module types to attach the shadow
        logger_cls: type of logger to be used in shadow module to process the outputs of
            quantized module and its float shadow module
    """
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite.prepare_model_with_stubs"
    )

    float_module_children = {}
    for name, mod in float_module.named_children():
        float_module_children[name] = mod

    reassign = {}
    for name, mod in q_module.named_children():
        if name not in float_module_children:
            continue

        float_mod = float_module_children[name]

        if type(float_mod) not in module_swap_list:
            prepare_model_with_stubs(float_mod, mod, module_swap_list, logger_cls)

        # Insert shadow module only if the module is not of the same type as
        # the floating point module
        if type(float_mod) in module_swap_list and not _is_identical_module_type(
            mod, float_mod
        ):
            reassign[name] = Shadow(mod, float_mod, logger_cls)

    for key, value in reassign.items():
        q_module._modules[key] = value


def _is_identical_module_type(mod1, mod2):
    # Compare if two modules have the same dtype
    mod1_module_types = [type(mod) for mod in mod1.modules()]
    mod2_module_types = [type(mod) for mod in mod2.modules()]
    return mod1_module_types == mod2_module_types


def compare_model_stub(
    float_model: nn.Module,
    q_model: nn.Module,
    module_swap_list: Set[type],
    *data,
    logger_cls=ShadowLogger,
) -> Dict[str, Dict]:
    r"""Compare quantized module in a model with its floating point counterpart,
    feeding both of them the same input. Return a dict with key corresponding to
    module names and each entry being a dictionary with two keys 'float' and
    'quantized', containing the output tensors of quantized and its matching
    float shadow module. This dict can be used to compare and compute the module
    level quantization error.

    This function first call prepare_model_with_stubs() to swap the quantized
    module that we want to compare with the Shadow module, which takes quantized
    module, corresponding float module and logger as input, and creates a forward
    path inside to make the float module to shadow quantized module sharing the
    same input. The logger can be customizable, default logger is ShadowLogger
    and it will save the outputs of the quantized module and float module that
    can be used to compute the module level quantization error.

    Example usage::

        module_swap_list = [torchvision.models.quantization.resnet.QuantizableBasicBlock]
        ob_dict = compare_model_stub(float_model,qmodel,module_swap_list, data)
        for key in ob_dict:
            print(key, compute_error(ob_dict[key]['float'], ob_dict[key]['quantized'].dequantize()))

    Args:
        float_model: float model used to generate the q_model
        q_model: model quantized from float_model
        module_swap_list: list of float module types at which shadow modules will
            be attached.
        data: input data used to run the prepared q_model
        logger_cls: type of logger to be used in shadow module to process the outputs of
            quantized module and its float shadow module
    """
    torch._C._log_api_usage_once("quantization_api._numeric_suite.compare_model_stub")
    prepare_model_with_stubs(float_model, q_model, module_swap_list, logger_cls)
    q_model(*data)
    ob_dict = get_logger_dict(q_model)
    return ob_dict


def get_matching_activations(
    float_module: nn.Module,
    q_module: nn.Module,
) -> Dict[str, Dict[str, torch.Tensor]]:
    r"""Find the matching activation between float and quantized modules.

    Args:
        float_module: float module used to generate the q_module
        q_module: module quantized from float_module

    Return:
        act_dict: dict with key corresponding to quantized module names and each
        entry being a dictionary with two keys 'float' and 'quantized', containing
        the matching float and quantized activations
    """
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite.get_matching_activations"
    )
    float_dict = get_logger_dict(float_module)
    quantized_dict = get_logger_dict(q_module)
    act_dict: Dict[str, Dict] = {}
    for key in quantized_dict:
        if len(quantized_dict[key]["tensor_val"]) == 0:
            continue
        match_key = _find_match(sorted(float_dict, reverse=True), key, "stats")
        if match_key is not None:
            act_dict[key] = {}
            act_dict[key]["float"] = float_dict[match_key]["tensor_val"]
            act_dict[key]["quantized"] = quantized_dict[key]["tensor_val"]
    return act_dict


def prepare_model_outputs(
    float_module: nn.Module,
    q_module: nn.Module,
    logger_cls=OutputLogger,
    allow_list=None,
) -> None:
    r"""Prepare the model by attaching the logger to both float module
    and quantized module if they are in the allow_list.

    Args:
        float_module: float module used to generate the q_module
        q_module: module quantized from float_module
        logger_cls: type of logger to be attached to float_module and q_module
        allow_list: list of module types to attach logger
    """
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite.prepare_model_outputs"
    )
    if allow_list is None:
        allow_list = get_default_compare_output_module_list()

    qconfig_debug = torch.ao.quantization.QConfig(activation=logger_cls, weight=None)
    float_module.qconfig = qconfig_debug  # type: ignore[assignment]
    prepare(
        float_module, inplace=True, allow_list=allow_list, prepare_custom_config_dict={}
    )
    q_module.qconfig = qconfig_debug  # type: ignore[assignment]
    prepare(
        q_module,
        inplace=True,
        allow_list=allow_list,
        observer_non_leaf_module_list=NON_LEAF_MODULE_TO_ADD_OBSERVER_ALLOW_LIST,
        prepare_custom_config_dict={},
    )


def compare_model_outputs(
    float_model: nn.Module,
    q_model: nn.Module,
    *data,
    logger_cls=OutputLogger,
    allow_list=None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    r"""Compare output activations between float and quantized models at
    corresponding locations for the same input. Return a dict with key corresponding
    to quantized module names and each entry being a dictionary with two keys
    'float' and 'quantized', containing the activations of quantized model and
    float model at matching locations. This dict can be used to compare and
    compute the propagation quantization error.

    Example usage::

        act_compare_dict = compare_model_outputs(float_model, qmodel, data)
        for key in act_compare_dict:
            print(
                key,
                compute_error(
                    act_compare_dict[key]['float'],
                    act_compare_dict[key]['quantized'].dequantize()
                )
            )

    Args:
        float_model: float model used to generate the q_model
        q_model: model quantized from float_model
        data: input data used to run the prepared float_model and q_model
        logger_cls: type of logger to be attached to float_module and q_module
        allow_list: list of module types to attach logger

    Return:
        act_compare_dict: dict with key corresponding to quantized module names
        and each entry being a dictionary with two keys 'float' and 'quantized',
        containing the matching float and quantized activations
    """
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite.compare_model_outputs"
    )
    if allow_list is None:
        allow_list = get_default_compare_output_module_list()
    prepare_model_outputs(float_model, q_model, logger_cls, allow_list)
    float_model(*data)
    q_model(*data)
    act_compare_dict = get_matching_activations(float_model, q_model)
    return act_compare_dict
