import torch
from torch.fx import GraphModule
from typing import Dict, Any, Optional
from .quantize_fx import (
    _check_is_graph_module,
    check_is_valid_convert_custom_config_dict
)
from .fx._convert_do_not_use import _convert_do_not_use

def _convert_fx_do_not_use(
        graph_module: GraphModule, is_reference: bool = False,
        convert_custom_config_dict: Dict[str, Any] = None,
        _remove_qconfig: bool = True,
        backend_config_dict: Optional[Dict[str, Any]] = None) -> torch.nn.Module:
    """
    Please do not use, this is a temporary function to migrate convert_fx
    to a new implementation
    """
    assert is_reference
    if convert_custom_config_dict is None:
        convert_custom_config_dict = {}

    _check_is_graph_module(graph_module)
    check_is_valid_convert_custom_config_dict(convert_custom_config_dict)

    quantized = _convert_do_not_use(
        graph_module, is_reference, convert_custom_config_dict,
        False, _remove_qconfig_flag=_remove_qconfig,
        backend_config_dict=backend_config_dict)

    preserved_attributes = convert_custom_config_dict.get("preserved_attributes", [])
    for attr_name in preserved_attributes:
        setattr(quantized, attr_name, getattr(graph_module, attr_name))
    return quantized
