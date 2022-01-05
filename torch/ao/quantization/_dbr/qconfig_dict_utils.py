from typing import Dict, Any

import torch

TYPE_TO_REPLACEMENT_TYPE = {
    torch.add: torch.Tensor.add,
    torch.Tensor.add_: torch.Tensor.add,
    torch.mul: torch.Tensor.mul,
    torch.Tensor.mul_: torch.Tensor.mul,
}

def normalize_object_types(qconfig_dict: Dict[str, Any]) -> None:
    """
    This function looks for entries in `qconfig_dict['object_type']`
    corresponding to PyTorch overrides of Python math functions
    such as `torch.add` and `torch.mul`. If any of these functions are found,
    it changes the type to the tensor variant of these functions.
    This is needed because the tensor variant is what is expected
    within the framework.
    """
    if 'object_type' not in qconfig_dict:
        return

    for idx, (target_type, qconfig) in enumerate(qconfig_dict['object_type']):
        replacement_type = TYPE_TO_REPLACEMENT_TYPE.get(target_type, None)
        if replacement_type is not None:
            qconfig_dict['object_type'][idx] = (replacement_type, qconfig)
