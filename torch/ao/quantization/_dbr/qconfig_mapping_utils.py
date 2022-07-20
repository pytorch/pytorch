import torch
from typing import Callable, Dict
from ..qconfig_mapping import QConfigMapping

TYPE_TO_REPLACEMENT_TYPE: Dict[Callable, Callable] = {
    torch.add: torch.Tensor.add,
    torch.Tensor.add_: torch.Tensor.add,
    torch.mul: torch.Tensor.mul,
    torch.Tensor.mul_: torch.Tensor.mul,
}

def normalize_object_types(qconfig_mapping: QConfigMapping) -> None:
    """
    This function looks for entries in `qconfig_mapping.object_type_qconfigs`
    corresponding to PyTorch overrides of Python math functions
    such as `torch.add` and `torch.mul`. If any of these functions are found,
    it changes the type to the tensor variant of these functions.
    This is needed because the tensor variant is what is expected
    within the framework.
    """
    for object_type, qconfig in list(qconfig_mapping.object_type_qconfigs.items()):
        replacement_type = TYPE_TO_REPLACEMENT_TYPE.get(object_type, None)  # type: ignore[arg-type]
        if replacement_type is not None:
            del qconfig_mapping.object_type_qconfigs[object_type]
            qconfig_mapping.object_type_qconfigs[replacement_type] = qconfig
