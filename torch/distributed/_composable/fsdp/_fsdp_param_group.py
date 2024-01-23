from typing import Dict, List, Optional

import torch
import torch.nn as nn

from torch.distributed.fsdp._common_utils import _named_parameters_with_duplicates

from ._fsdp_common import FSDPMeshInfo, ParamModuleInfo, TrainingState
from ._fsdp_param import FSDPParam


class FSDPParamGroup:
    """This class represents a parameter group to communicate together."""

    def __init__(
        self,
        params: List[nn.Parameter],
        module: nn.Module,
        mesh_info: FSDPMeshInfo,
        device: torch.device,
    ):
        self.module = module  # permit ref cycle because 1:1 lifetime
        param_module_infos = _get_param_module_infos(params, module)
        self.fsdp_params = [
            FSDPParam(param, module_info, mesh_info, device)
            for param, module_info in zip(params, param_module_infos)
        ]
        self.mesh_info = mesh_info
        self.device = device
        self._training_state = TrainingState.IDLE
        self._module_fqn: Optional[str] = None  # prefixed from root module


def _get_param_module_infos(
    params: List[nn.Parameter], module: nn.Module
) -> List[ParamModuleInfo]:
    """
    Shared parameter:
        lin1.weight = lin2.weight
    Shared module:
        mlp.lin1 = mlp.lin2
    We do not remove duplicates when traversing both modules and parameters to
    find shared modules' parameters and shared parameters within a module.
    """
    params_set = set(params)
    param_to_module_info: Dict[nn.Parameter, ParamModuleInfo] = {}
    for _, submodule in module.named_modules(remove_duplicate=False):
        for param_name, param in _named_parameters_with_duplicates(
            submodule, recurse=False
        ):
            if param in params_set:
                if param not in param_to_module_info:
                    param_to_module_info[param] = ParamModuleInfo(submodule, param_name)
                else:
                    param_to_module_info[param].shared_modules.append(submodule)
                    param_to_module_info[param].shared_param_names.append(param_name)
    if len(param_to_module_info) != len(params):
        raise AssertionError(f"Some parameters are not in the module tree of {module}")
    return [param_to_module_info[param] for param in params]
