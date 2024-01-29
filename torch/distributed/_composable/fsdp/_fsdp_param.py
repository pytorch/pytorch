from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn

from ._fsdp_common import FSDPMeshInfo


@dataclass
class ParamModuleInfo:
    """
    For a parameter, this stores the module and the parameter name to be able
    to do a parameter swap via ``setattr(module, param_name, ...)`` or to get
    the parameter via ``getattr(module, param_name)``. We additionally save
    shared modules and shared parameter names to update them accordingly.
    """

    # Parameter names are unprefixed, e.g. "weight", not "lin.weight"
    module: nn.Module
    param_name: str
    shared_modules: List[nn.Module] = field(default_factory=list)
    shared_param_names: List[str] = field(default_factory=list)


class FSDPParam:
    """
    This class manages a parameter with FSDP or FSDP variants applied,
    implementing dim-0 per-parameter sharding.
    """

    def __init__(
        self,
        param: nn.Parameter,
        module_info: ParamModuleInfo,
        mesh_info: FSDPMeshInfo,
        device: torch.device,
    ):
        self._module_info: ParamModuleInfo = module_info
        self.mesh_info = mesh_info
        self.device = device
