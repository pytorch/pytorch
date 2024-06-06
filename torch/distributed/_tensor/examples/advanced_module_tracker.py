import torch
from torch.autograd.graph import register_multi_grad_hook

from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)
from torch.utils._pytree import tree_flatten

from torch.utils.module_tracker import ModuleTracker


class ModuleParamaterShardingTracker(ModuleTracker):
    """
    Inherits ModuleTracker and expands on its functionality to track the parameters and sharding information of a model at a module-level
    """

    def __init__(self):
        super().__init__()
        self.module_parameters_dict = {}
        self.sharding_dict = {}

    def _fw_pre_hook(self, mod, input):
        name = super()._get_mod_name(mod)
        super()._get_append_fn(name, False)()

        args, _ = tree_flatten(input)
        tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
        if tensors:
            register_multi_grad_hook(tensors, super()._get_pop_fn(name, True))

        for param_name, param in mod.named_parameters(recurse=False):
            if name not in self.module_parameters_dict:
                self.module_parameters_dict[name] = {}

            self.module_parameters_dict[name][param_name] = param.data
            if type(param.data) == torch.distributed._tensor.api.DTensor:
                key_name = name + "." + param_name
                self.sharding_dict[key_name] = param.data.placements

    def __enter__(self):
        self.module_parameters_dict.clear()
        self.sharding_dict.clear()
        self._fw_pre_handle = register_module_forward_pre_hook(self._fw_pre_hook)
        self._fw_post_handle = register_module_forward_hook(super()._fw_post_hook)

    def __exit__(self, *args):
        super().__exit__(*args)

    def print_paramater_info(self):
        print(self.module_parameters_dict)

    def print_sharding_info(self):
        for key, value in self.sharding_dict.items():
            print(key + ": " + str(value))
