"""
_overwrite_module_params_on_conversion controls whether to assign new tensors to the parameters
instead of changing the existing parameters in-place when converting an `nn.Module`
using the following methods:
1. `module.cuda()` / `.cpu()` (for moving `module` between devices)
2. `module.float()` / `.double()` / `.half()` (for converting `module` to a different dtype)
3. `module.to()` / `.type()` (for changing `module`'s device or dtype)
4. `module._apply(fn)` (for generic functions applied to `module`)

Default: False

_swap_tensors_on_conversion controls whether to use torch.utils.swap_tensors instead
of setting .data to change the existing paramaters in-place when converting an `nn.Module`
using the methods listed above. It is not used when _overwrite_module_params_on_conversion = True
"""
_overwrite_module_params_on_conversion = False
_swap_module_params_on_conversion = False


def set_overwrite_module_params_on_conversion(value):
    global _overwrite_module_params_on_conversion
    _overwrite_module_params_on_conversion = value


def get_overwrite_module_params_on_conversion():
    return _overwrite_module_params_on_conversion


def set_swap_module_params_on_conversion(value):
    global _swap_module_params_on_conversion
    _swap_module_params_on_conversion = value


def get_swap_module_params_on_conversion():
    return _swap_module_params_on_conversion
