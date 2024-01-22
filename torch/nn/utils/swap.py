_swap_module_params_on_conversion: bool = False


def set_swap_module_params_on_conversion(value: bool):
    """
    Sets whether to use :func:`~torch.utils.swap_tensors` instead of setting ``.data`` to
    change the existing parameters in-place when converting an ``nn.Module`` using

    1. ``module.{device}()`` (e.g. ``module.cuda()``) for moving module between devices
    2. ``module.{dtype}()`` (e.g. ``module.float()``) for converting module to a different dtype
        (for converting module to a different dtype)
    3. ``module.to()``
    """
    global _swap_module_params_on_conversion
    _swap_module_params_on_conversion = value


def get_swap_module_params_on_conversion() -> bool:
    """
    Returns whether to use :func:`~torch.utils.swap_tensors` instead of setting .data to
    change the existing parameters in-place when converting an nn.Module.

    See :func:`~torch.nn.utils.set_swap_module_params_on_conversion` for more information.
    """
    return _swap_module_params_on_conversion
