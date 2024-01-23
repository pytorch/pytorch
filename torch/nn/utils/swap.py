_swap_module_params_on_conversion: bool = False


def set_swap_module_params_on_conversion(value: bool) -> None:
    """
    Sets whether to use :func:`~torch.utils.swap_tensors` instead of setting ``.data`` to
    change the existing parameters in-place when converting an ``nn.Module``.

    When enabled, the following methods will swap the existing parameters in-place:

    #. ``module.{device}()`` (e.g. ``module.cuda()``) for moving a module between devices
    #. ``module.{dtype}()`` (e.g. ``module.float()``) for converting a module to a different dtype
       (for converting a module to a different dtype)
    #. ``module.to()``
    #. ``module.load_state_dict(state_dict)``

    See also :meth:`~torch.Tensor.module_load_to` and :meth:`~torch.Tensor.module_load_from`.

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
