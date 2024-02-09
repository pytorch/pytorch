_overwrite_module_params_on_conversion: bool = False
_swap_module_params_on_conversion: bool = False


def set_overwrite_module_params_on_conversion(value: bool) -> None:
    """
    Sets whether to  assign new tensors to the parameters instead of changing the
    existing parameters in-place when converting an ``nn.Module``.

    When enabled, the following methods will assign new parameters to the module:

    #. ``module.{device}()`` (e.g. ``module.cuda()``) for moving a module between devices
    #. ``module.{dtype}()`` (e.g. ``module.float()``) for converting a module to a different dtype
       (for converting a module to a different dtype)
    #. ``module.to()``

    """
    global _overwrite_module_params_on_conversion
    _overwrite_module_params_on_conversion = value


def get_overwrite_module_params_on_conversion() -> bool:
    """
    Returns whether to assign new tensors to the parameters instead of changing the
    existing parameters in-place when converting an ``nn.Module`. Defaults to ``False``.

    See :func:`~torch.nn.utils.set_overwrite_module_params_on_conversion` for more information.
    """
    return _overwrite_module_params_on_conversion


def set_swap_module_params_on_conversion(value: bool) -> None:
    """
    Sets whether to use :func:`~torch.utils.swap_tensors` instead of setting ``.data`` to
    change the existing parameters in-place when converting an ``nn.Module``.

    .. note::
        If :func:`~torch.__future__.get_overwrite_module_params_on_conversion` returns ``True``,
        no swapping will occur.

    When enabled, the following methods will swap the existing parameters in-place:

    #. ``module.{device}()`` (e.g. ``module.cuda()``) for moving a module between devices
    #. ``module.{dtype}()`` (e.g. ``module.float()``) for converting a module to a different dtype
       (for converting a module to a different dtype)
    #. ``module.to()``

    """
    global _swap_module_params_on_conversion
    _swap_module_params_on_conversion = value


def get_swap_module_params_on_conversion() -> bool:
    """
    Returns whether to use :func:`~torch.utils.swap_tensors` instead of setting .data to
    change the existing parameters in-place when converting an nn.Module. Defaults to ``False``.

    See :func:`~torch.nn.utils.set_swap_module_params_on_conversion` for more information.
    """
    return _swap_module_params_on_conversion
