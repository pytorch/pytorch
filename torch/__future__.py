_overwrite_module_params_on_conversion: bool = False
_swap_module_params_on_conversion: bool = False


def set_overwrite_module_params_on_conversion(value: bool) -> None:
    """
    Sets whether to assign new tensors to the parameters instead of changing the
    existing parameters in-place when converting an ``nn.Module``.

    When enabled, the following methods will assign new parameters to the module:

    #. ``module.{device}()`` (e.g. ``module.cuda()``) for moving a module between devices
    #. ``module.{dtype}()`` (e.g. ``module.float()``) for converting a module to a different dtype
       (for converting a module to a different dtype)
    #. ``module.to()``

    Args:
        value (bool): Whether to assign new tensors or not.

    """
    global _overwrite_module_params_on_conversion
    _overwrite_module_params_on_conversion = value


def get_overwrite_module_params_on_conversion() -> bool:
    """
    Returns whether to assign new tensors to the parameters instead of changing the
    existing parameters in-place when converting an ``nn.Module``. Defaults to ``False``.

    See :func:`~torch.__future__.set_overwrite_module_params_on_conversion` for more information.
    """
    return _overwrite_module_params_on_conversion


def set_swap_module_params_on_conversion(value: bool) -> None:
    """
    Sets whether to use :func:`~torch.utils.swap_tensors` instead of setting ``.data`` to
    change the existing parameters in-place when converting an ``nn.Module`` and instead
    of ``param.copy_(state_dict[key])`` when loading a state dict into an ``nn.Module``.

    .. note::
        If :func:`~torch.__future__.get_overwrite_module_params_on_conversion` returns ``True``,
        for methods other than :meth:`~nn.Module.load_state_dict` no swapping will occur.

    When enabled, the following methods will swap the existing parameters in-place:

    #. ``module.{device}()`` (e.g. ``module.cuda()``) for moving a module between devices
    #. ``module.{dtype}()`` (e.g. ``module.float()``) for converting a module to a different dtype
       (for converting a module to a different dtype)
    #. ``module.to()``
    #. ``module.load_state_dict(state_dict)``

    The semantics for :meth:`~nn.Module.load_state_dict` when this is set are as follows:

    #. For each parameter/buffer, its corresponding``state_dict['key']`` is transformed via
       :meth:`~torch.Tensor.module_load` (i.e. ``res = param.module_load(state_dict['key'])``)
    #. If necessary, ``res`` will be wrapped in an :class:`~nn.Parameter`
    #. The parameter/buffer in the module will be swapped via :func:`~torch.utils.swap_tensors`
       with ``res``

    Args:
        value (bool): Whether to use :func:`~torch.utils.swap_tensors` or not.

    """
    global _swap_module_params_on_conversion
    _swap_module_params_on_conversion = value


def get_swap_module_params_on_conversion() -> bool:
    """
    Returns whether to use :func:`~torch.utils.swap_tensors` instead of setting .data to
    change the existing parameters in-place when converting an ``nn.Module``. Defaults to ``False``.

    See :func:`~torch.__future__.set_swap_module_params_on_conversion` for more information.
    """
    return _swap_module_params_on_conversion
