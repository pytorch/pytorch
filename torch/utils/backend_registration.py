from enum import Enum
import warnings
from typing import Any
import torch
from torch._C import _rename_privateuse1_backend

__all__ = ["rename_privateuse1_backend", "run_custom_mod_func", "get_custom_mod_func"]

def rename_privateuse1_backend(backend_name: str) -> None:
    r"""
    rename_privateuse1_backend(backend_name) -> None

    Note: support the custom device with privateuse1
    This is a registration API for external backends that would like to register their
    own device and C++ kernels out of tree.

    The steps are:
    (1) (In C++) implement kernels for various torch operations, and register them
        to the PrivateUse1 dispatch key.
    (2) (In python) call torch.register_privateuse1_backend("foo")

    You can now use "foo" as an ordinary device string in python.

    Note: this API can only be called once per process. Attempting to change
    the external backend after it's already been set will result in an error.

    Note(AMP): If you want to support AMP on your device, you can register a custom backend module.
    The backend must register a custom backend module with `torch._register_device_module("foo", BackendModule)`.
    BackendModule needs to have the following API's:

    (1) get_amp_supported_dtype() -> List[torch.dtype]
        get the supported dtypes on your `foo` device in AMP, maybe the `foo` device supports one more dtype.

    (2) is_autocast_enabled() -> bool
        check the AMP is enabled or not on your `foo` device.

    (3) get_autocast_dtype() -> torch.dtype
        get the supported dtype on your `foo` device in AMP, which is set by `set_autocast_dtype` or the
        default dtype, and the default dtype is `torch.float16`.

    (4) set_autocast_enabled(bool) -> None
        enable the AMP or not on your `foo` device.

    (5) set_autocast_dtype(dtype) -> None
        set the supported dtype on your `foo` device in AMP, and the dtype be contained in the dtypes got
        from `get_amp_supported_dtype`.

    Note(random): If you want to support to set seed for your device, BackendModule needs to have the following API's:

    (1) _is_in_bad_fork() -> bool
        Return `True` if now it is in bad_fork, else return `False`.

    (2) manual_seed_all(seed: int) -> None
        Sets the seed for generating random numbers for your devices.

    (3) device_count() -> int:
        Returns the number of `foo`s available.

    (4) get_rng_state(device: Union[int, str, torch.device] = 'foo') -> Tensor:
        Returns a list of ByteTensor representing the random number states of all devices.

    (5) set_rng_state(new_state: Tensor, device: Union[int, str, torch.device] = 'foo') -> None:
        Sets the random number generator state of the specified `foo` device.

    For more details, see https://pytorch.org/tutorials/advanced/extend_dispatcher.html#get-a-dispatch-key-for-your-backend
    For an existing example, see https://github.com/bdhirsh/pytorch_open_registration_example

    Example::

        >>> # xdoctest: +SKIP("failing")
        >>> torch.register_privateuse1_backend("foo")
        # This will work, assuming that you've implemented the right C++ kernels
        # to implement torch.ones.
        >>> a = torch.ones(2, device="foo")
        """
    return _rename_privateuse1_backend(backend_name)

class _LogLevel(Enum):
    WarningLevel = 0
    ErrorLevel = 1
    UndefinedLevel = 2

def _warn_or_error_func(_log_level_, message):
    if _log_level_ == _LogLevel.WarningLevel.value:
        warnings.warn(message, UserWarning, stacklevel=3)
    elif _log_level_ == _LogLevel.ErrorLevel.value:
        raise NotImplementedError(message)
    else:
        # TODO: optimize the log message in the future.
        pass

def run_custom_mod_func(_func_name_: str, _default_: Any = None, _log_level_: int = 0, *args, **kwargs) -> Any:
    r"""
    return results executing the func named `_func_name_` with `*args` and `**kwargs`,
    and the func is defined in custom device module which is registered with
    `torch.utils.rename_privateuse1_backend('foo')` and `torch._register_device_module('foo', BackendModule)`.
    If the custom device module or the func is not defined, it will give warning or error message.

    Args:
        _func_name_ (str): The function defined in custom device module.

        _default_: default return value.

        _log_level_ (int, _LogLevel): If the  custom device module or the func is not defined,
            it will give warning or error message. Default to _LogLevel.WarningLevel=1, it only
            give warning info and return the `default`.
            When set it to _LogLevel.WarningLevel=2, it will raise NotImplementedError.
            When set it to other value, there is no warning or error message, and return the `default`.

        *args, **kwargs: The arguments for the func of `_func_name_`.

    Example::

        class DummyfooModule:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def func_name(*args, **kwargs):
                ....
        torch.utils.rename_privateuse1_backend("foo")
        torch._register_device_module("foo", DummyfooModule)

        foo_is_available = run_custom_device_mod_func("is_available")
        result = run_custom_device_mod_func("func_name", None, 0, *args, **wkargs)
        # raise error/warning, you must have define func named `device_count` in `DummyfooModule` module
        run_custom_device_mod_func("device_count")
    """
    func_ = get_custom_mod_func(_func_name_, _default_, _log_level_)
    if callable(func_):
        return func_(*args, **kwargs)
    return _default_

def get_custom_mod_func(_func_name_: str, _default_: Any = None, _log_level_: int = 0):
    r""" run the callable func named `_func_name_` defined in custom device module.
    See details in `run_custom_mod_func`.
    """
    assert isinstance(_func_name_, str), f"_func_name_ must be `str`, but got `{type(_func_name_)}`."
    backend_name = torch._C._get_privateuse1_backend_name()
    message = f'Try to use torch.{backend_name}.{_func_name_}. The backend must register a custom backend '
    message += f"module with `torch._register_device_module('{backend_name}', BackendModule)`. And "
    message += f"BackendModule needs to have the following API's:\n `{_func_name_}(*args, **kwargs)`. \n"
    if hasattr(torch, backend_name):
        custom_device_mod = getattr(torch, backend_name)
        if hasattr(custom_device_mod, _func_name_):
            return getattr(custom_device_mod, _func_name_)
    _warn_or_error_func(_log_level_,
                        message + f"Not implemented for backend `{backend_name}` or func `{_func_name_}`. \n")
    return _default_
