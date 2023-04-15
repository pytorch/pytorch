import torch
from torch._C import _rename_privateuse1_backend, _get_privateuse1_backend_name
from typing import Union, Optional

__all__ = ["rename_privateuse1_backend", "generate_methods_for_privateuse1_backend"]

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

    And there are some common funcs:
    (1) is_available() -> bool:
        Returns a bool indicating if `foo` is currently available.
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


def _check_register_once(module, attr):
    if hasattr(module, attr):
        raise RuntimeError(f"The custom device module of {module} has already been registered with {attr}")


def generate_methods_for_privateuse1_backend() -> None:
    r"""
    generate_methods_for_privateuse1_backend() -> None

    Automatically generate attributes and methods for the custom backend after rename privateuse1 backend.

    When you implement kernels for various torch operations, and register them to the PrivateUse1 dispatch key.
    And call the function torch.rename_privateuse1_backend("foo") to rename your backend name.
    At this point, you can easily register specific methods and attributes by calling this function.
    Just like torch.Tensor.foo(), torch.Tensor.is_foo.

    Note: We recommend you use generic functions (check devices are equal or to(device=)).
    We provide these methods for convenience only and they will be "monkey patched" onto the objects
    and so will not be properly typed.

    Example::

        >>> # xdoctest: +SKIP("failing")
        >>> torch.utils.register_privateuse1_backend("foo")
        >>> torch.utils.generate_for_privateuse1_backend()
        # Then automatically generate backend-related attributes and methods.
        >>> a = torch.tensor(2).foo()
        >>> a.is_foo
        >>> hasattr(torch.nn.Module, 'foo')
        """
    custom_backend_name = _get_privateuse1_backend_name()

    @property  # type: ignore[misc]
    def wrap_tensor_backend(self: torch.Tensor) -> bool:
        return self.device.type == custom_backend_name

    _check_register_once(torch.Tensor, f'is_{custom_backend_name}')
    setattr(torch.Tensor, f'is_{custom_backend_name}', wrap_tensor_backend)

    def wrap_tensor_to(self: torch.Tensor, device: Optional[Union[int, torch.device]] = 0) -> torch.Tensor:
        if isinstance(device, torch.device):
            if device.type == custom_backend_name:
                return self.to(f'{custom_backend_name}:{device.index}')
            else:
                raise RuntimeError(f"Invalid device, must be {custom_backend_name} device")
        return self.to(f'{custom_backend_name}:{device}')

    _check_register_once(torch.Tensor, f'{custom_backend_name}')
    setattr(torch.Tensor, f'{custom_backend_name}', wrap_tensor_to)

    def wrap_module_to(self: torch.nn.modules.module.T,
                       device: Optional[Union[int, torch.device]] = None) -> torch.nn.modules.module.T:
        return self._apply(lambda t: getattr(t, f'{custom_backend_name}')(device))

    _check_register_once(torch.nn.Module, f'{custom_backend_name}')
    setattr(torch.nn.Module, f'{custom_backend_name}', wrap_module_to)
