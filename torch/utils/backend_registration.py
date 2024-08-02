# mypy: allow-untyped-defs
import torch
from torch.overrides import (
    handle_torch_function,
    has_torch_function_unary,
)
from torch._C import _rename_privateuse1_backend, _get_privateuse1_backend_name
from typing import List, Optional, Union

__all__ = ["rename_privateuse1_backend", "generate_methods_for_privateuse1_backend"]

# TODO: Should use `torch._C._get_privateuse1_backend_name()` to get
# renamed-backend name for `privateuse1`, but the func will cause an
# error with torch.jit.script, so we use the global variable named
# `_privateuse1_backend_name`.
_privateuse1_backend_name = "privateuseone"

def rename_privateuse1_backend(backend_name: str) -> None:
    r"""
    Rename the privateuse1 backend device to make it more convenient to use as a device name within PyTorch APIs.

    The steps are:

    (1) (In C++) implement kernels for various torch operations, and register them
        to the PrivateUse1 dispatch key.
    (2) (In python) call torch.utils.rename_privateuse1_backend("foo")

    You can now use "foo" as an ordinary device string in python.

    Note: this API can only be called once per process. Attempting to change
    the external backend after it's already been set will result in an error.

    Note(AMP): If you want to support AMP on your device, you can register a custom backend module.
    The backend must register a custom backend module with ``torch._register_device_module("foo", BackendModule)``.
    BackendModule needs to have the following API's:

    (1) ``get_amp_supported_dtype() -> List[torch.dtype]``
        get the supported dtypes on your "foo" device in AMP, maybe the "foo" device supports one more dtype.

    Note(random): If you want to support to set seed for your device, BackendModule needs to have the following API's:

    (1) ``_is_in_bad_fork() -> bool``
        Return ``True`` if now it is in bad_fork, else return ``False``.

    (2) ``manual_seed_all(seed int) -> None``
        Sets the seed for generating random numbers for your devices.

    (3) ``device_count() -> int``
        Returns the number of "foo"s available.

    (4) ``get_rng_state(device: Union[int, str, torch.device] = 'foo') -> Tensor``
        Returns a list of ByteTensor representing the random number states of all devices.

    (5) ``set_rng_state(new_state: Tensor, device: Union[int, str, torch.device] = 'foo') -> None``
        Sets the random number generator state of the specified "foo" device.

    And there are some common funcs:

    (1) ``is_available() -> bool``
        Returns a bool indicating if "foo" is currently available.

    (2) ``current_device() -> int``
        Returns the index of a currently selected device.

    For more details, see https://pytorch.org/tutorials/advanced/extend_dispatcher.html#get-a-dispatch-key-for-your-backend
    For an existing example, see https://github.com/bdhirsh/pytorch_open_registration_example

    Example::

        >>> # xdoctest: +SKIP("failing")
        >>> torch.utils.rename_privateuse1_backend("foo")
        # This will work, assuming that you've implemented the right C++ kernels
        # to implement torch.ones.
        >>> a = torch.ones(2, device="foo")

    """
    _rename_privateuse1_backend(backend_name)
    global _privateuse1_backend_name
    _privateuse1_backend_name = backend_name

def _check_register_once(module, attr):
    if hasattr(module, attr):
        raise RuntimeError(f"The custom device module of {module} has already been registered with {attr}")


def _normalization_device(custom_backend_name: str, device: Optional[Union[int, str, torch.device]] = None) -> int:
    def _get_current_device_index():
        _get_device_index = "current_device"
        if hasattr(torch, custom_backend_name) and \
                hasattr(getattr(torch, custom_backend_name), _get_device_index):
            return getattr(getattr(torch, custom_backend_name), _get_device_index)()
        else:
            # The default device index is 0.
            return 0

    if device is None:
        return _get_current_device_index()
    # if isinstance(device, str), this means that the parameter passed in is in the string format "foo:0"
    # convert str object to torch.device object, and then process it uniformly
    elif isinstance(device, str):
        device = torch.device(device)

    # variable devcie can only be torch.device type or int type
    if isinstance(device, torch.device):
        if device.type != custom_backend_name:
            raise RuntimeError(f"Invalid device, must be {custom_backend_name} device")
        elif device.index is None:
            device_idx = _get_current_device_index()
        else:
            device_idx = device.index
    # if isinstance(device, int), we can take the index number directly
    else:
        device_idx = device
    return device_idx


def _generate_tensor_methods_for_privateuse1_backend(custom_backend_name: str) -> None:
    @property  # type: ignore[misc]
    def wrap_tensor_backend(self: torch.Tensor) -> bool:
        if has_torch_function_unary(self):
            # TODO mypy doesn't support @property, see: https://github.com/python/mypy/issues/6185
            return handle_torch_function(wrap_tensor_backend.__get__, (self,), self)  # type: ignore[attr-defined]
        return self.device.type == custom_backend_name

    _check_register_once(torch.Tensor, f'is_{custom_backend_name}')
    wrap_tensor_backend.fget.__name__ = f'is_{custom_backend_name}'  # type: ignore[attr-defined]
    setattr(torch.Tensor, f'is_{custom_backend_name}', wrap_tensor_backend)

    def wrap_tensor_to(self: torch.Tensor, device: Optional[Union[int, torch.device]] = None, non_blocking=False,
                       **kwargs) -> torch.Tensor:
        r"""Perform Tensor device conversion. Call the to operator implementation.

        .. note::
            If the ``self`` Tensor already
            has the correct :class:`torch.device`, then ``self`` is returned.
            Otherwise, the returned tensor is a copy of ``self`` with the desired :class:`torch.device`.

        Args:
            device (int, optional): if specified, all parameters will be copied to that device
            non_blocking (bool): If ``True`` and the source is in pinned memory,
                the copy will be asynchronous with respect to the host. Otherwise,
                the argument has no effect.
            **kwargs (dict): For compatibility, may contain the key ``memory_format`` argument.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(wrap_tensor_to, (self,), self, device=device, non_blocking=False, **kwargs)
        device_idx = _normalization_device(custom_backend_name, device)
        return self.to(device=torch.device(f'{custom_backend_name}:{device_idx}'), non_blocking=non_blocking, **kwargs)

    _check_register_once(torch.Tensor, custom_backend_name)
    wrap_tensor_to.__name__ = custom_backend_name
    setattr(torch.Tensor, custom_backend_name, wrap_tensor_to)


def _generate_module_methods_for_privateuse1_backend(custom_backend_name: str) -> None:
    # Generate Module attributes and methods depends on Tensor methods,
    # so we need to check whether Tensor methods is already registered.
    if not hasattr(torch.Tensor, custom_backend_name):
        raise RuntimeError(
            f"Can not automatically generate {custom_backend_name}() method for torch.nn.Module."
            f"Because torch.Tensor doesn't has the method {custom_backend_name}()."
            f"For this error, you can try setting for_tensor=True.")

    def wrap_module_to(self: torch.nn.modules.module.T,
                       device: Optional[Union[int, torch.device]] = None) -> torch.nn.modules.module.T:
        r"""Move all model parameters and buffers to the custom device.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on device while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be copied to that device
        """
        return self._apply(lambda t: getattr(t, custom_backend_name)(device))

    _check_register_once(torch.nn.Module, custom_backend_name)
    setattr(torch.nn.Module, custom_backend_name, wrap_module_to)

def _generate_packed_sequence_methods_for_privateuse1_backend(custom_backend_name: str) -> None:
    # Generate PackedSequence Module attributes and methods depends on Tensor methods,
    # so we need to check whether Tensor methods is already registered.
    if not hasattr(torch.Tensor, f'is_{custom_backend_name}') or \
       not hasattr(torch.Tensor, custom_backend_name):
        raise RuntimeError(
            f"Can not automatically generate is_{custom_backend_name}() or "
            f"{custom_backend_name}() method for torch.nn.utils.rnn.PackedSequence."
            f"Because torch.Tensor doesn't has the method is_{custom_backend_name}()"
            f"or {custom_backend_name}()."
            f"For this error, you can try setting for_tensor=True.")

    @property  # type: ignore[misc]
    def wrap_tensor_backend(self: torch.nn.utils.rnn.PackedSequence) -> bool:
        return self.data.device.type == custom_backend_name

    _check_register_once(torch.nn.utils.rnn.PackedSequence, f'is_{custom_backend_name}')
    setattr(torch.nn.utils.rnn.PackedSequence, f'is_{custom_backend_name}', wrap_tensor_backend)

    def wrap_module_to(self: torch.nn.utils.rnn.PackedSequence,
                       *args, **kwargs) -> torch.nn.utils.rnn.PackedSequence:
        r"""Move all model parameters and buffers to the custom device.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on device while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be copied to that device
        """
        ex = torch.tensor((), dtype=self.data.dtype, device=self.data.device).to(*args, **kwargs)
        if ex.device.type == custom_backend_name:
            return self.to(*args, **kwargs)
        kwargs.update({'device': custom_backend_name})
        return self.to(*args, **kwargs)

    _check_register_once(torch.nn.utils.rnn.PackedSequence, custom_backend_name)
    setattr(torch.nn.utils.rnn.PackedSequence, custom_backend_name, wrap_module_to)

def _generate_storage_methods_for_privateuse1_backend(custom_backend_name: str,
                                                      unsupported_dtype: Optional[List[torch.dtype]] = None) -> None:
    # Attribute is registered in the _StorageBase class
    # and UntypedStorage obtains through inheritance.
    @property  # type: ignore[misc]
    def wrap_storage_backend(self: torch.storage._StorageBase) -> bool:
        r"""Return the internal :class:`torch.UntypedStorage`."""
        return self.device.type == custom_backend_name

    _check_register_once(torch.storage._StorageBase, f'is_{custom_backend_name}')
    setattr(torch.storage._StorageBase, f'is_{custom_backend_name}', wrap_storage_backend)

    def wrap_storage_to(self, device=None, non_blocking=False):
        r"""Return a copy of this object in custom device memory.

        If this object is already in device memory and on the correct device, then
        no copy is performed and the original object is returned.

        Args:
            device (int): The destination device id. Defaults to the current device.
            non_blocking (bool): If ``True`` and the source is in pinned memory,
            the copy will be asynchronous with respect to the host. Otherwise,
            the argument has no effect.
        """
        # There should be a judgment related to storage device and a judgment related to storage type,
        # but it depends on the extended function, so this part is temporarily omitted in the automatic generation.
        device_idx = _normalization_device(custom_backend_name, device)

        if getattr(self, f'is_{custom_backend_name}'):
            # storage has already on expected device.
            if self.get_device() == device_idx:
                return self
        # For sparse storage, custom need to extend the implementation by themselves.
        if self.is_sparse:
            raise RuntimeError(f"Can not support a sparse storage move to {custom_backend_name} backend")
        # create untyped_storage and copy data
        untyped_storage = torch.UntypedStorage(
            self.size(), device=torch.device(f'{custom_backend_name}:{device_idx}')
        )
        untyped_storage.copy_(self, non_blocking)
        return untyped_storage

    _check_register_once(torch.storage._StorageBase, custom_backend_name)
    setattr(torch.storage._StorageBase, custom_backend_name, wrap_storage_to)

    # Register the corresponding attribute for the TypedStorage class.
    # When the TypedStorage class is removed, the registration is also removed.

    @property  # type: ignore[misc]
    def wrap_typed_storage_backend(self: torch.storage.TypedStorage) -> bool:
        torch.storage._warn_typed_storage_removal()
        return self._untyped_storage.device.type == custom_backend_name

    _check_register_once(torch.TypedStorage, f'is_{custom_backend_name}')
    setattr(torch.storage.TypedStorage, f'is_{custom_backend_name}', wrap_typed_storage_backend)

    def wrap_typed_storage_to(self: torch.storage.TypedStorage,
                              device=None, non_blocking=False, **kwargs) -> torch.storage.TypedStorage:
        torch.storage._warn_typed_storage_removal()
        if unsupported_dtype and self.dtype in unsupported_dtype:
            raise RuntimeError(f"Cannot create {custom_backend_name} storage "
                               f"as {self.dtype} dtype is not supported by this backend")
        custom_backend_storage: torch.UntypedStorage = getattr(
            self._untyped_storage, custom_backend_name)(device, non_blocking, **kwargs)
        return self._new_wrapped_storage(custom_backend_storage)

    _check_register_once(torch.TypedStorage, custom_backend_name)
    setattr(torch.TypedStorage, custom_backend_name, wrap_typed_storage_to)


def generate_methods_for_privateuse1_backend(for_tensor: bool = True, for_module: bool = True,
                                             for_packed_sequence: bool = True,
                                             for_storage: bool = False,
                                             unsupported_dtype: Optional[List[torch.dtype]] = None) -> None:
    r"""
    Automatically generate attributes and methods for the custom backend after rename privateuse1 backend.

    In the default scenario, storage-related methods will not be generated automatically.

    When you implement kernels for various torch operations, and register them to the PrivateUse1 dispatch key.
    And call the function torch.rename_privateuse1_backend("foo") to rename your backend name.
    At this point, you can easily register specific methods and attributes by calling this function.
    Just like torch.Tensor.foo(), torch.Tensor.is_foo, torch.Storage.foo(), torch.Storage.is_foo.

    Note: We recommend you use generic functions (check devices are equal or to(device=)).
    We provide these methods for convenience only and they will be "monkey patched" onto the objects
    and so will not be properly typed. For Storage methods generate, if you need to support sparse data storage,
    you need to extend the implementation yourself.

    Args:
        for_tensor (bool): whether register related methods for torch.Tensor class.
        for_module (bool): whether register related methods for torch.nn.Module class.
        for_storage (bool): whether register related methods for torch.Storage class.
        unsupported_dtype (List[torch.dtype]): takes effect only when the storage method needs to be generated,
            indicating that the storage does not support the torch.dtype type.

    Example::

        >>> # xdoctest: +SKIP("failing")
        >>> torch.utils.rename_privateuse1_backend("foo")
        >>> torch.utils.generate_methods_for_privateuse1_backend()
        # Then automatically generate backend-related attributes and methods.
        >>> a = torch.tensor(2).foo()
        >>> a.is_foo
        >>> hasattr(torch.nn.Module, 'foo')
    """
    custom_backend_name = _get_privateuse1_backend_name()

    if for_tensor:
        _generate_tensor_methods_for_privateuse1_backend(custom_backend_name)

    if for_module:
        _generate_module_methods_for_privateuse1_backend(custom_backend_name)

    if for_storage:
        _generate_storage_methods_for_privateuse1_backend(custom_backend_name, unsupported_dtype)

    if for_packed_sequence:
        _generate_packed_sequence_methods_for_privateuse1_backend(custom_backend_name)

def _get_custom_mod_func(func_name: str):
    r"""
    Return the func named `func_name` defined in custom device module. If not defined,
    return `None`. And the func is registered with `torch.utils.rename_privateuse1_backend('foo')`
    and `torch._register_device_module('foo', BackendModule)`.
    If the custom device module or the func is not defined, it will give warning or error message.
    Args:
        func_name (str): return the callable func named func_name defined in custom device module.
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
        foo_is_available_func = torch.utils.backend_registration._get_custom_mod_func("is_available")
        if foo_is_available_func:
            foo_is_available = foo_is_available_func()
        func_ = torch.utils.backend_registration._get_custom_mod_func("func_name")
        if func_:
            result = func_(*args, **kwargs)
    Attention: This function is not meant to be used directly by users, which is why
    it is marked as private. It is a convenience function for backend implementers to
    more easily call the hooks into their backend extensions.
    """
    assert isinstance(func_name, str), f"func_name must be `str`, but got `{type(func_name)}`."
    backend_name = _get_privateuse1_backend_name()
    custom_device_mod = getattr(torch, backend_name, None)  # type: ignore[arg-type]
    function = getattr(custom_device_mod, func_name, None)  # type: ignore[arg-type]
    if custom_device_mod is None or function is None:
        message = f'Try to call torch.{backend_name}.{func_name}. The backend must register a custom backend '
        message += f"module with `torch._register_device_module('{backend_name}', BackendModule)`. And "
        message += f"BackendModule needs to have the following API's:\n `{func_name}(*args, **kwargs)`. \n"
        raise RuntimeError(message)
    return function
