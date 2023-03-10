from torch._C import _rename_privateuse1_backend

def rename_privateuse1_backend(backend_name: str) -> None:
    r"""
    rename_privateuse1_backend(backend_name) -> None

    This is a registration API for external backends that would like to register their
    own device and C++ kernels out of tree.

    The steps are:
    (1) (In C++) implement kernels for various torch operations, and register them
        to the PrivateUse1 dispatch key.
    (2) (In python) call torch.register_privateuse1_backend("foo")

    You can now use "foo" as an ordinary device string in python.

    Note: this API can only be called once per process. Attempting to change
    the external backend after it's already been set will result in an error.

    Note: and if you want to support AMP on your device, you can register a custom backend module.
    The backend must register a custom backend module with `torch._register_device_module("foo", BackendModule)`.
    BackendModule needs to have the following API's:

    (1) get_amp_supported_dtype() -> List[torch.dtype]
        get the supported dtypes on your `foo` device in AMP, maybe the `foo` device supports one more dtype.

    (2) is_autocast_foo_enabled() -> bool
        check the AMP is enabled or not on your `foo` device.

    (3) get_autocast_foo_dtype() -> torch.dtype
        get the supported dtype on your `foo` device in AMP, which is set by `set_autocast_foo_dtype` or the
        default dtype, and the default dtype is `torch.float16`.

    (4) set_autocast_foo_enabled(bool) -> None
        enable the AMP or not on your `foo` device.

    (5) set_autocast_foo_dtype(dtype) -> None
        set the supported dtype on your `foo` device in AMP, and the dtype be contained in the dtypes got
        from `get_amp_supported_dtype`.

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
