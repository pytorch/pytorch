from typing import List

import torch
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

    For more details, see https://pytorch.org/tutorials/advanced/extend_dispatcher.html#get-a-dispatch-key-for-your-backend
    For an existing example, see https://github.com/bdhirsh/pytorch_open_registration_example

    Example::

        >>> # xdoctest: +SKIP("failing")
        >>> torch.utils.register_privateuse1_backend("foo")
        # This will work, assuming that you've implemented the right C++ kernels
        # to implement torch.ones.
        >>> a = torch.ones(2, device="foo")
    """
    return _rename_privateuse1_backend(backend_name)

supported_amp_dtype_for_privateuse1 = set()

def set_amp_supported_dtype(dtypes) -> None:
    r"""
    set_amp_supported_dtype(dtypes) -> None

    This is a registration API for external backends that would like to register
    amp supported dtypes for their own device. And you should define your own device module to
    supported amp  simultaneously.

    Example::

        >>> # xdoctest: +SKIP("failing")
        >>> torch.utils.register_privateuse1_backend("foo")
        >>> torch.utils.set_amp_supported_dtype(torch.float16)
        """
    global supported_amp_dtype_for_privateuse1

    if isinstance(dtypes, torch.dtype):
        supported_amp_dtype_for_privateuse1.add(dtypes)
    elif isinstance(dtypes, (list, tuple)):
        for dtype in dtypes:
            set_amp_supported_dtype(dtype)
    else:
        raise ValueError("Excepted torch.dtype or list of torch.dtype, but got", dtypes)

def get_amp_supported_dtype() -> List[torch.dtype]:
    r"""
    get_amp_supported_dtype() -> List[torch.dtype]

    This is used to get supported dtypes for custom backend amp.
    """
    return list(supported_amp_dtype_for_privateuse1)
