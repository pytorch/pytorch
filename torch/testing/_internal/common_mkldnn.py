import contextlib
import functools
import inspect

import torch


# Test whether hardware BF32 math mode enabled. It is enabled only on:
# - MKLDNN is available
# - BF16 is supported by MKLDNN
def bf32_is_not_fp32():
    if not torch.backends.mkldnn.is_available():
        return False
    if not torch.ops.mkldnn._is_mkldnn_bf16_supported():
        return False
    return True


@contextlib.contextmanager
def bf32_off():
    try:
        with torch.backends.mkldnn.flags(enabled=None, allow_bf32=False):
            yield
    finally:
        pass


@contextlib.contextmanager
def bf32_on(self, bf32_precision=1e-5):
    try:
        with torch.backends.mkldnn.flags(enabled=None, allow_bf32=True):
            yield
    finally:
        pass


# This is a wrapper that wraps a test to run this test twice, one with
# allow_bf32=True, another with allow_bf32=False. When running with
# allow_bf32=True, it will use reduced precision as specified by the
# argument
def bf32_on_and_off(bf32_precision=1e-5):
    def with_bf32_disabled(self, function_call):
        with bf32_off():
            function_call()

    def with_bf32_enabled(self, function_call):
        with bf32_on(self, bf32_precision):
            function_call()

    def wrapper(f):
        params = inspect.signature(f).parameters
        arg_names = tuple(params.keys())

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            for k, v in zip(arg_names, args):
                kwargs[k] = v
            cond = bf32_is_not_fp32()
            if "device" in kwargs:
                cond = cond and (torch.device(kwargs["device"]).type == "cpu")
            if "dtype" in kwargs:
                cond = cond and (kwargs["dtype"] == torch.float)
            if cond:
                with_bf32_disabled(kwargs["self"], lambda: f(**kwargs))
                with_bf32_enabled(kwargs["self"], lambda: f(**kwargs))
            else:
                f(**kwargs)

        return wrapped

    return wrapper


# This is a wrapper that wraps a test to run it with bf32 turned off.
# This wrapper is designed to be used when a test uses matmul or convolutions
# but the purpose of that test is not testing matmul or convolutions.
# Disabling bf32 will enforce torch.float tensors to be always computed
# at full precision.
def with_bf32_off(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        with bf32_off():
            return f(*args, **kwargs)

    return wrapped
