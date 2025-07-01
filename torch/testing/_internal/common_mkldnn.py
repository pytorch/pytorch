# mypy: ignore-errors

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
    old_matmul_precision = torch.backends.mkldnn.matmul.fp32_precision
    old_conv_precision = torch.backends.mkldnn.conv.fp32_precision
    try:
        torch.backends.mkldnn.matmul.fp32_precision = "ieee"
        torch.backends.mkldnn.conv.fp32_precision = "ieee"
        yield
    finally:
        torch.backends.mkldnn.matmul.fp32_precision = old_matmul_precision
        torch.backends.mkldnn.conv.fp32_precision = old_conv_precision


@contextlib.contextmanager
def bf32_on(self, bf32_precision=1e-2):
    old_matmul_precision = torch.backends.mkldnn.matmul.fp32_precision
    old_conv_precision = torch.backends.mkldnn.conv.fp32_precision
    old_precision = self.precision
    try:
        torch.backends.mkldnn.matmul.fp32_precision = "bf16"
        torch.backends.mkldnn.conv.fp32_precision = "bf16"
        self.precision = bf32_precision
        yield
    finally:
        torch.backends.mkldnn.matmul.fp32_precision = old_matmul_precision
        torch.backends.mkldnn.conv.fp32_precision = old_conv_precision
        self.precision = old_precision


# This is a wrapper that wraps a test to run this test twice, one with
# allow_bf32=True, another with allow_bf32=False. When running with
# allow_bf32=True, it will use reduced precision as specified by the
# argument
def bf32_on_and_off(bf32_precision=1e-2):
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
            kwargs.update(zip(arg_names, args))
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


def tf32_is_not_fp32():
    if not torch.backends.mkldnn.is_available():
        return False
    if not torch._C._cpu._is_amx_fp16_supported():
        return False
    return True


@contextlib.contextmanager
def tf32_off():
    old_matmul_precision = torch.backends.mkldnn.matmul.fp32_precision
    old_conv_precision = torch.backends.mkldnn.conv.fp32_precision
    try:
        torch.backends.mkldnn.matmul.fp32_precision = "ieee"
        torch.backends.mkldnn.conv.fp32_precision = "ieee"
        yield
    finally:
        torch.backends.mkldnn.matmul.fp32_precision = old_matmul_precision
        torch.backends.mkldnn.conv.fp32_precision = old_conv_precision


@contextlib.contextmanager
def tf32_on(self, tf32_precision=1e-5):
    old_matmul_precision = torch.backends.mkldnn.matmul.fp32_precision
    old_conv_precision = torch.backends.mkldnn.conv.fp32_precision
    old_precision = self.precision
    try:
        torch.backends.mkldnn.matmul.fp32_precision = 'tf32'
        torch.backends.mkldnn.conv.fp32_precision = "tf32"
        self.precision = tf32_precision
        yield
    finally:
        torch.backends.mkldnn.matmul.fp32_precision = old_matmul_precision
        torch.backends.mkldnn.conv.fp32_precision = old_conv_precision
        self.precision = old_precision


# This is a wrapper that wraps a test to run this test twice, one with
# allow_tf32=True, another with allow_tf32=False. When running with
# allow_tf32=True, it will use reduced precision as specified by the
# argument
def tf32_on_and_off(tf32_precision=1e-5):
    def with_tf32_disabled(self, function_call):
        with tf32_off():
            function_call()

    def with_tf32_enabled(self, function_call):
        with tf32_on(self, tf32_precision):
            function_call()

    def wrapper(f):
        params = inspect.signature(f).parameters
        arg_names = tuple(params.keys())

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            kwargs.update(zip(arg_names, args))
            cond = tf32_is_not_fp32()
            if "device" in kwargs:
                cond = cond and (torch.device(kwargs["device"]).type == "cpu")
            if "dtype" in kwargs:
                cond = cond and (kwargs["dtype"] == torch.float)
            if cond:
                with_tf32_disabled(kwargs["self"], lambda: f(**kwargs))
                with_tf32_enabled(kwargs["self"], lambda: f(**kwargs))
            else:
                f(**kwargs)

        return wrapped

    return wrapper
