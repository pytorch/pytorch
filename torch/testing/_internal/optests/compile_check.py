import torch
from .make_fx import make_fx_check
from .aot_autograd import aot_autograd_check
from .common import TestFrameworkError
from torch._subclasses.schema_check_mode import SchemaCheckMode
import contextlib


def compile_check(
        func,
        args,
        kwargs,
        *,
        dynamic_only=False,
        inference_only=False,
        fullgraph=True,
        raise_error=True):
    """Check if torch.compile supports a function.

    Args:
        func (function): a Python function that takes at least one Tensor
            as input and returns a Tensor or a Tuple of Tensors.
        args (Tuple): args to the function
        kwargs (dict): kwargs to the function
        dynamic_only (bool, optional): If the function only works with dynamic
            shapes. This can happen if it returns Tensors whose shape are
            dependent on the data on the input Tensors. If True, we skip
            tests related to torch.compile with static shapes.
        inference_only (bool, optional): If the function only works with
            inputs that do not require grad. If True, we will skip
            autograd-related tests.
        fullgraph (bool, optional): If we expect the entire function
            to be captured with torch.compile without any graph breaks.
        raise_error (bool, optional): If True, this API raises errors
            when they are encountered.

    """
    def run_static_or_dynamic_tests(dynamic):
        tracing_mode = 'symbolic' if dynamic else 'fake'
        with ignore_test_framework_error():
            make_fx_check(func, args, kwargs, tracing_mode=tracing_mode)
        if not inference_only:
            with ignore_test_framework_error():
                aot_autograd_check(func, args, kwargs, dynamic=dynamic)
        with ignore_test_framework_error():
            check_compile(func, args, kwargs, fullgraph, dynamic=dynamic)

    if not raise_error:
        # In this case, we should return a nice report walking through
        # all of the tests and what failed. This is not yet implemented.
        raise NotImplementedError("NYI: raise_error=False")

    with ignore_test_framework_error():
        schema_check(func, args, kwargs)

    if not dynamic_only:
        run_static_or_dynamic_tests(dynamic=False)
    run_static_or_dynamic_tests(dynamic=True)


def schema_check(func, args, kwargs):
    with SchemaCheckMode():
        func(*args, **kwargs)


def check_compile(func, args, kwargs, fullgraph, dynamic):
    expected = func(*args, **kwargs)
    result = torch.compile(func, backend='aot_eager', fullgraph=fullgraph, dynamic=dynamic)(*args, **kwargs)
    torch.testing._comparison.assert_close(expected, result)


@contextlib.contextmanager
def ignore_test_framework_error():
    try:
        yield
    except TestFrameworkError:
        return
