r"""Importing this file must **not** initialize CUDA context. test_distributed
relies on this assumption to properly run. This means that when this is imported
no CUDA calls shall be made, including torch.cuda.device_count(), etc.

torch.testing._internal.common_cuda.py can freely initialize CUDA context when imported.
"""

import argparse
import contextlib
import copy
import ctypes
import errno
import functools
import gc
import inspect
import io
import json
import logging
import math
import operator
import os
import platform
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import warnings
from collections.abc import Mapping, Sequence
from contextlib import closing, contextmanager
from copy import deepcopy
from enum import Enum
from functools import partial, wraps
from itertools import product, chain
from pathlib import Path
from statistics import mean
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from unittest.mock import MagicMock

import expecttest
import numpy as np

import __main__  # type: ignore[import]
import torch
import torch.backends.cudnn
import torch.backends.mkl
import torch.backends.mps
import torch.backends.xnnpack
import torch.cuda
from torch import Tensor
from torch._C import ScriptDict, ScriptList  # type: ignore[attr-defined]
from torch._utils_internal import get_writable_path
from torch.nn import (
    ModuleDict,
    ModuleList,
    ParameterDict,
    ParameterList,
    Sequential,
)
from torch.onnx import (
    register_custom_op_symbolic,
    unregister_custom_op_symbolic,
)
from torch.testing import make_tensor
from torch.testing._comparison import (
    BooleanPair,
    NonePair,
    NumberPair,
    Pair,
    TensorLikePair,
)
from torch.testing._comparison import not_close_error_metas
from torch.testing._internal.common_dtype import get_all_dtypes
import torch.utils._pytree as pytree

from .composite_compliance import no_dispatch


# Class to keep track of test flags configurable by environment variables.
# Flags set here are intended to be read-only and should not be modified after
# definition.
# TODO: Expand this class to handle abritrary settings in addition to boolean flags?
class TestEnvironment:
    # Set of env vars to set for the repro command that is output on test failure.
    # Specifically, this includes env vars that are set to non-default values and
    # are not implied. Maps from env var name -> value (int)
    repro_env_vars: dict = {}

    # Defines a flag usable throughout the test suite, determining its value by querying
    # the specified environment variable.
    #
    # Args:
    #     name (str): The name of the flag. A global variable with this name will be set
    #         for convenient access throughout the test suite.
    #     env_var (str): The name of the primary environment variable from which to
    #         determine the value of this flag. If this is None or the environment variable
    #         is unset, the default value will be used unless otherwise implied (see
    #         implied_by_fn). Default: None
    #     default (bool): The default value to use for the flag if unset by the environment
    #         variable and unimplied. Default: False
    #     include_in_repro (bool): Indicates whether this flag should be included in the
    #         repro command that is output on test failure (i.e. whether it is possibly
    #         relevant to reproducing the test failure). Default: True
    #     enabled_fn (Callable): Callable returning whether the flag should be enabled
    #         given the environment variable value and the default value. Default: Lambda
    #         requiring "0" to disable if on by default OR "1" to enable if off by default.
    #     implied_by_fn (Callable): Thunk returning a bool to imply this flag as enabled
    #         by something outside of its primary environment variable setting. For example,
    #         this can be useful if the value of another environment variable implies the flag
    #         as enabled. Default: Lambda returning False to indicate no implications.
    @staticmethod
    def def_flag(
        name,
        env_var=None,
        default=False,
        include_in_repro=True,
        enabled_fn=lambda env_var_val, default: (
            (env_var_val != "0") if default else (env_var_val == "1")),
        implied_by_fn=lambda: False,
    ):
        enabled = default
        if env_var is not None:
            env_var_val = os.getenv(env_var)
            enabled = enabled_fn(env_var_val, default)
        implied = implied_by_fn()
        enabled = enabled or implied
        if include_in_repro and (env_var is not None) and (enabled != default) and not implied:
            TestEnvironment.repro_env_vars[env_var] = env_var_val

        # export flag globally for convenience
        assert name not in globals(), f"duplicate definition of flag '{name}'"
        globals()[name] = enabled

    # Returns a string prefix usable to set environment variables for any test
    # settings that should be explicitly set to match this instantiation of the
    # test suite.
    # Example: "PYTORCH_TEST_WITH_ASAN=1 PYTORCH_TEST_WITH_ROCM=1"
    @staticmethod
    def repro_env_var_prefix() -> str:
        return " ".join([f"{env_var}={value}"
                         for env_var, value in TestEnvironment.repro_env_vars.items()])


log = logging.getLogger(__name__)
torch.backends.disable_global_flags()

FILE_SCHEMA = "file://"
if sys.platform == 'win32':
    FILE_SCHEMA = "file:///"

# NB: This flag differs semantically from others in that setting the env var to any
# non-empty value will cause it to be true:
#   CI=1, CI="true", CI=0, etc. all set the flag to be true.
#   CI= and an unset CI set the flag to be false.
# GitHub sets the value to CI="true" to enable it.
TestEnvironment.def_flag("IS_CI", env_var="CI", include_in_repro=False,
                         enabled_fn=lambda env_var_value, _: bool(env_var_value))
TestEnvironment.def_flag(
    "IS_SANDCASTLE",
    env_var="SANDCASTLE",
    implied_by_fn=lambda: os.getenv("TW_JOB_USER") == "sandcastle",
    include_in_repro=False)
TestEnvironment.def_flag("IS_FBCODE", env_var="PYTORCH_TEST_FBCODE", include_in_repro=False)
TestEnvironment.def_flag("IS_REMOTE_GPU", env_var="PYTORCH_TEST_REMOTE_GPU",
                         include_in_repro=False)

TestEnvironment.def_flag("RETRY_TEST_CASES", env_var="PYTORCH_RETRY_TEST_CASES",
                         include_in_repro=False)
TestEnvironment.def_flag("OVERRIDE_FLAKY_SIGNAL", env_var="PYTORCH_OVERRIDE_FLAKY_SIGNAL",
                         include_in_repro=False)
TestEnvironment.def_flag(
    "DISABLE_RUNNING_SCRIPT_CHK",
    env_var="PYTORCH_DISABLE_RUNNING_SCRIPT_CHK",
    include_in_repro=False)
# NB: enabled by default unless in an fbcode context.
TestEnvironment.def_flag("PRINT_REPRO_ON_FAILURE", env_var="PYTORCH_PRINT_REPRO_ON_FAILURE",
                         default=(not IS_FBCODE), include_in_repro=False)

DEFAULT_DISABLED_TESTS_FILE = '.pytorch-disabled-tests.json'
DEFAULT_SLOW_TESTS_FILE = '.pytorch-slow-tests.json'

disabled_tests_dict = {}
slow_tests_dict = {}

def maybe_load_json(filename):
    if os.path.isfile(filename):
        with open(filename) as fp:
            return json.load(fp)
    log.warning("Attempted to load json file '%s' but it does not exist.", filename)
    return {}

# set them here in case the tests are running in a subprocess that doesn't call run_tests
if os.getenv("SLOW_TESTS_FILE", ""):
    slow_tests_dict = maybe_load_json(os.getenv("SLOW_TESTS_FILE", ""))
if os.getenv("DISABLED_TESTS_FILE", ""):
    disabled_tests_dict = maybe_load_json(os.getenv("DISABLED_TESTS_FILE", ""))

NATIVE_DEVICES = ('cpu', 'cuda', 'meta', torch._C._get_privateuse1_backend_name())

check_names = ['orin', 'concord', 'galen', 'xavier', 'nano', 'jetson', 'tegra']
IS_JETSON = any(name in platform.platform() for name in check_names)

def gcIfJetson(fn):
    # Irregular Jetson host/device memory setup requires cleanup to avoid tests being killed
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if IS_JETSON:
            gc.collect()
            torch.cuda.empty_cache()
        fn(*args, **kwargs)
    return wrapper

class _TestParametrizer:
    """
    Decorator class for parametrizing a test function, yielding a set of new tests spawned
    from the original generic test, each specialized for a specific set of test inputs. For
    example, parametrizing a test across the set of ops will result in a test function per op.

    The decision of how to parametrize / what to parametrize over is intended to be implemented
    by each derived class.

    In the details, the decorator adds a 'parametrize_fn' property to the test function. This function
    is intended to be called later by one of:
      * Device-specific test instantiation via instantiate_device_type_tests(). Note that for this
        case there is no need to explicitly parametrize over device type, as that is handled separately.
      * Device-agnostic parametrized test instantiation via instantiate_parametrized_tests().

    If the decorator is applied to a test function that already has a 'parametrize_fn' property, a new
    composite 'parametrize_fn' will be created that generates tests with the product of the parameters
    generated by the old and new parametrize_fns. This allows for convenient composability of decorators.
    """
    def _parametrize_test(self, test, generic_cls, device_cls):
        """
        Parametrizes the given test function across whatever dimension is specified by the derived class.
        Tests can be parametrized over any arbitrary dimension or combination of dimensions, such as all
        ops, all modules, or all ops + their associated dtypes.

        Args:
            test (fn): Test function to parametrize over
            generic_cls (class): Generic test class object containing tests (e.g. TestFoo)
            device_cls (class): Device-specialized test class object (e.g. TestFooCPU); set to None
                if the tests are not part of a device-specific set

        Returns:
            Generator object returning 4-tuples of:
                test (fn): Parametrized test function; must support a device arg and args for any params
                test_name (str): Parametrized suffix for the test (e.g. opname_int64); will be appended to
                    the base name of the test
                param_kwargs (dict): Param kwargs to pass to the test (e.g. {'op': 'add', 'dtype': torch.int64})
                decorator_fn (callable): Callable[[Dict], List] for list of decorators to apply given param_kwargs
        """
        raise NotImplementedError

    def __call__(self, fn):
        if hasattr(fn, 'parametrize_fn'):
            # Do composition with the product of args.
            old_parametrize_fn = fn.parametrize_fn
            new_parametrize_fn = self._parametrize_test
            fn.parametrize_fn = compose_parametrize_fns(old_parametrize_fn, new_parametrize_fn)
        else:
            fn.parametrize_fn = self._parametrize_test
        return fn


def compose_parametrize_fns(old_parametrize_fn, new_parametrize_fn):
    """
    Returns a parametrize_fn that parametrizes over the product of the parameters handled
    by the given parametrize_fns. Each given parametrize_fn should each have the signature
    f(test, generic_cls, device_cls).

    The test names will be a combination of the names produced by the parametrize_fns in
    "<new_name>_<old_name>" order. This order is done to match intuition for constructed names
    when composing multiple decorators; the names will be built in top to bottom order when stacking
    parametrization decorators.

    Args:
        old_parametrize_fn (callable) - First parametrize_fn to compose.
        new_parametrize_fn (callable) - Second parametrize_fn to compose.
    """

    def composite_fn(test, generic_cls, device_cls,
                     old_parametrize_fn=old_parametrize_fn,
                     new_parametrize_fn=new_parametrize_fn):
        old_tests = list(old_parametrize_fn(test, generic_cls, device_cls))
        for (old_test, old_test_name, old_param_kwargs, old_dec_fn) in old_tests:
            for (new_test, new_test_name, new_param_kwargs, new_dec_fn) in \
                    new_parametrize_fn(old_test, generic_cls, device_cls):
                redundant_params = set(old_param_kwargs.keys()).intersection(new_param_kwargs.keys())
                if redundant_params:
                    raise RuntimeError('Parametrization over the same parameter by multiple parametrization '
                                       'decorators is not supported. For test "{}", the following parameters '
                                       'are handled multiple times: {}'.format(
                                           test.__name__, redundant_params))
                full_param_kwargs = {**old_param_kwargs, **new_param_kwargs}
                merged_test_name = '{}{}{}'.format(new_test_name,
                                                   '_' if old_test_name != '' and new_test_name != '' else '',
                                                   old_test_name)

                def merged_decorator_fn(param_kwargs, old_dec_fn=old_dec_fn, new_dec_fn=new_dec_fn):
                    return list(old_dec_fn(param_kwargs)) + list(new_dec_fn(param_kwargs))

                yield (new_test, merged_test_name, full_param_kwargs, merged_decorator_fn)

    return composite_fn


def instantiate_parametrized_tests(generic_cls):
    """
    Instantiates tests that have been decorated with a parametrize_fn. This is generally performed by a
    decorator subclass of _TestParametrizer. The generic test will be replaced on the test class by
    parametrized tests with specialized names. This should be used instead of
    instantiate_device_type_tests() if the test class contains device-agnostic tests.

    You can also use it as a class decorator. E.g.

    ```
    @instantiate_parametrized_tests
    class TestFoo(TestCase):
        ...
    ```

    Args:
        generic_cls (class): Generic test class object containing tests (e.g. TestFoo)
    """
    for attr_name in tuple(dir(generic_cls)):
        class_attr = getattr(generic_cls, attr_name)
        if not hasattr(class_attr, 'parametrize_fn'):
            continue

        # Remove the generic test from the test class.
        delattr(generic_cls, attr_name)

        # Add parametrized tests to the test class.
        def instantiate_test_helper(cls, name, test, param_kwargs):
            @wraps(test)
            def instantiated_test(self, param_kwargs=param_kwargs):
                test(self, **param_kwargs)

            assert not hasattr(generic_cls, name), f"Redefinition of test {name}"
            setattr(generic_cls, name, instantiated_test)

        for (test, test_suffix, param_kwargs, decorator_fn) in class_attr.parametrize_fn(
                class_attr, generic_cls=generic_cls, device_cls=None):
            full_name = f'{test.__name__}_{test_suffix}'

            # Apply decorators based on full param kwargs.
            for decorator in decorator_fn(param_kwargs):
                test = decorator(test)

            instantiate_test_helper(cls=generic_cls, name=full_name, test=test, param_kwargs=param_kwargs)
    return generic_cls


class subtest:
    """
    Explicit subtest case for use with test parametrization.
    Allows for explicit naming of individual subtest cases as well as applying
    decorators to the parametrized test.

    Args:
        arg_values (iterable): Iterable of arg values (e.g. range(10)) or
            tuples of arg values (e.g. [(1, 2), (3, 4)]).
        name (str): Optional name to use for the test.
        decorators (iterable): Iterable of decorators to apply to the generated test.
    """
    __slots__ = ['arg_values', 'name', 'decorators']

    def __init__(self, arg_values, name=None, decorators=None):
        self.arg_values = arg_values
        self.name = name
        self.decorators = decorators if decorators else []


class parametrize(_TestParametrizer):
    """
    Decorator for applying generic test parametrizations.

    The interface for this decorator is modeled after `@pytest.mark.parametrize`.
    Basic usage between this decorator and pytest's is identical. The first argument
    should be a string containing comma-separated names of parameters for the test, and
    the second argument should be an iterable returning values or tuples of values for
    the case of multiple parameters.

    Beyond this basic usage, the decorator provides some additional functionality that
    pytest does not.

    1. Parametrized tests end up as generated test functions on unittest test classes.
    Since this differs from how pytest works, this decorator takes on the additional
    responsibility of naming these test functions. The default test names consists of
    the test's base name followed by each parameter name + value (e.g. "test_bar_x_1_y_foo"),
    but custom names can be defined using `name_fn` or the `subtest` structure (see below).

    2. The decorator specially handles parameter values of type `subtest`, which allows for
    more fine-grained control over both test naming and test execution. In particular, it can
    be used to tag subtests with explicit test names or apply arbitrary decorators (see examples
    below).

    Examples::

        @parametrize("x", range(5))
        def test_foo(self, x):
            ...

        @parametrize("x,y", [(1, 'foo'), (2, 'bar'), (3, 'baz')])
        def test_bar(self, x, y):
            ...

        @parametrize("x,y", [(1, 'foo'), (2, 'bar'), (3, 'baz')],
                     name_fn=lambda x, y: '{}_{}'.format(x, y))
        def test_bar_custom_names(self, x, y):
            ...

        @parametrize("x, y", [subtest((1, 2), name='double'),
                              subtest((1, 3), name='triple', decorators=[unittest.expectedFailure]),
                              subtest((1, 4), name='quadruple')])
        def test_baz(self, x, y):
            ...

    To actually instantiate the parametrized tests, one of instantiate_parametrized_tests() or
    instantiate_device_type_tests() should be called. The former is intended for test classes
    that contain device-agnostic tests, while the latter should be used for test classes that
    contain device-specific tests. Both support arbitrary parametrizations using the decorator.

    Args:
        arg_str (str): String of arg names separate by commas (e.g. "x,y").
        arg_values (iterable): Iterable of arg values (e.g. range(10)) or
            tuples of arg values (e.g. [(1, 2), (3, 4)]).
        name_fn (Callable): Optional function that takes in parameters and returns subtest name.
    """
    def __init__(self, arg_str, arg_values, name_fn=None):
        self.arg_names: List[str] = [s.strip() for s in arg_str.split(',') if s != '']
        self.arg_values = arg_values
        self.name_fn = name_fn

    def _formatted_str_repr(self, name, value):
        """ Returns a string representation for the given arg that is suitable for use in test function names. """
        if isinstance(value, torch.dtype):
            return dtype_name(value)
        elif isinstance(value, torch.device):
            return str(value)
        # Can't use isinstance as it would cause a circular import
        elif value.__class__.__name__ == 'OpInfo' or value.__class__.__name__ == 'ModuleInfo':
            return value.formatted_name
        else:
            # Include name and value separated by underscore.
            return f"{name}_{str(value).replace('.', '_')}"

    def _default_subtest_name(self, values):
        return '_'.join([self._formatted_str_repr(a, v) for a, v in zip(self.arg_names, values)])

    def _get_subtest_name(self, values, explicit_name=None):
        if explicit_name:
            subtest_name = explicit_name
        elif self.name_fn:
            subtest_name = self.name_fn(*values)
        else:
            subtest_name = self._default_subtest_name(values)
        return subtest_name

    def _parametrize_test(self, test, generic_cls, device_cls):
        if len(self.arg_names) == 0:
            # No additional parameters needed for the test.
            test_name = ''
            yield (test, test_name, {}, lambda _: [])
        else:
            # Each "values" item is expected to be either:
            # * A tuple of values with one for each arg. For a single arg, a single item is expected.
            # * A subtest instance with arg_values matching the previous.
            values = check_exhausted_iterator = object()
            for values in self.arg_values:
                maybe_name = None

                decorators = []
                if isinstance(values, subtest):
                    sub = values
                    values = sub.arg_values
                    maybe_name = sub.name

                    @wraps(test)
                    def test_wrapper(*args, **kwargs):
                        return test(*args, **kwargs)

                    decorators = sub.decorators
                    gen_test = test_wrapper
                else:
                    gen_test = test

                values = list(values) if len(self.arg_names) > 1 else [values]
                if len(values) != len(self.arg_names):
                    raise RuntimeError(f'Expected # values == # arg names, but got: {len(values)} '
                                       f'values and {len(self.arg_names)} names for test "{test.__name__}"')

                param_kwargs = dict(zip(self.arg_names, values))

                test_name = self._get_subtest_name(values, explicit_name=maybe_name)

                def decorator_fn(_, decorators=decorators):
                    return decorators

                yield (gen_test, test_name, param_kwargs, decorator_fn)

            if values is check_exhausted_iterator:
                raise ValueError('An empty arg_values was passed to @parametrize. '
                                 'Note that this may result from reuse of a generator.')


class ProfilingMode(Enum):
    LEGACY = 1
    SIMPLE = 2
    PROFILING = 3

def cppProfilingFlagsToProfilingMode():
    old_prof_exec_state = torch._C._jit_set_profiling_executor(True)
    old_prof_mode_state = torch._C._get_graph_executor_optimize(True)
    torch._C._jit_set_profiling_executor(old_prof_exec_state)
    torch._C._get_graph_executor_optimize(old_prof_mode_state)

    if old_prof_exec_state:
        if old_prof_mode_state:
            return ProfilingMode.PROFILING
        else:
            return ProfilingMode.SIMPLE
    else:
        return ProfilingMode.LEGACY

@contextmanager
def enable_profiling_mode_for_profiling_tests():
    if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
        old_prof_exec_state = torch._C._jit_set_profiling_executor(True)
        old_prof_mode_state = torch._C._get_graph_executor_optimize(True)
    try:
        yield
    finally:
        if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
            torch._C._jit_set_profiling_executor(old_prof_exec_state)
            torch._C._get_graph_executor_optimize(old_prof_mode_state)

@contextmanager
def enable_profiling_mode():
    old_prof_exec_state = torch._C._jit_set_profiling_executor(True)
    old_prof_mode_state = torch._C._get_graph_executor_optimize(True)
    try:
        yield
    finally:
        torch._C._jit_set_profiling_executor(old_prof_exec_state)
        torch._C._get_graph_executor_optimize(old_prof_mode_state)

@contextmanager
def num_profiled_runs(num_runs):
    old_num_runs = torch._C._jit_set_num_profiled_runs(num_runs)
    try:
        yield
    finally:
        torch._C._jit_set_num_profiled_runs(old_num_runs)

func_call = torch._C.ScriptFunction.__call__
meth_call = torch._C.ScriptMethod.__call__

def prof_callable(callable, *args, **kwargs):
    if 'profile_and_replay' in kwargs:
        del kwargs['profile_and_replay']
        if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
            with enable_profiling_mode_for_profiling_tests():
                callable(*args, **kwargs)
                return callable(*args, **kwargs)

    return callable(*args, **kwargs)

def prof_func_call(*args, **kwargs):
    return prof_callable(func_call, *args, **kwargs)

def prof_meth_call(*args, **kwargs):
    return prof_callable(meth_call, *args, **kwargs)

# TODO fix when https://github.com/python/mypy/issues/2427 is address
torch._C.ScriptFunction.__call__ = prof_func_call  # type: ignore[assignment]
torch._C.ScriptMethod.__call__ = prof_meth_call  # type: ignore[assignment]

def _get_test_report_path():
    # allow users to override the test file location. We need this
    # because the distributed tests run the same test file multiple
    # times with different configurations.
    override = os.environ.get('TEST_REPORT_SOURCE_OVERRIDE')
    test_source = override if override is not None else 'python-unittest'
    return os.path.join('test-reports', test_source)

is_running_via_run_test = "run_test.py" in getattr(__main__, "__file__", "")
parser = argparse.ArgumentParser(add_help=not is_running_via_run_test, allow_abbrev=False)
parser.add_argument('--subprocess', action='store_true',
                    help='whether to run each test in a subprocess')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--accept', action='store_true')
parser.add_argument('--jit-executor', '--jit_executor', type=str)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--test-bailouts', '--test_bailouts', action='store_true')
parser.add_argument('--use-pytest', action='store_true')
parser.add_argument('--save-xml', nargs='?', type=str,
                    const=_get_test_report_path(),
                    default=_get_test_report_path() if IS_CI else None)
parser.add_argument('--discover-tests', action='store_true')
parser.add_argument('--log-suffix', type=str, default="")
parser.add_argument('--run-parallel', type=int, default=1)
parser.add_argument('--import-slow-tests', type=str, nargs='?', const=DEFAULT_SLOW_TESTS_FILE)
parser.add_argument('--import-disabled-tests', type=str, nargs='?', const=DEFAULT_DISABLED_TESTS_FILE)
parser.add_argument('--rerun-disabled-tests', action='store_true')
parser.add_argument('--pytest-single-test', type=str, nargs=1)

# Only run when -h or --help flag is active to display both unittest and parser help messages.
def run_unittest_help(argv):
    unittest.main(argv=argv)

if '-h' in sys.argv or '--help' in sys.argv:
    help_thread = threading.Thread(target=run_unittest_help, args=(sys.argv,))
    help_thread.start()
    help_thread.join()

args, remaining = parser.parse_known_args()
if args.jit_executor == 'legacy':
    GRAPH_EXECUTOR = ProfilingMode.LEGACY
elif args.jit_executor == 'profiling':
    GRAPH_EXECUTOR = ProfilingMode.PROFILING
elif args.jit_executor == 'simple':
    GRAPH_EXECUTOR = ProfilingMode.SIMPLE
else:
    # infer flags based on the default settings
    GRAPH_EXECUTOR = cppProfilingFlagsToProfilingMode()

RERUN_DISABLED_TESTS = args.rerun_disabled_tests
# Rerun disabled tests many more times to make sure that they are not flaky anymore
MAX_NUM_RETRIES = 3 if not RERUN_DISABLED_TESTS else 50

SLOW_TESTS_FILE = args.import_slow_tests
DISABLED_TESTS_FILE = args.import_disabled_tests
LOG_SUFFIX = args.log_suffix
RUN_PARALLEL = args.run_parallel
TEST_BAILOUTS = args.test_bailouts
USE_PYTEST = args.use_pytest
PYTEST_SINGLE_TEST = args.pytest_single_test
TEST_DISCOVER = args.discover_tests
TEST_IN_SUBPROCESS = args.subprocess
TEST_SAVE_XML = args.save_xml
REPEAT_COUNT = args.repeat
SEED = args.seed
if not expecttest.ACCEPT:
    expecttest.ACCEPT = args.accept
UNITTEST_ARGS = [sys.argv[0]] + remaining
torch.manual_seed(SEED)

# CI Prefix path used only on CI environment
CI_TEST_PREFIX = str(Path(os.getcwd()))
CI_PT_ROOT = str(Path(os.getcwd()).parent)
CI_FUNCTORCH_ROOT = str(os.path.join(Path(os.getcwd()).parent, "functorch"))

def wait_for_process(p, timeout=None):
    try:
        return p.wait(timeout=timeout)
    except KeyboardInterrupt:
        # Give `p` a chance to handle KeyboardInterrupt. Without this,
        # `pytest` can't print errors it collected so far upon KeyboardInterrupt.
        exit_status = p.wait(timeout=5)
        if exit_status is not None:
            return exit_status
        else:
            p.kill()
            raise
    except subprocess.TimeoutExpired:
        # send SIGINT to give pytest a chance to make xml
        p.send_signal(signal.SIGINT)
        exit_status = None
        try:
            exit_status = p.wait(timeout=5)
        # try to handle the case where p.wait(timeout=5) times out as well as
        # otherwise the wait() call in the finally block can potentially hang
        except subprocess.TimeoutExpired:
            pass
        if exit_status is not None:
            return exit_status
        else:
            p.kill()
        raise
    except:  # noqa: B001,E722, copied from python core library
        p.kill()
        raise
    finally:
        # Always call p.wait() to ensure exit
        p.wait()

def shell(command, cwd=None, env=None, stdout=None, stderr=None, timeout=None):
    sys.stdout.flush()
    sys.stderr.flush()
    # The following cool snippet is copied from Py3 core library subprocess.call
    # only the with
    #   1. `except KeyboardInterrupt` block added for SIGINT handling.
    #   2. In Py2, subprocess.Popen doesn't return a context manager, so we do
    #      `p.wait()` in a `final` block for the code to be portable.
    #
    # https://github.com/python/cpython/blob/71b6c1af727fbe13525fb734568057d78cea33f3/Lib/subprocess.py#L309-L323
    assert not isinstance(command, str), "Command to shell should be a list or tuple of tokens"
    p = subprocess.Popen(command, universal_newlines=True, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
    return wait_for_process(p, timeout=timeout)


def retry_shell(command, cwd=None, env=None, stdout=None, stderr=None, timeout=None, retries=1):
    assert retries >= 0, f"Expecting non negative number for number of retries, got {retries}"
    try:
        exit_code = shell(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, timeout=timeout)
        if exit_code == 0 or retries == 0:
            return exit_code
        print(f"Got exit code {exit_code}, retrying (retries left={retries})", file=stdout, flush=True)
    except subprocess.TimeoutExpired:
        if retries == 0:
            print(f"Command took >{timeout // 60}min, returning 124", file=stdout, flush=True)
            return 124
        print(f"Command took >{timeout // 60}min, retrying (retries left={retries})", file=stdout, flush=True)
    return retry_shell(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, timeout=timeout, retries=retries - 1)


def discover_test_cases_recursively(suite_or_case):
    if isinstance(suite_or_case, unittest.TestCase):
        return [suite_or_case]
    rc = []
    for element in suite_or_case:
        print(element)
        rc.extend(discover_test_cases_recursively(element))
    return rc

def get_test_names(test_cases):
    return ['.'.join(case.id().split('.')[-2:]) for case in test_cases]

def _print_test_names():
    suite = unittest.TestLoader().loadTestsFromModule(__main__)
    test_cases = discover_test_cases_recursively(suite)
    for name in get_test_names(test_cases):
        print(name)

def chunk_list(lst, nchunks):
    return [lst[i::nchunks] for i in range(nchunks)]

# sanitize filename e.g., distributed/pipeline/sync/skip/test_api.py -> distributed.pipeline.sync.skip.test_api
def sanitize_test_filename(filename):
    # inspect.getfile returns absolute path in some CI jobs, converting it to relative path if needed
    if filename.startswith(CI_TEST_PREFIX):
        filename = filename[len(CI_TEST_PREFIX) + 1:]
    strip_py = re.sub(r'.py$', '', filename)
    return re.sub('/', r'.', strip_py)

def lint_test_case_extension(suite):
    succeed = True
    for test_case_or_suite in suite:
        test_case = test_case_or_suite
        if isinstance(test_case_or_suite, unittest.TestSuite):
            first_test = test_case_or_suite._tests[0] if len(test_case_or_suite._tests) > 0 else None
            if first_test is not None and isinstance(first_test, unittest.TestSuite):
                return succeed and lint_test_case_extension(test_case_or_suite)
            test_case = first_test

        if test_case is not None:
            test_class = test_case.id().split('.', 1)[1].split('.')[0]
            if not isinstance(test_case, TestCase):
                err = "This test class should extend from torch.testing._internal.common_utils.TestCase but it doesn't."
                print(f"{test_class} - failed. {err}")
                succeed = False
    return succeed


def get_report_path(argv=UNITTEST_ARGS, pytest=False):
    test_filename = sanitize_test_filename(argv[0])
    test_report_path = TEST_SAVE_XML + LOG_SUFFIX
    test_report_path = os.path.join(test_report_path, test_filename)
    if pytest:
        test_report_path = test_report_path.replace('python-unittest', 'python-pytest')
        os.makedirs(test_report_path, exist_ok=True)
        test_report_path = os.path.join(test_report_path, f"{test_filename}-{os.urandom(8).hex()}.xml")
        return test_report_path
    os.makedirs(test_report_path, exist_ok=True)
    return test_report_path


def sanitize_pytest_xml(xml_file: str):
    # pytext xml is different from unittext xml, this function makes pytest xml more similar to unittest xml
    # consider somehow modifying the XML logger in conftest to do this instead
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_file)
    for testcase in tree.iter('testcase'):
        full_classname = testcase.attrib['classname']
        # The test prefix is optional
        regex_result = re.search(r"^(test\.)?(?P<file>.*)\.(?P<classname>[^\.]*)$", full_classname)
        if regex_result is None:
            continue
        classname = regex_result.group("classname")
        file = regex_result.group("file").replace(".", "/")
        testcase.set("classname", classname)
        testcase.set("file", f"{file}.py")
    tree.write(xml_file)


def get_pytest_test_cases(argv: List[str]) -> List[str]:
    class TestCollectorPlugin:
        def __init__(self):
            self.tests = []

        def pytest_collection_finish(self, session):
            for item in session.items:
                self.tests.append(session.config.cwd_relative_nodeid(item.nodeid))

    test_collector_plugin = TestCollectorPlugin()
    import pytest
    pytest.main(
        [arg for arg in argv if arg != '-vv'] + ['--collect-only', '-qq', '--use-main-module'],
        plugins=[test_collector_plugin]
    )
    return test_collector_plugin.tests


def run_tests(argv=UNITTEST_ARGS):
    # import test files.
    if SLOW_TESTS_FILE:
        if os.path.exists(SLOW_TESTS_FILE):
            with open(SLOW_TESTS_FILE) as fp:
                global slow_tests_dict
                slow_tests_dict = json.load(fp)
                # use env vars so pytest-xdist subprocesses can still access them
                os.environ['SLOW_TESTS_FILE'] = SLOW_TESTS_FILE
        else:
            warnings.warn(f'slow test file provided but not found: {SLOW_TESTS_FILE}')
    if DISABLED_TESTS_FILE:
        if os.path.exists(DISABLED_TESTS_FILE):
            with open(DISABLED_TESTS_FILE) as fp:
                global disabled_tests_dict
                disabled_tests_dict = json.load(fp)
                os.environ['DISABLED_TESTS_FILE'] = DISABLED_TESTS_FILE
        else:
            warnings.warn(f'disabled test file provided but not found: {DISABLED_TESTS_FILE}')
    # Determine the test launch mechanism
    if TEST_DISCOVER:
        _print_test_names()
        return

    # Before running the tests, lint to check that every test class extends from TestCase
    suite = unittest.TestLoader().loadTestsFromModule(__main__)
    if not lint_test_case_extension(suite):
        sys.exit(1)

    if TEST_IN_SUBPROCESS:
        other_args = []
        if DISABLED_TESTS_FILE:
            other_args.append("--import-disabled-tests")
        if SLOW_TESTS_FILE:
            other_args.append("--import-slow-tests")
        if USE_PYTEST:
            other_args.append("--use-pytest")
        if RERUN_DISABLED_TESTS:
            other_args.append("--rerun-disabled-tests")

        test_cases = (
            get_pytest_test_cases(argv) if USE_PYTEST else
            [case.id().split('.', 1)[1] for case in discover_test_cases_recursively(suite)]
        )

        failed_tests = []

        for test_case_full_name in test_cases:

            cmd = (
                [sys.executable] + [argv[0]] + other_args + argv[1:] +
                (["--pytest-single-test"] if USE_PYTEST else []) +
                [test_case_full_name]
            )
            string_cmd = " ".join(cmd)

            timeout = None if RERUN_DISABLED_TESTS else 15 * 60

            exitcode = retry_shell(cmd, timeout=timeout, retries=0 if RERUN_DISABLED_TESTS else 1)

            if exitcode != 0:
                # This is sort of hacky, but add on relevant env variables for distributed tests.
                if 'TestDistBackendWithSpawn' in test_case_full_name:
                    backend = os.environ.get("BACKEND", "")
                    world_size = os.environ.get("WORLD_SIZE", "")
                    env_prefix = f"BACKEND={backend} WORLD_SIZE={world_size}"
                    string_cmd = env_prefix + " " + string_cmd
                # Log the command to reproduce the failure.
                print(f"Test exited with non-zero exitcode {exitcode}. Command to reproduce: {string_cmd}")
                failed_tests.append(test_case_full_name)

            assert len(failed_tests) == 0, "{} unit test(s) failed:\n\t{}".format(
                len(failed_tests), '\n\t'.join(failed_tests))

    elif RUN_PARALLEL > 1:
        test_cases = discover_test_cases_recursively(suite)
        test_batches = chunk_list(get_test_names(test_cases), RUN_PARALLEL)
        processes = []
        for i in range(RUN_PARALLEL):
            command = [sys.executable] + argv + [f'--log-suffix=-shard-{i + 1}'] + test_batches[i]
            processes.append(subprocess.Popen(command, universal_newlines=True))
        failed = False
        for p in processes:
            failed |= wait_for_process(p) != 0
        assert not failed, "Some test shards have failed"
    elif USE_PYTEST:
        pytest_args = argv + ["--use-main-module"]
        if TEST_SAVE_XML:
            test_report_path = get_report_path(pytest=True)
            print(f'Test results will be stored in {test_report_path}')
            pytest_args.append(f'--junit-xml-reruns={test_report_path}')
        if PYTEST_SINGLE_TEST:
            pytest_args = PYTEST_SINGLE_TEST + pytest_args[1:]

        import pytest
        os.environ["NO_COLOR"] = "1"
        exit_code = pytest.main(args=pytest_args)
        if TEST_SAVE_XML:
            sanitize_pytest_xml(test_report_path)

        if not RERUN_DISABLED_TESTS:
            # exitcode of 5 means no tests were found, which happens since some test configs don't
            # run tests from certain files
            exit(0 if exit_code == 5 else exit_code)
        else:
            # Only record the test report and always return a success code when running under rerun
            # disabled tests mode
            exit(0)
    elif TEST_SAVE_XML is not None:
        # import here so that non-CI doesn't need xmlrunner installed
        import xmlrunner  # type: ignore[import]
        from xmlrunner.result import _XMLTestResult  # type: ignore[import]

        class XMLTestResultVerbose(_XMLTestResult):
            """
            Adding verbosity to test outputs:
            by default test summary prints 'skip',
            but we want to also print the skip reason.
            GH issue: https://github.com/pytorch/pytorch/issues/69014

            This works with unittest_xml_reporting<=3.2.0,>=2.0.0
            (3.2.0 is latest at the moment)
            """
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def addSkip(self, test, reason):
                super().addSkip(test, reason)
                for c in self.callback.__closure__:
                    if isinstance(c.cell_contents, str) and c.cell_contents == 'skip':
                        # this message is printed in test summary;
                        # it stands for `verbose_str` captured in the closure
                        c.cell_contents = f"skip: {reason}"

            def printErrors(self) -> None:
                super().printErrors()
                self.printErrorList("XPASS", self.unexpectedSuccesses)
        test_report_path = get_report_path()
        verbose = '--verbose' in argv or '-v' in argv
        if verbose:
            print(f'Test results will be stored in {test_report_path}')
        unittest.main(argv=argv, testRunner=xmlrunner.XMLTestRunner(
            output=test_report_path,
            verbosity=2 if verbose else 1,
            resultclass=XMLTestResultVerbose))
    elif REPEAT_COUNT > 1:
        for _ in range(REPEAT_COUNT):
            if not unittest.main(exit=False, argv=argv).result.wasSuccessful():
                sys.exit(-1)
    else:
        unittest.main(argv=argv)

IS_LINUX = sys.platform == "linux"
IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_PPC = platform.machine() == "ppc64le"
IS_X86 = platform.machine() in ('x86_64', 'i386')
IS_ARM64 = platform.machine() == 'arm64'

def is_avx512_vnni_supported():
    if sys.platform != 'linux':
        return False
    with open("/proc/cpuinfo", encoding="ascii") as f:
        lines = f.read()
    return "vnni" in lines

IS_AVX512_VNNI_SUPPORTED = is_avx512_vnni_supported()

if IS_WINDOWS:
    @contextmanager
    def TemporaryFileName(*args, **kwargs):
        # Ideally we would like to not have to manually delete the file, but NamedTemporaryFile
        # opens the file, and it cannot be opened multiple times in Windows. To support Windows,
        # close the file after creation and try to remove it manually
        if 'delete' in kwargs:
            if kwargs['delete'] is not False:
                raise UserWarning("only TemporaryFileName with delete=False is supported on Windows.")
        else:
            kwargs['delete'] = False
        f = tempfile.NamedTemporaryFile(*args, **kwargs)
        try:
            f.close()
            yield f.name
        finally:
            os.unlink(f.name)
else:
    @contextmanager  # noqa: T484
    def TemporaryFileName(*args, **kwargs):
        with tempfile.NamedTemporaryFile(*args, **kwargs) as f:
            yield f.name

if IS_WINDOWS:
    @contextmanager
    def TemporaryDirectoryName(suffix=None):
        # On Windows the directory created by TemporaryDirectory is likely to be removed prematurely,
        # so we first create the directory using mkdtemp and then remove it manually
        try:
            dir_name = tempfile.mkdtemp(suffix=suffix)
            yield dir_name
        finally:
            shutil.rmtree(dir_name)
else:
    @contextmanager  # noqa: T484
    def TemporaryDirectoryName(suffix=None):
        with tempfile.TemporaryDirectory(suffix=suffix) as d:
            yield d

IS_FILESYSTEM_UTF8_ENCODING = sys.getfilesystemencoding() == 'utf-8'

def _check_module_exists(name: str) -> bool:
    r"""Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec(name)
        return spec is not None
    except ImportError:
        return False

TEST_NUMPY = _check_module_exists('numpy')
TEST_FAIRSEQ = _check_module_exists('fairseq')
TEST_SCIPY = _check_module_exists('scipy')
TEST_MKL = torch.backends.mkl.is_available()
TEST_MPS = torch.backends.mps.is_available()
TEST_CUDA = torch.cuda.is_available()
custom_device_mod = getattr(torch, torch._C._get_privateuse1_backend_name(), None)
TEST_PRIVATEUSE1 = True if (hasattr(custom_device_mod, "is_available") and custom_device_mod.is_available()) else False
TEST_NUMBA = _check_module_exists('numba')

TEST_DILL = _check_module_exists('dill')

TEST_LIBROSA = _check_module_exists('librosa') and not IS_ARM64

TEST_OPT_EINSUM = _check_module_exists('opt_einsum')

TEST_Z3 = _check_module_exists('z3')

BUILD_WITH_CAFFE2 = torch.onnx._CAFFE2_ATEN_FALLBACK

# Python 2.7 doesn't have spawn
TestEnvironment.def_flag("NO_MULTIPROCESSING_SPAWN", env_var="NO_MULTIPROCESSING_SPAWN")
TestEnvironment.def_flag("TEST_WITH_ASAN", env_var="PYTORCH_TEST_WITH_ASAN")
TestEnvironment.def_flag("TEST_WITH_DEV_DBG_ASAN", env_var="PYTORCH_TEST_WITH_DEV_DBG_ASAN")
TestEnvironment.def_flag("TEST_WITH_TSAN", env_var="PYTORCH_TEST_WITH_TSAN")
TestEnvironment.def_flag("TEST_WITH_UBSAN", env_var="PYTORCH_TEST_WITH_UBSAN")
TestEnvironment.def_flag("TEST_WITH_ROCM", env_var="PYTORCH_TEST_WITH_ROCM")

# Enables tests that are slow to run (disabled by default)
TestEnvironment.def_flag("TEST_WITH_SLOW", env_var="PYTORCH_TEST_WITH_SLOW")

# Disables non-slow tests (these tests enabled by default)
# This is usually used in conjunction with TEST_WITH_SLOW to
# run *only* slow tests.  (I could have done an enum, but
# it felt a little awkward.
TestEnvironment.def_flag("TEST_SKIP_FAST", env_var="PYTORCH_TEST_SKIP_FAST")

# Enables crossref tests, in addition to standard tests which
# are being run.  crossref tests work by installing a torch
# function mode that runs extra compute alongside the regular
# computation that happens with the test.  After both computations
# are done, we cross-reference them (thus the name) to check for
# correction, before throwing out the extra compute and proceeding
# as we had before.  By default, we don't run these tests.
TestEnvironment.def_flag("TEST_WITH_CROSSREF", env_var="PYTORCH_TEST_WITH_CROSSREF")

TestEnvironment.def_flag("TEST_SKIP_CUDAGRAPH", env_var="PYTORCH_TEST_SKIP_CUDAGRAPH")
TEST_CUDA_GRAPH = TEST_CUDA and (not TEST_SKIP_CUDAGRAPH) and (
    (torch.version.cuda and int(torch.version.cuda.split(".")[0]) >= 11) or
    (torch.version.hip and float(".".join(torch.version.hip.split(".")[0:2])) >= 5.3)
)

if TEST_CUDA and 'NUM_PARALLEL_PROCS' in os.environ:
    num_procs = int(os.getenv("NUM_PARALLEL_PROCS", "2"))
    # other libraries take up about 11% of space per process
    torch.cuda.set_per_process_memory_fraction(round(1 / num_procs - .11, 2))


def skipIfCrossRef(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if TEST_WITH_CROSSREF:
            raise unittest.SkipTest("test doesn't currently with crossref")
        else:
            fn(*args, **kwargs)
    return wrapper

class CrossRefMode(torch.overrides.TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        r = func(*args, **kwargs)
        return r

# Run PyTorch tests with TorchDynamo
TestEnvironment.def_flag("TEST_WITH_TORCHINDUCTOR", env_var="PYTORCH_TEST_WITH_INDUCTOR")
# AOT_EAGER not tested in ci, useful for debugging
TestEnvironment.def_flag("TEST_WITH_AOT_EAGER", env_var="PYTORCH_TEST_WITH_AOT_EAGER")
TestEnvironment.def_flag("TEST_WITH_TORCHDYNAMO", env_var="PYTORCH_TEST_WITH_DYNAMO",
                         implied_by_fn=lambda: TEST_WITH_TORCHINDUCTOR or TEST_WITH_AOT_EAGER)

if TEST_WITH_TORCHDYNAMO:
    import torch._dynamo
    # Do not spend time on helper functions that are called with different inputs
    torch._dynamo.config.cache_size_limit = 8
    # TODO: Remove this; this is grandfathered in because we suppressed errors
    # on test suite previously
    torch._dynamo.config.suppress_errors = True
    if TEST_WITH_TORCHINDUCTOR:
        import torch._inductor.config
        torch._inductor.config.fallback_random = True


def skipIfTorchDynamo(msg="test doesn't currently work with dynamo"):
    def decorator(fn):
        if not isinstance(fn, type):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                if TEST_WITH_TORCHDYNAMO:
                    raise unittest.SkipTest(msg)
                else:
                    fn(*args, **kwargs)
            return wrapper

        assert(isinstance(fn, type))
        if TEST_WITH_TORCHDYNAMO:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = msg

        return fn


    return decorator

def skipIfTorchInductor(msg="test doesn't currently work with torchinductor"):
    def decorator(fn):
        if not isinstance(fn, type):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                if TEST_WITH_TORCHINDUCTOR:
                    raise unittest.SkipTest(msg)
                else:
                    fn(*args, **kwargs)
            return wrapper

        assert(isinstance(fn, type))
        if TEST_WITH_TORCHINDUCTOR:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = msg

        return fn

    return decorator

def skipRocmIfTorchInductor(msg="test doesn't currently work with torchinductor on the ROCm stack"):
    def decorator(fn):
        if not isinstance(fn, type):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                if TEST_WITH_ROCM and TEST_WITH_TORCHINDUCTOR:
                    raise unittest.SkipTest(f"skipRocmIfTorchInductor: {msg}")
                else:
                    fn(*args, **kwargs)
            return wrapper

        assert(isinstance(fn, type))
        if TEST_WITH_ROCM and TEST_WITH_TORCHINDUCTOR:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = msg

        return fn

    return decorator

# Run PyTorch tests with translation validation on.
TEST_WITH_TV = os.getenv('PYTORCH_TEST_WITH_TV') == '1'

if TEST_WITH_TV:
    torch._dynamo.config.translation_validation = True

# Some tests take too long when dynamic_shapes is combined with
# translation_validation. Whenever that happens, we solve that by
# disabling translation_validation.
def disable_translation_validation_if_dynamic_shapes(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if torch._dynamo.config.dynamic_shapes:
            # Turning TV off due to high latency on dynamic shapes.
            torch._dynamo.config.translation_validation = False
        return fn(*args, **kwargs)
    return wrapper


# Determine whether to enable cuda memory leak check.
# CUDA mem leak check is expensive and thus we don't want to execute it on every
# test case / configuration.
# If this is True then CUDA memory leak checks are skipped. If this is false
#   then CUDA memory leak checks are performed.
# See: https://github.com/pytorch/pytorch/pull/59402#issuecomment-858811135
TestEnvironment.def_flag("TEST_CUDA_MEM_LEAK_CHECK", env_var="PYTORCH_TEST_CUDA_MEM_LEAK_CHECK")

# True if CI is running TBB-enabled Pytorch
IS_TBB = "tbb" in os.getenv("BUILD_ENVIRONMENT", "")

# Dict of NumPy dtype -> torch dtype (when the correspondence exists)
numpy_to_torch_dtype_dict = {
    np.bool_      : torch.bool,
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}


# numpy dtypes like np.float64 are not instances, but rather classes. This leads to rather absurd cases like
# np.float64 != np.dtype("float64") but np.float64 == np.dtype("float64").type.
# Especially when checking against a reference we can't be sure which variant we get, so we simply try both.
def numpy_to_torch_dtype(np_dtype):
    try:
        return numpy_to_torch_dtype_dict[np_dtype]
    except KeyError:
        return numpy_to_torch_dtype_dict[np_dtype.type]


def has_corresponding_torch_dtype(np_dtype):
    try:
        numpy_to_torch_dtype(np_dtype)
        return True
    except KeyError:
        return False


if IS_WINDOWS:
    # Size of `np.intc` is platform defined.
    # It is returned by functions like `bitwise_not`.
    # On Windows `int` is 32-bit
    # https://docs.microsoft.com/en-us/cpp/cpp/data-type-ranges?view=msvc-160
    numpy_to_torch_dtype_dict[np.intc] = torch.int

# Dict of torch dtype -> NumPy dtype
torch_to_numpy_dtype_dict = {value : key for (key, value) in numpy_to_torch_dtype_dict.items()}
torch_to_numpy_dtype_dict.update({
    torch.bfloat16: np.float32,
    torch.complex32: np.complex64
})

def skipIfRocm(func=None, *, msg="test doesn't currently work on the ROCm stack"):
    def dec_fn(fn):
        reason = f"skipIfRocm: {msg}"

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if TEST_WITH_ROCM:
                raise unittest.SkipTest(reason)
            else:
                return fn(*args, **kwargs)
        return wrapper
    if func:
        return dec_fn(func)
    return dec_fn


def runOnRocm(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if TEST_WITH_ROCM:
            fn(*args, **kwargs)
        else:
            raise unittest.SkipTest("test currently only works on the ROCm stack")
    return wrapper

def skipIfMps(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if TEST_MPS:
            raise unittest.SkipTest("test doesn't currently work with MPS")
        else:
            fn(*args, **kwargs)
    return wrapper

# Skips a test on CUDA if ROCm is available and its version is lower than requested.
def skipIfRocmVersionLessThan(version=None):
    def dec_fn(fn):
        @wraps(fn)
        def wrap_fn(self, *args, **kwargs):
            if TEST_WITH_ROCM:
                rocm_version = str(torch.version.hip)
                rocm_version = rocm_version.split("-")[0]    # ignore git sha
                rocm_version_tuple = tuple(int(x) for x in rocm_version.split("."))
                if rocm_version_tuple is None or version is None or rocm_version_tuple < tuple(version):
                    reason = f"ROCm {rocm_version_tuple} is available but {version} required"
                    raise unittest.SkipTest(reason)
            return fn(self, *args, **kwargs)
        return wrap_fn
    return dec_fn

def skipIfNotMiopenSuggestNHWC(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not TEST_WITH_MIOPEN_SUGGEST_NHWC:
            raise unittest.SkipTest("test doesn't currently work without MIOpen NHWC activation")
        else:
            fn(*args, **kwargs)
    return wrapper


# Reverts the linalg backend back to default to make sure potential failures in one
# test do not affect other tests
def setLinalgBackendsToDefaultFinally(fn):
    @wraps(fn)
    def _fn(*args, **kwargs):
        _preferred_backend = torch.backends.cuda.preferred_linalg_library()
        try:
            fn(*args, **kwargs)
        finally:
            torch.backends.cuda.preferred_linalg_library(_preferred_backend)
    return _fn


# Context manager for setting deterministic flag and automatically
# resetting it to its original value
class DeterministicGuard:
    def __init__(self, deterministic, *, warn_only=False):
        self.deterministic = deterministic
        self.warn_only = warn_only

    def __enter__(self):
        self.deterministic_restore = torch.are_deterministic_algorithms_enabled()
        self.warn_only_restore = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(
            self.deterministic,
            warn_only=self.warn_only)

    def __exit__(self, exception_type, exception_value, traceback):
        torch.use_deterministic_algorithms(
            self.deterministic_restore,
            warn_only=self.warn_only_restore)

class AlwaysWarnTypedStorageRemoval:
    def __init__(self, always_warn):
        assert isinstance(always_warn, bool)
        self.always_warn = always_warn

    def __enter__(self):
        self.always_warn_restore = torch.storage._get_always_warn_typed_storage_removal()
        torch.storage._set_always_warn_typed_storage_removal(self.always_warn)

    def __exit__(self, exception_type, exception_value, traceback):
        torch.storage._set_always_warn_typed_storage_removal(self.always_warn_restore)

# Context manager for setting cuda sync debug mode and reset it
# to original value
# we are not exposing it to the core because sync debug mode is
# global and thus not thread safe
class CudaSyncGuard:
    def __init__(self, sync_debug_mode):
        self.mode = sync_debug_mode

    def __enter__(self):
        self.debug_mode_restore = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode(self.mode)

    def __exit__(self, exception_type, exception_value, traceback):
        torch.cuda.set_sync_debug_mode(self.debug_mode_restore)

# This decorator can be used for API tests that call
# torch.use_deterministic_algorithms().  When the test is finished, it will
# restore the previous deterministic flag setting.
#
# If CUDA >= 10.2, this will set the environment variable
# CUBLAS_WORKSPACE_CONFIG=:4096:8 so that the error associated with that
# setting is not thrown during the test unless the test changes that variable
# on purpose. The previous CUBLAS_WORKSPACE_CONFIG setting will also be
# restored once the test is finished.
#
# Note that if a test requires CUDA to actually register the changed
# CUBLAS_WORKSPACE_CONFIG variable, a new subprocess must be created, because
# CUDA only checks the variable when the runtime initializes. Tests can be
# run inside a subprocess like so:
#
#   import subprocess, sys, os
#   script = '''
#   # Test code should go here
#   '''
#   try:
#       subprocess.check_output(
#           [sys.executable, '-c', script],
#           stderr=subprocess.STDOUT,
#           cwd=os.path.dirname(os.path.realpath(__file__)),
#           env=os.environ.copy())
#   except subprocess.CalledProcessError as e:
#       error_message = e.output.decode('utf-8')
#       # Handle exceptions raised by the subprocess here
#
def wrapDeterministicFlagAPITest(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with DeterministicGuard(
                torch.are_deterministic_algorithms_enabled(),
                warn_only=torch.is_deterministic_algorithms_warn_only_enabled()):
            class CuBLASConfigGuard:
                cublas_var_name = 'CUBLAS_WORKSPACE_CONFIG'

                def __enter__(self):
                    self.is_cuda10_2_or_higher = (
                        (torch.version.cuda is not None)
                        and ([int(x) for x in torch.version.cuda.split(".")] >= [10, 2]))
                    if self.is_cuda10_2_or_higher:
                        self.cublas_config_restore = os.environ.get(self.cublas_var_name)
                        os.environ[self.cublas_var_name] = ':4096:8'

                def __exit__(self, exception_type, exception_value, traceback):
                    if self.is_cuda10_2_or_higher:
                        cur_cublas_config = os.environ.get(self.cublas_var_name)
                        if self.cublas_config_restore is None:
                            if cur_cublas_config is not None:
                                del os.environ[self.cublas_var_name]
                        else:
                            os.environ[self.cublas_var_name] = self.cublas_config_restore
            with CuBLASConfigGuard():
                fn(*args, **kwargs)
    return wrapper

def skipIfCompiledWithoutNumpy(fn):
    # Even if the numpy module is present, if `USE_NUMPY=0` is used during the
    # build, numpy tests will fail
    numpy_support = TEST_NUMPY
    if numpy_support:
        try:
            # The numpy module is present, verify that PyTorch is compiled with
            # numpy support
            torch.from_numpy(np.array([2, 2]))
        except RuntimeError:
            numpy_support = False

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not numpy_support:
            raise unittest.SkipTest("PyTorch was compiled without numpy support")
        else:
            fn(*args, **kwargs)
    return wrapper

def _test_function(fn, device):
    def run_test_function(self):
        return fn(self, device)
    return run_test_function

def skipIfNoXNNPACK(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not torch.backends.xnnpack.enabled:
            raise unittest.SkipTest('XNNPACK must be enabled for these tests. Please build with USE_XNNPACK=1.')
        else:
            fn(*args, **kwargs)
    return wrapper

def skipIfNoLapack(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not torch._C.has_lapack:
            raise unittest.SkipTest('PyTorch compiled without Lapack')
        else:
            fn(*args, **kwargs)
    return wrapper

def skipIfNotRegistered(op_name, message):
    """Wraps the decorator to hide the import of the `core`.

    Args:
        op_name: Check if this op is registered in `core._REGISTERED_OPERATORS`.
        message: message to fail with.

    Usage:
        @skipIfNotRegistered('MyOp', 'MyOp is not linked!')
            This will check if 'MyOp' is in the caffe2.python.core
    """
    if not BUILD_WITH_CAFFE2:
        return unittest.skip("Pytorch is compiled without Caffe2")
    try:
        from caffe2.python import core
        skipper = unittest.skipIf(op_name not in core._REGISTERED_OPERATORS,
                                  message)
    except ImportError:
        skipper = unittest.skip("Cannot import `caffe2.python.core`")
    return skipper

def _decide_skip_caffe2(expect_caffe2, reason):
    def skip_dec(func):
        @wraps(func)
        def wrapper(self):
            if torch.onnx._CAFFE2_ATEN_FALLBACK != expect_caffe2:
                raise unittest.SkipTest(reason)
            return func(self)
        return wrapper
    return skip_dec

skipIfCaffe2 = _decide_skip_caffe2(False, "Not compatible with Caffe2")
skipIfNoCaffe2 = _decide_skip_caffe2(True, "Caffe2 is not available")

def skipIfNoSciPy(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not TEST_SCIPY:
            raise unittest.SkipTest("test require SciPy, but SciPy not found")
        else:
            fn(*args, **kwargs)
    return wrapper


def skipIfTBB(message="This test makes TBB sad"):
    def dec_fn(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if IS_TBB:
                raise unittest.SkipTest(message)
            else:
                fn(*args, **kwargs)
        return wrapper
    return dec_fn


def slowTest(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not TEST_WITH_SLOW:
            raise unittest.SkipTest("test is slow; run with PYTORCH_TEST_WITH_SLOW to enable test")
        else:
            fn(*args, **kwargs)
    wrapper.__dict__['slow_test'] = True
    return wrapper


def slowTestIf(condition):
    return slowTest if condition else lambda fn: fn


def skipCUDAMemoryLeakCheckIf(condition):
    def dec(fn):
        if getattr(fn, '_do_cuda_memory_leak_check', True):  # if current True
            fn._do_cuda_memory_leak_check = not condition
        return fn
    return dec

def skipCUDANonDefaultStreamIf(condition):
    def dec(fn):
        if getattr(fn, '_do_cuda_non_default_stream', True):  # if current True
            fn._do_cuda_non_default_stream = not condition
        return fn
    return dec

def suppress_warnings(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn(*args, **kwargs)
    return wrapper


def to_gpu(obj, type_map=None):
    if type_map is None:
        type_map = {}
    if isinstance(obj, torch.Tensor):
        assert obj.is_leaf
        t = type_map.get(obj.dtype, obj.dtype)
        with torch.no_grad():
            res = obj.clone().to(dtype=t, device="cuda")
            res.requires_grad = obj.requires_grad
        return res
    elif torch.is_storage(obj):
        return obj.new().resize_(obj.size()).copy_(obj)
    elif isinstance(obj, list):
        return [to_gpu(o, type_map) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(to_gpu(o, type_map) for o in obj)
    else:
        return deepcopy(obj)


def get_function_arglist(func):
    return inspect.getfullargspec(func).args


def set_rng_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    if TEST_NUMPY:
        np.random.seed(seed)


disable_functorch = torch._C._DisableFuncTorch


@contextlib.contextmanager
def freeze_rng_state():
    # no_dispatch needed for test_composite_compliance
    # Some OpInfos use freeze_rng_state for rng determinism, but
    # test_composite_compliance overrides dispatch for all torch functions
    # which we need to disable to get and set rng state
    with no_dispatch(), disable_functorch():
        rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state()
    try:
        yield
    finally:
        # Modes are not happy with torch.cuda.set_rng_state
        # because it clones the state (which could produce a Tensor Subclass)
        # and then grabs the new tensor's data pointer in generator.set_state.
        #
        # In the long run torch.cuda.set_rng_state should probably be
        # an operator.
        #
        # NB: Mode disable is to avoid running cross-ref tests on thes seeding
        with no_dispatch(), disable_functorch():
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)
            torch.set_rng_state(rng_state)

@contextlib.contextmanager
def set_default_dtype(dtype):
    saved_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(saved_dtype)

def iter_indices(tensor):
    if tensor.dim() == 0:
        return range(0)
    if tensor.dim() == 1:
        return range(tensor.size(0))
    return product(*(range(s) for s in tensor.size()))


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def is_iterable_of_tensors(iterable, include_empty=False):
    """ Returns True if iterable is an iterable of tensors and False o.w.

        If the iterable is empty, the return value is :attr:`include_empty`
    """
    # Tensor itself is iterable so we check this first
    if isinstance(iterable, torch.Tensor):
        return False

    try:
        if len(iterable) == 0:
            return include_empty

        for t in iter(iterable):
            if not isinstance(t, torch.Tensor):
                return False

    except TypeError as te:
        return False

    return True


class CudaNonDefaultStream:
    def __enter__(self):
        # Before starting CUDA test save currently active streams on all
        # CUDA devices and set new non default streams to all CUDA devices
        # to ensure CUDA tests do not use default stream by mistake.
        beforeDevice = torch.cuda.current_device()
        self.beforeStreams = []
        for d in range(torch.cuda.device_count()):
            self.beforeStreams.append(torch.cuda.current_stream(d))
            deviceStream = torch.cuda.Stream(device=d)
            self.beforeStreams[-1].synchronize()
            torch._C._cuda_setStream(stream_id=deviceStream.stream_id,
                                     device_index=deviceStream.device_index,
                                     device_type=deviceStream.device_type)
        torch._C._cuda_setDevice(beforeDevice)

    def __exit__(self, exec_type, exec_value, traceback):
        # After completing CUDA test load previously active streams on all
        # CUDA devices.
        beforeDevice = torch.cuda.current_device()
        for d in range(torch.cuda.device_count()):
            torch._C._cuda_setStream(stream_id=self.beforeStreams[d].stream_id,
                                     device_index=self.beforeStreams[d].device_index,
                                     device_type=self.beforeStreams[d].device_type)
        torch._C._cuda_setDevice(beforeDevice)

class CudaMemoryLeakCheck:
    def __init__(self, testcase, name=None):
        self.name = testcase.id() if name is None else name
        self.testcase = testcase

        # initialize context & RNG to prevent false positive detections
        # when the test is the first to initialize those
        from torch.testing._internal.common_cuda import initialize_cuda_context_rng
        initialize_cuda_context_rng()

    # Stores CUDA memory data provided by PyTorch's caching allocator and
    #   the CUDA driver.
    #
    # NOTE: The undocumented torch.cuda.mem_get_info() returns
    #   (#free bytes, #total bytes available) on the GPU
    def __enter__(self):
        self.caching_allocator_befores = []
        self.driver_befores = []

        # Performs a gc if required (required if any CUDA memory is held)
        num_devices = torch.cuda.device_count()
        for i in range(num_devices):
            caching_allocator_mem_allocated = torch.cuda.memory_allocated(i)
            # NOTE: gc is based exclusively on caching allocator memory
            #   because the driver will always have some bytes in use (context size?)
            if caching_allocator_mem_allocated > 0:
                gc.collect()
                torch._C._cuda_clearCublasWorkspaces()
                torch.cuda.empty_cache()
                break

        # Acquires caching allocator and driver statistics before the test is run
        for i in range(num_devices):
            self.caching_allocator_befores.append(torch.cuda.memory_allocated(i))
            bytes_free, bytes_total = torch.cuda.mem_get_info(i)
            driver_mem_allocated = bytes_total - bytes_free
            self.driver_befores.append(driver_mem_allocated)

    def __exit__(self, exec_type, exec_value, traceback):
        # Don't check for leaks if an exception was thrown
        if exec_type is not None:
            return

        # Compares caching allocator before/after statistics
        # An increase in allocated memory is a discrepancy indicating a possible
        #   memory leak
        discrepancy_detected = False
        num_devices = torch.cuda.device_count()
        for i in range(num_devices):
            # avoid counting cublasWorkspace allocations
            torch._C._cuda_clearCublasWorkspaces()
            caching_allocator_mem_allocated = torch.cuda.memory_allocated(i)

            if caching_allocator_mem_allocated > self.caching_allocator_befores[i]:
                discrepancy_detected = True
                break

        # Short-circuits if no discrepancy detected
        if not discrepancy_detected:
            return

        # Validates the discrepancy persists after garbage collection and
        #   is confirmed by the driver API

        # NOTE: driver API iscrepancies alone are ignored because with the jiterator
        #   some tests may permanently increase the CUDA context size and
        #   that will appear as a driver memory leak but is the expected behavior.

        # GCs and clears the cache
        gc.collect()
        torch.cuda.empty_cache()

        for i in range(num_devices):

            discrepancy_detected = True

            # Query memory multiple items to ensure leak was not transient
            for n in range(3):
                caching_allocator_mem_allocated = torch.cuda.memory_allocated(i)
                bytes_free, bytes_total = torch.cuda.mem_get_info(i)
                driver_mem_allocated = bytes_total - bytes_free

                caching_allocator_discrepancy = False
                driver_discrepancy = False

                if caching_allocator_mem_allocated > self.caching_allocator_befores[i]:
                    caching_allocator_discrepancy = True

                if driver_mem_allocated > self.driver_befores[i]:
                    driver_discrepancy = True

                if not(caching_allocator_discrepancy or driver_discrepancy):
                    # Leak was false positive, exit loop
                    discrepancy_detected = False
                    break

            if not discrepancy_detected:
                continue

            if caching_allocator_discrepancy and not driver_discrepancy:
                # Just raises a warning if the leak is not validated by the
                #   driver API
                # NOTE: this may be a problem with how the caching allocator collects its
                #   statistics or a leak too small to trigger the allocation of an
                #   additional block of memory by the CUDA driver
                msg = ("CUDA caching allocator reports a memory leak not "
                       "verified by the driver API in {}! "
                       "Caching allocator allocated memory was {} and is now reported as {} "
                       "on device {}. "
                       "CUDA driver allocated memory was {} and is now {}.").format(
                    self.name,
                    self.caching_allocator_befores[i],
                    caching_allocator_mem_allocated,
                    i,
                    self.driver_befores[i],
                    driver_mem_allocated)
                warnings.warn(msg)
            elif caching_allocator_discrepancy and driver_discrepancy:
                # A caching allocator discrepancy validated by the driver API is a
                #   failure (except on ROCm, see below)
                msg = ("CUDA driver API confirmed a leak in {}! "
                       "Caching allocator allocated memory was {} and is now reported as {} "
                       "on device {}. "
                       "CUDA driver allocated memory was {} and is now {}.").format(
                    self.name,
                    self.caching_allocator_befores[i],
                    caching_allocator_mem_allocated,
                    i,
                    self.driver_befores[i],
                    driver_mem_allocated)

                raise RuntimeError(msg)

@contextmanager
def skip_exception_type(exc_type):
    try:
        yield
    except exc_type as e:
        raise unittest.SkipTest(f"not implemented: {e}") from e

@contextmanager
def print_repro_on_failure(repro_str):
    try:
        yield
    except unittest.SkipTest:
        raise
    except Exception as e:
        # NB: Hacking the exception args is the cleanest way I've found to append
        # failure reproduction info without poisoning the stack trace.
        if len(e.args) >= 1:
            e.args = (f"{e.args[0]}\n{repro_str}", *e.args[1:])
        raise

#  "min_satisfying_examples" setting has been deprecated in hypothesis
#  3.56.0 and removed in hypothesis 4.x
try:
    import hypothesis

    def settings(*args, **kwargs):
        if 'min_satisfying_examples' in kwargs and hypothesis.version.__version_info__ >= (3, 56, 0):
            kwargs.pop('min_satisfying_examples')
        return hypothesis.settings(*args, **kwargs)


    hypothesis.settings.register_profile(
        "pytorch_ci",
        settings(
            derandomize=True,
            suppress_health_check=[hypothesis.HealthCheck.too_slow],
            database=None,
            max_examples=50,
            verbosity=hypothesis.Verbosity.normal))
    hypothesis.settings.register_profile(
        "dev",
        settings(
            suppress_health_check=[hypothesis.HealthCheck.too_slow],
            database=None,
            max_examples=10,
            verbosity=hypothesis.Verbosity.normal))
    hypothesis.settings.register_profile(
        "debug",
        settings(
            suppress_health_check=[hypothesis.HealthCheck.too_slow],
            database=None,
            max_examples=1000,
            verbosity=hypothesis.Verbosity.verbose))

    hypothesis.settings.load_profile(
        "pytorch_ci" if IS_CI else os.getenv('PYTORCH_HYPOTHESIS_PROFILE', 'dev')
    )
except ImportError:
    print('Fail to import hypothesis in common_utils, tests are not derandomized')

# Used in check_if_enable to see if a test method should be disabled by an issue,
# sanitizes a test method name from appended suffixes by @dtypes parametrization.
# e.g., an issue with title "DISABLED test_bitwise_ops (__main__.TestBinaryUfuncs)" should
# disabled ALL parametrized test_bitwise_ops tests, such test_bitwise_ops_cuda_int32
def remove_device_and_dtype_suffixes(test_name: str) -> str:
    # import statement is localized to avoid circular dependency issues with common_device_type.py
    from torch.testing._internal.common_device_type import get_device_type_test_bases
    device_suffixes = [x.device_type for x in get_device_type_test_bases()]
    dtype_suffixes = [str(dt)[len("torch."):] for dt in get_all_dtypes()]

    test_name_chunks = test_name.split("_")
    if len(test_name_chunks) > 0 and test_name_chunks[-1] in dtype_suffixes:
        if len(test_name_chunks) > 1 and test_name_chunks[-2] in device_suffixes:
            return "_".join(test_name_chunks[0:-2])
        return "_".join(test_name_chunks[0:-1])
    return test_name


def check_if_enable(test: unittest.TestCase):
    classname = str(test.__class__).split("'")[1].split(".")[-1]
    sanitized_testname = remove_device_and_dtype_suffixes(test._testMethodName)

    def matches_test(target: str):
        target_test_parts = target.split()
        if len(target_test_parts) < 2:
            # poorly formed target test name
            return False
        target_testname = target_test_parts[0]
        target_classname = target_test_parts[1][1:-1].split(".")[-1]
        # if test method name or its sanitized version exactly matches the disabled
        # test method name AND allow non-parametrized suite names to disable
        # parametrized ones (TestSuite disables TestSuiteCPU)
        return classname.startswith(target_classname) and (target_testname in (test._testMethodName, sanitized_testname))

    if any(matches_test(x) for x in slow_tests_dict.keys()):
        getattr(test, test._testMethodName).__dict__['slow_test'] = True
        if not TEST_WITH_SLOW:
            raise unittest.SkipTest("test is slow; run with PYTORCH_TEST_WITH_SLOW to enable test")

    if not IS_SANDCASTLE:
        should_skip = False
        skip_msg = ""

        for disabled_test, (issue_url, platforms) in disabled_tests_dict.items():
            if matches_test(disabled_test):
                platform_to_conditional: Dict = {
                    "mac": IS_MACOS,
                    "macos": IS_MACOS,
                    "win": IS_WINDOWS,
                    "windows": IS_WINDOWS,
                    "linux": IS_LINUX,
                    "rocm": TEST_WITH_ROCM,
                    "asan": TEST_WITH_ASAN,
                    "dynamo": TEST_WITH_TORCHDYNAMO,
                    "inductor": TEST_WITH_TORCHINDUCTOR,
                    "slow": TEST_WITH_SLOW,
                }

                invalid_platforms = list(filter(lambda p: p not in platform_to_conditional, platforms))
                if len(invalid_platforms) > 0:
                    invalid_plats_str = ", ".join(invalid_platforms)
                    valid_plats = ", ".join(platform_to_conditional.keys())

                    print(f"Test {disabled_test} is disabled for some unrecognized ",
                          f"platforms: [{invalid_plats_str}]. Please edit issue {issue_url} to fix the platforms ",
                          "assigned to this flaky test, changing \"Platforms: ...\" to a comma separated ",
                          f"subset of the following (or leave it blank to match all platforms): {valid_plats}")

                    # Sanitize the platforms list so that we continue to disable the test for any valid platforms given
                    platforms = list(filter(lambda p: p in platform_to_conditional, platforms))

                if platforms == [] or any(platform_to_conditional[platform] for platform in platforms):
                    should_skip = True
                    skip_msg = f"Test is disabled because an issue exists disabling it: {issue_url}" \
                        f" for {'all' if platforms == [] else ''}platform(s) {', '.join(platforms)}. " \
                        "If you're seeing this on your local machine and would like to enable this test, " \
                        "please make sure CI is not set and you are not using the flag --import-disabled-tests."
                    break

        if should_skip and not RERUN_DISABLED_TESTS:
            # Skip the disabled test when not running under --rerun-disabled-tests verification mode
            raise unittest.SkipTest(skip_msg)

        if not should_skip and RERUN_DISABLED_TESTS:
            skip_msg = "Test is enabled but --rerun-disabled-tests verification mode is set, so only" \
                " disabled tests are run"
            raise unittest.SkipTest(skip_msg)

    if TEST_SKIP_FAST:
        if hasattr(test, test._testMethodName) and not getattr(test, test._testMethodName).__dict__.get('slow_test', False):
            raise unittest.SkipTest("test is fast; we disabled it with PYTORCH_TEST_SKIP_FAST")


# `TestCase.assertEqual` is very permissive and coerced the inputs into a format that could be compared. This is very
# convenient when writing tests, but not so much while reviewing them. By default, the comparison `Pair` framework of
# `torch.testing._comparison.are_equal`, used for example by the public testing function
# `torch.testing.assert_close`, is more strict. In order to use the same framework and thus reduce the divergence
# between internal and external comparison logic as much as possible, we define some "relaxed" pairs here. They only
# change the supported inputs, but the comparison logic is the same.
# TODO: Revisit the relaxed pairs and check how much work it is to fix the tests that would fail without the relaxation.

class RelaxedBooleanPair(BooleanPair):
    """Pair for boolean-like inputs.

    In contrast to the builtin :class:`BooleanPair`, this class also supports one input being a number or a single
    element tensor-like.
    """
    _supported_number_types = NumberPair(0, 0)._supported_types

    def _process_inputs(self, actual, expected, *, id):
        # We require only one of the inputs of the inputs to be a boolean and the other can also be a boolean, a
        # number, or a single element tensor or array, whereas in default BooleanPair both inputs have to be booleans.
        tensor_or_array_types: Tuple[Type, ...] = (torch.Tensor, np.ndarray)
        other_supported_types = (*self._supported_types, *self._supported_number_types, *tensor_or_array_types)
        if not (
            (isinstance(actual, self._supported_types) and isinstance(expected, other_supported_types))
            or (isinstance(expected, self._supported_types) and isinstance(actual, other_supported_types))
        ):
            self._inputs_not_supported()

        return [self._to_bool(input, id=id) for input in (actual, expected)]

    def _to_bool(self, bool_like, *, id):
        if isinstance(bool_like, np.number):
            return bool(bool_like.item())
        elif type(bool_like) in self._supported_number_types:
            return bool(bool_like)
        elif isinstance(bool_like, (torch.Tensor, np.ndarray)):
            numel = bool_like.numel() if isinstance(bool_like, torch.Tensor) else bool_like.size
            if numel > 1:
                self._fail(
                    ValueError,
                    f"Only single element tensor-likes can be compared against a boolean. "
                    f"Got {numel} elements instead.",
                    id=id
                )

            return bool(bool_like.item())
        else:
            return super()._to_bool(bool_like, id=id)


class RelaxedNumberPair(NumberPair):
    """Pair for number-like inputs.

    In contrast to the builtin :class:`NumberPair`, this class also supports one input being a single element
    tensor-like or a :class:`enum.Enum`. (D)Type checks are disabled, meaning comparing 1 to 1.0 succeeds even when
    ``check_dtype=True`` is passed.

    In addition, this class uses looser default tolerances for :class:`float` and :class:`complex` inputs. Also
    supports overriding the absolute and relative tolerance through the ``@precisionOverride`` and
    ``@toleranceOverride`` decorators.
    """
    _TYPE_TO_DTYPE = {
        int: torch.int64,
        float: torch.float32,
        complex: torch.complex64,
    }

    def __init__(
            self, actual, expected, *, rtol_override=0.0, atol_override=0.0, check_dtype=None, **other_parameters
    ) -> None:
        super().__init__(actual, expected, check_dtype=False, **other_parameters)
        self.rtol = max(self.rtol, rtol_override)
        self.atol = max(self.atol, atol_override)

    def _process_inputs(self, actual, expected, *, id):
        # We require only one of the inputs of the inputs to be a number and the other can also be a number or a single
        # element tensor or array, whereas in default NumberPair both inputs have to be numbers.
        tensor_or_array_types: Tuple[Type, ...] = (torch.Tensor, np.ndarray)
        other_supported_types = (*self._supported_types, *tensor_or_array_types)
        if not (
                (isinstance(actual, self._supported_types) and isinstance(expected, other_supported_types))
                or (isinstance(expected, self._supported_types) and isinstance(actual, other_supported_types))
        ):
            self._inputs_not_supported()

        return [self._to_number(input, id=id) for input in (actual, expected)]

    def _to_number(self, number_like, *, id):
        if isinstance(number_like, (torch.Tensor, np.ndarray)):
            numel = number_like.numel() if isinstance(number_like, torch.Tensor) else number_like.size
            if numel > 1:
                self._fail(
                    ValueError,
                    f"Only single element tensor-likes can be compared against a number. "
                    f"Got {numel} elements instead.",
                    id=id
                )
            number = number_like.item()
            if isinstance(number, bool):
                number = int(number)

            return number
        elif isinstance(number_like, Enum):
            return int(number_like)  # type: ignore[call-overload]
        else:
            return super()._to_number(number_like, id=id)


class TensorOrArrayPair(TensorLikePair):
    """Pair for tensor-like inputs.

    On the one hand this class is stricter than the builtin :class:`TensorLikePair` since it only allows instances of
    :class:`torch.Tensor` and :class:`numpy.ndarray` rather than allowing any tensor-like than can be converted into a
    tensor. On the other hand this class is looser since it converts all inputs into tensors with no regard of their
    relationship, e.g. comparing a :class:`torch.Tensor` to :class:`numpy.ndarray` is fine.

    In addition, this class supports overriding the absolute and relative tolerance through the ``@precisionOverride``
    and ``@toleranceOverride`` decorators.
    """
    def __init__(self, actual, expected, *, rtol_override=0.0, atol_override=0.0, **other_parameters):
        super().__init__(actual, expected, **other_parameters)
        self.rtol = max(self.rtol, rtol_override)
        self.atol = max(self.atol, atol_override)

    def _process_inputs(self, actual, expected, *, id, allow_subclasses):
        self._check_inputs_isinstance(actual, expected, cls=(torch.Tensor, np.ndarray))

        actual, expected = (self._to_tensor(input) for input in (actual, expected))
        for tensor in (actual, expected):
            self._check_supported(tensor, id=id)
        return actual, expected


class TypedStoragePair(TensorLikePair):
    """Pair for :class:`torch.storage.TypedStorage` inputs."""
    def __init__(self, actual, expected, *, rtol_override=0.0, atol_override=0.0, **other_parameters):
        self._check_inputs_isinstance(actual, expected, cls=torch.storage.TypedStorage)
        super().__init__(actual, expected, **other_parameters)
        self.rtol = max(self.rtol, rtol_override)
        self.atol = max(self.atol, atol_override)

    def _to_tensor(self, typed_storage):
        return torch.tensor(
            typed_storage._untyped_storage,
            dtype={
                torch.quint8: torch.uint8,
                torch.quint4x2: torch.uint8,
                torch.quint2x4: torch.uint8,
                torch.qint32: torch.int32,
                torch.qint8: torch.int8
            }.get(typed_storage.dtype, typed_storage.dtype),
            device=typed_storage.device,
        )


class UnittestPair(Pair):
    """Fallback ABC pair that handles non-numeric inputs.

    To avoid recreating the mismatch messages of :meth:`unittest.TestCase.assertEqual`, this pair simply wraps it in
    order to use it with the :class:`Pair` "framework" from :func:`are_equal`.

    Define the :attr:`UnittestPair.CLS` in a subclass to indicate which class(es) of the inputs the pair should support.
    """
    CLS: Union[Type, Tuple[Type, ...]]
    TYPE_NAME: Optional[str] = None

    def __init__(self, actual, expected, **other_parameters):
        self._check_inputs_isinstance(actual, expected, cls=self.CLS)
        super().__init__(actual, expected, **other_parameters)

    def compare(self):
        test_case = unittest.TestCase()

        try:
            return test_case.assertEqual(self.actual, self.expected)
        except test_case.failureException as error:
            msg = str(error)

        type_name = self.TYPE_NAME or (self.CLS if isinstance(self.CLS, type) else self.CLS[0]).__name__
        self._fail(AssertionError, f"{type_name.title()} comparison failed: {msg}")


class StringPair(UnittestPair):
    CLS = (str, bytes)
    TYPE_NAME = "string"


class SetPair(UnittestPair):
    CLS = set


class TypePair(UnittestPair):
    CLS = type


class ObjectPair(UnittestPair):
    CLS = object


# This implements a variant of assertRaises/assertRaisesRegex where we first test
# if the exception is NotImplementedError, and if so just skip the test instead
# of failing it.
#
# This is implemented by inheriting from the (private) implementation of
# assertRaises from unittest.case, and slightly tweaking it for this new
# behavior.  The year is 2021: this private class hierarchy hasn't changed since
# 2010, seems low risk to inherit from.
class AssertRaisesContextIgnoreNotImplementedError(unittest.case._AssertRaisesContext):
    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None and issubclass(exc_type, NotImplementedError):
            self.test_case.skipTest(f"not_implemented: {exc_value}")  # type: ignore[attr-defined]
        return super().__exit__(exc_type, exc_value, tb)


@contextmanager
def set_warn_always_context(new_val: bool):
    old_val = torch.is_warn_always_enabled()
    torch.set_warn_always(new_val)
    try:
        yield
    finally:
        torch.set_warn_always(old_val)


class NoTest:
    # causes pytest to not recognize this class as a test
    __test__ = False


class TestCase(expecttest.TestCase):
    # NOTE: "precision" lets classes and generated tests set minimum
    # atol values when comparing tensors. Used by @precisionOverride and @toleranceOverride, for
    # example.
    # NOTE: "rel_tol" lets classes and generated tests set minimum
    # rtol values when comparing tensors. Used by @toleranceOverride, for example.
    _precision: float = 0
    _rel_tol: float = 0

    # checker to early terminate test suite if unrecoverable failure occurs.
    def _should_stop_test_suite(self):
        if torch.cuda.is_initialized():
            # CUDA device side error will cause subsequence test cases to fail.
            # stop entire test suite if catches RuntimeError during torch.cuda.synchronize().
            try:
                torch.cuda.synchronize()
            except RuntimeError as rte:
                print("TEST SUITE EARLY TERMINATION due to torch.cuda.synchronize() failure", file=sys.stderr)
                return True
            return False
        else:
            return False

    @property
    def precision(self) -> float:
        return self._precision

    @precision.setter
    def precision(self, prec: float) -> None:
        self._precision = prec

    @property
    def rel_tol(self) -> float:
        return self._rel_tol

    @rel_tol.setter
    def rel_tol(self, prec: float) -> None:
        self._rel_tol = prec

    _do_cuda_memory_leak_check = False
    _do_cuda_non_default_stream = False

    # When True, if a test case raises a NotImplementedError, instead of failing
    # the test, skip it instead.
    _ignore_not_implemented_error = False

    def __init__(self, method_name='runTest'):
        super().__init__(method_name)

        test_method = getattr(self, method_name, None)
        if test_method is not None:
            # Wraps the tested method if we should do CUDA memory check.
            if TEST_CUDA_MEM_LEAK_CHECK:
                self._do_cuda_memory_leak_check &= getattr(test_method, '_do_cuda_memory_leak_check', True)
                # FIXME: figure out the flaky -1024 anti-leaks on windows. See #8044
                if self._do_cuda_memory_leak_check and not IS_WINDOWS:
                    self.wrap_with_cuda_policy(method_name, self.assertLeaksNoCudaTensors)

            # Wraps the tested method if we should enforce non default CUDA stream.
            self._do_cuda_non_default_stream &= getattr(test_method, '_do_cuda_non_default_stream', True)
            if self._do_cuda_non_default_stream and not IS_WINDOWS:
                self.wrap_with_cuda_policy(method_name, self.enforceNonDefaultStream)

            if self._ignore_not_implemented_error:
                self.wrap_with_policy(method_name, lambda: skip_exception_type(NotImplementedError))

            if PRINT_REPRO_ON_FAILURE:
                env_var_prefix = TestEnvironment.repro_env_var_prefix()
                try:
                    def _get_rel_test_path(abs_test_path):
                        # Attempt to get relative path based on the "test" dir.
                        # In CI, the working dir is not guaranteed to be the base repo dir so
                        # we can't just compute relative path from that.
                        parts = Path(abs_test_path).parts
                        for i, part in enumerate(parts):
                            if part == "test":
                                base_dir = os.path.join(*parts[:i])
                                return os.path.relpath(abs_test_path, start=base_dir)

                        # Can't determine containing dir; just return the test filename.
                        # The path isn't strictly correct but it's arguably better than nothing.
                        return os.path.split(abs_test_path)[1]

                    test_filename = _get_rel_test_path(inspect.getfile(type(self)))
                    repro_str = f"""
To execute this test, run the following from the base repo dir:
    {env_var_prefix} python {test_filename} -k {method_name}

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0"""
                    self.wrap_with_policy(
                        method_name,
                        lambda repro_str=repro_str: print_repro_on_failure(repro_str=repro_str))
                except Exception as e:
                    # Don't fail entirely if we can't get the test filename
                    log.info("could not print repro string", extra=str(e))

    def assertLeaksNoCudaTensors(self, name=None):
        name = self.id() if name is None else name
        return CudaMemoryLeakCheck(self, name)

    def enforceNonDefaultStream(self):
        return CudaNonDefaultStream()

    def assertExpectedInline(self, actual, expect, skip=0):
        return super().assertExpectedInline(actual if isinstance(actual, str) else str(actual), expect, skip + 1)

    # Munges exceptions that internally contain stack traces, using munge_exc
    def assertExpectedInlineMunged(
        self, exc_type, callable, expect, *, suppress_suffix=True
    ):
        try:
            callable()
        except exc_type as e:
            self.assertExpectedInline(
                munge_exc(e, suppress_suffix=suppress_suffix, skip=1), expect, skip=1
            )
            return
        self.fail(msg="Did not raise when expected to")

    def assertLogs(self, logger=None, level=None):
        if logger is None:
            logger = logging.getLogger("torch")
        return super().assertLogs(logger, level)

    def assertNoLogs(self, logger=None, level=None):
        if logger is None:
            logger = logging.getLogger("torch")
        return super().assertNoLogs(logger, level)

    def wrap_with_cuda_policy(self, method_name, policy):
        test_method = getattr(self, method_name)
        # the import below may initialize CUDA context, so we do it only if
        # self._do_cuda_memory_leak_check or self._do_cuda_non_default_stream
        # is True.
        # TODO: sure looks like we unconditionally initialize the context here
        # -- ezyang
        from torch.testing._internal.common_cuda import TEST_CUDA
        fullname = self.id().lower()  # class_name.method_name
        if TEST_CUDA and ('gpu' in fullname or 'cuda' in fullname):
            setattr(self, method_name, self.wrap_method_with_policy(test_method, policy))

    def wrap_with_policy(self, method_name, policy):
        test_method = getattr(self, method_name)
        setattr(self, method_name, self.wrap_method_with_policy(test_method, policy))

    # A policy is a zero-argument function that returns a context manager.
    # We don't take the context manager directly as it may be necessary to
    # construct it once per test method
    def wrap_method_with_policy(self, method, policy):
        # Assumes that `method` is the tested function in `self`.
        # NOTE: Python Exceptions (e.g., unittest.Skip) keeps objects in scope
        #       alive, so this cannot be done in setUp and tearDown because
        #       tearDown is run unconditionally no matter whether the test
        #       passes or not. For the same reason, we can't wrap the `method`
        #       call in try-finally and always do the check.
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            with policy():
                method(*args, **kwargs)
        return types.MethodType(wrapper, self)

    def wrap_with_cuda_memory_check(self, method):
        return self.wrap_method_with_policy(method, self.assertLeaksNoCudaTensors)

    # Recursive function that incorporates retry logic when PYTORCH_RETRY_TEST_CASES=1 and enables early test
    # termination. [DISCLAIMER: ONLY WORKS WITH UNITTEST]
    # When report_only is True, flaky tests are only reported, but the signal remains the same (the test will still
    # show up red).
    # Otherwise, the flaky test will show up green while its stats are captured by test reports.
    def _run_with_retry(self, result=None, num_runs_left=0, report_only=True, num_red=0, num_green=0):
        using_unittest = isinstance(result, unittest.TestResult)
        if num_runs_left == 0:
            # The logic when RERUN_DISABLED_TESTS is set to true is as follows:
            # |-if the disabled test passes:
            # |-- if it's flaky:
            # |---  Do nothing because it's still flaky
            # |-- elif it isn't flaky anymore:
            # |---  Close the disabled ticket (later)
            # |
            # |- elif the disabled test fails after n retries:
            # |--  This is expected, report this but don't fail the job
            skipped_msg = {
                "num_red": num_red,
                "num_green": num_green,
                "max_num_retries": MAX_NUM_RETRIES,
                "rerun_disabled_test": RERUN_DISABLED_TESTS,
            }

            traceback_str = ""
            if RERUN_DISABLED_TESTS and using_unittest:
                # Hide all failures and errors when RERUN_DISABLED_TESTS is enabled. This is
                # a verification check, we don't want more red signals coming from it
                if result.failures:
                    _, traceback_str = result.failures.pop(-1)
                if result.errors:
                    _, traceback_str = result.errors.pop(-1)

                if traceback_str:
                    skipped_msg["traceback_str"] = traceback_str

                if num_green == 0:
                    # The disabled test fails, report as skipped but don't fail the job
                    result.addSkip(self, json.dumps(skipped_msg))

                if num_red == 0:
                    # The test passes after re-running multiple times. This acts as a signal
                    # to confirm that it's not flaky anymore
                    result.addSuccess(self)

            if num_green > 0 and num_red > 0 and using_unittest:
                skipped_msg["flaky"] = True
                # Still flaky, do nothing
                result.addSkip(self, json.dumps(skipped_msg))

            return

        if using_unittest:
            # Keep track of the number of tests marked as failures, errors, and skipped before starting
            failures_before = 0 if result is None else len(result.failures)
            errors_before = 0 if result is None else len(result.errors)
            skipped_before = 0 if result is None else len(result.skipped)

        super_run = super().run
        if TEST_WITH_TORCHINDUCTOR:
            super_run = torch._dynamo.optimize("inductor")(super_run)
        elif TEST_WITH_AOT_EAGER:
            super_run = torch._dynamo.optimize("aot_eager")(super_run)
        elif TEST_WITH_TORCHDYNAMO:
            # TorchDynamo optimize annotation
            super_run = torch._dynamo.optimize("eager")(super_run)

        super_run(result=result)

        # Early terminate test if necessary.
        if self._should_stop_test_suite():
            if result.wasSuccessful():
                case = TestCase()
                if TEST_SAVE_XML is not None:
                    # This is a big hacky, XMLRunner modifies expected type from TestCase to TestInfo
                    # Create dummy TestInfo to record results correctly
                    from xmlrunner.result import _TestInfo  # type: ignore[import]
                    case = _TestInfo(result, case)
                    case.output = _TestInfo.ERROR
                    case.elapsed_time = 0.0
                    case.test_description = "TestSuiteEarlyFailure"
                # This shouldn't really happen, but if does add fake failure
                # For more details see https://github.com/pytorch/pytorch/issues/71973
                result.failures.append((case, "TestSuite execution was aborted early"))
                assert result.wasSuccessful() is False
            result.stop()

        if not RETRY_TEST_CASES or not using_unittest:
            return

        err = sys.exc_info()
        num_retries_left = num_runs_left - 1
        if failures_before < len(result.failures):
            print(f"    {self._testMethodName} failed - num_retries_left: {num_retries_left}")
            if (report_only and num_retries_left < MAX_NUM_RETRIES) or (not report_only and num_retries_left > 0):
                _, traceback_str = result.failures.pop(-1)
                print(traceback_str)
                result.addExpectedFailure(self, err)
            self._run_with_retry(result=result, num_runs_left=num_retries_left, report_only=report_only,
                                 num_red=num_red + 1, num_green=num_green)
        elif errors_before < len(result.errors):
            print(f"    {self._testMethodName} errored - num_retries_left: {num_retries_left}")
            if (report_only and num_retries_left < MAX_NUM_RETRIES) or (not report_only and num_retries_left > 0):
                _, traceback_str = result.errors.pop(-1)
                print(traceback_str)
                result.addExpectedFailure(self, err)
            self._run_with_retry(result=result, num_runs_left=num_retries_left, report_only=report_only,
                                 num_red=num_red + 1, num_green=num_green)
        elif RERUN_DISABLED_TESTS and num_retries_left <= MAX_NUM_RETRIES and skipped_before == len(result.skipped):
            # Always re-run up to MAX_NUM_RETRIES when running under rerun disabled tests modes if the test successes.
            # The parameter num_retries_left can be equal to MAX_NUM_RETRIES here because num_runs_left is initially
            # set to MAX_NUM_RETRIES + 1, i.e. the first run successes
            #
            # Also if the result is skipped, this is due to check_if_enable skipping non-disabled tests, thus we
            # want to ignore them, not retrying and skipping multiple times
            print(f"    {self._testMethodName} succeeded - num_retries_left: {num_retries_left}")
            result.addSuccess(self)
            self._run_with_retry(result=result, num_runs_left=num_retries_left, report_only=report_only,
                                 num_red=num_red, num_green=num_green + 1)
        elif report_only and num_retries_left < MAX_NUM_RETRIES:
            # The original logic here is that num_retries_left must be smaller than MAX_NUM_RETRIES indicating
            # that at least one retry has been spent
            print(f"    {self._testMethodName} succeeded - num_retries_left: {num_retries_left}")
            result.addUnexpectedSuccess(self)
            self._run_with_retry(result=result, num_runs_left=num_retries_left, report_only=report_only,
                                 num_red=num_red, num_green=num_green + 1)
        elif not report_only and num_retries_left < MAX_NUM_RETRIES:
            # in this case, our test was rerun (as a retry has been used) and it just passed.
            # we incur one more recursive call with num_runs_left = 0 to allow for accurate flaky reporting
            self._run_with_retry(result=result, num_runs_left=0, report_only=report_only,
                                 num_red=num_red, num_green=num_green + 1)


    def run(self, result=None):
        with contextlib.ExitStack() as stack:
            if TEST_WITH_CROSSREF:
                stack.enter_context(CrossRefMode())
            num_runs = MAX_NUM_RETRIES + 1 if RETRY_TEST_CASES else 1
            self._run_with_retry(
                result=result,
                num_runs_left=num_runs,
                report_only=not OVERRIDE_FLAKY_SIGNAL,
                num_red=0,
                num_green=0)

    def setUp(self):
        check_if_enable(self)
        set_rng_seed(SEED)

        # Save global check sparse tensor invariants state that can be
        # restored from tearDown:
        self._check_invariants = torch.sparse.check_sparse_tensor_invariants.is_enabled()

        # Enable invariant checks for all sparse tensors constructions
        # including the unsafe ones. If this is not desired for some
        # test case, use check_invariants=False optional argument to
        # sparse tensor constructors or
        # @torch.sparse.check_sparse_tensor_invariants(False)
        # decorator to disable the invariant checks.
        torch.sparse.check_sparse_tensor_invariants.enable()

    def tearDown(self):
        # There exists test cases that override TestCase.setUp
        # definition, so we cannot assume that _check_invariants
        # attribute is defined in general.
        if hasattr(self, '_check_invariants'):
            # Restore the global check sparse tensor invariants state
            if self._check_invariants:
                torch.sparse.check_sparse_tensor_invariants.enable()
            else:
                torch.sparse.check_sparse_tensor_invariants.disable()

    @staticmethod
    def _make_crow_indices(n_rows, n_cols, nnz,
                           *, device, dtype, random=True):
        """Return crow_indices of a CSR tensor with size (n_rows, n_cols) and
        the number of specified elements nnz.

        If random is True, the column counts of rows are in random
        order. Otherwise, the column counts of rows are defined by the
        used sampling method.

        Sampling method
        ---------------

        The used sampling method was introduced in
        https://pearu.github.io/csr_sampling.html, and here we give
        only an overall description of the method.

        Notice that crow_indices can be defined as cumsum(counts)
        where counts is a sequence of non-negative integers satisfying
        the following conditions:

          len(counts) == n_rows + 1
          counts.max() <= n_cols

        while counts[i + 1] is interpreted as the number of specified
        elements in the i-th row.

        The used sampling method aims at increasing the diversity of
        CSR samples, that is, a CSR sample should contain (i) rows
        that are all filled, (ii) rows with no elements at all, and
        (iii) rows that are partially filled. At the same time and for
        the given total number of specified elements (nnz), there
        should be minimal preference to rows with a given number of
        elements.  To achieve this, the sampling method is built-up on
        using a sawteeth model for counts. In the simplest case, we
        would have

          counts = arange(n_rows + 1) % (n_cols + 1)

        that has equal number of all possible column counts per row.
        This formula can be used only for specific input values of
        n_rows, n_cols, and nnz. To generalize this model to any
        combinations of inputs, the counts model above is extended
        with an incomplete sawtooth, and the right and lower
        rectangular parts that will guarantee that

          counts.sum() == nnz

        for any combination of n_rows, n_cols, and nnz. Basically,
        we'll find a maximal window in (n_rows + 1, n_cols + 1)-grid
        that is able to hold a sequence of sawteeth and so-called
        final correction, while the external part of the window is
        filled with counts to meet the nnz constraint exactly.
        """
        assert 0 <= nnz <= n_rows * n_cols, (nnz, n_rows, n_cols)

        def sawteeth(n, m):
            # return the total number of counts in the sequence of
            # sawteeth where n and m define a window in (n_rows+1,
            # n_cols+1) rectangle where the sequence of sawteeth
            # perfectly fit.
            M = (n_cols - m) * (n_cols - m + 1) // 2
            K = (n_rows - n) % (n_cols - m + 1)
            return M * ((n_rows - n) // (n_cols - m + 1)) + K * (K - 1) // 2

        # Different from the original method description, here counts
        # has leading 0 required by crow_indices:
        counts = torch.zeros(n_rows + 1, dtype=dtype, device=torch.device('cpu'))

        n = m = 0
        N = sawteeth(n, m)
        if N and nnz >= max(N, n_cols):
            # determine the width of the sawteeth window. We use bisection to solve
            #   N(n, 0) == 0 or nnz - n * n_cols < max(N(n, 0), n_cols)
            # for n
            n_left = n
            n_right = n_rows - 1
            N_right = sawteeth(n_right, m)
            while n_right - n_left > 1:
                n_middle = (n_left + n_right) // 2
                N_middle = sawteeth(n_middle, m)
                if N_middle == 0 or nnz - n_middle * n_cols < max(N_middle, n_cols):
                    n_right, N_right = n_middle, N_middle
                else:
                    n_left = n_middle
            n, N = n_right, N_right
            # fill the right rectangle with counts:
            assert n
            counts[-n:].fill_(n_cols)

        if N and nnz - n * n_cols >= max(N, n_rows - n):
            # determine the height of the sawteeth window. We use bisection to solve
            #   N(n, m) == 0 or nnz - n * n_cols - m * (n_rows - n) < max(N(n, m), n_rows - n)
            # for m.
            m_left = m
            m_right = n_cols - 1
            N_right = sawteeth(n, m_right)
            while m_right - m_left > 1:
                m_middle = (m_left + m_right) // 2
                N_middle = sawteeth(n, m_middle)
                if N_middle == 0 or nnz - n * n_cols - m_middle * (n_rows - n) < max(N_middle, n_rows - n):
                    m_right, N_right = m_middle, N_middle
                else:
                    m_left = m_middle
            m, N = m_right, N_right
            # fill the bottom rectangle with counts:
            assert m
            counts[1:n_rows - n + 1].fill_(m)

        if N:
            # fill the sawteeth window with counts
            q, r = divmod(nnz - n * n_cols - m * (n_rows - n),
                          (n_cols - m) * (n_cols - m + 1) // 2)
            p = 1 + q * (n_cols - m + 1)
            k = math.isqrt(2 * r)
            if k * (k + 1) > 2 * r:
                k -= 1
            corr = r - k * (k + 1) // 2
            assert not ((p > 1) and (m > 0))  # full sawteeth are never on top of a bottom rectangle
            # sequence of full sawteeth:
            counts[1:p] = torch.arange(p - 1, dtype=dtype, device=counts.device) % (n_cols - m + 1)
            # incomplete sawtooth:
            counts[p:p + k + 1] += torch.arange(k + 1, dtype=dtype, device=counts.device)
        else:
            # given input does not support sawteeth
            p = 1
            corr = nnz - n * n_cols - m * (n_rows - n)

        # correction that will guarantee counts.sum() == nnz:
        counts[p] += corr

        if random:
            # randomize crow_indices by shuffling the sawteeth
            # sequence:
            perm = torch.randperm(n_rows, device=counts.device)
            counts[1:] = counts[1:][perm]

        # compute crow_indices:
        crow_indices = counts
        crow_indices.cumsum_(dim=0)
        return crow_indices.to(device=device)

    def genSparseCompressedTensor(self, size, nnz, *, layout, device, dtype, index_dtype, blocksize=(), dense_dims=0):
        from operator import mul
        from functools import reduce
        sparse_dim = 2
        assert all(size[d] > 0 for d in range(len(size))) or nnz == 0, 'invalid arguments'
        assert len(size) >= sparse_dim
        if blocksize:
            assert len(blocksize) == 2, (size, blocksize)
            assert size[-2 - dense_dims] % blocksize[0] == 0, (size, blocksize)
            assert size[-1 - dense_dims] % blocksize[1] == 0, (size, blocksize)
            blocksize0, blocksize1 = blocksize
        else:
            blocksize0 = blocksize1 = 1

        size = tuple(size)
        dense_size = size[(len(size) - dense_dims):]

        def random_sparse_compressed(n_compressed_dims, n_plain_dims, nnz):
            compressed_indices = self._make_crow_indices(n_compressed_dims, n_plain_dims, nnz, device=device, dtype=index_dtype)
            plain_indices = torch.zeros(nnz, dtype=index_dtype, device=device)
            for i in range(n_compressed_dims):
                count = compressed_indices[i + 1] - compressed_indices[i]
                plain_indices[compressed_indices[i]:compressed_indices[i + 1]], _ = torch.sort(
                    torch.randperm(n_plain_dims, dtype=index_dtype, device=device)[:count])
            low = -1 if dtype != torch.uint8 else 0
            high = 1 if dtype != torch.uint8 else 2
            values = make_tensor((nnz,) + blocksize + dense_size, device=device, dtype=dtype, low=low, high=high)
            return values, compressed_indices, plain_indices

        batch_shape = size[:-2 - dense_dims]
        n_batch = reduce(mul, batch_shape, 1)

        if layout in {torch.sparse_csr, torch.sparse_bsr}:
            n_compressed_dims, n_plain_dims = size[-2 - dense_dims] // blocksize0, size[-1 - dense_dims] // blocksize1
        else:
            n_compressed_dims, n_plain_dims = size[-1 - dense_dims] // blocksize1, size[-2 - dense_dims] // blocksize0
        blocknnz = nnz // (blocksize0 * blocksize1)
        sparse_tensors = [random_sparse_compressed(n_compressed_dims, n_plain_dims, blocknnz) for _ in range(n_batch)]
        sparse_tensors_it = map(list, zip(*sparse_tensors))

        values = torch.stack(next(sparse_tensors_it)).reshape(*batch_shape, blocknnz, *blocksize, *dense_size)
        compressed_indices = torch.stack(next(sparse_tensors_it)).reshape(*batch_shape, -1)
        plain_indices = torch.stack(next(sparse_tensors_it)).reshape(*batch_shape, -1)
        return torch.sparse_compressed_tensor(compressed_indices, plain_indices,
                                              values, size=size, dtype=dtype, layout=layout, device=device)

    def genSparseCSRTensor(self, size, nnz, *, device, dtype, index_dtype, dense_dims=0):
        return self.genSparseCompressedTensor(size, nnz, layout=torch.sparse_csr, device=device,
                                              dtype=dtype, index_dtype=index_dtype, blocksize=(), dense_dims=dense_dims)

    def genSparseCSCTensor(self, size, nnz, *, device, dtype, index_dtype, dense_dims=0):
        return self.genSparseCompressedTensor(size, nnz, layout=torch.sparse_csc, device=device,
                                              dtype=dtype, index_dtype=index_dtype, blocksize=(), dense_dims=0)

    def genSparseBSRTensor(self, size, blocksize, nnz, *, device, dtype, index_dtype, dense_dims=0):
        assert len(blocksize) == 2
        return self.genSparseCompressedTensor(size, nnz, layout=torch.sparse_bsr, device=device,
                                              dtype=dtype, index_dtype=index_dtype, blocksize=blocksize, dense_dims=dense_dims)

    def genSparseBSCTensor(self, size, blocksize, nnz, *, device, dtype, index_dtype, dense_dims=0):
        assert len(blocksize) == 2
        return self.genSparseCompressedTensor(size, nnz, layout=torch.sparse_bsc, device=device,
                                              dtype=dtype, index_dtype=index_dtype, blocksize=blocksize, dense_dims=dense_dims)

    def genSparseTensor(self, size, sparse_dim, nnz, is_uncoalesced, device, dtype):
        # Assert not given impossible combination, where the sparse dims have
        # empty numel, but nnz > 0 makes the indices containing values.
        assert all(size[d] > 0 for d in range(sparse_dim)) or nnz == 0, 'invalid arguments'

        v_size = [nnz] + list(size[sparse_dim:])
        v = make_tensor(v_size, device=device, dtype=dtype, low=-1, high=1)
        i = torch.rand(sparse_dim, nnz, device=device)
        i.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(i))
        i = i.to(torch.long)
        if is_uncoalesced:
            i1 = i[:, :(nnz // 2), ...]
            i2 = i[:, :((nnz + 1) // 2), ...]
            i = torch.cat([i1, i2], 1)
        x = torch.sparse_coo_tensor(i, v, torch.Size(size), dtype=dtype, device=device)

        if not is_uncoalesced:
            x = x.coalesce()
        else:
            # FIXME: `x` is a sparse view of `v`. Currently rebase_history for
            #        sparse views is not implemented, so this workaround is
            #        needed for inplace operations done on `x`, e.g., copy_().
            #        Remove after implementing something equivalent to CopySlice
            #        for sparse views.
            # NOTE: We do clone() after detach() here because we need to be able to change size/storage of x afterwards
            x = x.detach().clone()._coalesced_(False)
        return x, x._indices().clone(), x._values().clone()

    def generate_simple_inputs(self, layout,
                               device=None,
                               dtype=None,
                               index_dtype=None,
                               enable_batch=True,
                               enable_hybrid=True,
                               enable_zero_sized=True,
                               enable_non_contiguous_indices=True,
                               enable_non_contiguous_values=True,
                               enable_batch_variable_nse=False,
                               output_tensor=True,
                               patterns=None):
        """Generator of simple inputs for tensor constructors of the given layout.

        The generated tensor inputs have the following properties:

        - tensor shapes are minimal but not trivial
        - tensor values are sorted sequences for COO and CSR formats, e.g. [1, 2, 3, 4]
        - the generated tensors represent the same mathematical tensor for all layouts
        - the generated tensors include regular, zero-sized, and optionally, batched or/and hybrid tensors.
        - the generated tensors include contiguous or non-contiguous tensors both in indices and values

        If output_tensor is True, yield tensors with the given
        layout. Otherwise, yield inputs to the corresponding tensor
        constructors:

          - sparse compressed input is defined as
            (compressed_indices, plain_indices, values), dict(size=expected_size_from_shape_inference, device=device, dtype=dtype)

          - sparse COO input is defined as
            (indices, values), dict(size=expected_size_from_shape_inference, device=device, dtype=dtype)

          - strided input is defined as
            (values,), dict(device=device, dtype=dtype)
        """
        if index_dtype is None:
            index_dtype = torch.int64

        is_compressed_sparse_layout = layout in {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}

        if output_tensor:
            for args, kwargs in self.generate_simple_inputs(layout, device=device, dtype=dtype, index_dtype=index_dtype,
                                                            enable_batch=enable_batch, enable_hybrid=enable_hybrid,
                                                            enable_zero_sized=enable_zero_sized,
                                                            enable_non_contiguous_indices=enable_non_contiguous_indices,
                                                            enable_non_contiguous_values=enable_non_contiguous_values,
                                                            enable_batch_variable_nse=enable_batch_variable_nse,
                                                            output_tensor=False):
                if layout is torch.strided:
                    assert len(args) == 1
                    size = kwargs.pop('size', None)  # to ensure that a zero-sized tensor has the desired shape
                    assert size is not None
                    yield args[0].reshape(size)
                elif layout is torch.sparse_coo:
                    yield torch.sparse_coo_tensor(*args, **kwargs)
                elif is_compressed_sparse_layout:
                    kwargs.update(layout=layout)
                    yield torch.sparse_compressed_tensor(*args, **kwargs)
                else:
                    assert 0  # unreachable
            return

        def get_blockpattern(pattern, blocksize):
            basesize = pattern.shape
            assert basesize[0] % blocksize[0] == 0, (basesize, blocksize)
            assert basesize[1] % blocksize[1] == 0, (basesize, blocksize)
            blockpattern = pattern.reshape(-1,
                                           blocksize[0],
                                           basesize[1] // blocksize[1],
                                           blocksize[1]).transpose(-3, -2).any(-1).any(-1)
            block_ids = torch.arange(1, blockpattern.numel() + 1).reshape(blockpattern.shape)
            return (blockpattern != 0) * block_ids

        def get_sparse_data(pattern):
            basesize = pattern.shape
            assert len(basesize) == 2, basesize  # pattern is expected to be a matrix

            # We cannot use `torch.sparse_xyz_tensor(pattern)` to
            # compute the sparse layout indices and values because
            # generate_simple_inputs is used to generate the inputs to
            # test `torch.sparse_xyz_tensor` factory functions, so
            # we'll compute the indices and values independently of
            # the factory functions.

            indices = torch.where(pattern != 0)
            coo_indices = torch.stack(indices)
            crow_indices = torch.zeros(basesize[0] + 1, dtype=torch.int64)
            crow_indices[1:] = torch.cumsum(coo_indices[0].bincount(minlength=basesize[0]), 0)
            col_indices = coo_indices[1]
            strided_values = torch.zeros(basesize, dtype=torch.int64)

            # the property of `values == range(1, 1+nnz)` is used in
            # get_sparse_data_with_block to relate BSR and BSC values,
            # so, don't change the following line:
            values = torch.arange(1, 1 + len(indices[0]), dtype=torch.int64)
            strided_values[indices] = values

            indices_T = torch.where(pattern.transpose(0, 1) != 0)
            coo_indices_T = torch.stack(indices_T)
            ccol_indices = torch.zeros(basesize[1] + 1, dtype=torch.int64)
            ccol_indices[1:] = torch.cumsum(coo_indices_T[0].bincount(minlength=basesize[1]), 0)
            row_indices = coo_indices_T[1]
            csc_values = strided_values.transpose(0, 1)[indices_T]

            return {torch.sparse_coo: (coo_indices, values),
                    torch.sparse_csr: (crow_indices, col_indices, values),
                    torch.sparse_csc: (ccol_indices, row_indices, csc_values),
                    torch.strided: (strided_values,)}

        def get_sparse_data_with_block(pattern, blocksize):
            nonblock_data = get_sparse_data(pattern)
            blockpattern = get_blockpattern(pattern, blocksize)
            block_data = get_sparse_data(blockpattern)

            strided_values = nonblock_data[torch.strided][0]
            block_indices = block_data[torch.sparse_coo][0]
            bsr_values = torch.stack([strided_values[bi * blocksize[0]:(bi + 1) * blocksize[0],
                                                     bj * blocksize[1]:(bj + 1) * blocksize[1]]
                                      for bi, bj in block_indices.transpose(0, 1)])

            # here we use the property `values == range(1, 1+nnz)` and
            # `values` relation to `csc_values` (see get_sparse_data)
            # to get BSC blocks via reordering the BSR blocks:
            bsc_values = bsr_values[block_data[torch.sparse_csc][2] - 1]

            return {torch.sparse_bsr: (*block_data[torch.sparse_csr][:2], bsr_values),
                    torch.sparse_bsc: (*block_data[torch.sparse_csc][:2], bsc_values),
                    **nonblock_data}

        def get_batch_sparse_data(pattern, blocksize):
            size = pattern.shape
            if len(size) <= 2:  # non-batch
                return get_sparse_data_with_block(pattern, blocksize)

            # batch data is created recursively:
            batch_data = {}
            for i, item in enumerate(pattern):
                for layout, d in get_batch_sparse_data(item, blocksize).items():
                    target = batch_data.get(layout)
                    if layout is torch.sparse_coo:
                        # a "batch COO" means a COO with the leading
                        # sparse dimensions interpreted as batch
                        # dimensions
                        ext_coo_indices1 = torch.cat((torch.full((1, len(d[1])), i, dtype=torch.int64), d[0]))
                        if target is None:
                            target = batch_data[layout] = (ext_coo_indices1, d[1])
                        else:
                            target[0].set_(torch.cat((target[0], ext_coo_indices1), 1))
                            target[1].set_(torch.cat((target[1], d[1])))
                    else:
                        if target is None:
                            target = batch_data[layout] = tuple(d[j].unsqueeze(0) for j in range(len(d)))
                        else:
                            for j in range(len(d)):
                                target[j].set_(torch.cat((target[j], d[j].unsqueeze(0))))
            return batch_data

        def generate_values(base, densesize):
            """Generates a tensor of shape densesize with values equal to

              base + i_1 * 10^0 + ... + i_d * 10^{d - 1}

            at indices i_1, ..., i_d (with 0 <= i_j < densesize[j] for any 1 <= j <=
            len(densesize))

            This mapping produces unique values as long as
            densesize[i] < 10 for all i in range(len(densesize)).
            """

            if not densesize:
                return base
            if not isinstance(base, int) and base.ndim > 0:
                return torch.stack([generate_values(b, densesize) for b in base])
            if base == 0:
                return torch.zeros(densesize, dtype=torch.int64)
            r = torch.arange(densesize[0], dtype=torch.int64)
            for i, d in enumerate(densesize[1:]):
                y = torch.arange(d, dtype=torch.int64) * (10 ** (i + 1))
                r = r[..., None] + y[None, ...]
            r.add_(base)
            return r

        if patterns is None:
            # A pattern is a 3-tuple with the following items:
            #
            # - a list of integers with the depth of two or more. The
            #   integers define the sparsity patterns of the generated
            #   inputs: zero values correspond to unspecified
            #   elements/blocks, and non-zero values to the specified
            #   elements.
            #
            #   For debugging convenience, the elements with the same
            #   value typically belong to the same block. However, it
            #   is not a hard requirement: as long as the shape of a
            #   pattern divides with block sizes, the pattern will be
            #   a valid one.
            #
            #   If the depth of the list is larger than two, inputs
            #   with batch dimensions will be generated.
            #
            # - a list of 2-tuples of block sizes, used to generate
            #   BSR/BSC tensors with various block size parameters
            #
            # - a list of tuples of dense dimensions, used to generate
            #   hybrid tensors with various dense dimensions
            #
            patterns = [
                # a simple 3 x 2 tensor: non-hybrid, hybrid with 1 and 2 dense dimensions
                ([[1, 2, 0],
                  [1, 0, 3]], [(2, 1), (1, 3)], [(), (2,), (4, 5)]),
                # 2 x 3 batch of 3 x 2 tensors: non-hybrid and hybrid with 2 dense dimensions
                ([[[[1, 2, 0],
                    [1, 0, 3]],
                   [[1, 2, 3],
                    [1, 0, 0]],
                   [[1, 0, 0],
                    [1, 2, 3]]],
                  [[[0, 2, 0],
                    [1, 2, 3]],
                   [[1, 0, 3],
                    [1, 2, 0]],
                   [[1, 2, 3],
                    [0, 2, 0]]]], [(2, 1), (2, 3)], [(), (2,)]),
                # tensor with non-trivial blocksize
                ([[0, 1, 0, 2, 0, 2],
                  [0, 1, 0, 0, 2, 0],
                  [3, 3, 3, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 5, 0, 6, 6, 6],
                  [5, 0, 5, 6, 6, 6],
                  [0, 0, 0, 0, 8, 8],
                  [7, 7, 7, 0, 8, 8]], [(2, 3)], [(), (4, 5)]),
                # batch tensor with variable NSE
                # Requires https://github.com/pytorch/pytorch/pull/84843 or similar.
                ([[[1, 2],
                   [3, 4]],
                  [[1, 0],
                   [0, 0]]], [(1, 1)], ([()] if enable_batch_variable_nse else []))]

        def non_contiguous_copy(t, dim=-1, offset=0):
            # return a copy of t that is non-contiguous along the
            # given dimension and with the given storage offset
            self.assertTrue(t.is_contiguous())
            if dim < 0:
                dim = dim + t.ndim
            assert dim >= 0 and dim < t.ndim
            step = max(2, offset + 1)
            tmp = torch.zeros((*t.shape[:dim], t.shape[dim] * step, *t.shape[dim + 1:]), dtype=t.dtype, device=t.device)
            dim_slices = (*((slice(None),) * dim), slice(offset, None, step))
            r = tmp[dim_slices].copy_(t)
            self.assertFalse(r.is_contiguous())
            self.assertEqual(t, r)
            return r

        # the main loop of the method:
        for pattern, blocksizes, densesizes in patterns:
            if not enable_hybrid:
                densesizes = [s for s in densesizes if not s]
            if not (densesizes and blocksizes):
                continue
            pattern = torch.tensor(pattern, dtype=torch.int64)
            if not enable_batch and pattern.ndim > 2:
                continue
            for blocksize in blocksizes:
                data = get_batch_sparse_data(pattern, blocksize)[layout]
                for densesize in densesizes:
                    indices = [a.to(device=device, dtype=index_dtype) for a in data[:-1]]
                    values = generate_values(data[-1], densesize).to(device=device, dtype=dtype)
                    yield (*indices, values), dict(device=device, dtype=dtype,
                                                   size=pattern.shape + densesize)

                    if enable_non_contiguous_indices and pattern.ndim > 2:
                        # sparse compressed indices can be sliced only along batch dimensions
                        for (dim, offset) in {(0, 1), (-2, 0)}:
                            indices_copy = [non_contiguous_copy(a, dim=dim, offset=offset) for a in indices]
                            yield (*indices_copy, values), dict(device=device, dtype=dtype,
                                                                size=pattern.shape + densesize)

                            if enable_non_contiguous_values:
                                values_copy = non_contiguous_copy(values, dim=-1, offset=1)
                                yield (*indices_copy, values_copy), dict(device=device, dtype=dtype,
                                                                         size=pattern.shape + densesize)

                    if enable_non_contiguous_values:
                        values_copy = non_contiguous_copy(values, dim=-1, offset=1)
                        yield (*indices, values_copy), dict(device=device, dtype=dtype,
                                                            size=pattern.shape + densesize)

        # zero-sized tensor inputs, non-batch, non-hybrid/hybrid
        if enable_zero_sized:
            for basesize, blocksizes, densesizes in [
                    ((2, 0), [(1, 2)], [(), (2,), (2, 3)] if enable_hybrid else [()]),
                    ((0, 2), [(1, 2), (2, 1), (3, 2)], [()]),
                    ((0, 0), [(1, 2)], [()]),
            ]:
                for blocksize in blocksizes:
                    for densesize in densesizes:
                        if layout == torch.strided:
                            indices = ()
                            values = torch.empty((basesize + densesize), device=device, dtype=dtype)
                        elif layout == torch.sparse_coo:
                            indices = (torch.empty(len(basesize), 0, device=device, dtype=index_dtype),)
                            values = torch.empty((0, *densesize), device=device, dtype=dtype)
                        elif layout == torch.sparse_csr:
                            crow_indices = torch.tensor([0] * (basesize[0] + 1), device=device, dtype=index_dtype)
                            col_indices = torch.empty(0, device=device, dtype=index_dtype)
                            indices = (crow_indices, col_indices)
                            values = torch.empty((0, *densesize), device=device, dtype=dtype)
                        elif layout == torch.sparse_csc:
                            ccol_indices = torch.tensor([0] * (basesize[1] + 1), device=device, dtype=index_dtype)
                            row_indices = torch.empty(0, device=device, dtype=index_dtype)
                            indices = (ccol_indices, row_indices)
                            values = torch.empty((0, *densesize), device=device, dtype=dtype)
                        elif layout == torch.sparse_bsr:
                            crow_indices = torch.tensor([0] * (basesize[0] // blocksize[0] + 1), device=device, dtype=index_dtype)
                            col_indices = torch.empty(0, device=device, dtype=index_dtype)
                            indices = (crow_indices, col_indices)
                            values = torch.empty((0, *blocksize, *densesize), device=device, dtype=dtype)
                        elif layout == torch.sparse_bsc:
                            ccol_indices = torch.tensor([0] * (basesize[1] // blocksize[1] + 1), device=device, dtype=index_dtype)
                            row_indices = torch.empty(0, device=device, dtype=index_dtype)
                            indices = (ccol_indices, row_indices)
                            values = torch.empty((0, *blocksize, *densesize), device=device, dtype=dtype)
                        else:
                            assert 0  # unreachable
                        yield (*indices, values), dict(device=device, dtype=dtype, size=basesize + densesize)

    def safeToDense(self, t):
        # coalesce is only implemented for COO
        if t.layout == torch.sparse_coo:
            t = t.coalesce()
        return t.to_dense()

    # Compares a torch function with a reference function for a given sample input (object of SampleInput)
    # Note: only values are compared, type comparison is not done here
    def compare_with_reference(self, torch_fn, ref_fn, sample_input, **kwargs):
        numpy_sample = sample_input.numpy()
        n_inp, n_args, n_kwargs = numpy_sample.input, numpy_sample.args, numpy_sample.kwargs
        t_inp, t_args, t_kwargs = sample_input.input, sample_input.args, sample_input.kwargs

        actual = torch_fn(t_inp, *t_args, **t_kwargs)
        expected = ref_fn(n_inp, *n_args, **n_kwargs)

        self.assertEqual(actual, expected, exact_device=False, **kwargs)

    # Compares the given Torch and NumPy functions on the given tensor-like object.
    # NOTE: both torch_fn and np_fn should be functions that take a single
    #   tensor (array). If the torch and/or NumPy function require additional
    #   arguments then wrap the function in a lambda or pass a partial function.
    # TODO: add args/kwargs for passing to assertEqual (e.g. rtol, atol)
    def compare_with_numpy(self, torch_fn, np_fn, tensor_like,
                           device=None, dtype=None, **kwargs):
        assert TEST_NUMPY

        if isinstance(tensor_like, torch.Tensor):
            assert device is None
            assert dtype is None
            t_cpu = tensor_like.detach().cpu()
            if t_cpu.dtype is torch.bfloat16:
                t_cpu = t_cpu.float()
            a = t_cpu.numpy()
            t = tensor_like
        else:
            d = copy.copy(torch_to_numpy_dtype_dict)
            d[torch.bfloat16] = np.float32
            a = np.array(tensor_like, dtype=d[dtype])
            t = torch.tensor(tensor_like, device=device, dtype=dtype)

        np_result = np_fn(a)
        torch_result = torch_fn(t).cpu()

        # Converts arrays to tensors
        if isinstance(np_result, np.ndarray):
            try:
                np_result = torch.from_numpy(np_result)
            except Exception:
                # NOTE: copying an array before conversion is necessary when,
                #   for example, the array has negative strides.
                np_result = torch.from_numpy(np_result.copy())
            if t.dtype is torch.bfloat16 and torch_result.dtype is torch.bfloat16 and np_result.dtype is torch.float:
                torch_result = torch_result.to(torch.float)

        self.assertEqual(np_result, torch_result, **kwargs)

    def assertEqualIgnoreType(self, *args, **kwargs) -> None:
        # If you are seeing this function used, that means test is written wrongly
        # and deserves detailed investigation
        return self.assertEqual(*args, exact_dtype=False, **kwargs)

    def assertEqualBroadcasting(self, x, y, *args, **kwargs) -> None:
        r"""Tests if tensor x equals to y, if y to be broadcast to x.shape.
        """
        if not isinstance(y, Iterable):
            # int, float, etc. or different shape tensors
            y = torch.ones_like(x) * y
        if not isinstance(y, torch.Tensor):
            # iterable, but not a tensor
            y = torch.ones_like(x) * torch.tensor(y)
        return self.assertEqual(x, y, *args, **kwargs)

    def assertEqual(
            self,
            x,
            y,
            msg: Optional[Union[str, Callable[[str], str]]] = None,
            *,
            atol: Optional[float] = None,
            rtol: Optional[float] = None,
            equal_nan=True,
            exact_dtype=True,
            # TODO: default this to True
            exact_device=False,
            exact_layout=False,
            exact_stride=False,
            exact_is_coalesced=False
    ):
        # Hide this function from `pytest`'s traceback
        __tracebackhide__ = True

        # numpy's dtypes are a superset of what PyTorch supports. In case we encounter an unsupported dtype, we fall
        # back to an elementwise comparison. Note that this has to happen here and not for example in
        # `TensorOrArrayPair`, since at that stage we can no longer split the array into its elements and perform
        # multiple comparisons.
        if any(
            isinstance(input, np.ndarray) and not has_corresponding_torch_dtype(input.dtype) for input in (x, y)
        ):
            def to_list(input):
                return input.tolist() if isinstance(input, (torch.Tensor, np.ndarray)) else list(input)

            x = to_list(x)
            y = to_list(y)
        # When comparing a sequence of numbers to a tensor, we need to convert the sequence to a tensor here.
        # Otherwise, the pair origination of `are_equal` will fail, because the sequence is recognized as container
        # that should be checked elementwise while the tensor is not.
        elif isinstance(x, torch.Tensor) and isinstance(y, Sequence):
            y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
        elif isinstance(x, Sequence) and isinstance(y, torch.Tensor):
            x = torch.as_tensor(x, dtype=y.dtype, device=y.device)

        # If x or y are tensors and nested then we unbind them to a list of tensors this should allow us to compare
        # a nested tensor to a nested tensor and a nested tensor to a list of expected tensors
        if isinstance(x, torch.Tensor) and x.is_nested:
            x = x.unbind()
        if isinstance(y, torch.Tensor) and y.is_nested:
            y = y.unbind()

        error_metas = not_close_error_metas(
            x,
            y,
            pair_types=(
                NonePair,
                RelaxedBooleanPair,
                RelaxedNumberPair,
                TensorOrArrayPair,
                TypedStoragePair,
                StringPair,
                SetPair,
                TypePair,
                ObjectPair,
            ),
            sequence_types=(
                Sequence,
                Sequential,
                ModuleList,
                ParameterList,
                ScriptList,
                torch.utils.data.dataset.Subset,
            ),
            mapping_types=(Mapping, ModuleDict, ParameterDict, ScriptDict),
            rtol=rtol,
            rtol_override=self.rel_tol,
            atol=atol,
            atol_override=self.precision,
            equal_nan=equal_nan,
            check_device=exact_device,
            check_dtype=exact_dtype,
            check_layout=exact_layout,
            check_stride=exact_stride,
            check_is_coalesced=exact_is_coalesced,
        )

        if error_metas:
            # See [ErrorMeta Cycles]
            error_metas = [error_metas]
            # TODO: compose all metas into one AssertionError
            raise error_metas.pop()[0].to_error(
                # This emulates unittest.TestCase's behavior if a custom message passed and
                # TestCase.longMessage (https://docs.python.org/3/library/unittest.html#unittest.TestCase.longMessage)
                # is True (default)
                (lambda generated_msg: f"{generated_msg}\n{msg}") if isinstance(msg, str) and self.longMessage else msg
            )

    def assertNotEqual(self, x, y, msg: Optional[str] = None, *,                                       # type: ignore[override]
                       atol: Optional[float] = None, rtol: Optional[float] = None, **kwargs) -> None:
        with self.assertRaises(AssertionError, msg=msg):
            self.assertEqual(x, y, msg, atol=atol, rtol=rtol, **kwargs)

    def assertEqualTypeString(self, x, y) -> None:
        # This API is used simulate deprecated x.type() == y.type()
        self.assertEqual(x.device, y.device)
        self.assertEqual(x.dtype, y.dtype)
        self.assertEqual(x.is_sparse, y.is_sparse)

    def assertObjectIn(self, obj: Any, iterable: Iterable[Any]) -> None:
        for elem in iterable:
            if id(obj) == id(elem):
                return
        raise AssertionError("object not found in iterable")

    # Reimplemented to provide special behavior when
    # _ignore_not_implemented_error is True
    def assertRaises(self, expected_exception, *args, **kwargs):
        if self._ignore_not_implemented_error:
            context: Optional[AssertRaisesContextIgnoreNotImplementedError] = \
                AssertRaisesContextIgnoreNotImplementedError(expected_exception, self)  # type: ignore[call-arg]
            try:
                return context.handle('assertRaises', args, kwargs)  # type: ignore[union-attr]
            finally:
                # see https://bugs.python.org/issue23890
                context = None
        else:
            return super().assertRaises(expected_exception, *args, **kwargs)

    # Reimplemented to provide special behavior when
    # _ignore_not_implemented_error is True
    def assertRaisesRegex(self, expected_exception, expected_regex, *args, **kwargs):
        # Verifies that an exception with the type expected_exception and message
        # matching the regular expression defined by expected_regex is thrown.
        # If the test is instantiated for a non-native device type (like XLA)
        # then the message is not validated.

        # Checks whether the test is instantiated for a device type by testing
        # if the test class has defined the device_type attribute and,
        # if so, tests whether the instantiated device type is native or not
        if hasattr(self, 'device_type') and self.device_type not in NATIVE_DEVICES and self.device_type != "mps":  # type: ignore[attr-defined]
            # empty string matches any string
            expected_regex = ''

        if self._ignore_not_implemented_error:
            context = AssertRaisesContextIgnoreNotImplementedError(  # type: ignore[call-arg]
                expected_exception, self, expected_regex)
            return context.handle('assertRaisesRegex', args, kwargs)  # type: ignore[attr-defined]
        else:
            return super().assertRaisesRegex(expected_exception, expected_regex, *args, **kwargs)

    # Verifies that no unraisable exceptions are raised by callable.  Unlike regular
    # exceptions, these do not actually propagate to the caller and are
    # suppressed.  We must test for them specially.
    def assertNoUnraisable(self, callable, *args, **kwargs):
        raised = None

        def record_unraisable(unraisable):
            nonlocal raised
            raised = unraisable

        # Disable GC when running the callable to prevent spurious flakiness
        # from unlucky GCs inside the callable
        prev = gc.isenabled()
        gc.disable()
        try:
            with unittest.mock.patch("sys.unraisablehook", record_unraisable):
                callable(*args, **kwargs)
        finally:
            if prev:
                gc.enable()

        self.assertIsNone(raised)

    # TODO: Support context manager interface
    # NB: The kwargs forwarding to callable robs the 'subname' parameter.
    # If you need it, manually apply your callable in a lambda instead.
    def assertExpectedRaises(self, exc_type, callable, *args, **kwargs):
        subname = None
        if 'subname' in kwargs:
            subname = kwargs['subname']
            del kwargs['subname']
        try:
            callable(*args, **kwargs)
        except exc_type as e:
            self.assertExpected(str(e), subname)
            return
        # Don't put this in the try block; the AssertionError will catch it
        self.fail(msg="Did not raise when expected to")

    def assertNotWarn(self, callable, msg=''):
        r"""
        Test if :attr:`callable` does not raise a warning.
        """
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            with set_warn_always_context(True):
                callable()
            self.assertTrue(len(ws) == 0, msg)

    @contextmanager
    def assertWarnsOnceRegex(self, category, regex=''):
        """Context manager for code that *must always* warn

        This filters expected warnings from the test and fails if
        the expected warning is not caught. It uses set_warn_always() to force
        TORCH_WARN_ONCE to behave like TORCH_WARN
        """
        pattern = re.compile(regex)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            with set_warn_always_context(True):
                yield
            if len(ws) == 0:
                self.fail('no warning caught')
            self.assertTrue(any(type(w.message) is category for w in ws))
            self.assertTrue(
                any(re.match(pattern, str(w.message)) for w in ws),
                f'{pattern}, {[w.message for w in ws if type(w.message) is category]}')

    def assertExpected(self, s, subname=None):
        r"""
        Test that a string matches the recorded contents of a file
        derived from the name of this test and subname.  This file
        is placed in the 'expect' directory in the same directory
        as the test script. You can automatically update the recorded test
        output using --accept.

        If you call this multiple times in a single function, you must
        give a unique subname each time.
        """
        if not isinstance(s, str):
            raise TypeError("assertExpected is strings only")

        def remove_prefix(text, prefix):
            if text.startswith(prefix):
                return text[len(prefix):]
            return text
        # NB: we take __file__ from the module that defined the test
        # class, so we place the expect directory where the test script
        # lives, NOT where test/common_utils.py lives.  This doesn't matter in
        # PyTorch where all test scripts are in the same directory as
        # test/common_utils.py, but it matters in onnx-pytorch
        module_id = self.__class__.__module__
        munged_id = remove_prefix(self.id(), module_id + ".")
        test_file = os.path.realpath(sys.modules[module_id].__file__)
        expected_file = os.path.join(os.path.dirname(test_file),
                                     "expect",
                                     munged_id)

        subname_output = ""
        if subname:
            expected_file += "-" + subname
            subname_output = f" ({subname})"
        expected_file += ".expect"
        expected = None

        def accept_output(update_type):
            print(f"Accepting {update_type} for {munged_id}{subname_output}:\n\n{s}")
            with open(expected_file, 'w') as f:
                # Adjust for producer_version, leave s unmodified
                s_tag = re.sub(r'(producer_version): "[0-9.]*"',
                               r'\1: "CURRENT_VERSION"', s)
                f.write(s_tag)

        try:
            with open(expected_file) as f:
                expected = f.read()
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
            elif expecttest.ACCEPT:
                return accept_output("output")
            else:
                raise RuntimeError(
                      f"I got this output for {munged_id}{subname_output}:\n\n{s}\n\n"
                      "No expect file exists; to accept the current output, run:\n"
                      f"python {__main__.__file__} {munged_id} --accept") from None

        # a hack for JIT tests
        if IS_WINDOWS:
            expected = re.sub(r'CppOp\[(.+?)\]', 'CppOp[]', expected)
            s = re.sub(r'CppOp\[(.+?)\]', 'CppOp[]', s)

        # Adjust for producer_version
        expected = expected.replace(
            'producer_version: "CURRENT_VERSION"',
            f'producer_version: "{torch.onnx.producer_version}"'
        )
        if expecttest.ACCEPT:
            if expected != s:
                return accept_output("updated output")
        else:
            if hasattr(self, "assertMultiLineEqual"):
                # Python 2.7 only
                # NB: Python considers lhs "old" and rhs "new".
                self.assertMultiLineEqual(expected, s)
            else:
                self.assertEqual(s, expected)

    def assertExpectedStripMangled(self, s, subname=None):
        s = re.sub(r'__torch__[^ ]+', '', s)
        self.assertExpected(s, subname)

    def assertGreaterAlmostEqual(self, first, second, places=None, msg=None, delta=None):
        """Assert that ``first`` is greater than or almost equal to ``second``.

        The equality of ``first`` and ``second`` is determined in a similar way to
        the ``assertAlmostEqual`` function of the standard library.
        """
        if delta is not None and places is not None:
            raise TypeError("specify delta or places not both")

        if first >= second:
            return

        diff = second - first
        if delta is not None:
            if diff <= delta:
                return

            standardMsg = f"{first} not greater than or equal to {second} within {delta} delta"
        else:
            if places is None:
                places = 7

            if round(diff, places) == 0:
                return

            standardMsg = f"{first} not greater than or equal to {second} within {places} places"

        msg = self._formatMessage(msg, standardMsg)
        raise self.failureException(msg)

    def assertAtenOp(self, onnx_model, operator, overload_name=""):
        all_aten_nodes = [p for p in onnx_model.graph.node
                          if p.op_type == "ATen" and p.domain == "org.pytorch.aten"]
        self.assertTrue(all_aten_nodes)

        for op in all_aten_nodes:
            attrs = {attr.name: attr.s.decode() for attr in op.attribute}
            if attrs.get("operator") == operator:
                break

        self.assertEqual(attrs["operator"], operator)
        self.assertEqual(attrs.get("overload_name", ""), overload_name)

    def check_nondeterministic_alert(self, fn, caller_name, should_alert=True):
        '''Checks that an operation produces a nondeterministic alert when
        expected while `torch.use_deterministic_algorithms(True)` is set.

        Args:
          fn (callable): Function to check for a nondeterministic alert

          caller_name (str): Name of the operation that produces the
              nondeterministic alert. This name is expected to appear at the
              beginning of the error/warning message.

          should_alert (bool, optional): If True, then the check will only pass
              if calling `fn` produces a nondeterministic error/warning with the
              expected message. If False, then the check will only pass if
              calling `fn` does not produce an error. Default: `True`.
        '''

        alert_message = '^' + caller_name + ' does not have a deterministic implementation, but you set'

        # Check that errors are thrown correctly
        with DeterministicGuard(True):
            if should_alert:
                with self.assertRaisesRegex(
                        RuntimeError,
                        alert_message,
                        msg='expected a non-deterministic error, but it was not raised'):
                    fn()

            else:
                # If a nondeterministic error is not expected, make sure
                # that it is not raised
                try:
                    fn()
                except RuntimeError as e:
                    if 'does not have a deterministic implementation' in str(e):
                        self.fail(
                            'did not expect non-deterministic error message, '
                            + 'but got one anyway: "' + str(e) + '"')
                    # Reraise exceptions unrelated to nondeterminism
                    raise

        # Check that warnings are thrown correctly
        with DeterministicGuard(True, warn_only=True):
            if should_alert:
                with self.assertWarnsRegex(
                        UserWarning,
                        alert_message):
                    fn()
            else:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    fn()
                    for warning in w:
                        if isinstance(warning, UserWarning):
                            self.assertTrue(re.search(alert_message, str(warning)) is None)

    # run code in subprocess and capture exceptions.
    @staticmethod
    def run_process_no_exception(code, env=None):
        import subprocess

        popen = subprocess.Popen(
            [sys.executable, '-c', code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env)
        (stdout, stderr) = popen.communicate()
        return (stdout, stderr)

    # returns captured stderr
    @staticmethod
    def runWithPytorchAPIUsageStderr(code):
        env = os.environ.copy()
        env["PYTORCH_API_USAGE_STDERR"] = "1"
        # remove CI flag since this is a wrapped test process.
        # CI flag should be set in the parent process only.
        if "CI" in env.keys():
            del env["CI"]
        (stdout, stderr) = TestCase.run_process_no_exception(code, env=env)
        return stderr.decode('ascii')


def download_file(url, binary=True):
    from urllib.parse import urlsplit
    from urllib import request, error

    filename = os.path.basename(urlsplit(url)[2])
    data_dir = get_writable_path(os.path.join(os.path.dirname(__file__), 'data'))
    path = os.path.join(data_dir, filename)

    if os.path.exists(path):
        return path
    try:
        data = request.urlopen(url, timeout=15).read()
        with open(path, 'wb' if binary else 'w') as f:
            f.write(data)
        return path
    except error.URLError as e:
        msg = f"could not download test file '{url}'"
        warnings.warn(msg, RuntimeWarning)
        raise unittest.SkipTest(msg) from e

def find_free_port():
    """
    Finds an available port and returns that port number.

    NOTE: If this function is being used to allocate a port to Store (or
    indirectly via init_process_group or init_rpc), it should be used
    in conjuction with the `retry_on_connect_failures` decorator as there is a potential
    race condition where the allocated port may become unavailable before it can be used
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('localhost', 0))
        _, port = sock.getsockname()
        return port

# Errors that we can get in c10d initialization for which we should retry tests for.
ADDRESS_IN_USE = "Address already in use"
CONNECT_TIMEOUT = "connect() timed out."

def retry_on_connect_failures(func=None, connect_errors=(ADDRESS_IN_USE)):
    """Reruns a test if the test returns a RuntimeError and the exception
    contains one of the strings in connect_errors."""
    # This if block is executed when using this function as a decorator with arguments.
    if func is None:
        return partial(retry_on_connect_failures, connect_errors=connect_errors)

    @wraps(func)
    def wrapper(*args, **kwargs):
        n_retries = 10
        tries_remaining = n_retries
        while True:
            try:
                return func(*args, **kwargs)
            except RuntimeError as error:
                if any(connect_error in str(error) for connect_error in connect_errors):
                    tries_remaining -= 1
                    if tries_remaining == 0:
                        raise RuntimeError(f"Failing after {n_retries} retries with error: {str(error)}") from error
                    time.sleep(random.random())
                    continue
                raise
    return wrapper


# Decorator to retry upon certain Exceptions.
def retry(ExceptionToCheck, tries=3, delay=3, skip_after_retries=False):
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except ExceptionToCheck as e:
                    msg = "%s, Retrying in %d seconds..." % (str(e), mdelay)
                    print(msg)
                    time.sleep(mdelay)
                    mtries -= 1
            try:
                return f(*args, **kwargs)
            except ExceptionToCheck as e:
                raise unittest.SkipTest(f"Skipping after {tries} consecutive {str(e)}") from e if skip_after_retries else e
        return f_retry  # true decorator
    return deco_retry


# FIXME: modernize these to be consistent with make_tensor
#   and review including them in torch.testing
# Methods for matrix generation

def random_square_matrix_of_rank(l, rank, dtype=torch.double, device='cpu'):
    assert rank <= l
    A = torch.randn(l, l, dtype=dtype, device=device)
    u, s, vh = torch.linalg.svd(A, full_matrices=False)
    for i in range(l):
        if i >= rank:
            s[i] = 0
        elif s[i] == 0:
            s[i] = 1
    return (u * s.to(dtype).unsqueeze(-2)) @ vh

def random_well_conditioned_matrix(*shape, dtype, device, mean=1.0, sigma=0.001):
    """
    Returns a random rectangular matrix (batch of matrices)
    with singular values sampled from a Gaussian with
    mean `mean` and standard deviation `sigma`.
    The smaller the `sigma`, the better conditioned
    the output matrix is.
    """
    primitive_dtype = {
        torch.float: torch.float,
        torch.double: torch.double,
        torch.cfloat: torch.float,
        torch.cdouble: torch.double
    }
    x = torch.rand(shape, dtype=dtype, device=device)
    m = x.size(-2)
    n = x.size(-1)
    u, _, vh = torch.linalg.svd(x, full_matrices=False)
    s = (torch.randn(*(shape[:-2] + (min(m, n),)), dtype=primitive_dtype[dtype], device=device) * sigma + mean) \
        .sort(-1, descending=True).values.to(dtype)
    return (u * s.unsqueeze(-2)) @ vh

# Returns a noncontiguous (tensor with the same shape and values as t
# The noncontiguous tensor is constructed such that elements in the innermost
#   dimension are separated by zeros or (whenever possible) nans
# TODO: consider more complicated noncontiguity schemes
def noncontiguous_like(t):
    # Short-circuits if t is already noncontiguous
    if not t.is_contiguous():
        return t

    # Choose a "weird" value that won't be accessed
    if t.dtype.is_floating_point or t.dtype.is_complex:
        value = math.nan
    elif t.dtype == torch.bool:
        value = True
    else:
        value = 12

    result = t.new_empty(t.shape + (2,))
    result[..., 0] = value
    result[..., 1] = t.detach()
    result = result[..., 1]
    result.requires_grad_(t.requires_grad)
    return result

# TODO: remove this (prefer make_symmetric_matrices below)
def random_symmetric_matrix(l, *batches, **kwargs):
    dtype = kwargs.get('dtype', torch.double)
    device = kwargs.get('device', 'cpu')
    A = torch.randn(*(batches + (l, l)), dtype=dtype, device=device)
    A = (A + A.mT).div_(2)
    return A

# Creates a symmetric matrix or batch of symmetric matrices
# Shape must be a square matrix or batch of square matrices
def make_symmetric_matrices(*shape, device, dtype):
    assert shape[-1] == shape[-2]
    t = make_tensor(shape, device=device, dtype=dtype)
    t = (t + t.mT).div_(2)
    return t

def random_hermitian_matrix(l, *batches, **kwargs):
    dtype = kwargs.get('dtype', torch.double)
    device = kwargs.get('device', 'cpu')
    A = torch.randn(*(batches + (l, l)), dtype=dtype, device=device)
    A = (A + A.mH).div_(2)
    return A


def random_symmetric_psd_matrix(l, *batches, **kwargs):
    """
    Returns a batch of random symmetric positive-semi-definite matrices.
    The shape of the result is batch_dims + (matrix_size, matrix_size)
    The following example creates a tensor of size 2 x 4 x 3 x 3
    >>> # xdoctest: +SKIP("undefined variables")
    >>> matrices = random_symmetric_psd_matrix(3, 2, 4, dtype=dtype, device=device)
    """
    dtype = kwargs.get('dtype', torch.double)
    device = kwargs.get('device', 'cpu')
    A = torch.randn(*(batches + (l, l)), dtype=dtype, device=device)
    return A @ A.mT


def random_hermitian_psd_matrix(matrix_size, *batch_dims, dtype=torch.double, device='cpu'):
    """
    Returns a batch of random Hermitian positive-semi-definite matrices.
    The shape of the result is batch_dims + (matrix_size, matrix_size)
    The following example creates a tensor of size 2 x 4 x 3 x 3
    >>> # xdoctest: +SKIP("undefined variables")
    >>> matrices = random_hermitian_psd_matrix(3, 2, 4, dtype=dtype, device=device)
    """
    A = torch.randn(*(batch_dims + (matrix_size, matrix_size)), dtype=dtype, device=device)
    return A @ A.mH


# TODO: remove this (prefer make_symmetric_pd_matrices below)
def random_symmetric_pd_matrix(matrix_size, *batch_dims, **kwargs):
    dtype = kwargs.get('dtype', torch.double)
    device = kwargs.get('device', 'cpu')
    A = torch.randn(*(batch_dims + (matrix_size, matrix_size)),
                    dtype=dtype, device=device)
    return torch.matmul(A, A.mT) \
        + torch.eye(matrix_size, dtype=dtype, device=device) * 1e-5


# Creates a symmetric positive-definite matrix or batch of
#   such matrices
def make_symmetric_pd_matrices(*shape, device, dtype):
    assert shape[-1] == shape[-2]
    t = make_tensor(shape, device=device, dtype=dtype)
    i = torch.eye(shape[-1], device=device, dtype=dtype) * 1e-5
    return t @ t.mT + i

def random_hermitian_pd_matrix(matrix_size, *batch_dims, dtype, device):
    """
    Returns a batch of random Hermitian positive-definite matrices.
    The shape of the result is batch_dims + (matrix_size, matrix_size)
    The following example creates a tensor of size 2 x 4 x 3 x 3
    >>> # xdoctest: +SKIP("undefined variables")
    >>> matrices = random_hermitian_pd_matrix(3, 2, 4, dtype=dtype, device=device)
    """
    A = torch.randn(*(batch_dims + (matrix_size, matrix_size)),
                    dtype=dtype, device=device)
    return A @ A.mH + torch.eye(matrix_size, dtype=dtype, device=device)

# Creates a full rank matrix with distinct singular values or
#   a batch of such matrices
def make_fullrank_matrices_with_distinct_singular_values(*shape, device, dtype, requires_grad=False):
    with torch.no_grad():
        t = make_tensor(shape, device=device, dtype=dtype)
        u, _, vh = torch.linalg.svd(t, full_matrices=False)
        real_dtype = t.real.dtype if t.dtype.is_complex else t.dtype
        k = min(shape[-1], shape[-2])
        # We choose the singular values to be "around one"
        # This is to make the matrix well conditioned
        # s = [2, 3, ..., k+1]
        s = torch.arange(2, k + 2, dtype=real_dtype, device=device)
        # s = [2, -3, 4, ..., (-1)^k k+1]
        s[1::2] *= -1.
        # 1 + 1/s so that the singular values are in the range [2/3, 3/2]
        # This gives a condition number of 9/4, which should be good enough
        s.reciprocal_().add_(1.)
        # Note that the singular values need not be ordered in an SVD so
        # we don't need need to sort S
        x = (u * s.to(u.dtype)) @ vh
    x.requires_grad_(requires_grad)
    return x

def random_matrix(rows, columns, *batch_dims, **kwargs):
    """Return rectangular matrix or batches of rectangular matrices.

    Parameters:
      dtype - the data type
      device - the device kind
      singular - when True, the output will be singular
    """
    dtype = kwargs.get('dtype', torch.double)
    device = kwargs.get('device', 'cpu')
    silent = kwargs.get("silent", False)
    singular = kwargs.get("singular", False)
    if silent and not torch._C.has_lapack:
        return torch.ones(rows, columns, dtype=dtype, device=device)

    A = torch.randn(batch_dims + (rows, columns), dtype=dtype, device=device)
    if A.numel() == 0:
        return A
    u, _, vh = torch.linalg.svd(A, full_matrices=False)
    k = min(rows, columns)
    s = torch.linspace(1 / (k + 1), 1, k, dtype=dtype, device=device)
    if singular:
        # make matrix singular
        s[k - 1] = 0
        if k > 2:
            # increase the order of singularity so that the pivoting
            # in LU factorization will be non-trivial
            s[0] = 0
    return (u * s.unsqueeze(-2)) @ vh


def random_lowrank_matrix(rank, rows, columns, *batch_dims, **kwargs):
    """Return rectangular matrix or batches of rectangular matrices with
    given rank.
    """
    B = random_matrix(rows, rank, *batch_dims, **kwargs)
    C = random_matrix(rank, columns, *batch_dims, **kwargs)
    return B.matmul(C)


def random_sparse_matrix(rows, columns, density=0.01, **kwargs):
    """Return rectangular random sparse matrix within given density.

    The density of the result approaches to given density as the size
    of the matrix is increased and a relatively small value of density
    is specified but higher than min(rows, columns)/(rows * columns)
    for non-singular matrices.
    """
    dtype = kwargs.get('dtype', torch.double)
    device = kwargs.get('device', 'cpu')
    singular = kwargs.get("singular", False)

    k = min(rows, columns)
    nonzero_elements = max(min(rows, columns), int(rows * columns * density))

    row_indices = [i % rows for i in range(nonzero_elements)]
    column_indices = [i % columns for i in range(nonzero_elements)]
    random.shuffle(column_indices)
    indices = [row_indices, column_indices]
    values = torch.randn(nonzero_elements, dtype=dtype, device=device)
    # ensure that the diagonal dominates
    values *= torch.tensor([-float(i - j)**2 for i, j in zip(*indices)], dtype=dtype, device=device).exp()
    indices_tensor = torch.tensor(indices)
    A = torch.sparse_coo_tensor(indices_tensor, values, (rows, columns), device=device)
    return A.coalesce()


def random_sparse_pd_matrix(matrix_size, density=0.01, **kwargs):
    """Return random sparse positive-definite matrix with given density.

    The eigenvalues of the matrix are defined as::
      arange(1, matrix_size+1)/matrix_size

    Algorithm:
      A = diag(arange(1, matrix_size+1)/matrix_size)
      while <A density is smaller than required>:
          <choose random i, j in range(matrix_size), theta in [0, 2*pi]>
          R = <rotation matrix (i,j,theta)>
          A = R^T A R
    """
    import math
    torch = kwargs.get('torch', globals()['torch'])
    dtype = kwargs.get('dtype', torch.double)
    device = kwargs.get('device', 'cpu')
    data = {(i, i): float(i + 1) / matrix_size
            for i in range(matrix_size)}


    def multiply(data, N, i, j, cs, sn, left=True):
        for k in range(N):
            if left:
                ik, jk = (k, i), (k, j)
            else:
                ik, jk = (i, k), (j, k)
            aik, ajk = data.get(ik, 0), data.get(jk, 0)
            aik, ajk = cs * aik + sn * ajk, -sn * aik + cs * ajk
            if aik:
                data[ik] = aik
            else:
                data.pop(ik, None)
            if ajk:
                data[jk] = ajk
            else:
                data.pop(jk, None)

    target_nnz = density * matrix_size * matrix_size
    while len(data) < target_nnz:
        i = random.randint(0, matrix_size - 1)
        j = random.randint(0, matrix_size - 1)
        if i != j:
            theta = random.uniform(0, 2 * math.pi)
            cs = math.cos(theta)
            sn = math.sin(theta)
            multiply(data, matrix_size, i, j, cs, sn, left=True)
            multiply(data, matrix_size, i, j, cs, sn, left=False)
    icoords, jcoords, values = [], [], []
    for (i, j), v in sorted(data.items()):
        icoords.append(i)
        jcoords.append(j)
        values.append(v)
    indices_tensor = torch.tensor([icoords, jcoords])
    return torch.sparse_coo_tensor(indices_tensor, values, (matrix_size, matrix_size), dtype=dtype, device=device)

# FIXME: remove this by updating test suites using it
def do_test_dtypes(self, dtypes, layout, device):
    for dtype in dtypes:
        if dtype != torch.float16:
            out = torch.zeros((2, 3), dtype=dtype, layout=layout, device=device)
            self.assertIs(dtype, out.dtype)
            self.assertIs(layout, out.layout)
            self.assertEqual(device, out.device)

# FIXME: remove this by updating test suites using it
def do_test_empty_full(self, dtypes, layout, device):
    shape = torch.Size([2, 3])

    def check_value(tensor, dtype, layout, device, value, requires_grad):
        self.assertEqual(shape, tensor.shape)
        self.assertIs(dtype, tensor.dtype)
        self.assertIs(layout, tensor.layout)
        self.assertEqual(tensor.requires_grad, requires_grad)
        if tensor.is_cuda and device is not None:
            self.assertEqual(device, tensor.device)
        if value is not None:
            fill = tensor.new(shape).fill_(value)
            self.assertEqual(tensor, fill)

    def get_int64_dtype(dtype):
        module = '.'.join(str(dtype).split('.')[1:-1])
        if not module:
            return torch.int64
        return operator.attrgetter(module)(torch).int64

    default_dtype = torch.get_default_dtype()
    check_value(torch.empty(shape), default_dtype, torch.strided, -1, None, False)
    check_value(torch.full(shape, -5.), default_dtype, torch.strided, -1, None, False)
    for dtype in dtypes:
        for rg in {dtype.is_floating_point, False}:
            int64_dtype = get_int64_dtype(dtype)
            v = torch.empty(shape, dtype=dtype, device=device, layout=layout, requires_grad=rg)
            check_value(v, dtype, layout, device, None, rg)
            out = v.new()
            check_value(torch.empty(shape, out=out, device=device, layout=layout, requires_grad=rg),
                        dtype, layout, device, None, rg)
            check_value(v.new_empty(shape), dtype, layout, device, None, False)
            check_value(v.new_empty(shape, dtype=int64_dtype, device=device, requires_grad=False),
                        int64_dtype, layout, device, None, False)
            check_value(torch.empty_like(v), dtype, layout, device, None, False)
            check_value(torch.empty_like(v, dtype=int64_dtype, layout=layout, device=device, requires_grad=False),
                        int64_dtype, layout, device, None, False)

            if dtype is not torch.float16 and layout != torch.sparse_coo:
                fv = 3
                v = torch.full(shape, fv, dtype=dtype, layout=layout, device=device, requires_grad=rg)
                check_value(v, dtype, layout, device, fv, rg)
                check_value(v.new_full(shape, fv + 1), dtype, layout, device, fv + 1, False)
                out = v.new()
                check_value(torch.full(shape, fv + 2, out=out, device=device, layout=layout, requires_grad=rg),
                            dtype, layout, device, fv + 2, rg)
                check_value(v.new_full(shape, fv + 3, dtype=int64_dtype, device=device, requires_grad=False),
                            int64_dtype, layout, device, fv + 3, False)
                check_value(torch.full_like(v, fv + 4), dtype, layout, device, fv + 4, False)
                check_value(torch.full_like(v, fv + 5,
                                            dtype=int64_dtype, layout=layout, device=device, requires_grad=False),
                            int64_dtype, layout, device, fv + 5, False)

# FIXME: improve load_tests() documentation here
running_script_path = None
def set_running_script_path():
    global running_script_path
    try:
        running_file = os.path.abspath(os.path.realpath(sys.argv[0]))
        if running_file.endswith('.py'):  # skip if the running file is not a script
            running_script_path = running_file
    except Exception:
        pass

def check_test_defined_in_running_script(test_case):
    if running_script_path is None:
        return
    test_case_class_file = os.path.abspath(os.path.realpath(inspect.getfile(test_case.__class__)))
    assert test_case_class_file == running_script_path, f"Class of loaded TestCase \"{test_case.id()}\" " \
        f"is not defined in the running script \"{running_script_path}\", but in \"{test_case_class_file}\". Did you " \
        "accidentally import a unittest.TestCase from another file?"

def load_tests(loader, tests, pattern):
    set_running_script_path()
    test_suite = unittest.TestSuite()
    for test_group in tests:
        if not DISABLE_RUNNING_SCRIPT_CHK:
            for test in test_group:
                check_test_defined_in_running_script(test)
        if test_group._tests:
            test_suite.addTest(test_group)
    return test_suite

# FIXME: document this and move it to test_serialization
class BytesIOContext(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

# Tentative value for nondet_tol for gradcheck when backward implementation
# relies on nondeterministic operations, i.e., those listed here:
# https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
#
# For more information see https://github.com/pytorch/pytorch/issues/56202
GRADCHECK_NONDET_TOL = 1e-12

TestEnvironment.def_flag("TEST_WITH_SLOW_GRADCHECK", env_var="PYTORCH_TEST_WITH_SLOW_GRADCHECK")

skipIfSlowGradcheckEnv = unittest.skipIf(
    TEST_WITH_SLOW_GRADCHECK,
    "Tests that don't use gradcheck don't need to run on slow_gradcheck CI"
)

def gradcheck(fn, inputs, **kwargs):
    # Wrapper around gradcheck that enables certain keys by default.
    # Use this testing-internal gradcheck instead of autograd.gradcheck so that new features like vmap and
    # forward-mode AD are tested by default. We create this wrapper because we'd like to keep new checks
    # to be disabled to default for the public-facing api to avoid breaking user code.
    #
    # All PyTorch devs doing testing should use this wrapper instead of autograd.gradcheck.
    default_values = {
        "check_batched_grad": True,
        "fast_mode": True,
    }

    if TEST_WITH_SLOW_GRADCHECK:
        default_values["fast_mode"] = False

    for key, value in default_values.items():
        # default value override values explicitly set to None
        k = kwargs.get(key, None)
        kwargs[key] = k if k is not None else value

    return torch.autograd.gradcheck(fn, inputs, **kwargs)

def gradgradcheck(fn, inputs, grad_outputs=None, **kwargs):
    # Wrapper around gradgradcheck that enables certain keys by default
    # See gradcheck above for an explanation of why we need something like this.
    #
    # All PyTorch devs doing testing should use this wrapper instead of autograd.gradgradcheck
    default_values = {
        "check_batched_grad": True,
        "fast_mode": True,
    }

    if TEST_WITH_SLOW_GRADCHECK:
        default_values["fast_mode"] = False

    for key, value in default_values.items():
        # default value override values explicitly set to None
        k = kwargs.get(key, None)
        kwargs[key] = k if k is not None else value

    return torch.autograd.gradgradcheck(fn, inputs, grad_outputs, **kwargs)


def _assertGradAndGradgradChecks(test_case, apply_fn, inputs, **kwargs):
    # call assert function rather than returning a bool since it's nicer
    # if we get whether this failed on the gradcheck or the gradgradcheck.
    test_case.assertTrue(gradcheck(apply_fn, inputs, **kwargs))
    test_case.assertTrue(gradgradcheck(apply_fn, inputs, **kwargs))


@contextmanager
def set_cwd(path: str) -> Iterator[None]:
    old_cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_cwd)


# FIXME: delete this
# Using @toleranceOverride specific to your test is the recommended way
# of doing this. These are just some values that worked for test_nn.
dtype2prec_DONTUSE = {torch.float: 1e-5,
                      torch.double: 1e-5,
                      torch.half: 1e-2,
                      torch.bfloat16: 1e-1}

# FIXME: move to test_sparse or sparse utils
# This is a wrapper that wraps a test to run this test twice, one with
# coalesced=True, another with coalesced=False for coalesced/uncoalesced sparse tensors.
def coalescedonoff(f):
    @wraps(f)
    def wrapped(self, *args, **kwargs):
        f(self, *args, **kwargs, coalesced=True)
        f(self, *args, **kwargs, coalesced=False)
    return wrapped


def is_coalesced_indices(s):
    indices = s._indices()
    hash_coeffs = (1,) + s.shape[s.sparse_dim() - 1:0:-1]
    hash_indices = torch.tensor(hash_coeffs, device=s.device).cumprod(-1).flip(-1)
    if s.sparse_dim() > 1:
        hash_indices.unsqueeze_(-1)
        hash_indices = (indices * hash_indices).sum(0)
    else:
        hash_indices = indices * hash_indices

    # check if indices are sorted
    res = torch.allclose(hash_indices, hash_indices.sort()[0])

    # check if there are no repeated indices
    res = res and torch.allclose(hash_indices, hash_indices.unique())

    return res


@contextlib.contextmanager
def disable_gc():
    if gc.isenabled():
        try:
            gc.disable()
            yield
        finally:
            gc.enable()
    else:
        yield


def find_library_location(lib_name: str) -> Path:
    # return the shared library file in the installed folder if exist,
    # else the file in the build folder
    torch_root = Path(torch.__file__).resolve().parent
    path = torch_root / 'lib' / lib_name
    if os.path.exists(path):
        return path
    torch_root = Path(__file__).resolve().parent.parent.parent
    return torch_root / 'build' / 'lib' / lib_name

def skip_but_pass_in_sandcastle(reason):
    """
    Similar to unittest.skip, however in the sandcastle environment it just
    "passes" the test instead to avoid creating tasks complaining about tests
    skipping continuously.
    """
    def decorator(func):
        if not IS_SANDCASTLE:
            func.__unittest_skip__ = True
            func.__unittest_skip_why__ = reason
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f'Skipping {func.__name__} on sandcastle for following reason: {reason}', file=sys.stderr)
            return
        return wrapper

    return decorator

def mock_wrapper(method):
    """
    Returns a function that calls the real implementation of a method
    in addition to passing args to a mock object.
    """
    mock = MagicMock()

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        mock(*args, **kwargs)
        return method(self, *args, **kwargs)
    wrapper.mock = mock  # type: ignore[attr-defined]
    return wrapper

def get_tensors_from(args, kwargs):
    """ Returns a set of all Tensor objects in the given args and kwargs. """
    return set([arg for arg in args if isinstance(arg, Tensor)] +
               [v for v in kwargs.values() if isinstance(v, Tensor)])


# Returns scalar tensor representation of a list of integer byte values
def bytes_to_scalar(byte_list: List[int], dtype: torch.dtype, device: torch.device):
    dtype_to_ctype: Dict[torch.dtype, Any] = {
        torch.int8: ctypes.c_int8,
        torch.uint8: ctypes.c_uint8,
        torch.int16: ctypes.c_int16,
        torch.int32: ctypes.c_int32,
        torch.int64: ctypes.c_int64,
        torch.bool: ctypes.c_bool,
        torch.float32: ctypes.c_float,
        torch.complex64: ctypes.c_float,
        torch.float64: ctypes.c_double,
        torch.complex128: ctypes.c_double,
    }
    ctype = dtype_to_ctype[dtype]
    num_bytes = ctypes.sizeof(ctype)

    def check_bytes(byte_list):
        for byte in byte_list:
            assert 0 <= byte <= 255

    if dtype.is_complex:
        assert len(byte_list) == (num_bytes * 2)
        check_bytes(byte_list)
        real = ctype.from_buffer((ctypes.c_byte * num_bytes)(
            *byte_list[:num_bytes])).value
        imag = ctype.from_buffer((ctypes.c_byte * num_bytes)(
            *byte_list[num_bytes:])).value
        res = real + 1j * imag
    else:
        assert len(byte_list) == num_bytes
        check_bytes(byte_list)
        res = ctype.from_buffer((ctypes.c_byte * num_bytes)(
            *byte_list)).value

    return torch.tensor(res, device=device, dtype=dtype)


def copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


def xfail_inherited_tests(tests):
    """
    Given a list of test names which are defined by a superclass of the
    class this decorates, mark them as expected failure.  This is useful
    if you are doing poor man's parameterized tests by subclassing a generic
    test class.
    """
    def deco(cls):
        for t in tests:
            # NB: expectedFailure operates by mutating the method in question,
            # which is why you have to copy the function first
            setattr(cls, t, unittest.expectedFailure(copy_func(getattr(cls, t))))
        return cls
    return deco


def skip_but_pass_in_sandcastle_if(condition, reason):
    """
    Similar to unittest.skipIf, however in the sandcastle environment it just
    "passes" the test instead to avoid creating tasks complaining about tests
    skipping continuously.
    """
    def decorator(func):
        if condition:
            if IS_SANDCASTLE:
                @wraps(func)
                def wrapper(*args, **kwargs):
                    print(f'Skipping {func.__name__} on sandcastle for following reason: {reason}', file=sys.stderr)
                return wrapper
            else:
                func.__unittest_skip__ = True
                func.__unittest_skip_why__ = reason

        return func

    return decorator

def dtype_name(dtype):
    """ Returns the pretty name of the dtype (e.g. torch.int64 -> int64). """
    return str(dtype).split('.')[1]


dtype_abbrs = {
    torch.bfloat16: 'bf16',
    torch.float64: 'f64',
    torch.float32: 'f32',
    torch.float16: 'f16',
    torch.complex32: 'c32',
    torch.complex64: 'c64',
    torch.complex128: 'c128',
    torch.int8: 'i8',
    torch.int16: 'i16',
    torch.int32: 'i32',
    torch.int64: 'i64',
    torch.bool: 'b8',
    torch.uint8: 'u8',
}


def set_single_threaded_if_parallel_tbb(fn):
    """Set test to be single threaded for parallel tbb.

    See https://github.com/pytorch/pytorch/issues/64571#issuecomment-914691883
    """
    if not IS_TBB:
        return fn

    @wraps(fn)
    def wrap_fn(*args, **kwargs):
        num_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            return fn(*args, **kwargs)
        finally:
            torch.set_num_threads(num_threads)
    return wrap_fn


@functools.lru_cache
def get_cycles_per_ms() -> float:
    """Measure and return approximate number of cycles per millisecond for torch.cuda._sleep
    """

    def measure() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda._sleep(1000000)
        end.record()
        end.synchronize()
        cycles_per_ms = 1000000 / start.elapsed_time(end)
        return cycles_per_ms

    # Get 10 values and remove the 2 max and 2 min and return the avg.
    # This is to avoid system disturbance that skew the results, e.g.
    # the very first cuda call likely does a bunch of init, which takes
    # much longer than subsequent calls.
    #
    # Tested on both Tesla V100, Quadro GP100, Titan RTX, RTX 3090 GPUs
    # and seems to return stable values. Therefore, we enable caching
    # using lru_cache decorator above.
    num = 10
    vals = []
    for _ in range(num):
        vals.append(measure())
    vals = sorted(vals)
    return mean(vals[2 : num - 2])


# OpInfo utils

T = TypeVar('T')
def first_sample(self: unittest.TestCase, samples: Iterable[T]) -> T:
    """
    Returns the first sample from an iterable of samples, like those returned by OpInfo.
    The test will be skipped if no samples are available.
    """
    try:
        return next(iter(samples))
    except StopIteration as e:
        raise unittest.SkipTest('Skipped! Need at least 1 sample input') from e

# this helper method is to recursively
# clone the tensor-type input of operators tested by OpInfo
def clone_input_helper(input):
    if isinstance(input, torch.Tensor):
        return torch.clone(input)

    if isinstance(input, Sequence):
        return tuple(map(clone_input_helper, input))

    return input

@contextmanager
def custom_op(opname, symbolic_fn, opset_version):
    """Context manager/decorator to test ONNX export with custom operator"""
    try:
        register_custom_op_symbolic(opname, symbolic_fn, opset_version)
        yield
    finally:
        unregister_custom_op_symbolic(opname, opset_version)


def outs_and_grads(fn, graph_inps, inps):
    outs = fn(*graph_inps)
    for out in pytree.tree_flatten(outs)[0]:
        if isinstance(out, torch.Tensor) and out.requires_grad:
            out.sum().backward(retain_graph=True)
    grads = [inp.grad for inp in pytree.tree_flatten(inps)[0] if isinstance(inp, torch.Tensor)]
    for inp in pytree.tree_flatten(inps)[0]:
        if isinstance(inp, torch.Tensor):
            inp.grad = None
    return outs, grads

def compare_equal_outs_and_grads(test, m1, m2, inps):
    r1, g1 = outs_and_grads(m1, inps, inps)
    r2, g2 = outs_and_grads(m2, inps, inps)
    test.assertEqual(r1, r2)
    test.assertEqual(g1, g2)

class TestGradients(TestCase):
    exact_dtype = True

    # Copies inputs to inplace operations to avoid inplace modifications
    #   to leaves requiring gradient
    def _get_safe_inplace(self, inplace_variant):
        @wraps(inplace_variant)
        def _fn(t, *args, **kwargs):
            return inplace_variant(t.clone(), *args, **kwargs)

        return _fn

    def _check_helper(self, device, dtype, op, variant, check, *, check_forward_ad=False, check_backward_ad=True,
                      check_batched_grad=None, check_batched_forward_grad=False):
        assert check in ('gradcheck', 'bwgrad_bwgrad', 'fwgrad_bwgrad')
        # NB: check_backward_ad does not affect gradgradcheck (always True)
        if variant is None:
            self.skipTest("Skipped! Variant not implemented.")
        if not op.supports_dtype(dtype, torch.device(device).type):
            self.skipTest(f"Skipped! {op.name} does not support dtype {str(dtype)}")

        def is_inplace(variant):
            if hasattr(variant, "__wrapped__"):
                return variant.__wrapped__ is op.get_inplace()
            return variant is op.get_inplace()

        include_conjugated_inputs = op.test_conjugated_samples and dtype.is_complex

        samples = op.sample_inputs(device, dtype, requires_grad=True, include_conjugated_inputs=include_conjugated_inputs,
                                   small_inputs_only=TEST_WITH_SLOW_GRADCHECK)

        for sample in samples:
            if sample.broadcasts_input and is_inplace(variant):
                continue

            # Gradcheck expects tensors as its input, but autograd actually supports tensorlists
            #   and tensors passed as kwargs. The following creates a function that accepts just
            #   the tensors that require grad as varargs, and then recomposes them back into the
            #   original input.

            # Creates gradcheck inputs by identifying tensors requiring grad
            all_args = None
            if is_iterable_of_tensors(sample.input):
                all_args = chain(sample.input, sample.args, sample.kwargs.values())
            else:
                all_args = tuple(chain((sample.input,), sample.args, sample.kwargs.values()))
            gradcheck_args = tuple(x for x in all_args if (isinstance(x, torch.Tensor) and x.requires_grad))

            # Verifies sample input tensors should have no grad
            # This may happen if the same tensor is used in two different SampleInputs
            for t in gradcheck_args:
                self.assertIsNone(t.grad,
                                  "A sampled input has a gradient before running autograd. "
                                  "This usually means that (at least) one input tensor is reused "
                                  "across different SampleInputs. "
                                  "Please create a new tensor for each SampleInput.")

            def _input_recomposition_helper(inputs, inp, input_idx):
                if is_iterable_of_tensors(inp):
                    tensor_list = []
                    for x in inp:
                        if isinstance(x, torch.Tensor) and x.requires_grad:
                            tensor_list.append(inputs[input_idx])
                            input_idx = input_idx + 1
                        else:
                            tensor_list.append(x)
                    return tensor_list, input_idx
                elif isinstance(inp, torch.Tensor) and inp.requires_grad:
                    return inputs[input_idx], input_idx + 1
                else:
                    return inp, input_idx

            def fn(*inputs):
                # Puts inputs back into sample properly
                positional_args = []
                input_idx = 0
                inp, input_idx = _input_recomposition_helper(inputs, sample.input, input_idx)
                positional_args.append(inp)

                for x in sample.args:
                    inp, input_idx = _input_recomposition_helper(inputs, x, input_idx)
                    positional_args.append(inp)

                # Recreates kwargs
                kwargs = {}
                for k, v in sample.kwargs.items():
                    inp, input_idx = _input_recomposition_helper(inputs, v, input_idx)
                    kwargs[k] = inp

                output = op.gradcheck_wrapper(variant, *positional_args, **kwargs)
                if sample.output_process_fn_grad is not None:
                    return sample.output_process_fn_grad(output)
                return output

            if check == 'gradcheck':
                if check_batched_grad is None:
                    check_batched_grad = op.check_batched_grad
                self.assertTrue(gradcheck(fn, gradcheck_args,
                                          check_batched_grad=check_batched_grad,
                                          check_grad_dtypes=True,
                                          nondet_tol=op.gradcheck_nondet_tol,
                                          fast_mode=op.gradcheck_fast_mode,
                                          check_forward_ad=check_forward_ad,
                                          check_backward_ad=check_backward_ad,
                                          check_undefined_grad=True,
                                          check_batched_forward_grad=check_batched_forward_grad))
            elif check in ('bwgrad_bwgrad', 'fwgrad_bwgrad'):  # gradgrad check
                self.assertFalse(check_forward_ad, msg="Cannot run forward AD check for gradgradcheck")
                for gen_non_contig_grad_outputs in (False, True):
                    kwargs = {
                        "gen_non_contig_grad_outputs": gen_non_contig_grad_outputs,
                        "check_batched_grad": op.check_batched_gradgrad,
                        "check_grad_dtypes": True,
                        "nondet_tol": op.gradcheck_nondet_tol,
                        "fast_mode": op.gradcheck_fast_mode
                    }
                    if check == "fwgrad_bwgrad":
                        kwargs["check_fwd_over_rev"] = True
                        kwargs["check_rev_over_rev"] = False
                        kwargs["check_batched_grad"] = False
                        kwargs["check_undefined_grad"] = False

                    self.assertTrue(gradgradcheck(fn, gradcheck_args, **kwargs))
            else:
                self.assertTrue(False, msg="Unknown check requested!")

    def _grad_test_helper(self, device, dtype, op, variant, *, check_forward_ad=False, check_backward_ad=True,
                          check_batched_grad=None, check_batched_forward_grad=False):
        return self._check_helper(device, dtype, op, variant, 'gradcheck', check_forward_ad=check_forward_ad,
                                  check_backward_ad=check_backward_ad, check_batched_grad=check_batched_grad,
                                  check_batched_forward_grad=check_batched_forward_grad)

    def _skip_helper(self, op, device, dtype):
        if dtype not in op.supported_backward_dtypes(torch.device(device).type):
            self.skipTest("Skipped! Op doesn't support autograd for this dtype.")
        if not op.supports_autograd and not op.supports_forward_ad:
            self.skipTest("Skipped! autograd not supported.")

def make_lazy_class(cls):

    def lazy_init(self, cb):
        self._cb = cb
        self._value = None

    cls.__init__ = lazy_init

    for basename in [
        "add", "sub", "mul", "truediv", "floordiv", "mod", "divmod", "pow",
        "lshift", "rshift", "and", "or", "xor", "neg", "pos", "abs", "invert",
        "eq", "ne", "lt", "le", "gt", "ge", "bool", "int", "index",
    ]:
        name = f"__{basename}__"

        def inner_wrapper(name):
            use_operator = basename not in ("bool", "int")

            def wrapped(self, *args, **kwargs):
                if self._cb is not None:
                    self._value = self._cb()
                    self._cb = None
                if not use_operator:
                    return getattr(self._value, name)(*args, **kwargs)
                else:
                    return getattr(operator, name)(self._value, *args, **kwargs)
            return wrapped

        setattr(cls, name, inner_wrapper(name))

    return cls

@make_lazy_class
class LazyVal:
    pass


def munge_exc(e, *, suppress_suffix=True, suppress_prefix=True, file=None, skip=0):
    if file is None:
        file = inspect.stack()[1 + skip].filename  # skip one frame

    s = str(e)

    # Remove everything that looks like stack frames in NOT this file
    def repl_frame(m):
        if m.group(1) != file:
            return ""
        # Don't accept top-level, even for this script, these will wobble
        # depending on how the testing script was invoked
        if m.group(2) == "<module>":
            return ""

        return m.group(0)

    s = re.sub(r'  File "([^"]+)", line \d+, in (.+)\n    .+\n( +[~^]+ *\n)?', repl_frame, s)
    s = re.sub(r"line \d+", "line N", s)
    s = re.sub(file, os.path.basename(file), s)
    s = re.sub(os.path.join(os.path.dirname(torch.__file__), ""), "", s)
    s = re.sub(r"\\", "/", s)  # for Windows
    if suppress_suffix:
        s = re.sub(r"\n*Set TORCH_LOGS.+", "", s, flags=re.DOTALL)
        s = re.sub(r"\n*You can suppress this exception.+", "", s, flags=re.DOTALL)
    if suppress_prefix:
        s = re.sub(r"Cannot export model.+\n\n", "", s)
    s = re.sub(r" +$", "", s, flags=re.M)
    return s
