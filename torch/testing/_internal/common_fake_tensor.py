# mypy: allow-untyped-defs

import os
import unittest
from contextlib import nullcontext
from functools import wraps
from unittest.mock import patch

from torch._dynamo import config
from torch.testing._internal.common_utils import _TestParametrizer


class fake_tensor_test(_TestParametrizer):
    """Run an existing test with each FakeTensorMode implementation."""

    def __init__(self, *, processes=False):
        self.processes = processes

    def _parametrize_test(self, test, generic_cls, device_cls):
        for name, use_cpp in (("python", False), ("cpp", True)):

            @wraps(test)
            def run(*args, use_cpp_fake_tensor, **kwargs):
                env = {"CPP_FAKETENSOR": str(int(use_cpp_fake_tensor))}
                with (
                    config.patch(use_cpp_fake_tensor=use_cpp_fake_tensor),
                    patch.dict(os.environ, env) if self.processes else nullcontext(),
                ):
                    return test(*args, **kwargs)

            yield run, name, {"use_cpp_fake_tensor": use_cpp}, lambda _: []


def instantiate_fake_tensor_tests(scope, *, processes=False):
    """Expand a module's already-instantiated suites into Python/C++ variants.

    Call after device and method parametrization, before run_tests(). In
    particular, OpInfo decorators must see the original generic class names.
    Keep the Python class name for suites that import or reference each other;
    the additional C++ class has a ``Cpp`` prefix.
    Replace method-level backend parametrization when opting in the whole suite.
    Use processes=True when a suite launches fresh Python interpreters.
    """
    for name, cls in list(scope.items()):
        if (
            not isinstance(cls, type)
            or not issubclass(cls, unittest.TestCase)
            or cls.__module__ != scope["__name__"]
            or not unittest.TestLoader().getTestCaseNames(cls)
        ):
            continue
        for use_cpp in (False, True):
            variant = _fake_tensor_test_class(cls, use_cpp, processes)
            variant.__name__ = variant.__qualname__ = f"Cpp{name}" if use_cpp else name
            variant.__module__ = cls.__module__
            scope[variant.__name__] = variant


def _fake_tensor_test_class(base, use_cpp, processes):
    class FakeTensorTest(base):
        _fake_tensor_backend = use_cpp

        @classmethod
        def setUpClass(cls):
            # Register restoration before setup, including failing/skipped setup.
            cls.enterClassContext(
                config.patch(use_cpp_fake_tensor=cls._fake_tensor_backend)
            )
            if processes:
                cls.enterClassContext(
                    patch.dict(
                        os.environ, CPP_FAKETENSOR=str(int(cls._fake_tensor_backend))
                    )
                )
            super().setUpClass()

    return FakeTensorTest
