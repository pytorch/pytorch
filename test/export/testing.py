import functools
import unittest
from unittest.mock import patch


def make_test_cls_with_mocked_export(
    cls, cls_prefix, fn_suffix, mocked_export_fn, xfail_prop=None
):
    MockedTestClass = type(f"{cls_prefix}{cls.__name__}", cls.__bases__, {})
    MockedTestClass.__qualname__ = MockedTestClass.__name__

    for name in dir(cls):
        if name.startswith("test_"):
            fn = getattr(cls, name)
            if not callable(fn):
                setattr(MockedTestClass, name, getattr(cls, name))
                continue
            new_name = f"{name}{fn_suffix}"
            new_fn = _make_fn_with_mocked_export(fn, mocked_export_fn)
            new_fn.__name__ = new_name
            if xfail_prop is not None and hasattr(fn, xfail_prop):
                new_fn = unittest.expectedFailure(new_fn)
            setattr(MockedTestClass, new_name, new_fn)
        # NB: Doesn't handle slots correctly, but whatever
        elif not hasattr(MockedTestClass, name):
            setattr(MockedTestClass, name, getattr(cls, name))

    return MockedTestClass


def _make_fn_with_mocked_export(fn, mocked_export_fn):
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        try:
            from . import test_export
        except ImportError:
            import test_export

        with patch(f"{test_export.__name__}.export", mocked_export_fn):
            return fn(*args, **kwargs)

    return _fn


# Controls tests generated in test/export/test_export_nonstrict.py
def expectedFailureNonStrict(fn):
    fn._expected_failure_non_strict = True
    return fn


# Controls tests generated in test/export/test_retraceability.py
def expectedFailureRetraceability(fn):
    fn._expected_failure_retrace = True
    return fn


# Controls tests generated in test/export/test_serdes.py
def expectedFailureSerDer(fn):
    fn._expected_failure_serdes = True
    return fn
