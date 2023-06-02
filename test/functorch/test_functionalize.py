# Owner(s): ["module: functorch"]

import functools

import test_aotdispatch
from torch.testing._internal.common_utils import run_tests, skipIfRocm


def make_functionalize_fn(fn):
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        return fn(*args, **kwargs)

    return _fn


def make_functionalize_test(cls):
    def wrapper(wrapped_cls):
        @functools.wraps(wrapped_cls, updated=[])
        class FunctionalizeTest(cls):
            pass

        for name in dir(cls):
            if name.startswith("test_"):
                fn = getattr(cls, name)
                if not callable(fn):
                    continue

                fn = make_functionalize_fn(fn)
                # https://github.com/pytorch/pytorch/issues/96560
                fn = skipIfRocm(fn)

                new_name = f"{name}_functionalize"
                fn.__name__ = new_name
                setattr(FunctionalizeTest, name, None)
                setattr(FunctionalizeTest, new_name, fn)

        return FunctionalizeTest

    return wrapper


@make_functionalize_test(test_aotdispatch.TestAOTAutograd)
class FunctionalizeTestPythonKeyAOT:
    pass


@make_functionalize_test(test_aotdispatch.TestPartitioning)
class FunctionalizeTestPythonKeyPartitioning:
    pass


if __name__ == "__main__":
    run_tests()
