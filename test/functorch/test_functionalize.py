# Owner(s): ["module: functorch"]

import functorch
from unittest.mock import patch
import functools
from torch.testing._internal.common_utils import run_tests
import test_aotdispatch


def make_functionalize_fn(fn):
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        with patch.object(functorch.compile.config, "use_functionalize", True):
            return fn(*args, **kwargs)

    return _fn


def make_functionalize_test(cls):
    class FunctionalizeTest(cls):
        pass

    FunctionalizeTest.__name__ = f"Functionalize{cls.__name__}"

    for name in dir(cls):
        if name.startswith("test_"):
            fn = getattr(cls, name)
            if not callable(fn):
                continue

            new_name = f"{name}_functionalize"
            fn = make_functionalize_fn(fn)
            fn.__name__ = new_name
            setattr(FunctionalizeTest, name, None)
            setattr(FunctionalizeTest, new_name, fn)

    return FunctionalizeTest


FunctionalizeTestPythonKeyAOT = make_functionalize_test(test_aotdispatch.TestAOTAutograd)
FunctionalizeTestPythonKeyPartitioning = make_functionalize_test(test_aotdispatch.TestPartitioning)

if __name__ == "__main__":
    run_tests()
