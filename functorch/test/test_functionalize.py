# Owner(s): ["module: functorch"]

import functorch
from unittest.mock import patch
import functools
from torch.testing._internal.common_utils import run_tests
import test_compile_cache
import test_pythonkey


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


FunctionalizeTestCompileCache = make_functionalize_test(test_compile_cache.TestCompileCache)
FunctionalizeTestCompileCacheStaticArgs = make_functionalize_test(test_compile_cache.TestCompileCacheStaticArgs)
FunctionalizeTestPythonKeyAOT = make_functionalize_test(test_pythonkey.TestAOTAutograd)
FunctionalizeTestPythonKeyPartitioning = make_functionalize_test(test_pythonkey.TestPartitioning)

if __name__ == "__main__":
    run_tests()
