# Owner(s): ["module: dynamo"]
import unittest
import warnings

from torch._dynamo import config
from torch._dynamo.testing import make_test_cls_with_patches
from torch.fx.experimental import _config as fx_config

try:
    from . import (
        test_ctx_manager,
        test_export,
        test_functions,
        test_higher_order_ops,
        test_misc,
        test_modules,
        test_repros,
        test_sdpa,
        test_subgraphs,
    )
except ImportError:
    import test_ctx_manager
    import test_export
    import test_functions
    import test_higher_order_ops
    import test_misc
    import test_modules
    import test_repros
    import test_sdpa
    import test_subgraphs


test_classes = {}


def make_dynamic_cls(cls, backend):
    suffix = "_generated_Backend_test_" + backend

    cls_prefix = "GeneratedBackendTest" + backend

    test_class = make_test_cls_with_patches(
        cls,
        cls_prefix,
        suffix,
        (config, "_test_backend_override", backend),
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


tests = [
    test_functions.FunctionTests,
]
for backend in ["aot_eager", "inductor"]:
    for test in tests:
        make_dynamic_cls(test, backend)
del test

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
