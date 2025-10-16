# Owner(s): ["module: dynamo"]
import unittest

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import make_test_cls_with_patches


try:
    from . import test_activation_checkpointing, test_ctx_manager
except ImportError:
    import test_activation_checkpointing
    import test_ctx_manager

test_classes = {}


def make_nested_cls(cls):
    config = torch._dynamo.config

    test_class = make_test_cls_with_patches(
        cls,
        "NestedGraphBreaks",
        "_nested_graph_breaks",
        (config, "nested_graph_breaks", True),
        (config, "debug_force_nested_calls", True),
        (config, "debug_disable_compile_counter", True),
        xfail_prop="_expected_failure_nested_graph_breaks",
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__


tests = [
    getattr(
        test_activation_checkpointing, "ActivationCheckpointingViaTagsTestsCUDA", None
    ),
    test_ctx_manager.CtxManagerTests,
]

test = None
for test in tests:
    if not test:
        continue
    make_nested_cls(test)

del test, tests

xfails = []

case = None

for case in xfails:
    unittest.expectedFailure(case)

del case, xfails

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
