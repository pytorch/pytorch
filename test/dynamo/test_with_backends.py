# Owner(s): ["module: dynamo"]

from torch._dynamo import config
from torch._dynamo.testing import make_test_cls_with_patches

try:
    from . import test_functions
except ImportError:
    import test_functions


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
backends = ["aot_eager"]
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA

if HAS_CPU or HAS_CUDA:
    backends.append("inductor")
for backend in backends:
    for test in tests:
        make_dynamic_cls(test, backend)
del test

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
