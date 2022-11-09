# Owner(s): ["module: dynamo"]
import unittest

from torch._dynamo.testing import make_test_cls_with_patches

try:
    from . import test_functions, test_misc, test_modules, test_repros, test_unspec
except ImportError:
    import test_functions
    import test_misc
    import test_modules
    import test_repros
    import test_unspec


def make_no_fake_cls(cls):
    return make_test_cls_with_patches(
        cls, "NoFakeTensors", "_no_fake_tensors", ("fake_tensor_propagation", False)
    )


NoFakeTensorsFunctionTests = make_no_fake_cls(test_functions.FunctionTests)
NoFakeTensorsMiscTests = make_no_fake_cls(test_misc.MiscTests)
NoFakeTensorsReproTests = make_no_fake_cls(test_repros.ReproTests)
NoFakeTensorsNNModuleTests = make_no_fake_cls(test_modules.NNModuleTests)
NoFakeTensorsUnspecTests = make_no_fake_cls(test_unspec.UnspecTests)

unittest.expectedFailure(
    NoFakeTensorsReproTests.test_guard_fail_tensor_bool_no_fake_tensors
)
NoFakeTensorsReproTests.test_numpy_list_no_fake_tensors.__unittest_expecting_failure__ = (
    False
)
NoFakeTensorsUnspecTests.test_builtin_getitem_no_fake_tensors.__unittest_expecting_failure__ = (
    False
)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
