# Owner(s): ["module: dynamo"]

from torch._dynamo.testing import make_test_cls_with_patches

try:
    from . import test_functions, test_misc, test_modules, test_repros, test_unspec
except ImportError:
    import test_functions
    import test_misc
    import test_modules
    import test_repros
    import test_unspec


def make_dynamic_cls(cls):
    return make_test_cls_with_patches(
        cls, "DynamicShapes", "_dynamic_shapes", ("dynamic_shapes", True)
    )


DynamicShapesFunctionTests = make_dynamic_cls(test_functions.FunctionTests)
DynamicShapesMiscTests = make_dynamic_cls(test_misc.MiscTests)
DynamicShapesReproTests = make_dynamic_cls(test_repros.ReproTests)
DynamicShapesNNModuleTests = make_dynamic_cls(test_modules.NNModuleTests)
DynamicShapesUnspecTests = make_dynamic_cls(test_unspec.UnspecTests)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
