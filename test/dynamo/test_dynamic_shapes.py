# Owner(s): ["module: dynamo"]
from torch._dynamo import config
from torch._dynamo.testing import make_test_cls_with_patches

try:
    from . import (
        test_export,
        test_functions,
        test_misc,
        test_modules,
        test_repros,
        test_subgraphs,
        test_unspec,
    )
except ImportError:
    import test_export
    import test_functions
    import test_misc
    import test_modules
    import test_repros
    import test_subgraphs
    import test_unspec

import unittest


test_classes = {}


def make_dynamic_cls(cls, assume_static_by_default):
    assume_static_by_default_suffix = (
        "_static_default" if assume_static_by_default else ""
    )
    cls_prefix = "StaticDefault" if assume_static_by_default else ""
    test_class = make_test_cls_with_patches(
        cls,
        f"{cls_prefix}DynamicShapes",
        f"_dynamic_shapes{assume_static_by_default_suffix}",
        (config, "dynamic_shapes", True),
        (config, "assume_static_by_default", assume_static_by_default),
    )
    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    return test_class


tests = [
    test_functions.FunctionTests,
    test_misc.MiscTests,
    test_repros.ReproTests,
    test_modules.NNModuleTests,
    test_unspec.UnspecTests,
    test_export.ExportTests,
    test_subgraphs.SubGraphTests,
]
for test in tests:
    for assume_static_by_default in [True, False]:
        make_dynamic_cls(test, assume_static_by_default=assume_static_by_default)

DynamicShapesReproTests = test_classes["DynamicShapesReproTests"]
DynamicShapesReproTestsDefaultStatic = test_classes[
    "StaticDefaultDynamicShapesReproTests"
]
DynamicShapesSubGraphTests = test_classes["DynamicShapesSubGraphTests"]
DynamicShapesSubGraphTestsDefaultStatic = test_classes[
    "StaticDefaultDynamicShapesSubGraphTests"
]

unittest.expectedFailure(
    DynamicShapesReproTestsDefaultStatic.test_convert_boxes_to_pooler_format_dynamic_shapes_static_default
)

unittest.expectedFailure(
    DynamicShapesReproTestsDefaultStatic.test_do_paste_mask_dynamic_shapes_static_default
)

unittest.expectedFailure(
    DynamicShapesReproTestsDefaultStatic.test_hf_t5_forward_dynamic_shapes_static_default
)

unittest.expectedFailure(
    DynamicShapesReproTestsDefaultStatic.test_sort_out2_dynamic_shapes_static_default
)

unittest.expectedFailure(
    DynamicShapesReproTests.test_do_paste_mask_dynamic_shapes
    # aten.min.dim - couldn't find symbolic meta function/decomposition
)

unittest.expectedFailure(
    DynamicShapesReproTests.test_convert_boxes_to_pooler_format_dynamic_shapes
    # Could not infer dtype of torch._C.SymIntNode
)

unittest.expectedFailure(
    DynamicShapesReproTests.test_hf_t5_forward_dynamic_shapes
    # Cannot call sizes() on tensor with symbolic sizes/strides
)

unittest.expectedFailure(
    DynamicShapesReproTests.test_sort_out2_dynamic_shapes
    # Cannot call sizes() on tensor with symbolic sizes/strides
)

unittest.expectedFailure(
    DynamicShapesMiscTests.test_autocast_sdpa_dynamic_shapes
    # Cannot call sizes() on tensor with symbolic sizes/strides
)


# DynamicShapesSubGraphTests
unittest.expectedFailure(
    DynamicShapesSubGraphTests.test_enumerate_not_break_graph_dynamic_shapes
)

# DynamicShapesSubGraphTests
unittest.expectedFailure(
    DynamicShapesSubGraphTestsDefaultStatic.test_enumerate_not_break_graph_dynamic_shapes_static_default
)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
