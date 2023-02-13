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


def make_dynamic_cls(cls, assume_static_by_default):
    assume_static_by_default_suffix = (
        "_static_default" if assume_static_by_default else ""
    )
    return make_test_cls_with_patches(
        cls,
        "DynamicShapes",
        f"_dynamic_shapes{assume_static_by_default_suffix}",
        (config, "dynamic_shapes", True),
        (config, "assume_static_by_default", assume_static_by_default),
    )


# Note - this is a little redundant, but some of these tests have specific
# exclusions.
# Note #2 - Putting this in a for loop over [True, False] for
# assume_static_by_default clobbers the name, which makes the tests not run.
DynamicShapesFunctionTests = make_dynamic_cls(
    test_functions.FunctionTests, assume_static_by_default=False
)
DynamicShapesFunctionTestsDefaultStatic = make_dynamic_cls(
    test_functions.FunctionTests, assume_static_by_default=True
)
DynamicShapesMiscTests = make_dynamic_cls(
    test_misc.MiscTests, assume_static_by_default=False
)
DynamicShapesMiscTestsDefaultStatic = make_dynamic_cls(
    test_misc.MiscTests, assume_static_by_default=True
)
DynamicShapesReproTests = make_dynamic_cls(
    test_repros.ReproTests, assume_static_by_default=False
)
DynamicShapesReproTestsDefaultStatic = make_dynamic_cls(
    test_repros.ReproTests, assume_static_by_default=True
)
DynamicShapesNNModuleTests = make_dynamic_cls(
    test_modules.NNModuleTests, assume_static_by_default=False
)
DynamicShapesNNModuleTestsDefaultStatic = make_dynamic_cls(
    test_modules.NNModuleTests, assume_static_by_default=True
)
DynamicShapesUnspecTests = make_dynamic_cls(
    test_unspec.UnspecTests, assume_static_by_default=False
)
DynamicShapesUnspecTestsDefaultStatic = make_dynamic_cls(
    test_unspec.UnspecTests, assume_static_by_default=True
)
DynamicShapesExportTests = make_dynamic_cls(
    test_export.ExportTests, assume_static_by_default=False
)
DynamicShapesExportTestsDefaultStatic = make_dynamic_cls(
    test_export.ExportTests, assume_static_by_default=True
)
DynamicShapesSubGraphTests = make_dynamic_cls(
    test_subgraphs.SubGraphTests, assume_static_by_default=False
)
DynamicShapesSubGraphTestsDefaultStatic = make_dynamic_cls(
    test_subgraphs.SubGraphTests, assume_static_by_default=True
)


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
