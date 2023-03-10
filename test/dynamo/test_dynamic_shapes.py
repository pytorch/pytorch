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
    )
except ImportError:
    import test_export
    import test_functions
    import test_misc
    import test_modules
    import test_repros
    import test_subgraphs

import unittest


test_classes = {}

ALL_DYNAMIC_XFAILS = {
    "MiscTests": [
        "test_autocast_sdpa",
        "test_parsing_sdpa",
    ],
    "ReproTests": [
        # aten.min.dim - couldn't find symbolic meta function/decomposition
        "test_do_paste_mask",
        # Could not infer dtype of torch._C.SymIntNode
        "test_convert_boxes_to_pooler_format",
        # Cannot call sizes() on tensor with symbolic sizes/strides
        "test_hf_t5_forward",
        "test_sort_out2",
    ],
    "SubGraphTests": [
        "test_enumerate_not_break_graph",
    ],
}

XFAIL_HITS = 0


def make_dynamic_cls(cls, *, static_default=False, unspec=False):
    suffix = "_dynamic_shapes"
    if static_default:
        suffix += "_static_default"
    if unspec:
        suffix += "_unspec"

    cls_prefix = "DynamicShapes"
    if static_default:
        cls_prefix = f"StaticDefault{cls_prefix}"
    if unspec:
        cls_prefix = f"Unspec{cls_prefix}"

    test_class = make_test_cls_with_patches(
        cls,
        cls_prefix,
        suffix,
        (config, "dynamic_shapes", True),
        (config, "assume_static_by_default", static_default),
        (config, "specialize_int", not unspec),
    )

    xfail_tests = ALL_DYNAMIC_XFAILS.get(cls.__name__)
    if xfail_tests is not None:
        global XFAIL_HITS
        XFAIL_HITS += 1
        for t in xfail_tests:
            unittest.expectedFailure(getattr(test_class, f"{t}{suffix}"))

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    return test_class


tests = [
    test_functions.FunctionTests,
    test_misc.MiscTests,
    test_repros.ReproTests,
    test_modules.NNModuleTests,
    test_export.ExportTests,
    test_subgraphs.SubGraphTests,
]
for test in tests:
    make_dynamic_cls(test)
    make_dynamic_cls(test, unspec=True)
    make_dynamic_cls(test, static_default=True)

assert XFAIL_HITS == len(ALL_DYNAMIC_XFAILS) * 3

# Unspec only failures

unittest.expectedFailure(
    UnspecDynamicShapesMiscTests.test_slice_input_dynamic_shapes_unspec
    # NotImplementedError: SymNodeVariable() is not a constant
)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
