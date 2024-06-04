# mypy: allow-untyped-defs
import logging
import os
import sys

# NOTE: [dynamo_test_failures.py]
#
# We generate xFailIfTorchDynamo* for all tests in `dynamo_expected_failures`
# We generate skipIfTorchDynamo* for all tests in `dynamo_skips`
#
# For an easier-than-manual way of generating and updating these lists,
# see scripts/compile_tests/update_failures.py
#
# If you're adding a new test, and it's failing PYTORCH_TEST_WITH_DYNAMO=1,
# either add the appropriate decorators to your test or add skips for them
# via test/dynamo_skips and test/dynamo_expected_failures.
#
# *These are not exactly unittest.expectedFailure and unittest.skip. We'll
# always execute the test and then suppress the signal, if necessary.
# If your tests crashes, or is slow, please use @skipIfTorchDynamo instead.
#
# The expected failure and skip files are located in test/dynamo_skips and
# test/dynamo_expected_failures. They're individual files rather than a list so
# git will merge changes easier.


def find_test_dir():
    # Find the path to the dynamo expected failure and skip files.
    from os.path import abspath, basename, dirname, exists, join, normpath

    if sys.platform == "win32":
        return None

    # Check relative to this file (local build):
    test_dir = normpath(join(dirname(abspath(__file__)), "../../../test"))
    if exists(join(test_dir, "dynamo_expected_failures")):
        return test_dir

    # Check relative to __main__ (installed builds relative to test file):
    main = sys.modules["__main__"]
    file = getattr(main, "__file__", None)
    if file is None:
        # Generated files do not have a module.__file__
        return None
    test_dir = dirname(abspath(file))
    while dirname(test_dir) != test_dir:
        if basename(test_dir) == "test" and exists(
            join(test_dir, "dynamo_expected_failures")
        ):
            return test_dir
        test_dir = dirname(test_dir)

    # Not found
    return None


test_dir = find_test_dir()
if not test_dir:
    logger = logging.getLogger(__name__)
    logger.warning(
        "test/dynamo_expected_failures directory not found - known dynamo errors won't be skipped."
    )

# Tests that run without strict mode in PYTORCH_TEST_WITH_INDUCTOR=1.
# Please don't add anything to this list.
FIXME_inductor_non_strict = {
    "test_modules",
    "test_ops",
    "test_ops_gradients",
    "test_torch",
}

# We generate unittest.expectedFailure for all of the following tests
# when run under PYTORCH_TEST_WITH_DYNAMO=1.
# see NOTE [dynamo_test_failures.py] for more details
#
# This lists exists so we can more easily add large numbers of failing tests,
if test_dir is None:
    dynamo_expected_failures = set()
    dynamo_skips = set()
else:
    failures_directory = os.path.join(test_dir, "dynamo_expected_failures")
    skips_directory = os.path.join(test_dir, "dynamo_skips")

    dynamo_expected_failures = set(os.listdir(failures_directory))
    dynamo_skips = set(os.listdir(skips_directory))

# TODO: due to case sensitivity problems, for now list these files by hand
extra_dynamo_skips = {
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_T_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_t_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_T_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_t_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_T_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_t_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_T_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_t_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_T_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_t_cpu_float32",
}
dynamo_skips = dynamo_skips.union(extra_dynamo_skips)


# verify some invariants
for test in dynamo_expected_failures.union(dynamo_skips):
    if len(test.split(".")) != 2:
        raise AssertionError(f'Invalid test name: "{test}"')

intersection = dynamo_expected_failures.intersection(dynamo_skips)
if len(intersection) > 0:
    raise AssertionError(
        "there should be no overlap between dynamo_expected_failures "
        "and dynamo_skips, got " + str(intersection)
    )
