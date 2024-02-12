# mypy: ignore-errors
import csv
import os

# NOTE: [dynamo_test_failures.py]
#
# We generate xFailIfTorchDynamo* for all tests in `dynamo_expected_failures`
# We generate skipIfTorchDynamo* for all tests in `dynamo_skips`
#
# For an easier-than-manual way of generating and updating these lists,
# see scripts/compile_tests/update_failures.py
#
# If you're adding a new test, and it's failing PYTORCH_TEST_WITH_DYNAMO=1,
# either add the appropriate decorators to your test or list them in this file.
#
# *These are not exactly unittest.expectedFailure and unittest.skip. We'll
# always execute the test and then suppress the signal, if necessary.
# If your tests crashes, or is slow, please use @skipIfTorchDynamo instead.

# Tests that run without strict mode in PYTORCH_TEST_WITH_INDUCTOR=1.
# Please don't add anything to this list.
FIXME_inductor_non_strict = {
    "test_modules",
    "test_ops",
    "test_ops_gradients",
    "test_torch",
}

cache = None


def get_config():
    global cache
    if cache is not None:
        return cache
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "dynamo_test_failures.csv")
        test_failures = {}
        with open(csv_path) as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                # There should be 3 lines in between everything in dynamo_test_failures.csv!
                if i % 4 != 0:
                    assert len(row) == 0
                    continue
                if i == 0:
                    # ignore header
                    continue
                assert len(row) >= 2
                test_failures[row[0]] = row[1]
    except FileNotFoundError:
        # CSV not packaged with PyTorch or some weird build configuration
        test_failures = {}

    # We generate unittest.expectedFailure/unittest.skip for all of the following tests
    # when run under PYTORCH_TEST_WITH_DYNAMO=1.
    # see NOTE [dynamo_test_failures.py] for more details
    dynamo_expected_failures = [
        test for test, value in test_failures.items() if value == "xfail"
    ]
    dynamo_skips = [test for test, value in test_failures.items() if value == "skip"]

    def check_sorted(lst):
        return lst == sorted(lst)

    check_sorted(dynamo_expected_failures)
    check_sorted(dynamo_skips)

    dynamo_expected_failures = set(dynamo_expected_failures)
    dynamo_skips = set(dynamo_skips)

    cache = (dynamo_expected_failures, dynamo_skips)
    return cache
