# mypy: allow-untyped-defs
import logging
import os
import sys


# NOTE: [hpu_test_failures.py]
# The expected failure and skip files are located in test/hpu_expected_failures and
# test/hpu_skips.


def find_test_dir():
    # Find the path to the hpu expected failure and skip files.
    from os.path import abspath, basename, dirname, exists, join, normpath

    if sys.platform == "win32":
        return None

    # Check relative to this file (local build):
    test_dir = normpath(join(dirname(abspath(__file__)), "../../../test"))

    if exists(join(test_dir, "hpu_expected_failures")):
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
            join(test_dir, "hpu_expected_failures")
        ):
            return test_dir
        test_dir = dirname(test_dir)

    # Not found
    return None


test_dir = find_test_dir()
if not test_dir:
    logger = logging.getLogger(__name__)
    logger.warning(
        "test/hpu_expected_failures directory not found - known hpu errors won't be skipped."
    )

if test_dir is None:
    hpu_expected_failures = set()
    hpu_skips = set()

else:
    hpu_failures_directory = os.path.join(test_dir, "hpu_expected_failures")
    hpu_skips_directory = os.path.join(test_dir, "hpu_skips")

    hpu_expected_failures = set(os.listdir(hpu_failures_directory))
    hpu_skips = set(os.listdir(hpu_skips_directory))

# TODO: due to case sensitivity problems, for now list these files by hand
extra_hpu_expected_failures: set[str] = set()
hpu_expected_failures = hpu_expected_failures.union(extra_hpu_expected_failures)

extra_hpu_skips: set[str] = set()
hpu_skips = hpu_skips.union(extra_hpu_skips)


# verify some invariants
for test in hpu_expected_failures | hpu_skips:
    if len(test.split(".")) != 2:
        raise AssertionError(f'Invalid test name: "{test}"')

hpu_intersection = hpu_expected_failures.intersection(hpu_skips)
if len(hpu_intersection) > 0:
    raise AssertionError(
        "there should be no overlap between hpu_expected_failures "
        "and hpu_skips, got " + str(hpu_intersection)
    )
