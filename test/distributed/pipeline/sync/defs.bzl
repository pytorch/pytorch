load("@bazel_skylib//lib:paths.bzl", "paths")
load(
    "//caffe2/test:defs.bzl",
    "define_tests",
)

def define_pipeline_tests():
    test_files = native.glob(["**/test_*.py"])

    TESTS = {}

    for test_file in test_files:
        test_file_name = paths.basename(test_file)
        test_name = test_file_name.replace("test_", "").replace(".py", "")
        TESTS[test_name] = [test_file]

    define_tests(
        pytest = True,
        tests = TESTS,
        external_deps = [("pytest", None)],
        resources = ["conftest.py"],
    )
