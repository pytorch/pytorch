load("@bazel_skylib//lib:paths.bzl", "paths")
load(
    "//caffe2/test:defs.bzl",
    "define_mp_tests",
)

def define_fsdp_tests():
    test_files = native.glob(["**/test_*.py"])

    TESTS = {}

    additional_deps = {}
    for test_file in test_files:
        test_file_name = paths.basename(test_file)
        test_name = test_file_name.replace("test_", "").replace(".py", "")
        TESTS[test_name] = [test_file]
        additional_deps[test_name] = ["//pytorch/vision:torchvision"]

    define_mp_tests(
        tests = TESTS,
        additional_deps = additional_deps,
    )
