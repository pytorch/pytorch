sh_test(
    name = "system_library_test",
    size = "small",
    srcs = ["system_library_test.sh"],
    data = [
        ":unittest.bash",
        "//cc:system_library.bzl",
        "@bazel_tools//tools/bash/runfiles",
    ],
)
