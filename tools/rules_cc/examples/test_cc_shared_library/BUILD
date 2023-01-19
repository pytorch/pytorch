load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("//examples:experimental_cc_shared_library.bzl", "LINKABLE_MORE_THAN_ONCE", "cc_shared_library")
load(":starlark_tests.bzl", "additional_inputs_test", "build_failure_test", "linking_suffix_test", "paths_test")

package(
    default_visibility = ["//examples/test_cc_shared_library:__subpackages__"],
)

cc_test(
    name = "cc_test",
    srcs = ["main.cc"],
    dynamic_deps = ["foo_so"],
    deps = ["foo"],
)

cc_binary(
    name = "binary",
    srcs = ["main.cc"],
    dynamic_deps = ["foo_so"],
    deps = ["foo"],
)

cc_shared_library(
    name = "foo_so",
    additional_linker_inputs = [
        ":foo.lds",
        ":additional_script.txt",
    ],
    dynamic_deps = ["bar_so"],
    preloaded_deps = ["preloaded_dep"],
    roots = [
        "baz",
        "foo",
        "a_suffix",
    ],
    static_deps = [
        "//examples/test_cc_shared_library:qux",
        "//examples/test_cc_shared_library:qux2",
    ],
    user_link_flags = [
        "-Wl,-rpath,kittens",
        "-Wl,--version-script=$(location :foo.lds)",
        "-Wl,--script=$(location :additional_script.txt)",
    ],
)

cc_library(
    name = "preloaded_dep",
    srcs = ["preloaded_dep.cc"],
    hdrs = ["preloaded_dep.h"],
)

cc_library(
    name = "foo",
    srcs = ["foo.cc"],
    hdrs = ["foo.h"],
    deps = [
        "preloaded_dep",
        "bar",
        "baz",
        # Not exported.
        "qux",
        "qux2",
    ],
)

cc_library(
    name = "a_suffix",
    srcs = ["a_suffix.cc"],
    hdrs = ["a_suffix.h"],
)

cc_library(
    name = "baz",
    srcs = ["baz.cc"],
    hdrs = ["baz.h"],
    deps = ["bar3"],
)

cc_library(
    name = "qux",
    srcs = ["qux.cc"],
    hdrs = ["qux.h"],
)

cc_library(
    name = "qux2",
    srcs = ["qux2.cc"],
    hdrs = ["qux2.h"],
    tags = [LINKABLE_MORE_THAN_ONCE],
)

cc_shared_library(
    name = "bar_so",
    additional_linker_inputs = [
        ":bar.lds",
    ],
    exports_filter = [
        "bar3",  # Exported transitive dependency
        "//examples/test_cc_shared_library3:bar",
    ],
    permissions = [
        "//examples/test_cc_shared_library3:permissions",
    ],
    roots = [
        "bar",
        "bar2",
        "@test_repo//:bar",
    ],
    static_deps = [
        "//examples/test_cc_shared_library:barX",
        "@test_repo//:bar",
        "//examples/test_cc_shared_library:qux2",
    ],
    user_link_flags = [
        "-Wl,--version-script=$(location :bar.lds)",
    ],
)

cc_library(
    name = "barX",
    srcs = ["bar.cc"],
    hdrs = ["bar.h"],
    deps = [
        "@test_repo//:bar",
    ],
)

cc_library(
    name = "bar",
    srcs = ["bar.cc"],
    hdrs = ["bar.h"],
    deps = [
        "barX",
        "qux2",
    ],
)

cc_library(
    name = "bar2",
    srcs = ["bar2.cc"],
    hdrs = ["bar2.h"],
    deps = ["bar3"],
)

cc_library(
    name = "bar3",
    srcs = ["bar3.cc"],
    hdrs = ["bar3.h"],
    deps = [
        "//examples/test_cc_shared_library3:bar",
    ],
)

sh_test(
    name = "cc_shared_library_integration_test",
    srcs = ["cc_shared_library_integration_test.sh"],
    data = [
        ":bar_so",
        ":binary",
        ":cc_test",
        ":foo_so",
    ],
)

linking_suffix_test(
    name = "linking_action_test",
    target_under_test = ":foo_so",
)

additional_inputs_test(
    name = "additional_inputs_test",
    target_under_test = ":foo_so",
)

build_failure_test(
    name = "link_once_repeated_test",
    message = "already linked statically in " +
              "//examples/test_cc_shared_library:foo_so but not exported.",
    target_under_test = "//examples/test_cc_shared_library/failing_targets:should_fail_binary",
)

paths_test(
    name = "path_matching_test",
)

build_failure_test(
    name = "export_without_permissions_test",
    message = "doesn't have the necessary permissions",
    target_under_test = "//examples/test_cc_shared_library/failing_targets:permissions_fail_so",
)

build_failure_test(
    name = "forbidden_target_permissions_test",
    message = "can only list targets that are in the same package or a sub-package",
    target_under_test = "//examples/test_cc_shared_library/failing_targets:permissions_fail",
)

bzl_library(
    name = "test_cc_shared_library_starlark_tests_bzl",
    srcs = ["starlark_tests.bzl"],
    visibility = ["//visibility:private"],
)
