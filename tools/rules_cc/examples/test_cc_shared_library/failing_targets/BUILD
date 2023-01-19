load("//cc:defs.bzl", "cc_binary")
load("//examples:experimental_cc_shared_library.bzl", "cc_shared_library", "cc_shared_library_permissions")

package(
    default_visibility = ["//examples/test_cc_shared_library:__pkg__"],
)

TAGS = [
    "manual",
    "nobuilder",
]

cc_binary(
    name = "should_fail_binary",
    dynamic_deps = ["//examples/test_cc_shared_library:foo_so"],
    tags = TAGS,
    deps = [
        "//examples/test_cc_shared_library:foo",
        "//examples/test_cc_shared_library:qux",
    ],
)

cc_shared_library(
    name = "permissions_fail_so",
    roots = [
        "//examples/test_cc_shared_library3:bar",
    ],
    tags = TAGS,
)

cc_shared_library_permissions(
    name = "permissions_fail",
    tags = TAGS,
    targets = [
        "//examples/test_cc_shared_library:foo",
    ],
)
