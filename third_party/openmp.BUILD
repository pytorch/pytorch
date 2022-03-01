load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

cmake(
    name = "openmp",
    lib_source = ":all_srcs",
    # This is only used for Apple builds right now, since the default
    # Clang toolchain on macOS doesn't include OpenMP.
    out_shared_libs = ["libomp.dylib"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)
