load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

filegroup(
    name = "distribution",
    srcs = [
        "BUILD",
        "LICENSE",
        "internal_deps.bzl",
        "internal_setup.bzl",
    ],
    visibility = ["@//distro:__pkg__"],
)

bzl_library(
    name = "internal_deps_bzl",
    srcs = ["internal_deps.bzl"],
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "internal_setup_bzl",
    srcs = ["internal_setup.bzl"],
    visibility = ["//visibility:private"],
)
