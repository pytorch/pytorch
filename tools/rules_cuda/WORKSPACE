workspace(name = "rules_cuda")

load("//cuda:dependencies.bzl", "rules_cuda_dependencies")

rules_cuda_dependencies()

load("@rules_cc//cc:repositories.bzl", "rules_cc_toolchains")

rules_cc_toolchains()

load("@bazel_skylib//lib:versions.bzl", "versions")

versions.check(minimum_bazel_version = "4.0.0")

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()
