"""Repository rules entry point module for rules_cc."""

# WARNING: This file only exists for backwards-compatibility.
# rules_cc uses the Bazel federation, so please add any new dependencies to
# rules_cc_deps() in
# https://github.com/bazelbuild/bazel-federation/blob/master/repositories.bzl
# Third party dependencies can be added to
# https://github.com/bazelbuild/bazel-federation/blob/master/third_party_repositories.bzl
# Ideally we'd delete this entire file.

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//cc/private/toolchain:cc_configure.bzl", "cc_configure")

def rules_cc_dependencies():
    _maybe(
        http_archive,
        name = "bazel_skylib",
        sha256 = "2ea8a5ed2b448baf4a6855d3ce049c4c452a6470b1efd1504fdb7c1c134d220a",
        strip_prefix = "bazel-skylib-0.8.0",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/archive/0.8.0.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/archive/0.8.0.tar.gz",
        ],
    )

# buildifier: disable=unnamed-macro
def rules_cc_toolchains(*args):
    cc_configure(*args)

def _maybe(repo_rule, name, **kwargs):
    if not native.existing_rule(name):
        repo_rule(name = name, **kwargs)
