# Copyright 2016 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rules for configuring the C++ toolchain (experimental)."""

load("@bazel_tools//tools/osx:xcode_configure.bzl", "run_xcode_locator")
load(
    ":lib_cc_configure.bzl",
    "get_cpu_value",
    "resolve_labels",
)
load(":osx_cc_configure.bzl", "configure_osx_toolchain")
load(":unix_cc_configure.bzl", "configure_unix_toolchain")
load(":windows_cc_configure.bzl", "configure_windows_toolchain")

def _generate_cpp_only_build_file(repository_ctx, cpu_value, paths):
    repository_ctx.template(
        "BUILD",
        paths["@rules_cc//cc/private/toolchain:BUILD.toolchains.tpl"],
        {"%{name}": cpu_value},
    )

def cc_autoconf_toolchains_impl(repository_ctx):
    """Generate BUILD file with 'toolchain' targets for the local host C++ toolchain.

    Args:
      repository_ctx: repository context
    """
    paths = resolve_labels(repository_ctx, [
        "@rules_cc//cc/private/toolchain:BUILD.toolchains.tpl",
        "@bazel_tools//tools/osx/crosstool:BUILD.toolchains",
        "@bazel_tools//tools/osx/crosstool:osx_archs.bzl",
        "@bazel_tools//tools/osx:xcode_locator.m",
    ])
    env = repository_ctx.os.environ
    cpu_value = get_cpu_value(repository_ctx)

    # Should we try to find C++ toolchain at all? If not, we don't have to generate toolchains for C++ at all.
    should_detect_cpp_toolchain = "BAZEL_DO_NOT_DETECT_CPP_TOOLCHAIN" not in env or env["BAZEL_DO_NOT_DETECT_CPP_TOOLCHAIN"] != "1"

    # Should we unconditionally *not* use xcode? If so, we don't have to run Xcode locator ever.
    should_use_cpp_only_toolchain = "BAZEL_USE_CPP_ONLY_TOOLCHAIN" in env and env["BAZEL_USE_CPP_ONLY_TOOLCHAIN"] == "1"

    # Should we unconditionally use xcode? If so, we don't have to run Xcode locator now.
    should_use_xcode = "BAZEL_USE_XCODE_TOOLCHAIN" in env and env["BAZEL_USE_XCODE_TOOLCHAIN"] == "1"

    if not should_detect_cpp_toolchain:
        repository_ctx.file("BUILD", "# C++ toolchain autoconfiguration was disabled by BAZEL_DO_NOT_DETECT_CPP_TOOLCHAIN env variable.")
    elif cpu_value == "darwin" and not should_use_cpp_only_toolchain:
        xcode_toolchains = []

        # Only detect xcode if the user didn't tell us it will be there.
        if not should_use_xcode:
            # TODO(#6926): Unify C++ and ObjC toolchains so we don't have to run xcode locator to generate toolchain targets.
            # And also so we don't have to keep this code in sync with @rules_cc//cc/private/toolchain:osx_cc_configure.bzl.
            (xcode_toolchains, _xcodeloc_err) = run_xcode_locator(
                repository_ctx,
                paths["@bazel_tools//tools/osx:xcode_locator.m"],
            )

        if should_use_xcode or xcode_toolchains:
            repository_ctx.symlink(paths["@bazel_tools//tools/osx/crosstool:BUILD.toolchains"], "BUILD")
            repository_ctx.symlink(paths["@bazel_tools//tools/osx/crosstool:osx_archs.bzl"], "osx_archs.bzl")
        else:
            _generate_cpp_only_build_file(repository_ctx, cpu_value, paths)
    else:
        _generate_cpp_only_build_file(repository_ctx, cpu_value, paths)

cc_autoconf_toolchains = repository_rule(
    environ = [
        "BAZEL_USE_CPP_ONLY_TOOLCHAIN",
        "BAZEL_DO_NOT_DETECT_CPP_TOOLCHAIN",
    ],
    implementation = cc_autoconf_toolchains_impl,
    configure = True,
)

def cc_autoconf_impl(repository_ctx, overriden_tools = dict()):
    """Generate BUILD file with 'cc_toolchain' targets for the local host C++ toolchain.

    Args:
       repository_ctx: repository context
       overriden_tools: dict of tool paths to use instead of autoconfigured tools
    """

    env = repository_ctx.os.environ
    cpu_value = get_cpu_value(repository_ctx)
    if "BAZEL_DO_NOT_DETECT_CPP_TOOLCHAIN" in env and env["BAZEL_DO_NOT_DETECT_CPP_TOOLCHAIN"] == "1":
        paths = resolve_labels(repository_ctx, [
            "@rules_cc//cc/private/toolchain:BUILD.empty",
            "@rules_cc//cc/private/toolchain:empty_cc_toolchain_config.bzl",
        ])
        repository_ctx.symlink(paths["@rules_cc//cc/private/toolchain:empty_cc_toolchain_config.bzl"], "cc_toolchain_config.bzl")
        repository_ctx.symlink(paths["@rules_cc//cc/private/toolchain:BUILD.empty"], "BUILD")
    elif cpu_value == "freebsd":
        paths = resolve_labels(repository_ctx, [
            "@rules_cc//cc/private/toolchain:BUILD.static.freebsd",
            "@rules_cc//cc/private/toolchain:freebsd_cc_toolchain_config.bzl",
        ])

        # This is defaulting to a static crosstool, we should eventually
        # autoconfigure this platform too.  Theorically, FreeBSD should be
        # straightforward to add but we cannot run it in a docker container so
        # skipping until we have proper tests for FreeBSD.
        repository_ctx.symlink(paths["@rules_cc//cc/private/toolchain:freebsd_cc_toolchain_config.bzl"], "cc_toolchain_config.bzl")
        repository_ctx.symlink(paths["@rules_cc//cc/private/toolchain:BUILD.static.freebsd"], "BUILD")
    elif cpu_value == "x64_windows":
        # TODO(ibiryukov): overriden_tools are only supported in configure_unix_toolchain.
        # We might want to add that to Windows too(at least for msys toolchain).
        configure_windows_toolchain(repository_ctx)
    elif (cpu_value == "darwin" and
          ("BAZEL_USE_CPP_ONLY_TOOLCHAIN" not in env or env["BAZEL_USE_CPP_ONLY_TOOLCHAIN"] != "1")):
        configure_osx_toolchain(repository_ctx, overriden_tools)
    else:
        configure_unix_toolchain(repository_ctx, cpu_value, overriden_tools)

MSVC_ENVVARS = [
    "BAZEL_VC",
    "BAZEL_VC_FULL_VERSION",
    "BAZEL_VS",
    "BAZEL_WINSDK_FULL_VERSION",
    "VS90COMNTOOLS",
    "VS100COMNTOOLS",
    "VS110COMNTOOLS",
    "VS120COMNTOOLS",
    "VS140COMNTOOLS",
    "VS150COMNTOOLS",
    "VS160COMNTOOLS",
    "TMP",
    "TEMP",
]

cc_autoconf = repository_rule(
    environ = [
        "ABI_LIBC_VERSION",
        "ABI_VERSION",
        "BAZEL_COMPILER",
        "BAZEL_HOST_SYSTEM",
        "BAZEL_CXXOPTS",
        "BAZEL_LINKOPTS",
        "BAZEL_LINKLIBS",
        "BAZEL_PYTHON",
        "BAZEL_SH",
        "BAZEL_TARGET_CPU",
        "BAZEL_TARGET_LIBC",
        "BAZEL_TARGET_SYSTEM",
        "BAZEL_USE_CPP_ONLY_TOOLCHAIN",
        "BAZEL_USE_XCODE_TOOLCHAIN",
        "BAZEL_DO_NOT_DETECT_CPP_TOOLCHAIN",
        "BAZEL_USE_LLVM_NATIVE_COVERAGE",
        "BAZEL_LLVM",
        "BAZEL_IGNORE_SYSTEM_HEADERS_VERSIONS",
        "USE_CLANG_CL",
        "CC",
        "CC_CONFIGURE_DEBUG",
        "CC_TOOLCHAIN_NAME",
        "CPLUS_INCLUDE_PATH",
        "GCOV",
        "HOMEBREW_RUBY_PATH",
        "SYSTEMROOT",
    ] + MSVC_ENVVARS,
    implementation = cc_autoconf_impl,
    configure = True,
)

# buildifier: disable=unnamed-macro
def cc_configure():
    """A C++ configuration rules that generate the crosstool file."""
    cc_autoconf_toolchains(name = "local_config_cc_toolchains")
    cc_autoconf(name = "local_config_cc")
    native.bind(name = "cc_toolchain", actual = "@local_config_cc//:toolchain")
    native.register_toolchains(
        # Use register_toolchain's target pattern expansion to register all toolchains in the package.
        "@local_config_cc_toolchains//:all",
    )
