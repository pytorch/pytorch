# pylint: disable=g-bad-file-header
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
"""Configuring the C++ toolchain on macOS."""

load("@bazel_tools//tools/osx:xcode_configure.bzl", "run_xcode_locator")
load(
    ":lib_cc_configure.bzl",
    "escape_string",
    "resolve_labels",
    "write_builtin_include_directory_paths",
)
load(
    ":unix_cc_configure.bzl",
    "configure_unix_toolchain",
    "get_env",
    "get_escaped_cxx_inc_directories",
)

def _get_escaped_xcode_cxx_inc_directories(repository_ctx, cc, xcode_toolchains):
    """Compute the list of default C++ include paths on Xcode-enabled darwin.

    Args:
      repository_ctx: The repository context.
      cc: The default C++ compiler on the local system.
      xcode_toolchains: A list containing the xcode toolchains available
    Returns:
      include_paths: A list of builtin include paths.
    """

    # TODO(cparsons): Falling back to the default C++ compiler builtin include
    # paths shouldn't be unnecessary once all actions are using xcrun.
    include_dirs = get_escaped_cxx_inc_directories(repository_ctx, cc, "-xc++")
    for toolchain in xcode_toolchains:
        include_dirs.append(escape_string(toolchain.developer_dir))

    # Assume that all paths that point to /Applications/ are built in include paths
    include_dirs.append("/Applications/")
    return include_dirs

def compile_cc_file(repository_ctx, src_name, out_name):
    xcrun_result = repository_ctx.execute([
        "env",
        "-i",
        "xcrun",
        "--sdk",
        "macosx",
        "clang",
        "-mmacosx-version-min=10.9",
        "-std=c++11",
        "-lc++",
        "-o",
        out_name,
        src_name,
    ], 30)
    if (xcrun_result.return_code != 0):
        error_msg = (
            "return code {code}, stderr: {err}, stdout: {out}"
        ).format(
            code = xcrun_result.return_code,
            err = xcrun_result.stderr,
            out = xcrun_result.stdout,
        )
        fail(out_name + " failed to generate. Please file an issue at " +
             "https://github.com/bazelbuild/bazel/issues with the following:\n" +
             error_msg)

def configure_osx_toolchain(repository_ctx, overriden_tools):
    """Configure C++ toolchain on macOS.

    Args:
      repository_ctx: The repository context.
      overriden_tools: dictionary of overriden tools.
    """
    paths = resolve_labels(repository_ctx, [
        "@rules_cc//cc/private/toolchain:osx_cc_wrapper.sh.tpl",
        "@rules_cc//cc/private/toolchain:libtool_check_unique.cc",
        "@bazel_tools//tools/objc:libtool.sh",
        "@bazel_tools//tools/objc:make_hashed_objlist.py",
        "@bazel_tools//tools/objc:xcrunwrapper.sh",
        "@bazel_tools//tools/osx/crosstool:BUILD.tpl",
        "@bazel_tools//tools/osx/crosstool:cc_toolchain_config.bzl",
        "@bazel_tools//tools/osx/crosstool:wrapped_clang.cc",
        "@bazel_tools//tools/osx:xcode_locator.m",
    ])

    env = repository_ctx.os.environ
    should_use_xcode = "BAZEL_USE_XCODE_TOOLCHAIN" in env and env["BAZEL_USE_XCODE_TOOLCHAIN"] == "1"
    xcode_toolchains = []

    # Make the following logic in sync with @rules_cc//cc/private/toolchain:cc_configure.bzl#cc_autoconf_toolchains_impl
    (xcode_toolchains, xcodeloc_err) = run_xcode_locator(
        repository_ctx,
        paths["@bazel_tools//tools/osx:xcode_locator.m"],
    )
    if should_use_xcode and not xcode_toolchains:
        fail("BAZEL_USE_XCODE_TOOLCHAIN is set to 1 but Bazel couldn't find Xcode installed on the " +
             "system. Verify that 'xcode-select -p' is correct.")
    if xcode_toolchains:
        # For Xcode toolchains, there's no reason to use anything other than
        # wrapped_clang, so that we still get the Bazel Xcode placeholder
        # substitution and other behavior for actions that invoke this
        # cc_wrapper.sh script. The wrapped_clang binary is already hardcoded
        # into the Objective-C crosstool actions, anyway, so this ensures that
        # the C++ actions behave consistently.
        cc = repository_ctx.path("wrapped_clang")

        cc_path = '"$(/usr/bin/dirname "$0")"/wrapped_clang'
        repository_ctx.template(
            "cc_wrapper.sh",
            paths["@rules_cc//cc/private/toolchain:osx_cc_wrapper.sh.tpl"],
            {
                "%{cc}": escape_string(cc_path),
                "%{env}": escape_string(get_env(repository_ctx)),
            },
        )
        repository_ctx.symlink(
            paths["@bazel_tools//tools/objc:xcrunwrapper.sh"],
            "xcrunwrapper.sh",
        )
        repository_ctx.symlink(
            paths["@bazel_tools//tools/objc:libtool.sh"],
            "libtool",
        )
        repository_ctx.symlink(
            paths["@bazel_tools//tools/objc:make_hashed_objlist.py"],
            "make_hashed_objlist.py",
        )
        repository_ctx.symlink(
            paths["@bazel_tools//tools/osx/crosstool:cc_toolchain_config.bzl"],
            "cc_toolchain_config.bzl",
        )
        libtool_check_unique_src_path = str(repository_ctx.path(
            paths["@rules_cc//cc/private/toolchain:libtool_check_unique.cc"],
        ))
        compile_cc_file(repository_ctx, libtool_check_unique_src_path, "libtool_check_unique")
        wrapped_clang_src_path = str(repository_ctx.path(
            paths["@bazel_tools//tools/osx/crosstool:wrapped_clang.cc"],
        ))
        compile_cc_file(repository_ctx, wrapped_clang_src_path, "wrapped_clang")
        repository_ctx.symlink("wrapped_clang", "wrapped_clang_pp")

        tool_paths = {}
        gcov_path = repository_ctx.os.environ.get("GCOV")
        if gcov_path != None:
            if not gcov_path.startswith("/"):
                gcov_path = repository_ctx.which(gcov_path)
            tool_paths["gcov"] = gcov_path

        escaped_include_paths = _get_escaped_xcode_cxx_inc_directories(repository_ctx, cc, xcode_toolchains)
        write_builtin_include_directory_paths(repository_ctx, cc, escaped_include_paths)
        escaped_cxx_include_directories = []
        for path in escaped_include_paths:
            escaped_cxx_include_directories.append(("        \"%s\"," % path))
        if xcodeloc_err:
            escaped_cxx_include_directories.append("# Error: " + xcodeloc_err + "\n")
        repository_ctx.template(
            "BUILD",
            paths["@bazel_tools//tools/osx/crosstool:BUILD.tpl"],
            {
                "%{cxx_builtin_include_directories}": "\n".join(escaped_cxx_include_directories),
                "%{tool_paths_overrides}": ",\n        ".join(
                    ['"%s": "%s"' % (k, v) for k, v in tool_paths.items()],
                ),
            },
        )
    else:
        configure_unix_toolchain(repository_ctx, cpu_value = "darwin", overriden_tools = overriden_tools)
