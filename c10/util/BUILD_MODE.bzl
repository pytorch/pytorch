""" build mode definitions for caffe2/c10/util """

load("@fbcode//:BUILD_MODE.bzl", get_parent_modes = "get_empty_modes")
load("@fbcode_macros//build_defs:create_build_mode.bzl", "extend_build_modes")

_extra_cflags = [
]

_common_flags = [
]

_extra_clang_flags = _common_flags + [
    # Clang-specific warning flags (not supported by GCC)
    # The -pedantic flag enables strict ISO C++ compliance checks
    # GCC's -pedantic rejects __int128 which is used in folly headers
    "-pedantic",
]

_extra_gcc_flags = _common_flags + [
    # GCC doesn't work well with -pedantic due to __int128 usage in folly
]

_tags = [
]

_modes = extend_build_modes(
    get_parent_modes(),
    c_flags = _extra_cflags,
    clang_flags = _extra_clang_flags,
    gcc_flags = _extra_gcc_flags,
    tags = _tags,
)

def get_modes():
    """ Return modes for this file """
    return _modes
