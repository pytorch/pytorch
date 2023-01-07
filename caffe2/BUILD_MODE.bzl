""" build mode definitions for caffe2/caffe2 """

load("@fbcode//:BUILD_MODE.bzl", get_parent_modes = "all_modes_keep_gpu_sections_all_modes_use_lld")
load("@fbcode_macros//build_defs:create_build_mode.bzl", "extend_build_mode")

def update_mode_struct(name, mode_struct):
    if name == "dev":
        return extend_build_mode(
            mode_struct,
            # TODO(ipbrady): Modules introduce floating point inaccuracies (T43879333)
            cxx_modules = False,
        )
    else:
        return mode_struct

_modes = {
    mode_name: update_mode_struct(mode_name, mode_struct)
    for mode_name, mode_struct in get_parent_modes().items()
}

def get_modes():
    """ Return modes for this file """
    return _modes
