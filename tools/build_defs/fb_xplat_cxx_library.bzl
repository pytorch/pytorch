# Only used for PyTorch open source BUCK build
# @lint-ignore-every BUCKRESTRICTEDSYNTAX

load(":buck_helpers.bzl", "filter_attributes", "to_oss_deps")

def fb_xplat_cxx_library(
        name,
        deps = [],
        exported_deps = [],
        **kwgs):
    if read_config("pt", "is_oss", "0") == "0":
        fail("This file is for open source pytorch build. Do not use it in fbsource!")

    cxx_library(
        name = name,
        deps = to_oss_deps(deps),
        exported_deps = to_oss_deps(exported_deps),
        **filter_attributes(kwgs)
    )
