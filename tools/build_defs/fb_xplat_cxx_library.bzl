# Only used for PyTorch open source BUCK build
# @lint-ignore-every BUCKRESTRICTEDSYNTAX

load(
    ":buck_helpers.bzl",
    "filter_attributes",
)

def fb_xplat_cxx_library(
        name,
        extra_flags = {},
        **kwgs):
    if read_config("pt", "is_oss", "0") == "0":
        fail("This file is for open source pytorch build. Do not use it in fbsource!")

    cxx_library(
        name = name,
        **filter_attributes(kwgs)
    )
