# Only used for PyTorch open source BUCK build
# @lint-ignore-every BUCKRESTRICTEDSYNTAX
load(":buck_helpers.bzl", "filter_attributes")

def fb_xplat_cxx_test(
        name,
        deps = [],
        **kwgs):
    if read_config("pt", "is_oss", "0") == "0":
        fail("This file is for open source pytorch build. Do not use it in fbsource!")

    cxx_test(
        name = name,
        deps = deps + [
            "//third_party:gtest",
        ],
        **filter_attributes(kwgs)
    )
