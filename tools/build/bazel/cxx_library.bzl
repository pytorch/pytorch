load("@rules_cc//cc:defs.bzl", "cc_library")
load("//lib:visibility.bzl", "from_buck_visibility")

def cxx_library(
        name,
        exported_headers = None,
        headers = None,
        visibility = None):
    srcs = []
    if type(headers) == "dict":
        srcs.extend(headers.values())
    cc_library(
        name = name,
        visibility = from_buck_visibility(visibility),
        hdrs = exported_headers,
        srcs = srcs,
    )
