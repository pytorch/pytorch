load("@fbcode_macros//build_defs:cpp_library.bzl", "cpp_library")
load("//caffe2/tools:sgx_target_definitions.bzl", "is_sgx")

def add_cpuinfo_lib():
    cpp_library(
        name = "cpuinfo",
        exported_deps = [
            "fbsource//third-party/cpuinfo_sgx:cpuinfo_coffeelake",
        ] if is_sgx else [
            "fbsource//third-party/cpuinfo:cpuinfo",
        ],
    )
