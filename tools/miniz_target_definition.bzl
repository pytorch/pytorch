load("@fbcode_macros//build_defs:cpp_library.bzl", "cpp_library")
load("//caffe2/tools:sgx_target_definitions.bzl", "is_sgx")

def add_miniz_lib():
    cpp_library(
        name = "miniz",
        srcs = [
            "third_party/miniz-2.1.0/fb/FollyCrcPlugin.cpp",
            "third_party/miniz-2.1.0/fb/miniz-fb.c",
        ],
        headers = {
            "caffe2/third_party/miniz-2.1.0/miniz.c": "third_party/miniz-2.1.0/miniz.c",
            "miniz-fb.h": "third_party/miniz-2.1.0/fb/miniz-fb.h",
            "miniz.h": "third_party/miniz-2.1.0/miniz.h",
        },
        header_namespace = "",
        # -fexceptions is required, otherwise, when we use @mode/opt-clang-thinlto,
        # c functions become noexcept, and we may not be able to catch exceptions
        # during model loading.
        compiler_flags = ["-DUSE_EXTERNAL_MZCRC", "-fexceptions"] + (["-DMINIZ_NO_STDIO"] if is_sgx else []),
        # folly is only required as a dependency if USE_EXTERNAL_MZCRC
        # above is defined, and FollyCrcPlugin.cpp is added.
        # Neither are strictly needed, but run significantly faster.
        exported_deps = ["//folly/hash:checksum"],
    )
