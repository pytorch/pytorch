# Adopted from: https://github.com/NVIDIA/TRTorch/blob/master/third_party/cudnn/local/BUILD

cc_library(
    name = "cudnn_headers",
    hdrs = ["include/cudnn.h"] + glob([
        "include/cudnn+.h",
        "include/cudnn_*.h",
    ]),
    includes = ["include/"],
    visibility = ["//visibility:private"],
)

cc_import(
    name = "cudnn_lib",
    shared_library = "lib64/libcudnn.so",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "cudnn",
    visibility = ["//visibility:public"],
    deps = [
        "cudnn_headers",
        "cudnn_lib",
    ],
)
