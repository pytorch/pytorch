# Adopted from: https://github.com/tensorflow/runtime/blob/master/third_party/rules_cuda/private/BUILD.local_cuda
# Library targets are created corresponding to BUILD.bazel's needs.

cc_library(
    name = "cuda_headers",
    hdrs = glob([
        "include/**",
        "targets/x86_64-linux/include/**",
    ]),
    includes = [
        "include",
        "targets/x86_64-linux/include",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_driver",
    srcs = ["lib64/stubs/libcuda.so"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda",
    srcs = ["targets/x86_64-linux/lib/libcudart.so"],
    visibility = ["//visibility:public"],
    deps = [":cuda_headers"],
)

cc_library(
    name = "cufft",
    srcs = ["targets/x86_64-linux/lib/libcufft.so"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cublas",
    srcs = [
        "targets/x86_64-linux/lib/libcublasLt.so",
        "targets/x86_64-linux/lib/libcublas.so",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "curand",
    srcs = ["targets/x86_64-linux/lib/libcurand.so"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cusolver",
    srcs = ["targets/x86_64-linux/lib/libcusolver.so"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cusparse",
    srcs = ["targets/x86_64-linux/lib/libcusparse.so"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cusparselt",
    srcs = ["targets/x86_64-linux/lib/libcusparseLt.so"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nvrtc",
    srcs = [
        "targets/x86_64-linux/lib/libnvrtc.so",
        "targets/x86_64-linux/lib/libnvrtc-builtins.so",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nvToolsExt",
    srcs = [ "lib64/libnvToolsExt.so"],
    visibility = ["//visibility:public"],
)
