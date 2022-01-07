"""
Collect all the CUDA stuff from @local_config_cuda in a single target
for convenience.
"""

cc_library(
    name = "cuda",
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_cuda//cuda:cublas",
        "@local_config_cuda//cuda:cuda_driver",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "@local_config_cuda//cuda:cufft",
        "@local_config_cuda//cuda:curand",
    ],
)

cc_library(
    name = "cupti",
    deps = [
        "@local_config_cuda//cuda:cupti_headers",
        "@local_config_cuda//cuda:cupti_link",
    ],
)

[
    alias(
        name = lib,
        actual = "@local_config_cuda//cuda:{}".format(lib),
        visibility = ["//visibility:public"],
    )
    for lib in [
        "cublas",
        "cufft",
        "cusolver",
        "cusparse",
        "curand",
        "nvrtc",
        "cuda_driver",
        "nvToolsExt",
    ]
]
