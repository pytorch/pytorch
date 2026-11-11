load("//arvr/tools/build_defs:oxx.bzl", "oxx_static_library")
load("//xplat/caffe2/fb:c10_xplat.bzl", "C10_CPU_SUPPORTED_PLATFORMS", "C10_CUDA_SUPPORTED_PLATFORMS", "c10_subpackage_labels")

cpu_supported_platforms = C10_CPU_SUPPORTED_PLATFORMS

cuda_supported_platforms = C10_CUDA_SUPPORTED_PLATFORMS

def define_c10_ovrsource(name, is_mobile):
    if is_mobile:
        pp_flags = ["-DC10_MOBILE=1"]
    else:
        pp_flags = []

    # The sources and most headers are in the subdirectories that are their own packages; see
    # `fb/c10_xplat.bzl`.
    oxx_static_library(
        name = name,
        compatible_with = cpu_supported_platforms,
        compiler_flags = select({
            "DEFAULT": [],
            "ovr_config//compiler:cl": [
                "/w",
            ],
            "ovr_config//toolchain/clang:win": [
                "-Wno-error",
                "-Wno-shadow",
                "-Wno-undef",
                "-Wno-unused-variable",
            ],
        }),
        include_directories = [".."],
        preprocessor_flags = [
            "-DNO_EXPORT",
            "-DC10_BUILD_MAIN_LIB=1",
            "-DSUPPORTS_BACKTRACE=0",
        ],
        fbobjc_compiler_flags = ["-Wno-error=global-constructors", "-Wno-error=missing-prototypes"],
        public_include_directories = [".."],
        public_preprocessor_flags = pp_flags,
        reexport_all_header_dependencies = False,
        visibility = [
            "//xplat/caffe2/c10:c10_ovrsource",
        ],
        deps = select({
            "DEFAULT": [],
            "ovr_config//os:linux": [
                "//third-party/numactl:numactl",
            ],
        }),
        exported_deps = c10_subpackage_labels("all_mobile_ovrsource" if is_mobile else "all_full_ovrsource") + [
            "//xplat/caffe2/c10/test:util_headers_ovrsource",  # some external tests use these
            "//xplat/caffe2/torch/headeronly:torch_headeronly_ovrsource",
            "//arvr/third-party/gflags:gflags",
            "//third-party/cpuinfo:cpuinfo",
            "//third-party/fmt:fmt",
            "//third-party/glog:glog",
        ],
    )

def define_ovrsource_targets():
    oxx_static_library(
        name = "c10_ovrsource",
        compatible_with = cpu_supported_platforms,
        exported_deps = select({
            "DEFAULT": [":c10_full_ovrsource"],
            "ovr_config//os:android": [":c10_mobile_ovrsource"],
            "ovr_config//os:iphoneos": [":c10_mobile_ovrsource"],
        }),
        visibility = ["PUBLIC"],
    )

    """
    Most users should use c10_ovrsource, not these targets directly.
    """
    define_c10_ovrsource("c10_mobile_ovrsource", True)
    define_c10_ovrsource("c10_full_ovrsource", False)

    native.alias(
        name = "c10_cuda_ovrsource",
        actual = "//xplat/caffe2/c10/cuda:c10_cuda_ovrsource",
        visibility = ["PUBLIC"],
    )
