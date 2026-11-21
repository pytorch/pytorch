load("//arvr/tools/build_defs:oxx.bzl", "oxx_static_library")
load("//xplat/caffe2/fb:c10_xplat.bzl", "C10_CPU_SUPPORTED_PLATFORMS", "C10_CUDA_SUPPORTED_PLATFORMS", "c10_subpackage_labels")

cpu_supported_platforms = C10_CPU_SUPPORTED_PLATFORMS

cuda_supported_platforms = C10_CUDA_SUPPORTED_PLATFORMS

# rocktenn apparently has its own copy of glog that comes with libmp.dll, so we
# had better not try to use glog from c10 lest the glog symbols not be eliminated.
C10_USE_GLOG = native.read_config("c10", "use_glog", "1") == "1"

# If you don't use any functionality that relies on static initializer in c10 (the
# most notable ones are the allocators), you can turn off link_whole this way.
# In practice, this is only used by rocktenn as well.
C10_LINK_WHOLE = native.read_config("c10", "link_whole", "1") == "1"

def define_c10_ovrsource(name, is_mobile):
    pp_flags = []
    if is_mobile:
        pp_flags.append("-DC10_MOBILE=1")
    if C10_USE_GLOG:
        pp_flags.append("-DC10_USE_GLOG")

    # The sources and most headers are in the subdirectories that are their own packages; see
    # `fb/c10_xplat.bzl`.
    oxx_static_library(
        name = name,
        compatible_with = cpu_supported_platforms,
        link_whole = C10_LINK_WHOLE,
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
            # For some godforsaken reason, this is always required even when not C10_USE_GLOG
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
