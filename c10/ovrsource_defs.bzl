load("//arvr/tools/build_defs:genrule_utils.bzl", "gen_cmake_header")
load("//arvr/tools/build_defs:oxx.bzl", "oxx_static_library")

cpu_supported_platforms = [
    "ovr_config//os:android",
    "ovr_config//os:iphoneos",
    "ovr_config//os:linux-x86_64",
    "ovr_config//os:macos",
    "ovr_config//os:windows-x86_64",
    "ovr_config//runtime:arm64-linux-ubuntu-neon",
    "ovr_config//os:linux-arm64",
]

cuda_supported_platforms = [
    "ovr_config//os:linux-cuda",
    "ovr_config//os:windows-cuda",
]

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

    oxx_static_library(
        name = name,
        srcs = native.glob([
            "core/*.cpp",
            "core/impl/*.cpp",
            "mobile/*.cpp",
            "util/*.cpp",
        ]),
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
        public_raw_headers = native.glob([
            "core/*.h",
            "macros/*.h",
            "mobile/*.h",
            "test/util/*.h",  # some external tests use this
            "util/*.h",
        ]),
        raw_headers = native.glob([
            "core/impl/*.h",
        ]),
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
        exported_deps = [
            "//xplat/caffe2/torch/headeronly:torch_headeronly_ovrsource",
            "//arvr/third-party/gflags:gflags",
            "//third-party/cpuinfo:cpuinfo",
            "//third-party/fmt:fmt",
            # For some godforsaken reason, this is always required even when not C10_USE_GLOG
            "//third-party/glog:glog",
        ],
    )

def define_ovrsource_targets():
    c10_cuda_macros = gen_cmake_header(
        src = "cuda/impl/cuda_cmake_macros.h.in",
        defines = [
            ("#cmakedefine C10_CUDA_BUILD_SHARED_LIBS", ""),
        ],
        header = "c10/cuda/impl/cuda_cmake_macros.h",
        prefix = "ovrsource",
    )

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

    oxx_static_library(
        name = "c10_cuda_ovrsource",
        srcs = native.glob([
            "cuda/*.cpp",
            "cuda/impl/*.cpp",
        ]),
        compatible_with = cuda_supported_platforms,
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
        link_whole = True,
        preprocessor_flags = [
            "-DNO_EXPORT",
            "-DC10_CUDA_BUILD_MAIN_LIB=1",
        ],
        raw_headers = native.glob([
            "cuda/*.h",
            "cuda/impl/*.h",
        ]),
        reexport_all_header_dependencies = False,
        visibility = ["PUBLIC"],
        deps = [
            "//third-party/cuda:libcuda",
            "//third-party/cuda:libcudart",
        ],
        exported_deps = c10_cuda_macros + [
            ":c10_ovrsource",
        ],
    )
