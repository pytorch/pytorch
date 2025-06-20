load("//arvr/tools/build_defs:genrule_utils.bzl", "gen_cmake_header")
load("//arvr/tools/build_defs:oxx.bzl", "oxx_static_library")


def define_c10_ovrsource(name, is_mobile):
    if is_mobile:
        pp_flags = ["-DC10_MOBILE=1"]
    else:
        pp_flags = []

    oxx_static_library(
        name = name,
        srcs = native.glob([
            "core/*.cpp",
            "core/impl/*.cpp",
            "mobile/*.cpp",
            "util/*.cpp",
        ]),
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
        # tests = C10_CPU_TEST_TARGETS,
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
            "//xplat/caffe2:torch_standalone_headers",
            "//arvr/third-party/gflags:gflags",
            "//third-party/cpuinfo:cpuinfo",
            "//third-party/fmt:fmt",
            "//third-party/glog:glog",
        ],
    )


def define_ovrsource_targets():
    common_c10_cmake_defines = [
        ("#cmakedefine C10_BUILD_SHARED_LIBS", ""),
        ("#cmakedefine C10_USE_NUMA", ""),
        ("#cmakedefine C10_USE_MSVC_STATIC_RUNTIME", ""),
        ("#cmakedefine C10_USE_ROCM_KERNEL_ASSERT", ""),
    ]

    mobile_c10_cmake_defines = [
        ("#cmakedefine C10_USE_GLOG", ""),
        ("#cmakedefine C10_USE_GFLAGS", ""),
    ]

    non_mobile_c10_cmake_defines = [
        ("#cmakedefine C10_USE_GLOG", "#define C10_USE_GLOG 1"),
        ("#cmakedefine C10_USE_GFLAGS", "#define C10_USE_GFLAGS 1"),
    ]

    gen_cmake_header(
        src = "macros/cmake_macros.h.in",
        defines = common_c10_cmake_defines + mobile_c10_cmake_defines,
        header = "torch/standalone/macros/cmake_macros.h",
        prefix = "ovrsource_c10_mobile_",
    )

    gen_cmake_header(
        src = "macros/cmake_macros.h.in",
        defines = common_c10_cmake_defines + non_mobile_c10_cmake_defines,
        header = "torch/standalone/macros/cmake_macros.h",
        prefix = "ovrsource_c10_non_mobile_",
    )

    oxx_static_library(
        name = "ovrsource_c10_cmake_macros.h",
        compatible_with = [
            "ovr_config//os:android",
            "ovr_config//os:iphoneos",
            "ovr_config//os:linux",
            "ovr_config//os:macos",
            "ovr_config//os:windows",
        ],
        deps = select({
            "ovr_config//os:android": [":ovrsource_c10_mobile_cmake_macros.h"],
            "ovr_config//os:iphoneos": [":ovrsource_c10_mobile_cmake_macros.h"],
            "ovr_config//os:linux": [":ovrsource_c10_non_mobile_cmake_macros.h"],
            "ovr_config//os:macos": [":ovrsource_c10_non_mobile_cmake_macros.h"],
            "ovr_config//os:windows": [":ovrsource_c10_non_mobile_cmake_macros.h"],
        }),
    )
