load("//arvr/tools/build_defs:genrule_utils.bzl", "gen_cmake_header")
load("//arvr/tools/build_defs:oxx.bzl", "oxx_static_library")

cpu_supported_platforms = [
    "ovr_config//os:android",
    "ovr_config//os:iphoneos",
    "ovr_config//os:linux-x86_64",
    "ovr_config//os:macos",
    "ovr_config//os:windows-x86_64",
    "ovr_config//runtime:arm64-linux-ubuntu-neon",
]

cuda_supported_platforms = [
    "ovr_config//os:linux-cuda",
    "ovr_config//os:windows-cuda",
]

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
            "-DHAVE_MMAP=1",
            "-DHAVE_SHM_OPEN=1",
            "-DHAVE_SHM_UNLINK=1",
        ],
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
        deps = [
            "//third-party/rt",
        ] + select({
            "DEFAULT": [],
            "ovr_config//os:linux": [
                "//third-party/numactl:numactl",
            ],
        }),
        exported_deps = [
            ":ovrsource_c10_cmake_macros.h",
            "//arvr/third-party/gflags:gflags",
            "//third-party/cpuinfo:cpuinfo",
            "//third-party/fmt:fmt",
            "//third-party/glog:glog",
        ],
    )

def define_ovrsource_targets():
    # C10_CPU_TEST_FILES = native.glob([
    #     "test/core/*.cpp",
    #     "test/util/*.cpp",
    # ])

    # C10_GPU_TEST_FILES = native.glob([
    #     "cuda/test/**/*.cpp",
    # ])

    # C10_CPU_TEST_TARGETS = [
    #     ":" + paths.basename(test)[:-len(".cpp")] + "_ovrsource"
    #     for test in C10_CPU_TEST_FILES
    # ]

    # C10_GPU_TEST_TARGETS = [
    #     ":" + paths.basename(test)[:-len(".cpp")] + "_ovrsource"
    #     for test in C10_GPU_TEST_FILES
    # ]

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
        header = "c10/macros/cmake_macros.h",
        prefix = "ovrsource_c10_mobile_",
    )

    gen_cmake_header(
        src = "macros/cmake_macros.h.in",
        defines = common_c10_cmake_defines + non_mobile_c10_cmake_defines,
        header = "c10/macros/cmake_macros.h",
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
        # tests = C10_GPU_TEST_TARGETS,
        visibility = ["PUBLIC"],
        deps = [
            "//third-party/cuda:libcuda",
            "//third-party/cuda:libcudart",
        ],
        exported_deps = c10_cuda_macros + [
            ":c10_ovrsource",
        ],
    )

    # [
    #     oxx_test(
    #         name = paths.basename(test)[:-len(".cpp")] + "_ovrsource",
    #         srcs = [test],
    #         compatible_with = cpu_supported_platforms,
    #         compiler_flags = select({
    #             "DEFAULT": [],
    #             "ovr_config//compiler:cl": [
    #                 "/w",
    #             ],
    #             "ovr_config//compiler:clang": [
    #                 "-Wno-error",
    #                 "-Wno-self-assign-overloaded",
    #                 "-Wno-self-move",
    #                 "-Wno-shadow",
    #                 "-Wno-undef",
    #                 "-Wno-unused-function",
    #                 "-Wno-unused-variable",
    #             ],
    #         }),
    #         framework = "gtest",
    #         oncall = "ovrsource_pytorch",
    #         raw_headers = native.glob([
    #             "test/**/*.h",
    #         ]),
    #         deps = [
    #             ":c10_ovrsource",
    #         ],
    #     )
    #     for test in C10_CPU_TEST_FILES
    # ]

    # [
    #     oxx_test(
    #         name = paths.basename(test)[:-len(".cpp")] + "_ovrsource",
    #         srcs = [test],
    #         compatible_with = cuda_supported_platforms,
    #         compiler_flags = select({
    #             "DEFAULT": [],
    #             "ovr_config//compiler:cl": [
    #                 "/w",
    #             ],
    #             "ovr_config//compiler:clang": [
    #                 "-Wno-error",
    #             ],
    #         }),
    #         framework = "gtest",
    #         oncall = "ovrsource_pytorch",
    #         raw_headers = native.glob([
    #             "test/**/*.h",
    #         ]),
    #         runtime_shared_libraries = [
    #             "//third-party/cuda:cudart",
    #         ],
    #         deps = [
    #             ":c10_cuda_ovrsource",
    #         ],
    #     )
    #     for test in C10_GPU_TEST_FILES
    # ]
