load("//tools/build_defs:fb_xplat_cxx_library.bzl", "fb_xplat_cxx_library")
load("//tools/build_defs:fbsource_utils.bzl", "is_arvr_mode")
load("//tools/build_defs:glob_defs.bzl", "subdir_glob")
load("//tools/build_defs:platform_defs.bzl", "ANDROID", "APPLE", "CXX", "IOS", "MACOSX", "WINDOWS")
load(
    "@fbsource//xplat/caffe2/third_party:xnnpack_buck_shim.bzl",
    "LOGGING_SRCS",
    "OPERATOR_SRCS",
    "SUBGRAPH_SRCS",
    "TABLE_SRCS",
    "XNNPACK_SRCS",
    "get_xnnpack_headers",
    "prod_srcs_for_arch_wrapper",
)

XNN_COMMON_PREPROCESSOR_FLAGS = [
    "-DXNN_PRIVATE=",
    "-DXNN_INTERNAL=",
    "-DXNN_LOG_LEVEL=0"
]

# This defines XNNPACK targets for both fbsource BUCK and OSS BUCK
# Note that the file path is relative to the BUCK file that called from, not to this bzl file.
# So for fbsource build it points to xplat/third-party/XNNPACK/XNNPACK,
# and for OSS it points to pytorch/third_party/XNNPACK
def define_xnnpack(third_party, labels = [], XNNPACK_WINDOWS_AVX512F_ENABLED = False):
    WINDOWS_FLAGS = [
        "/D__x86_64__",
        "/EHsc",
        "/wd4090",  # 'function': different 'const' qualifiers
        "/wd4146",  # unary minus operator applied to unsigned type, result still unsigned
    ] + ([
        "/D__AVX512F__",  # needed to avoid linkage errors
        "-mavx2",
        "/D__builtin_clz=__lzcnt",  # Intrinsics are spelled differently in MSVC
        "/Drestrict=",  # MSVC doesn't understand [restrict XNN_NUM_ELEMENTS(N)] syntax
    ] if XNNPACK_WINDOWS_AVX512F_ENABLED else [])

    WINDOWS_CLANG_COMPILER_FLAGS = [
        "-Wno-error",
        "-Wno-error=undef",
        "-Wno-error=incompatible-pointer-types",
        "-Wno-error=incompatible-pointer-types-discards-qualifiers",
    ]

    XNN_COMMON_MICROKERNEL_EXPORTED_DEPS = [
        ":interface",
        third_party("FP16"),
        third_party("FXdiv"),
    ]

    fb_xplat_cxx_library(
        name = "interface",
        header_namespace = "",
        exported_headers = {
            "xnnpack.h": "XNNPACK/include/xnnpack.h",
        },
        apple_sdks = (IOS, MACOSX),
        labels = labels,
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        exported_deps = [
            # Dependency only on pthreadpool interface
            third_party("pthreadpool_header"),
        ],
    )

    fb_xplat_cxx_library(
        name = "subgraph",
        srcs = SUBGRAPH_SRCS + ["XNNPACK/src/datatype.c"],
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ],
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS + [
            "-DXNN_ENABLE_SPARSE=0",
            "-DXNN_ENABLE_MEMOPT",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = [
            ":interface",
            third_party("FP16"),
            third_party("FXdiv"),
            third_party("clog"),
        ],
    )

    fb_xplat_cxx_library(
        name = "tables",
        srcs = TABLE_SRCS,
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ],
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = [
            ":interface",
            third_party("FP16"),
            third_party("FXdiv"),
            third_party("clog"),
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_scalar",
        srcs = prod_srcs_for_arch_wrapper("scalar"),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
            "-fno-fast-math",
            "-fno-math-errno",
            "-ffp-contract=off",
        ],
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_sse",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("sse"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("sse"),
            "ovr_config//runtime:fbcode": prod_srcs_for_arch_wrapper("sse"),

        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-msse",
            ],
            "ovr_config//cpu:x86_64": [
                "-msse",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-msse"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-msse"],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_sse_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-msse",
            ],
            "ovr_config//cpu:x86_64": [
                "-msse",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-msse"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-msse"],
        windows_srcs = prod_srcs_for_arch_wrapper("sse"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_sse2",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("sse2"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("sse2"),
            "ovr_config//runtime:fbcode": prod_srcs_for_arch_wrapper("sse2"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-msse2",
            ],
            "ovr_config//cpu:x86_64": [
                "-msse2",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-msse2"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-msse2"],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_sse2_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-msse2",
            ],
            "ovr_config//cpu:x86_64": [
                "-msse2",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-msse2"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-msse2"],
        windows_srcs = prod_srcs_for_arch_wrapper("sse2"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_ssse3",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("ssse3"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("ssse3"),
            "ovr_config//runtime:fbcode": prod_srcs_for_arch_wrapper("ssse3"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mssse3",
            ],
            "ovr_config//cpu:x86_64": [
                "-mssse3",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mssse3"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mssse3"],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_ssse3_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mssse3",
            ],
            "ovr_config//cpu:x86_64": [
                "-mssse3",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mssse3"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mssse3"],
        windows_srcs = prod_srcs_for_arch_wrapper("ssse3"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_sse41",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("sse41"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("sse41"),
            "ovr_config//runtime:fbcode": prod_srcs_for_arch_wrapper("sse41"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-msse4.1",
            ],
            "ovr_config//cpu:x86_64": [
                "-msse4.1",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-msse4.1"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-msse4.1"],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_sse41_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-msse4.1",
            ],
            "ovr_config//cpu:x86_64": [
                "-msse4.1",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-msse4.1"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-msse4.1"],
        windows_srcs = prod_srcs_for_arch_wrapper("sse41"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("avx"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("avx"),
            "ovr_config//runtime:fbcode": prod_srcs_for_arch_wrapper("avx"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mavx",
            ],
            "ovr_config//cpu:x86_64": [
                "-mavx",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mavx"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mavx"],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
            "-mavx",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mavx",
            ],
            "ovr_config//cpu:x86_64": [
                "-mavx",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mavx"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mavx"],
        windows_srcs = prod_srcs_for_arch_wrapper("avx"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx512vnnigfni",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("avx512vnnigfni"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("avx512vnnigfni"),
            "ovr_config//runtime:fbcode": prod_srcs_for_arch_wrapper("avx512vnnigfni"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mavx",
                "-mgfni",
                "-mavx512vl",
                "-mavx512vnni",
                "-mavx512bw",
                "-mavx512dq",
            ],
            "ovr_config//cpu:x86_64": [
                "-mavx",
                "-mgfni",
                "-mavx512vl",
                "-mavx512vnni",
                "-mavx512bw",
                "-mavx512dq",
            ],
        }) + select({
            "DEFAULT": [],
            "ovr_config//os:windows": WINDOWS_FLAGS + ["-mavx"] + select({
                "DEFAULT": [],
                "ovr_config//compiler:clang-windows": WINDOWS_CLANG_COMPILER_FLAGS,
            }),
        }),
        labels = labels,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx512vnnigfni_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mavx512f",
                "-mavx512cd",
                "-mavx512bw",
                "-mavx512dq",
                "-mavx512vl",
                "-mavx512vnni",
                "-mgfni",
            ],
            "ovr_config//cpu:x86_64": [
                "-mavx512f",
                "-mavx512cd",
                "-mavx512bw",
                "-mavx512dq",
                "-mavx512vl",
                "-mavx512vnni",
                "-mgfni",
            ],
        }) + select({
            "DEFAULT": [],
            "ovr_config//os:windows": WINDOWS_FLAGS + ["-mavx"] + select({
                "DEFAULT": [],
                "ovr_config//compiler:clang-windows": WINDOWS_CLANG_COMPILER_FLAGS,
            }),
        }),
        labels = labels,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_srcs = prod_srcs_for_arch_wrapper("avx512vnnigfni"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx512vnni",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("avx512vnni"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("avx512vnni"),
            "ovr_config//runtime:fbcode": prod_srcs_for_arch_wrapper("avx512vnni"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mavx512f",
                "-mavx512cd",
                "-mavx512bw",
                "-mavx512dq",
                "-mavx512vl",
                "-mavx512vnni",
            ],
            "ovr_config//cpu:x86_64": [
                "-mavx512f",
                "-mavx512cd",
                "-mavx512bw",
                "-mavx512dq",
                "-mavx512vl",
                "-mavx512vnni",
            ],
        }) + select({
            "DEFAULT": [],
            "ovr_config//os:windows": WINDOWS_FLAGS + ["-mavx"] + select({
                "DEFAULT": [],
                "ovr_config//compiler:clang-windows": WINDOWS_CLANG_COMPILER_FLAGS,
            }),
        }),
        labels = labels,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        exported_preprocessor_flags = [
            "-DXNN_ENABLE_AVX512VNNI"
        ],
        visibility = ["PUBLIC"],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx512vnni_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mavx512f",
                "-mavx512cd",
                "-mavx512bw",
                "-mavx512dq",
                "-mavx512vl",
                "-mavx512vnni",
            ],
            "ovr_config//cpu:x86_64": [
                "-mavx512f",
                "-mavx512cd",
                "-mavx512bw",
                "-mavx512dq",
                "-mavx512vl",
                "-mavx512vnni",
            ],
        }) + select({
            "DEFAULT": [],
            "ovr_config//os:windows": WINDOWS_FLAGS + ["-mavx"] + select({
                "DEFAULT": [],
                "ovr_config//compiler:clang-windows": WINDOWS_CLANG_COMPILER_FLAGS,
            }),
        }),
        labels = labels,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        exported_preprocessor_flags = [
            "-DXNN_ENABLE_AVX512VNNI"
        ],
        visibility = ["PUBLIC"],
        windows_srcs = prod_srcs_for_arch_wrapper("avx512vnni"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avxvnni",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("avxvnni"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("avxvnni"),
            "ovr_config//runtime:fbcode": prod_srcs_for_arch_wrapper("avxvnni"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
            "-mavxvnni",
            "-mf16c",
            "-mfma",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mavx2",
                "-mavxvnni",
                "-mf16c",
                "-mfma",
            ],
            "ovr_config//cpu:x86_64": [
                "-mavx2",
                "-mavxvnni",
                "-mf16c",
                "-mfma",
            ],
        }),
        labels = labels,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mavx"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mavx"],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avxvnni_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mavx2",
                "-mavxvnni",
            ],
            "ovr_config//cpu:x86_64": [
                "-mavx2",
                "-mavxvnni",
            ],
        }),
        labels = labels,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mavx"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mavx"],
        windows_srcs = prod_srcs_for_arch_wrapper("avxvnni"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_f16c",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("f16c"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("f16c"),
            "ovr_config//runtime:fbcode": prod_srcs_for_arch_wrapper("f16c"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mf16c",
            ],
            "ovr_config//cpu:x86_64": [
                "-mf16c",
            ],
        }),
        labels = labels,
        platforms = (APPLE, ANDROID, CXX, WINDOWS),
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mf16c"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mf16c"],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_f16c_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
            "-mf16c",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mf16c",
            ],
            "ovr_config//cpu:x86_64": [
                "-mf16c",
            ],
        }),
        labels = labels,
        platforms = (APPLE, ANDROID, CXX, WINDOWS),
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mf16c"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mf16c"],
        windows_srcs = prod_srcs_for_arch_wrapper("f16c"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_fma3",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("fma3"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("fma3"),
            "ovr_config//runtime:fbcode": prod_srcs_for_arch_wrapper("fma3"),

        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mfma",
                "-mf16c",
            ],
            "ovr_config//cpu:x86_64": [
                "-mfma",
                "-mf16c",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + [
            "-mfma",
            "-mf16c",
        ],
        windows_compiler_flags_override = WINDOWS_FLAGS + [
            "-mfma",
            "-mf16c",
        ],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_fma3_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
            "-mfma",
            "-mf16c",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mfma",
                "-mf16c",
            ],
            "ovr_config//cpu:x86_64": [
                "-mfma",
                "-mf16c",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + [
            "-mfma",
            "-mf16c",
        ],
        windows_compiler_flags_override = WINDOWS_FLAGS + [
            "-mfma",
            "-mf16c",
        ],
        windows_srcs = prod_srcs_for_arch_wrapper("fma3"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx2",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("avx2"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("avx2"),
            "ovr_config//runtime:fbcode": prod_srcs_for_arch_wrapper("avx2"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mavx2",
                "-mfma",
                "-mf16c",
            ],
            "ovr_config//cpu:x86_64": [
                "-mavx2",
                "-mfma",
                "-mf16c",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + [
            "-mavx2",
            "-mfma",
            "-mf16c",
        ],
        windows_compiler_flags_override = WINDOWS_FLAGS + [
            "-mavx2",
            "-mfma",
            "-mf16c",
        ],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx2_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
            "-mavx2",
            "-mfma",
            "-mf16c",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mavx2",
                "-mfma",
                "-mf16c",
            ],
            "ovr_config//cpu:x86_64": [
                "-mavx2",
                "-mfma",
                "-mf16c",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + [
            "-mavx2",
            "-mfma",
            "-mf16c",
        ],
        windows_compiler_flags_override = WINDOWS_FLAGS + [
            "/D__AVX2__",
            "-mavx2",
            "-mfma",
            "-mf16c",
        ],
        windows_srcs = prod_srcs_for_arch_wrapper("avx2"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx512",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("avx512f"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("avx512f"),
            "ovr_config//runtime:fbcode": prod_srcs_for_arch_wrapper("avx512f"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mavx512f",
            ],
            "ovr_config//cpu:x86_64": [
                "-mavx512f",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mavx512f"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mavx512f"],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx512vbmi",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("avx512vbmi"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("avx512vbmi"),
            "ovr_config//runtime:fbcode": prod_srcs_for_arch_wrapper("avx512vbmi"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mavx512f",
                "-mavx512cd",
                "-mavx512bw",
                "-mavx512dq",
                "-mavx512vl",
                "-mavx512vbmi",
            ],
            "ovr_config//cpu:x86_64": [
                "-mavx512f",
                "-mavx512cd",
                "-mavx512bw",
                "-mavx512dq",
                "-mavx512vl",
                "-mavx512vbmi",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + [
            "-mavx512f",
            "-mavx512cd",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",
            "-mavx512vbmi",
        ],
        windows_compiler_flags_override = WINDOWS_FLAGS + [
            "-mavx512f",
            "-mavx512cd",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",
            "-mavx512vbmi",
        ],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx512_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
            "-mavx512f",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mavx512f",
            ],
            "ovr_config//cpu:x86_64": [
                "-mavx512f",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mavx512f"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mavx512f"],
        windows_srcs = prod_srcs_for_arch_wrapper("avx512f"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx512skx",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("avx512skx"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("avx512skx"),
            "ovr_config//runtime:fbcode": prod_srcs_for_arch_wrapper("avx512skx"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mavx512f",
                "-mavx512cd",
                "-mavx512bw",
                "-mavx512dq",
                "-mavx512vl",
            ],
            "ovr_config//cpu:x86_64": [
                "-mavx512f",
                "-mavx512cd",
                "-mavx512bw",
                "-mavx512dq",
                "-mavx512vl",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + [
            "-mavx512f",
            "-mavx512cd",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",
        ],
        windows_compiler_flags_override = WINDOWS_FLAGS + [
            "-mavx512f",
            "-mavx512cd",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",

        ],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx512skx_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
            "-mavx512f",
            "-mavx512cd",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mavx512f",
                "-mavx512cd",
                "-mavx512bw",
                "-mavx512dq",
                "-mavx512vl",
            ],
            "ovr_config//cpu:x86_64": [
                "-mavx512f",
                "-mavx512cd",
                "-mavx512bw",
                "-mavx512dq",
                "-mavx512vl",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + [
            "-mavx512f",
            "-mavx512cd",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",
        ],
        windows_compiler_flags_override = WINDOWS_FLAGS + [
            "-mavx512f",
            "-mavx512cd",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",
            "/D__AVX512BW__",
        ],
        windows_srcs = prod_srcs_for_arch_wrapper("avx512skx"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_armsimd32",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": prod_srcs_for_arch_wrapper("armsimd32"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
            "-fno-fast-math",
            "-fno-math-errno",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": [
                "-marm",
                "-march=armv6",
                "-mfpu=vfp",
                "-munaligned-access",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_neon",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": prod_srcs_for_arch_wrapper("neon"),
            "ovr_config//cpu:arm64": prod_srcs_for_arch_wrapper("neon"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": [
                "-marm",
                "-march=armv7-a",
                "-mfpu=neon",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_neon_aarch64",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm64": prod_srcs_for_arch_wrapper("neon_aarch64"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ],
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_neon_fma",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": prod_srcs_for_arch_wrapper("neonfma"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": [
                "-marm",
                "-march=armv7-a",
                "-mfpu=neon-vfpv4",
            ],
        }) + select({
            "DEFAULT": [],
            "ovr_config//os:iphoneos-arm32": [
                "-mcpu=cyclone",
                "-mtune=generic",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_neonfma_aarch64",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm64": prod_srcs_for_arch_wrapper("neonfma") + prod_srcs_for_arch_wrapper("neonfma_aarch64"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ],
        labels = labels,
        platforms = (APPLE, ANDROID, CXX, WINDOWS),
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_fp16arith",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": prod_srcs_for_arch_wrapper("fp16arith"),
            "ovr_config//cpu:arm64": prod_srcs_for_arch_wrapper("fp16arith") + prod_srcs_for_arch_wrapper("fp16arith_aarch64"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
            "-fno-fast-math",
            "-fno-math-errno",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": [
                "-marm",
                "-march=armv8.2-a+fp16",
                # GCC emits wrong directives for assembler with -mfpu=fp-armv8
                "-mfpu=neon-fp-armv8",
                # For vsqrth_f16 polyfill using sqrtf
                "-fno-math-errno",
                # For vminh_f16/vmaxh_f16 polyfills using compare + select
                "-ffinite-math-only",
            ],
            "ovr_config//cpu:arm64": [
                "-march=armv8.2-a+fp16",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_neon_fp16",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": prod_srcs_for_arch_wrapper("neonfp16"),
            "ovr_config//cpu:arm64": prod_srcs_for_arch_wrapper("neonfp16"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": [
                "-marm",
                "-march=armv7-a",
                "-mfpu=neon-fp16",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_neon_v8",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": prod_srcs_for_arch_wrapper("neonv8"),
            "ovr_config//cpu:arm64": prod_srcs_for_arch_wrapper("neonv8"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm64": ["-march=armv8-a"],
            "ovr_config//cpu:arm32": [
                "-marm",
                "-march=armv8-a",
                "-mfpu=neon-fp-armv8",
            ],
        }) + select({
            "DEFAULT": [],
            "ovr_config//os:android-arm32": [
                "-marm",
                "-march=armv8-a",
                "-mfpu=neon-fp-armv8",
            ],
        }) + select({
            "DEFAULT": [],
            "ovr_config//os:iphoneos-arm32": [
                "-mcpu=cyclone",
                "-mtune=generic",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_neon_dot",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": prod_srcs_for_arch_wrapper("neondot"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": [
                "-march=armv8.2-a+dotprod",
                "-mfpu=neon-fp-armv8",
                "-mfloat-abi=softfp",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_neon_dot_aarch64",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm64": prod_srcs_for_arch_wrapper("neondot") + prod_srcs_for_arch_wrapper("neondot_aarch64"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm64": ["-march=armv8.2-a+dotprod"],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_neon_dot_fp16arith",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": prod_srcs_for_arch_wrapper("neondotfp16arith"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": [
                "-marm",
                "-march=armv8.2-a+dotprod+fp16",
                "-mfpu=neon-fp-armv8",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_neon_dot_fp16arith_aarch64",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm64": prod_srcs_for_arch_wrapper("neondotfp16arith") + prod_srcs_for_arch_wrapper("neondotfp16arith_aarch64"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm64": [
                "-march=armv8.2-a+dotprod+fp16",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_neon_fp16arith",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": prod_srcs_for_arch_wrapper("neonfp16arith"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": [
                "-marm",
                "-march=armv8.2-a+fp16",
                "-mfpu=neon-fp-armv8",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_neon_fp16arith_aarch64",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm64": prod_srcs_for_arch_wrapper("neonfp16arith") + prod_srcs_for_arch_wrapper("neonfp16arith_aarch64"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm64": ["-march=armv8.2-a+fp16"],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_neonfma_i8mm",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": prod_srcs_for_arch_wrapper("neonfma_i8mm"),
            "ovr_config//cpu:arm64": prod_srcs_for_arch_wrapper("neonfma_i8mm"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": [
                "-marm",
                "-march=armv8.2-a+i8mm+fp16",
                "-mfpu=neon-fp-armv8",
            ],
            "ovr_config//cpu:arm64": [
                "-march=armv8.2-a+i8mm+fp16",
            ],
        }),
        labels = labels,
        platforms = (APPLE, ANDROID, CXX, WINDOWS),
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_neoni8mm",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm64": prod_srcs_for_arch_wrapper("neoni8mm"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm64": [
                "-march=armv8.2-a+i8mm+fp16",
            ],
        }),
        labels = labels,
        platforms = (APPLE, ANDROID, CXX, WINDOWS),
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_asm_aarch32",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": prod_srcs_for_arch_wrapper("aarch32"),
        }),
        headers = subdir_glob([
            ("XNNPACK/src", "xnnpack/assembly.h"),
            ("XNNPACK/src", "**/*.S"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": [
                "-marm",
                "-march=armv8.2-a+dotprod+fp16",
                "-mfpu=neon-fp-armv8",
            ],
        }),
        labels = labels,
        platforms = (APPLE, ANDROID, CXX, WINDOWS),
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_asm_aarch64",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm64": prod_srcs_for_arch_wrapper("aarch64"),
        }),
        headers = subdir_glob([
            ("XNNPACK/src", "xnnpack/assembly.h"),
            ("XNNPACK/src", "**/*.S"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm64": [
                "-march=armv8.2-a+fp16+dotprod",
            ],
        }),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "arm64_lib",
        apple_sdks = (IOS, MACOSX),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        visibility = ["PUBLIC"],
        deps = [
            ":ukernels_asm_aarch64",
            ":ukernels_neon",
            ":ukernels_neon_aarch64",
            ":ukernels_neon_dot_fp16arith",
            ":ukernels_neon_dot_fp16arith_aarch64",
            ":ukernels_neon_dot",
            ":ukernels_neon_dot_aarch64",
            ":ukernels_neon_fma",
            ":ukernels_neon_fp16",
            ":ukernels_neon_fp16arith",
            ":ukernels_neon_fp16arith_aarch64",
            ":ukernels_neon_v8",
            ":ukernels_neonfma_aarch64",
            ":ukernels_neonfma_i8mm",
            ":ukernels_neoni8mm",
        ],
    )

    fb_xplat_cxx_library(
        name = "x86_and_x86_64_lib",
        apple_sdks = (IOS, MACOSX),
        labels = labels,
        preferred_linkage = "static",
        visibility = ["PUBLIC"],
        deps = [
            ":ukernels_avx",
            ":ukernels_avx2",
            ":ukernels_avx512",
            ":ukernels_avx512skx",
            ":ukernels_f16c",
            ":ukernels_fma3",
            ":ukernels_sse",
            ":ukernels_sse2",
            ":ukernels_sse41",
            ":ukernels_ssse3",
            ":ukernels_avx512vbmi",
            ":ukernels_avx512vnni",
            ":ukernels_avx512vnnigfni",
            # ":ukernels_avxvnni" Excluding avxvnni microkernels because they fail on older compilers
        ],
    )

    fb_xplat_cxx_library(
        name = "x86_and_x86_64_lib_ovr_win32",
        apple_sdks = (IOS, MACOSX),
        labels = labels,
        preferred_linkage = "static",
        visibility = ["PUBLIC"],
        deps = [
            ":ukernels_avx2_ovr_win32",
            ":ukernels_avx512_ovr_win32",
            ":ukernels_avx512skx_ovr_win32",
            ":ukernels_avx_ovr_win32",
            ":ukernels_f16c_ovr_win32",
            ":ukernels_fma3_ovr_win32",
            ":ukernels_sse2_ovr_win32",
            ":ukernels_sse41_ovr_win32",
            ":ukernels_sse_ovr_win32",
            ":ukernels_ssse3_ovr_win32",
            ":ukernels_avx512vbmi",
            # ":ukernels_avx512vnni_ovr_win32", # Build crashes on Windows Clang 17.0.3, re-enable when fixed (T199959765)
            # ":ukernels_avx512vnnigfni_ovr_win32",
            # ":ukernels_avxvnni_ovr_win32" Excluding avxvnni microkernels because they fail on older compilers
        ],
        exported_preprocessor_flags = [
            "-DXNN_ENABLE_AVX512VNNIGFNI=0"
        ]
    )

    fb_xplat_cxx_library(
        name = "arm_lib",
        apple_sdks = (IOS, MACOSX),
        labels = labels,
        preferred_linkage = "static",
        visibility = ["PUBLIC"],
        deps = [
            ":ukernels_armsimd32",
            ":ukernels_asm_aarch32",
            ":ukernels_asm_aarch64",
            ":ukernels_neon",
            ":ukernels_neon_aarch64",
            ":ukernels_neon_dot",
            ":ukernels_neon_dot_aarch64",
            ":ukernels_neon_dot_fp16arith",
            ":ukernels_neon_dot_fp16arith_aarch64",
            ":ukernels_neon_fma",
            ":ukernels_neon_fp16",
            ":ukernels_neon_fp16arith",
            ":ukernels_neon_fp16arith_aarch64",
            ":ukernels_neon_v8",
            ":ukernels_neonfma_aarch64",
            ":ukernels_neonfma_i8mm",
            ":ukernels_neoni8mm",
            ":ukernels_fp16arith",
        ],
    )

    fb_xplat_cxx_library(
        name = "armv7_lib",
        apple_sdks = (IOS, MACOSX),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        visibility = ["PUBLIC"],
        deps = [
            ":ukernels_asm_aarch32",
            ":ukernels_neon",
            ":ukernels_neon_dot",
            ":ukernels_neon_fma",
            ":ukernels_neon_v8",
        ],
    )

    fb_xplat_cxx_library(
        name = "prod_ukernels",
        apple_sdks = (IOS, MACOSX),
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        visibility = ["PUBLIC"],
        deps = [
            ":ukernels_scalar",
        ] + select({
            "DEFAULT": [
                ":arm_lib",
                ":x86_and_x86_64_lib",
            ],
            "ovr_config//os:windows": [":x86_and_x86_64_lib_ovr_win32"] if XNNPACK_WINDOWS_AVX512F_ENABLED else [
                ":arm_lib",
                ":x86_and_x86_64_lib",
            ],
            # doesn't cover iphonesimulator-x86_64
            "ovr_config//runtime:arm64-linux-ubuntu-neon": [":arm64_lib"],
            "ovr_config//runtime:fbcode-arm64": [":arm64_lib"],
            "ovr_config//runtime:platform010": [":x86_and_x86_64_lib"],
        }),
    )

    fb_xplat_cxx_library(
        name = "XNNPACK",
        apple_sdks = (IOS, MACOSX),
        labels = labels,
        deps = [
            ":tables",
            ":prod_ukernels",
            third_party("cpuinfo"),
            third_party("pthreadpool"),
        ],
        exported_headers = {
            "xnnpack.h": "XNNPACK/include/xnnpack.h",
        },
        header_namespace = "",
        headers = get_xnnpack_headers(),
        exported_deps = [
            ":subgraph",
        ],
        compiler_flags = select({
            "DEFAULT": [],
            "ovr_config//os:macos": ["-fvisibility=default"],
        }),
        platforms = (APPLE, ANDROID, CXX, WINDOWS),
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS + [
            "-DXNN_NO_Q8_OPERATORS",
            "-DXNN_NO_F16_OPERATORS",
            "-DXNN_NO_NCHW_OPERATORS",
            "-DXNN_NO_QU8_OPERATORS",
            "-DXNN_NO_U8_OPERATORS",
            "-DXNN_NO_X32_OPERATORS",
            "-DXNN_NO_X8_OPERATORS",
            "-DXNN_ENABLE_MEMOPT",
            "-DXNN_ENABLE_SPARSE=0",
            "-DXNN_ENABLE_ASSEMBLY",
            "-DXNN_ENABLE_GEMM_M_SPECIALIZATION",
            "-DXNN_ENABLE_ARM_DOTPROD",
            "-DXNN_ENABLE_CPUINFO",
            "-DXNN_ENABLE_ARM_I8MM=1",
            "-DXNN_ENABLE_ARM_FP16_VECTOR=1",
            "-DXNN_ENABLE_AVXVNNI=0",
            "-DXNN_ENABLE_ARM_SME=0",
            "-DXNN_ENABLE_ARM_SME2=0",
        ],
        srcs = XNNPACK_SRCS + LOGGING_SRCS + OPERATOR_SRCS + [
            "XNNPACK/src/init.c",
            "XNNPACK/src/configs/hardware-config.c",
            "XNNPACK/src/microkernel-utils.c",
            "XNNPACK/src/operator-run.c",
            "XNNPACK/src/reference/packing.cc",
            "XNNPACK/src/packw.c",
            "XNNPACK/src/cache.c",
            "XNNPACK/src/indirection.c",
            "XNNPACK/src/operator-utils.c",
            "XNNPACK/src/normalization.c",
            "XNNPACK/src/allocator.c",
            "XNNPACK/src/memory.c",
            "XNNPACK/src/mutex.c",
            "XNNPACK/src/microparams-init.c",
            "XNNPACK/src/params.c",
            "XNNPACK/src/reference/unary-elementwise.cc",
            "XNNPACK/src/reference/binary-elementwise.cc",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = (WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS) if XNNPACK_WINDOWS_AVX512F_ENABLED else WINDOWS_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS if XNNPACK_WINDOWS_AVX512F_ENABLED else [],
    )
