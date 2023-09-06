load("//tools/build_defs:fb_xplat_cxx_library.bzl", "fb_xplat_cxx_library")
load("//tools/build_defs:fbsource_utils.bzl", "is_arvr_mode")
load("//tools/build_defs:glob_defs.bzl", "subdir_glob")
load("//tools/build_defs:platform_defs.bzl", "ANDROID", "APPLE", "APPLETVOS", "CXX", "IOS", "MACOSX", "WINDOWS")
load(
    ":xnnpack_src_defs.bzl",
    "HOT_SRCS",
    "JIT_SRCS",
    "LOGGING_SRCS",
    "OPERATOR_SRCS",
    "SUBGRAPH_SRCS",
    "TABLE_SRCS",
)
load(
    ":xnnpack_wrapper_defs.bzl",
    "AARCH32_ASM_MICROKERNEL_SRCS",
    "AARCH64_ASM_MICROKERNEL_SRCS",
    "PROD_AARCH64_NEONFP16ARITH_MICROKERNEL_SRCS",
    "PROD_AARCH64_NEON_MICROKERNEL_SRCS",
    "PROD_AVX2_MICROKERNEL_SRCS",
    "PROD_AVX512F_MICROKERNEL_SRCS",
    "PROD_AVX512SKX_MICROKERNEL_SRCS",
    "PROD_AVX_MICROKERNEL_SRCS",
    "PROD_F16C_MICROKERNEL_SRCS",
    "PROD_FMA3_MICROKERNEL_SRCS",
    "PROD_NEONDOT_MICROKERNEL_SRCS",
    "PROD_NEONFMA_MICROKERNEL_SRCS",
    "PROD_NEONFP16_MICROKERNEL_SRCS",
    "PROD_NEONV8_MICROKERNEL_SRCS",
    "PROD_NEON_MICROKERNEL_SRCS",
    "PROD_SCALAR_AARCH32_MICROKERNEL_SRCS",
    "PROD_SCALAR_PORTABLE_MICROKERNEL_SRCS",
    "PROD_SSE2_MICROKERNEL_SRCS",
    "PROD_SSE41_MICROKERNEL_SRCS",
    "PROD_SSE_MICROKERNEL_SRCS",
    "PROD_SSSE3_MICROKERNEL_SRCS",
    "PROD_XOP_MICROKERNEL_SRCS",
    "ALL_NEONFMA_AARCH64_MICROKERNEL_SRCS",
    "ALL_NEON_AARCH64_MICROKERNEL_SRCS",
    "PROD_AVX512VBMI_MICROKERNEL_SRCS",
    "ALL_AVX512VBMI_MICROKERNEL_SRCS",
)

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

    fb_xplat_cxx_library(
        name = "interface",
        header_namespace = "",
        exported_headers = {
            "xnnpack.h": "XNNPACK/include/xnnpack.h",
        },
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        labels = labels,
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        exported_deps = [
            # Dependency only on pthreadpool interface
            third_party("pthreadpool_header"),
        ],
    )

    fb_xplat_cxx_library(
        name = "operators",
        # srcs have to include HOT_SRCS to be able to build on ARVR
        srcs = OPERATOR_SRCS + [
            "XNNPACK/src/binary-elementwise-config.c",
            "XNNPACK/src/packing.c",
            "XNNPACK/src/cache.c",
            "XNNPACK/src/indirection.c",
            "XNNPACK/src/operator-utils.c",
            "XNNPACK/src/normalization.c",
            "XNNPACK/src/allocator.c",
            "XNNPACK/src/memory.c",
            "XNNPACK/src/mutex.c",
        ],
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-Oz",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
            "-DXNN_ENABLE_GEMM_M_SPECIALIZATION=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = [
            ":interface",
            ":ukernels_f16c",
            third_party("cpuinfo"),
            third_party("FP16"),
            third_party("FXdiv"),
            third_party("clog"),
        ],
    )

    fb_xplat_cxx_library(
        name = "subgraph",
        srcs = SUBGRAPH_SRCS,
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
            "-DXNN_ENABLE_JIT=0",
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
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
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
        name = "jit_memory",
        # srcs have to include HOT_SRCS to be able to build on ARVR
        srcs = JIT_SRCS,
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-Oz",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platforms = (APPLE, ANDROID, CXX, WINDOWS),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = [
            ":interface",
            third_party("clog"),
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_scalar",
        srcs = PROD_SCALAR_PORTABLE_MICROKERNEL_SRCS,
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.c"),
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = [
            ":interface",
            third_party("FP16"),
            third_party("FXdiv"),
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_sse",
        srcs = PROD_SSE_MICROKERNEL_SRCS if is_arvr_mode() else [],
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.c"),
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-msse",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                PROD_SSE_MICROKERNEL_SRCS,
            ),
        ] if not is_arvr_mode() else []),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-msse"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-msse"],
        deps = [
            ":interface",
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_sse_ovr_win32",
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.c"),
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-msse",
                ],
            ),
        ],
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-msse"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-msse"],
        windows_srcs = PROD_SSE_MICROKERNEL_SRCS,
        deps = [
            ":interface",
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_sse2",
        srcs = PROD_SSE2_MICROKERNEL_SRCS if is_arvr_mode() else [],
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.c"),
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-msse2",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                PROD_SSE2_MICROKERNEL_SRCS,
            ),
        ] if not is_arvr_mode() else []),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-msse2"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-msse2"],
        deps = [
            ":interface",
            third_party("FP16"),
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_sse2_ovr_win32",
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.c"),
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-msse2",
                ],
            ),
        ],
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-msse2"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-msse2"],
        windows_srcs = PROD_SSE2_MICROKERNEL_SRCS,
        deps = [
            ":interface",
            third_party("FP16"),
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_ssse3",
        srcs = PROD_SSSE3_MICROKERNEL_SRCS if is_arvr_mode() else [],
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.c"),
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-mssse3",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                PROD_SSSE3_MICROKERNEL_SRCS,
            ),
        ] if not is_arvr_mode() else []),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mssse3"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mssse3"],
        deps = [
            ":interface",
            third_party("FP16"),
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_ssse3_ovr_win32",
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.c"),
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-mssse3",
                ],
            ),
        ],
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mssse3"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mssse3"],
        windows_srcs = PROD_SSSE3_MICROKERNEL_SRCS,
        deps = [
            ":interface",
            third_party("FP16"),
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_sse41",
        srcs = PROD_SSE41_MICROKERNEL_SRCS if is_arvr_mode() else [],
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.c"),
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-msse4.1",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                PROD_SSE41_MICROKERNEL_SRCS,
            ),
        ] if not is_arvr_mode() else []),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-msse4.1"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-msse4.1"],
        deps = [
            ":interface",
            third_party("FP16"),
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_sse41_ovr_win32",
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.c"),
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-msse4.1",
                ],
            ),
        ],
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-msse4.1"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-msse4.1"],
        windows_srcs = PROD_SSE41_MICROKERNEL_SRCS,
        deps = [
            ":interface",
            third_party("FP16"),
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx",
        srcs = PROD_AVX_MICROKERNEL_SRCS if is_arvr_mode() else [],
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.h"),
            ("XNNPACK/src", "**/*.c"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
            "-mavx",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-mavx",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                PROD_AVX_MICROKERNEL_SRCS,
            ),
        ] if not is_arvr_mode() else []),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mavx"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mavx"],
        deps = [
            ":interface",
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx_ovr_win32",
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.h"),
            ("XNNPACK/src", "**/*.c"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
            "-mavx",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-mavx",
                ],
            ),
        ],
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mavx"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mavx"],
        windows_srcs = PROD_AVX_MICROKERNEL_SRCS,
        deps = [
            ":interface",
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_f16c",
        srcs = PROD_F16C_MICROKERNEL_SRCS if is_arvr_mode() else [],
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.h"),
            ("XNNPACK/src", "**/*.c"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
            "-mf16c",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-mf16c",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                PROD_F16C_MICROKERNEL_SRCS,
            ),
        ] if not is_arvr_mode() else []),
        platforms = (APPLE, ANDROID, CXX, WINDOWS),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mf16c"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mf16c"],
        deps = [
            ":interface",
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_f16c_ovr_win32",
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.h"),
            ("XNNPACK/src", "**/*.c"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
            "-mf16c",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-mf16c",
                ],
            ),
        ],
        platforms = (APPLE, ANDROID, CXX, WINDOWS),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mf16c"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mf16c"],
        windows_srcs = PROD_F16C_MICROKERNEL_SRCS,
        deps = [
            ":interface",
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_xop",
        srcs = PROD_XOP_MICROKERNEL_SRCS if is_arvr_mode() else [],
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.h"),
            ("XNNPACK/src", "**/*.c"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
            "-mxop",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_preprocessor_flags = [
            (
                "windows-x86_64",
                [
                    "-Drestrict=",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                PROD_XOP_MICROKERNEL_SRCS,
            ),
        ] if not is_arvr_mode() else []),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mxop"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mxop"],
        deps = [
            ":interface",
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_xop_ovr_win32",
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.h"),
            ("XNNPACK/src", "**/*.c"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
            "-mxop",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_preprocessor_flags = [
            (
                "windows-x86_64",
                [
                    "-Drestrict=",
                ],
            ),
        ],
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mxop"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mxop"],
        windows_srcs = PROD_XOP_MICROKERNEL_SRCS,
        deps = [
            ":interface",
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_fma3",
        srcs = PROD_FMA3_MICROKERNEL_SRCS if is_arvr_mode() else [],
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.h"),
            ("XNNPACK/src", "**/*.c"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
            "-mfma",
            "-mf16c",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "^(i[3-6]86|x86|x86_64|AMD64)$",
                [
                    "-mfma",
                    "-mf16c",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                PROD_FMA3_MICROKERNEL_SRCS,
            ),
        ] if not is_arvr_mode() else []),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + [
            "-mfma",
            "-mf16c",
        ],
        windows_compiler_flags_override = WINDOWS_FLAGS + [
            "-mfma",
            "-mf16c",
        ],
        deps = [
            ":interface",
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_fma3_ovr_win32",
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.h"),
            ("XNNPACK/src", "**/*.c"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
            "-mfma",
            "-mf16c",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "^(i[3-6]86|x86|x86_64|AMD64)$",
                [
                    "-mfma",
                    "-mf16c",
                ],
            ),
        ],
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + [
            "-mfma",
            "-mf16c",
        ],
        windows_compiler_flags_override = WINDOWS_FLAGS + [
            "-mfma",
            "-mf16c",
        ],
        windows_srcs = PROD_FMA3_MICROKERNEL_SRCS,
        deps = [
            ":interface",
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx2",
        srcs = PROD_AVX2_MICROKERNEL_SRCS if is_arvr_mode() else [],
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.c"),
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
            "-mavx2",
            "-mfma",
            "-mf16c",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-mavx2",
                    "-mfma",
                    "-mf16c",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                PROD_AVX2_MICROKERNEL_SRCS,
            ),
        ] if not is_arvr_mode() else []),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
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
        deps = [
            ":interface",
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx2_ovr_win32",
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.c"),
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
            "-mavx2",
            "-mfma",
            "-mf16c",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-mavx2",
                    "-mfma",
                    "-mf16c",
                ],
            ),
        ],
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
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
        windows_srcs = PROD_AVX2_MICROKERNEL_SRCS,
        deps = [
            ":interface",
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx512",
        srcs = PROD_AVX512F_MICROKERNEL_SRCS if is_arvr_mode() else [],
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.c"),
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
            "-mavx512f",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-mavx512f",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                PROD_AVX512F_MICROKERNEL_SRCS,
            ),
        ] if not is_arvr_mode() else []),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mavx512f"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mavx512f"],
        deps = [
            ":interface",
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx512vbmi",
        srcs = PROD_AVX512VBMI_MICROKERNEL_SRCS if is_arvr_mode() else [],
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.c"),
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
            "-mavx512f",
            "-mavx512cd",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",
            "-mavx512vbmi",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "^(i[3-6]86|x86|x86_64|AMD64)$",
                [
                    "-mavx512f",
                    "-mavx512cd",
                    "-mavx512bw",
                    "-mavx512dq",
                    "-mavx512vl",
                    "-mavx512vbmi",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                PROD_AVX512VBMI_MICROKERNEL_SRCS,
            ),
        ] if not is_arvr_mode() else []),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
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
        deps = [
            ":interface",
        ],
    )


    fb_xplat_cxx_library(
        name = "ukernels_avx512_ovr_win32",
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.c"),
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
            "-mavx512f",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-mavx512f",
                ],
            ),
        ],
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mavx512f"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mavx512f"],
        windows_srcs = PROD_AVX512F_MICROKERNEL_SRCS,
        deps = [
            ":interface",
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx512skx",
        srcs = PROD_AVX512SKX_MICROKERNEL_SRCS if is_arvr_mode() else [],
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.c"),
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
            "-mavx512f",
            "-mavx512cd",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "^(i[3-6]86|x86|x86_64|AMD64)$",
                [
                    "-mavx512f",
                    "-mavx512cd",
                    "-mavx512bw",
                    "-mavx512dq",
                    "-mavx512vl",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                PROD_AVX512SKX_MICROKERNEL_SRCS,
            ),
        ] if not is_arvr_mode() else []),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
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
        deps = [
            ":interface",
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx512skx_ovr_win32",
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.c"),
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
            "-mavx512f",
            "-mavx512cd",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "^(i[3-6]86|x86|x86_64|AMD64)$",
                [
                    "-mavx512f",
                    "-mavx512cd",
                    "-mavx512bw",
                    "-mavx512dq",
                    "-mavx512vl",
                ],
            ),
        ],
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
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
        windows_srcs = PROD_AVX512SKX_MICROKERNEL_SRCS,
        deps = [
            ":interface",
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_neon",
        srcs = PROD_NEON_MICROKERNEL_SRCS,
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.c"),
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "^(android-armv7|iphoneos-armv7)$",
                [
                    "-march=armv7-a",
                    "-mfpu=neon",
                    "-mfloat-abi=softfp",
                ],
            ),
        ],
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = [
            ":interface",
            third_party("FP16"),
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_neon_fma",
        srcs = PROD_NEONFMA_MICROKERNEL_SRCS,
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.c"),
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "armv7",
                [
                    "-mfpu=neon-vfpv4",
                ],
            ),
        ],
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = [
            ":interface",
            third_party("FP16"),
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_neon_fp16",
        srcs = PROD_NEONFP16_MICROKERNEL_SRCS,
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.c"),
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "armv7",
                [
                    "-mfpu=neon-fp16",
                ],
            ),
        ],
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = [
            ":interface",
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_neon_v8",
        srcs = PROD_NEONV8_MICROKERNEL_SRCS,
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.c"),
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "(aarch64|arm64)",
                [
                    "-march=armv8-a",
                ],
            ),
            (
                "^android-armv7$",
                [
                    "-march=armv8-a",
                    "-mfpu=neon-fp-armv8",
                    "-mfloat-abi=softfp",
                ],
            ),
            (
                "^iphoneos-armv7$",
                [
                    "-mcpu=cyclone",
                    "-mtune=generic",
                ],
            ),
        ],
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = [
            ":interface",
            third_party("FP16"),
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_neon_dot",
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.c"),
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "(aarch64|arm64)",
                [
                    "-march=armv8.2-a+dotprod",
                ],
            ),
            (
                "^android-armv7$",
                [
                    "-march=armv8.2-a+dotprod",
                    "-mfpu=neon-fp-armv8",
                    "-mfloat-abi=softfp",
                ],
            ),
        ],
        platform_srcs = [
            # excluding iphoneos-armv7, matching everything else
            (
                "^((?!iphoneos-armv7).)*$",
                PROD_NEONDOT_MICROKERNEL_SRCS,
            ),
        ],
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = [
            ":interface",
            third_party("FP16"),
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_neon_aarch64",
        srcs = ALL_NEON_AARCH64_MICROKERNEL_SRCS,
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.c"),
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = [
            ":interface",
            third_party("FP16"),
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_neon_fp16arith_aarch64",
        srcs = PROD_AARCH64_NEONFP16ARITH_MICROKERNEL_SRCS,
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.c"),
            ("XNNPACK/src", "**/*.h"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "(aarch64|arm64)",
                [
                    "-march=armv8.2-a+fp16",
                ],
            ),
        ],
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = [
            ":interface",
            third_party("FP16"),
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_scalar_aarch32",
        srcs = PROD_SCALAR_AARCH32_MICROKERNEL_SRCS,
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.h"),
            ("XNNPACK/src", "**/*.c"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "^(android-armv7|iphoneos-armv7)$",
                [
                    "-march=armv7-a",
                    "-mfpu=neon",
                    "-mfloat-abi=softfp",
                ],
            ),
        ],
        platforms = (APPLE, ANDROID, CXX, WINDOWS),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = [
            ":interface",
            third_party("FP16"),
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_neonfma_aarch64",
        srcs = ALL_NEONFMA_AARCH64_MICROKERNEL_SRCS,
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.h"),
            ("XNNPACK/src", "**/*.c"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "^(android-armv8|iphoneos-armv8)$",
                [
                    "-march=armv8-a",
                ],
            ),
        ],
        platforms = (APPLE, ANDROID, CXX, WINDOWS),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = [
            ":interface",
            third_party("FP16"),
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_asm_aarch32",
        srcs = AARCH32_ASM_MICROKERNEL_SRCS,
        headers = subdir_glob([
            ("XNNPACK/src", "xnnpack/assembly.h"),
            ("XNNPACK/src", "**/*.S"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "^android-armv7$",
                [
                    "-march=armv8.2-a+dotprod",
                    "-mfpu=neon-fp-armv8",
                    "-mfloat-abi=softfp",
                ],
            ),
        ],
        platforms = (APPLE, ANDROID, CXX, WINDOWS),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = [
            ":interface",
            ":jit_memory",
            third_party("FP16"),
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_asm_aarch64",
        srcs = AARCH64_ASM_MICROKERNEL_SRCS,
        headers = subdir_glob([
            ("XNNPACK/src", "xnnpack/assembly.h"),
            ("XNNPACK/src", "**/*.S"),
        ]),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        compiler_flags = [
            "-O2",
        ],
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "(aarch64|arm64)",
                [
                    "-march=armv8.2-a+fp16+dotprod",
                ],
            ),
        ],
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = [
            ":interface",
            ":jit_memory",
            third_party("FP16"),
        ],
    )

    fb_xplat_cxx_library(
        name = "arm64_lib",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        labels = labels,
        preferred_linkage = "static",
        visibility = ["PUBLIC"],
        deps = [
            ":jit_memory",
            ":ukernels_asm_aarch64",
            ":ukernels_neon",
            ":ukernels_neon_aarch64",
            ":ukernels_neon_dot",
            ":ukernels_neon_fma",
            ":ukernels_neon_fp16",
            ":ukernels_neon_fp16arith_aarch64",
            ":ukernels_neon_v8",
            ":ukernels_neonfma_aarch64",
        ],
    )

    fb_xplat_cxx_library(
        name = "x86_and_x86_64_lib",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
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
            ":ukernels_xop",
            ":ukernels_avx512vbmi",
        ],
    )

    fb_xplat_cxx_library(
        name = "x86_and_x86_64_lib_ovr_win32",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
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
            ":ukernels_xop_ovr_win32",
            ":ukernels_avx512vbmi",
        ],
    )

    fb_xplat_cxx_library(
        name = "arm_lib",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        labels = labels,
        preferred_linkage = "static",
        visibility = ["PUBLIC"],
        deps = [
            ":jit_memory",
            ":ukernels_asm_aarch32",
            ":ukernels_asm_aarch64",
            ":ukernels_neon",
            ":ukernels_neon_aarch64",
            ":ukernels_neon_dot",
            ":ukernels_neon_fma",
            ":ukernels_neon_fp16",
            ":ukernels_neon_fp16arith_aarch64",
            ":ukernels_neon_v8",
            ":ukernels_scalar_aarch32",
            ":ukernels_neonfma_aarch64",
        ],
    )

    fb_xplat_cxx_library(
        name = "armv7_lib",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        labels = labels,
        preferred_linkage = "static",
        visibility = ["PUBLIC"],
        deps = [
            ":jit_memory",
            ":ukernels_asm_aarch32",
            ":ukernels_neon",
            ":ukernels_neon_dot",
            ":ukernels_neon_fma",
            ":ukernels_neon_v8",
            ":ukernels_scalar_aarch32",
        ],
    )

    fb_xplat_cxx_library(
        name = "XNNPACK",
        apple_sdks = (IOS, MACOSX, APPLETVOS),
        labels = labels,
        deps = [
            ":operators",
            ":subgraph",
            ":tables",
            ":ukernels_scalar",
            third_party("cpuinfo"),
            third_party("pthreadpool"),
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
            "ovr_config//runtime:platform010": [":x86_and_x86_64_lib"],
        }),
        exported_headers = {
            "xnnpack.h": "XNNPACK/include/xnnpack.h",
        },
        fbobjc_preprocessor_flags = [
            "-DXNN_PRIVATE=",
            "-DXNN_INTERNAL=",
        ],
        header_namespace = "",
        headers = subdir_glob([
            ("XNNPACK/src", "**/*.h"),
            ("XNNPACK/include", "**/*.h"),
        ]),
        platforms = (APPLE, ANDROID, CXX, WINDOWS),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
            "-DXNN_NO_Q8_OPERATORS",
            "-DXNN_NO_F16_OPERATORS",
            "-DXNN_NO_NCHW_OPERATORS",
            "-DXNN_NO_QU8_OPERATORS",
            "-DXNN_NO_U8_OPERATORS",
            "-DXNN_NO_X32_OPERATORS",
            "-DXNN_NO_X8_OPERATORS",
            "-DXNN_NO_XX_OPERATORS",
            "-DXNN_ENABLE_MEMOPT",
            "-DXNN_ENABLE_SPARSE=0",
            "-DXNN_ENABLE_JIT=0",
            "-DXNN_ENABLE_ASSEMBLY",
            "-DXNN_ENABLE_GEMM_M_SPECIALIZATION",
            "-DXNN_ENABLE_ARM_DOTPROD",
        ],
        srcs = [
            "XNNPACK/src/init.c",
            "XNNPACK/src/params.c",
            "XNNPACK/src/operator-run.c",
            "XNNPACK/src/microparams-init.c",
            "XNNPACK/src/x8-lut-config.c",
            "XNNPACK/src/hardware-config.c",
            "XNNPACK/src/transpose-config.c",
            "XNNPACK/src/amalgam/scalar.c",
            "XNNPACK/src/operators/post-operation.c",
        ] + LOGGING_SRCS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = (WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS) if XNNPACK_WINDOWS_AVX512F_ENABLED else WINDOWS_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS if XNNPACK_WINDOWS_AVX512F_ENABLED else [],
    )
