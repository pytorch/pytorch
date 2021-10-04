load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("@//third_party:sleef.bzl", "sleef_cc_library")

SLEEF_COPTS = [
    "-DHAVE_MALLOC_USABLE_SIZE=1",
    "-DHAVE_MMAP=1",
    "-DHAVE_SHM_OPEN=1",
    "-DHAVE_SHM_UNLINK=1",
    "-DIDEEP_USE_MKL",
    "-DDNNL_CPU_RUNTIME=TBB",
    "-DONNX_ML=1",
    "-DONNX_NAMESPACE=onnx",
    "-DTH_BLAS_MKL",
    "-D_FILE_OFFSET_BITS=64",
    "-ffp-contract=off",
    "-fno-math-errno",
    "-fno-trapping-math",
    "-DCAFFE2_USE_GLOO",
    "-DCUDA_HAS_FP16=1",
    "-DHAVE_GCC_GET_CPUID",
    "-DUSE_AVX",
    "-DUSE_AVX2",
    "-DTH_HAVE_THREAD",
    "-std=gnu99",
]

SLEEF_COMMON_TARGET_COPTS = [
    "-DSLEEF_STATIC_LIBS=1",
    "-DENABLE_ALIAS=1",
]

SLEEF_PRIVATE_HEADERS = glob([
    "build/include/*.h",
    "src/arch/*.h",
    "src/common/*.h",
    "src/libm/*.h",
    "src/libm/include/*.h",
])

SLEEF_PUBLIC_HEADERS = [
    ":sleef_h",
]

SLEEF_PRIVATE_INCLUDES = [
    "-Iexternal/sleef/src/arch",
    "-Iexternal/sleef/src/common",
]

SLEEF_PUBLIC_INCLUDES = [
    "build/include",
]

SLEEF_VISIBILITY = [
    "//visibility:public",
]

cc_binary(
    name = "mkalias",
    srcs = [
        "src/libm/funcproto.h",
        "src/libm/mkalias.c",
    ],
)

genrule(
    name = "alias_avx512f_h",
    outs = ["alias_avx512f.h"],
    cmd = "{ " + "; ".join([
        "$(location :mkalias) -16 __m512 __m512i e avx512f",
        "$(location :mkalias) 8 __m512d __m256i e avx512f",
    ]) + "; } > $@",
    tools = [":mkalias"],
)

cc_binary(
    name = "mkdisp",
    srcs = [
        "src/libm/funcproto.h",
        "src/libm/mkdisp.c",
    ],
    copts = SLEEF_COPTS,
)

genrule(
    name = "dispavx_c",
    srcs = ["src/libm/dispavx.c.org"],
    outs = ["dispavx.c"],
    cmd = "{ cat $(location src/libm/dispavx.c.org); $(location :mkdisp) 4 8 __m256d __m256 __m128i avx fma4 avx2; } > $@",
    tools = [":mkdisp"],
)

genrule(
    name = "dispsse_c",
    srcs = ["src/libm/dispsse.c.org"],
    outs = ["dispsse.c"],
    cmd = "{ cat $(location src/libm/dispsse.c.org); $(location :mkdisp) 2 4 __m128d __m128 __m128i sse2 sse4 avx2128; } > $@",
    tools = [":mkdisp"],
)

cc_binary(
    name = "mkrename",
    srcs = [
        "src/libm/funcproto.h",
        "src/libm/mkrename.c",
    ],
)

genrule(
    name = "renameavx_h",
    outs = ["renameavx.h"],
    cmd = "$(location :mkrename) cinz_ 4 8 avx > $@",
    tools = [":mkrename"],
)

genrule(
    name = "renameavx2_h",
    outs = ["renameavx2.h"],
    cmd = "$(location :mkrename) finz_ 4 8 avx2 > $@",
    tools = [":mkrename"],
)

genrule(
    name = "renameavx2128_h",
    outs = ["renameavx2128.h"],
    cmd = "$(location :mkrename) finz_ 2 4 avx2128 > $@",
    tools = [":mkrename"],
)

genrule(
    name = "renameavx512f_h",
    outs = ["renameavx512f.h"],
    cmd = "$(location :mkrename) finz_ 8 16 avx512f > $@",
    tools = [":mkrename"],
)

genrule(
    name = "renameavx512fnofma_h",
    outs = ["renameavx512fnofma.h"],
    cmd = "$(location :mkrename) cinz_ 8 16 avx512fnofma > $@",
    tools = [":mkrename"],
)

genrule(
    name = "renamefma4_h",
    outs = ["renamefma4.h"],
    cmd = "$(location :mkrename) finz_ 4 8 fma4 > $@",
    tools = [":mkrename"],
)

genrule(
    name = "renamepurec_scalar_h",
    outs = ["renamepurec_scalar.h"],
    cmd = "$(location :mkrename) cinz_ 1 1 purec > $@",
    tools = [":mkrename"],
)

genrule(
    name = "renamepurecfma_scalar_h",
    outs = ["renamepurecfma_scalar.h"],
    cmd = "$(location :mkrename) finz_ 1 1 purecfma > $@",
    tools = [":mkrename"],
)

genrule(
    name = "renamesse2_h",
    outs = ["renamesse2.h"],
    cmd = "$(location :mkrename) cinz_ 2 4 sse2 > $@",
    tools = [":mkrename"],
)

genrule(
    name = "renamesse4_h",
    outs = ["renamesse4.h"],
    cmd = "$(location :mkrename) cinz_ 2 4 sse4 > $@",
    tools = [":mkrename"],
)

genrule(
    name = "sleef_h",
    srcs = [
        "src/libm/sleeflibm_header.h.org.in",
        "src/libm/sleeflibm_footer.h.org",
    ],
    outs = ["build/include/sleef.h"],
    cmd = "{ " + "; ".join([
        "cat $(location src/libm/sleeflibm_header.h.org.in)",
        "$(location :mkrename) cinz_ 2 4 __m128d __m128 __m128i __m128i __SSE2__",
        "$(location :mkrename) cinz_ 2 4 __m128d __m128 __m128i __m128i __SSE2__ sse2",
        "$(location :mkrename) cinz_ 2 4 __m128d __m128 __m128i __m128i __SSE2__ sse4",
        "$(location :mkrename) cinz_ 4 8 __m256d __m256 __m128i \"struct { __m128i x, y; }\" __AVX__",
        "$(location :mkrename) cinz_ 4 8 __m256d __m256 __m128i \"struct { __m128i x, y; }\" __AVX__ avx",
        "$(location :mkrename) finz_ 4 8 __m256d __m256 __m128i \"struct { __m128i x, y; }\" __AVX__ fma4",
        "$(location :mkrename) finz_ 4 8 __m256d __m256 __m128i __m256i __AVX__ avx2",
        "$(location :mkrename) finz_ 2 4 __m128d __m128 __m128i __m128i __SSE2__ avx2128",
        "$(location :mkrename) finz_ 8 16 __m512d __m512 __m256i __m512i __AVX512F__",
        "$(location :mkrename) finz_ 8 16 __m512d __m512 __m256i __m512i __AVX512F__ avx512f",
        "$(location :mkrename) cinz_ 8 16 __m512d __m512 __m256i __m512i __AVX512F__ avx512fnofma",
        "$(location :mkrename) cinz_ 1 1 double float int32_t int32_t __STDC__ purec",
        "$(location :mkrename) finz_ 1 1 double float int32_t int32_t FP_FAST_FMA purecfma",
        "cat $(location src/libm/sleeflibm_footer.h.org)",
    ]) + "; } > $@",
    tools = [":mkrename"],
)

cc_library(
    name = "sleef",
    srcs = [
        "src/libm/rempitab.c",
        "src/libm/sleefdp.c",
        "src/libm/sleefld.c",
        "src/libm/sleefqp.c",
        "src/libm/sleefsp.c",
    ],
    hdrs = SLEEF_PUBLIC_HEADERS,
    copts = SLEEF_PRIVATE_INCLUDES + SLEEF_COPTS + SLEEF_COMMON_TARGET_COPTS + [
        "-DDORENAME=1",
        "-DENABLEFLOAT128=1",
        "-Wno-unused-result",
    ],
    includes = SLEEF_PUBLIC_INCLUDES,
    # -lgcc resolves
    # U __addtf3
    # U __eqtf2
    # U __fixtfdi
    # U __floatditf
    # U __gttf2
    # U __lttf2
    # U __multf3
    # U __subtf3
    # in bazel-bin/external/sleef/_objs/sleef/sleefqp.pic.o
    linkopts = [
        "-lgcc",
    ],
    linkstatic = True,
    visibility = SLEEF_VISIBILITY,
    # The purpose of the lists in deps is to keep related pairs of
    # libraries together. In particular, each pair that contains a *det*
    # library originates with a sleef_cc_library().
    deps = [
        ":common",
        ":dispavx",
        ":dispsse",
    ] + [
        ":sleefavx",
        ":sleefdetavx",
    ] + [
        ":sleefavx2",
        ":sleefdetavx2",
    ] + [
        ":sleefavx2128",
        ":sleefdetavx2128",
    ] + [
        ":sleefavx512f",
        ":sleefdetavx512f",
    ] + [
        ":sleefavx512fnofma",
        ":sleefdetavx512fnofma",
    ] + [
        ":sleeffma4",
        ":sleefdetfma4",
    ] + [
        ":sleefsse2",
        ":sleefdetsse2",
    ] + [
        ":sleefsse4",
        ":sleefdetsse4",
    ] + [
        ":sleefpurec_scalar",
        ":sleefdetpurec_scalar",
    ] + [
        ":sleefpurecfma_scalar",
        ":sleefdetpurecfma_scalar",
    ],
    alwayslink = True,
)

cc_library(
    name = "common",
    srcs = SLEEF_PRIVATE_HEADERS + [
        "src/common/common.c",
    ],
    copts = SLEEF_PRIVATE_INCLUDES + SLEEF_COPTS + [
        "-Wno-unused-result",
    ],
    linkstatic = True,
    visibility = SLEEF_VISIBILITY,
    alwayslink = True,
)

cc_library(
    name = "dispavx",
    srcs = SLEEF_PRIVATE_HEADERS + SLEEF_PUBLIC_HEADERS + [
        ":dispavx_c",
    ],
    copts = SLEEF_PRIVATE_INCLUDES + SLEEF_COPTS + SLEEF_COMMON_TARGET_COPTS + [
        "-DENABLE_AVX2=1",
        "-DENABLE_FMA4=1",
        "-mavx",
    ],
    includes = SLEEF_PUBLIC_INCLUDES,
    linkstatic = True,
    visibility = SLEEF_VISIBILITY,
    alwayslink = True,
)

cc_library(
    name = "dispsse",
    srcs = SLEEF_PRIVATE_HEADERS + SLEEF_PUBLIC_HEADERS + [
        ":dispsse_c",
    ],
    copts = SLEEF_PRIVATE_INCLUDES + SLEEF_COPTS + SLEEF_COMMON_TARGET_COPTS + [
        "-DENABLE_AVX2=1",
        "-DENABLE_FMA4=1",
        "-msse2",
    ],
    includes = SLEEF_PUBLIC_INCLUDES,
    linkstatic = True,
    visibility = SLEEF_VISIBILITY,
    alwayslink = True,
)

sleef_cc_library(
    name = "sleefavx512f",
    srcs = SLEEF_PRIVATE_HEADERS + [
        "src/libm/sleefsimddp.c",
        "src/libm/sleefsimdsp.c",
        ":alias_avx512f_h",
        ":renameavx512f_h",
    ],
    copts = SLEEF_PRIVATE_INCLUDES + SLEEF_COPTS + SLEEF_COMMON_TARGET_COPTS + [
        "-DDORENAME=1",
        "-DALIAS_NO_EXT_SUFFIX=\\\"alias_avx512f.h\\\"",
        "-DENABLE_AVX512F=1",
        "-mavx512f",
    ],
    linkstatic = True,
    visibility = SLEEF_VISIBILITY,
    alwayslink = True,
)

sleef_cc_library(
    name = "sleefavx512fnofma",
    srcs = SLEEF_PRIVATE_HEADERS + [
        "src/libm/sleefsimddp.c",
        "src/libm/sleefsimdsp.c",
        ":renameavx512fnofma_h",
    ],
    copts = SLEEF_PRIVATE_INCLUDES + SLEEF_COPTS + SLEEF_COMMON_TARGET_COPTS + [
        "-DDORENAME=1",
        "-DENABLE_AVX512FNOFMA=1",
        "-mavx512f",
    ],
    linkstatic = True,
    visibility = SLEEF_VISIBILITY,
    alwayslink = True,
)

sleef_cc_library(
    name = "sleefavx",
    srcs = SLEEF_PRIVATE_HEADERS + [
        "src/libm/sleefsimddp.c",
        "src/libm/sleefsimdsp.c",
        ":renameavx_h",
    ],
    copts = SLEEF_PRIVATE_INCLUDES + SLEEF_COPTS + SLEEF_COMMON_TARGET_COPTS + [
        "-DDORENAME=1",
        "-DENABLE_AVX=1",
        "-mavx",
    ],
    linkstatic = True,
    visibility = SLEEF_VISIBILITY,
    alwayslink = True,
)

sleef_cc_library(
    name = "sleefavx2",
    srcs = SLEEF_PRIVATE_HEADERS + [
        "src/libm/sleefsimddp.c",
        "src/libm/sleefsimdsp.c",
        ":renameavx2_h",
    ],
    copts = SLEEF_PRIVATE_INCLUDES + SLEEF_COPTS + SLEEF_COMMON_TARGET_COPTS + [
        "-DDORENAME=1",
        "-DENABLE_AVX2=1",
        "-mavx2",
        "-mfma",
    ],
    linkstatic = True,
    visibility = SLEEF_VISIBILITY,
    alwayslink = True,
)

sleef_cc_library(
    name = "sleefavx2128",
    srcs = SLEEF_PRIVATE_HEADERS + [
        "src/libm/sleefsimddp.c",
        "src/libm/sleefsimdsp.c",
        ":renameavx2128_h",
    ],
    copts = SLEEF_PRIVATE_INCLUDES + SLEEF_COPTS + SLEEF_COMMON_TARGET_COPTS + [
        "-DDORENAME=1",
        "-DENABLE_AVX2128=1",
        "-mavx2",
        "-mfma",
    ],
    linkstatic = True,
    visibility = SLEEF_VISIBILITY,
    alwayslink = True,
)

sleef_cc_library(
    name = "sleeffma4",
    srcs = SLEEF_PRIVATE_HEADERS + [
        "src/libm/sleefsimddp.c",
        "src/libm/sleefsimdsp.c",
        ":renamefma4_h",
    ],
    copts = SLEEF_PRIVATE_INCLUDES + SLEEF_COPTS + SLEEF_COMMON_TARGET_COPTS + [
        "-DDORENAME=1",
        "-DENABLE_FMA4=1",
        "-mfma4",
    ],
    linkstatic = True,
    visibility = SLEEF_VISIBILITY,
    alwayslink = True,
)

sleef_cc_library(
    name = "sleefsse2",
    srcs = SLEEF_PRIVATE_HEADERS + [
        "src/libm/sleefsimddp.c",
        "src/libm/sleefsimdsp.c",
        ":renamesse2_h",
    ],
    copts = SLEEF_PRIVATE_INCLUDES + SLEEF_COPTS + SLEEF_COMMON_TARGET_COPTS + [
        "-DDORENAME=1",
        "-DENABLE_SSE2=1",
        "-msse2",
    ],
    linkstatic = True,
    visibility = SLEEF_VISIBILITY,
    alwayslink = True,
)

sleef_cc_library(
    name = "sleefsse4",
    srcs = SLEEF_PRIVATE_HEADERS + [
        "src/libm/sleefsimddp.c",
        "src/libm/sleefsimdsp.c",
        ":renamesse4_h",
    ],
    copts = SLEEF_PRIVATE_INCLUDES + SLEEF_COPTS + SLEEF_COMMON_TARGET_COPTS + [
        "-DDORENAME=1",
        "-DENABLE_SSE4=1",
        "-msse4.1",
    ],
    linkstatic = True,
    visibility = SLEEF_VISIBILITY,
    alwayslink = True,
)

sleef_cc_library(
    name = "sleefpurec_scalar",
    srcs = SLEEF_PRIVATE_HEADERS + [
        "src/libm/sleefsimddp.c",
        "src/libm/sleefsimdsp.c",
        ":renamepurec_scalar_h",
    ],
    copts = SLEEF_PRIVATE_INCLUDES + SLEEF_COPTS + SLEEF_COMMON_TARGET_COPTS + [
        "-DDORENAME=1",
        "-DENABLE_PUREC_SCALAR=1",
    ],
    linkstatic = True,
    visibility = SLEEF_VISIBILITY,
    alwayslink = True,
)

sleef_cc_library(
    name = "sleefpurecfma_scalar",
    srcs = SLEEF_PRIVATE_HEADERS + [
        "src/libm/sleefsimddp.c",
        "src/libm/sleefsimdsp.c",
        ":renamepurecfma_scalar_h",
    ],
    copts = SLEEF_PRIVATE_INCLUDES + SLEEF_COPTS + SLEEF_COMMON_TARGET_COPTS + [
        "-DDORENAME=1",
        "-DENABLE_PURECFMA_SCALAR=1",
        "-mavx2",
        "-mfma",
    ],
    linkstatic = True,
    visibility = SLEEF_VISIBILITY,
    alwayslink = True,
)
