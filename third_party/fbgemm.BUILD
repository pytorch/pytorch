load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "fbgemm_src_headers",
    hdrs = [
        "src/RefImplementations.h",
    ],
    include_prefix = "fbgemm",
)

cc_library(
    name = "fbgemm_base",
    srcs = [
        "src/EmbeddingSpMDM.cc",
        "src/EmbeddingSpMDMNBit.cc",
        "src/ExecuteKernel.cc",
        "src/ExecuteKernelU8S8.cc",
        "src/Fbgemm.cc",
        "src/FbgemmBfloat16Convert.cc",
        "src/FbgemmConv.cc",
        "src/FbgemmFP16.cc",
        "src/FbgemmFloat16Convert.cc",
        "src/FbgemmI64.cc",
        "src/FbgemmI8Spmdm.cc",
        "src/GenerateI8Depthwise.cc",
        "src/GenerateKernelU8S8S32ACC16.cc",
        "src/GenerateKernelU8S8S32ACC16Avx512.cc",
        "src/GenerateKernelU8S8S32ACC16Avx512VNNI.cc",
        "src/GenerateKernelU8S8S32ACC32.cc",
        "src/GenerateKernelU8S8S32ACC32Avx512.cc",
        "src/GenerateKernelU8S8S32ACC32Avx512VNNI.cc",
        "src/GroupwiseConvAcc32Avx2.cc",
        "src/PackAMatrix.cc",
        "src/PackAWithIm2Col.cc",
        "src/PackBMatrix.cc",
        "src/PackMatrix.cc",
        "src/PackAWithQuantRowOffset.cc",
        "src/PackAWithRowOffset.cc",
        "src/PackWeightMatrixForGConv.cc",
        "src/PackWeightsForConv.cc",
        "src/QuantUtils.cc",
        "src/RefImplementations.cc",
        "src/RowWiseSparseAdagradFused.cc",
        "src/SparseAdagrad.cc",
        "src/Utils.cc",
        # Private headers
        "src/CodeCache.h",
        "src/CodeGenHelpers.h",
        "src/ExecuteKernel.h",
        "src/ExecuteKernelGeneric.h",
        "src/ExecuteKernelU8S8.h",
        "src/FbgemmFP16Common.h",
        "src/GenerateI8Depthwise.h",
        "src/GenerateKernel.h",
        "src/GroupwiseConv.h",
        "src/RefImplementations.h",
        "src/TransposeUtils.h",
    ],
    hdrs = [
        "include/fbgemm/FbgemmConvert.h",
        "include/fbgemm/FbgemmI64.h",
    ],
    includes = [
        ".",
        "src",
    ],
    deps = [
        ":fbgemm_avx2",
        ":fbgemm_avx512",
        ":fbgemm_headers",
        ":fbgemm_src_headers",
        "@asmjit",
        "@cpuinfo",
    ],
    linkstatic = 1,
)

cc_library(
    name = "fbgemm_avx2_circular",
    srcs = [
        "src/FbgemmFloat16ConvertAvx2.cc",
    ],
    copts = [
        "-mavx2",
        "-mf16c",
    ],
    deps = [
        ":fbgemm_base",
    ],
    linkstatic = 1,
)

cc_library(
    name = "fbgemm",
    visibility = ["//visibility:public"],
    deps = [
        ":fbgemm_base",
        ":fbgemm_avx2_circular",
    ],
    linkstatic = 1,
)

cc_library(
    name = "fbgemm_avx2",
    srcs = [
        "src/EmbeddingSpMDMAvx2.cc",
        "src/FbgemmBfloat16ConvertAvx2.cc",
        # "src/FbgemmFloat16ConvertAvx2.cc",
        "src/FbgemmI8Depthwise3DAvx2.cc",
        "src/FbgemmI8Depthwise3x3Avx2.cc",
        "src/FbgemmI8DepthwiseAvx2.cc",
        "src/FbgemmI8DepthwisePerChannelQuantAvx2.cc",
        "src/OptimizedKernelsAvx2.cc",
        "src/PackDepthwiseConvMatrixAvx2.cc",
        "src/QuantUtilsAvx2.cc",
        "src/UtilsAvx2.cc",
        # Inline Assembly sources
        "src/FbgemmFP16UKernelsAvx2.cc",
        # Private headers
        "src/FbgemmFP16Common.h",
        "src/FbgemmFP16UKernelsAvx2.h",
        "src/FbgemmI8Depthwise2DAvx2-inl.h",
        "src/FbgemmI8DepthwiseAvx2-inl.h",
        "src/GenerateI8Depthwise.h",
        "src/MaskAvx2.h",
        "src/OptimizedKernelsAvx2.h",
        "src/TransposeUtils.h",
        "src/TransposeUtilsAvx2.h",
    ],
    copts = [
        "-m64",
        "-mavx2",
        "-mfma",
        "-mf16c",
        "-masm=intel",
    ],
    deps = [
        ":fbgemm_headers",
    ],
    linkstatic = 1,
)

cc_library(
    name = "fbgemm_avx2_headers",
    includes = [
        "src",
    ],
    hdrs = [
        "src/FbgemmFP16UKernelsAvx2.h",
        "src/MaskAvx2.h",
        "src/OptimizedKernelsAvx2.h",
    ],
)

cc_library(
    name = "fbgemm_avx512",
    srcs = [
        "src/FbgemmBfloat16ConvertAvx512.cc",
        "src/FbgemmFloat16ConvertAvx512.cc",
        "src/UtilsAvx512.cc",
        # Inline Assembly sources
        "src/FbgemmFP16UKernelsAvx512.cc",
        "src/FbgemmFP16UKernelsAvx512_256.cc",
        # Private headers
        "src/FbgemmFP16UKernelsAvx512.h",
        "src/FbgemmFP16Common.h",
        "src/MaskAvx2.h",
        "src/TransposeUtils.h",
        "src/TransposeUtilsAvx2.h",
    ],
    hdrs = [
        "src/FbgemmFP16UKernelsAvx512_256.h",
    ],
    copts = [
        "-m64",
        "-mfma",
        "-mavx512f",
        "-mavx512bw",
        "-mavx512dq",
        "-mavx512vl",
        "-masm=intel",
    ],
    deps = [
        ":fbgemm_headers",
    ],
    linkstatic = 1,
)

cc_library(
    name = "fbgemm_avx512_headers",
    includes = [
        "src",
    ],
    hdrs = [
        "src/FbgemmFP16UKernelsAvx512.h",
        "src/FbgemmFP16UKernelsAvx512_256.h",
    ],
)

cc_library(
    name = "fbgemm_headers",
    hdrs = [
        "include/fbgemm/ConvUtils.h",
        "include/fbgemm/Fbgemm.h",
        "include/fbgemm/FbgemmBuild.h",
        "include/fbgemm/FbgemmConvert.h",
        "include/fbgemm/FbgemmEmbedding.h",
        "include/fbgemm/FbgemmFP16.h",
        "include/fbgemm/FbgemmI64.h",
        "include/fbgemm/FbgemmI8DepthwiseAvx2.h",
        "include/fbgemm/FbgemmI8Spmdm.h",
        "include/fbgemm/OutputProcessing-inl.h",
        "include/fbgemm/PackingTraits-inl.h",
        "include/fbgemm/QuantUtils.h",
        "include/fbgemm/QuantUtilsAvx2.h",
        "include/fbgemm/Types.h",
        "include/fbgemm/Utils.h",
        "include/fbgemm/UtilsAvx2.h",
    ],
    includes = [
        "include",
    ],
    visibility = ["//visibility:public"],
)
