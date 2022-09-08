# useful command for debugging which files are included:
# buck targets caffe2/caffe2: --json | jq -r "map(select(.srcs)) | map({key: .name, value: .srcs | sort}) | from_entries"
load("@fbsource//tools/build_defs:type_defs.bzl", "is_list")
load("//tools/build/buck:flags.bzl", "get_flags")

flags = get_flags()

_BASE_PATHS = (
    "core/*",
    "core/boxing/*",
    "core/boxing/impl/*",
    "core/dispatch/*",
    "core/op_registration/*",
    "cuda_rtc/*",
    "db/*",
    "experiments/operators/*",
    "ideep/**/*",
    "observers/*",
    "onnx/**/*",
    "operators/**/*",
    "observers/*",
    "predictor/*",
    "queue/*",
    "sgd/*",
    "share/contrib/zstd/*",
    "transforms/*",
    "utils/**/*",
)

_BASE_SGX_PATHS = (
    "core/*",
    "core/boxing/*",
    "core/boxing/impl/*",
    "core/dispatch/*",
    "core/op_registration/*",
    "cuda_rtc/*",
    "db/*",
    "experiments/operators/*",
    "observers/*",
    "onnx/**/*",
    "operators/**/*",
    "observers/*",
    "predictor/*",
    "queue/*",
    "sgd/*",
    "serialize/*",
    "share/contrib/zstd/*",
    "transforms/*",
    "utils/**/*",
)

def get_sgx_patterns(ext):
    if not is_list(ext):
        ext = [ext]
    return [path + e for path in _BASE_SGX_PATHS for e in ext]

def get_patterns(ext):
    if not is_list(ext):
        ext = [ext]
    return [path + e for path in _BASE_PATHS for e in ext]

def get_simd_preprocessor_flags():
    return [
        "-DUSE_FBGEMM",
    ]

def get_simd_compiler_flags():
    if flags.USE_SSE_ONLY:
        return ["-mno-avx"]

    simd_compiler_flags = [
        "-mavx",
    ] + get_simd_preprocessor_flags()

    # Every uarch with AVX512 support has AVX2 support
    if (flags.USE_AVX2 or flags.USE_AVX512):
        simd_compiler_flags += [
            "-mavx2",
            "-mfma",
        ]

    if flags.USE_AVX512:
        simd_compiler_flags += [
            "-mavx512f",
            "-mavx512dq",
            "-mavx512vl",
        ]

    return simd_compiler_flags
