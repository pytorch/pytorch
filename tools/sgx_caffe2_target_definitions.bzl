load("@fbcode_macros//build_defs:cpp_library.bzl", "cpp_library")
load("//caffe2/caffe2:defs.bzl", "get_sgx_patterns")
load("//caffe2/tools:perf_kernel_defs.bzl", "define_perf_kernels")
load("//caffe2/tools:sgx_target_definitions.bzl", "is_sgx")

def add_sgx_caffe_libs():
    # we do not need to define these targets if we are in not SGX mode
    if not is_sgx:
        return

    core_file_patterns = [
        "core/allocator.cc",
        "core/logging.cc",
        "core/flags.cc",
        "core/common.cc",
        "core/context.cc",
        "core/event.cc",
        "core/context_base.cc",
        "core/numa.cc",
        "core/blob_serialization.cc",
        "core/tensor.cc",
        "core/types.cc",
        "core/blob_stats.cc",
        "opt/converter.cc",
        "opt/annotations.cc",
        "utils/cpuid.cc",
        "utils/threadpool/ThreadPool.cc",
        "utils/threadpool/pthreadpool-cpp.cc",
        "utils/threadpool/thread_pool_guard.cpp",
        "utils/proto_utils.cc",
    ]

    core_srcs = native.glob(
        core_file_patterns,
    )

    core_external_deps = [
        "protobuf",
        "glog",
        "sparsehash",
        "zstd",
    ]

    core_internal_deps = [
        "fbsource//third-party/fmt:fmt",
        "//caffe/proto:fb_protobuf",
        "//caffe2/caffe2/proto:fb_protobuf",
        "//caffe2/c10:c10",
        "//common/base:exception",
        "//common/logging:logging",
    ]

    internal_deps = core_internal_deps + [
        # "//libfb/py/mkl:mkl_dep_handle_lp64",
        "//onnx/onnx:onnx_lib",
        "//foxi:foxi_loader",
        "//caffe2/caffe2/fb/onnxifi:fbonnxifi_loader_stub",
        # "//rocksdb:rocksdb",
        "//caffe2:cpuinfo",
        "//xplat/QNNPACK:QNNPACK",
        "//folly/experimental/symbolizer:symbolizer",
        "//folly/hash:hash",
        "//folly/io:iobuf",
        "//folly:conv",
        "//folly:dynamic",
        "//folly:executor",
        "//folly:format",
        "//folly:json",
        "//folly:map_util",
        "//folly:memory",
        "//folly:mpmc_queue",
        "//folly:optional",
        "//folly:random",
        "//folly:range",
        "//folly/synchronization:rw_spin_lock",
        "//folly:singleton",
        "//folly:string",
        "//folly:synchronized",
        "//folly:thread_local",
        "//folly:traits",
        "//caffe2:ATen-core-headers",
        # important dependency to claim space for future refactorings
        "//caffe2:ATen-cpu",
        "//caffe2/caffe2/perfkernels:perfkernels",
        "//xplat/third-party/FP16:FP16",
        "fbsource//third-party/neon2sse:neon2sse",
    ]

    exclude = [
        # hip files are obtained from defs_hip.bzl
        # do not include in the cpu/cuda build
        "**/hip/**/*",
        "test/caffe2_gtest_main.cc",
        "quantization/server/**/*",
        "fb/async/comm/**/*",
        "fb/monitoring/**/*",
        "fb/session/**/*",
        # utils/knobs.cc and utils/knob_patcher.cc are only used in the open-source build
        # The internal build uses versions from fb/utils/ instead.
        "utils/knobs.cc",
        "utils/knob_patcher.cc",
    ]

    core_file_patterns = [
        "core/allocator.cc",
        "core/logging.cc",
        "core/flags.cc",
        "core/common.cc",
        "core/context.cc",
        "core/event.cc",
        "core/context_base.cc",
        "core/numa.cc",
        "core/blob_serialization.cc",
        "core/tensor.cc",
        "core/types.cc",
        "core/blob_stats.cc",
        "opt/converter.cc",
        "opt/annotations.cc",
        "utils/cpuid.cc",
        "utils/threadpool/ThreadPool.cc",
        "utils/threadpool/pthreadpool-cpp.cc",
        "utils/threadpool/thread_pool_guard.cpp",
        "utils/proto_utils.cc",
    ]

    test_file_patterns = get_sgx_patterns([
        "_test.cc",
        "_test.cpp",
    ])

    gpu_file_patterns = get_sgx_patterns([
        "_gpu.cc",
        "_cudnn.cc",
    ])

    cpu_file_patterns = get_sgx_patterns([
        ".cc",
        ".cpp",
    ])

    cpp_srcs = native.glob(
        cpu_file_patterns,
        exclude = exclude + gpu_file_patterns + test_file_patterns + core_file_patterns,
    )

    pp_flags = [
        "-Icaffe2",
        "-Imodules",
        "-DEIGEN_NO_DEBUG",
        "-DCAFFE2_USE_GOOGLE_GLOG",
        "-DCAFFE2_NO_CROSS_ARCH_WARNING",
        "-DCAFFE2_USE_EXCEPTION_PTR",
        # Work-around for incompatible thread pools in Caffe2 and NNPACK
        "-DFBCODE_CAFFE2",
        "-DUSE_PTHREADPOOL",
        "-DC10_MOBILE",
    ]

    compiler_flags = [
        "-Wno-unknown-pragmas",
        "-Wno-narrowing",
        "-Wno-missing-braces",
        "-Wno-strict-overflow",
        "-mno-avx",
        "-Wno-error=unused-result",
    ]

    cpu_header_patterns = [
        "**/*.h",
    ]

    cpp_headers = native.glob(
        cpu_header_patterns,
        exclude = exclude,
    )

    cpp_library(
        name = "caffe2_sgx_headers",
        headers = cpp_headers,
        propagated_pp_flags = pp_flags,
        exported_deps = core_internal_deps + [
            "//folly/io/async:async_base",
            "//caffe2/aten:ATen-core-sgx-headers",
        ],
        exported_external_deps = core_external_deps,
    )

    cpp_library(
        name = "caffe2_sgx_core",
        srcs = core_srcs + [
            "serialize/inline_container.cc",
            "serialize/crc.cc",
            "serialize/file_adapter.cc",
            "serialize/istream_adapter.cc",
            "serialize/read_adapter_interface.cc",
        ],
        compiler_flags = compiler_flags,
        link_whole = True,
        propagated_pp_flags = pp_flags,
        exported_deps = core_internal_deps + [
            "//caffe2/aten:ATen-sgx-core",
            "//caffe2/caffe2/core/nomnigraph:nomnigraph",
            "//xplat/third-party/pthreadpool:pthreadpool",
            "//caffe2:miniz",
        ],
        exported_external_deps = core_external_deps,
    )

def add_sgx_perf_kernel_libs():
    # we do not need to define these targets if we are in not SGX mode
    if not is_sgx:
        return

    dependencies = [
        "//caffe2/caffe2:caffe2_sgx_headers",
        "//caffe2/aten:ATen-core-sgx-headers",
    ]

    compiler_common_flags = [
        "-DCAFFE2_PERF_WITH_AVX2",
        "-DCAFFE2_PERF_WITH_AVX",
    ]

    external_deps = []

    # these are esentially disabled for hte sgx build but we still need them
    # to avoid linking issues
    levels_and_flags = ([
        (
            "avx2",
            [
                "-mavx2",
                "-mfma",
                "-mavx",
                "-mf16c",
            ],
        ),
        (
            "avx",
            [
                "-mavx",
                "-mf16c",
            ],
        ),
    ])

    define_perf_kernels(
        prefix = "sgx_",
        levels_and_flags = levels_and_flags,
        compiler_common_flags = compiler_common_flags,
        dependencies = dependencies,
        external_deps = external_deps,
    )
