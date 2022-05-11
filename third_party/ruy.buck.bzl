RUY_SRCS = [
    'ruy/allocator.cc',
    'ruy/apply_multiplier.cc',
    'ruy/block_map.cc',
    'ruy/blocking_counter.cc',
    'ruy/context.cc',
    'ruy/context_get_ctx.cc',
    'ruy/cpuinfo.cc',
    'ruy/ctx.cc',
    'ruy/denormal.cc',
    'ruy/frontend.cc',
    'ruy/have_built_path_for_avx.cc',
    'ruy/have_built_path_for_avx2_fma.cc',
    'ruy/have_built_path_for_avx512.cc',
    'ruy/kernel_arm32.cc',
    'ruy/kernel_arm64.cc',
    'ruy/kernel_avx.cc',
    'ruy/kernel_avx2_fma.cc',
    'ruy/kernel_avx512.cc',
    'ruy/pack_arm.cc',
    'ruy/pack_avx.cc',
    'ruy/pack_avx2_fma.cc',
    'ruy/pack_avx512.cc',
    'ruy/pmu.cc',
    'ruy/prepacked_cache.cc',
    'ruy/prepare_packed_matrices.cc',
    'ruy/profiler/instrumentation.cc',
    'ruy/profiler/profiler.cc',
    'ruy/profiler/test_instrumented_library.cc',
    'ruy/profiler/treeview.cc',
    'ruy/system_aligned_alloc.cc',
    'ruy/thread_pool.cc',
    'ruy/trmul.cc',
    'ruy/tune.cc',
    'ruy/wait.cc'
  ]


def define_ruy():

    native.genrule(
        name = "ruy_srcs",
        outs = {src: [src] for src in RUY_SRCS},
        cmd = "rsync -a $(location :ruy_http_archive)/ $OUT",
        # default_outs = ["."],
    )

    native.genrule(
        name = "ruy_headers",
        out = "headers",
        cmd = "rsync -a $(location :ruy_http_archive)/ $OUT",
        # default_outs = ["."],
    )

    native.http_archive(
        name = "ruy_http_archive",
        strip_prefix = "ruy-a09683b8da7164b9c5704f88aef2dc65aa583e5d",
        sha256 = "1e5b0eceb645caf930ee5bdb5014f3d4d1e6f41860b7ef53612f2843d457ce0a",
        urls = [
            "https://github.com/google/ruy/archive/a09683b8da7164b9c5704f88aef2dc65aa583e5d.zip",
        ],
        out = "",
    )

    native.cxx_library(
        name = "ruy_lib",
        srcs = [":ruy_srcs[{}]".format(src) for src in RUY_SRCS],
        compiler_flags = ["-Os"],
        preferred_linkage = "static",
        exported_preprocessor_flags = [
            '-I$(location :ruy_headers)',
        ],
        visibility = [
            "PUBLIC",
        ],
        deps = [":ruy_http_archive"],
    )
