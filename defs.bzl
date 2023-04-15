def get_blas_gomp_arch_deps():
    return [
        ("x86_64", [
            "third-party//IntelComposerXE:{}".format(native.read_config("fbcode", "mkl_lp64", "mkl_lp64_omp")),
        ]),
        ("aarch64", [
            "third-party//OpenBLAS:OpenBLAS",
            "third-party//openmp:omp",
        ]),
    ]

default_compiler_flags = [
    "-Wall",
    "-Wextra",
    "-Wno-unused-function",
    "-Wno-unused-parameter",
    "-Wno-error=strict-aliasing",
    "-Wno-shadow-compatible-local",
    "-Wno-maybe-uninitialized",  # aten is built with gcc as part of HHVM
    "-Wno-unknown-pragmas",
    "-Wno-strict-overflow",
    # See https://fb.facebook.com/groups/fbcode/permalink/1813348245368673/
    # These trigger on platform007
    "-Wno-stringop-overflow",
    "-Wno-class-memaccess",
    "-DHAVE_MMAP",
    "-DUSE_GCC_ATOMICS=1",
    "-D_FILE_OFFSET_BITS=64",
    "-DHAVE_SHM_OPEN=1",
    "-DHAVE_SHM_UNLINK=1",
    "-DHAVE_MALLOC_USABLE_SIZE=1",
    "-DTH_HAVE_THREAD",
    "-DCPU_CAPABILITY_DEFAULT",
    "-DTH_INDEX_BASE=0",
    "-DMAGMA_V2",
    "-DNO_CUDNN_DESTROY_HANDLE",
    "-DUSE_EXPERIMENTAL_CUDNN_V8_API",  # enable cudnn v8 api
    "-DUSE_FBGEMM",
    "-DUSE_QNNPACK",
    "-DUSE_PYTORCH_QNNPACK",
    # The dynamically loaded NVRTC trick doesn't work in fbcode,
    # and it's not necessary anyway, because we have a stub
    # nvrtc library which we load canonically anyway
    "-DUSE_DIRECT_NVRTC",
    "-DUSE_RUY_QMATMUL",
] + select({
    # XNNPACK depends on an updated version of pthreadpool interface, whose implementation
    # includes <pthread.h> - a header not available on Windows.
    "DEFAULT": ["-DUSE_XNNPACK"],
    "ovr_config//os:windows": [],
}) + (["-O1"] if native.read_config("fbcode", "build_mode_test_label", "") == "dev-nosan" else [])

compiler_specific_flags = {
    "clang": [
        "-Wno-absolute-value",
        "-Wno-pass-failed",
        "-Wno-braced-scalar-init",
    ],
    "gcc": [
        "-Wno-error=array-bounds",
    ],
}

def get_cpu_parallel_backend_flags():
    parallel_backend = native.read_config("pytorch", "parallel_backend", "openmp")
    defs = []
    if parallel_backend == "openmp":
        defs.append("-DAT_PARALLEL_OPENMP_FBCODE=1")
    elif parallel_backend == "tbb":
        defs.append("-DAT_PARALLEL_NATIVE_TBB_FBCODE=1")
    elif parallel_backend == "native":
        defs.append("-DAT_PARALLEL_NATIVE_FBCODE=1")
    else:
        fail("Unsupported parallel backend: " + parallel_backend)
    if native.read_config("pytorch", "exp_single_thread_pool", "0") == "1":
        defs.append("-DAT_EXPERIMENTAL_SINGLE_THREAD_POOL=1")
    mkl_ver = native.read_config("fbcode", "mkl_lp64", "mkl_lp64_omp")
    if mkl_ver == "mkl_lp64_seq":
        defs.append("-DATEN_MKL_SEQUENTIAL_FBCODE=1")
    return defs

def is_cpu_static_dispatch_build():
    mode = native.read_config("fbcode", "caffe2_static_dispatch_mode", "none")
    return mode == "cpu"
