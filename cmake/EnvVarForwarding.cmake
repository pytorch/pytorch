# Forward environment variables to CMake variables.
#
# Forwarding rule: an environment variable reaches CMake (as a cache variable of
# the same name) if it (a) starts with BUILD_, USE_, or CMAKE_, (b) ends with
# EXITCODE or EXITCODE__TRYRUN_OUTPUT, or (c) appears in the _ENV_ALIASES /
# _ENV_PASSTHROUGH / _LOW_PRIORITY_ALIASES lists below. Anything else is not
# forwarded -- set it as a CMake option with -D / cmake.define instead.
#
# ============================================================================
# Build environment variable reference. Vars below that match the forwarding
# rule are passed through by this module; the rest are handled where noted.
# See CONTRIBUTING.md for the recommended developer setup.
# ============================================================================
#
# Everyday knobs:
#   DEBUG=1                  build with -O0 -g; mapped to the CMake build type
#                            by [[tool.scikit-build.overrides]] in pyproject.toml
#   REL_WITH_DEB_INFO=1      optimized build with -g, same mechanism as DEBUG
#   MAX_JOBS                 compile parallelism; aliased to
#                            CMAKE_BUILD_PARALLEL_LEVEL by [tool.scikit-build.env]
#                            in pyproject.toml
#   CC / CXX / CFLAGS        compiler and flags; read by CMake / scikit-build-core
#                            directly (CFLAGS also applies to C++ unless CXXFLAGS
#                            is set)
#   USE_CUDA=0, BUILD_TEST=0, ...   feature toggles, next section
#   TORCH_CUDA_ARCH_LIST     CUDA arches to build for, e.g. "8.0;9.0"
#
# Feature toggles (USE_*/BUILD_*, forwarded by prefix):
#   USE_CUDA=0                disables CUDA build
#   USE_CUDNN=0               disables the cuDNN build
#   USE_CUSPARSELT=0          disables the cuSPARSELt build
#   USE_CUDSS=0               disables the cuDSS build
#   USE_CUFILE=0              disables the cuFile build
#   USE_FBGEMM=0              disables the FBGEMM build
#   USE_MSLK=0                disables the MSLK build
#   USE_KINETO=0              disables libkineto profiling
#   USE_NUMPY=0               disables the NumPy build
#   USE_ITT=0                 disables Intel(R) VTune ITT functionality
#   USE_NNPACK=0              disables NNPACK build
#   USE_DISTRIBUTED=0         disables distributed (c10d, gloo, mpi, etc.) build
#   USE_TENSORPIPE=0          disables the Tensorpipe backend
#   USE_GLOO=0                disables the gloo backend
#   USE_MPI=0                 disables the MPI backend
#   USE_SYSTEM_NCCL=0         use the submoduled nccl instead of system nccl
#   USE_OPENMP=0              disables OpenMP parallelization
#   USE_MKLDNN=0              disables MKLDNN
#   USE_MKLDNN_ACL           enables Compute Library backend for MKLDNN on Arm
#                            (USE_MKLDNN must be explicitly enabled)
#   USE_STATIC_MKL           prefer to link MKL statically (Unix only)
#   USE_FLASH_ATTENTION=0    disables flash attention for scaled dot product attn
#   USE_MEM_EFF_ATTENTION=0  disables memory efficient attention for SDPA
#   USE_ROCM_KERNEL_ASSERT=1 enables kernel assert on ROCm
#   USE_ROCM_CK_GEMM=1       builds the CK GEMM backend on ROCm
#   USE_ROCM_CK_SDPA=1       builds the CK SDPA backend on ROCm
#   USE_LAYERNORM_FAST_RECIPROCAL  fast reciprocals for layer norm (default on)
#   USE_MIMALLOC             static-link mimalloc into c10 (default: Windows/AArch64)
#   USE_CUSTOM_DEBINFO="a.cpp;b.cpp"  build debug info only for the listed files
#   USE_SYSTEM_LIBS          use system-provided third-party libraries; expands
#                            to the individual USE_SYSTEM_* toggles in CMake
#   BUILD_TEST=0             disables the test build
#   BUILD_BINARY             enables the extra binaries/ build
#   BUILD_LIBTORCH_WHL       builds libtorch.so and deps as a wheel
#   BUILD_PYTHON_ONLY        builds the python wheel against a separate libtorch.so
#
# Architecture selection (forwarded by prefix or read from the environment):
#   TORCH_CUDA_ARCH_LIST     CUDA arches to build for, e.g. "6.0;7.0" (passthrough)
#   TORCH_XPU_ARCH_LIST      XPU arches, e.g. "ats-m150,lnl-m" (passthrough)
#   PYTORCH_ROCM_ARCH        AMD GPU targets, e.g. "gfx900;gfx906" (read from env
#                            in cmake/public/utils.cmake)
#
# Library/backend selection (passthrough or read from the environment):
#   BLAS                     MKL, Eigen, ATLAS, FlexiBLAS, or OpenBLAS; fails the
#                            build if the requested BLAS is not found (passthrough)
#   MKL_THREADING            MKL threading mode: SEQ, TBB, or OMP (default)
#   MKLDNN_CPU_RUNTIME       MKL-DNN threading mode: TBB or OMP (default)
#   ATEN_THREADING           OMP or NATIVE intra-/inter-op parallel backend
#   ONNX_NAMESPACE           namespace for the ONNX built here
#   ATEN_AVX512_256=TRUE     let ATen AVX2 kernels use 32 ymm registers (read from
#                            the environment in cmake/Codegen.cmake)
#
# Library location hints (passthrough, alias, read from env, or CMake-native):
#   CUDA_HOME (Linux/macOS) / CUDA_PATH (Windows)  CUDA install location
#   CUDAHOSTCXX              host compiler for nvcc (alias)
#   CUDA_NVCC_EXECUTABLE     nvcc to use (passthrough; CI points this at a cache)
#   CUDNN_LIBRARY / CUDNN_INCLUDE_DIR / CUDNN_LIB_DIR   cuDNN location (CUDNN_LIB_DIR
#                            is an alias for CUDNN_LIBRARY)
#   MIOPEN_PATH              MIOpen install root (read from env in LoadHIP.cmake).
#                            NOTE: the old MIOPEN_LIB_DIR/INCLUDE_DIR/LIBRARY env
#                            vars are no longer used.
#   NCCL_ROOT / NCCL_LIB_DIR / NCCL_INCLUDE_DIR   nccl location (read from env in
#                            cmake/Modules/FindNCCL.cmake)
#   ACL_ROOT_DIR             Arm Compute Library location (read from env in
#                            cmake/Modules/FindACL.cmake)
#   LIBRARY_PATH / LD_LIBRARY_PATH   searched for libraries (compiler/linker native)
#
# Handled outside this module (NOT forwarded here; see also the everyday knobs
# above, which are all handled via pyproject.toml or natively):
#   PYTORCH_BUILD_VERSION / PYTORCH_BUILD_NUMBER   wheel version; consumed by the
#                            version metadata provider (tools/metadata)
#
# CMake options, set with -D / cmake.define (NOT environment variables):
#   DEBUG_CUDA               when compiling DEBUG, also build CUDA kernels with
#                            debug flags (may OOM nvcc). This was always a CMake
#                            option; the setup.py comment that listed it as an env
#                            var was inaccurate -- it was never forwarded.
#
# Removed with setup.py (no longer available):
#   CMAKE_FRESH              force a fresh configure. Delete the build/ directory
#                            to reconfigure from scratch instead.
#   CMAKE_ONLY               configure without building; no equivalent (this was a
#                            setup.py-only debugging aid).
#   USE_NINJA                select the generator via CMAKE_GENERATOR=Ninja instead
#                            (ninja is the default when available).

# Additional env vars that are forwarded with a different CMake variable name.
set(_ENV_ALIASES
  "CUDNN_LIB_DIR=CUDNN_LIBRARY"
  "USE_CUDA_STATIC_LINK=CAFFE2_STATIC_LINK_CUDA"
)

# Additional env vars forwarded with the same name.
set(_ENV_PASSTHROUGH
  UBSAN_FLAGS
  BLAS
  WITH_BLAS
  CUDA_HOST_COMPILER
  CUDA_NVCC_EXECUTABLE
  CUDA_SEPARABLE_COMPILATION
  CUDNN_LIBRARY
  CUDNN_INCLUDE_DIR
  CUDNN_ROOT
  EXPERIMENTAL_SINGLE_THREAD_POOL
  INSTALL_TEST
  INTEL_MKL_DIR
  INTEL_OMP_DIR
  MKL_THREADING
  MKLDNN_CPU_RUNTIME
  MSVC_Z7_OVERRIDE
  CAFFE2_USE_MSVC_STATIC_RUNTIME
  Numa_INCLUDE_DIR
  Numa_LIBRARIES
  ONNX_ML
  ONNX_NAMESPACE
  ATEN_THREADING
  WERROR
  OPENSSL_ROOT_DIR
  STATIC_DISPATCH_BACKEND
  SELECTED_OP_LIST
  TORCH_CUDA_ARCH_LIST
  TORCH_XPU_ARCH_LIST
  TRACING_BASED
)

# Low-priority aliases: if the canonical var is not set, use the alias.
set(_LOW_PRIORITY_ALIASES
  "CUDA_HOST_COMPILER=CMAKE_CUDA_HOST_COMPILER"
  "CUDAHOSTCXX=CUDA_HOST_COMPILER"
  "CMAKE_CUDA_HOST_COMPILER=CUDA_HOST_COMPILER"
  "CMAKE_CUDA_COMPILER=CUDA_NVCC_EXECUTABLE"
  "CUDACXX=CUDA_NVCC_EXECUTABLE"
)

# Forward aliased env vars (env name -> different cmake name)
foreach(_alias IN LISTS _ENV_ALIASES)
  string(REPLACE "=" ";" _parts "${_alias}")
  list(GET _parts 0 _env_name)
  list(GET _parts 1 _cmake_name)
  if(DEFINED ENV{${_env_name}} AND NOT DEFINED ${_cmake_name})
    set(${_cmake_name} "$ENV{${_env_name}}" CACHE STRING "From env ${_env_name}" FORCE)
  endif()
endforeach()

# Forward passthrough env vars (same name)
foreach(_var IN LISTS _ENV_PASSTHROUGH)
  if(DEFINED ENV{${_var}} AND NOT DEFINED ${_var})
    set(${_var} "$ENV{${_var}}" CACHE STRING "From env ${_var}" FORCE)
  endif()
endforeach()

# Forward all BUILD_*, USE_*, CMAKE_* environment variables (plus names ending
# in EXITCODE / EXITCODE__TRYRUN_OUTPUT) into the CMake cache, mirroring the -D
# flags setup.py used to pass.
#
# CMake cannot enumerate environment variables, and serializing the whole
# environment to text and re-parsing it in CMake is unsafe: values such as PS1
# contain ';' and '\' (and some exported shell functions even contain newlines),
# all of which collide with CMake's list, escape, and line semantics and
# silently corrupt unrelated variables. The top-level CMakeLists.txt already
# requires Python (find_package(Python COMPONENTS Interpreter REQUIRED)) before
# including this module, so read os.environ directly there -- the full
# environment is never serialized -- and have it emit only the selected,
# properly escaped cache assignments for CMake to evaluate.

# Applies one forwarded variable. An explicitly-set environment variable takes
# priority, matching the -D semantics this module emulates: override the cache
# (do not merely fill when undefined) so a value left by an earlier env-less
# configure -- an option() default or a ninja-triggered reconfigure -- cannot
# permanently shadow the environment.
function(_envfwd_apply _name _value)
  if(NOT DEFINED ${_name} OR NOT "${${_name}}" STREQUAL "${_value}")
    set(${_name} "${_value}" CACHE STRING "From environment" FORCE)
  endif()
endfunction()

# Reads os.environ and prints `_envfwd_apply("<name>" "<value>")` for each
# selected variable, escaping the value for a CMake double-quoted argument.
set(_envfwd_script [==[
import os, re, sys

select = re.compile(r"^(BUILD_|USE_|CMAKE_)|(EXITCODE|EXITCODE__TRYRUN_OUTPUT)$")

def q(s):
    # Escape for a CMake double-quoted argument. Backslash and quote are
    # structural; '$' is escaped to suppress ${}/$ENV{} expansion. ';' and
    # newlines are literal inside quotes and need no escaping.
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")

sys.stdout.write("\n".join(
    '_envfwd_apply("%s" "%s")' % (q(name), q(value))
    for name, value in os.environ.items()
    if select.search(name)
))
]==])

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "${_envfwd_script}"
  OUTPUT_VARIABLE _envfwd_code
  RESULT_VARIABLE _envfwd_rc
)
if(NOT _envfwd_rc EQUAL 0)
  message(FATAL_ERROR
    "EnvVarForwarding: failed to read the environment via Python (exit ${_envfwd_rc}).")
endif()
cmake_language(EVAL CODE "${_envfwd_code}")

# Low-priority aliases
foreach(_alias IN LISTS _LOW_PRIORITY_ALIASES)
  string(REPLACE "=" ";" _parts "${_alias}")
  list(GET _parts 0 _env_name)
  list(GET _parts 1 _cmake_name)
  if(DEFINED ENV{${_env_name}} AND NOT DEFINED ${_cmake_name})
    set(${_cmake_name} "$ENV{${_env_name}}" CACHE STRING "From env alias ${_env_name}" FORCE)
  endif()
endforeach()

# Ensure Python's sys.prefix (the venv/conda env root) and purelib are on
# CMAKE_PREFIX_PATH so CMake can find packages installed there.
#
# - sys.prefix is needed because conda-style envs put libraries under
#   <prefix>/lib (Linux) or <prefix>/Library/lib (Windows). CMake 3.28
#   removed the find_library() heuristic that derived <prefix>/lib from
#   <prefix>/bin entries on PATH, so without sys.prefix on the prefix
#   path, find_package(MKL) and similar fail to locate conda-provided
#   libraries (e.g. mkl_intel_lp64, libiomp5md). The Linux CI scripts
#   used to set CMAKE_PREFIX_PATH=$CONDA_PREFIX explicitly as a
#   workaround for the same issue (see gh-119557); having it here makes
#   that redundant and gives the same coverage to Windows pull-CI and
#   to local builds outside of CI.
# - purelib is needed for python-package CMake configs (e.g., pybind11,
#   numpy headers).
if(Python_EXECUTABLE)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
      "import sys, sysconfig; print(sys.prefix); print(sysconfig.get_path('purelib'))"
    OUTPUT_VARIABLE _py_paths
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(_py_paths AND NOT "${_py_paths}" STREQUAL "")
    string(REPLACE "\n" ";" _py_paths "${_py_paths}")
    # On Windows, conda envs lay out installed libraries under
    # <prefix>/Library/{lib,include,bin}, which CMake's find_library does
    # not search by default. Prepend <prefix>/Library so the standard
    # <prefix>/lib heuristic resolves <prefix>/Library/lib (where MKL,
    # OpenSSL, libiomp5md, etc. live in conda-on-Windows installs).
    list(GET _py_paths 0 _py_prefix)
    if(WIN32 AND EXISTS "${_py_prefix}/Library")
      list(PREPEND _py_paths "${_py_prefix}/Library")
    endif()
    list(PREPEND CMAKE_PREFIX_PATH ${_py_paths})
    # Preserve paths from the CMAKE_PREFIX_PATH environment variable.
    # Setting the cmake variable shadows the env var, so we must merge it in
    # explicitly.
    if(DEFINED ENV{CMAKE_PREFIX_PATH} AND NOT "$ENV{CMAKE_PREFIX_PATH}" STREQUAL "")
      if(WIN32)
        # On Windows the env var is already ;-separated and : appears in drive
        # letters (e.g. C:\conda\envs\py310), so use it as-is.
        set(_env_prefix "$ENV{CMAKE_PREFIX_PATH}")
      else()
        string(REPLACE ":" ";" _env_prefix "$ENV{CMAKE_PREFIX_PATH}")
      endif()
      list(APPEND CMAKE_PREFIX_PATH ${_env_prefix})
    endif()
    list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
  endif()
endif()

# BUILD_PYTHON_ONLY implies BUILD_LIBTORCHLESS=ON.
if(BUILD_PYTHON_ONLY)
  set(BUILD_LIBTORCHLESS ON CACHE BOOL "Build without libtorch" FORCE)
endif()

# Installing pre-built nightly binaries instead of building is handled by
# tools/nightly.py, not by the build: a PEP 517 build cannot skip itself.
# Fail loudly rather than let USE_NIGHTLY be silently ignored.
if(USE_NIGHTLY)
  message(FATAL_ERROR
    "USE_NIGHTLY is not supported with the scikit-build-core build system. "
    "Use 'python tools/nightly.py checkout' instead (it checks out the nightly "
    "commit and installs matching pre-built binaries; see --help), or install "
    "a nightly wheel directly: "
    "pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu"
  )
endif()

# Conflict check
if(BUILD_LIBTORCH_WHL AND BUILD_PYTHON_ONLY)
  message(FATAL_ERROR
    "Conflict: BUILD_LIBTORCH_WHL and BUILD_PYTHON_ONLY cannot both be ON.")
endif()
