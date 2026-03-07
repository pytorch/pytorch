# Forward environment variables to CMake variables.
#
# This replicates the behavior of setup.py / tools/setup_helpers/cmake.py which
# passes all BUILD_*, USE_*, and CMAKE_* environment variables as -D flags, plus
# a set of additional variables that don't follow the prefix convention.

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
  JAVA_HOME
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
  PYTHON_LIB_REL_PATH
)

# Low-priority aliases: if the canonical var is not set, use the alias.
set(_LOW_PRIORITY_ALIASES
  "CUDA_HOST_COMPILER=CMAKE_CUDA_HOST_COMPILER"
  "CUDAHOSTCXX=CUDA_HOST_COMPILER"
  "CMAKE_CUDA_HOST_COMPILER=CUDA_HOST_COMPILER"
  "CMAKE_CUDA_COMPILER=CUDA_NVCC_EXECUTABLE"
  "CUDACXX=CUDA_NVCC_EXECUTABLE"
)

# Forward aliased env vars (env name → different cmake name)
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

# Forward all BUILD_*, USE_* env vars not already set as CMake variables.
# This matches the existing behavior where setup.py passed everything with
# these prefixes through to CMake.
# We use execute_process + env to get the full list since CMake has no
# built-in way to enumerate environment variables.
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E environment
  OUTPUT_VARIABLE _all_env
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\n" ";" _env_lines "${_all_env}")
foreach(_line IN LISTS _env_lines)
  if(_line MATCHES "^(BUILD_[A-Za-z_0-9]+|USE_[A-Za-z_0-9]+)=(.*)")
    set(_var_name "${CMAKE_MATCH_1}")
    set(_var_value "${CMAKE_MATCH_2}")
    if(NOT DEFINED ${_var_name})
      set(${_var_name} "${_var_value}" CACHE STRING "From environment" FORCE)
    endif()
  endif()
endforeach()

# Low-priority aliases
foreach(_alias IN LISTS _LOW_PRIORITY_ALIASES)
  string(REPLACE "=" ";" _parts "${_alias}")
  list(GET _parts 0 _env_name)
  list(GET _parts 1 _cmake_name)
  if(DEFINED ENV{${_env_name}} AND NOT DEFINED ${_cmake_name})
    set(${_cmake_name} "$ENV{${_env_name}}" CACHE STRING "From env alias ${_env_name}" FORCE)
  endif()
endforeach()

# Ensure Python's purelib is on CMAKE_PREFIX_PATH so CMake can find
# packages installed there (e.g., pybind11, numpy).
# Use Python_EXECUTABLE if set (by scikit-build-core), otherwise find python3.
if(NOT Python_EXECUTABLE)
  find_program(Python_EXECUTABLE NAMES python3 python)
endif()
if(Python_EXECUTABLE)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('purelib'))"
    OUTPUT_VARIABLE _py_purelib
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(_py_purelib AND NOT "${_py_purelib}" STREQUAL "")
    list(PREPEND CMAKE_PREFIX_PATH "${_py_purelib}")
    # Preserve paths from the CMAKE_PREFIX_PATH environment variable.
    # Setting the cmake variable shadows the env var, so we must merge it in
    # explicitly. This ensures conda's prefix (e.g. /opt/conda/envs/py_3.10)
    # is present so cmake can find conda-provided libraries (libgomp, libnuma).
    if(DEFINED ENV{CMAKE_PREFIX_PATH} AND NOT "$ENV{CMAKE_PREFIX_PATH}" STREQUAL "")
      string(REPLACE ":" ";" _env_prefix "$ENV{CMAKE_PREFIX_PATH}")
      list(APPEND CMAKE_PREFIX_PATH ${_env_prefix})
    endif()
    list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)

    # Add -rpath-link dirs so the conda compat linker (-B compiler_compat) can
    # find versioned shared libraries (libnuma.so.1, libgomp.so.1,
    # libroctracer64.so.4, ...) when resolving transitive DT_NEEDED during
    # test-binary and torch_shm_manager linking.
    # Must be done here (before add_subdirectory) so subdirectory scopes inherit
    # the flags. System lib dirs come first to prefer system-native libraries
    # over conda's sysroot versions (which trigger GLIBC_PRIVATE link failures).
    set(_rp_flags "")
    # System multiarch lib dir (Debian/Ubuntu: /lib/x86_64-linux-gnu, etc.).
    if(CMAKE_LIBRARY_ARCHITECTURE)
      foreach(_d
          "/lib/${CMAKE_LIBRARY_ARCHITECTURE}"
          "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
        if(IS_DIRECTORY "${_d}")
          string(APPEND _rp_flags " -Wl,-rpath-link,${_d}")
        endif()
      endforeach()
    endif()
    # Traditional 64-bit lib dirs (RHEL/manylinux).
    foreach(_d "/lib64" "/usr/lib64")
      if(IS_DIRECTORY "${_d}")
        string(APPEND _rp_flags " -Wl,-rpath-link,${_d}")
      endif()
    endforeach()
    # ENV{CMAKE_PREFIX_PATH} entries (conda env, etc.).
    foreach(_prefix IN LISTS _env_prefix)
      foreach(_sub "lib" "lib64")
        if(IS_DIRECTORY "${_prefix}/${_sub}")
          string(APPEND _rp_flags " -Wl,-rpath-link,${_prefix}/${_sub}")
        endif()
      endforeach()
    endforeach()
    # ROCm: use ROCM_PATH env var (set in ROCm CI), fall back to /opt/rocm.
    if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
      set(_rocm_root "$ENV{ROCM_PATH}")
    else()
      set(_rocm_root "/opt/rocm")
    endif()
    foreach(_sub "lib" "lib64")
      if(IS_DIRECTORY "${_rocm_root}/${_sub}")
        string(APPEND _rp_flags " -Wl,-rpath-link,${_rocm_root}/${_sub}")
      endif()
    endforeach()
    # CUDA: use CUDA_PATH env var, fall back to /usr/local/cuda.
    if(DEFINED ENV{CUDA_PATH} AND NOT "$ENV{CUDA_PATH}" STREQUAL "")
      set(_cuda_root "$ENV{CUDA_PATH}")
    else()
      set(_cuda_root "/usr/local/cuda")
    endif()
    foreach(_sub "lib64" "lib")
      if(IS_DIRECTORY "${_cuda_root}/${_sub}")
        string(APPEND _rp_flags " -Wl,-rpath-link,${_cuda_root}/${_sub}")
      endif()
    endforeach()
    # MAGMA (ROCm): libmagma.so lives in MAGMA_HOME/lib, not in /opt/rocm/lib.
    if(DEFINED ENV{MAGMA_HOME} AND NOT "$ENV{MAGMA_HOME}" STREQUAL "")
      string(APPEND _rp_flags " -Wl,-rpath-link,$ENV{MAGMA_HOME}/lib")
    elseif(IS_DIRECTORY "${_rocm_root}/magma/lib")
      string(APPEND _rp_flags " -Wl,-rpath-link,${_rocm_root}/magma/lib")
    endif()
    # Aotriton (ROCm): installed into the source tree's torch/lib during the
    # build phase. The linker ignores non-existent rpath-link dirs, so it is
    # safe to add this unconditionally.
    string(APPEND _rp_flags " -Wl,-rpath-link,${PROJECT_SOURCE_DIR}/torch/lib")
    # torch-xpu-ops sycltla libs (XPU): libtorch-xpu-ops-sycltla-mha_*.so are
    # built into the cmake binary dir's lib/ subdir during the build phase.
    string(APPEND _rp_flags " -Wl,-rpath-link,${CMAKE_BINARY_DIR}/lib")
    # Intel oneAPI MKL SYCL (XPU): libmkl_sycl_blas.so.5, libmkl_sycl_dft.so.5,
    # libmkl_sycl_lapack.so.5 are in the MKL component, not the compiler component
    # that appears in CMAKE_PREFIX_PATH. Glob the standard install path.
    file(GLOB _mkl_lib_dirs "/opt/intel/oneapi/mkl/*/lib")
    foreach(_d IN LISTS _mkl_lib_dirs)
      string(APPEND _rp_flags " -Wl,-rpath-link,${_d}")
    endforeach()
    if(_rp_flags)
      string(APPEND CMAKE_EXE_LINKER_FLAGS "${_rp_flags}")
      string(APPEND CMAKE_SHARED_LINKER_FLAGS "${_rp_flags}")
      set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS}"    CACHE STRING "" FORCE)
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}" CACHE STRING "" FORCE)
    endif()
  endif()
endif()

# MAX_JOBS → CMAKE_BUILD_PARALLEL_LEVEL (scikit-build-core respects this)
if(DEFINED ENV{MAX_JOBS} AND NOT DEFINED CMAKE_BUILD_PARALLEL_LEVEL)
  set(ENV{CMAKE_BUILD_PARALLEL_LEVEL} "$ENV{MAX_JOBS}")
endif()

# BUILD_PYTHON_ONLY implies BUILD_LIBTORCHLESS=ON and requires LIBTORCH_LIB_PATH.
# This matches setup.py behavior.
if(DEFINED ENV{BUILD_PYTHON_ONLY})
  string(TOUPPER "$ENV{BUILD_PYTHON_ONLY}" _bpo_val)
  if(_bpo_val MATCHES "^(ON|1|YES|TRUE|Y)$")
    set(ENV{BUILD_LIBTORCHLESS} "ON")
    if(NOT DEFINED BUILD_LIBTORCHLESS)
      set(BUILD_LIBTORCHLESS ON CACHE BOOL "Build without libtorch" FORCE)
    endif()
  endif()
endif()

# USE_NIGHTLY bypasses the build entirely and downloads a pre-built wheel.
# This is not supported via CMake — use the standalone script instead.
if(DEFINED ENV{USE_NIGHTLY})
  message(FATAL_ERROR
    "USE_NIGHTLY is not supported with the scikit-build-core build system. "
    "Use 'python tools/nightly_wheel.py' instead, or install directly with pip: "
    "pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu"
  )
endif()

# Conflict check
if(DEFINED ENV{BUILD_LIBTORCH_WHL} AND DEFINED ENV{BUILD_PYTHON_ONLY})
  string(TOUPPER "$ENV{BUILD_LIBTORCH_WHL}" _bltw)
  string(TOUPPER "$ENV{BUILD_PYTHON_ONLY}" _bpo)
  if(_bltw MATCHES "^(ON|1|YES|TRUE|Y)$" AND _bpo MATCHES "^(ON|1|YES|TRUE|Y)$")
    message(FATAL_ERROR
      "Conflict: BUILD_LIBTORCH_WHL and BUILD_PYTHON_ONLY cannot both be ON.")
  endif()
endif()

# Build type logic previously in tools/setup_helpers/env.py.
# CMAKE_BUILD_TYPE always takes precedence. If not set, check DEBUG and
# REL_WITH_DEB_INFO env vars.
if(NOT CMAKE_BUILD_TYPE AND NOT DEFINED ENV{CMAKE_BUILD_TYPE})
  if(DEFINED ENV{DEBUG})
    string(TOUPPER "$ENV{DEBUG}" _debug_val)
    if(_debug_val MATCHES "^(ON|1|YES|TRUE|Y)$")
      set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
    endif()
  endif()
  if(NOT CMAKE_BUILD_TYPE AND DEFINED ENV{REL_WITH_DEB_INFO})
    string(TOUPPER "$ENV{REL_WITH_DEB_INFO}" _rwdi_val)
    if(_rwdi_val MATCHES "^(ON|1|YES|TRUE|Y)$")
      set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
    endif()
  endif()
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
  endif()
elseif(DEFINED ENV{CMAKE_BUILD_TYPE} AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "$ENV{CMAKE_BUILD_TYPE}" CACHE STRING "Build type" FORCE)
endif()
