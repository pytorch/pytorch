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

# Forward all BUILD_*, USE_*, CMAKE_* env vars not already set as CMake
# variables, plus vars ending in EXITCODE or EXITCODE__TRYRUN_OUTPUT.
# This matches the existing behavior where setup.py passed everything with
# these prefixes/suffixes through to CMake.
# We use execute_process + env to get the full list since CMake has no
# built-in way to enumerate environment variables.
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E environment
  OUTPUT_VARIABLE _all_env
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\n" ";" _env_lines "${_all_env}")
foreach(_line IN LISTS _env_lines)
  if(_line MATCHES "^([A-Za-z_0-9]+)=(.*)")
    set(_var_name "${CMAKE_MATCH_1}")
    set(_var_value "${CMAKE_MATCH_2}")
    # Only forward vars with BUILD_/USE_/CMAKE_ prefix or *EXITCODE* suffix.
    string(REGEX MATCH "^(BUILD_|USE_|CMAKE_)" _has_prefix "${_var_name}")
    string(REGEX MATCH "(EXITCODE|EXITCODE__TRYRUN_OUTPUT)$" _has_suffix "${_var_name}")
    if(NOT _has_prefix AND NOT _has_suffix)
      continue()
    endif()
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
