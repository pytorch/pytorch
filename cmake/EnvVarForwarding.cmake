# Forward environment variables to CMake variables.
#
# Forwarding rule: an environment variable reaches CMake (as a cache variable of
# the same name) if it (a) starts with BUILD_, USE_, or CMAKE_, (b) ends with
# EXITCODE or EXITCODE__TRYRUN_OUTPUT, or (c) appears in the _ENV_ALIASES /
# _ENV_PASSTHROUGH lists below. Anything else is not
# forwarded -- set it as a CMake option with -D / cmake.define instead.
#
# Precedence: a forwarded variable overrides the cache on every configure, so the
# environment wins over -D / cmake.define, which win over pyproject.toml, and all
# three win over what the build directory remembers from the last configure. To
# change one setting for a single command, put the variable in front of the
# command. An alias applies only when its canonical name is not in the
# environment itself.
#
# The full reference -- every build environment variable, what it does, and how it reaches
# CMake -- lives in the "Build environment variable reference" section of CONTRIBUTING.md.
# Keep it in sync with the lists below.

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
  FLASH_ATTENTION_MAX_JOBS
  INSTALL_TEST
  INTEL_MKL_DIR
  INTEL_OMP_DIR
  LLVM_BOLT_PROFILES_DIR
  MKL_THREADING
  MKLDNN_CPU_RUNTIME
  MSVC_Z7_OVERRIDE
  CAFFE2_USE_MSVC_STATIC_RUNTIME
  CAFFE2_CUSTOM_PROTOC_EXECUTABLE
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
  TORCH_CK_GEMM_ARCH_LIST
  TORCH_XPU_ARCH_LIST
  TRACING_BASED
)

# Env vars forwarded under a different CMake name ("env name=cmake name"). Where
# several aliases name the same variable, the first listed one wins.
set(_ENV_ALIASES
  "CUDNN_LIB_DIR=CUDNN_LIBRARY"
  "USE_CUDA_STATIC_LINK=CAFFE2_STATIC_LINK_CUDA"
  "CUDA_HOST_COMPILER=CMAKE_CUDA_HOST_COMPILER"
  "CUDAHOSTCXX=CUDA_HOST_COMPILER"
  "CMAKE_CUDA_HOST_COMPILER=CUDA_HOST_COMPILER"
  "CMAKE_CUDA_COMPILER=CUDA_NVCC_EXECUTABLE"
  "CUDACXX=CUDA_NVCC_EXECUTABLE"
)

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
# Names this module forwarded, and their EFFECTIVE values at the end of configure,
# written for native-AOT stage 2: it reconfigures this build before relinking, and has to
# know whether that changed any setting. It cannot read CMakeCache.txt for this -- a
# dependent option that loses availability keeps its stale value there (retyped INTERNAL)
# while the build uses the force value -- so record what the build actually used.
set(ENVFWD_EFFECTIVE_FILE "env_forwarded.txt")

# The list lives in the cache rather than this configure's scope: a configure forwards
# only what its own environment holds, so a later reconfigure from a barer shell would
# record fewer names and stage 2 would read a missing name as a changed setting.
function(_envfwd_note _name)
  set(_seen "${ENVFWD_FORWARDED_NAMES}")
  list(APPEND _seen "${_name}")
  list(REMOVE_DUPLICATES _seen)
  list(SORT _seen)
  set(ENVFWD_FORWARDED_NAMES "${_seen}" CACHE INTERNAL
      "Variables EnvVarForwarding has forwarded into this build")
endfunction()

function(_envfwd_write_effective)
  set(_names "${ENVFWD_FORWARDED_NAMES}")
  set(_lines "")
  foreach(_n IN LISTS _names)
    # One line per name, so a newline inside a value would corrupt the comparison.
    string(REPLACE "\n" "\\n" _v "${${_n}}")
    string(APPEND _lines "${_n}=${_v}\n")
  endforeach()
  file(WRITE "${CMAKE_BINARY_DIR}/${ENVFWD_EFFECTIVE_FILE}" "${_lines}")
endfunction()

# End of the top-level scope, so the value recorded is the one the build settled on.
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL _envfwd_write_effective)

function(_envfwd_apply _name _value)
  _envfwd_note("${_name}")
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

# Passthrough names, same semantics as the prefix names above.
foreach(_var IN LISTS _ENV_PASSTHROUGH)
  if(DEFINED ENV{${_var}})
    _envfwd_apply("${_var}" "$ENV{${_var}}")
  endif()
endforeach()

# Aliases apply with the same semantics, but only when the canonical name is not
# in the environment itself (the prefix and passthrough passes above already
# applied it), and the first listed alias of a name wins over later ones.
set(_envfwd_aliased "")
foreach(_alias IN LISTS _ENV_ALIASES)
  string(REPLACE "=" ";" _parts "${_alias}")
  list(GET _parts 0 _env_name)
  list(GET _parts 1 _cmake_name)
  if(DEFINED ENV{${_env_name}} AND NOT DEFINED ENV{${_cmake_name}}
     AND NOT "${_cmake_name}" IN_LIST _envfwd_aliased)
    list(APPEND _envfwd_aliased "${_cmake_name}")
    _envfwd_apply("${_cmake_name}" "$ENV{${_env_name}}")
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
