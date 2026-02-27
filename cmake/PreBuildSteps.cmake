# Pre-build steps previously handled by setup.py:
# 1. Git submodule initialization
# 2. NCCL checkout from pinned version

find_package(Git QUIET)

# --- Submodule initialization ---
# Check if submodules are present; if not, initialize them.
# Matches the logic in setup.py::check_submodules().
if(NOT DEFINED USE_SYSTEM_LIBS OR NOT USE_SYSTEM_LIBS)
  set(_check_dirs
    "${PROJECT_SOURCE_DIR}/third_party/gloo"
    "${PROJECT_SOURCE_DIR}/third_party/cpuinfo"
    "${PROJECT_SOURCE_DIR}/third_party/onnx"
    "${PROJECT_SOURCE_DIR}/third_party/fbgemm"
    "${PROJECT_SOURCE_DIR}/third_party/cutlass"
  )
  set(_all_missing TRUE)
  foreach(_dir IN LISTS _check_dirs)
    if(EXISTS "${_dir}" AND IS_DIRECTORY "${_dir}")
      file(GLOB _contents "${_dir}/*")
      if(_contents)
        set(_all_missing FALSE)
        break()
      endif()
    endif()
  endforeach()

  if(_all_missing AND GIT_FOUND)
    message(STATUS "Initializing git submodules...")
    execute_process(
      COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      RESULT_VARIABLE _submodule_result
    )
    if(NOT _submodule_result EQUAL 0)
      message(FATAL_ERROR
        "Git submodule initialization failed. Please run:\n"
        "  git submodule update --init --recursive"
      )
    endif()
  endif()
endif()

# --- NCCL checkout ---
# Clone NCCL from the pinned tag if building with NCCL and not using
# system NCCL.  Conditions match build_pytorch_libs.py::build_pytorch().
if(NOT USE_SYSTEM_NCCL)
  # Only attempt if USE_DISTRIBUTED, USE_CUDA, USE_NCCL are not explicitly OFF.
  # At this point these may not be fully resolved yet, so we check for
  # explicit OFF values. The default is to attempt the checkout.
  set(_skip_nccl FALSE)
  foreach(_var USE_DISTRIBUTED USE_CUDA USE_NCCL)
    if(DEFINED ${_var})
      string(TOUPPER "${${_var}}" _val)
      if(_val MATCHES "^(OFF|0|NO|FALSE|N)$")
        set(_skip_nccl TRUE)
        break()
      endif()
    endif()
  endforeach()

  if(NOT _skip_nccl)
    set(_nccl_dir "${PROJECT_SOURCE_DIR}/third_party/nccl")
    if(NOT EXISTS "${_nccl_dir}" AND GIT_FOUND)
      # Read pinned version
      set(_nccl_pin_file "${PROJECT_SOURCE_DIR}/.ci/docker/ci_commit_pins/nccl.txt")
      if(EXISTS "${_nccl_pin_file}")
        file(READ "${_nccl_pin_file}" _nccl_tag)
        string(STRIP "${_nccl_tag}" _nccl_tag)
        message(STATUS "Checking out NCCL release tag: ${_nccl_tag}")
        execute_process(
          COMMAND ${GIT_EXECUTABLE} clone --depth 1 --branch "${_nccl_tag}"
                  https://github.com/NVIDIA/nccl
          WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/third_party"
          RESULT_VARIABLE _nccl_result
        )
        if(NOT _nccl_result EQUAL 0)
          message(WARNING "NCCL checkout failed (tag: ${_nccl_tag}). "
                          "NCCL support may not be available.")
        endif()
      endif()
    endif()
  endif()
endif()
