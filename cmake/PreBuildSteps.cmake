# Pre-build steps previously handled by setup.py:
# 1. Git submodule initialization and verification
# 2. NCCL checkout from pinned version
#
# This file is included early in CMakeLists.txt, before option() declarations.
# It relies on env vars (forwarded by EnvVarForwarding.cmake) and CMake -D
# cache variables, both of which are available at this point.

find_package(Git QUIET)

# --- Submodule initialization and verification ---
# Matches the logic in setup.py::check_submodules().
if(NOT DEFINED USE_SYSTEM_LIBS OR NOT USE_SYSTEM_LIBS)
  # Read submodule paths from .gitmodules if available, otherwise use defaults.
  set(_gitmodules_file "${PROJECT_SOURCE_DIR}/.gitmodules")
  if(EXISTS "${_gitmodules_file}")
    file(STRINGS "${_gitmodules_file}" _gitmodule_lines REGEX "^[[:space:]]*path")
    set(_submodule_folders)
    foreach(_line IN LISTS _gitmodule_lines)
      string(REGEX REPLACE ".*=[[:space:]]*" "" _path "${_line}")
      list(APPEND _submodule_folders "${PROJECT_SOURCE_DIR}/${_path}")
    endforeach()
  else()
    set(_submodule_folders
      "${PROJECT_SOURCE_DIR}/third_party/gloo"
      "${PROJECT_SOURCE_DIR}/third_party/cpuinfo"
      "${PROJECT_SOURCE_DIR}/third_party/onnx"
      "${PROJECT_SOURCE_DIR}/third_party/fbgemm"
      "${PROJECT_SOURCE_DIR}/third_party/cutlass"
    )
  endif()

  set(_all_missing TRUE)
  foreach(_dir IN LISTS _submodule_folders)
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

  # Verify submodules contain expected files (catches corrupt/partial checkouts).
  set(_expected_files CMakeLists.txt Makefile setup.py LICENSE LICENSE.md LICENSE.txt)
  foreach(_dir IN LISTS _submodule_folders)
    set(_found FALSE)
    foreach(_file IN LISTS _expected_files)
      if(EXISTS "${_dir}/${_file}")
        set(_found TRUE)
        break()
      endif()
    endforeach()
    if(NOT _found)
      message(FATAL_ERROR
        "Submodule ${_dir} appears incomplete (none of "
        "${_expected_files} found).\n"
        "Please run: git submodule update --init --recursive"
      )
    endif()
  endforeach()
  # Extra check for fbgemm's nested dependency
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/third_party/fbgemm/external/asmjit/CMakeLists.txt")
    message(FATAL_ERROR
      "third_party/fbgemm/external/asmjit appears incomplete.\n"
      "Please run: git submodule update --init --recursive"
    )
  endif()
endif()

# --- NCCL checkout ---
# Clone NCCL from the pinned tag if building with NCCL and not using
# system NCCL.  Conditions match build_pytorch_libs.py::build_pytorch().
if(NOT USE_SYSTEM_NCCL)
  # Only attempt if USE_DISTRIBUTED, USE_CUDA, USE_NCCL are not explicitly OFF.
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
    if(NOT EXISTS "${_nccl_dir}")
      # Select pin file: try a CUDA-version-specific pin (e.g. nccl-cu126.txt)
      # first, fall back to nccl.txt.  Adding a new pin file is sufficient to
      # support a new CUDA version — no CMake changes needed.
      set(_nccl_pin_name "nccl.txt")
      if(DEFINED ENV{DESIRED_CUDA})
        set(_cuda_ver "$ENV{DESIRED_CUDA}")
      elseif(DEFINED ENV{CUDA_VERSION})
        set(_cuda_ver "$ENV{CUDA_VERSION}")
      else()
        set(_cuda_ver "")
      endif()
      set(_cuda_suffix "")
      if(_cuda_ver MATCHES "^([0-9]+)\\.([0-9]+)")
        set(_cuda_suffix "cu${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
      elseif(_cuda_ver MATCHES "^(cu[0-9]+)$")
        set(_cuda_suffix "${CMAKE_MATCH_1}")
      endif()
      if(_cuda_suffix)
        set(_versioned_pin "${PROJECT_SOURCE_DIR}/.ci/docker/ci_commit_pins/nccl-${_cuda_suffix}.txt")
        if(EXISTS "${_versioned_pin}")
          set(_nccl_pin_name "nccl-${_cuda_suffix}.txt")
        endif()
      endif()

      set(_nccl_pin_file "${PROJECT_SOURCE_DIR}/.ci/docker/ci_commit_pins/${_nccl_pin_name}")
      if(EXISTS "${_nccl_pin_file}")
        file(READ "${_nccl_pin_file}" _nccl_tag)
        string(STRIP "${_nccl_tag}" _nccl_tag)
        message(STATUS "Checking out NCCL release tag: ${_nccl_tag} (from ${_nccl_pin_name})")
        include(FetchContent)
        FetchContent_Declare(
          nccl
          GIT_REPOSITORY https://github.com/NVIDIA/nccl
          GIT_TAG        "${_nccl_tag}"
          GIT_SHALLOW    TRUE
          SOURCE_DIR     "${_nccl_dir}"
        )
        FetchContent_Populate(nccl)
      endif()
    endif()
  endif()
endif()
