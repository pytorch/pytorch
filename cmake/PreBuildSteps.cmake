# Pre-build steps previously handled by setup.py:
# 1. Git submodule initialization and verification
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
    file(STRINGS "${_gitmodules_file}" _gitmodule_lines REGEX "^[ \t]*path")
    set(_submodule_folders)
    foreach(_line IN LISTS _gitmodule_lines)
      string(REGEX REPLACE ".*=[ \t]*" "" _path "${_line}")
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

  # Only attempt `git submodule update` when building from a git checkout.
  # Source tarballs / nightly build trees have no .git directory; running
  # `git submodule` there would fail and abort the build even when the
  # submodule trees are already populated from the tarball.
  if(_all_missing AND GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
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
