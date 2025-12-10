# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

macro(cmake_nvcc_filter_implicit_info lang lang_var_)
  # Remove the CUDA Toolkit include directories from the set of
  # implicit system include directories.
  # This resolves the issue that NVCC doesn't specify these
  # includes as SYSTEM includes when compiling device code, and sometimes
  # they contain headers that generate warnings, so let users mark them
  # as SYSTEM explicitly
  if(${lang_var_}TOOLKIT_INCLUDE_DIRECTORIES)
    list(REMOVE_ITEM CMAKE_${lang}_IMPLICIT_INCLUDE_DIRECTORIES
      ${${lang_var_}TOOLKIT_INCLUDE_DIRECTORIES}
      )
  endif()
endmacro()
