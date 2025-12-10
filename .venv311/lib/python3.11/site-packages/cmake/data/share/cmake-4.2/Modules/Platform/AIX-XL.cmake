# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__AIX_COMPILER_XL)
  return()
endif()
set(__AIX_COMPILER_XL 1)

macro(__aix_compiler_xl lang)
  set(CMAKE_SHARED_LIBRARY_RUNTIME_${lang}_FLAG "-Wl,-blibpath:")
  set(CMAKE_SHARED_LIBRARY_RUNTIME_${lang}_FLAG_SEP ":")
  string(APPEND CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS " -Wl,-bnoipath")
  set(CMAKE_SHARED_LIBRARY_LINK_${lang}_FLAGS "-Wl,-bexpall")
  set(CMAKE_SHARED_LIBRARY_${lang}_FLAGS " ")
  set(CMAKE_SHARED_MODULE_${lang}_FLAGS  " ")

  set(CMAKE_${lang}_LINK_FLAGS "-Wl,-bnoipath")

  set(_OBJECTS " <OBJECTS>")
  if(DEFINED CMAKE_XL_CreateExportList AND CMAKE_XL_CreateExportList STREQUAL "")
    # Prior to CMake 3.16, CMAKE_XL_CreateExportList held the path to the XL CreateExportList tool.
    # Users could set it to an empty value to skip automatic exports in favor of manual -bE: flags.
    # Preserve that behavior for compatibility (even though it was undocumented).
    set(_OBJECTS "")
  endif()

  # Construct the export list ourselves to pass only the object files so
  # that we export only the symbols actually provided by the sources.
  set(CMAKE_${lang}_CREATE_SHARED_LIBRARY
    "\"${CMAKE_ROOT}/Modules/Platform/AIX/ExportImportList\" -o <OBJECT_DIR>/exports.exp -c <CMAKE_${lang}_COMPILER> <AIX_EXPORTS>${_OBJECTS}"
    "<CMAKE_${lang}_COMPILER> <CMAKE_SHARED_LIBRARY_${lang}_FLAGS> -Wl,-bE:<OBJECT_DIR>/exports.exp <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>"
    )

  # Create an archive for shared library if CMAKE_AIX_SHARED_LIBRARY_ARCHIVE is used.
  string(REPLACE " <SONAME_FLAG><TARGET_SONAME> -o <TARGET>" " -o <TARGET_SONAME>"
    CMAKE_${lang}_CREATE_SHARED_LIBRARY_ARCHIVE "${CMAKE_${lang}_CREATE_SHARED_LIBRARY}")
  list(APPEND CMAKE_${lang}_CREATE_SHARED_LIBRARY_ARCHIVE
    "<CMAKE_AR> -X32_64 rc <TARGET> <TARGET_SONAME>"
    "rm -f <TARGET_SONAME>"
  )

  set(CMAKE_${lang}_LINK_EXECUTABLE_WITH_EXPORTS
    "\"${CMAKE_ROOT}/Modules/Platform/AIX/ExportImportList\" -o <TARGET_IMPLIB> -c <CMAKE_${lang}_COMPILER> -l . <AIX_EXPORTS> <OBJECTS>"
    "<CMAKE_${lang}_COMPILER> <FLAGS> -Wl,-bE:<TARGET_IMPLIB> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")

  unset(_OBJECTS)
endmacro()
