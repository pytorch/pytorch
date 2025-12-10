# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__AIX_COMPILER_GNU)
  return()
endif()
set(__AIX_COMPILER_GNU 1)

macro(__aix_compiler_gnu lang)
  set(CMAKE_SHARED_LIBRARY_RUNTIME_${lang}_FLAG "-Wl,-blibpath:")
  set(CMAKE_SHARED_LIBRARY_RUNTIME_${lang}_FLAG_SEP ":")
  string(APPEND CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS " -Wl,-bnoipath")
  set(CMAKE_SHARED_LIBRARY_LINK_${lang}_FLAGS "-Wl,-bexpall")
  set(CMAKE_${lang}_USE_IMPLICIT_LINK_DIRECTORIES_IN_RUNTIME_PATH 1)
  set(CMAKE_${lang}_VERBOSE_LINK_FLAG "-Wl,-v")

  set(CMAKE_${lang}_LINK_FLAGS "-Wl,-bnoipath")

  if(CMAKE_${lang}_COMPILER_VERSION VERSION_LESS 7 OR CMAKE_SYSTEM_VERSION VERSION_LESS 7.1)
    unset(CMAKE_${lang}_COMPILE_OPTIONS_VISIBILITY)
  endif()

  # Construct the export list ourselves to pass only the object files so
  # that we export only the symbols actually provided by the sources.
  set(CMAKE_${lang}_CREATE_SHARED_LIBRARY
    "\"${CMAKE_ROOT}/Modules/Platform/AIX/ExportImportList\" -o <OBJECT_DIR>/exports.exp -c <CMAKE_${lang}_COMPILER> <AIX_EXPORTS> <OBJECTS>"
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
endmacro()
