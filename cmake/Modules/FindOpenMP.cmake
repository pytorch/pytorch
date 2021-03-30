# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#.rst:
# FindOpenMP
# ----------
#
# Finds OpenMP support
#
# This module can be used to detect OpenMP support in a compiler.  If
# the compiler supports OpenMP, the flags required to compile with
# OpenMP support are returned in variables for the different languages.
# The variables may be empty if the compiler does not need a special
# flag to support OpenMP.
#
# Variables
# ^^^^^^^^^
#
# The module exposes the components ``C``, ``CXX``, and ``Fortran``.
# Each of these controls the various languages to search OpenMP support for.
#
# Depending on the enabled components the following variables will be set:
#
# ``OpenMP_FOUND``
#   Variable indicating that OpenMP flags for all requested languages have been found.
#   If no components are specified, this is true if OpenMP settings for all enabled languages
#   were detected.
# ``OpenMP_VERSION``
#   Minimal version of the OpenMP standard detected among the requested languages,
#   or all enabled languages if no components were specified.
#
# This module will set the following variables per language in your
# project, where ``<lang>`` is one of C, CXX, or Fortran:
#
# ``OpenMP_<lang>_FOUND``
#   Variable indicating if OpenMP support for ``<lang>`` was detected.
# ``OpenMP_<lang>_FLAGS``
#   OpenMP compiler flags for ``<lang>``, separated by spaces.
#
# For linking with OpenMP code written in ``<lang>``, the following
# variables are provided:
#
# ``OpenMP_<lang>_LIB_NAMES``
#   :ref:`;-list <CMake Language Lists>` of libraries for OpenMP programs for ``<lang>``.
# ``OpenMP_<libname>_LIBRARY``
#   Location of the individual libraries needed for OpenMP support in ``<lang>``.
# ``OpenMP_<lang>_LIBRARIES``
#   A list of libraries needed to link with OpenMP code written in ``<lang>``.
#
# Additionally, the module provides :prop_tgt:`IMPORTED` targets:
#
# ``OpenMP::OpenMP_<lang>``
#   Target for using OpenMP from ``<lang>``.
#
# Specifically for Fortran, the module sets the following variables:
#
# ``OpenMP_Fortran_HAVE_OMPLIB_HEADER``
#   Boolean indicating if OpenMP is accessible through ``omp_lib.h``.
# ``OpenMP_Fortran_HAVE_OMPLIB_MODULE``
#   Boolean indicating if OpenMP is accessible through the ``omp_lib`` Fortran module.
#
# The module will also try to provide the OpenMP version variables:
#
# ``OpenMP_<lang>_SPEC_DATE``
#   Date of the OpenMP specification implemented by the ``<lang>`` compiler.
# ``OpenMP_<lang>_VERSION_MAJOR``
#   Major version of OpenMP implemented by the ``<lang>`` compiler.
# ``OpenMP_<lang>_VERSION_MINOR``
#   Minor version of OpenMP implemented by the ``<lang>`` compiler.
# ``OpenMP_<lang>_VERSION``
#   OpenMP version implemented by the ``<lang>`` compiler.
#
# The specification date is formatted as given in the OpenMP standard:
# ``yyyymm`` where ``yyyy`` and ``mm`` represents the year and month of
# the OpenMP specification implemented by the ``<lang>`` compiler.

cmake_policy(PUSH)
cmake_policy(SET CMP0012 NEW) # if() recognizes numbers and booleans
cmake_policy(SET CMP0054 NEW) # if() quoted variables not dereferenced
cmake_policy(SET CMP0057 NEW) # if IN_LIST

function(_OPENMP_FLAG_CANDIDATES LANG)
  if(NOT OpenMP_${LANG}_FLAG)
    unset(OpenMP_FLAG_CANDIDATES)

    set(OMP_FLAG_GNU "-fopenmp")
    set(OMP_FLAG_Clang "-fopenmp=libomp" "-fopenmp=libiomp5" "-fopenmp")

    if(WIN32)
      # Prefer Intel OpenMP header which can be provided by CMAKE_INCLUDE_PATH.
      # Note that CMAKE_INCLUDE_PATH is searched before CMAKE_SYSTEM_INCLUDE_PATH (MSVC path in this case)
      find_path(__header_dir "omp.h")
    else()
      # AppleClang may need a header file, search for omp.h with hints to brew
      # default include dir
      find_path(__header_dir "omp.h" HINTS "/usr/local/include")
    endif()
    set(OMP_FLAG_AppleClang "-Xpreprocessor -fopenmp" "-Xpreprocessor -fopenmp -I${__header_dir}")

    set(OMP_FLAG_HP "+Oopenmp")
    if(WIN32)
      set(OMP_FLAG_Intel "-Qopenmp")
    elseif(CMAKE_${LANG}_COMPILER_ID STREQUAL "Intel" AND
           "${CMAKE_${LANG}_COMPILER_VERSION}" VERSION_LESS "15.0.0.20140528")
      set(OMP_FLAG_Intel "-openmp")
    else()
      set(OMP_FLAG_Intel "-qopenmp")
    endif()
    set(OMP_FLAG_MIPSpro "-mp")
    if(__header_dir MATCHES ".*Microsoft Visual Studio.*")
      # MSVC header. No need to pass it as additional include.
      set(OMP_FLAG_MSVC "-openmp:experimental" "-openmp")
    else()
      set(OMP_FLAG_MSVC "-openmp:experimental -I${__header_dir}" "-openmp -I${__header_dir}")
    endif()
    set(OMP_FLAG_PathScale "-openmp")
    set(OMP_FLAG_NAG "-openmp")
    set(OMP_FLAG_Absoft "-openmp")
    set(OMP_FLAG_PGI "-mp")
    set(OMP_FLAG_Flang "-fopenmp")
    set(OMP_FLAG_SunPro "-xopenmp")
    set(OMP_FLAG_XL "-qsmp=omp")
    # Cray compiler activate OpenMP with -h omp, which is enabled by default.
    set(OMP_FLAG_Cray " " "-h omp")

    # If we know the correct flags, use those
    if(DEFINED OMP_FLAG_${CMAKE_${LANG}_COMPILER_ID})
      set(OpenMP_FLAG_CANDIDATES "${OMP_FLAG_${CMAKE_${LANG}_COMPILER_ID}}")
    # Fall back to reasonable default tries otherwise
    else()
      set(OpenMP_FLAG_CANDIDATES "-openmp" "-fopenmp" "-mp" " ")
    endif()
    set(OpenMP_${LANG}_FLAG_CANDIDATES "${OpenMP_FLAG_CANDIDATES}" PARENT_SCOPE)
  else()
    set(OpenMP_${LANG}_FLAG_CANDIDATES "${OpenMP_${LANG}_FLAG}" PARENT_SCOPE)
  endif()
endfunction()

# sample openmp source code to test
set(OpenMP_C_CXX_TEST_SOURCE
"
#include <omp.h>
int main(void) {
#ifdef _OPENMP
  omp_get_max_threads();
  return 0;
#else
  breaks_on_purpose
#endif
}
")

# in Fortran, an implementation may provide an omp_lib.h header
# or omp_lib module, or both (OpenMP standard, section 3.1)
# Furthmore !$ is the Fortran equivalent of #ifdef _OPENMP (OpenMP standard, 2.2.2)
# Without the conditional compilation, some compilers (e.g. PGI) might compile OpenMP code
# while not actually enabling OpenMP, building code sequentially
set(OpenMP_Fortran_TEST_SOURCE
  "
      program test
      @OpenMP_Fortran_INCLUDE_LINE@
  !$  integer :: n
      n = omp_get_num_threads()
      end program test
  "
)

function(_OPENMP_WRITE_SOURCE_FILE LANG SRC_FILE_CONTENT_VAR SRC_FILE_NAME SRC_FILE_FULLPATH)
  set(WORK_DIR ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/FindOpenMP)
  if("${LANG}" STREQUAL "C")
    set(SRC_FILE "${WORK_DIR}/${SRC_FILE_NAME}.c")
    file(WRITE "${SRC_FILE}" "${OpenMP_C_CXX_${SRC_FILE_CONTENT_VAR}}")
  elseif("${LANG}" STREQUAL "CXX")
    set(SRC_FILE "${WORK_DIR}/${SRC_FILE_NAME}.cpp")
    file(WRITE "${SRC_FILE}" "${OpenMP_C_CXX_${SRC_FILE_CONTENT_VAR}}")
  elseif("${LANG}" STREQUAL "Fortran")
    set(SRC_FILE "${WORK_DIR}/${SRC_FILE_NAME}.f90")
    file(WRITE "${SRC_FILE}_in" "${OpenMP_Fortran_${SRC_FILE_CONTENT_VAR}}")
    configure_file("${SRC_FILE}_in" "${SRC_FILE}" @ONLY)
  endif()
  set(${SRC_FILE_FULLPATH} "${SRC_FILE}" PARENT_SCOPE)
endfunction()

include(CMakeParseImplicitLinkInfo)

function(_OPENMP_GET_FLAGS LANG FLAG_MODE OPENMP_FLAG_VAR OPENMP_LIB_NAMES_VAR)
  _OPENMP_FLAG_CANDIDATES("${LANG}")
  _OPENMP_WRITE_SOURCE_FILE("${LANG}" "TEST_SOURCE" OpenMPTryFlag _OPENMP_TEST_SRC)

  unset(OpenMP_VERBOSE_COMPILE_OPTIONS)
  if(UNIX)
    separate_arguments(OpenMP_VERBOSE_OPTIONS UNIX_COMMAND "${CMAKE_${LANG}_VERBOSE_FLAG}")
  else()
    separate_arguments(OpenMP_VERBOSE_OPTIONS WINDOWS_COMMAND "${CMAKE_${LANG}_VERBOSE_FLAG}")
  endif()
  foreach(_VERBOSE_OPTION IN LISTS OpenMP_VERBOSE_OPTIONS)
    if(NOT _VERBOSE_OPTION MATCHES "^-Wl,")
      list(APPEND OpenMP_VERBOSE_COMPILE_OPTIONS ${_VERBOSE_OPTION})
    endif()
  endforeach()

  foreach(OPENMP_FLAG IN LISTS OpenMP_${LANG}_FLAG_CANDIDATES)
    set(OPENMP_FLAGS_TEST "${OPENMP_FLAG}")
    if(OpenMP_VERBOSE_COMPILE_OPTIONS)
      string(APPEND OPENMP_FLAGS_TEST " ${OpenMP_VERBOSE_COMPILE_OPTIONS}")
    endif()
    string(REGEX REPLACE "[-/=+]" "" OPENMP_PLAIN_FLAG "${OPENMP_FLAG}")

    # NOTE [ Linking both MKL and OpenMP ]
    #
    # It is crucial not to link two `libomp` libraries together, even when they
    # are both Intel or GNU. Otherwise, you will end up with this nasty error,
    # and may get incorrect results.
    #
    #   OMP: Error #15: Initializing libomp.dylib, but found libiomp5.dylib
    #   already initialized.
    #
    #   OMP: Hint This means that multiple copies of the OpenMP runtime have
    #   been linked into the program. That is dangerous, since it can degrade
    #   performance or cause incorrect results. The best thing to do is to
    #   ensure that only a single OpenMP runtime is linked into the process,
    #   e.g. by avoiding static linking of the OpenMP runtime in any library. As
    #   an unsafe, unsupported, undocumented workaround you can set the
    #   environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to
    #   continue to execute, but that may cause crashes or silently produce
    #   incorrect results. For more information, please see
    #   http://openmp.llvm.org/
    #
    # So here, before we test each flag combination, we first try directly
    # linking against any `libomp` MKL has found (if any). This allows us to
    # do sensible things in tricky (yet common) conditions like:
    #   - using `clang` (so no native GNU OpenMP), and
    #   - having `brew` `libomp` installed at `/usr/local/`, and
    #   - having `conda` `mkl` installed at `$HOME/conda/`, with includes a copy
    #     of `libiomp5`.
    # Rather than blindly picking one, we pick what ever `FindMKL.cmake` choses
    # to avoid conflicts.
    #
    # Crucially, we only do so for non-GNU compilers. For GNU ones,
    # `FindMKL.cmake` calls `FindOpenMP.cmake` when trying to find `gomp` and
    # thus will cause infinite recursion if this is not taken care of. Moreover,
    # for them, since the compiler provices the OpenMP library, it is most
    # likely that only one viable gomp library can be found in search path by
    # `FindOpenMP.cmake`, so the chance of having conflicts is slow.
    #
    # TODO: refactor to solve this weird dependency where
    #         - for non-GNU, FindOpenMP.cmake replies on FindMKL.cmake to finish first, but
    #         - for GNU,     FindMKL.cmake replies on FindOpenMP.cmake to finish first.

    if(NOT "${CMAKE_${LANG}_COMPILER_ID}" STREQUAL "GNU")
      find_package(MKL QUIET)
      if(MKL_FOUND AND (NOT "${MKL_OPENMP_LIBRARY}" STREQUAL ""))
        # If we already link OpenMP via MKL, use that. Otherwise at run-time
        # OpenMP will complain about being initialized twice (OMP: Error #15),
        # can may cause incorrect behavior.
        set(OpenMP_libomp_LIBRARY "${MKL_OPENMP_LIBRARY}" CACHE STRING "libomp location for OpenMP")
      else()
        find_library(OpenMP_libomp_LIBRARY
          NAMES omp gomp iomp5
          HINTS ${CMAKE_${LANG}_IMPLICIT_LINK_DIRECTORIES}
          DOC "libomp location for OpenMP"
        )
      endif()
      mark_as_advanced(OpenMP_libomp_LIBRARY)

      if (OpenMP_libomp_LIBRARY)
        try_compile( OpenMP_COMPILE_RESULT_${FLAG_MODE}_${OPENMP_PLAIN_FLAG} ${CMAKE_BINARY_DIR} ${_OPENMP_TEST_SRC}
          CMAKE_FLAGS "-DCOMPILE_DEFINITIONS:STRING=${OPENMP_FLAGS_TEST}"
          LINK_LIBRARIES ${CMAKE_${LANG}_VERBOSE_FLAG} ${OpenMP_libomp_LIBRARY}
          OUTPUT_VARIABLE OpenMP_TRY_COMPILE_OUTPUT
        )
        if(OpenMP_COMPILE_RESULT_${FLAG_MODE}_${OPENMP_PLAIN_FLAG})
          set("${OPENMP_FLAG_VAR}" "${OPENMP_FLAG}" PARENT_SCOPE)
          set("${OPENMP_LIB_NAMES_VAR}" "libomp" PARENT_SCOPE)
          break()
        endif()
      endif()
    endif()

    try_compile( OpenMP_COMPILE_RESULT_${FLAG_MODE}_${OPENMP_PLAIN_FLAG} ${CMAKE_BINARY_DIR} ${_OPENMP_TEST_SRC}
      CMAKE_FLAGS "-DCOMPILE_DEFINITIONS:STRING=${OPENMP_FLAGS_TEST}"
      LINK_LIBRARIES ${CMAKE_${LANG}_VERBOSE_FLAG}
      OUTPUT_VARIABLE OpenMP_TRY_COMPILE_OUTPUT
    )

    if(OpenMP_COMPILE_RESULT_${FLAG_MODE}_${OPENMP_PLAIN_FLAG})
      set("${OPENMP_FLAG_VAR}" "${OPENMP_FLAG}" PARENT_SCOPE)

      if(CMAKE_${LANG}_VERBOSE_FLAG)
        unset(OpenMP_${LANG}_IMPLICIT_LIBRARIES)
        unset(OpenMP_${LANG}_IMPLICIT_LINK_DIRS)
        unset(OpenMP_${LANG}_IMPLICIT_FWK_DIRS)
        unset(OpenMP_${LANG}_LOG_VAR)

        file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
        "Detecting ${LANG} OpenMP compiler ABI info compiled with the following output:\n${OpenMP_TRY_COMPILE_OUTPUT}\n\n")

        cmake_parse_implicit_link_info("${OpenMP_TRY_COMPILE_OUTPUT}"
          OpenMP_${LANG}_IMPLICIT_LIBRARIES
          OpenMP_${LANG}_IMPLICIT_LINK_DIRS
          OpenMP_${LANG}_IMPLICIT_FWK_DIRS
          OpenMP_${LANG}_LOG_VAR
          "${CMAKE_${LANG}_IMPLICIT_OBJECT_REGEX}"
        )

        file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
        "Parsed ${LANG} OpenMP implicit link information from above output:\n${OpenMP_${LANG}_LOG_VAR}\n\n")

        unset(_OPENMP_LIB_NAMES)
        foreach(_OPENMP_IMPLICIT_LIB IN LISTS OpenMP_${LANG}_IMPLICIT_LIBRARIES)
          get_filename_component(_OPENMP_IMPLICIT_LIB_DIR "${_OPENMP_IMPLICIT_LIB}" DIRECTORY)
          get_filename_component(_OPENMP_IMPLICIT_LIB_NAME "${_OPENMP_IMPLICIT_LIB}" NAME)
          get_filename_component(_OPENMP_IMPLICIT_LIB_PLAIN "${_OPENMP_IMPLICIT_LIB}" NAME_WE)
          string(REGEX REPLACE "([][+.*?()^$])" "\\\\\\1" _OPENMP_IMPLICIT_LIB_PLAIN_ESC "${_OPENMP_IMPLICIT_LIB_PLAIN}")
          string(REGEX REPLACE "([][+.*?()^$])" "\\\\\\1" _OPENMP_IMPLICIT_LIB_PATH_ESC "${_OPENMP_IMPLICIT_LIB}")
          if(NOT ( "${_OPENMP_IMPLICIT_LIB}" IN_LIST CMAKE_${LANG}_IMPLICIT_LINK_LIBRARIES
            OR "${CMAKE_${LANG}_STANDARD_LIBRARIES}" MATCHES "(^| )(-Wl,)?(-l)?(${_OPENMP_IMPLICIT_LIB_PLAIN_ESC}|${_OPENMP_IMPLICIT_LIB_PATH_ESC})( |$)"
            OR "${CMAKE_${LANG}_LINK_EXECUTABLE}" MATCHES "(^| )(-Wl,)?(-l)?(${_OPENMP_IMPLICIT_LIB_PLAIN_ESC}|${_OPENMP_IMPLICIT_LIB_PATH_ESC})( |$)" ) )
            if(_OPENMP_IMPLICIT_LIB_DIR)
              set(OpenMP_${_OPENMP_IMPLICIT_LIB_PLAIN}_LIBRARY "${_OPENMP_IMPLICIT_LIB}" CACHE FILEPATH
                "Path to the ${_OPENMP_IMPLICIT_LIB_PLAIN} library for OpenMP")
            else()
              find_library(OpenMP_${_OPENMP_IMPLICIT_LIB_PLAIN}_LIBRARY
                NAMES "${_OPENMP_IMPLICIT_LIB_NAME}"
                DOC "Path to the ${_OPENMP_IMPLICIT_LIB_PLAIN} library for OpenMP"
                HINTS ${OpenMP_${LANG}_IMPLICIT_LINK_DIRS}
                CMAKE_FIND_ROOT_PATH_BOTH
                NO_DEFAULT_PATH
              )
            endif()
            mark_as_advanced(OpenMP_${_OPENMP_IMPLICIT_LIB_PLAIN}_LIBRARY)
            list(APPEND _OPENMP_LIB_NAMES ${_OPENMP_IMPLICIT_LIB_PLAIN})
          endif()
        endforeach()
        set("${OPENMP_LIB_NAMES_VAR}" "${_OPENMP_LIB_NAMES}" PARENT_SCOPE)
      else()
        # We do not know how to extract implicit OpenMP libraries for this compiler.
        # Assume that it handles them automatically, e.g. the Intel Compiler on
        # Windows should put the dependency in its object files.
        set("${OPENMP_LIB_NAMES_VAR}" "" PARENT_SCOPE)
      endif()
      break()
    elseif((CMAKE_${LANG}_COMPILER_ID STREQUAL "AppleClang") AND
           (NOT CMAKE_${LANG}_COMPILER_VERSION VERSION_LESS "7.0"))

      # LLVM 3.7 supports OpenMP 3.1, and continues to add more features to
      # support newer OpenMP standards in new versions.
      # http://releases.llvm.org/3.7.0/tools/clang/docs/ReleaseNotes.html#openmp-support
      #
      # Apple Clang 7.0 is the first version based on LLVM 3.7 or later.
      # https://en.wikipedia.org/wiki/Xcode#Latest_versions
      #
      # Check for separate OpenMP library on AppleClang 7+
      find_library(OpenMP_libomp_LIBRARY
        NAMES omp gomp iomp5
        HINTS ${CMAKE_${LANG}_IMPLICIT_LINK_DIRECTORIES}
        DOC "libomp location for OpenMP"
      )
      mark_as_advanced(OpenMP_libomp_LIBRARY)

      if(OpenMP_libomp_LIBRARY)
        try_compile( OpenMP_COMPILE_RESULT_${FLAG_MODE}_${OPENMP_PLAIN_FLAG} ${CMAKE_BINARY_DIR} ${_OPENMP_TEST_SRC}
          CMAKE_FLAGS "-DCOMPILE_DEFINITIONS:STRING=${OPENMP_FLAGS_TEST}"
          LINK_LIBRARIES ${CMAKE_${LANG}_VERBOSE_FLAG} ${OpenMP_libomp_LIBRARY}
          OUTPUT_VARIABLE OpenMP_TRY_COMPILE_OUTPUT
        )
        if(OpenMP_COMPILE_RESULT_${FLAG_MODE}_${OPENMP_PLAIN_FLAG})
          set("${OPENMP_FLAG_VAR}" "${OPENMP_FLAG}" PARENT_SCOPE)
          set("${OPENMP_LIB_NAMES_VAR}" "libomp" PARENT_SCOPE)
          break()
        endif()
      endif()
    else()
      file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
        "Detecting ${LANG} OpenMP failed with the following output:\n${OpenMP_TRY_COMPILE_OUTPUT}\n\n")
    endif()
    if (NOT ${OpenMP_${LANG}_FIND_QUIETLY})
      message(STATUS "OpenMP try_compile log:\n${OpenMP_TRY_COMPILE_OUTPUT}\n\n")
    endif()
    set("${OPENMP_LIB_NAMES_VAR}" "NOTFOUND" PARENT_SCOPE)
    set("${OPENMP_FLAG_VAR}" "NOTFOUND" PARENT_SCOPE)
  endforeach()

  unset(OpenMP_VERBOSE_COMPILE_OPTIONS)
endfunction()

set(OpenMP_C_CXX_CHECK_VERSION_SOURCE
"
#include <stdio.h>
#include <omp.h>
const char ompver_str[] = { 'I', 'N', 'F', 'O', ':', 'O', 'p', 'e', 'n', 'M',
                            'P', '-', 'd', 'a', 't', 'e', '[',
                            ('0' + ((_OPENMP/100000)%10)),
                            ('0' + ((_OPENMP/10000)%10)),
                            ('0' + ((_OPENMP/1000)%10)),
                            ('0' + ((_OPENMP/100)%10)),
                            ('0' + ((_OPENMP/10)%10)),
                            ('0' + ((_OPENMP/1)%10)),
                            ']', '\\0' };
int main(void)
{
  puts(ompver_str);
  return 0;
}
")

set(OpenMP_Fortran_CHECK_VERSION_SOURCE
"
      program omp_ver
      @OpenMP_Fortran_INCLUDE_LINE@
      integer, parameter :: zero = ichar('0')
      integer, parameter :: ompv = openmp_version
      character, dimension(24), parameter :: ompver_str =&
      (/ 'I', 'N', 'F', 'O', ':', 'O', 'p', 'e', 'n', 'M', 'P', '-',&
         'd', 'a', 't', 'e', '[',&
         char(zero + mod(ompv/100000, 10)),&
         char(zero + mod(ompv/10000, 10)),&
         char(zero + mod(ompv/1000, 10)),&
         char(zero + mod(ompv/100, 10)),&
         char(zero + mod(ompv/10, 10)),&
         char(zero + mod(ompv/1, 10)), ']' /)
      print *, ompver_str
      end program omp_ver
")

function(_OPENMP_GET_SPEC_DATE LANG SPEC_DATE)
  _OPENMP_WRITE_SOURCE_FILE("${LANG}" "CHECK_VERSION_SOURCE" OpenMPCheckVersion _OPENMP_TEST_SRC)

  set(BIN_FILE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/FindOpenMP/ompver_${LANG}.bin")
  string(REGEX REPLACE "[-/=+]" "" OPENMP_PLAIN_FLAG "${OPENMP_FLAG}")
  try_compile(OpenMP_SPECTEST_${LANG}_${OPENMP_PLAIN_FLAG} "${CMAKE_BINARY_DIR}" "${_OPENMP_TEST_SRC}"
              CMAKE_FLAGS "-DCOMPILE_DEFINITIONS:STRING=${OpenMP_${LANG}_FLAGS}"
              COPY_FILE ${BIN_FILE}
              OUTPUT_VARIABLE OpenMP_TRY_COMPILE_OUTPUT)

  if(${OpenMP_SPECTEST_${LANG}_${OPENMP_PLAIN_FLAG}})
    file(STRINGS ${BIN_FILE} specstr LIMIT_COUNT 1 REGEX "INFO:OpenMP-date")
    set(regex_spec_date ".*INFO:OpenMP-date\\[0*([^]]*)\\].*")
    if("${specstr}" MATCHES "${regex_spec_date}")
      set(${SPEC_DATE} "${CMAKE_MATCH_1}" PARENT_SCOPE)
    endif()
  else()
    file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
        "Detecting ${LANG} OpenMP version failed with the following output:\n${OpenMP_TRY_COMPILE_OUTPUT}\n\n")
  endif()
endfunction()

macro(_OPENMP_SET_VERSION_BY_SPEC_DATE LANG)
  set(OpenMP_SPEC_DATE_MAP
    # Preview versions
    "201611=5.0" # OpenMP 5.0 preview 1
    # Combined versions, 2.5 onwards
    "201511=4.5"
    "201307=4.0"
    "201107=3.1"
    "200805=3.0"
    "200505=2.5"
    # C/C++ version 2.0
    "200203=2.0"
    # Fortran version 2.0
    "200011=2.0"
    # Fortran version 1.1
    "199911=1.1"
    # C/C++ version 1.0 (there's no 1.1 for C/C++)
    "199810=1.0"
    # Fortran version 1.0
    "199710=1.0"
  )

  if(OpenMP_${LANG}_SPEC_DATE)
    string(REGEX MATCHALL "${OpenMP_${LANG}_SPEC_DATE}=([0-9]+)\\.([0-9]+)" _version_match "${OpenMP_SPEC_DATE_MAP}")
  else()
    set(_version_match "")
  endif()
  if(NOT _version_match STREQUAL "")
    set(OpenMP_${LANG}_VERSION_MAJOR ${CMAKE_MATCH_1})
    set(OpenMP_${LANG}_VERSION_MINOR ${CMAKE_MATCH_2})
    set(OpenMP_${LANG}_VERSION "${OpenMP_${LANG}_VERSION_MAJOR}.${OpenMP_${LANG}_VERSION_MINOR}")
  else()
    unset(OpenMP_${LANG}_VERSION_MAJOR)
    unset(OpenMP_${LANG}_VERSION_MINOR)
    unset(OpenMP_${LANG}_VERSION)
  endif()
  unset(_version_match)
  unset(OpenMP_SPEC_DATE_MAP)
endmacro()

foreach(LANG IN ITEMS C CXX)
  if(CMAKE_${LANG}_COMPILER_LOADED)
    if(NOT DEFINED OpenMP_${LANG}_FLAGS OR "${OpenMP_${LANG}_FLAGS}" STREQUAL "NOTFOUND"
      OR NOT DEFINED OpenMP_${LANG}_LIB_NAMES OR "${OpenMP_${LANG}_LIB_NAMES}" STREQUAL "NOTFOUND")
      _OPENMP_GET_FLAGS("${LANG}" "${LANG}" OpenMP_${LANG}_FLAGS_WORK OpenMP_${LANG}_LIB_NAMES_WORK)
    endif()

    set(OpenMP_${LANG}_FLAGS "${OpenMP_${LANG}_FLAGS_WORK}"
      CACHE STRING "${LANG} compiler flags for OpenMP parallelization")
    set(OpenMP_${LANG}_LIB_NAMES "${OpenMP_${LANG}_LIB_NAMES_WORK}"
      CACHE STRING "${LANG} compiler libraries for OpenMP parallelization")
    mark_as_advanced(OpenMP_${LANG}_FLAGS OpenMP_${LANG}_LIB_NAMES)
  endif()
endforeach()

if(CMAKE_Fortran_COMPILER_LOADED)
  if(NOT DEFINED OpenMP_Fortran_FLAGS OR "${OpenMP_Fortran_FLAGS}" STREQUAL "NOTFOUND"
    OR NOT DEFINED OpenMP_Fortran_LIB_NAMES OR "${OpenMP_Fortran_LIB_NAMES}" STREQUAL "NOTFOUND"
    OR NOT DEFINED OpenMP_Fortran_HAVE_OMPLIB_MODULE)
    set(OpenMP_Fortran_INCLUDE_LINE "use omp_lib\n      implicit none")
    _OPENMP_GET_FLAGS("Fortran" "FortranHeader" OpenMP_Fortran_FLAGS_WORK OpenMP_Fortran_LIB_NAMES_WORK)
    if(OpenMP_Fortran_FLAGS_WORK)
      set(OpenMP_Fortran_HAVE_OMPLIB_MODULE TRUE CACHE BOOL INTERNAL "")
    endif()

    set(OpenMP_Fortran_FLAGS "${OpenMP_Fortran_FLAGS_WORK}"
      CACHE STRING "Fortran compiler flags for OpenMP parallelization")
    set(OpenMP_Fortran_LIB_NAMES "${OpenMP_Fortran_LIB_NAMES_WORK}"
      CACHE STRING "Fortran compiler libraries for OpenMP parallelization")
    mark_as_advanced(OpenMP_Fortran_FLAGS OpenMP_Fortran_LIB_NAMES)
  endif()

  if(NOT DEFINED OpenMP_Fortran_FLAGS OR "${OpenMP_Fortran_FLAGS}" STREQUAL "NOTFOUND"
    OR NOT DEFINED OpenMP_Fortran_LIB_NAMES OR "${OpenMP_Fortran_LIB_NAMES}" STREQUAL "NOTFOUND"
    OR NOT DEFINED OpenMP_Fortran_HAVE_OMPLIB_HEADER)
    set(OpenMP_Fortran_INCLUDE_LINE "implicit none\n      include 'omp_lib.h'")
    _OPENMP_GET_FLAGS("Fortran" "FortranModule" OpenMP_Fortran_FLAGS_WORK OpenMP_Fortran_LIB_NAMES_WORK)
    if(OpenMP_Fortran_FLAGS_WORK)
      set(OpenMP_Fortran_HAVE_OMPLIB_HEADER TRUE CACHE BOOL INTERNAL "")
    endif()

    set(OpenMP_Fortran_FLAGS "${OpenMP_Fortran_FLAGS_WORK}"
      CACHE STRING "Fortran compiler flags for OpenMP parallelization")

    set(OpenMP_Fortran_LIB_NAMES "${OpenMP_Fortran_LIB_NAMES}"
      CACHE STRING "Fortran compiler libraries for OpenMP parallelization")
  endif()

  if(OpenMP_Fortran_HAVE_OMPLIB_MODULE)
    set(OpenMP_Fortran_INCLUDE_LINE "use omp_lib\n      implicit none")
  else()
    set(OpenMP_Fortran_INCLUDE_LINE "implicit none\n      include 'omp_lib.h'")
  endif()
endif()

if(NOT OpenMP_FIND_COMPONENTS)
  set(OpenMP_FINDLIST C CXX Fortran)
else()
  set(OpenMP_FINDLIST ${OpenMP_FIND_COMPONENTS})
endif()

unset(_OpenMP_MIN_VERSION)

include(FindPackageHandleStandardArgs)

foreach(LANG IN LISTS OpenMP_FINDLIST)
  if(CMAKE_${LANG}_COMPILER_LOADED)
    if (NOT OpenMP_${LANG}_SPEC_DATE AND OpenMP_${LANG}_FLAGS)
      _OPENMP_GET_SPEC_DATE("${LANG}" OpenMP_${LANG}_SPEC_DATE_INTERNAL)
      set(OpenMP_${LANG}_SPEC_DATE "${OpenMP_${LANG}_SPEC_DATE_INTERNAL}" CACHE
        INTERNAL "${LANG} compiler's OpenMP specification date")
      _OPENMP_SET_VERSION_BY_SPEC_DATE("${LANG}")
    endif()

    set(OpenMP_${LANG}_FIND_QUIETLY ${OpenMP_FIND_QUIETLY})
    set(OpenMP_${LANG}_FIND_REQUIRED ${OpenMP_FIND_REQUIRED})
    set(OpenMP_${LANG}_FIND_VERSION ${OpenMP_FIND_VERSION})
    set(OpenMP_${LANG}_FIND_VERSION_EXACT ${OpenMP_FIND_VERSION_EXACT})

    set(_OPENMP_${LANG}_REQUIRED_VARS OpenMP_${LANG}_FLAGS)
    if("${OpenMP_${LANG}_LIB_NAMES}" STREQUAL "NOTFOUND")
      set(_OPENMP_${LANG}_REQUIRED_LIB_VARS OpenMP_${LANG}_LIB_NAMES)
    else()
      foreach(_OPENMP_IMPLICIT_LIB IN LISTS OpenMP_${LANG}_LIB_NAMES)
        list(APPEND _OPENMP_${LANG}_REQUIRED_LIB_VARS OpenMP_${_OPENMP_IMPLICIT_LIB}_LIBRARY)
      endforeach()
    endif()

    find_package_handle_standard_args(OpenMP_${LANG}
      REQUIRED_VARS OpenMP_${LANG}_FLAGS ${_OPENMP_${LANG}_REQUIRED_LIB_VARS}
      VERSION_VAR OpenMP_${LANG}_VERSION
    )

    if(OpenMP_${LANG}_FOUND)
      if(DEFINED OpenMP_${LANG}_VERSION)
        if(NOT _OpenMP_MIN_VERSION OR _OpenMP_MIN_VERSION VERSION_GREATER OpenMP_${LANG}_VERSION)
          set(_OpenMP_MIN_VERSION OpenMP_${LANG}_VERSION)
        endif()
      endif()
      set(OpenMP_${LANG}_LIBRARIES "")
      foreach(_OPENMP_IMPLICIT_LIB IN LISTS OpenMP_${LANG}_LIB_NAMES)
        list(APPEND OpenMP_${LANG}_LIBRARIES "${OpenMP_${_OPENMP_IMPLICIT_LIB}_LIBRARY}")
      endforeach()

      if(NOT TARGET OpenMP::OpenMP_${LANG})
        add_library(OpenMP::OpenMP_${LANG} INTERFACE IMPORTED)
      endif()
      if(OpenMP_${LANG}_FLAGS)
        if(UNIX)
          separate_arguments(_OpenMP_${LANG}_OPTIONS UNIX_COMMAND "${OpenMP_${LANG}_FLAGS}")
        else()
          separate_arguments(_OpenMP_${LANG}_OPTIONS WINDOWS_COMMAND "${OpenMP_${LANG}_FLAGS}")
        endif()
        set_property(TARGET OpenMP::OpenMP_${LANG} PROPERTY
          INTERFACE_COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:${LANG}>:${_OpenMP_${LANG}_OPTIONS}>")
        unset(_OpenMP_${LANG}_OPTIONS)
      endif()
      if(OpenMP_${LANG}_LIBRARIES)
        set_property(TARGET OpenMP::OpenMP_${LANG} PROPERTY
          INTERFACE_LINK_LIBRARIES "${OpenMP_${LANG}_LIBRARIES}")
      endif()
    endif()
  endif()
endforeach()

unset(_OpenMP_REQ_VARS)
foreach(LANG IN ITEMS C CXX Fortran)
  if((NOT OpenMP_FIND_COMPONENTS AND CMAKE_${LANG}_COMPILER_LOADED) OR LANG IN_LIST OpenMP_FIND_COMPONENTS)
    list(APPEND _OpenMP_REQ_VARS "OpenMP_${LANG}_FOUND")
  endif()
endforeach()

find_package_handle_standard_args(OpenMP
    REQUIRED_VARS ${_OpenMP_REQ_VARS}
    VERSION_VAR ${_OpenMP_MIN_VERSION}
    HANDLE_COMPONENTS)

set(OPENMP_FOUND ${OpenMP_FOUND})

if(CMAKE_Fortran_COMPILER_LOADED AND OpenMP_Fortran_FOUND)
  if(NOT DEFINED OpenMP_Fortran_HAVE_OMPLIB_MODULE)
    set(OpenMP_Fortran_HAVE_OMPLIB_MODULE FALSE CACHE BOOL INTERNAL "")
  endif()
  if(NOT DEFINED OpenMP_Fortran_HAVE_OMPLIB_HEADER)
    set(OpenMP_Fortran_HAVE_OMPLIB_HEADER FALSE CACHE BOOL INTERNAL "")
  endif()
endif()

if(NOT ( CMAKE_C_COMPILER_LOADED OR CMAKE_CXX_COMPILER_LOADED OR CMAKE_Fortran_COMPILER_LOADED ))
  message(SEND_ERROR "FindOpenMP requires the C, CXX or Fortran languages to be enabled")
endif()

unset(OpenMP_C_CXX_TEST_SOURCE)
unset(OpenMP_Fortran_TEST_SOURCE)
unset(OpenMP_C_CXX_CHECK_VERSION_SOURCE)
unset(OpenMP_Fortran_CHECK_VERSION_SOURCE)
unset(OpenMP_Fortran_INCLUDE_LINE)

cmake_policy(POP)
