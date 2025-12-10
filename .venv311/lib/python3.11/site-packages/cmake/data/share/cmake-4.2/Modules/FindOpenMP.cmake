# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindOpenMP
----------

Finds Open Multi-Processing (OpenMP) support in a compiler:

.. code-block:: cmake

  find_package(OpenMP [<version>] [COMPONENTS <langs>...] [...])

If the compiler supports OpenMP, the flags required to compile with OpenMP
support are returned in variables for the different languages.  The variables
may be empty if the compiler does not need a special flag to support OpenMP.

.. versionadded:: 3.5
  Clang support.

Components
^^^^^^^^^^

This module supports components that can be specified using the standard
syntax:

.. code-block:: cmake

  find_package(OpenMP [COMPONENTS <langs>...])

Each of these components controls the various languages to search OpenMP
support for.  The following components are exposed:

``C``
  .. versionadded:: 3.10

``CXX``
  .. versionadded:: 3.10

``Fortran``
  .. versionadded:: 3.10

``CUDA``
  .. versionadded:: 3.31

  The ``CUDA`` language component is supported when using a CUDA compiler
  that supports OpenMP on the host.

If no components are specified, module checks for all of them automatically,
depending on the enabled languages in the project.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``OpenMP::OpenMP_<lang>``
  .. versionadded:: 3.9

  Target encapsulating the OpenMP usage requirements for language ``<lang>``.

Input Variables
^^^^^^^^^^^^^^^

The following variables may be set to influence this module's behavior:

``OpenMP_RUNTIME_MSVC``
  .. versionadded:: 3.30

  Specify the `OpenMP Runtime <msvc-openmp_>`_ when compiling with MSVC.
  If set to a non-empty value, such as ``experimental`` or ``llvm``, it
  will be passed as the value of the ``-openmp:`` flag.

.. _`msvc-openmp`: https://learn.microsoft.com/en-us/cpp/build/reference/openmp-enable-openmp-2-0-support

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``OpenMP_FOUND``
  .. versionadded:: 3.10

  Boolean indicating that OpenMP flags for all requested languages were
  found.  If no components are specified, this variable is set to boolean
  true if OpenMP settings for all enabled languages were detected.

``OpenMP_VERSION``
  .. versionadded:: 4.2

  Minimal version of the OpenMP standard detected among the requested
  languages, or all enabled languages if no components were specified.

This module will set the following variables per language in the
project, where ``<lang>`` is one of C, CXX, CUDA, or Fortran:

``OpenMP_<lang>_FOUND``
  .. versionadded:: 3.9

  Boolean indicating whether the (requested version of) OpenMP support for
  ``<lang>`` was detected.

``OpenMP_<lang>_VERSION``
  .. versionadded:: 3.9

  OpenMP version implemented by the ``<lang>`` compiler, if available.

``OpenMP_<lang>_VERSION_MAJOR``
  .. versionadded:: 3.9

  Major version of OpenMP implemented by the ``<lang>`` compiler, if
  available.

``OpenMP_<lang>_VERSION_MINOR``
  .. versionadded:: 3.9

  Minor version of OpenMP implemented by the ``<lang>`` compiler, if
  available.

``OpenMP_<lang>_SPEC_DATE``
  .. versionadded:: 3.7

  Date of the OpenMP specification implemented by the ``<lang>`` compiler,
  if available.

  The specification date is formatted as given in the OpenMP standard:
  ``yyyymm`` where ``yyyy`` and ``mm`` represents the year and month of the
  OpenMP specification implemented by the ``<lang>`` compiler.

``OpenMP_<lang>_FLAGS``
  OpenMP compiler flags for ``<lang>``, separated by spaces.

``OpenMP_<lang>_INCLUDE_DIRS``
  .. versionadded:: 3.16

  Directories that must be added to the header search path for ``<lang>``
  when using OpenMP.

For linking with OpenMP code written in ``<lang>``, the following
variables are provided:

``OpenMP_<lang>_LIB_NAMES``
  .. versionadded:: 3.9

  A :ref:`semicolon-separated list <CMake Language Lists>` of OpenMP
  programs libraries for ``<lang>``.

``OpenMP_<libname>_LIBRARY``
  .. versionadded:: 3.9

  Location of the individual libraries needed for OpenMP support in
  ``<lang>``.  The ``<libname>`` names are stored in the
  ``OpenMP_<lang>_LIB_NAMES`` list.

``OpenMP_<lang>_LIBRARIES``
  .. versionadded:: 3.9

  A list of libraries needed to link with OpenMP code written in ``<lang>``.

Specifically for Fortran, the module sets the following variables:

``OpenMP_Fortran_HAVE_OMPLIB_HEADER``
  .. versionadded:: 3.9

  Boolean indicating whether OpenMP is accessible through the ``omp_lib.h``
  Fortran header.

``OpenMP_Fortran_HAVE_OMPLIB_MODULE``
  .. versionadded:: 3.9

  Boolean indicating whether OpenMP is accessible through the ``omp_lib``
  Fortran module.

For some compilers, it may be necessary to add a header search path to find
the relevant OpenMP headers.  This location may be language-specific.  Where
this is needed, the module may attempt to find the location, but it can be
provided directly by setting the ``OpenMP_<lang>_INCLUDE_DIR`` cache variable.
Note that this variable is an *input* control to the module.  Project code
should use the ``OpenMP_<lang>_INCLUDE_DIRS`` *output* variable if it needs
to know what include directories are needed, or preferably the
``OpenMP::OpenMP_<lang>`` imported target.

Examples
^^^^^^^^

Finding OpenMP support and linking the imported target to a project target
using the C language component:

.. code-block:: cmake

  find_package(OpenMP)
  target_link_libraries(project_target PRIVATE OpenMP::OpenMP_C)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

function(_OPENMP_FLAG_CANDIDATES LANG)
  if(NOT OpenMP_${LANG}_FLAG)
    unset(OpenMP_FLAG_CANDIDATES)

    set(OMP_FLAG_GNU "-fopenmp")
    set(OMP_FLAG_LCC "-fopenmp")
    set(OMP_FLAG_Clang "-fopenmp=libomp" "-fopenmp=libiomp5" "-fopenmp" "-Xclang -fopenmp")
    set(OMP_FLAG_AppleClang "-Xclang -fopenmp")
    set(OMP_FLAG_HP "+Oopenmp")
    if(WIN32)
      set(OMP_FLAG_Intel "-Qopenmp")
    elseif(CMAKE_${LANG}_COMPILER_ID STREQUAL "Intel" AND
           "${CMAKE_${LANG}_COMPILER_VERSION}" VERSION_LESS "15.0.0.20140528")
      set(OMP_FLAG_Intel "-openmp")
    else()
      set(OMP_FLAG_Intel "-qopenmp")
    endif()
    if(CMAKE_${LANG}_COMPILER_ID STREQUAL "IntelLLVM" AND
      "x${CMAKE_${LANG}_COMPILER_FRONTEND_VARIANT}" STREQUAL "xMSVC")
      set(OMP_FLAG_IntelLLVM "-Qiopenmp")
    else()
      set(OMP_FLAG_IntelLLVM "-fiopenmp")
    endif()
    if(OpenMP_RUNTIME_MSVC)
      set(OMP_FLAG_MSVC "-openmp:${OpenMP_RUNTIME_MSVC}")
    else()
      set(OMP_FLAG_MSVC "-openmp")
    endif()
    set(OMP_FLAG_PathScale "-openmp")
    set(OMP_FLAG_NAG "-openmp")
    set(OMP_FLAG_Absoft "-openmp")
    set(OMP_FLAG_NVHPC "-mp")
    set(OMP_FLAG_PGI "-mp")
    set(OMP_FLAG_Flang "-fopenmp")
    set(OMP_FLAG_LLVMFlang "-fopenmp")
    set(OMP_FLAG_SunPro "-xopenmp")
    set(OMP_FLAG_XL "-qsmp=omp")
    # Cray compiler activate OpenMP with -h omp, which is enabled by default.
    set(OMP_FLAG_Cray " " "-h omp")
    set(OMP_FLAG_CrayClang "-fopenmp")
    set(OMP_FLAG_Fujitsu "-Kopenmp" "-KOMP")
    set(OMP_FLAG_FujitsuClang "-fopenmp" "-Kopenmp")

    if(CMAKE_${LANG}_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_${LANG}_HOST_COMPILER_ID)
      set(compiler_id "${CMAKE_${LANG}_HOST_COMPILER_ID}")
    else()
      set(compiler_id "${CMAKE_${LANG}_COMPILER_ID}")
    endif()

    # If we know the correct flags, use those
    if(DEFINED OMP_FLAG_${compiler_id})
      set(OpenMP_FLAG_CANDIDATES "${OMP_FLAG_${compiler_id}}")
    else()
      # Fall back to reasonable default tries otherwise
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
#elif !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
#  error \"_OPENMP not defined!\"
#endif
  return 0;
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

macro(_OPENMP_PREPARE_SOURCE LANG CONTENT_ID NAME_PREFIX FULLNAME_VAR CONTENT_VAR)
  if("${LANG}" STREQUAL "C")
    set(${FULLNAME_VAR} "${NAME_PREFIX}.c")
    set(${CONTENT_VAR} "${OpenMP_C_CXX_${CONTENT_ID}}")
  elseif("${LANG}" STREQUAL "CXX")
    set(${FULLNAME_VAR} "${NAME_PREFIX}.cpp")
    set(${CONTENT_VAR} "${OpenMP_C_CXX_${CONTENT_ID}}")
  elseif("${LANG}" STREQUAL "CUDA")
    set(${FULLNAME_VAR} "${NAME_PREFIX}.cu")
    set(${CONTENT_VAR} "${OpenMP_C_CXX_${CONTENT_ID}}")
  elseif("${LANG}" STREQUAL "Fortran")
    set(${FULLNAME_VAR} "${NAME_PREFIX}.F90")
    string(CONFIGURE "${OpenMP_Fortran_${CONTENT_ID}}" ${CONTENT_VAR} @ONLY)
  endif()
endmacro()

include(${CMAKE_CURRENT_LIST_DIR}/CMakeParseImplicitLinkInfo.cmake)

function(_OPENMP_GET_FLAGS LANG FLAG_MODE OPENMP_FLAG_VAR OPENMP_LIB_NAMES_VAR)
  _OPENMP_FLAG_CANDIDATES("${LANG}")
  _OPENMP_PREPARE_SOURCE("${LANG}" TEST_SOURCE OpenMPTryFlag
    _OPENMP_TEST_SRC_NAME _OPENMP_TEST_SRC_CONTENT)

  separate_arguments(OpenMP_VERBOSE_OPTIONS NATIVE_COMMAND "${CMAKE_${LANG}_VERBOSE_FLAG}")

  foreach(OPENMP_FLAG IN LISTS OpenMP_${LANG}_FLAG_CANDIDATES)
    string(REGEX REPLACE "[-/=+]" "" OPENMP_PLAIN_FLAG "${OPENMP_FLAG}")
    unset(_includeDirFlags)
    if(OpenMP_${LANG}_INCLUDE_DIR)
      set(_includeDirFlags "-DINCLUDE_DIRECTORIES:STRING=${OpenMP_${LANG}_INCLUDE_DIR}")
    endif()
    if(CMAKE_${LANG}_COMPILER_ID STREQUAL "NVIDIA")
      # With NVCC we drive linking directly through the host compiler, but
      # without language-wide flags since they may be specific to nvcc.
      # Pass the candidate OpenMP flag to the host compiler when linking.
      set(_OpenMP_LINK_OPTIONS "${OPENMP_FLAG}")
      # Exclude CUDA runtime libraries that we may add ourselves.
      # See the Compiler/NVIDIA module.  Do not exclude pthread,
      # as that is typically a dependency of OpenMP too.
      set(_OpenMP_EXCLUDE_IMPLICIT_LIBS cudart cudart_static cudadevrt rt dl)
    else()
      set(_OpenMP_LINK_OPTIONS "")
      set(_OpenMP_EXCLUDE_IMPLICIT_LIBS "")
    endif()
    try_compile( OpenMP_COMPILE_RESULT_${FLAG_MODE}_${OPENMP_PLAIN_FLAG}
      SOURCE_FROM_VAR "${_OPENMP_TEST_SRC_NAME}" _OPENMP_TEST_SRC_CONTENT
      LOG_DESCRIPTION "Detecting ${LANG} OpenMP compiler info"
      CMAKE_FLAGS "-DCOMPILE_DEFINITIONS:STRING=${OPENMP_FLAG}" ${_includeDirFlags}
      LINK_OPTIONS ${OpenMP_VERBOSE_OPTIONS} ${_OpenMP_LINK_OPTIONS}
      OUTPUT_VARIABLE OpenMP_TRY_COMPILE_OUTPUT
    )

    if(OpenMP_COMPILE_RESULT_${FLAG_MODE}_${OPENMP_PLAIN_FLAG} AND
       NOT "${CMAKE_${LANG}_COMPILER_ID};${CMAKE_${LANG}_SIMULATE_ID}" STREQUAL "Clang;MSVC")
      set("${OPENMP_FLAG_VAR}" "${OPENMP_FLAG}" PARENT_SCOPE)

      if(CMAKE_${LANG}_VERBOSE_FLAG)
        unset(OpenMP_${LANG}_IMPLICIT_LIBRARIES)
        unset(OpenMP_${LANG}_IMPLICIT_LINK_DIRS)
        unset(OpenMP_${LANG}_IMPLICIT_FWK_DIRS)
        unset(OpenMP_${LANG}_LOG_VAR)

        cmake_parse_implicit_link_info("${OpenMP_TRY_COMPILE_OUTPUT}"
          OpenMP_${LANG}_IMPLICIT_LIBRARIES
          OpenMP_${LANG}_IMPLICIT_LINK_DIRS
          OpenMP_${LANG}_IMPLICIT_FWK_DIRS
          OpenMP_${LANG}_LOG_VAR
          "${CMAKE_${LANG}_IMPLICIT_OBJECT_REGEX}"
          LANGUAGE ${LANG}
        )

        # For LCC we should additionally alanyze -print-search-dirs output
        # to check for additional implicit_dirs.
        # Note: This won't work if CMP0129 policy is set to OLD!
        if("${CMAKE_${LANG}_COMPILER_ID}" STREQUAL "LCC")
          execute_process(
            COMMAND ${CMAKE_${LANG}_COMPILER} -print-search-dirs
            OUTPUT_VARIABLE output_lines
            COMMAND_ERROR_IS_FATAL ANY
            ERROR_QUIET)
          if("${output_lines}" MATCHES ".*\nlibraries:[ \t]+(.*:)\n.*")
            string(REPLACE ":" ";" implicit_dirs_addon "${CMAKE_MATCH_1}")
            list(PREPEND OpenMP_${LANG}_IMPLICIT_LINK_DIRS ${implicit_dirs_addon})
            string(APPEND OpenMP_${LANG}_LOG_VAR
              "  Extended OpenMP library search paths: [${implicit_dirs}]\n")
          endif()
        endif()

        message(CONFIGURE_LOG
          "Parsed ${LANG} OpenMP implicit link information from above output:\n${OpenMP_${LANG}_LOG_VAR}\n\n")

        unset(_OPENMP_LIB_NAMES)
        foreach(_OPENMP_IMPLICIT_LIB IN LISTS OpenMP_${LANG}_IMPLICIT_LIBRARIES)
          get_filename_component(_OPENMP_IMPLICIT_LIB_DIR "${_OPENMP_IMPLICIT_LIB}" DIRECTORY)
          get_filename_component(_OPENMP_IMPLICIT_LIB_NAME "${_OPENMP_IMPLICIT_LIB}" NAME)
          get_filename_component(_OPENMP_IMPLICIT_LIB_PLAIN "${_OPENMP_IMPLICIT_LIB}" NAME_WE)
          string(REGEX REPLACE "([][+.*?()^$])" "\\\\\\1" _OPENMP_IMPLICIT_LIB_PLAIN_ESC "${_OPENMP_IMPLICIT_LIB_PLAIN}")
          string(REGEX REPLACE "([][+.*?()^$])" "\\\\\\1" _OPENMP_IMPLICIT_LIB_PATH_ESC "${_OPENMP_IMPLICIT_LIB}")
          if(NOT ( "${_OPENMP_IMPLICIT_LIB}" IN_LIST CMAKE_${LANG}_IMPLICIT_LINK_LIBRARIES
            OR "${_OPENMP_IMPLICIT_LIB}" IN_LIST _OpenMP_EXCLUDE_IMPLICIT_LIBS
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
                NO_PACKAGE_ROOT_PATH
                NO_CMAKE_PATH
                NO_CMAKE_ENVIRONMENT_PATH
              )
            endif()
            mark_as_advanced(OpenMP_${_OPENMP_IMPLICIT_LIB_PLAIN}_LIBRARY)
            list(APPEND _OPENMP_LIB_NAMES ${_OPENMP_IMPLICIT_LIB_PLAIN})
          endif()
        endforeach()
        list(REVERSE _OPENMP_LIB_NAMES)
        list(REMOVE_DUPLICATES _OPENMP_LIB_NAMES)
        list(REVERSE _OPENMP_LIB_NAMES)
        set("${OPENMP_LIB_NAMES_VAR}" "${_OPENMP_LIB_NAMES}" PARENT_SCOPE)
      else()
        # We do not know how to extract implicit OpenMP libraries for this compiler.
        # Assume that it handles them automatically, e.g. the Intel Compiler on
        # Windows should put the dependency in its object files.
        set("${OPENMP_LIB_NAMES_VAR}" "" PARENT_SCOPE)
      endif()
      break()
    elseif((CMAKE_${LANG}_COMPILER_ID STREQUAL "AppleClang"
      AND CMAKE_${LANG}_COMPILER_VERSION VERSION_GREATER_EQUAL "7.0") OR
      (CMAKE_${LANG}_COMPILER_ID STREQUAL "Clang" AND APPLE))

      # Check for separate OpenMP library on AppleClang 7+
      find_library(OpenMP_libomp_LIBRARY
        NAMES omp gomp iomp5
        HINTS ${CMAKE_${LANG}_IMPLICIT_LINK_DIRECTORIES}
      )
      mark_as_advanced(OpenMP_libomp_LIBRARY)

      if(OpenMP_libomp_LIBRARY)
        # Try without specifying include directory first. We only want to
        # explicitly add a search path if the header can't be found on the
        # default header search path already.
        try_compile( OpenMP_COMPILE_RESULT_${FLAG_MODE}_${OPENMP_PLAIN_FLAG}
          SOURCE_FROM_VAR "${_OPENMP_TEST_SRC_NAME}" _OPENMP_TEST_SRC_CONTENT
          LOG_DESCRIPTION "Trying ${LANG} OpenMP compiler with '${OpenMP_libomp_LIBRARY}'"
          COMPILE_DEFINITIONS ${OPENMP_FLAG}
          LINK_LIBRARIES ${OpenMP_libomp_LIBRARY}
        )
        if(NOT OpenMP_COMPILE_RESULT_${FLAG_MODE}_${OPENMP_PLAIN_FLAG})
          find_path(OpenMP_${LANG}_INCLUDE_DIR omp.h)
          mark_as_advanced(OpenMP_${LANG}_INCLUDE_DIR)
          set(OpenMP_${LANG}_INCLUDE_DIR "${OpenMP_${LANG}_INCLUDE_DIR}" PARENT_SCOPE)
          if(OpenMP_${LANG}_INCLUDE_DIR)
            try_compile( OpenMP_COMPILE_RESULT_${FLAG_MODE}_${OPENMP_PLAIN_FLAG}
              SOURCE_FROM_VAR "${_OPENMP_TEST_SRC_NAME}" _OPENMP_TEST_SRC_CONTENT
              LOG_DESCRIPTION "Trying ${LANG} OpenMP compiler with '${OpenMP_libomp_LIBRARY}' and '${OpenMP_${LANG}_INCLUDE_DIR}'"
              CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${OpenMP_${LANG}_INCLUDE_DIR}"
              COMPILE_DEFINITIONS ${OPENMP_FLAG}
              LINK_LIBRARIES ${OpenMP_libomp_LIBRARY}
            )
          endif()
        endif()
        if(OpenMP_COMPILE_RESULT_${FLAG_MODE}_${OPENMP_PLAIN_FLAG})
          set("${OPENMP_FLAG_VAR}" "${OPENMP_FLAG}" PARENT_SCOPE)
          set("${OPENMP_LIB_NAMES_VAR}" "libomp" PARENT_SCOPE)
          break()
        endif()
      endif()
    elseif(CMAKE_${LANG}_COMPILER_ID STREQUAL "LLVMFlang")
      find_library(OpenMP_libomp_LIBRARY
        NAMES omp
        HINTS ${CMAKE_${LANG}_IMPLICIT_LINK_DIRECTORIES}
      )
      mark_as_advanced(OpenMP_libomp_LIBRARY)

      if(OpenMP_libomp_LIBRARY)
        # Try without specifying include directory first. We only want to
        # explicitly add a search path if the header can't be found on the
        # default header search path already.
        try_compile( OpenMP_COMPILE_RESULT_${FLAG_MODE}_${OPENMP_PLAIN_FLAG}
          SOURCE_FROM_VAR "${_OPENMP_TEST_SRC_NAME}" _OPENMP_TEST_SRC_CONTENT
          LOG_DESCRIPTION "Trying ${LANG} OpenMP compiler with '${OpenMP_libomp_LIBRARY}'"
          COMPILE_DEFINITIONS ${OPENMP_FLAG}
          LINK_LIBRARIES ${OpenMP_libomp_LIBRARY}
        )
        if(NOT OpenMP_COMPILE_RESULT_${FLAG_MODE}_${OPENMP_PLAIN_FLAG})
          find_path(OpenMP_${LANG}_INCLUDE_DIR omp_lib.mod)
          mark_as_advanced(OpenMP_${LANG}_INCLUDE_DIR)
          set(OpenMP_${LANG}_INCLUDE_DIR "${OpenMP_${LANG}_INCLUDE_DIR}" PARENT_SCOPE)
          if(OpenMP_${LANG}_INCLUDE_DIR)
            try_compile( OpenMP_COMPILE_RESULT_${FLAG_MODE}_${OPENMP_PLAIN_FLAG}
              SOURCE_FROM_VAR "${_OPENMP_TEST_SRC_NAME}" _OPENMP_TEST_SRC_CONTENT
              LOG_DESCRIPTION "Trying ${LANG} OpenMP compiler with '${OpenMP_libomp_LIBRARY}' and '${OpenMP_${LANG}_INCLUDE_DIR}'"
              CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${OpenMP_${LANG}_INCLUDE_DIR}"
              COMPILE_DEFINITIONS ${OPENMP_FLAG}
              LINK_LIBRARIES ${OpenMP_libomp_LIBRARY}
            )
          endif()
        endif()
        if(OpenMP_COMPILE_RESULT_${FLAG_MODE}_${OPENMP_PLAIN_FLAG})
          set("${OPENMP_FLAG_VAR}" "${OPENMP_FLAG}" PARENT_SCOPE)
          set("${OPENMP_LIB_NAMES_VAR}" "libomp" PARENT_SCOPE)
          break()
        endif()
      endif()
    elseif(CMAKE_${LANG}_COMPILER_ID STREQUAL "Clang" AND WIN32)
      # Check for separate OpenMP library for Clang on Windows
      find_library(OpenMP_libomp_LIBRARY
        NAMES libomp libgomp libiomp5
        HINTS ${CMAKE_${LANG}_IMPLICIT_LINK_DIRECTORIES}
      )
      mark_as_advanced(OpenMP_libomp_LIBRARY)
      if(OpenMP_libomp_LIBRARY)
        try_compile( OpenMP_COMPILE_RESULT_${FLAG_MODE}_${OPENMP_PLAIN_FLAG}
          SOURCE_FROM_VAR "${_OPENMP_TEST_SRC_NAME}" _OPENMP_TEST_SRC_CONTENT
          LOG_DESCRIPTION "Trying ${LANG} OpenMP compiler with '${OpenMP_libomp_LIBRARY}'"
          COMPILE_DEFINITIONS ${OPENMP_FLAG}
          LINK_LIBRARIES ${OpenMP_libomp_LIBRARY}
        )
        if(OpenMP_COMPILE_RESULT_${FLAG_MODE}_${OPENMP_PLAIN_FLAG})
          set("${OPENMP_FLAG_VAR}" "${OPENMP_FLAG}" PARENT_SCOPE)
          set("${OPENMP_LIB_NAMES_VAR}" "libomp" PARENT_SCOPE)
          break()
        endif()
      endif()
    endif()
    set("${OPENMP_LIB_NAMES_VAR}" "NOTFOUND" PARENT_SCOPE)
    set("${OPENMP_FLAG_VAR}" "NOTFOUND" PARENT_SCOPE)
  endforeach()
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
  _OPENMP_PREPARE_SOURCE("${LANG}" CHECK_VERSION_SOURCE OpenMPCheckVersion
    _OPENMP_TEST_SRC_NAME _OPENMP_TEST_SRC_CONTENT)

  unset(_includeDirFlags)
  if(OpenMP_${LANG}_INCLUDE_DIR)
    set(_includeDirFlags "-DINCLUDE_DIRECTORIES:STRING=${OpenMP_${LANG}_INCLUDE_DIR}")
  endif()

  set(BIN_FILE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/FindOpenMP/ompver_${LANG}.bin")
  string(REGEX REPLACE "[-/=+]" "" OPENMP_PLAIN_FLAG "${OPENMP_FLAG}")
  try_compile(OpenMP_SPECTEST_${LANG}_${OPENMP_PLAIN_FLAG}
    SOURCE_FROM_VAR "${_OPENMP_TEST_SRC_NAME}" _OPENMP_TEST_SRC_CONTENT
    LOG_DESCRIPTION "Detecting ${LANG} OpenMP version"
    CMAKE_FLAGS "-DCOMPILE_DEFINITIONS:STRING=${OpenMP_${LANG}_FLAGS}" ${_includeDirFlags}
    COPY_FILE "${BIN_FILE}"
    )

  if(${OpenMP_SPECTEST_${LANG}_${OPENMP_PLAIN_FLAG}})
    file(STRINGS ${BIN_FILE} specstr LIMIT_COUNT 1 REGEX "INFO:OpenMP-date")
    set(regex_spec_date ".*INFO:OpenMP-date\\[0*([^]]*)\\].*")
    if("${specstr}" MATCHES "${regex_spec_date}")
      set(${SPEC_DATE} "${CMAKE_MATCH_1}" PARENT_SCOPE)
    endif()
  endif()
endfunction()

macro(_OPENMP_SET_VERSION_BY_SPEC_DATE LANG)
  set(OpenMP_SPEC_DATE_MAP
    "202111=5.2"
    "202011=5.1"
    # Preview versions
    "201611=5.0" # OpenMP 5.0 preview 1
    # Combined versions, 2.5 onwards
    "201811=5.0"
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
  if(MSVC)
    list(APPEND OpenMP_SPEC_DATE_MAP "2019=2.0")
  endif()

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

foreach(LANG IN ITEMS C CXX CUDA)
  if(CMAKE_${LANG}_COMPILER_LOADED)
    if(NOT DEFINED OpenMP_${LANG}_FLAGS OR "${OpenMP_${LANG}_FLAGS}" STREQUAL "NOTFOUND"
      OR NOT DEFINED OpenMP_${LANG}_LIB_NAMES OR "${OpenMP_${LANG}_LIB_NAMES}" STREQUAL "NOTFOUND")
      _OPENMP_GET_FLAGS("${LANG}" "${LANG}" OpenMP_${LANG}_FLAGS_WORK OpenMP_${LANG}_LIB_NAMES_WORK)
      if(NOT DEFINED OpenMP_${LANG}_FLAGS OR "${OpenMP_${LANG}_FLAGS}" STREQUAL "NOTFOUND")
        set(OpenMP_${LANG}_FLAGS "${OpenMP_${LANG}_FLAGS_WORK}"
          CACHE STRING "${LANG} compiler flags for OpenMP parallelization" FORCE)
      endif()
      if(NOT DEFINED OpenMP_${LANG}_LIB_NAMES OR "${OpenMP_${LANG}_LIB_NAMES}" STREQUAL "NOTFOUND")
        set(OpenMP_${LANG}_LIB_NAMES "${OpenMP_${LANG}_LIB_NAMES_WORK}"
          CACHE STRING "${LANG} compiler libraries for OpenMP parallelization" FORCE)
      endif()
      mark_as_advanced(OpenMP_${LANG}_FLAGS OpenMP_${LANG}_LIB_NAMES)
    endif()
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

    if(NOT DEFINED OpenMP_Fortran_FLAGS OR "${OpenMP_Fortran_FLAGS}" STREQUAL "NOTFOUND")
      set(OpenMP_Fortran_FLAGS "${OpenMP_Fortran_FLAGS_WORK}"
        CACHE STRING "Fortran compiler flags for OpenMP parallelization" FORCE)
    endif()
    if(NOT DEFINED OpenMP_Fortran_LIB_NAMES OR "${OpenMP_Fortran_LIB_NAMES}" STREQUAL "NOTFOUND")
      set(OpenMP_Fortran_LIB_NAMES "${OpenMP_Fortran_LIB_NAMES_WORK}"
        CACHE STRING "Fortran compiler libraries for OpenMP parallelization" FORCE)
    endif()
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

    if(NOT DEFINED OpenMP_Fortran_FLAGS OR "${OpenMP_Fortran_FLAGS}" STREQUAL "NOTFOUND")
      set(OpenMP_Fortran_FLAGS "${OpenMP_Fortran_FLAGS_WORK}"
        CACHE STRING "Fortran compiler flags for OpenMP parallelization" FORCE)
    endif()
    if(NOT DEFINED OpenMP_Fortran_LIB_NAMES OR "${OpenMP_Fortran_LIB_NAMES}" STREQUAL "NOTFOUND")
      set(OpenMP_Fortran_LIB_NAMES "${OpenMP_Fortran_LIB_NAMES_WORK}"
        CACHE STRING "Fortran compiler libraries for OpenMP parallelization" FORCE)
    endif()
  endif()

  if(OpenMP_Fortran_HAVE_OMPLIB_MODULE)
    set(OpenMP_Fortran_INCLUDE_LINE "use omp_lib\n      implicit none")
  else()
    set(OpenMP_Fortran_INCLUDE_LINE "implicit none\n      include 'omp_lib.h'")
  endif()
endif()

if(NOT OpenMP_FIND_COMPONENTS)
  set(OpenMP_FINDLIST C CXX CUDA Fortran)
else()
  set(OpenMP_FINDLIST ${OpenMP_FIND_COMPONENTS})
endif()

unset(OpenMP_VERSION)

include(FindPackageHandleStandardArgs)

foreach(LANG IN LISTS OpenMP_FINDLIST)
  if(CMAKE_${LANG}_COMPILER_LOADED)
    if (NOT OpenMP_${LANG}_SPEC_DATE AND OpenMP_${LANG}_FLAGS)
      _OPENMP_GET_SPEC_DATE("${LANG}" OpenMP_${LANG}_SPEC_DATE_INTERNAL)
      set(OpenMP_${LANG}_SPEC_DATE "${OpenMP_${LANG}_SPEC_DATE_INTERNAL}" CACHE
        INTERNAL "${LANG} compiler's OpenMP specification date")
    endif()
    _OPENMP_SET_VERSION_BY_SPEC_DATE("${LANG}")

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
      NAME_MISMATCHED
      REQUIRED_VARS OpenMP_${LANG}_FLAGS ${_OPENMP_${LANG}_REQUIRED_LIB_VARS}
      VERSION_VAR OpenMP_${LANG}_VERSION
    )

    if(OpenMP_${LANG}_FOUND)
      if(DEFINED OpenMP_${LANG}_VERSION)
        if(NOT OpenMP_VERSION OR OpenMP_VERSION VERSION_GREATER OpenMP_${LANG}_VERSION)
          set(OpenMP_VERSION "${OpenMP_${LANG}_VERSION}")
        endif()
      endif()
      set(OpenMP_${LANG}_LIBRARIES "")
      foreach(_OPENMP_IMPLICIT_LIB IN LISTS OpenMP_${LANG}_LIB_NAMES)
        list(APPEND OpenMP_${LANG}_LIBRARIES "${OpenMP_${_OPENMP_IMPLICIT_LIB}_LIBRARY}")
      endforeach()
      if(OpenMP_${LANG}_INCLUDE_DIR)
        set(OpenMP_${LANG}_INCLUDE_DIRS ${OpenMP_${LANG}_INCLUDE_DIR})
      else()
        set(OpenMP_${LANG}_INCLUDE_DIRS "")
      endif()

      if(NOT TARGET OpenMP::OpenMP_${LANG})
        add_library(OpenMP::OpenMP_${LANG} INTERFACE IMPORTED)
      endif()
      if(OpenMP_${LANG}_FLAGS)
        set_property(TARGET OpenMP::OpenMP_${LANG} PROPERTY
          INTERFACE_COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:${LANG}>:SHELL:${OpenMP_${LANG}_FLAGS}>")
        if(CMAKE_${LANG}_COMPILER_ID STREQUAL "Fujitsu"
          OR ${CMAKE_${LANG}_COMPILER_ID} STREQUAL "IntelLLVM"
          OR CMAKE_${LANG}_COMPILER_ID MATCHES "^(Cray|CrayClang)$")
          set_property(TARGET OpenMP::OpenMP_${LANG} PROPERTY
            INTERFACE_LINK_OPTIONS "SHELL:${OpenMP_${LANG}_FLAGS}")
        endif()
      endif()
      if(OpenMP_${LANG}_INCLUDE_DIRS)
        set_property(TARGET OpenMP::OpenMP_${LANG} PROPERTY
          INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${OpenMP_${LANG}_INCLUDE_DIRS}>")
      endif()
      if(OpenMP_${LANG}_LIBRARIES)
        set_property(TARGET OpenMP::OpenMP_${LANG} PROPERTY
          INTERFACE_LINK_LIBRARIES "${OpenMP_${LANG}_LIBRARIES}")
      endif()
    endif()
  endif()
endforeach()

unset(_OpenMP_REQ_VARS)
foreach(LANG IN ITEMS C CXX CUDA Fortran)
  if((NOT OpenMP_FIND_COMPONENTS AND CMAKE_${LANG}_COMPILER_LOADED) OR LANG IN_LIST OpenMP_FIND_COMPONENTS)
    list(APPEND _OpenMP_REQ_VARS "OpenMP_${LANG}_FOUND")
  endif()
endforeach()

find_package_handle_standard_args(OpenMP
    REQUIRED_VARS ${_OpenMP_REQ_VARS}
    VERSION_VAR OpenMP_VERSION
    HANDLE_COMPONENTS)

if(CMAKE_Fortran_COMPILER_LOADED AND OpenMP_Fortran_FOUND)
  if(NOT DEFINED OpenMP_Fortran_HAVE_OMPLIB_MODULE)
    set(OpenMP_Fortran_HAVE_OMPLIB_MODULE FALSE CACHE BOOL INTERNAL "")
  endif()
  if(NOT DEFINED OpenMP_Fortran_HAVE_OMPLIB_HEADER)
    set(OpenMP_Fortran_HAVE_OMPLIB_HEADER FALSE CACHE BOOL INTERNAL "")
  endif()
endif()

if(NOT ( CMAKE_C_COMPILER_LOADED OR CMAKE_CXX_COMPILER_LOADED OR CMAKE_CUDA_COMPILER_LOADED OR CMAKE_Fortran_COMPILER_LOADED ))
  message(SEND_ERROR "FindOpenMP requires the C, CXX, CUDA, or Fortran languages to be enabled")
endif()

unset(OpenMP_C_CXX_TEST_SOURCE)
unset(OpenMP_Fortran_TEST_SOURCE)
unset(OpenMP_C_CXX_CHECK_VERSION_SOURCE)
unset(OpenMP_Fortran_CHECK_VERSION_SOURCE)
unset(OpenMP_Fortran_INCLUDE_LINE)

cmake_policy(POP)
