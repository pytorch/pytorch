# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindHDF5
--------

Finds Hierarchical Data Format (HDF5), a library for reading and writing
self-describing array data:

.. code-block:: cmake

  find_package(HDF5 [<version>] [COMPONENTS <components>...] [...])

If the HDF5 library is built using its CMake-based build system, it will as
of HDF5 version 1.8.15 provide its own CMake Package Configuration file
(``hdf5-config.cmake``) for use with the :command:`find_package` command in
*config mode*.  By default, this module searches for this file and, if found,
returns the results based on the found configuration.

If the upstream configuration file is not found, this module falls back to
*module mode* and invokes the HDF5 wrapper compiler typically installed
with the HDF5 library.  Depending on the configuration, this wrapper
compiler is named either ``h5cc`` (serial) or ``h5pcc`` (parallel).  If
found, the wrapper is queried with the ``-show`` argument to determine the
compiler and linker flags required for building an HDF5 client application.
Both serial and parallel versions of the HDF5 wrapper are considered.  The
first directory containing either is used.  If both versions are found in the
same directory, the serial version is preferred by default.  To change this
behavior, set the variable ``HDF5_PREFER_PARALLEL`` to ``TRUE``.

In addition to finding the include directories and libraries needed to compile
an HDF5 application, this module also attempts to find additional tools
provided by the HDF5 distribution, which can be useful for regression testing
or development workflows.

Components
^^^^^^^^^^

This module supports optional components, which can be specified with the
:command:`find_package` command:

.. code-block:: cmake

  find_package(HDF5 [COMPONENTS <components>...])

Supported components include:

``C``
  Finds the ``HDF5`` C library (C bindings).

``CXX``
  Finds the ``HDF5`` C++ library (C++ bindings).

``Fortran``
  Finds the ``HDF5`` Fortran library (Fortran bindings).

``HL``
  This component can be used in combination with other components to find the
  high-level (HL) HDF5 library variants for C, CXX, and/or Fortran, which
  provide high-level functions.

If no components are specified, then this module will by default search for the
``C`` component.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``HDF5::HDF5``
  .. versionadded:: 3.19

  Target encapsulating the usage requirements for all found HDF5 binding
  libraries (``HDF5_LIBRARIES``), available if HDF5 and all required components
  are found.

``hdf5::hdf5``
  .. versionadded:: 3.19

  Target encapsulating the usage requirements for the HDF5 C library, available
  if HDF5 library and its ``C`` component are found.

``hdf5::hdf5_cpp``
  .. versionadded:: 3.19

  Target encapsulating the usage requirements for the HDF5 C and C++ libraries,
  available if HDF5 library, and its ``C`` and ``CXX`` components are found.

``hdf5::hdf5_fortran``
  .. versionadded:: 3.19

  Target encapsulating the usage requirements for the HDF5 Fortran library,
  available if HDF5 library and its ``Fortran`` component are found.

``hdf5::hdf5_hl``
  .. versionadded:: 3.19

  Target encapsulating the usage requirements for the HDF5 high-level C library,
  available if HDF5 library and its ``C``, and ``HL`` components are found.

``hdf5::hdf5_hl_cpp``
  .. versionadded:: 3.19

  High-level C++ library.

  Target encapsulating the usage requirements for the HDF5 high-level C and
  high-level C++ libraries, available if HDF5 library and its ``C``, ``CXX``,
  and ``HL`` components are found.

``hdf5::hdf5_hl_fortran``
  .. versionadded:: 3.19

  Target encapsulating the usage requirements for the HDF5 high-level Fortran
  library, available if HDF5 library and its ``Fortran``, and ``HL`` components
  are found.

``hdf5::h5diff``
  .. versionadded:: 3.19

  Imported executable target encapsulating the usage requirements for the
  ``h5diff`` executable, available if ``h5diff`` is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``HDF5_FOUND``
  Boolean indicating whether (the requested version of) HDF5 was found.

``HDF5_VERSION``
  .. versionadded:: 3.3

  The version of HDF5 library found.

``HDF5_INCLUDE_DIRS``
  Include directories containing header files needed to use HDF5.

``HDF5_DEFINITIONS``
  Required compiler definitions for using HDF5.

``HDF5_LIBRARIES``
  Libraries of all requested bindings needed to link against to use HDF5.

``HDF5_HL_LIBRARIES``
  Required libraries for the HDF5 high-level API for all bindings,
  if the ``HL`` component is enabled.

``HDF5_IS_PARALLEL``
  Boolean indicating whether the HDF5 library has parallel IO support.

For each enabled language binding component, a corresponding
``HDF5_<LANG>_LIBRARIES`` variable, and potentially
``HDF5_<LANG>_DEFINITIONS``, will be defined.  If the ``HL`` component is
enabled, then ``HDF5_<LANG>_HL_LIBRARIES`` variables will also be defined:

``HDF5_C_DEFINITIONS``
  Required compiler definitions for HDF5 C bindings.

``HDF5_CXX_DEFINITIONS``
  Required compiler definitions for HDF5 C++ bindings.

``HDF5_Fortran_DEFINITIONS``
  Required compiler definitions for HDF5 Fortran bindings.

``HDF5_C_INCLUDE_DIRS``
  Required include directories for HDF5 C bindings.

``HDF5_CXX_INCLUDE_DIRS``
  Required include directories for HDF5 C++ bindings.

``HDF5_Fortran_INCLUDE_DIRS``
  Required include directories for HDF5 Fortran bindings.

``HDF5_C_LIBRARIES``
  Required libraries for the HDF5 C bindings.

``HDF5_CXX_LIBRARIES``
  Required libraries for the HDF5 C++ bindings.

``HDF5_Fortran_LIBRARIES``
  Required libraries for the HDF5 Fortran bindings.

``HDF5_C_HL_LIBRARIES``
  Required libraries for the high-level C bindings, if the ``HL`` component
  is enabled.

``HDF5_CXX_HL_LIBRARIES``
  Required libraries for the high-level C++ bindings, if the ``HL``
  component is enabled.

``HDF5_Fortran_HL_LIBRARIES``
  Required libraries for the high-level Fortran bindings, if the ``HL``
  component is enabled.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``HDF5_C_COMPILER_EXECUTABLE``
  The path to the HDF5 C wrapper compiler.

``HDF5_CXX_COMPILER_EXECUTABLE``
  The path to the HDF5 C++ wrapper compiler.

``HDF5_Fortran_COMPILER_EXECUTABLE``
  The path to the HDF5 Fortran wrapper compiler.

``HDF5_C_COMPILER_EXECUTABLE_NO_INTERROGATE``
  .. versionadded:: 3.6

  The path to the primary C compiler which is also the HDF5 wrapper.
  This variable is used only in *module mode*.

``HDF5_CXX_COMPILER_EXECUTABLE_NO_INTERROGATE``
  .. versionadded:: 3.6

  The path to the primary C++ compiler which is also the HDF5 wrapper.
  This variable is used only in *module mode*.

``HDF5_Fortran_COMPILER_EXECUTABLE_NO_INTERROGATE``
  .. versionadded:: 3.6

  The path to the primary Fortran compiler which is also the HDF5 wrapper.
  This variable is used only in *module mode*.

``HDF5_DIFF_EXECUTABLE``
  The path to the HDF5 dataset comparison tool (``h5diff``).

Hints
^^^^^

The following variables can be set before calling the ``find_package(HDF5)``
to guide the search for HDF5 library:

``HDF5_PREFER_PARALLEL``
  .. versionadded:: 3.4

  Set this to boolean true to prefer parallel HDF5 (by default, serial is
  preferred).  This variable is used only in *module mode*.

``HDF5_FIND_DEBUG``
  .. versionadded:: 3.9

  Set this to boolean true to get extra debugging output by this module.

``HDF5_NO_FIND_PACKAGE_CONFIG_FILE``
  .. versionadded:: 3.8

  Set this to boolean true to skip finding and using CMake package configuration
  file (``hdf5-config.cmake``).

``HDF5_USE_STATIC_LIBRARIES``
  Set this to boolean value to determine whether or not to prefer a
  static link to a dynamic link for ``HDF5`` and all of its dependencies.

  .. versionadded:: 3.10
    Support for ``HDF5_USE_STATIC_LIBRARIES`` on Windows.

Examples
^^^^^^^^

Examples: Finding HDF5
""""""""""""""""""""""

Finding HDF5:

.. code-block:: cmake

  find_package(HDF5)

Specifying a minimum required version of HDF5 to find:

.. code-block:: cmake

  find_package(HDF5 1.8.15)

Finding HDF5 and making it required (if HDF5 is not found, processing stops with
an error message):

.. code-block:: cmake

  find_package(HDF5 1.8.15 REQUIRED)

Searching for static HDF5 libraries:

.. code-block:: cmake

  set(HDF5_USE_STATIC_LIBRARIES TRUE)
  find_package(HDF5)

Specifying components to find high-level C and C++ functions:

.. code-block:: cmake

  find_package(HDF5 COMPONENTS C CXX HL)

Examples: Using HDF5
""""""""""""""""""""

Finding HDF5 and linking it to a project target:

.. code-block:: cmake

  find_package(HDF5)
  target_link_libraries(project_target PRIVATE HDF5::HDF5)

Using Fortran HDF5 and HDF5-HL functions:

.. code-block:: cmake

  find_package(HDF5 COMPONENTS Fortran HL)
  target_link_libraries(project_target PRIVATE HDF5::HDF5)
#]=======================================================================]

include(${CMAKE_CURRENT_LIST_DIR}/SelectLibraryConfigurations.cmake)
include(FindPackageHandleStandardArgs)

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

# We haven't found HDF5 yet. Clear its state in case it is set in the parent
# scope somewhere else. We can't rely on it because different components may
# have been requested for this call.
set(HDF5_FOUND OFF)
set(HDF5_LIBRARIES)
set(HDF5_HL_LIBRARIES)

# List of the valid HDF5 components
set(HDF5_VALID_LANGUAGE_BINDINGS C CXX Fortran)

# Validate the list of find components.
if(NOT HDF5_FIND_COMPONENTS)
  set(HDF5_LANGUAGE_BINDINGS "C")
else()
  set(HDF5_LANGUAGE_BINDINGS)
  # add the extra specified components, ensuring that they are valid.
  set(HDF5_FIND_HL OFF)
  foreach(_component IN LISTS HDF5_FIND_COMPONENTS)
    list(FIND HDF5_VALID_LANGUAGE_BINDINGS ${_component} _component_location)
    if(NOT _component_location EQUAL -1)
      list(APPEND HDF5_LANGUAGE_BINDINGS ${_component})
    elseif(_component STREQUAL "HL")
      set(HDF5_FIND_HL ON)
    elseif(_component STREQUAL "Fortran_HL") # only for compatibility
      list(APPEND HDF5_LANGUAGE_BINDINGS Fortran)
      set(HDF5_FIND_HL ON)
      set(HDF5_FIND_REQUIRED_Fortran_HL FALSE)
      set(HDF5_FIND_REQUIRED_Fortran TRUE)
      set(HDF5_FIND_REQUIRED_HL TRUE)
    else()
      message(FATAL_ERROR "${_component} is not a valid HDF5 component.")
    endif()
  endforeach()
  unset(_component)
  unset(_component_location)
  if(NOT HDF5_LANGUAGE_BINDINGS)
    get_property(_langs GLOBAL PROPERTY ENABLED_LANGUAGES)
    foreach(_lang IN LISTS _langs)
      if(_lang MATCHES "^(C|CXX|Fortran)$")
        list(APPEND HDF5_LANGUAGE_BINDINGS ${_lang})
      endif()
    endforeach()
  endif()
  list(REMOVE_ITEM HDF5_FIND_COMPONENTS Fortran_HL) # replaced by Fortran and HL
  list(REMOVE_DUPLICATES HDF5_LANGUAGE_BINDINGS)
endif()

# Determine whether to search for serial or parallel executable first
if(HDF5_PREFER_PARALLEL)
  set(HDF5_C_COMPILER_NAMES h5pcc h5cc)
  set(HDF5_CXX_COMPILER_NAMES h5pc++ h5c++)
  set(HDF5_Fortran_COMPILER_NAMES h5pfc h5fc)
else()
  set(HDF5_C_COMPILER_NAMES h5cc h5pcc)
  set(HDF5_CXX_COMPILER_NAMES h5c++ h5pc++)
  set(HDF5_Fortran_COMPILER_NAMES h5fc h5pfc)
endif()

# Prefer h5hl<LANG> compilers if HDF5_FIND_HL is enabled
if(HDF5_FIND_HL)
  list(PREPEND HDF5_C_COMPILER_NAMES h5hlcc)
  list(PREPEND HDF5_CXX_COMPILER_NAMES h5hlc++)
  list(PREPEND HDF5_Fortran_COMPILER_NAMES h5hlfc)
endif()

# Test first if the current compilers automatically wrap HDF5
function(_HDF5_test_regular_compiler_C success version is_parallel)
  if(NOT ${success} OR
     NOT EXISTS ${_HDF5_TEST_DIR}/compiler_has_h5_c)
    file(WRITE "${_HDF5_TEST_DIR}/${_HDF5_TEST_SRC}"
      "#include <hdf5.h>\n"
      "const char* info_ver = \"INFO\" \":\" H5_VERSION;\n"
      "#ifdef H5_HAVE_PARALLEL\n"
      "const char* info_parallel = \"INFO\" \":\" \"PARALLEL\";\n"
      "#endif\n"
      "int main(int argc, char **argv) {\n"
      "  int require = 0;\n"
      "  require += info_ver[argc];\n"
      "#ifdef H5_HAVE_PARALLEL\n"
      "  require += info_parallel[argc];\n"
      "#endif\n"
      "  hid_t fid;\n"
      "  fid = H5Fcreate(\"foo.h5\",H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);\n"
      "  return 0;\n"
      "}")
    try_compile(${success} SOURCES "${_HDF5_TEST_DIR}/${_HDF5_TEST_SRC}"
      COPY_FILE ${_HDF5_TEST_DIR}/compiler_has_h5_c
    )
  endif()
  if(${success} AND EXISTS ${_HDF5_TEST_DIR}/compiler_has_h5_c)
    file(STRINGS ${_HDF5_TEST_DIR}/compiler_has_h5_c INFO_STRINGS
      REGEX "^INFO:"
    )
    string(REGEX MATCH "^INFO:([0-9]+\\.[0-9]+\\.[0-9]+)(-patch([0-9]+))?"
      INFO_VER "${INFO_STRINGS}"
    )
    set(${version} ${CMAKE_MATCH_1})
    if(CMAKE_MATCH_3)
      set(${version} ${HDF5_C_VERSION}.${CMAKE_MATCH_3})
    endif()
    set(${version} ${${version}} PARENT_SCOPE)

    if(INFO_STRINGS MATCHES "INFO:PARALLEL")
      set(${is_parallel} TRUE PARENT_SCOPE)
    else()
      set(${is_parallel} FALSE PARENT_SCOPE)
    endif()
  endif()
endfunction()

function(_HDF5_test_regular_compiler_CXX success version is_parallel)
  if(NOT ${success} OR
     NOT EXISTS ${_HDF5_TEST_DIR}/compiler_has_h5_cxx)
    file(WRITE "${_HDF5_TEST_DIR}/${_HDF5_TEST_SRC}"
      "#include <H5Cpp.h>\n"
      "#ifndef H5_NO_NAMESPACE\n"
      "using namespace H5;\n"
      "#endif\n"
      "const char* info_ver = \"INFO\" \":\" H5_VERSION;\n"
      "#ifdef H5_HAVE_PARALLEL\n"
      "const char* info_parallel = \"INFO\" \":\" \"PARALLEL\";\n"
      "#endif\n"
      "int main(int argc, char **argv) {\n"
      "  int require = 0;\n"
      "  require += info_ver[argc];\n"
      "#ifdef H5_HAVE_PARALLEL\n"
      "  require += info_parallel[argc];\n"
      "#endif\n"
      "  H5File file(\"foo.h5\", H5F_ACC_TRUNC);\n"
      "  return 0;\n"
      "}")
    try_compile(${success} SOURCES "${_HDF5_TEST_DIR}/${_HDF5_TEST_SRC}"
      COPY_FILE ${_HDF5_TEST_DIR}/compiler_has_h5_cxx
    )
  endif()
  if(${success} AND EXISTS ${_HDF5_TEST_DIR}/compiler_has_h5_cxx)
    file(STRINGS ${_HDF5_TEST_DIR}/compiler_has_h5_cxx INFO_STRINGS
      REGEX "^INFO:"
    )
    string(REGEX MATCH "^INFO:([0-9]+\\.[0-9]+\\.[0-9]+)(-patch([0-9]+))?"
      INFO_VER "${INFO_STRINGS}"
    )
    set(${version} ${CMAKE_MATCH_1})
    if(CMAKE_MATCH_3)
      set(${version} ${HDF5_CXX_VERSION}.${CMAKE_MATCH_3})
    endif()
    set(${version} ${${version}} PARENT_SCOPE)

    if(INFO_STRINGS MATCHES "INFO:PARALLEL")
      set(${is_parallel} TRUE PARENT_SCOPE)
    else()
      set(${is_parallel} FALSE PARENT_SCOPE)
    endif()
  endif()
endfunction()

function(_HDF5_test_regular_compiler_Fortran success is_parallel)
  if(NOT ${success})
    file(WRITE "${_HDF5_TEST_DIR}/${_HDF5_TEST_SRC}"
      "program hdf5_hello\n"
      "  use hdf5\n"
      "  integer error\n"
      "  call h5open_f(error)\n"
      "  call h5close_f(error)\n"
      "end\n")
    try_compile(${success} SOURCES "${_HDF5_TEST_DIR}/${_HDF5_TEST_SRC}")
    if(${success})
      execute_process(COMMAND ${CMAKE_Fortran_COMPILER} -showconfig
        OUTPUT_VARIABLE config_output
        ERROR_VARIABLE config_error
        RESULT_VARIABLE config_result
        )
      if(config_output MATCHES "Parallel HDF5: ([A-Za-z0-9]+)")
        # The value may be anything used when HDF5 was configured,
        # so see if CMake interprets it as "true".
        set(parallelHDF5 "${CMAKE_MATCH_1}")
        if(parallelHDF5)
          set(${is_parallel} TRUE PARENT_SCOPE)
        else()
          set(${is_parallel} FALSE PARENT_SCOPE)
        endif()
      else()
        set(${is_parallel} FALSE PARENT_SCOPE)
      endif()
    endif()
  endif()
endfunction()

# Invoke the HDF5 wrapper compiler.  The compiler return value is stored to the
# return_value argument, the text output is stored to the output variable.
function( _HDF5_invoke_compiler language output_var return_value_var version_var is_parallel_var)
  set(is_parallel FALSE)
  if(HDF5_USE_STATIC_LIBRARIES)
    set(lib_type_args -noshlib)
  else()
    set(lib_type_args -shlib)
  endif()
  # Verify that the compiler wrapper can actually compile: sometimes the compiler
  # wrapper exists, but not the compiler.  E.g. Miniconda / Anaconda Python
  execute_process(
    COMMAND ${HDF5_${language}_COMPILER_EXECUTABLE} "${_HDF5_TEST_SRC}"
    WORKING_DIRECTORY ${_HDF5_TEST_DIR}
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output
    RESULT_VARIABLE return_value
    )
  if(NOT return_value EQUAL 0)
    message(CONFIGURE_LOG
      "HDF5 ${language} compiler wrapper is unable to compile a minimal HDF5 program.\n\n${output}")
    if(NOT HDF5_FIND_QUIETLY)
      message(STATUS
        "HDF5 ${language} compiler wrapper is unable to compile a minimal HDF5 program.")
    endif()
  else()
    execute_process(
      COMMAND ${HDF5_${language}_COMPILER_EXECUTABLE} -show ${lib_type_args} "${_HDF5_TEST_SRC}"
      WORKING_DIRECTORY ${_HDF5_TEST_DIR}
      OUTPUT_VARIABLE output
      ERROR_VARIABLE output
      RESULT_VARIABLE return_value
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    if(NOT return_value EQUAL 0)
      message(CONFIGURE_LOG
        "Unable to determine HDF5 ${language} flags from HDF5 wrapper.\n\n${output}")
      if(NOT HDF5_FIND_QUIETLY)
        message(STATUS
          "Unable to determine HDF5 ${language} flags from HDF5 wrapper.")
      endif()
    endif()
    execute_process(
      COMMAND ${HDF5_${language}_COMPILER_EXECUTABLE} -showconfig
      OUTPUT_VARIABLE config_output
      ERROR_VARIABLE config_output
      RESULT_VARIABLE return_value
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    if(NOT return_value EQUAL 0)
      message(CONFIGURE_LOG
        "Unable to determine HDF5 ${language} version_var from HDF5 wrapper.\n\n${output}")
      if(NOT HDF5_FIND_QUIETLY)
        message(STATUS
          "Unable to determine HDF5 ${language} version_var from HDF5 wrapper.")
      endif()
    endif()
    string(REGEX MATCH "HDF5 Version: ([a-zA-Z0-9\\.\\-]*)" version "${config_output}")
    if(version)
      string(REPLACE "HDF5 Version: " "" version "${version}")
      string(REPLACE "-patch" "." version "${version}")
    endif()
    if(config_output MATCHES "Parallel HDF5: ([A-Za-z0-9]+)")
      # The value may be anything used when HDF5 was configured,
      # so see if CMake interprets it as "true".
      set(parallelHDF5 "${CMAKE_MATCH_1}")
      if(parallelHDF5)
        set(is_parallel TRUE)
      endif()
    endif()
  endif()
  foreach(var output return_value version is_parallel)
    set(${${var}_var} ${${var}} PARENT_SCOPE)
  endforeach()
endfunction()

# Parse a compile line for definitions, includes, library paths, and libraries.
function(_HDF5_parse_compile_line compile_line_var include_paths definitions
    library_paths libraries libraries_hl)

  separate_arguments(_compile_args NATIVE_COMMAND "${${compile_line_var}}")

  foreach(_arg IN LISTS _compile_args)
    if("${_arg}" MATCHES "^-I(.*)$")
      # include directory
      list(APPEND include_paths "${CMAKE_MATCH_1}")
    elseif("${_arg}" MATCHES "^-D(.*)$")
      # compile definition
      list(APPEND definitions "-D${CMAKE_MATCH_1}")
    elseif("${_arg}" MATCHES "^-L(.*)$")
      # library search path
      list(APPEND library_paths "${CMAKE_MATCH_1}")
    elseif("${_arg}" MATCHES "^-l(hdf5.*hl.*)$")
      # library name (hl)
      list(APPEND libraries_hl "${CMAKE_MATCH_1}")
    elseif("${_arg}" MATCHES "^-l(.*)$")
      # library name
      list(APPEND libraries "${CMAKE_MATCH_1}")
    elseif("${_arg}" MATCHES "^(.:)?[/\\].*\\.(a|so|dylib|sl|lib)$")
      # library file
      if(NOT EXISTS "${_arg}")
        continue()
      endif()
      get_filename_component(_lpath "${_arg}" DIRECTORY)
      get_filename_component(_lname "${_arg}" NAME_WE)
      string(REGEX REPLACE "^lib" "" _lname "${_lname}")
      list(APPEND library_paths "${_lpath}")
      if(_lname MATCHES "hdf5.*hl")
        list(APPEND libraries_hl "${_lname}")
      else()
        list(APPEND libraries "${_lname}")
      endif()
    endif()
  endforeach()
  foreach(var include_paths definitions library_paths libraries libraries_hl)
    set(${${var}_var} ${${var}} PARENT_SCOPE)
  endforeach()
endfunction()

# Select a preferred imported configuration from a target
function(_HDF5_select_imported_config target imported_conf)
    # We will first assign the value to a local variable _imported_conf, then assign
    # it to the function argument at the end.
    get_target_property(_imported_conf ${target} MAP_IMPORTED_CONFIG_${CMAKE_BUILD_TYPE})
    if (NOT _imported_conf)
        # Get available imported configurations by examining target properties
        get_target_property(_imported_conf ${target} IMPORTED_CONFIGURATIONS)
        if(HDF5_FIND_DEBUG)
            message(STATUS "Found imported configurations: ${_imported_conf}")
        endif()
        # Find the imported configuration that we prefer.
        # We do this by making list of configurations in order of preference,
        # starting with ${CMAKE_BUILD_TYPE} and ending with the first imported_conf
        set(_preferred_confs ${CMAKE_BUILD_TYPE})
        list(GET _imported_conf 0 _fallback_conf)
        list(APPEND _preferred_confs RELWITHDEBINFO RELEASE DEBUG ${_fallback_conf})
        if(HDF5_FIND_DEBUG)
            message(STATUS "Start search through imported configurations in the following order: ${_preferred_confs}")
        endif()
        # Now find the first of these that is present in imported_conf
        foreach (_conf IN LISTS _preferred_confs)
            if (${_conf} IN_LIST _imported_conf)
               set(_imported_conf ${_conf})
               break()
            endif()
        endforeach()
    endif()
    if(HDF5_FIND_DEBUG)
        message(STATUS "Selected imported configuration: ${_imported_conf}")
    endif()
    # assign value to function argument
    set(${imported_conf} ${_imported_conf} PARENT_SCOPE)
endfunction()


if(NOT HDF5_ROOT)
    set(HDF5_ROOT $ENV{HDF5_ROOT})
endif()
if(HDF5_ROOT)
    set(_HDF5_SEARCH_OPTS NO_DEFAULT_PATH)
else()
    set(_HDF5_SEARCH_OPTS)
endif()

# Try to find HDF5 using an installed hdf5-config.cmake
if(NOT HDF5_FOUND AND NOT HDF5_NO_FIND_PACKAGE_CONFIG_FILE)
    find_package(HDF5 QUIET NO_MODULE
      HINTS ${HDF5_ROOT}
      ${_HDF5_SEARCH_OPTS}
      )
    if( HDF5_FOUND)
        if(HDF5_FIND_DEBUG)
            message(STATUS "Found HDF5 at ${HDF5_DIR} via NO_MODULE. Now trying to extract locations etc.")
        endif()
        set(HDF5_IS_PARALLEL ${HDF5_ENABLE_PARALLEL})
        set(HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIR})
        set(HDF5_LIBRARIES)
        if (NOT TARGET hdf5 AND NOT TARGET hdf5-static AND NOT TARGET hdf5-shared)
            # Some HDF5 versions (e.g. 1.8.18) used hdf5::hdf5 etc
            set(_target_prefix "hdf5::")
        endif()
        set(HDF5_C_TARGET ${_target_prefix}hdf5)
        set(HDF5_C_HL_TARGET ${_target_prefix}hdf5_hl)
        set(HDF5_CXX_TARGET ${_target_prefix}hdf5_cpp)
        set(HDF5_CXX_HL_TARGET ${_target_prefix}hdf5_hl_cpp)
        set(HDF5_Fortran_TARGET ${_target_prefix}hdf5_fortran)
        set(HDF5_Fortran_HL_TARGET ${_target_prefix}hdf5_hl_fortran)
        set(HDF5_DEFINITIONS "")
        if(HDF5_USE_STATIC_LIBRARIES)
            set(_suffix "-static")
        else()
            set(_suffix "-shared")
        endif()
        foreach(_lang ${HDF5_LANGUAGE_BINDINGS})

            #Older versions of hdf5 don't have a static/shared suffix so
            #if we detect that occurrence clear the suffix
            if(_suffix AND NOT TARGET ${HDF5_${_lang}_TARGET}${_suffix})
              if(NOT TARGET ${HDF5_${_lang}_TARGET})
                #can't find this component with or without the suffix
                #so bail out, and let the following locate HDF5
                set(HDF5_FOUND FALSE)
                break()
              endif()
              set(_suffix "")
            endif()

            if(HDF5_FIND_DEBUG)
                message(STATUS "Trying to get properties of target ${HDF5_${_lang}_TARGET}${_suffix}")
            endif()
            # Find library for this target. Complicated as on Windows with a DLL, we need to search for the import-lib.
            _HDF5_select_imported_config(${HDF5_${_lang}_TARGET}${_suffix} _hdf5_imported_conf)
            get_target_property(_hdf5_lang_location ${HDF5_${_lang}_TARGET}${_suffix} IMPORTED_IMPLIB_${_hdf5_imported_conf} )
            if (NOT _hdf5_lang_location)
                # no import lib, just try LOCATION
                get_target_property(_hdf5_lang_location ${HDF5_${_lang}_TARGET}${_suffix} LOCATION_${_hdf5_imported_conf})
                if (NOT _hdf5_lang_location)
                    get_target_property(_hdf5_lang_location ${HDF5_${_lang}_TARGET}${_suffix} LOCATION)
                endif()
            endif()
            if( _hdf5_lang_location )
                set(HDF5_${_lang}_LIBRARY ${_hdf5_lang_location})
                list(APPEND HDF5_LIBRARIES ${HDF5_${_lang}_TARGET}${_suffix})
                set(HDF5_${_lang}_LIBRARIES ${HDF5_${_lang}_TARGET}${_suffix})
                set(HDF5_${_lang}_FOUND TRUE)
            endif()
            if(HDF5_FIND_HL)
                get_target_property(_hdf5_lang_hl_location ${HDF5_${_lang}_HL_TARGET}${_suffix} IMPORTED_IMPLIB_${_hdf5_imported_conf} )
                if (NOT _hdf5_lang_hl_location)
                    get_target_property(_hdf5_lang_hl_location ${HDF5_${_lang}_HL_TARGET}${_suffix} LOCATION_${_hdf5_imported_conf})
                    if (NOT _hdf5_hl_lang_location)
                        get_target_property(_hdf5_hl_lang_location ${HDF5_${_lang}_HL_TARGET}${_suffix} LOCATION)
                    endif()
                endif()
                if( _hdf5_lang_hl_location )
                    set(HDF5_${_lang}_HL_LIBRARY ${_hdf5_lang_hl_location})
                    list(APPEND HDF5_HL_LIBRARIES ${HDF5_${_lang}_HL_TARGET}${_suffix})
                    set(HDF5_${_lang}_HL_LIBRARIES ${HDF5_${_lang}_HL_TARGET}${_suffix})
                    set(HDF5_HL_FOUND TRUE)
                endif()
                unset(_hdf5_lang_hl_location)
            endif()
            unset(_hdf5_imported_conf)
            unset(_hdf5_lang_location)
        endforeach()
    endif()
endif()

if(NOT HDF5_FOUND)
  set(_HDF5_NEED_TO_SEARCH FALSE)
  set(_HDF5_TEST_DIR ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/hdf5)
  set(HDF5_COMPILER_NO_INTERROGATE TRUE)
  # Only search for languages we've enabled
  foreach(_lang IN LISTS HDF5_LANGUAGE_BINDINGS)
    set(HDF5_${_lang}_LIBRARIES)
    set(HDF5_${_lang}_HL_LIBRARIES)

    # First check to see if our regular compiler is one of wrappers
    if(_lang STREQUAL "C")
      set(_HDF5_TEST_SRC cmake_hdf5_test.c)
      if(CMAKE_CXX_COMPILER_LOADED AND NOT CMAKE_C_COMPILER_LOADED)
        # CXX project without C enabled
        set(_HDF5_TEST_SRC cmake_hdf5_test.cxx)
      endif()
      _HDF5_test_regular_compiler_C(
        HDF5_${_lang}_COMPILER_NO_INTERROGATE
        HDF5_${_lang}_VERSION
        HDF5_${_lang}_IS_PARALLEL)
    elseif(_lang STREQUAL "CXX")
      set(_HDF5_TEST_SRC cmake_hdf5_test.cxx)
      _HDF5_test_regular_compiler_CXX(
        HDF5_${_lang}_COMPILER_NO_INTERROGATE
        HDF5_${_lang}_VERSION
        HDF5_${_lang}_IS_PARALLEL)
    elseif(_lang STREQUAL "Fortran")
      set(_HDF5_TEST_SRC cmake_hdf5_test.f90)
      _HDF5_test_regular_compiler_Fortran(
        HDF5_${_lang}_COMPILER_NO_INTERROGATE
        HDF5_${_lang}_IS_PARALLEL)
    else()
      continue()
    endif()
    if(HDF5_${_lang}_COMPILER_NO_INTERROGATE)
      if(HDF5_FIND_DEBUG)
        message(STATUS "HDF5: Using hdf5 compiler wrapper for all ${_lang} compiling")
      endif()
      set(HDF5_${_lang}_FOUND TRUE)
      set(HDF5_${_lang}_COMPILER_EXECUTABLE_NO_INTERROGATE
          "${CMAKE_${_lang}_COMPILER}"
          CACHE FILEPATH "HDF5 ${_lang} compiler wrapper")
      set(HDF5_${_lang}_DEFINITIONS)
      set(HDF5_${_lang}_INCLUDE_DIRS)
      set(HDF5_${_lang}_LIBRARIES)
      set(HDF5_${_lang}_HL_LIBRARIES)

      mark_as_advanced(HDF5_${_lang}_COMPILER_EXECUTABLE_NO_INTERROGATE)

      set(HDF5_${_lang}_FOUND TRUE)
      set(HDF5_HL_FOUND TRUE)
    else()
      set(HDF5_COMPILER_NO_INTERROGATE FALSE)
      # If this language isn't using the wrapper, then try to seed the
      # search options with the wrapper
      find_program(HDF5_${_lang}_COMPILER_EXECUTABLE
        NAMES ${HDF5_${_lang}_COMPILER_NAMES} NAMES_PER_DIR
        HINTS ${HDF5_ROOT}
        PATH_SUFFIXES bin Bin
        DOC "HDF5 ${_lang} Wrapper compiler.  Used only to detect HDF5 compile flags."
        ${_HDF5_SEARCH_OPTS}
      )
      mark_as_advanced( HDF5_${_lang}_COMPILER_EXECUTABLE )
      unset(HDF5_${_lang}_COMPILER_NAMES)

      if(HDF5_${_lang}_COMPILER_EXECUTABLE)
        _HDF5_invoke_compiler(${_lang} HDF5_${_lang}_COMPILE_LINE
          HDF5_${_lang}_RETURN_VALUE HDF5_${_lang}_VERSION HDF5_${_lang}_IS_PARALLEL)
        if(HDF5_${_lang}_RETURN_VALUE EQUAL 0)
          if(HDF5_FIND_DEBUG)
            message(STATUS "HDF5: Using hdf5 compiler wrapper to determine ${_lang} configuration")
          endif()
          _HDF5_parse_compile_line( HDF5_${_lang}_COMPILE_LINE
            HDF5_${_lang}_INCLUDE_DIRS
            HDF5_${_lang}_DEFINITIONS
            HDF5_${_lang}_LIBRARY_DIRS
            HDF5_${_lang}_LIBRARY_NAMES
            HDF5_${_lang}_HL_LIBRARY_NAMES
          )
          set(HDF5_${_lang}_LIBRARIES)

          foreach(_lib IN LISTS HDF5_${_lang}_LIBRARY_NAMES)
            set(_HDF5_SEARCH_NAMES_LOCAL)
            if("x${_lib}" MATCHES "hdf5")
              # hdf5 library
              set(_HDF5_SEARCH_OPTS_LOCAL ${_HDF5_SEARCH_OPTS})
              if(HDF5_USE_STATIC_LIBRARIES)
                if(WIN32)
                  set(_HDF5_SEARCH_NAMES_LOCAL lib${_lib})
                else()
                  set(_HDF5_SEARCH_NAMES_LOCAL lib${_lib}.a)
                endif()
              endif()
            else()
              # external library
              set(_HDF5_SEARCH_OPTS_LOCAL)
            endif()
            find_library(HDF5_${_lang}_LIBRARY_${_lib}
              NAMES ${_HDF5_SEARCH_NAMES_LOCAL} ${_lib} NAMES_PER_DIR
              HINTS ${HDF5_${_lang}_LIBRARY_DIRS}
                    ${HDF5_ROOT}
              ${_HDF5_SEARCH_OPTS_LOCAL}
              )
            unset(_HDF5_SEARCH_OPTS_LOCAL)
            unset(_HDF5_SEARCH_NAMES_LOCAL)
            if(HDF5_${_lang}_LIBRARY_${_lib})
              list(APPEND HDF5_${_lang}_LIBRARIES ${HDF5_${_lang}_LIBRARY_${_lib}})
            else()
              list(APPEND HDF5_${_lang}_LIBRARIES ${_lib})
            endif()
          endforeach()
          if(HDF5_FIND_HL)
            set(HDF5_${_lang}_HL_LIBRARIES)
            foreach(_lib IN LISTS HDF5_${_lang}_HL_LIBRARY_NAMES)
              set(_HDF5_SEARCH_NAMES_LOCAL)
              if("x${_lib}" MATCHES "hdf5")
                # hdf5 library
                set(_HDF5_SEARCH_OPTS_LOCAL ${_HDF5_SEARCH_OPTS})
                if(HDF5_USE_STATIC_LIBRARIES)
                  if(WIN32)
                    set(_HDF5_SEARCH_NAMES_LOCAL lib${_lib})
                  else()
                    set(_HDF5_SEARCH_NAMES_LOCAL lib${_lib}.a)
                  endif()
                endif()
              else()
                # external library
                set(_HDF5_SEARCH_OPTS_LOCAL)
              endif()
              find_library(HDF5_${_lang}_LIBRARY_${_lib}
                NAMES ${_HDF5_SEARCH_NAMES_LOCAL} ${_lib} NAMES_PER_DIR
                HINTS ${HDF5_${_lang}_LIBRARY_DIRS}
                      ${HDF5_ROOT}
                ${_HDF5_SEARCH_OPTS_LOCAL}
                )
              unset(_HDF5_SEARCH_OPTS_LOCAL)
              unset(_HDF5_SEARCH_NAMES_LOCAL)
              if(HDF5_${_lang}_LIBRARY_${_lib})
                list(APPEND HDF5_${_lang}_HL_LIBRARIES ${HDF5_${_lang}_LIBRARY_${_lib}})
              else()
                list(APPEND HDF5_${_lang}_HL_LIBRARIES ${_lib})
              endif()
            endforeach()
            set(HDF5_HL_FOUND TRUE)
          endif()

          set(HDF5_${_lang}_FOUND TRUE)
          list(REMOVE_DUPLICATES HDF5_${_lang}_DEFINITIONS)
          list(REMOVE_DUPLICATES HDF5_${_lang}_INCLUDE_DIRS)
        else()
          set(_HDF5_NEED_TO_SEARCH TRUE)
        endif()
      else()
        set(_HDF5_NEED_TO_SEARCH TRUE)
      endif()
    endif()
    if(HDF5_${_lang}_VERSION)
      if(NOT HDF5_VERSION)
        set(HDF5_VERSION ${HDF5_${_lang}_VERSION})
      elseif(NOT HDF5_VERSION VERSION_EQUAL HDF5_${_lang}_VERSION)
        message(WARNING "HDF5 Version found for language ${_lang}, ${HDF5_${_lang}_VERSION} is different than previously found version ${HDF5_VERSION}")
      endif()
    endif()
    if(DEFINED HDF5_${_lang}_IS_PARALLEL)
      if(NOT DEFINED HDF5_IS_PARALLEL)
        set(HDF5_IS_PARALLEL ${HDF5_${_lang}_IS_PARALLEL})
      elseif(NOT HDF5_IS_PARALLEL AND HDF5_${_lang}_IS_PARALLEL)
        message(WARNING "HDF5 found for language ${_lang} is parallel but previously found language is not parallel.")
      elseif(HDF5_IS_PARALLEL AND NOT HDF5_${_lang}_IS_PARALLEL)
        message(WARNING "HDF5 found for language ${_lang} is not parallel but previously found language is parallel.")
      endif()
    endif()
  endforeach()
  unset(_HDF5_TEST_DIR)
  unset(_HDF5_TEST_SRC)
  unset(_lib)
else()
  set(_HDF5_NEED_TO_SEARCH TRUE)
endif()

if(NOT HDF5_FOUND AND HDF5_COMPILER_NO_INTERROGATE)
  # No arguments necessary, all languages can use the compiler wrappers
  set(HDF5_FOUND TRUE)
  set(HDF5_METHOD "Included by compiler wrappers")
  set(HDF5_REQUIRED_VARS HDF5_METHOD)
elseif(NOT HDF5_FOUND AND NOT _HDF5_NEED_TO_SEARCH)
  # Compiler wrappers aren't being used by the build but were found and used
  # to determine necessary include and library flags
  set(HDF5_INCLUDE_DIRS)
  set(HDF5_LIBRARIES)
  set(HDF5_HL_LIBRARIES)
  foreach(_lang IN LISTS HDF5_LANGUAGE_BINDINGS)
    if(HDF5_${_lang}_FOUND)
      if(NOT HDF5_${_lang}_COMPILER_NO_INTERROGATE)
        list(APPEND HDF5_DEFINITIONS ${HDF5_${_lang}_DEFINITIONS})
        list(APPEND HDF5_INCLUDE_DIRS ${HDF5_${_lang}_INCLUDE_DIRS})
        list(APPEND HDF5_LIBRARIES ${HDF5_${_lang}_LIBRARIES})
        if(HDF5_FIND_HL)
          list(APPEND HDF5_HL_LIBRARIES ${HDF5_${_lang}_HL_LIBRARIES})
        endif()
      endif()
    endif()
  endforeach()
  list(REMOVE_DUPLICATES HDF5_DEFINITIONS)
  list(REMOVE_DUPLICATES HDF5_INCLUDE_DIRS)
  set(HDF5_FOUND TRUE)
  set(HDF5_REQUIRED_VARS HDF5_LIBRARIES)
  if(HDF5_FIND_HL)
    list(APPEND HDF5_REQUIRED_VARS HDF5_HL_LIBRARIES)
  endif()
endif()

find_program( HDF5_DIFF_EXECUTABLE
    NAMES h5diff
    HINTS ${HDF5_ROOT}
    PATH_SUFFIXES bin Bin
    ${_HDF5_SEARCH_OPTS}
    DOC "HDF5 file differencing tool." )
mark_as_advanced( HDF5_DIFF_EXECUTABLE )

if( NOT HDF5_FOUND )
    # seed the initial lists of libraries to find with items we know we need
    set(HDF5_C_LIBRARY_NAMES          hdf5)
    set(HDF5_C_HL_LIBRARY_NAMES       hdf5_hl ${HDF5_C_LIBRARY_NAMES} )

    set(HDF5_CXX_LIBRARY_NAMES        hdf5_cpp    ${HDF5_C_LIBRARY_NAMES})
    set(HDF5_CXX_HL_LIBRARY_NAMES     hdf5_hl_cpp ${HDF5_C_HL_LIBRARY_NAMES} ${HDF5_CXX_LIBRARY_NAMES})

    set(HDF5_Fortran_LIBRARY_NAMES    hdf5_fortran   ${HDF5_C_LIBRARY_NAMES})
    set(HDF5_Fortran_HL_LIBRARY_NAMES hdf5_hl_fortran hdf5hl_fortran ${HDF5_C_HL_LIBRARY_NAMES} ${HDF5_Fortran_LIBRARY_NAMES})

    # suffixes as seen on Linux, MSYS2, ...
    set(_lib_suffixes hdf5)
    if(NOT HDF5_PREFER_PARALLEL)
      list(APPEND _lib_suffixes hdf5/serial)
    endif()
    if(HDF5_USE_STATIC_LIBRARIES)
      set(_inc_suffixes include/static)
    else()
      set(_inc_suffixes include/shared)
    endif()

    foreach(_lang IN LISTS HDF5_LANGUAGE_BINDINGS)
        set(HDF5_${_lang}_LIBRARIES)
        set(HDF5_${_lang}_HL_LIBRARIES)

        # The "main" library.
        set(_hdf5_main_library "")

        # find the HDF5 libraries
        foreach(LIB IN LISTS HDF5_${_lang}_LIBRARY_NAMES)
            if(HDF5_USE_STATIC_LIBRARIES)
                # According to bug 1643 on the CMake bug tracker, this is the
                # preferred method for searching for a static library.
                # See https://gitlab.kitware.com/cmake/cmake/-/issues/1643.  We search
                # first for the full static library name, but fall back to a
                # generic search on the name if the static search fails.
                set( THIS_LIBRARY_SEARCH_DEBUG
                    lib${LIB}d.a lib${LIB}_debug.a lib${LIB}d lib${LIB}_D lib${LIB}_debug
                    lib${LIB}d-static.a lib${LIB}_debug-static.a ${LIB}d-static ${LIB}_D-static ${LIB}_debug-static )
                set( THIS_LIBRARY_SEARCH_RELEASE lib${LIB}.a lib${LIB} lib${LIB}-static.a ${LIB}-static)
            else()
                set( THIS_LIBRARY_SEARCH_DEBUG ${LIB}d ${LIB}_D ${LIB}_debug ${LIB}d-shared ${LIB}_D-shared ${LIB}_debug-shared)
                set( THIS_LIBRARY_SEARCH_RELEASE ${LIB} ${LIB}-shared)
                if(WIN32)
                  list(APPEND HDF5_DEFINITIONS "-DH5_BUILT_AS_DYNAMIC_LIB")
                endif()
            endif()
            find_library(HDF5_${LIB}_LIBRARY_DEBUG
                NAMES ${THIS_LIBRARY_SEARCH_DEBUG}
                HINTS ${HDF5_ROOT} PATH_SUFFIXES lib Lib ${_lib_suffixes}
                ${_HDF5_SEARCH_OPTS}
            )
            find_library(HDF5_${LIB}_LIBRARY_RELEASE
                NAMES ${THIS_LIBRARY_SEARCH_RELEASE}
                HINTS ${HDF5_ROOT} PATH_SUFFIXES lib Lib ${_lib_suffixes}
                ${_HDF5_SEARCH_OPTS}
            )

            # Set the "main" library if not already set.
            if (NOT _hdf5_main_library)
              if (HDF5_${LIB}_LIBRARY_RELEASE)
                set(_hdf5_main_library "${HDF5_${LIB}_LIBRARY_RELEASE}")
              elseif (HDF5_${LIB}_LIBRARY_DEBUG)
                set(_hdf5_main_library "${HDF5_${LIB}_LIBRARY_DEBUG}")
              endif ()
            endif ()

            select_library_configurations( HDF5_${LIB} )
            list(APPEND HDF5_${_lang}_LIBRARIES ${HDF5_${LIB}_LIBRARY})
        endforeach()
        if(HDF5_${_lang}_LIBRARIES)
            set(HDF5_${_lang}_FOUND TRUE)
        endif()

        # Append the libraries for this language binding to the list of all
        # required libraries.
        list(APPEND HDF5_LIBRARIES ${HDF5_${_lang}_LIBRARIES})

        # find the HDF5 include directories
        set(_hdf5_inc_extra_paths)
        set(_hdf5_inc_extra_suffixes)
        if("${_lang}" STREQUAL "Fortran")
            set(HDF5_INCLUDE_FILENAME hdf5.mod HDF5.mod)

            # Add library-based search paths for Fortran modules.
            if (NOT _hdf5_main_library STREQUAL "")
              # gfortran module directory
              if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU" OR CMAKE_Fortran_COMPILER_ID STREQUAL "LCC")
                get_filename_component(_hdf5_library_dir "${_hdf5_main_library}" DIRECTORY)
                list(APPEND _hdf5_inc_extra_paths "${_hdf5_library_dir}")
                unset(_hdf5_library_dir)
                list(APPEND _hdf5_inc_extra_suffixes gfortran/modules)
              endif ()
            endif ()
        elseif("${_lang}" STREQUAL "CXX")
            set(HDF5_INCLUDE_FILENAME H5Cpp.h)
        else()
            set(HDF5_INCLUDE_FILENAME hdf5.h)
        endif()

        unset(_hdf5_main_library)

        find_path(HDF5_${_lang}_INCLUDE_DIR ${HDF5_INCLUDE_FILENAME}
            HINTS ${HDF5_ROOT}
            PATHS $ENV{HOME}/.local/include ${_hdf5_inc_extra_paths}
            PATH_SUFFIXES include Include ${_inc_suffixes} ${_lib_suffixes} ${_hdf5_inc_extra_suffixes}
            ${_HDF5_SEARCH_OPTS}
        )
        mark_as_advanced(HDF5_${_lang}_INCLUDE_DIR)
        unset(_hdf5_inc_extra_paths)
        unset(_hdf5_inc_extra_suffixes)
        # set the _DIRS variable as this is what the user will normally use
        set(HDF5_${_lang}_INCLUDE_DIRS ${HDF5_${_lang}_INCLUDE_DIR})
        list(APPEND HDF5_INCLUDE_DIRS ${HDF5_${_lang}_INCLUDE_DIR})

        if(HDF5_FIND_HL)
            foreach(LIB IN LISTS HDF5_${_lang}_HL_LIBRARY_NAMES)
                if(HDF5_USE_STATIC_LIBRARIES)
                    # According to bug 1643 on the CMake bug tracker, this is the
                    # preferred method for searching for a static library.
                    # See https://gitlab.kitware.com/cmake/cmake/-/issues/1643.  We search
                    # first for the full static library name, but fall back to a
                    # generic search on the name if the static search fails.
                    set( THIS_LIBRARY_SEARCH_DEBUG
                        lib${LIB}d.a lib${LIB}_debug.a lib${LIB}d lib${LIB}_D lib${LIB}_debug
                        lib${LIB}d-static.a lib${LIB}_debug-static.a lib${LIB}d-static lib${LIB}_D-static lib${LIB}_debug-static )
                    set( THIS_LIBRARY_SEARCH_RELEASE lib${LIB}.a lib${LIB} lib${LIB}-static.a lib${LIB}-static)
                else()
                    set( THIS_LIBRARY_SEARCH_DEBUG ${LIB}d ${LIB}_D ${LIB}_debug ${LIB}d-shared ${LIB}_D-shared ${LIB}_debug-shared)
                    set( THIS_LIBRARY_SEARCH_RELEASE ${LIB} ${LIB}-shared)
                endif()
                find_library(HDF5_${LIB}_LIBRARY_DEBUG
                    NAMES ${THIS_LIBRARY_SEARCH_DEBUG}
                    HINTS ${HDF5_ROOT} PATH_SUFFIXES lib Lib ${_lib_suffixes}
                    ${_HDF5_SEARCH_OPTS}
                )
                find_library(HDF5_${LIB}_LIBRARY_RELEASE
                    NAMES ${THIS_LIBRARY_SEARCH_RELEASE}
                    HINTS ${HDF5_ROOT} PATH_SUFFIXES lib Lib ${_lib_suffixes}
                    ${_HDF5_SEARCH_OPTS}
                )

                select_library_configurations( HDF5_${LIB} )
                list(APPEND HDF5_${_lang}_HL_LIBRARIES ${HDF5_${LIB}_LIBRARY})
            endforeach()

            # Append the libraries for this language binding to the list of all
            # required libraries.
            list(APPEND HDF5_HL_LIBRARIES ${HDF5_${_lang}_HL_LIBRARIES})
        endif()
    endforeach()
    if(HDF5_FIND_HL AND HDF5_HL_LIBRARIES)
        set(HDF5_HL_FOUND TRUE)
    endif()

    list(REMOVE_DUPLICATES HDF5_DEFINITIONS)
    list(REMOVE_DUPLICATES HDF5_INCLUDE_DIRS)

    # If the HDF5 include directory was found, open H5pubconf.h to determine if
    # HDF5 was compiled with parallel IO support
    set( HDF5_IS_PARALLEL FALSE )
    foreach( _dir IN LISTS HDF5_INCLUDE_DIRS )
      foreach(_hdr "${_dir}/H5pubconf.h" "${_dir}/H5pubconf-64.h" "${_dir}/H5pubconf-32.h")
        if( EXISTS "${_hdr}" )
            file( STRINGS "${_hdr}"
                HDF5_HAVE_PARALLEL_DEFINE
                REGEX "HAVE_PARALLEL 1" )
            if( HDF5_HAVE_PARALLEL_DEFINE )
                set( HDF5_IS_PARALLEL TRUE )
            endif()
            unset(HDF5_HAVE_PARALLEL_DEFINE)

            file( STRINGS "${_hdr}"
                HDF5_VERSION_DEFINE
                REGEX "^[ \t]*#[ \t]*define[ \t]+H5_VERSION[ \t]+" )
            if( "${HDF5_VERSION_DEFINE}" MATCHES
                "H5_VERSION[ \t]+\"([0-9\\.]+)(-patch([0-9]+))?\"" )
                set( HDF5_VERSION "${CMAKE_MATCH_1}" )
                if( CMAKE_MATCH_3 )
                  set( HDF5_VERSION ${HDF5_VERSION}.${CMAKE_MATCH_3})
                endif()
            endif()
            unset(HDF5_VERSION_DEFINE)
        endif()
      endforeach()
    endforeach()
    unset(_hdr)
    unset(_dir)
    set( HDF5_IS_PARALLEL ${HDF5_IS_PARALLEL} CACHE BOOL
        "HDF5 library compiled with parallel IO support" )
    mark_as_advanced( HDF5_IS_PARALLEL )

    set(HDF5_REQUIRED_VARS HDF5_LIBRARIES HDF5_INCLUDE_DIRS)
    if(HDF5_FIND_HL)
        list(APPEND HDF5_REQUIRED_VARS HDF5_HL_LIBRARIES)
    endif()
endif()

# For backwards compatibility we set HDF5_INCLUDE_DIR to the value of
# HDF5_INCLUDE_DIRS
if( HDF5_INCLUDE_DIRS )
  set( HDF5_INCLUDE_DIR "${HDF5_INCLUDE_DIRS}" )
endif()

# If HDF5_REQUIRED_VARS is empty at this point, then it's likely that
# something external is trying to explicitly pass already found
# locations
if(NOT HDF5_REQUIRED_VARS)
    set(HDF5_REQUIRED_VARS HDF5_LIBRARIES HDF5_INCLUDE_DIRS)
endif()

find_package_handle_standard_args(HDF5
    REQUIRED_VARS ${HDF5_REQUIRED_VARS}
    VERSION_VAR   HDF5_VERSION
    HANDLE_COMPONENTS
)

unset(_HDF5_SEARCH_OPTS)

if( HDF5_FOUND AND NOT HDF5_DIR)
  # hide HDF5_DIR for the non-advanced user to avoid confusion with
  # HDF5_DIR-NOT_FOUND while HDF5 was found.
  mark_as_advanced(HDF5_DIR)
endif()

if (HDF5_FOUND)
  if (NOT TARGET HDF5::HDF5)
    add_library(HDF5::HDF5 INTERFACE IMPORTED)
    string(REPLACE "-D" "" _hdf5_definitions "${HDF5_DEFINITIONS}")
    set_target_properties(HDF5::HDF5 PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIRS}"
      INTERFACE_COMPILE_DEFINITIONS "${_hdf5_definitions}")
    unset(_hdf5_definitions)
    target_link_libraries(HDF5::HDF5 INTERFACE ${HDF5_LIBRARIES})
  endif ()

  foreach (hdf5_lang IN LISTS HDF5_LANGUAGE_BINDINGS)
    if (hdf5_lang STREQUAL "C")
      set(hdf5_target_name "hdf5")
    elseif (hdf5_lang STREQUAL "CXX")
      set(hdf5_target_name "hdf5_cpp")
    elseif (hdf5_lang STREQUAL "Fortran")
      set(hdf5_target_name "hdf5_fortran")
    else ()
      continue ()
    endif ()

    if (NOT TARGET "hdf5::${hdf5_target_name}")
      if (HDF5_COMPILER_NO_INTERROGATE)
        add_library("hdf5::${hdf5_target_name}" INTERFACE IMPORTED)
        string(REPLACE "-D" "" _hdf5_definitions "${HDF5_${hdf5_lang}_DEFINITIONS}")
        set_target_properties("hdf5::${hdf5_target_name}" PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${HDF5_${hdf5_lang}_INCLUDE_DIRS}"
          INTERFACE_COMPILE_DEFINITIONS "${_hdf5_definitions}")
      else()
        if (DEFINED "HDF5_${hdf5_target_name}_LIBRARY")
          set(_hdf5_location "${HDF5_${hdf5_target_name}_LIBRARY}")
          set(_hdf5_location_release "${HDF5_${hdf5_target_name}_LIBRARY_RELEASE}")
          set(_hdf5_location_debug "${HDF5_${hdf5_target_name}_LIBRARY_DEBUG}")
        elseif (DEFINED "HDF5_${hdf5_lang}_LIBRARY")
          set(_hdf5_location "${HDF5_${hdf5_lang}_LIBRARY}")
          set(_hdf5_location_release "${HDF5_${hdf5_lang}_LIBRARY_RELEASE}")
          set(_hdf5_location_debug "${HDF5_${hdf5_lang}_LIBRARY_DEBUG}")
        elseif (DEFINED "HDF5_${hdf5_lang}_LIBRARY_${hdf5_target_name}")
          set(_hdf5_location "${HDF5_${hdf5_lang}_LIBRARY_${hdf5_target_name}}")
        else ()
          # Error if we still don't have the location.
          message(SEND_ERROR
            "HDF5 was found, but a different variable was set which contains "
            "the location of the `hdf5::${hdf5_target_name}` library.")
        endif ()
        add_library("hdf5::${hdf5_target_name}" UNKNOWN IMPORTED)
        string(REPLACE "-D" "" _hdf5_definitions "${HDF5_${hdf5_lang}_DEFINITIONS}")
        if (NOT HDF5_${hdf5_lang}_INCLUDE_DIRS)
         set(HDF5_${hdf5_lang}_INCLUDE_DIRS ${HDF5_INCLUDE_DIRS})
        endif ()
        set_target_properties("hdf5::${hdf5_target_name}" PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${HDF5_${hdf5_lang}_INCLUDE_DIRS}"
          INTERFACE_COMPILE_DEFINITIONS "${_hdf5_definitions}")
        if (_hdf5_location_release)
          set_property(TARGET "hdf5::${hdf5_target_name}" APPEND PROPERTY
            IMPORTED_CONFIGURATIONS RELEASE)
          set_property(TARGET "hdf5::${hdf5_target_name}" PROPERTY
            IMPORTED_LOCATION_RELEASE "${_hdf5_location_release}")
        endif()
        if (_hdf5_location_debug)
          set_property(TARGET "hdf5::${hdf5_target_name}" APPEND PROPERTY
            IMPORTED_CONFIGURATIONS DEBUG)
          set_property(TARGET "hdf5::${hdf5_target_name}" PROPERTY
            IMPORTED_LOCATION_DEBUG "${_hdf5_location_debug}")
        endif()
        if (NOT _hdf5_location_release AND NOT _hdf5_location_debug)
          set_property(TARGET "hdf5::${hdf5_target_name}" PROPERTY
            IMPORTED_LOCATION "${_hdf5_location}")
        endif()
        if (_hdf5_libtype STREQUAL "SHARED")
          set_property(TARGET "hdf5::${hdf5_target_name}" APPEND
            PROPERTY
              INTERFACE_COMPILE_DEFINITIONS H5_BUILT_AS_DYNAMIC_LIB)
        elseif (_hdf5_libtype STREQUAL "STATIC")
          set_property(TARGET "hdf5::${hdf5_target_name}" APPEND
            PROPERTY
              INTERFACE_COMPILE_DEFINITIONS H5_BUILT_AS_STATIC_LIB)
        endif ()
        unset(_hdf5_definitions)
        unset(_hdf5_libtype)
        unset(_hdf5_location)
        unset(_hdf5_location_release)
        unset(_hdf5_location_debug)
      endif ()
    endif ()

    if (NOT HDF5_FIND_HL)
      continue ()
    endif ()

    set(hdf5_alt_target_name "")
    if (hdf5_lang STREQUAL "C")
      set(hdf5_target_name "hdf5_hl")
    elseif (hdf5_lang STREQUAL "CXX")
      set(hdf5_target_name "hdf5_hl_cpp")
    elseif (hdf5_lang STREQUAL "Fortran")
      set(hdf5_target_name "hdf5_hl_fortran")
      set(hdf5_alt_target_name "hdf5hl_fortran")
    else ()
      continue ()
    endif ()

    if (NOT TARGET "hdf5::${hdf5_target_name}")
      if (HDF5_COMPILER_NO_INTERROGATE)
        add_library("hdf5::${hdf5_target_name}" INTERFACE IMPORTED)
        string(REPLACE "-D" "" _hdf5_definitions "${HDF5_${hdf5_lang}_HL_DEFINITIONS}")
        set_target_properties("hdf5::${hdf5_target_name}" PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${HDF5_${hdf5_lang}_HL_INCLUDE_DIRS}"
          INTERFACE_COMPILE_DEFINITIONS "${_hdf5_definitions}")
      else()
        if (DEFINED "HDF5_${hdf5_target_name}_LIBRARY")
          set(_hdf5_location "${HDF5_${hdf5_target_name}_LIBRARY}")
          set(_hdf5_location_release "${HDF5_${hdf5_target_name}_LIBRARY_RELEASE}")
          set(_hdf5_location_debug "${HDF5_${hdf5_target_name}_LIBRARY_DEBUG}")
        elseif (DEFINED "HDF5_${hdf5_lang}_HL_LIBRARY")
          set(_hdf5_location "${HDF5_${hdf5_lang}_HL_LIBRARY}")
          set(_hdf5_location_release "${HDF5_${hdf5_lang}_HL_LIBRARY_RELEASE}")
          set(_hdf5_location_debug "${HDF5_${hdf5_lang}_HL_LIBRARY_DEBUG}")
        elseif (DEFINED "HDF5_${hdf5_lang}_LIBRARY_${hdf5_target_name}")
          set(_hdf5_location "${HDF5_${hdf5_lang}_LIBRARY_${hdf5_target_name}}")
        elseif (hdf5_alt_target_name AND DEFINED "HDF5_${hdf5_lang}_LIBRARY_${hdf5_alt_target_name}")
          set(_hdf5_location "${HDF5_${hdf5_lang}_LIBRARY_${hdf5_alt_target_name}}")
        else ()
          # Error if we still don't have the location.
          message(SEND_ERROR
            "HDF5 was found, but a different variable was set which contains "
            "the location of the `hdf5::${hdf5_target_name}` library.")
        endif ()
        add_library("hdf5::${hdf5_target_name}" UNKNOWN IMPORTED)
        string(REPLACE "-D" "" _hdf5_definitions "${HDF5_${hdf5_lang}_HL_DEFINITIONS}")
        set_target_properties("hdf5::${hdf5_target_name}" PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${HDF5_${hdf5_lang}_HL_INCLUDE_DIRS}"
          INTERFACE_COMPILE_DEFINITIONS "${_hdf5_definitions}")
        if (_hdf5_location_release)
          set_property(TARGET "hdf5::${hdf5_target_name}" APPEND PROPERTY
            IMPORTED_CONFIGURATIONS RELEASE)
          set_property(TARGET "hdf5::${hdf5_target_name}" PROPERTY
            IMPORTED_LOCATION_RELEASE "${_hdf5_location_release}")
        endif()
        if (_hdf5_location_debug)
          set_property(TARGET "hdf5::${hdf5_target_name}" APPEND PROPERTY
            IMPORTED_CONFIGURATIONS DEBUG)
          set_property(TARGET "hdf5::${hdf5_target_name}" PROPERTY
            IMPORTED_LOCATION_DEBUG "${_hdf5_location_debug}")
        endif()
        if (NOT _hdf5_location_release AND NOT _hdf5_location_debug)
          set_property(TARGET "hdf5::${hdf5_target_name}" PROPERTY
            IMPORTED_LOCATION "${_hdf5_location}")
        endif()
        if (_hdf5_libtype STREQUAL "SHARED")
          set_property(TARGET "hdf5::${hdf5_target_name}" APPEND
            PROPERTY
              INTERFACE_COMPILE_DEFINITIONS H5_BUILT_AS_DYNAMIC_LIB)
        elseif (_hdf5_libtype STREQUAL "STATIC")
          set_property(TARGET "hdf5::${hdf5_target_name}" APPEND
            PROPERTY
              INTERFACE_COMPILE_DEFINITIONS H5_BUILT_AS_STATIC_LIB)
        endif ()
        unset(_hdf5_definitions)
        unset(_hdf5_libtype)
        unset(_hdf5_location)
      endif ()
    endif ()
  endforeach ()
  unset(hdf5_lang)

  if (HDF5_DIFF_EXECUTABLE AND NOT TARGET hdf5::h5diff)
    add_executable(hdf5::h5diff IMPORTED)
    set_target_properties(hdf5::h5diff PROPERTIES
      IMPORTED_LOCATION "${HDF5_DIFF_EXECUTABLE}")
  endif ()
endif ()

if (HDF5_FIND_DEBUG)
  message(STATUS "HDF5_DIR: ${HDF5_DIR}")
  message(STATUS "HDF5_DEFINITIONS: ${HDF5_DEFINITIONS}")
  message(STATUS "HDF5_INCLUDE_DIRS: ${HDF5_INCLUDE_DIRS}")
  message(STATUS "HDF5_LIBRARIES: ${HDF5_LIBRARIES}")
  message(STATUS "HDF5_HL_LIBRARIES: ${HDF5_HL_LIBRARIES}")
  foreach(_lang IN LISTS HDF5_LANGUAGE_BINDINGS)
    message(STATUS "HDF5_${_lang}_DEFINITIONS: ${HDF5_${_lang}_DEFINITIONS}")
    message(STATUS "HDF5_${_lang}_INCLUDE_DIR: ${HDF5_${_lang}_INCLUDE_DIR}")
    message(STATUS "HDF5_${_lang}_INCLUDE_DIRS: ${HDF5_${_lang}_INCLUDE_DIRS}")
    message(STATUS "HDF5_${_lang}_LIBRARY: ${HDF5_${_lang}_LIBRARY}")
    message(STATUS "HDF5_${_lang}_LIBRARIES: ${HDF5_${_lang}_LIBRARIES}")
    message(STATUS "HDF5_${_lang}_HL_LIBRARY: ${HDF5_${_lang}_HL_LIBRARY}")
    message(STATUS "HDF5_${_lang}_HL_LIBRARIES: ${HDF5_${_lang}_HL_LIBRARIES}")
  endforeach()
  message(STATUS "Defined targets (if any):")
  foreach(_lang IN  ITEMS "" "_cpp" "_fortran")
    foreach(_hl IN  ITEMS "" "_hl")
      foreach(_prefix IN ITEMS "hdf5::" "")
        foreach(_suffix IN ITEMS "-static" "-shared" "")
          set (_target ${_prefix}hdf5${_hl}${_lang}${_suffix})
          if (TARGET  ${_target})
            message(STATUS "... ${_target}")
          else()
            #message(STATUS "... ${_target} does not exist")
          endif()
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endif()
unset(_lang)
unset(_HDF5_NEED_TO_SEARCH)

cmake_policy(POP)
