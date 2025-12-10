# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
GenerateExportHeader
--------------------

This module provides commands for generating a header file containing
preprocessor macro definitions to control C/C++ symbol visibility.

Load this module in a CMake project with:

.. code-block:: cmake

  include(GenerateExportHeader)

.. versionadded:: 3.12
  Support for C projects.  Previous versions supported C++ projects only.

When developing C or C++ projects, especially for cross-platform use, symbol
visibility determines which functions, classes, global variables, templates,
and other symbols are made visible to users of the library.

For example, on Windows, symbols must be explicitly marked with
``__declspec(dllexport)`` when building a shared library, and
``__declspec(dllimport)`` when using it.  Other platforms may use attributes
like ``__attribute__((visibility("default")))``.

This module simplifies the creation and usage of preprocessor macros to
manage these requirements, avoiding repetitive and error-prone ``#ifdef``
blocks in source code.

Some symbol visibility can also be controlled with compiler options. In
CMake, target properties such as :prop_tgt:`<LANG>_VISIBILITY_PRESET` and
:prop_tgt:`VISIBILITY_INLINES_HIDDEN` enable compiler visibility flags,
where appropriate.  See also related convenience variables
:variable:`CMAKE_<LANG>_VISIBILITY_PRESET` and
:variable:`CMAKE_VISIBILITY_INLINES_HIDDEN` to enable it for all targets in
current scope.  These are commonly used in combination with this module to
further simplify C/C++ code, removing the need for some of the preprocessor
macros in the source code.

Commands
^^^^^^^^

This module provides the following commands:

Generating Export Header
""""""""""""""""""""""""

.. command:: generate_export_header

  Generates a header file suitable for inclusion in source code, containing
  preprocessor *export* macros for controlling the visibility of symbols:

  .. code-block:: cmake

    generate_export_header(
      <target>
      [BASE_NAME <base-name>]
      [EXPORT_FILE_NAME <export-file-name>]
      [EXPORT_MACRO_NAME <export-macro-name>]
      [NO_EXPORT_MACRO_NAME <no-export-macro-name>]
      [DEPRECATED_MACRO_NAME <deprecated-macro-name>]
      [DEFINE_NO_DEPRECATED]
      [NO_DEPRECATED_MACRO_NAME <no-deprecated-macro-name>]
      [STATIC_DEFINE <static-define>]
      [PREFIX_NAME <prefix>]
      [CUSTOM_CONTENT_FROM_VARIABLE <variable>]
      [INCLUDE_GUARD_NAME <include-guard-name>]
    )

  By default, this command generates a header file named
  ``<target-name-lowercase>_export.h`` in the current binary directory
  (:variable:`CMAKE_CURRENT_BINARY_DIR`).  This header defines a set of
  preprocessor macros used to mark API symbols as exported, hidden, or
  deprecated across different platforms and build types (e.g., static or
  shared builds), and is intended to be installed along with the library's
  public headers, because it affects public API declarations:

  * ``<MACRO>_EXPORT``: Marks symbols for export or import, making them
    visible as part of the public API when building or consuming a shared
    library.

  * ``<MACRO>_NO_EXPORT``: Marks symbols that should not be exported.
    If the :prop_tgt:`<LANG>_VISIBILITY_PRESET` target property is set to
    ``hidden``, using this macro in source code is typically redundant.

  * ``<MACRO>_DEPRECATED``: Marks symbols as deprecated.  When such symbols
    are used, the compiler emits a warning at compile-time.

  * ``<MACRO>_DEPRECATED_EXPORT``: Combines export/import and deprecation
    markers for a symbol that is both part of the public API and deprecated.

  * ``<MACRO>_DEPRECATED_NO_EXPORT``: Marks a deprecated symbol that should
    not be exported (internal and deprecated).

  * ``<MACRO>_NO_DEPRECATED``: A macro that can be used in source code to
    conditionally exclude deprecated code parts from the build via
    preprocessor logic.

  The ``<MACRO>`` part is derived by default from the uppercase name of the
  target or the explicitly provided ``<base-name>``.  All macro names can be
  customized using the optional arguments.

  .. rubric:: The arguments are:

  ``<target>``
    Name of a target for which the export header will be generated.
    Supported target types:

    * ``STATIC`` library (in this case, export-related macros are defined
      without values)
    * ``SHARED`` library
    * ``MODULE`` library
    * .. versionadded:: 3.1
        ``OBJECT`` library

  ``BASE_NAME <base-name>``
    If specified, it overrides the default file name and macro names.

  ``EXPORT_FILE_NAME <export-file-name>``
    If specified, it overrides the full path and the name of the generated
    export header file (``<base-name-lowercase>_export.h``) to
    ``<export-file-name>``.  If given as a relative path, it will be
    interpreted relative to the current binary directory
    (:variable:`CMAKE_CURRENT_BINARY_DIR`).

  ``EXPORT_MACRO_NAME <export-macro-name>``
    If specified, it overrides the default macro name for the export
    directive.

  ``NO_EXPORT_MACRO_NAME <no-export-macro-name>``
    If specified, the ``<no-export-macro-name>`` will be used for the macro
    name that designates the attribute for items that shouldn't be exported.

  ``DEPRECATED_MACRO_NAME <deprecated-macro-name>``
    If specified, the following names will be used:

    * ``<deprecated-macro-name>`` (macro for marking deprecated symbols)
    * ``<deprecated-macro-name>_EXPORT`` (macro for deprecated symbols with
      export markers)
    * ``<deprecated-macro-name>_NO_EXPORT`` (macro for deprecated symbols
      with no-export markers)

    instead of the default names in format of
    ``<MACRO>_DEPRECATED{,_EXPORT,_NO_EXPORT}``.

  ``DEFINE_NO_DEPRECATED``
    If specified, this will define a macro named ``<MACRO>_NO_DEPRECATED``.

  ``NO_DEPRECATED_MACRO_NAME <no-deprecated-macro-name>``
    Used in combination with ``DEFINE_NO_DEPRECATED`` option.  If specified,
    then a macro named ``<no-deprecated-macro-name>`` is used instead of the
    default ``<MACRO>_NO_DEPRECATED``.

  ``STATIC_DEFINE <static-define>``
    If specified, the ``<static-define>`` macro name will be used instead
    of the default ``<MACRO>_STATIC_DEFINE``.  This macro controls the
    symbol export behavior in the generated header for static libraries.
    It is typically used when building both shared and static variants of a
    library from the same sources using a single generated export header.
    When this macro is defined for static library, the export-related macros
    will expand to nothing.  This is important also on Windows, where symbol
    decoration is required only for shared libraries, not for static ones.

  ``PREFIX_NAME <prefix>``
    If specified, the additional ``<prefix>`` is prepended to all generated
    macro names.

  ``CUSTOM_CONTENT_FROM_VARIABLE <variable>``
    .. versionadded:: 3.7

    If specified, the content from the ``<variable>`` value is appended to
    the generated header file content after the preprocessor macros
    definitions.

  ``INCLUDE_GUARD_NAME <include-guard-name>``
    .. versionadded:: 3.11

    If specified, the ``<include-guard-name>`` is used as the preprocessor
    macro name to guard multiple inclusions of the generated header instead
    of the default name ``<export-macro-name>_H``.

    .. code-block:: c++
      :caption: ``<base-name-lowercase>_export.h``

      #ifndef <include-guard-name>
      #define <include-guard-name>
      // ...
      #endif /* <include-guard-name> */

Deprecated Command
""""""""""""""""""

.. command:: add_compiler_export_flags

  .. deprecated:: 3.0

    Set the target properties
    :prop_tgt:`CXX_VISIBILITY_PRESET <<LANG>_VISIBILITY_PRESET>` and
    :prop_tgt:`VISIBILITY_INLINES_HIDDEN` instead.

  Adds C++ compiler options ``-fvisibility=hidden`` (and
  ``-fvisibility-inlines-hidden``, if supported) to hide all symbols by
  default to either :variable:`CMAKE_CXX_FLAGS <CMAKE_<LANG>_FLAGS>`
  variable or to a specified variable:

  .. code-block:: cmake

    add_compiler_export_flags([<output_variable>])

  This command is a no-op on Windows which does not need extra compiler flags
  for exporting support.

  ``<output-variable>``
    Optional variable name that will be populated with a string of
    space-separated C++ compile options required to enable visibility
    support for the compiler/architecture in use.  If this argument is
    specified, the :variable:`CMAKE_CXX_FLAGS <CMAKE_<LANG>_FLAGS>` variable
    will not be modified.

Examples
^^^^^^^^

Example: Generating Export Header
"""""""""""""""""""""""""""""""""

The following example demonstrates how to use this module to generate an
export header in the current binary directory (``example_export.h``) and use
it in a C++ library named ``example`` to control symbols visibility.  The
generated header defines the preprocessor macros ``EXAMPLE_EXPORT``,
``EXAMPLE_NO_EXPORT``, ``EXAMPLE_DEPRECATED``, ``EXAMPLE_DEPRECATED_EXPORT``,
and ``EXAMPLE_DEPRECATED_NO_EXPORT``, and is installed along with the
library's other public headers:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``
  :emphasize-lines: 10,11

  cmake_minimum_required(VERSION 3.24)
  project(GenerateExportHeaderExample)

  # Set default visibility of all symbols to hidden
  set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
  set(CMAKE_VISIBILITY_INLINES_HIDDEN TRUE)

  add_library(example)

  include(GenerateExportHeader)
  generate_export_header(example)

  target_sources(
    example
    PRIVATE example.cxx
    PUBLIC
      FILE_SET HEADERS
        FILES example.h
      FILE_SET generated_headers
        TYPE HEADERS
        BASE_DIRS $<TARGET_PROPERTY:example,BINARY_DIR>
        FILES ${CMAKE_CURRENT_BINARY_DIR}/example_export.h
  )

  target_include_directories(example PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

  install(
    TARGETS example
    FILE_SET HEADERS
    FILE_SET generated_headers
  )

And in the ABI header files:

.. code-block:: c++
  :caption: ``example.h``

  #include "example_export.h"

  // This class is part of the public API and is exported
  class EXAMPLE_EXPORT SomeClass
  {
  public:
    SomeClass();
    void doSomething();

    // This method is deprecated
    EXAMPLE_DEPRECATED void legacyMethod();
  };

  // This function is exported and deprecated
  EXAMPLE_DEPRECATED_EXPORT void legacyPublicFunction();

  // This function is deprecated but not exported
  EXAMPLE_DEPRECATED void legacyInternalFunction();

.. code-block:: c++
  :caption: ``example.cxx``

  #include <iostream>
  #include "example.h"

  SomeClass::SomeClass() = default;

  void SomeClass::doSomething()
  {
    std::cout << "SomeClass::doSomething() called" << std::endl;
  }

  void SomeClass::legacyMethod()
  {
    std::cout << "SomeClass::legacyMethod() is deprecated" << std::endl;
  }

  void legacyPublicFunction()
  {
    std::cout << "legacyPublicFunction() is deprecated" << std::endl;
  }

  void internalLegacyFunction()
  {
    std::cout << "legacyInternalFunction() is deprecated" << std::endl;
  }

Examples: Customizing Generated Header
""""""""""""""""""""""""""""""""""""""

The ``BASE_NAME`` argument can be used to override the generated file name
and the names used for the macros.  The following will generate a file
named ``other_name_export.h`` containing export-related macros such as
``OTHER_NAME_EXPORT``, ``OTHER_NAME_NO_EXPORT``, ``OTHER_NAME_DEPRECATED``,
etc.

.. code-block:: cmake

  add_library(example example.cxx)
  include(GenerateExportHeader)
  generate_export_header(example BASE_NAME "other_name")

The ``BASE_NAME`` may be overridden by specifying other command options.
For example, the following creates a macro ``OTHER_NAME_EXPORT`` instead of
``EXAMPLE_EXPORT``, but other macros and the generated header file name are
set to their default values:

.. code-block:: cmake

  add_library(example example.cxx)
  include(GenerateExportHeader)
  generate_export_header(example EXPORT_MACRO_NAME "OTHER_NAME_EXPORT")

The following example creates ``KDE_DEPRECATED`` macro instead of
default ``EXAMPLE_DEPRECATED``:

.. code-block:: cmake

  add_library(example example.cxx)
  include(GenerateExportHeader)
  generate_export_header(example DEPRECATED_MACRO_NAME "KDE_DEPRECATED")

The ``DEFINE_NO_DEPRECATED`` option can be used to define a macro which can
be used to remove deprecated code from preprocessor output:

.. code-block:: cmake

  option(EXCLUDE_DEPRECATED "Exclude deprecated parts of the library")
  if(EXCLUDE_DEPRECATED)
    set(NO_BUILD_DEPRECATED DEFINE_NO_DEPRECATED)
  endif()

  include(GenerateExportHeader)
  generate_export_header(example ${NO_BUILD_DEPRECATED})

.. code-block:: c++
  :caption: ``example.h``

  class EXAMPLE_EXPORT SomeClass
  {
  public:
  #ifndef EXAMPLE_NO_DEPRECATED
    EXAMPLE_DEPRECATED void legacyMethod();
  #endif
  };

.. code-block:: c++
  :caption: ``example.cxx``

  #ifndef EXAMPLE_NO_DEPRECATED
  void SomeClass::legacyMethod() {  }
  #endif

The ``PREFIX_NAME`` argument can be used to prepend all generated macro names
with some prefix.  For example, the following will generate macros such as
``VTK_SOMELIB_EXPORT``, etc.

.. code-block:: cmake

  include(GenerateExportHeader)
  generate_export_header(somelib PREFIX_NAME "VTK_")

Appending additional content to generated header can be done with the
``CUSTOM_CONTENT_FROM_VARIABLE`` argument:

.. code-block:: cmake

  include(GenerateExportHeader)
  set(content [[#include "project_api.h"]])
  generate_export_header(example CUSTOM_CONTENT_FROM_VARIABLE content)

Example: Building Shared and Static Library
"""""""""""""""""""""""""""""""""""""""""""

In the following example both a shared and a static library are built from
the same sources, and the ``<MACRO>_STATIC_DEFINE`` macro compile definition
is defined to ensure the same generated export header works for both:

.. code-block:: cmake

  add_library(example_shared SHARED example.cxx)
  add_library(example_static STATIC example.cxx)

  include(GenerateExportHeader)
  generate_export_header(example_shared BASE_NAME "example")

  # Define macro to disable export attributes for static build
  target_compile_definitions(example_static PRIVATE EXAMPLE_STATIC_DEFINE)

Example: Upgrading Deprecated Command
"""""""""""""""""""""""""""""""""""""

In earlier versions of CMake, ``add_compiler_export_flags()`` command was
used to add symbol visibility compile options:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  add_library(example example.cxx)

  include(GenerateExportHeader)

  add_compiler_export_flags(flags)
  string(REPLACE " " ";" flags "${flags}")
  set_property(TARGET example APPEND PROPERTY COMPILE_OPTIONS "${flags}")

  generate_export_header(example)

In new code, the following target properties are used to achieve the same
functionality:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  add_library(example example.cxx)

  include(GenerateExportHeader)

  set_target_properties(
    example
    PROPERTIES
      CXX_VISIBILITY_PRESET hidden
      VISIBILITY_INLINES_HIDDEN TRUE
  )

  generate_export_header(example)

See Also
^^^^^^^^

* The :prop_tgt:`DEFINE_SYMBOL` target property to customize the preprocessor
  macro name used by the generated header.  This macro determines whether
  the library header is being included during the library's own compilation
  or when it is used by another project (e.g., after installation).
* The :prop_tgt:`ENABLE_EXPORTS` target property.
* The :prop_tgt:`WINDOWS_EXPORT_ALL_SYMBOLS` target property.
#]=======================================================================]

include(CheckCompilerFlag)
include(CheckSourceCompiles)

# TODO: Install this macro separately?
macro(_check_cxx_compiler_attribute _ATTRIBUTE _RESULT)
  check_source_compiles(CXX "${_ATTRIBUTE} int somefunc() { return 0; }
    int main() { return somefunc();}" ${_RESULT}
  )
endmacro()

# TODO: Install this macro separately?
macro(_check_c_compiler_attribute _ATTRIBUTE _RESULT)
  check_source_compiles(C "${_ATTRIBUTE} int somefunc(void) { return 0; }
    int main(void) { return somefunc();}" ${_RESULT}
  )
endmacro()

macro(_test_compiler_hidden_visibility)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
    AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.2")
    set(GCC_TOO_OLD TRUE)
  elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU"
    AND CMAKE_C_COMPILER_VERSION VERSION_LESS "4.2")
    set(GCC_TOO_OLD TRUE)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "12.0")
    set(_INTEL_TOO_OLD TRUE)
  endif()

  # Exclude XL here because it misinterprets -fvisibility=hidden even though
  # the check_compiler_flag passes
  if(NOT GCC_TOO_OLD
      AND NOT _INTEL_TOO_OLD
      AND NOT WIN32
      AND NOT CYGWIN
      AND NOT CMAKE_CXX_COMPILER_ID MATCHES "^(IBMClang|XLClang|XL)$"
      AND NOT CMAKE_CXX_COMPILER_ID MATCHES "^(PGI|NVHPC)$"
      AND NOT CMAKE_CXX_COMPILER_ID MATCHES Watcom)
    if (CMAKE_CXX_COMPILER_LOADED)
      check_compiler_flag(CXX -fvisibility=hidden COMPILER_HAS_HIDDEN_VISIBILITY)
      check_compiler_flag(CXX -fvisibility-inlines-hidden
        COMPILER_HAS_HIDDEN_INLINE_VISIBILITY)
    else()
      check_compiler_flag(C -fvisibility=hidden COMPILER_HAS_HIDDEN_VISIBILITY)
      check_compiler_flag(C -fvisibility-inlines-hidden
        COMPILER_HAS_HIDDEN_INLINE_VISIBILITY)
    endif()
  endif()
endmacro()

macro(_test_compiler_has_deprecated)
  # NOTE:  Some Embarcadero compilers silently compile __declspec(deprecated)
  # without error, but this is not a documented feature and the attribute does
  # not actually generate any warnings.
  if(CMAKE_CXX_COMPILER_ID MATCHES Borland
      OR CMAKE_CXX_COMPILER_ID MATCHES Embarcadero
      OR CMAKE_CXX_COMPILER_ID MATCHES HP
      OR GCC_TOO_OLD
      OR CMAKE_CXX_COMPILER_ID MATCHES "^(PGI|NVHPC)$"
      OR CMAKE_CXX_COMPILER_ID MATCHES Watcom)
    set(COMPILER_HAS_DEPRECATED "" CACHE INTERNAL
      "Compiler support for a deprecated attribute")
  else()
    if (CMAKE_CXX_COMPILER_LOADED)
      _check_cxx_compiler_attribute("__attribute__((__deprecated__))"
        COMPILER_HAS_DEPRECATED_ATTR)
      if(COMPILER_HAS_DEPRECATED_ATTR)
        set(COMPILER_HAS_DEPRECATED "${COMPILER_HAS_DEPRECATED_ATTR}"
          CACHE INTERNAL "Compiler support for a deprecated attribute")
      else()
        _check_cxx_compiler_attribute("__declspec(deprecated)"
          COMPILER_HAS_DEPRECATED)
      endif()
    else()
      _check_c_compiler_attribute("__attribute__((__deprecated__))"
        COMPILER_HAS_DEPRECATED_ATTR)
      if(COMPILER_HAS_DEPRECATED_ATTR)
        set(COMPILER_HAS_DEPRECATED "${COMPILER_HAS_DEPRECATED_ATTR}"
          CACHE INTERNAL "Compiler support for a deprecated attribute")
      else()
        _check_c_compiler_attribute("__declspec(deprecated)"
          COMPILER_HAS_DEPRECATED)
      endif()

    endif()
  endif()
endmacro()

macro(_DO_SET_MACRO_VALUES TARGET_LIBRARY)
  set(DEFINE_DEPRECATED)
  set(DEFINE_EXPORT)
  set(DEFINE_IMPORT)
  set(DEFINE_NO_EXPORT)

  if (COMPILER_HAS_DEPRECATED_ATTR AND NOT WIN32)
    set(DEFINE_DEPRECATED "__attribute__ ((__deprecated__))")
  elseif(COMPILER_HAS_DEPRECATED)
    set(DEFINE_DEPRECATED "__declspec(deprecated)")
  endif()

  get_property(type TARGET ${TARGET_LIBRARY} PROPERTY TYPE)

  if(NOT ${type} STREQUAL "STATIC_LIBRARY")
    if(WIN32 OR CYGWIN)
      set(DEFINE_EXPORT "__declspec(dllexport)")
      set(DEFINE_IMPORT "__declspec(dllimport)")
    elseif(COMPILER_HAS_HIDDEN_VISIBILITY)
      set(DEFINE_EXPORT "__attribute__((visibility(\"default\")))")
      set(DEFINE_IMPORT "__attribute__((visibility(\"default\")))")
      set(DEFINE_NO_EXPORT "__attribute__((visibility(\"hidden\")))")
    endif()
  endif()
endmacro()

function(_DO_GENERATE_EXPORT_HEADER TARGET_LIBRARY)
  # Option overrides
  set(options DEFINE_NO_DEPRECATED)
  set(oneValueArgs PREFIX_NAME BASE_NAME EXPORT_MACRO_NAME EXPORT_FILE_NAME
    DEPRECATED_MACRO_NAME NO_EXPORT_MACRO_NAME STATIC_DEFINE
    NO_DEPRECATED_MACRO_NAME CUSTOM_CONTENT_FROM_VARIABLE INCLUDE_GUARD_NAME)
  set(multiValueArgs)

  cmake_parse_arguments(_GEH "${options}" "${oneValueArgs}" "${multiValueArgs}"
    ${ARGN})

  set(BASE_NAME "${TARGET_LIBRARY}")

  if(_GEH_BASE_NAME)
    set(BASE_NAME ${_GEH_BASE_NAME})
  endif()

  string(TOUPPER ${BASE_NAME} BASE_NAME_UPPER)
  string(TOLOWER ${BASE_NAME} BASE_NAME_LOWER)

  # Default options
  set(EXPORT_MACRO_NAME "${_GEH_PREFIX_NAME}${BASE_NAME_UPPER}_EXPORT")
  set(NO_EXPORT_MACRO_NAME "${_GEH_PREFIX_NAME}${BASE_NAME_UPPER}_NO_EXPORT")
  set(EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/${BASE_NAME_LOWER}_export.h")
  set(DEPRECATED_MACRO_NAME "${_GEH_PREFIX_NAME}${BASE_NAME_UPPER}_DEPRECATED")
  set(STATIC_DEFINE "${_GEH_PREFIX_NAME}${BASE_NAME_UPPER}_STATIC_DEFINE")
  set(NO_DEPRECATED_MACRO_NAME
    "${_GEH_PREFIX_NAME}${BASE_NAME_UPPER}_NO_DEPRECATED")

  if(_GEH_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unknown keywords given to generate_export_header(): \"${_GEH_UNPARSED_ARGUMENTS}\"")
  endif()

  if(_GEH_EXPORT_MACRO_NAME)
    set(EXPORT_MACRO_NAME ${_GEH_PREFIX_NAME}${_GEH_EXPORT_MACRO_NAME})
  endif()
  string(MAKE_C_IDENTIFIER ${EXPORT_MACRO_NAME} EXPORT_MACRO_NAME)
  if(_GEH_EXPORT_FILE_NAME)
    if(IS_ABSOLUTE ${_GEH_EXPORT_FILE_NAME})
      set(EXPORT_FILE_NAME ${_GEH_EXPORT_FILE_NAME})
    else()
      set(EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/${_GEH_EXPORT_FILE_NAME}")
    endif()
  endif()
  if(_GEH_DEPRECATED_MACRO_NAME)
    set(DEPRECATED_MACRO_NAME ${_GEH_PREFIX_NAME}${_GEH_DEPRECATED_MACRO_NAME})
  endif()
  string(MAKE_C_IDENTIFIER ${DEPRECATED_MACRO_NAME} DEPRECATED_MACRO_NAME)
  if(_GEH_NO_EXPORT_MACRO_NAME)
    set(NO_EXPORT_MACRO_NAME ${_GEH_PREFIX_NAME}${_GEH_NO_EXPORT_MACRO_NAME})
  endif()
  string(MAKE_C_IDENTIFIER ${NO_EXPORT_MACRO_NAME} NO_EXPORT_MACRO_NAME)
  if(_GEH_STATIC_DEFINE)
    set(STATIC_DEFINE ${_GEH_PREFIX_NAME}${_GEH_STATIC_DEFINE})
  endif()
  string(MAKE_C_IDENTIFIER ${STATIC_DEFINE} STATIC_DEFINE)

  if(_GEH_DEFINE_NO_DEPRECATED)
    set(DEFINE_NO_DEPRECATED 1)
  else()
    set(DEFINE_NO_DEPRECATED 0)
  endif()

  if(_GEH_NO_DEPRECATED_MACRO_NAME)
    set(NO_DEPRECATED_MACRO_NAME
      ${_GEH_PREFIX_NAME}${_GEH_NO_DEPRECATED_MACRO_NAME})
  endif()
  string(MAKE_C_IDENTIFIER ${NO_DEPRECATED_MACRO_NAME} NO_DEPRECATED_MACRO_NAME)

  if(_GEH_INCLUDE_GUARD_NAME)
    set(INCLUDE_GUARD_NAME ${_GEH_INCLUDE_GUARD_NAME})
  else()
    set(INCLUDE_GUARD_NAME "${EXPORT_MACRO_NAME}_H")
  endif()

  get_target_property(EXPORT_IMPORT_CONDITION ${TARGET_LIBRARY} DEFINE_SYMBOL)

  if(NOT EXPORT_IMPORT_CONDITION)
    set(EXPORT_IMPORT_CONDITION ${TARGET_LIBRARY}_EXPORTS)
  endif()
  string(MAKE_C_IDENTIFIER ${EXPORT_IMPORT_CONDITION} EXPORT_IMPORT_CONDITION)

  if(_GEH_CUSTOM_CONTENT_FROM_VARIABLE)
    if(DEFINED "${_GEH_CUSTOM_CONTENT_FROM_VARIABLE}")
      set(CUSTOM_CONTENT "${${_GEH_CUSTOM_CONTENT_FROM_VARIABLE}}")
    else()
      set(CUSTOM_CONTENT "")
    endif()
  endif()

  configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/exportheader.cmake.in"
    "${EXPORT_FILE_NAME}" @ONLY)
endfunction()

function(GENERATE_EXPORT_HEADER TARGET_LIBRARY)
  get_property(type TARGET ${TARGET_LIBRARY} PROPERTY TYPE)
  if(NOT ${type} STREQUAL "STATIC_LIBRARY"
      AND NOT ${type} STREQUAL "SHARED_LIBRARY"
      AND NOT ${type} STREQUAL "OBJECT_LIBRARY"
      AND NOT ${type} STREQUAL "MODULE_LIBRARY")
    message(WARNING "This macro can only be used with libraries")
    return()
  endif()
  _test_compiler_hidden_visibility()
  _test_compiler_has_deprecated()
  _do_set_macro_values(${TARGET_LIBRARY})
  _do_generate_export_header(${TARGET_LIBRARY} ${ARGN})
endfunction()

function(add_compiler_export_flags)
  if(NOT CMAKE_MINIMUM_REQUIRED_VERSION VERSION_LESS 2.8.12)
    message(DEPRECATION "The add_compiler_export_flags function is obsolete. Use the CXX_VISIBILITY_PRESET and VISIBILITY_INLINES_HIDDEN target properties instead.")
  endif()

  _test_compiler_hidden_visibility()
  _test_compiler_has_deprecated()

  option(USE_COMPILER_HIDDEN_VISIBILITY
    "Use HIDDEN visibility support if available." ON)
  mark_as_advanced(USE_COMPILER_HIDDEN_VISIBILITY)
  if(NOT (USE_COMPILER_HIDDEN_VISIBILITY AND COMPILER_HAS_HIDDEN_VISIBILITY))
    # Just return if there are no flags to add.
    return()
  endif()

  set (EXTRA_FLAGS "-fvisibility=hidden")

  if(COMPILER_HAS_HIDDEN_INLINE_VISIBILITY)
    set (EXTRA_FLAGS "${EXTRA_FLAGS} -fvisibility-inlines-hidden")
  endif()

  # Either return the extra flags needed in the supplied argument, or to the
  # CMAKE_CXX_FLAGS if no argument is supplied.
  if(ARGC GREATER 0)
    set(${ARGV0} "${EXTRA_FLAGS}" PARENT_SCOPE)
  else()
    string(APPEND CMAKE_CXX_FLAGS " ${EXTRA_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" PARENT_SCOPE)
  endif()
endfunction()

# FIXME(#24994): The following module(s) are included only for compatibility
# with projects that accidentally relied on them with CMake 3.26 and below.
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
