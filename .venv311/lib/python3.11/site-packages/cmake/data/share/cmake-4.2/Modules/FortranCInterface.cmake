# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FortranCInterface
-----------------

This module provides variables and commands to detect Fortran/C Interface.

Load this module in a CMake project with:

.. code-block:: cmake

  include(FortranCInterface)

This module automatically detects the API by which C and Fortran languages
interact.

Variables
^^^^^^^^^

Result Variables
""""""""""""""""

Including this module defines the following variables that indicate if the
mangling is found:

``FortranCInterface_GLOBAL_FOUND``
  Boolean indicating whether global subroutines and functions are available.

``FortranCInterface_MODULE_FOUND``
  Boolean indicating whether module subroutines and functions (declared by
  ``MODULE PROCEDURE``) are available.

Input Variables
"""""""""""""""

This module also provides the following variables to specify
the detected mangling, though a typical use case does not need
to reference them and can use the `Commands`_ below.

``FortranCInterface_GLOBAL_PREFIX``
  Prefix for a global symbol without an underscore.

``FortranCInterface_GLOBAL_SUFFIX``
  Suffix for a global symbol without an underscore.

``FortranCInterface_GLOBAL_CASE``
  The case for a global symbol without an underscore,
  either ``UPPER`` or ``LOWER``.

``FortranCInterface_GLOBAL__PREFIX``
  Prefix for a global symbol with an underscore.

``FortranCInterface_GLOBAL__SUFFIX``
  Suffix for a global symbol with an underscore.

``FortranCInterface_GLOBAL__CASE``
  The case for a global symbol with an underscore,
  either ``UPPER`` or ``LOWER``.

``FortranCInterface_MODULE_PREFIX``
  Prefix for a module symbol without an underscore.

``FortranCInterface_MODULE_MIDDLE``
  Middle of a module symbol without an underscore that appears
  between the name of the module and the name of the symbol.

``FortranCInterface_MODULE_SUFFIX``
  Suffix for a module symbol without an underscore.

``FortranCInterface_MODULE_CASE``
  The case for a module symbol without an underscore,
  either ``UPPER`` or ``LOWER``.

``FortranCInterface_MODULE_ORDER``
  .. versionadded:: 4.1

  Order of components for module symbols without an underscore:

  ``MODULE_THEN_SYMBOL``
    The module name appears *before* the symbol name, i.e.,
    ``<PREFIX><module><MIDDLE><symbol><SUFFIX>``.

  ``SYMBOL_THEN_MODULE``
    The module name appears *after* the symbol name, i.e.,
    ``<PREFIX><symbol><MIDDLE><module><SUFFIX>``.

``FortranCInterface_MODULE__PREFIX``
  Prefix for a module symbol with an underscore.

``FortranCInterface_MODULE__MIDDLE``
  Middle of a module symbol with an underscore that appears
  between the name of the module and the name of the symbol.

``FortranCInterface_MODULE__SUFFIX``
  Suffix for a module symbol with an underscore.

``FortranCInterface_MODULE__CASE``
  The case for a module symbol with an underscore,
  either ``UPPER`` or ``LOWER``.

``FortranCInterface_MODULE__ORDER``
  .. versionadded:: 4.1

  Order of components for module symbols with an underscore:

  ``MODULE_THEN_SYMBOL``
    The module name appears *before* the symbol name, i.e.,
    ``<PREFIX><module><MIDDLE><symbol><SUFFIX>``.

  ``SYMBOL_THEN_MODULE``
    The module name appears *after* the symbol name, i.e.,
    ``<PREFIX><symbol><MIDDLE><module><SUFFIX>``.

Variables For Additional Manglings
""""""""""""""""""""""""""""""""""

This module is aware of possible ``GLOBAL`` and ``MODULE`` manglings for
many Fortran compilers, but it also provides an interface to specify
new possible manglings.  The following variables can be set before including
this module to specify additional manglings:

``FortranCInterface_GLOBAL_SYMBOLS``

``FortranCInterface_MODULE_SYMBOLS``

before including this module to specify manglings of the symbols
``MySub``, ``My_Sub``, ``MyModule:MySub``, and ``My_Module:My_Sub``.

Commands
^^^^^^^^

This module provides the following commands:

.. command:: FortranCInterface_HEADER

  Generates a C header file containing macros to mangle symbol names:

  .. code-block:: cmake

    FortranCInterface_HEADER(
      <file>
      [MACRO_NAMESPACE <macro-ns>]
      [SYMBOL_NAMESPACE <ns>]
      [SYMBOLS [<module>:]<function> ...]
    )

  This command generates a ``<file>`` with definitions of the following
  macros:

  .. code-block:: c

    #define FortranCInterface_GLOBAL (name,NAME) ...
    #define FortranCInterface_GLOBAL_(name,NAME) ...
    #define FortranCInterface_MODULE (mod,name, MOD,NAME) ...
    #define FortranCInterface_MODULE_(mod,name, MOD,NAME) ...

  These macros mangle four categories of Fortran symbols, respectively:

  * Global symbols without '_': ``call mysub()``
  * Global symbols with '_'   : ``call my_sub()``
  * Module symbols without '_': ``use mymod; call mysub()``
  * Module symbols with '_'   : ``use mymod; call my_sub()``

  If mangling for a category is not known, its macro is left undefined.
  All macros require raw names in both lower case and upper case.

  The options are:

  ``MACRO_NAMESPACE``
    Replace the default ``FortranCInterface_`` prefix with a given
    namespace ``<macro-ns>``.

  ``SYMBOL_NAMESPACE``
    Prefix all preprocessor definitions generated by the ``SYMBOLS``
    option with a given namespace ``<ns>``.

  ``SYMBOLS``
    List symbols to mangle automatically with C preprocessor definitions::

      <function>          ==> #define <ns><function> ...
      <module>:<function> ==> #define <ns><module>_<function> ...

    If the mangling for some symbol is not known then no preprocessor
    definition is created, and a warning is displayed.

.. command:: FortranCInterface_VERIFY

  Verifies that the Fortran and C/C++ compilers work together:

  .. code-block:: cmake

    FortranCInterface_VERIFY([CXX] [QUIET])

  This command tests whether a simple test executable using Fortran and C
  (and C++ when the ``CXX`` option is given) compiles and links successfully.
  The result is stored in the cache entry ``FortranCInterface_VERIFIED_C``
  (or ``FortranCInterface_VERIFIED_CXX`` if ``CXX`` is given) as a boolean.
  If the check fails and ``QUIET`` is not given the command terminates with a
  fatal error message describing the problem.  The purpose of this check
  is to stop a build early for incompatible compiler combinations.  The
  test is built in the ``Release`` configuration.

Examples
^^^^^^^^

Examples: Basic Usage
"""""""""""""""""""""

The following example creates a ``FC.h`` header that defines mangling macros
``FC_GLOBAL()``, ``FC_GLOBAL_()``, ``FC_MODULE()``, and ``FC_MODULE_()``:

.. code-block:: cmake

  include(FortranCInterface)
  FortranCInterface_HEADER(FC.h MACRO_NAMESPACE "FC_")

The next example creates a ``FCMangle.h`` header that defines the same
``FC_*()`` mangling macros as the previous example plus preprocessor symbols
``FC_mysub`` and ``FC_mymod_my_sub``:

.. code-block:: cmake

  include(FortranCInterface)
  FortranCInterface_HEADER(
    FCMangle.h
    MACRO_NAMESPACE "FC_"
    SYMBOL_NAMESPACE "FC_"
    SYMBOLS mysub mymod:my_sub
  )

Example: Additional Manglings
"""""""""""""""""""""""""""""

The following example shows how to specify manglings of the symbols
``MySub``, ``My_Sub``, ``MyModule:MySub``, and ``My_Module:My_Sub``.
The following code tells this module to try given ``GLOBAL`` and ``MODULE``
manglings.  (The carets point at raw symbol names for clarity in this
example but are not needed.)

.. code-block:: cmake

  set(FortranCInterface_GLOBAL_SYMBOLS mysub_ my_sub__ MYSUB_)
    #                                  ^^^^^  ^^^^^^   ^^^^^
  set(FortranCInterface_MODULE_SYMBOLS
      __mymodule_MOD_mysub __my_module_MOD_my_sub)
    #   ^^^^^^^^     ^^^^^   ^^^^^^^^^     ^^^^^^

  include(FortranCInterface)

  # ...
#]=======================================================================]

#-----------------------------------------------------------------------------
# Execute at most once in a project.
if(FortranCInterface_SOURCE_DIR)
  return()
endif()

#-----------------------------------------------------------------------------
# Verify that C and Fortran are available.
foreach(lang C Fortran)
  if(NOT CMAKE_${lang}_COMPILER_LOADED)
    message(FATAL_ERROR
      "FortranCInterface requires the ${lang} language to be enabled.")
  endif()
endforeach()

#-----------------------------------------------------------------------------
set(FortranCInterface_SOURCE_DIR ${CMAKE_ROOT}/Modules/FortranCInterface)

# MinGW's make tool does not always like () in the path
if("${CMAKE_GENERATOR}" MATCHES "MinGW" AND
    "${FortranCInterface_SOURCE_DIR}" MATCHES "[()]")
  file(COPY ${FortranCInterface_SOURCE_DIR}/
    DESTINATION ${CMAKE_BINARY_DIR}/CMakeFiles/FortranCInterfaceMinGW)
  set(FortranCInterface_SOURCE_DIR ${CMAKE_BINARY_DIR}/CMakeFiles/FortranCInterfaceMinGW)
endif()

# Create the interface detection project if it does not exist.
if(NOT FortranCInterface_BINARY_DIR)
  set(FortranCInterface_BINARY_DIR ${CMAKE_BINARY_DIR}/CMakeFiles/FortranCInterface)
  include(${FortranCInterface_SOURCE_DIR}/Detect.cmake)
endif()

# Load the detection results.
include(${FortranCInterface_BINARY_DIR}/Output.cmake)

#-----------------------------------------------------------------------------
function(FortranCInterface_HEADER file)
  # Parse arguments.
  if(IS_ABSOLUTE "${file}")
    set(FILE "${file}")
  else()
    set(FILE "${CMAKE_CURRENT_BINARY_DIR}/${file}")
  endif()
  set(MACRO_NAMESPACE "FortranCInterface_")
  set(SYMBOL_NAMESPACE)
  set(SYMBOLS)
  set(doing)
  foreach(arg ${ARGN})
    if("x${arg}" MATCHES "^x(SYMBOLS|SYMBOL_NAMESPACE|MACRO_NAMESPACE)$")
      set(doing "${arg}")
    elseif("x${doing}" MATCHES "^x(SYMBOLS)$")
      list(APPEND "${doing}" "${arg}")
    elseif("x${doing}" MATCHES "^x(SYMBOL_NAMESPACE|MACRO_NAMESPACE)$")
      set("${doing}" "${arg}")
      set(doing)
    else()
      message(AUTHOR_WARNING "Unknown argument: \"${arg}\"")
    endif()
  endforeach()

  # Generate macro definitions.
  set(HEADER_CONTENT)
  set(_desc_GLOBAL  "/* Mangling for Fortran global symbols without underscores. */")
  set(_desc_GLOBAL_ "/* Mangling for Fortran global symbols with underscores. */")
  set(_desc_MODULE  "/* Mangling for Fortran module symbols without underscores. */")
  set(_desc_MODULE_ "/* Mangling for Fortran module symbols with underscores. */")
  foreach(macro GLOBAL GLOBAL_ MODULE MODULE_)
    if(FortranCInterface_${macro}_MACRO)
      string(APPEND HEADER_CONTENT "
${_desc_${macro}}
#define ${MACRO_NAMESPACE}${macro}${FortranCInterface_${macro}_MACRO}
")
    endif()
  endforeach()

  # Generate symbol mangling definitions.
  if(SYMBOLS)
    string(APPEND HEADER_CONTENT "
/*--------------------------------------------------------------------------*/
/* Mangle some symbols automatically.                                       */
")
  endif()
  foreach(f ${SYMBOLS})
    if("${f}" MATCHES ":")
      # Module symbol name.  Parse "<module>:<function>" syntax.
      string(REPLACE ":" ";" pieces "${f}")
      list(GET pieces 0 module)
      list(GET pieces 1 function)
      string(TOUPPER "${module}" m_upper)
      string(TOLOWER "${module}" m_lower)
      string(TOUPPER "${function}" f_upper)
      string(TOLOWER "${function}" f_lower)
      if("${function}" MATCHES "_")
        set(form "_")
      else()
        set(form "")
      endif()
      if(FortranCInterface_MODULE${form}_MACRO)
        string(APPEND HEADER_CONTENT "#define ${SYMBOL_NAMESPACE}${module}_${function} ${MACRO_NAMESPACE}MODULE${form}(${m_lower},${f_lower}, ${m_upper},${f_upper})\n")
      else()
        message(AUTHOR_WARNING "No FortranCInterface mangling known for ${f}")
      endif()
    else()
      # Global symbol name.
      if("${f}" MATCHES "_")
        set(form "_")
      else()
        set(form "")
      endif()
      string(TOUPPER "${f}" f_upper)
      string(TOLOWER "${f}" f_lower)
      if(FortranCInterface_GLOBAL${form}_MACRO)
        string(APPEND HEADER_CONTENT "#define ${SYMBOL_NAMESPACE}${f} ${MACRO_NAMESPACE}GLOBAL${form}(${f_lower}, ${f_upper})\n")
      else()
        message(AUTHOR_WARNING "No FortranCInterface mangling known for ${f}")
      endif()
    endif()
  endforeach()

  # Store the content.
  configure_file(${FortranCInterface_SOURCE_DIR}/Macro.h.in ${FILE} @ONLY)
endfunction()

function(FortranCInterface_VERIFY)
  # Check arguments.

  set(lang C)
  set(quiet 0)
  set(verify_cxx 0)
  foreach(arg ${ARGN})
    if("${arg}" STREQUAL "QUIET")
      set(quiet 1)
    elseif("${arg}" STREQUAL "CXX")
      set(lang CXX)
      set(verify_cxx 1)
    else()
      message(FATAL_ERROR
        "FortranCInterface_VERIFY - called with unknown argument:\n  ${arg}")
    endif()
  endforeach()

  if(NOT CMAKE_${lang}_COMPILER_LOADED)
    message(FATAL_ERROR
      "FortranCInterface_VERIFY(${lang}) requires ${lang} to be enabled.")
  endif()

  # Build the verification project if not yet built.
  if(NOT DEFINED FortranCInterface_VERIFIED_${lang})
    set(_desc "Verifying Fortran/${lang} Compiler Compatibility")
    message(CHECK_START "${_desc}")

    # Perform verification with only one architecture.
    # FIXME: Add try_compile whole-project option to forward architectures.
    if(CMAKE_OSX_ARCHITECTURES MATCHES "^([^;]+)(;|$)")
      set(_FortranCInterface_OSX_ARCH "-DCMAKE_OSX_ARCHITECTURES=${CMAKE_MATCH_1}")
    else()
      set(_FortranCInterface_OSX_ARCH "")
    endif()

    set(_FortranCInterface_EXE_LINKER_FLAGS "-DCMAKE_EXE_LINKER_FLAGS:STRING=${CMAKE_EXE_LINKER_FLAGS}")

    # Build a sample project which reports symbols.
    set(CMAKE_TRY_COMPILE_CONFIGURATION Release)
    try_compile(FortranCInterface_VERIFY_${lang}_COMPILED
      PROJECT VerifyFortranC
      TARGET VerifyFortranC
      SOURCE_DIR ${FortranCInterface_SOURCE_DIR}/Verify
      BINARY_DIR ${FortranCInterface_BINARY_DIR}/Verify${lang}
      CMAKE_FLAGS -DVERIFY_CXX=${verify_cxx}
                  -DCMAKE_VERBOSE_MAKEFILE=ON
                 "-DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}"
                 "-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}"
                 "-DCMAKE_Fortran_FLAGS:STRING=${CMAKE_Fortran_FLAGS}"
                 "-DCMAKE_C_FLAGS_RELEASE:STRING=${CMAKE_C_FLAGS_RELEASE}"
                 "-DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}"
                 "-DCMAKE_Fortran_FLAGS_RELEASE:STRING=${CMAKE_Fortran_FLAGS_RELEASE}"
                 "-DFortranCInterface_BINARY_DIR=${FortranCInterface_BINARY_DIR}"
                 "-DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}"
                 ${_FortranCInterface_OSX_ARCH}
                 ${_FortranCInterface_EXE_LINKER_FLAGS}
      OUTPUT_VARIABLE _output)
    file(WRITE "${FortranCInterface_BINARY_DIR}/Verify${lang}/output.txt" "${_output}")

    # Report results.
    if(FortranCInterface_VERIFY_${lang}_COMPILED)
      message(CHECK_PASS "Success")
      set(FortranCInterface_VERIFIED_${lang} 1 CACHE INTERNAL "Fortran/${lang} compatibility")
    else()
      message(CHECK_FAIL "Failed")
      set(FortranCInterface_VERIFIED_${lang} 0 CACHE INTERNAL "Fortran/${lang} compatibility")
    endif()
    unset(FortranCInterface_VERIFY_${lang}_COMPILED CACHE)
  endif()

  # Error if compilers are incompatible.
  if(NOT FortranCInterface_VERIFIED_${lang} AND NOT quiet)
    file(READ "${FortranCInterface_BINARY_DIR}/Verify${lang}/output.txt" _output)
    string(REPLACE "\n" "\n  " _output "${_output}")
    message(FATAL_ERROR
      "The Fortran compiler:\n  ${CMAKE_Fortran_COMPILER}\n"
      "and the ${lang} compiler:\n  ${CMAKE_${lang}_COMPILER}\n"
      "failed to compile a simple test project using both languages.  "
      "The output was:\n  ${_output}")
  endif()
endfunction()
