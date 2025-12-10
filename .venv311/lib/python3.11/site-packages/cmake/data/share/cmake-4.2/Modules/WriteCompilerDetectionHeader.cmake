# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
WriteCompilerDetectionHeader
----------------------------

.. deprecated:: 3.20
  This module is available only if policy :policy:`CMP0120`
  is not set to ``NEW``.  Do not use it in new code.

.. versionadded:: 3.1

This module provides a command to generate header with preprocessor macros.

Load this module in a CMake project with:

.. code-block:: cmake

  include(WriteCompilerDetectionHeader)

Commands
^^^^^^^^

This module provides the following command:

.. command:: write_compiler_detection_header

This function can be used to generate a file suitable for preprocessor
inclusion which contains macros to be used in source code:

.. code-block:: cmake

   write_compiler_detection_header(
             FILE <file>
             PREFIX <prefix>
             [OUTPUT_FILES_VAR <output_files_var> OUTPUT_DIR <output_dir>]
             COMPILERS <compiler> [...]
             FEATURES <feature> [...]
             [BARE_FEATURES <feature> [...]]
             [VERSION <version>]
             [PROLOG <prolog>]
             [EPILOG <epilog>]
             [ALLOW_UNKNOWN_COMPILERS]
             [ALLOW_UNKNOWN_COMPILER_VERSIONS]
   )

This generates the file ``<file>`` with macros which all have the prefix
``<prefix>``.

By default, all content is written directly to the ``<file>``.  The
``OUTPUT_FILES_VAR`` may be specified to cause the compiler-specific
content to be written to separate files.  The separate files are then
available in the ``<output_files_var>`` and may be consumed by the caller
for installation for example.  The ``OUTPUT_DIR`` specifies a relative
path from the main ``<file>`` to the compiler-specific files. For example:

.. code-block:: cmake

   write_compiler_detection_header(
     FILE climbingstats_compiler_detection.h
     PREFIX ClimbingStats
     OUTPUT_FILES_VAR support_files
     OUTPUT_DIR compilers
     COMPILERS GNU Clang MSVC Intel
     FEATURES cxx_variadic_templates
   )
   install(FILES
     ${CMAKE_CURRENT_BINARY_DIR}/climbingstats_compiler_detection.h
     DESTINATION include
   )
   install(FILES
     ${support_files}
     DESTINATION include/compilers
   )


``VERSION`` may be used to specify the API version to be generated.
Future versions of CMake may introduce alternative APIs.  A given
API is selected by any ``<version>`` value greater than or equal
to the version of CMake that introduced the given API and less
than the version of CMake that introduced its succeeding API.
The value of the :variable:`CMAKE_MINIMUM_REQUIRED_VERSION`
variable is used if no explicit version is specified.
(As of CMake version |release| there is only one API version.)

``PROLOG`` may be specified as text content to write at the start of the
header. ``EPILOG`` may be specified as text content to write at the end
of the header

At least one ``<compiler>`` and one ``<feature>`` must be listed.  Compilers
which are known to CMake, but not specified are detected and a preprocessor
``#error`` is generated for them.  A preprocessor macro matching
``<PREFIX>_COMPILER_IS_<compiler>`` is generated for each compiler
known to CMake to contain the value ``0`` or ``1``.

Possible compiler identifiers are documented with the
:variable:`CMAKE_<LANG>_COMPILER_ID` variable.
Available features in this version of CMake are listed in the
:prop_gbl:`CMAKE_C_KNOWN_FEATURES` and
:prop_gbl:`CMAKE_CXX_KNOWN_FEATURES` global properties.
See the :manual:`cmake-compile-features(7)` manual for information on
compile features.

.. versionadded:: 3.2
  Added ``MSVC`` and ``AppleClang`` compiler support.

.. versionadded:: 3.6
  Added ``Intel`` compiler support.

.. versionchanged:: 3.8
  The ``{c,cxx}_std_*`` meta-features are ignored if requested.

.. versionadded:: 3.8
  ``ALLOW_UNKNOWN_COMPILERS`` and ``ALLOW_UNKNOWN_COMPILER_VERSIONS`` cause
  the module to generate conditions that treat unknown compilers as simply
  lacking all features.  Without these options the default behavior is to
  generate a ``#error`` for unknown compilers and versions.

.. versionadded:: 3.12
  ``BARE_FEATURES`` will define the compatibility macros with the name used in
  newer versions of the language standard, so the code can use the new feature
  name unconditionally.

Feature Test Macros
===================

For each compiler, a preprocessor macro is generated matching
``<PREFIX>_COMPILER_IS_<compiler>`` which has the content either ``0``
or ``1``, depending on the compiler in use. Preprocessor macros for
compiler version components are generated matching
``<PREFIX>_COMPILER_VERSION_MAJOR`` ``<PREFIX>_COMPILER_VERSION_MINOR``
and ``<PREFIX>_COMPILER_VERSION_PATCH`` containing decimal values
for the corresponding compiler version components, if defined.

A preprocessor test is generated based on the compiler version
denoting whether each feature is enabled.  A preprocessor macro
matching ``<PREFIX>_COMPILER_<FEATURE>``, where ``<FEATURE>`` is the
upper-case ``<feature>`` name, is generated to contain the value
``0`` or ``1`` depending on whether the compiler in use supports the
feature:

.. code-block:: cmake

   write_compiler_detection_header(
     FILE climbingstats_compiler_detection.h
     PREFIX ClimbingStats
     COMPILERS GNU Clang AppleClang MSVC Intel
     FEATURES cxx_variadic_templates
   )

.. code-block:: c++

   #if ClimbingStats_COMPILER_CXX_VARIADIC_TEMPLATES
   template<typename... T>
   void someInterface(T t...) { /* ... */ }
   #else
   // Compatibility versions
   template<typename T1>
   void someInterface(T1 t1) { /* ... */ }
   template<typename T1, typename T2>
   void someInterface(T1 t1, T2 t2) { /* ... */ }
   template<typename T1, typename T2, typename T3>
   void someInterface(T1 t1, T2 t2, T3 t3) { /* ... */ }
   #endif

Symbol Macros
=============

Some additional symbol-defines are created for particular features for
use as symbols which may be conditionally defined empty:

.. code-block:: c++

   class MyClass ClimbingStats_FINAL
   {
       ClimbingStats_CONSTEXPR int someInterface() { return 42; }
   };

The ``ClimbingStats_FINAL`` macro will expand to ``final`` if the
compiler (and its flags) support the ``cxx_final`` feature, and the
``ClimbingStats_CONSTEXPR`` macro will expand to ``constexpr``
if ``cxx_constexpr`` is supported.

If ``BARE_FEATURES cxx_final`` was given as argument the ``final`` keyword
will be defined for old compilers, too.

The following features generate corresponding symbol defines and if they
are available as ``BARE_FEATURES``:

========================== =================================== ================= ======
        Feature                          Define                      Symbol       bare
========================== =================================== ================= ======
``c_restrict``              ``<PREFIX>_RESTRICT``               ``restrict``      yes
``cxx_constexpr``           ``<PREFIX>_CONSTEXPR``              ``constexpr``     yes
``cxx_deleted_functions``   ``<PREFIX>_DELETED_FUNCTION``       ``= delete``
``cxx_extern_templates``    ``<PREFIX>_EXTERN_TEMPLATE``        ``extern``
``cxx_final``               ``<PREFIX>_FINAL``                  ``final``         yes
``cxx_noexcept``            ``<PREFIX>_NOEXCEPT``               ``noexcept``      yes
``cxx_noexcept``            ``<PREFIX>_NOEXCEPT_EXPR(X)``       ``noexcept(X)``
``cxx_override``            ``<PREFIX>_OVERRIDE``               ``override``      yes
========================== =================================== ================= ======

Compatibility Implementation Macros
===================================

Some features are suitable for wrapping in a macro with a backward
compatibility implementation if the compiler does not support the feature.

When the ``cxx_static_assert`` feature is not provided by the compiler,
a compatibility implementation is available via the
``<PREFIX>_STATIC_ASSERT(COND)`` and
``<PREFIX>_STATIC_ASSERT_MSG(COND, MSG)`` function-like macros. The macros
expand to ``static_assert`` where that compiler feature is available, and
to a compatibility implementation otherwise. In the first form, the
condition is stringified in the message field of ``static_assert``.  In
the second form, the message ``MSG`` is passed to the message field of
``static_assert``, or ignored if using the backward compatibility
implementation.

The ``cxx_attribute_deprecated`` feature provides a macro definition
``<PREFIX>_DEPRECATED``, which expands to either the standard
``[[deprecated]]`` attribute or a compiler-specific decorator such
as ``__attribute__((__deprecated__))`` used by GNU compilers.

The ``cxx_alignas`` feature provides a macro definition
``<PREFIX>_ALIGNAS`` which expands to either the standard ``alignas``
decorator or a compiler-specific decorator such as
``__attribute__ ((__aligned__))`` used by GNU compilers.

The ``cxx_alignof`` feature provides a macro definition
``<PREFIX>_ALIGNOF`` which expands to either the standard ``alignof``
decorator or a compiler-specific decorator such as ``__alignof__``
used by GNU compilers.

============================= ================================ ===================== ======
          Feature                          Define                     Symbol          bare
============================= ================================ ===================== ======
``cxx_alignas``                ``<PREFIX>_ALIGNAS``             ``alignas``
``cxx_alignof``                ``<PREFIX>_ALIGNOF``             ``alignof``
``cxx_nullptr``                ``<PREFIX>_NULLPTR``             ``nullptr``           yes
``cxx_static_assert``          ``<PREFIX>_STATIC_ASSERT``       ``static_assert``
``cxx_static_assert``          ``<PREFIX>_STATIC_ASSERT_MSG``   ``static_assert``
``cxx_attribute_deprecated``   ``<PREFIX>_DEPRECATED``          ``[[deprecated]]``
``cxx_attribute_deprecated``   ``<PREFIX>_DEPRECATED_MSG``      ``[[deprecated]]``
``cxx_thread_local``           ``<PREFIX>_THREAD_LOCAL``        ``thread_local``
============================= ================================ ===================== ======

A use-case which arises with such deprecation macros is the deprecation
of an entire library.  In that case, all public API in the library may
be decorated with the ``<PREFIX>_DEPRECATED`` macro.  This results in
very noisy build output when building the library itself, so the macro
may be may be defined to empty in that case when building the deprecated
library:

.. code-block:: cmake

  add_library(compat_support ${srcs})
  target_compile_definitions(compat_support
    PRIVATE
      CompatSupport_DEPRECATED=
  )

.. _`WCDH Example Usage`:

Example Usage
=============

.. note::

  This section was migrated from the :manual:`cmake-compile-features(7)`
  manual since it relies on the ``WriteCompilerDetectionHeader`` module
  which is removed by policy :policy:`CMP0120`.

Compile features may be preferred if available, without creating a hard
requirement.  For example, a library may provide alternative
implementations depending on whether the ``cxx_variadic_templates``
feature is available:

.. code-block:: c++

  #if Foo_COMPILER_CXX_VARIADIC_TEMPLATES
  template<int I, int... Is>
  struct Interface;

  template<int I>
  struct Interface<I>
  {
    static int accumulate()
    {
      return I;
    }
  };

  template<int I, int... Is>
  struct Interface
  {
    static int accumulate()
    {
      return I + Interface<Is...>::accumulate();
    }
  };
  #else
  template<int I1, int I2 = 0, int I3 = 0, int I4 = 0>
  struct Interface
  {
    static int accumulate() { return I1 + I2 + I3 + I4; }
  };
  #endif

Such an interface depends on using the correct preprocessor defines for the
compiler features.  CMake can generate a header file containing such
defines using the :module:`WriteCompilerDetectionHeader` module.  The
module contains the ``write_compiler_detection_header`` function which
accepts parameters to control the content of the generated header file:

.. code-block:: cmake

  write_compiler_detection_header(
    FILE "${CMAKE_CURRENT_BINARY_DIR}/foo_compiler_detection.h"
    PREFIX Foo
    COMPILERS GNU
    FEATURES
      cxx_variadic_templates
  )

Such a header file may be used internally in the source code of a project,
and it may be installed and used in the interface of library code.

For each feature listed in ``FEATURES``, a preprocessor definition
is created in the header file, and defined to either ``1`` or ``0``.

Additionally, some features call for additional defines, such as the
``cxx_final`` and ``cxx_override`` features. Rather than being used in
``#ifdef`` code, the ``final`` keyword is abstracted by a symbol
which is defined to either ``final``, a compiler-specific equivalent, or
to empty.  That way, C++ code can be written to unconditionally use the
symbol, and compiler support determines what it is expanded to:

.. code-block:: c++

  struct Interface {
    virtual void Execute() = 0;
  };

  struct Concrete Foo_FINAL {
    void Execute() Foo_OVERRIDE;
  };

In this case, ``Foo_FINAL`` will expand to ``final`` if the
compiler supports the keyword, or to empty otherwise.

In this use-case, the project code may wish to enable a particular language
standard if available from the compiler. The :prop_tgt:`CXX_STANDARD`
target property may be set to the desired language standard for a particular
target, and the :variable:`CMAKE_CXX_STANDARD` variable may be set to
influence all following targets:

.. code-block:: cmake

  write_compiler_detection_header(
    FILE "${CMAKE_CURRENT_BINARY_DIR}/foo_compiler_detection.h"
    PREFIX Foo
    COMPILERS GNU
    FEATURES
      cxx_final cxx_override
  )

  # Includes foo_compiler_detection.h and uses the Foo_FINAL symbol
  # which will expand to 'final' if the compiler supports the requested
  # CXX_STANDARD.
  add_library(foo foo.cpp)
  set_property(TARGET foo PROPERTY CXX_STANDARD 11)

  # Includes foo_compiler_detection.h and uses the Foo_FINAL symbol
  # which will expand to 'final' if the compiler supports the feature,
  # even though CXX_STANDARD is not set explicitly.  The requirement of
  # cxx_constexpr causes CMake to set CXX_STANDARD internally, which
  # affects the compile flags.
  add_library(foo_impl foo_impl.cpp)
  target_compile_features(foo_impl PRIVATE cxx_constexpr)

The ``write_compiler_detection_header`` function also creates compatibility
code for other features which have standard equivalents.  For example, the
``cxx_static_assert`` feature is emulated with a template and abstracted
via the ``<PREFIX>_STATIC_ASSERT`` and ``<PREFIX>_STATIC_ASSERT_MSG``
function-macros.
#]=======================================================================]

# Guard against inclusion by absolute path.
cmake_policy(GET CMP0120 _WCDH_policy)
if(_WCDH_policy STREQUAL "NEW")
  message(FATAL_ERROR "The WriteCompilerDetectionHeader module has been removed by policy CMP0120.")
elseif(_WCDH_policy STREQUAL "")
  message(AUTHOR_WARNING
    "The WriteCompilerDetectionHeader module will be removed by policy CMP0120.  "
    "Projects should be ported away from the module, perhaps by bundling a copy "
    "of the generated header or using a third-party alternative."
    )
endif()

include(${CMAKE_CURRENT_LIST_DIR}/CMakeCompilerIdDetection.cmake)

function(_load_compiler_variables CompilerId lang)
  include("${CMAKE_ROOT}/Modules/Compiler/${CompilerId}-${lang}-FeatureTests.cmake" OPTIONAL)
  set(_cmake_oldestSupported_${CompilerId} ${_cmake_oldestSupported} PARENT_SCOPE)
  foreach(feature ${ARGN})
    set(_cmake_feature_test_${CompilerId}_${feature} ${_cmake_feature_test_${feature}} PARENT_SCOPE)
  endforeach()
  include("${CMAKE_ROOT}/Modules/Compiler/${CompilerId}-${lang}-DetermineCompiler.cmake" OPTIONAL
      RESULT_VARIABLE determinedCompiler)
  if (NOT determinedCompiler)
    include("${CMAKE_ROOT}/Modules/Compiler/${CompilerId}-DetermineCompiler.cmake" OPTIONAL)
  endif()
  set(_compiler_id_version_compute_${CompilerId} ${_compiler_id_version_compute} PARENT_SCOPE)
endfunction()

macro(_simpledefine FEATURE_NAME FEATURE_TESTNAME FEATURE_STRING FEATURE_DEFAULT_STRING)
  if (feature STREQUAL "${FEATURE_NAME}")
        set(def_value "${prefix_arg}_${FEATURE_TESTNAME}")
        string(APPEND file_content "
#  if defined(${def_name}) && ${def_name}
#    define ${def_value} ${FEATURE_STRING}
#  else
#    define ${def_value} ${FEATURE_DEFAULT_STRING}
#  endif
\n")
  endif()
endmacro()

macro(_simplebaredefine FEATURE_NAME FEATURE_STRING FEATURE_DEFAULT_STRING)
  if (feature STREQUAL "${FEATURE_NAME}")
        string(APPEND file_content "
#  if !(defined(${def_name}) && ${def_name})
#    define ${FEATURE_STRING} ${FEATURE_DEFAULT_STRING}
#  endif
\n")
  endif()
endmacro()

function(_check_feature_lists C_FEATURE_VAR CXX_FEATURE_VAR)
  foreach(feature ${ARGN})
    if (feature MATCHES "^c_std_")
      # ignored
    elseif (feature MATCHES "^cxx_std_")
      # ignored
    elseif (feature MATCHES "^cxx_")
      list(APPEND _langs CXX)
      list(APPEND ${CXX_FEATURE_VAR} ${feature})
    elseif (feature MATCHES "^c_")
      list(APPEND _langs C)
      list(APPEND ${C_FEATURE_VAR} ${feature})
    else()
      message(FATAL_ERROR "Unsupported feature ${feature}.")
    endif()
  endforeach()
  set(${C_FEATURE_VAR} ${${C_FEATURE_VAR}} PARENT_SCOPE)
  set(${CXX_FEATURE_VAR} ${${CXX_FEATURE_VAR}} PARENT_SCOPE)
  set(_langs ${_langs} PARENT_SCOPE)
endfunction()

function(write_compiler_detection_header
    file_keyword file_arg
    prefix_keyword prefix_arg
    )
  if (NOT "x${file_keyword}" STREQUAL "xFILE")
    message(FATAL_ERROR "write_compiler_detection_header: FILE parameter missing.")
  endif()
  if (NOT "x${prefix_keyword}" STREQUAL "xPREFIX")
    message(FATAL_ERROR "write_compiler_detection_header: PREFIX parameter missing.")
  endif()
  set(options ALLOW_UNKNOWN_COMPILERS ALLOW_UNKNOWN_COMPILER_VERSIONS)
  set(oneValueArgs VERSION EPILOG PROLOG OUTPUT_FILES_VAR OUTPUT_DIR)
  set(multiValueArgs COMPILERS FEATURES BARE_FEATURES)
  cmake_parse_arguments(_WCD "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT _WCD_COMPILERS)
    message(FATAL_ERROR "Invalid arguments.  write_compiler_detection_header requires at least one compiler.")
  endif()
  if (NOT _WCD_FEATURES AND NOT _WCD_BARE_FEATURES)
    message(FATAL_ERROR "Invalid arguments.  write_compiler_detection_header requires at least one feature.")
  endif()

  if(_WCD_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unparsed arguments: ${_WCD_UNPARSED_ARGUMENTS}")
  endif()

  if (prefix_arg STREQUAL "")
    message(FATAL_ERROR "A prefix must be specified")
  endif()
  string(MAKE_C_IDENTIFIER ${prefix_arg} cleaned_prefix)
  if (NOT prefix_arg STREQUAL cleaned_prefix)
    message(FATAL_ERROR "The prefix must be a valid C identifier.")
  endif()

  if(NOT _WCD_VERSION)
    set(_WCD_VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})
  endif()
  set(_min_version 3.1.0) # Version which introduced this function
  if (_WCD_VERSION VERSION_LESS _min_version)
    set(err "VERSION compatibility for write_compiler_detection_header is set to ${_WCD_VERSION}, which is too low.")
    string(APPEND err "  It must be set to at least ${_min_version}.  ")
    string(APPEND err "  Either set the VERSION parameter to the write_compiler_detection_header function, or update")
    string(APPEND err " your minimum required CMake version with the cmake_minimum_required command.")
    message(FATAL_ERROR "${err}")
  endif()

  if(_WCD_OUTPUT_FILES_VAR)
    if(NOT _WCD_OUTPUT_DIR)
      message(FATAL_ERROR "If OUTPUT_FILES_VAR is specified, then OUTPUT_DIR must also be specified.")
    endif()
  endif()
  if(_WCD_OUTPUT_DIR)
    if(NOT _WCD_OUTPUT_FILES_VAR)
      message(FATAL_ERROR "If OUTPUT_DIR is specified, then OUTPUT_FILES_VAR must also be specified.")
    endif()
    get_filename_component(main_file_dir ${file_arg} DIRECTORY)
    if (NOT IS_ABSOLUTE ${main_file_dir})
      set(main_file_dir "${CMAKE_CURRENT_BINARY_DIR}/${main_file_dir}")
    endif()
    if (NOT IS_ABSOLUTE ${_WCD_OUTPUT_DIR})
      set(_WCD_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/${_WCD_OUTPUT_DIR}")
    endif()
    get_filename_component(out_file_dir ${_WCD_OUTPUT_DIR} ABSOLUTE)
    string(FIND ${out_file_dir} ${main_file_dir} idx)
    if (NOT idx EQUAL 0)
      message(FATAL_ERROR "The compiler-specific output directory must be within the same directory as the main file.")
    endif()

    if (main_file_dir STREQUAL out_file_dir)
      unset(_WCD_OUTPUT_DIR)
    else()
      string(REPLACE "${main_file_dir}/" "" _WCD_OUTPUT_DIR "${out_file_dir}/")
    endif()
  endif()

  set(compilers
    GNU
    Clang
    AppleClang
    MSVC
    SunPro
    Intel
  )

  set(_hex_compilers ADSP Borland Embarcadero SunPro)

  foreach(_comp ${_WCD_COMPILERS})
    list(FIND compilers ${_comp} idx)
    if (idx EQUAL -1)
      message(FATAL_ERROR "Unsupported compiler ${_comp}.")
    endif()
    if (NOT _need_hex_conversion)
      list(FIND _hex_compilers ${_comp} idx)
      if (NOT idx EQUAL -1)
        set(_need_hex_conversion TRUE)
      endif()
    endif()
  endforeach()

  set(file_content "
// This is a generated file. Do not edit!

#ifndef ${prefix_arg}_COMPILER_DETECTION_H
#define ${prefix_arg}_COMPILER_DETECTION_H
")

  if (_WCD_PROLOG)
    string(APPEND file_content "\n${_WCD_PROLOG}\n")
  endif()

  if (_need_hex_conversion)
    string(APPEND file_content "
#define ${prefix_arg}_DEC(X) (X)
#define ${prefix_arg}_HEX(X) ( \\
    ((X)>>28 & 0xF) * 10000000 + \\
    ((X)>>24 & 0xF) *  1000000 + \\
    ((X)>>20 & 0xF) *   100000 + \\
    ((X)>>16 & 0xF) *    10000 + \\
    ((X)>>12 & 0xF) *     1000 + \\
    ((X)>>8  & 0xF) *      100 + \\
    ((X)>>4  & 0xF) *       10 + \\
    ((X)     & 0xF) \\
    )\n")
  endif()

  _check_feature_lists(C_features CXX_features ${_WCD_FEATURES})
  _check_feature_lists(C_bare_features CXX_bare_features ${_WCD_BARE_FEATURES})
  list(REMOVE_DUPLICATES _langs)

  if(_WCD_OUTPUT_FILES_VAR)
    get_filename_component(main_file_name ${file_arg} NAME)
    set(compiler_file_content_
"#ifndef ${prefix_arg}_COMPILER_DETECTION_H
#  error This file may only be included from ${main_file_name}
#endif\n")
  endif()

  foreach(_lang ${_langs})
    set(target_compilers)
    foreach(compiler ${_WCD_COMPILERS})
      _load_compiler_variables(${compiler} ${_lang} ${${_lang}_features})
      if(_cmake_oldestSupported_${compiler})
        list(APPEND target_compilers ${compiler})
      endif()
    endforeach()

    get_property(known_features GLOBAL PROPERTY CMAKE_${_lang}_KNOWN_FEATURES)
    foreach(feature ${${_lang}_features})
      list(FIND known_features ${feature} idx)
      if (idx EQUAL -1)
        message(FATAL_ERROR "Unsupported feature ${feature}.")
      endif()
    endforeach()

    if(_lang STREQUAL CXX)
      string(APPEND file_content "\n#ifdef __cplusplus\n")
    else()
      string(APPEND file_content "\n#ifndef __cplusplus\n")
    endif()

    compiler_id_detection(ID_CONTENT ${_lang} PREFIX ${prefix_arg}_
      ID_DEFINE
    )

    string(APPEND file_content "${ID_CONTENT}\n")

    set(pp_if "if")
    foreach(compiler ${target_compilers})
      string(APPEND file_content "\n#  ${pp_if} ${prefix_arg}_COMPILER_IS_${compiler}\n")

      if(_WCD_OUTPUT_FILES_VAR)
        set(compile_file_name "${_WCD_OUTPUT_DIR}${prefix_arg}_COMPILER_INFO_${compiler}_${_lang}.h")
        string(APPEND file_content "\n#    include \"${compile_file_name}\"\n")
      endif()

      if(_WCD_OUTPUT_FILES_VAR)
        set(compiler_file_content compiler_file_content_${compiler}_${_lang})
      else()
        set(compiler_file_content file_content)
      endif()

      if(NOT _WCD_ALLOW_UNKNOWN_COMPILER_VERSIONS)
        string(APPEND ${compiler_file_content} "
#    if !(${_cmake_oldestSupported_${compiler}})
#      error Unsupported compiler version
#    endif\n")
      endif()

      set(PREFIX ${prefix_arg}_)
      if (_need_hex_conversion)
        set(MACRO_DEC ${prefix_arg}_DEC)
        set(MACRO_HEX ${prefix_arg}_HEX)
      else()
        set(MACRO_DEC)
        set(MACRO_HEX)
      endif()
      string(CONFIGURE "${_compiler_id_version_compute_${compiler}}" VERSION_BLOCK @ONLY)
      string(APPEND ${compiler_file_content} "${VERSION_BLOCK}\n")
      set(PREFIX)
      set(MACRO_DEC)
      set(MACRO_HEX)

      set(pp_if "elif")
      foreach(feature ${${_lang}_features})
        string(TOUPPER ${feature} feature_upper)
        set(feature_PP "COMPILER_${feature_upper}")
        set(_define_item "\n#    define ${prefix_arg}_${feature_PP} 0\n")
        if (_cmake_feature_test_${compiler}_${feature} STREQUAL "1")
          set(_define_item "\n#    define ${prefix_arg}_${feature_PP} 1\n")
        elseif (_cmake_feature_test_${compiler}_${feature})
          set(_define_item "\n#      define ${prefix_arg}_${feature_PP} 0\n")
          set(_define_item "\n#    if ${_cmake_feature_test_${compiler}_${feature}}\n#      define ${prefix_arg}_${feature_PP} 1\n#    else${_define_item}#    endif\n")
        endif()
        string(APPEND ${compiler_file_content} "${_define_item}")
      endforeach()
    endforeach()
    if(pp_if STREQUAL "elif")
      if(_WCD_ALLOW_UNKNOWN_COMPILERS)
        string(APPEND file_content "
#  endif\n")
      else()
        string(APPEND file_content "
#  else
#    error Unsupported compiler
#  endif\n")
      endif()
    endif()
    foreach(feature ${${_lang}_features})
      string(TOUPPER ${feature} feature_upper)
      set(feature_PP "COMPILER_${feature_upper}")
      set(def_name ${prefix_arg}_${feature_PP})
      _simpledefine(c_restrict RESTRICT restrict "")
      _simpledefine(cxx_constexpr CONSTEXPR constexpr "")
      _simpledefine(cxx_final FINAL final "")
      _simpledefine(cxx_override OVERRIDE override "")
      if (feature STREQUAL cxx_static_assert)
        set(def_value "${prefix_arg}_STATIC_ASSERT(X)")
        set(def_value_msg "${prefix_arg}_STATIC_ASSERT_MSG(X, MSG)")
        set(def_fallback "enum { ${prefix_arg}_STATIC_ASSERT_JOIN(${prefix_arg}StaticAssertEnum, __LINE__) = sizeof(${prefix_arg}StaticAssert<X>) }")
        string(APPEND file_content "#  if defined(${def_name}) && ${def_name}
#    define ${def_value} static_assert(X, #X)
#    define ${def_value_msg} static_assert(X, MSG)
#  else
#    define ${prefix_arg}_STATIC_ASSERT_JOIN(X, Y) ${prefix_arg}_STATIC_ASSERT_JOIN_IMPL(X, Y)
#    define ${prefix_arg}_STATIC_ASSERT_JOIN_IMPL(X, Y) X##Y
template<bool> struct ${prefix_arg}StaticAssert;
template<> struct ${prefix_arg}StaticAssert<true>{};
#    define ${def_value} ${def_fallback}
#    define ${def_value_msg} ${def_fallback}
#  endif
\n")
      endif()
      if (feature STREQUAL cxx_alignas)
        set(def_value "${prefix_arg}_ALIGNAS(X)")
        string(APPEND file_content "
#  if defined(${def_name}) && ${def_name}
#    define ${def_value} alignas(X)
#  elif ${prefix_arg}_COMPILER_IS_GNU || ${prefix_arg}_COMPILER_IS_Clang || ${prefix_arg}_COMPILER_IS_AppleClang
#    define ${def_value} __attribute__ ((__aligned__(X)))
#  elif ${prefix_arg}_COMPILER_IS_MSVC
#    define ${def_value} __declspec(align(X))
#  else
#    define ${def_value}
#  endif
\n")
      endif()
      if (feature STREQUAL cxx_alignof)
        set(def_value "${prefix_arg}_ALIGNOF(X)")
        string(APPEND file_content "
#  if defined(${def_name}) && ${def_name}
#    define ${def_value} alignof(X)
#  elif ${prefix_arg}_COMPILER_IS_GNU || ${prefix_arg}_COMPILER_IS_Clang || ${prefix_arg}_COMPILER_IS_AppleClang
#    define ${def_value} __alignof__(X)
#  elif ${prefix_arg}_COMPILER_IS_MSVC
#    define ${def_value} __alignof(X)
#  endif
\n")
      endif()
      _simpledefine(cxx_deleted_functions DELETED_FUNCTION "= delete" "")
      _simpledefine(cxx_extern_templates EXTERN_TEMPLATE extern "")
      if (feature STREQUAL cxx_noexcept)
        set(def_value "${prefix_arg}_NOEXCEPT")
        string(APPEND file_content "
#  if defined(${def_name}) && ${def_name}
#    define ${def_value} noexcept
#    define ${def_value}_EXPR(X) noexcept(X)
#  else
#    define ${def_value}
#    define ${def_value}_EXPR(X)
#  endif
\n")
      endif()
      if (feature STREQUAL cxx_nullptr)
        set(def_value "${prefix_arg}_NULLPTR")
        string(APPEND file_content "
#  if defined(${def_name}) && ${def_name}
#    define ${def_value} nullptr
#  elif ${prefix_arg}_COMPILER_IS_GNU
#    define ${def_value} __null
#  else
#    define ${def_value} 0
#  endif
\n")
      endif()
      if (feature STREQUAL cxx_thread_local)
        set(def_value "${prefix_arg}_THREAD_LOCAL")
        string(APPEND file_content "
#  if defined(${def_name}) && ${def_name}
#    define ${def_value} thread_local
#  elif ${prefix_arg}_COMPILER_IS_GNU || ${prefix_arg}_COMPILER_IS_Clang || ${prefix_arg}_COMPILER_IS_AppleClang
#    define ${def_value} __thread
#  elif ${prefix_arg}_COMPILER_IS_MSVC
#    define ${def_value} __declspec(thread)
#  else
// ${def_value} not defined for this configuration.
#  endif
\n")
      endif()
      if (feature STREQUAL cxx_attribute_deprecated)
        set(def_name ${prefix_arg}_${feature_PP})
        set(def_value "${prefix_arg}_DEPRECATED")
        string(APPEND file_content "
#  ifndef ${def_value}
#    if defined(${def_name}) && ${def_name}
#      define ${def_value} [[deprecated]]
#      define ${def_value}_MSG(MSG) [[deprecated(MSG)]]
#    elif ${prefix_arg}_COMPILER_IS_GNU || ${prefix_arg}_COMPILER_IS_Clang
#      define ${def_value} __attribute__((__deprecated__))
#      define ${def_value}_MSG(MSG) __attribute__((__deprecated__(MSG)))
#    elif ${prefix_arg}_COMPILER_IS_MSVC
#      define ${def_value} __declspec(deprecated)
#      define ${def_value}_MSG(MSG) __declspec(deprecated(MSG))
#    else
#      define ${def_value}
#      define ${def_value}_MSG(MSG)
#    endif
#  endif
\n")
      endif()
    endforeach()

    foreach(feature ${${_lang}_bare_features})
      string(TOUPPER ${feature} feature_upper)
      set(feature_PP "COMPILER_${feature_upper}")
      set(def_name ${prefix_arg}_${feature_PP})
      _simplebaredefine(c_restrict restrict "")
      _simplebaredefine(cxx_constexpr constexpr "")
      _simplebaredefine(cxx_final final "")
      _simplebaredefine(cxx_override override "")
      if (feature STREQUAL cxx_nullptr)
        set(def_value "nullptr")
        string(APPEND file_content "
#  if !(defined(${def_name}) && ${def_name})
#    if ${prefix_arg}_COMPILER_IS_GNU
#      define ${def_value} __null
#    else
#      define ${def_value} 0
#    endif
#  endif
\n")
      endif()
      _simplebaredefine(cxx_noexcept noexcept "")
    endforeach()

    string(APPEND file_content "#endif\n")

  endforeach()

  if(_WCD_OUTPUT_FILES_VAR)
    foreach(compiler ${_WCD_COMPILERS})
      foreach(_lang ${_langs})
        if(compiler_file_content_${compiler}_${_lang})
          set(compile_file_name "${_WCD_OUTPUT_DIR}${prefix_arg}_COMPILER_INFO_${compiler}_${_lang}.h")
          set(full_path "${main_file_dir}/${compile_file_name}")
          list(APPEND ${_WCD_OUTPUT_FILES_VAR} ${full_path})
          file(
            CONFIGURE
            OUTPUT "${full_path}"
            CONTENT "${compiler_file_content_}${compiler_file_content_${compiler}_${_lang}}\n"
            @ONLY
          )
        endif()
      endforeach()
    endforeach()
    set(${_WCD_OUTPUT_FILES_VAR} ${${_WCD_OUTPUT_FILES_VAR}} PARENT_SCOPE)
  endif()

  if (_WCD_EPILOG)
    string(APPEND file_content "\n${_WCD_EPILOG}\n")
  endif()
  string(APPEND file_content "\n#endif")

  file(CONFIGURE OUTPUT "${file_arg}" CONTENT "${file_content}\n" @ONLY)
endfunction()
