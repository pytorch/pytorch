# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CheckCCompilerFlag
------------------

This module provides a command to check whether the C compiler supports a
given flag.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CheckCCompilerFlag)

Commands
^^^^^^^^

This module provides the following command:

.. command:: check_c_compiler_flag

  Checks once whether the C compiler supports a given flag:

  .. code-block:: cmake

    check_c_compiler_flag(<flag> <variable>)

  This command checks once that the ``<flag>`` is accepted by the C compiler
  without producing a diagnostic message.  Multiple flags can be specified
  in one argument as a string using a :ref:`semicolon-separated list
  <CMake Language Lists>`.

  The result of the check is stored in the internal cache variable specified
  by ``<variable>``, with boolean true for success and boolean false for
  failure.

  A successful result only indicates that the compiler did not report an
  error when given the flag.  Whether the flag has any effect, or the
  intended one, is outside the scope of this module.

  .. note::

    Since the underlying :command:`try_compile` command also uses flags from
    variables like :variable:`CMAKE_<LANG>_FLAGS`, unknown or unsupported
    flags in those variables may result in a false negative for this check.

  .. rubric:: Variables Affecting the Check

  The following variables may be set before calling this command to modify
  the way the check is run:

  .. include:: /module/include/CMAKE_REQUIRED_FLAGS.rst

  .. include:: /module/include/CMAKE_REQUIRED_DEFINITIONS.rst

  .. include:: /module/include/CMAKE_REQUIRED_INCLUDES.rst

  .. include:: /module/include/CMAKE_REQUIRED_LINK_OPTIONS.rst

  .. include:: /module/include/CMAKE_REQUIRED_LIBRARIES.rst

  .. include:: /module/include/CMAKE_REQUIRED_LINK_DIRECTORIES.rst

  .. include:: /module/include/CMAKE_REQUIRED_QUIET.rst

  .. include:: /module/include/CMAKE_TRY_COMPILE_TARGET_TYPE.rst

Examples
^^^^^^^^

The following example demonstrates how to use this module to check support
for the C compiler flag ``-fno-optimize-strlen``, which disables
optimizations related to the ``strlen()`` C function in GCC and Clang
compilers.  The result of the check is stored in the internal cache
variable ``HAVE_FNO_OPTIMIZE_STRLEN``, and the flag is conditionally enabled
using the :command:`target_compile_options` command.  The
:genex:`$<COMPILE_LANGUAGE:...> <COMPILE_LANGUAGE:languages>` generator
expression ensures that the flag is added only to ``C`` source files.

.. code-block:: cmake

  include(CheckCCompilerFlag)
  check_c_compiler_flag(-fno-optimize-strlen HAVE_FNO_OPTIMIZE_STRLEN)

  if(HAVE_FNO_OPTIMIZE_STRLEN)
    target_compile_options(
      example
      PRIVATE $<$<COMPILE_LANGUAGE:C>:-fno-optimize-strlen>
    )
  endif()

See Also
^^^^^^^^

* The :module:`CheckCompilerFlag` module for a more general command to check
  whether a compiler flag is supported.
#]=======================================================================]

include_guard(GLOBAL)
include(Internal/CheckCompilerFlag)

macro (CHECK_C_COMPILER_FLAG _FLAG _RESULT)
  cmake_check_compiler_flag(C "${_FLAG}" ${_RESULT})
endmacro ()

# FIXME(#24994): The following module is included only for compatibility
# with projects that accidentally relied on it with CMake 3.26 and below.
include(CheckCSourceCompiles)
