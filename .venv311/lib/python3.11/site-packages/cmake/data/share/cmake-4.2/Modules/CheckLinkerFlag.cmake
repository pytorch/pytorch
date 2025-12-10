# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CheckLinkerFlag
---------------

.. versionadded:: 3.18

This module provides a command to check whether a given link flag is
supported by the compiler.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CheckLinkerFlag)

Commands
^^^^^^^^

This module provides the following command:

.. command:: check_linker_flag

  Checks once whether the compiler supports a given link flag:

  .. code-block:: cmake

    check_linker_flag(<lang> <flag> <variable>)

  This command checks once whether the linker flag ``<flag>`` is accepted
  by the ``<lang>`` compiler without producing a diagnostic message.

  The arguments are:

  ``<lang>``
    The language of the compiler used for the check.  Supported languages are
    ``C``, ``CXX``, ``CUDA``, ``Fortran``, ``HIP``, ``OBJC``, ``OBJCXX``,
    and ``Swift``.

    .. versionadded:: 3.19
      Support for ``CUDA`` language.

    .. versionadded:: 3.21
      Support for ``HIP`` language.

    .. versionadded:: 3.26
      Support for ``Swift`` language.

  ``<flag>``
    Linker flag(s) to check.  Multiple flags can be specified in one
    argument as a string using a :ref:`semicolon-separated list
    <CMake Language Lists>`.

    The underlying implementation uses the :prop_tgt:`LINK_OPTIONS` target
    property to test the specified flag.  The ``LINKER:`` (and ``SHELL:``)
    prefixes may be used, as described in the
    `Handling Compiler Driver Differences`_ section.

  ``<variable>``
    The name of the variable to store the check result.  This variable will
    be created as an internal cache variable.

  This command temporarily sets the ``CMAKE_REQUIRED_LINK_OPTIONS`` variable
  and calls the :command:`check_source_compiles` command from the
  :module:`CheckSourceCompiles` module.

  A successful result only indicates that the compiler did not report an
  error when given the link flag.  Whether the flag has any effect, or the
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

  .. include:: /module/include/CMAKE_REQUIRED_LIBRARIES.rst

  .. include:: /module/include/CMAKE_REQUIRED_LINK_DIRECTORIES.rst

  .. include:: /module/include/CMAKE_REQUIRED_QUIET.rst

.. include:: /command/include/LINK_OPTIONS_LINKER.rst

Examples
^^^^^^^^

Example: Checking Linker Flag
"""""""""""""""""""""""""""""

The following example shows how to use this module to check the ``-z relro``
linker flag, which is supported on many Unix-like systems to enable read-only
relocations for improved binary security.  If the flag is supported by the
linker, it is conditionally added to the executable target using the
:command:`target_link_options`.  The ``LINKER:`` prefix is used to pass the
flag to the linker in a portable and compiler-independent way.

.. code-block:: cmake

  include(CheckLinkerFlag)

  check_linker_flag(C "LINKER:-z,relro" HAVE_Z_RELRO)

  add_executable(example main.c)

  if(HAVE_Z_RELRO)
    target_link_options(example PRIVATE "LINKER:-z,relro")
  endif()

Example: Checking Multiple Flags
""""""""""""""""""""""""""""""""

In the following example, multiple linker flags are checked simultaneously:

.. code-block:: cmake

  include(CheckLinkerFlag)

  check_linker_flag(C "LINKER:-z,relro;LINKER:-z,now" HAVE_FLAGS)

  add_executable(example main.c)

  if(HAVE_FLAGS)
    target_link_options(example PRIVATE LINKER:-z,relro LINKER:-z,now)
  endif()

See Also
^^^^^^^^

* The :variable:`CMAKE_LINKER_TYPE` variable to specify the linker, which
  will be used also by this module.
* The :module:`CheckCompilerFlag` module to check whether a compiler flag
  is supported.
#]=======================================================================]

include_guard(GLOBAL)
include(Internal/CheckLinkerFlag)

function(CHECK_LINKER_FLAG _lang _flag _var)
  cmake_check_linker_flag(${_lang} "${_flag}" ${_var})
endfunction()
