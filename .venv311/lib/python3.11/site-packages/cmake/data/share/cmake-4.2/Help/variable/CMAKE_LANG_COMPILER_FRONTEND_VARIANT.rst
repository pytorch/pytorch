CMAKE_<LANG>_COMPILER_FRONTEND_VARIANT
--------------------------------------

.. versionadded:: 3.14

Identification string of the compiler frontend variant.

Some compilers have multiple, different frontends for accepting command
line options.  (For example ``Clang`` originally only had a frontend
compatible with the ``GNU`` compiler but since its port to Windows
(``Clang-Cl``) it now also supports a frontend compatible with ``MSVC``.)
When CMake detects such a compiler it sets this
variable to what would have been the :variable:`CMAKE_<LANG>_COMPILER_ID` for
the compiler whose frontend it resembles.

.. note::
  In other words, this variable describes what command line options
  and language extensions the compiler frontend expects.

.. versionchanged:: 3.26
  This variable is set for ``GNU``, ``MSVC``, and ``AppleClang``
  compilers that have only one frontend variant.
