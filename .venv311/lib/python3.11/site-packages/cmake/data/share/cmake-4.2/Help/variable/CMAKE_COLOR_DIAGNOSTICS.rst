CMAKE_COLOR_DIAGNOSTICS
-----------------------

.. versionadded:: 3.24

Enable color diagnostics throughout the generated build system.

This variable uses three states: ``ON``, ``OFF`` and not defined.

When not defined:

* :ref:`Makefile Generators` initialize the :variable:`CMAKE_COLOR_MAKEFILE`
  variable to ``ON``.  It controls color buildsystem messages.

* GNU/Clang compilers are not invoked with any color diagnostics flag.

When ``ON``:

* :ref:`Makefile Generators` produce color buildsystem messages by default.
  :variable:`CMAKE_COLOR_MAKEFILE` is not initialized, but may be
  explicitly set to ``OFF`` to disable color buildsystem messages.

* GNU/Clang compilers are invoked with a flag enabling color diagnostics
  (``-fcolor-diagnostics``).

When ``OFF``:

* :ref:`Makefile Generators` do not produce color buildsystem messages by
  default.  :variable:`CMAKE_COLOR_MAKEFILE` is not initialized, but may be
  explicitly set to ``ON`` to enable color buildsystem messages.

* GNU/Clang compilers are invoked with a flag disabling color diagnostics
  (``-fno-color-diagnostics``).

If the :envvar:`CMAKE_COLOR_DIAGNOSTICS` environment variable is set, its
value is used.  Otherwise, ``CMAKE_COLOR_DIAGNOSTICS`` is not defined by
default.

See the :envvar:`CLICOLOR` and :envvar:`CLICOLOR_FORCE` environment
variables to control color output from CMake command-line tools.
