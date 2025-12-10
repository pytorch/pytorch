CLICOLOR
--------

.. versionadded:: 3.21

.. include:: include/ENV_VAR.rst

Set to ``0`` to tell command-line tools not to print color
messages even if connected to a terminal.
This is a `common convention`_ among command-line tools in general.

See also the :envvar:`NO_COLOR` and :envvar:`CLICOLOR_FORCE` environment
variables.  If either of them is activated, it takes precedence over
:envvar:`!CLICOLOR`.

See the :variable:`CMAKE_COLOR_DIAGNOSTICS` variable to control
color in a generated build system.

.. _`common convention`: https://web.archive.org/web/20250410160803/https://bixense.com/clicolors/
