CLICOLOR_FORCE
--------------

.. versionadded:: 3.5

.. include:: include/ENV_VAR.rst

Set to a non-empty value, other than ``0``, to tell command-line
tools to print color messages even if not connected to a terminal.
This is a `common convention`_ among command-line tools in general.

See also the :envvar:`NO_COLOR` and :envvar:`CLICOLOR` environment variables.
If :envvar:`NO_COLOR` is activated, it takes precedence over
:envvar:`!CLICOLOR_FORCE`.  Otherwise, if :envvar:`!CLICOLOR_FORCE` is
activated, it takes precedence over :envvar:`CLICOLOR`.

See the :variable:`CMAKE_COLOR_DIAGNOSTICS` variable to control
color in a generated build system.

.. _`common convention`: https://web.archive.org/web/20250410160803/https://bixense.com/clicolors/
