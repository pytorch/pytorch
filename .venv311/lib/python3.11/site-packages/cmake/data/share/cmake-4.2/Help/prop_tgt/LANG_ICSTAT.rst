<LANG>_ICSTAT
-------------

.. versionadded:: 4.1

This property is supported only when ``<LANG>`` is ``C`` or ``CXX``.

Specify a :ref:`semicolon-separated list <CMake Language Lists>`
containing a command line for the ``icstat`` static analysis tool.
The :ref:`Makefile Generators` and the :ref:`Ninja Generators` will
run ``icstat`` along with the compiler and report any problems.
The build will fail if the ``icstat`` tool returns non-zero.

This property is initialized by the value of the
:variable:`CMAKE_<LANG>_ICSTAT` variable if it is set when a target is
created.  It also supports
:manual:`generator expressions <cmake-generator-expressions(7)>`.

This lint may be suppressed for individual source files by setting
the :prop_sf:`SKIP_LINTING` source file property.
