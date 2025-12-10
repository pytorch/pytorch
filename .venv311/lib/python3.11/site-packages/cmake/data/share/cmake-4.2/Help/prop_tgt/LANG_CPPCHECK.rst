<LANG>_CPPCHECK
---------------

.. versionadded:: 3.10

This property is supported only when ``<LANG>`` is ``C`` or ``CXX``.

Specify a :ref:`semicolon-separated list <CMake Language Lists>` containing a command line
for the ``cppcheck`` static analysis tool.  The :ref:`Makefile Generators`
and the :generator:`Ninja` generator will run ``cppcheck`` along with the
compiler and report any problems.  If the command-line specifies the
exit code options to ``cppcheck`` then the build  will fail if the
tool returns non-zero.

This property is initialized by the value of the
:variable:`CMAKE_<LANG>_CPPCHECK` variable if it is set when a target is
created.

.. versionadded:: 3.27

  This property supports
  :manual:`generator expressions <cmake-generator-expressions(7)>`.

.. versionadded:: 3.27

  :prop_sf:`SKIP_LINTING` can be set on individual source files to exclude
  them from the linting tools defined by :prop_tgt:`<LANG>_CPPLINT`,
  :prop_tgt:`<LANG>_CLANG_TIDY`, ``<LANG>_CPPCHECK``, and
  :prop_tgt:`<LANG>_INCLUDE_WHAT_YOU_USE`.  When :prop_sf:`SKIP_LINTING` is
  set to true on a source file, those tools will not be run on that specific
  file.
