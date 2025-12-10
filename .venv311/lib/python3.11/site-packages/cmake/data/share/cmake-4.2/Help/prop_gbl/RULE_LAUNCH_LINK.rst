RULE_LAUNCH_LINK
----------------

Specify a launcher for link rules.

.. note::
  This property is intended for internal use by :manual:`ctest(1)`.  Projects
  and developers should use the :prop_tgt:`<LANG>_LINKER_LAUNCHER` target
  properties or the associated :variable:`CMAKE_<LANG>_LINKER_LAUNCHER`
  variables instead.

:ref:`Makefile Generators` and the :generator:`Ninja` generator prefix
link and archive commands with the given launcher command line.
This is intended to allow launchers to intercept build problems
with high granularity.  Other generators ignore this property
because their underlying build systems provide no hook to wrap
individual commands with a launcher.
