CMAKE_UNITY_BUILD
-----------------

.. versionadded:: 3.16

This variable is used to initialize the :prop_tgt:`UNITY_BUILD`
property of targets when they are created.  Setting it to true
enables batch compilation of multiple sources within each target.
This feature is known as a *Unity* or *Jumbo* build.

Projects should not set this variable, it is intended as a developer
control to be set on the :manual:`cmake(1)` command line or other
equivalent methods.  The developer must have the ability to enable or
disable unity builds according to the capabilities of their own machine
and compiler.

By default, this variable is not set, which will result in unity builds
being disabled.

.. note::
  This option currently does not work well in combination with
  the :variable:`CMAKE_EXPORT_COMPILE_COMMANDS` variable.
