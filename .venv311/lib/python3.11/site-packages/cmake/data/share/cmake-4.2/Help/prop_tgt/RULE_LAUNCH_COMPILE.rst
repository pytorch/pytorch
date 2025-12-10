RULE_LAUNCH_COMPILE
-------------------

Specify a launcher for compile rules.

.. note::
  This property is intended for internal use by :manual:`ctest(1)`.  Projects
  and developers should use the :prop_tgt:`<LANG>_COMPILER_LAUNCHER` target
  properties or the associated :variable:`CMAKE_<LANG>_COMPILER_LAUNCHER`
  variables instead.

See the :prop_gbl:`global property <RULE_LAUNCH_COMPILE>` of the same name
for details.  This overrides the global and directory property for a target.
