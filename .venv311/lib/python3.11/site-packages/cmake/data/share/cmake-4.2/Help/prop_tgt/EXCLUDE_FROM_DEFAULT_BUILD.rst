EXCLUDE_FROM_DEFAULT_BUILD
--------------------------

Exclude a target from the solution build configuration.

This property is only used by :ref:`Visual Studio Generators`. When set to
``TRUE``, the target will be excluded from the build when the "Build Solution"
command is run.

This property has a per-configuration version:
:prop_tgt:`EXCLUDE_FROM_DEFAULT_BUILD_<CONFIG>`.

.. note::
  Solution build configurations do not take project dependencies into account.
  If a target is excluded, it will not be built, even if another target
  included in the configuration depends on it. This behavior differs from the
  CMake-generated ``ALL_BUILD`` target and the :prop_tgt:`EXCLUDE_FROM_ALL`
  property.
