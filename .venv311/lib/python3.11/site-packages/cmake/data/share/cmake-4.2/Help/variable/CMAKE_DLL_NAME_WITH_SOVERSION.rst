CMAKE_DLL_NAME_WITH_SOVERSION
-----------------------------

.. versionadded:: 3.27

This variable is used to initialize the :prop_tgt:`DLL_NAME_WITH_SOVERSION`
property on shared library targets for the Windows platform, which is selected
when the :variable:`WIN32` variable is set.

See this target property for additional information.

Please note that setting this variable has no effect if versioned filenames
are globally disabled with the :variable:`CMAKE_PLATFORM_NO_VERSIONED_SONAME`
variable.
