VS_WINDOWS_TARGET_PLATFORM_MIN_VERSION
--------------------------------------

.. versionadded:: 3.4

Visual Studio Windows Target Platform Minimum Version

For Windows 10. Specifies the minimum version of the OS that is being
targeted. For example ``10.0.10240.0``. If the value is not specified, the
value of the :variable:`CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION` variable
will be used on WindowsStore projects.  Otherwise the target platform
minimum version will not be specified for the project.

.. versionadded:: 3.27
  This property is initialized by the value of the
  :variable:`CMAKE_VS_WINDOWS_TARGET_PLATFORM_MIN_VERSION` variable
  if it is set when a target is created.
