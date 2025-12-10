VS_IOT_EXTENSIONS_VERSION
-------------------------

.. versionadded:: 3.4

Visual Studio Windows 10 IoT Extensions Version

Specifies the version of the IoT Extensions that should be included in the
target. For example ``10.0.10240.0``. If the value is not specified, the IoT
Extensions will not be included. To use the same version of the extensions as
the Windows 10 SDK that is being used, you can use the
:variable:`CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION` variable.
