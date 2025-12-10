CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION
----------------------------------------

.. versionadded:: 3.4

Visual Studio Windows Target Platform Version.

When targeting Windows 10 and above, :ref:`Visual Studio Generators` for
VS 2015 and above support specification of a Windows SDK version:

* If :variable:`CMAKE_GENERATOR_PLATFORM` specifies a ``version=`` field,
  as documented by :ref:`Visual Studio Platform Selection`, that SDK
  version is selected.

* Otherwise, if the ``WindowsSDKVersion`` environment variable
  is set to an available SDK version, that version is selected.
  This is intended for use in environments established by ``vcvarsall.bat``
  or similar scripts.

  .. versionadded:: 3.27
    This is enabled by policy :policy:`CMP0149`.

* Otherwise, if :variable:`CMAKE_SYSTEM_VERSION` is set to an available
  SDK version, that version is selected.

  .. versionchanged:: 3.27
    This is disabled by policy :policy:`CMP0149`.

* Otherwise, CMake uses the latest Windows SDK version available.

The chosen Windows target version number is provided
in ``CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION``.  If no Windows 10 SDK
is available this value will be empty.

One may set a ``CMAKE_WINDOWS_KITS_10_DIR`` *environment variable*
to an absolute path to tell CMake to look for Windows 10 SDKs in
a custom location.  The specified directory is expected to contain
``Include/10.0.*`` directories.

See also :variable:`CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION_MAXIMUM`.
