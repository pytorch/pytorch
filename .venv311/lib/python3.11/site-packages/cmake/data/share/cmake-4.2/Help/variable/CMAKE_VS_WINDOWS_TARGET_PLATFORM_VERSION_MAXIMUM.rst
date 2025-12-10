CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION_MAXIMUM
------------------------------------------------

.. versionadded:: 3.19

Override the :ref:`Windows 10 SDK Maximum Version for VS 2015` and beyond.

The ``CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION_MAXIMUM`` variable may
be set to a false value (e.g. ``OFF``, ``FALSE``, or ``0``) or the SDK version
to use as the maximum (e.g. ``10.0.14393.0``).  If unset, the default depends
on which version of Visual Studio is targeted by the current generator.

This can be used to exclude Windows SDK versions from consideration for
:variable:`CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION`.
