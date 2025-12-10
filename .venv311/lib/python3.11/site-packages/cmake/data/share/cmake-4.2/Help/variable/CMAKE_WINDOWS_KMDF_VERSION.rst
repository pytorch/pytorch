CMAKE_WINDOWS_KMDF_VERSION
--------------------------

.. versionadded:: 3.31

Specify the `Kernel-Mode Drive Framework`_ target version.

A :variable:`toolchain file <CMAKE_TOOLCHAIN_FILE>` that sets
:variable:`CMAKE_SYSTEM_NAME` to ``WindowsKernelModeDriver``
must also set ``CMAKE_WINDOWS_KMDF_VERSION`` to specify the
KMDF target version.

.. _`Kernel-Mode Drive Framework`: https://learn.microsoft.com/en-us/windows-hardware/drivers/wdf/kmdf-version-history
