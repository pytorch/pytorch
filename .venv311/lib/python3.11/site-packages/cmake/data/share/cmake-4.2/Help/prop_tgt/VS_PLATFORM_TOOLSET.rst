VS_PLATFORM_TOOLSET
-------------------

.. versionadded:: 3.18

Overrides the platform toolset used to build a target.

Only supported when the compiler used by the given toolset is the
same as the compiler used to build the whole source tree.

This is especially useful to create driver projects with the toolsets
"WindowsUserModeDriver10.0" or "WindowsKernelModeDriver10.0".
