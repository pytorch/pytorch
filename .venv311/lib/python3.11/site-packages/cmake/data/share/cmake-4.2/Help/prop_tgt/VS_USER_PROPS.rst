VS_USER_PROPS
-------------

.. versionadded:: 3.8

Sets the user props file to be included in the visual studio
C++ project file. The standard path is
``$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props``, which is
in most cases the same as
``%LOCALAPPDATA%\\Microsoft\\MSBuild\\v4.0\\Microsoft.Cpp.Win32.user.props``
or ``%LOCALAPPDATA%\\Microsoft\\MSBuild\\v4.0\\Microsoft.Cpp.x64.user.props``.

The ``*.user.props`` files can be used for Visual Studio wide
configuration which is independent from cmake.
