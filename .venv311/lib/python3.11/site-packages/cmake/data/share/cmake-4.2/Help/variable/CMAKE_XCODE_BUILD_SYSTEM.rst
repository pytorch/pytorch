CMAKE_XCODE_BUILD_SYSTEM
------------------------

.. versionadded:: 3.19

Xcode build system selection.

The :generator:`Xcode` generator defines this variable to indicate which
variant of the Xcode build system will be used.  The value is the
version of Xcode in which the corresponding build system first became
mature enough for use by CMake.  The possible values are:

``1``
  The original Xcode build system.
  This is the default when using Xcode 11.x or below and supported
  up to Xcode 13.x.

``12``
  The Xcode "new build system" introduced by Xcode 10.
  It became mature enough for use by CMake in Xcode 12.
  This is the default when using Xcode 12.x or above.

The ``CMAKE_XCODE_BUILD_SYSTEM`` variable is informational and should not
be modified by project code.  See the :ref:`Xcode Build System Selection`
documentation section to select the Xcode build system.
