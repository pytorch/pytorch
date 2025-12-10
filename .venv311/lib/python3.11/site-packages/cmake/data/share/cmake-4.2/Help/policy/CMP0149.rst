CMP0149
-------

.. versionadded:: 3.27

:ref:`Visual Studio Generators` select latest Windows SDK by default.

Visual Studio Generators select a Windows SDK version to put in the
``WindowsTargetPlatformVersion`` setting in ``.vcxproj`` files.
CMake sets the :variable:`CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION`
variable to the selected SDK version.

Prior to CMake 3.27, the SDK version was always selected by the value of
the :variable:`CMAKE_SYSTEM_VERSION` variable.  Users or toolchain files
could set that variable to one of the exact Windows SDK versions available
on the host system.  Since :variable:`CMAKE_SYSTEM_VERSION` defaults to
:variable:`CMAKE_HOST_SYSTEM_VERSION`, and it is not guaranteed that a
matching Windows SDK version is available, CMake had to fall back to
using the latest Windows SDK version if no exact match was available.
This approach was problematic:

* The latest Windows SDK might or might not be selected based on whether
  the host version of Windows happens to match an available SDK version.

* An old Windows SDK version might be selected that has not been updated
  for newer language standards such as C11.

CMake 3.27 and higher prefer to ignore the exact value of
:variable:`CMAKE_SYSTEM_VERSION` and by default select the latest SDK
version available.  An exact SDK version may be specified explicitly
using a ``version=`` field in the :variable:`CMAKE_GENERATOR_PLATFORM`
variable.  See :ref:`Visual Studio Platform Selection`.

This policy provides compatibility for projects, toolchain files, and
build scripts that have not been ported away from using
:variable:`CMAKE_SYSTEM_VERSION` to specify an exact SDK version.

.. note::

  This policy must be set before the first :command:`project` or
  :command:`enable_language` command invocation at the top of the
  project.  That is when :ref:`Visual Studio Generators` select a
  Windows SDK.

The ``OLD`` behavior for this policy is to use the exact value of
:variable:`CMAKE_SYSTEM_VERSION` if possible.  The ``NEW`` behavior
for this policy is to ignore it.

.. |INTRODUCED_IN_CMAKE_VERSION| replace:: 3.27
.. |WARNS_OR_DOES_NOT_WARN| replace:: does *not* warn
.. include:: include/STANDARD_ADVICE.rst

.. include:: include/DEPRECATED.rst
