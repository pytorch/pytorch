Visual Studio 15 2017
---------------------

.. versionadded:: 3.7.1

Generates Visual Studio 15 (VS 2017) project files.

Project Types
^^^^^^^^^^^^^

Only Visual C++ and C# projects may be generated (and Fortran with
Intel compiler integration).  Other types of projects (JavaScript,
Powershell, Python, etc.) are not supported.

Instance Selection
^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.11

VS 2017 supports multiple installations on the same machine.  The
:variable:`CMAKE_GENERATOR_INSTANCE` variable may be used to select one.

Platform Selection
^^^^^^^^^^^^^^^^^^

The default target platform name (architecture) is ``Win32``.

The :variable:`CMAKE_GENERATOR_PLATFORM` variable may be set, perhaps
via the :option:`cmake -A` option, to specify a target platform
name (architecture).  For example:

* ``cmake -G "Visual Studio 15 2017" -A Win32``
* ``cmake -G "Visual Studio 15 2017" -A x64``
* ``cmake -G "Visual Studio 15 2017" -A ARM``
* ``cmake -G "Visual Studio 15 2017" -A ARM64``

.. versionchanged:: 4.0

  Previously, for compatibility with CMake versions prior to 3.1,
  one could specify a target platform name optionally at the
  end of the generator name.  This has been removed.
  This was supported only for:

  ``Visual Studio 15 2017 Win64``
    Specify target platform ``x64``.

  ``Visual Studio 15 2017 ARM``
    Specify target platform ``ARM``.

Toolset Selection
^^^^^^^^^^^^^^^^^

The ``v141`` toolset that comes with Visual Studio 15 2017 is selected by
default.  The :variable:`CMAKE_GENERATOR_TOOLSET` option may be set, perhaps
via the :option:`cmake -T` option, to specify another toolset.

.. |VS_TOOLSET_HOST_ARCH_DEFAULT| replace::
   By default this generator uses the 32-bit variant even on a 64-bit host.

.. include:: include/VS_TOOLSET_HOST_ARCH_LEGACY.rst
