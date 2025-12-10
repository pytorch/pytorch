Visual Studio 17 2022
---------------------

.. versionadded:: 3.21

Generates Visual Studio 17 (VS 2022) project files.

Project Types
^^^^^^^^^^^^^

Only Visual C++ and C# projects may be generated (and Fortran with
Intel compiler integration).  Other types of projects (JavaScript,
Powershell, Python, etc.) are not supported.

Instance Selection
^^^^^^^^^^^^^^^^^^

VS 2022 supports multiple installations on the same machine.  The
:variable:`CMAKE_GENERATOR_INSTANCE` variable may be used to select one.

Platform Selection
^^^^^^^^^^^^^^^^^^

The default target platform name (architecture) is that of the host
and is provided in the :variable:`CMAKE_VS_PLATFORM_NAME_DEFAULT` variable.

The :variable:`CMAKE_GENERATOR_PLATFORM` variable may be set, perhaps
via the :option:`cmake -A` option, to specify a target platform
name (architecture).  For example:

* ``cmake -G "Visual Studio 17 2022" -A Win32``
* ``cmake -G "Visual Studio 17 2022" -A x64``
* ``cmake -G "Visual Studio 17 2022" -A ARM``
* ``cmake -G "Visual Studio 17 2022" -A ARM64``

Toolset Selection
^^^^^^^^^^^^^^^^^

The ``v143`` toolset that comes with VS 17 2022 is selected by default.
The :variable:`CMAKE_GENERATOR_TOOLSET` option may be set, perhaps
via the :option:`cmake -T` option, to specify another toolset.

.. |VS_TOOLSET_HOST_ARCH_DEFAULT| replace::
   By default this generator uses the 64-bit variant on x64 hosts and
   the 32-bit variant otherwise.

.. include:: include/VS_TOOLSET_HOST_ARCH.rst
