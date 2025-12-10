Visual Studio 14 2015
---------------------

.. deprecated:: 4.2

  This generator is deprecated and will be removed in a future version
  of CMake.  It will still be possible to build with VS 14 2015 tools
  using the :generator:`Visual Studio 15 2017` (or above) generator
  with :variable:`CMAKE_GENERATOR_TOOLSET` set to ``v140``, or by
  using the :generator:`NMake Makefiles` generator.

.. versionadded:: 3.1

Generates Visual Studio 14 (VS 2015) project files.

Project Types
^^^^^^^^^^^^^

Only Visual C++ and C# projects may be generated (and Fortran with
Intel compiler integration).  Other types of projects (JavaScript,
Powershell, Python, etc.) are not supported.

Platform Selection
^^^^^^^^^^^^^^^^^^

The default target platform name (architecture) is ``Win32``.

The :variable:`CMAKE_GENERATOR_PLATFORM` variable may be set, perhaps
via the :option:`cmake -A` option, to specify a target platform
name (architecture).  For example:

* ``cmake -G "Visual Studio 14 2015" -A Win32``
* ``cmake -G "Visual Studio 14 2015" -A x64``
* ``cmake -G "Visual Studio 14 2015" -A ARM``

.. versionchanged:: 4.0

  Previously, for compatibility with CMake versions prior to 3.1,
  one could specify a target platform name optionally at the
  end of the generator name.  This has been removed.
  This was supported only for:

  ``Visual Studio 14 2015 Win64``
    Specify target platform ``x64``.

  ``Visual Studio 14 2015 ARM``
    Specify target platform ``ARM``.

Toolset Selection
^^^^^^^^^^^^^^^^^

The ``v140`` toolset that comes with Visual Studio 14 2015 is selected by
default.  The :variable:`CMAKE_GENERATOR_TOOLSET` option may be set, perhaps
via the :option:`cmake -T` option, to specify another toolset.

.. |VS_TOOLSET_HOST_ARCH_DEFAULT| replace::
   By default this generator uses the 32-bit variant even on a 64-bit host.

.. include:: include/VS_TOOLSET_HOST_ARCH_LEGACY.rst

.. _`Windows 10 SDK Maximum Version for VS 2015`:

Windows 10 SDK Maximum Version for VS 2015
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.19

Microsoft stated in a "Windows 10 October 2018 Update" blog post that Windows
10 SDK versions (15063, 16299, 17134, 17763) are not supported by VS 2015 and
are only supported by VS 2017 and later.  Therefore by default CMake
automatically ignores Windows 10 SDKs beyond ``10.0.14393.0``.

However, there are other recommendations for certain driver/Win32 builds that
indicate otherwise.  A user can override this behavior by either setting the
:variable:`CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION_MAXIMUM` to a false value
or setting the :variable:`CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION_MAXIMUM` to
the string value of the required maximum (e.g. ``10.0.15063.0``).
