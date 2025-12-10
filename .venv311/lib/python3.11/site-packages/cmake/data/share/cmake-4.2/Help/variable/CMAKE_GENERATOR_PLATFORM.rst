CMAKE_GENERATOR_PLATFORM
------------------------

.. versionadded:: 3.1

Generator-specific target platform specification provided by user.

Some CMake generators support a target platform name to be given
to the native build system to choose a compiler toolchain.
If the user specifies a platform name (e.g. via the :option:`cmake -A`
option or via the :envvar:`CMAKE_GENERATOR_PLATFORM` environment variable)
the value will be available in this variable.

The value of this variable should never be modified by project code.
A toolchain file specified by the :variable:`CMAKE_TOOLCHAIN_FILE`
variable may initialize ``CMAKE_GENERATOR_PLATFORM``.  Once a given
build tree has been initialized with a particular value for this
variable, changing the value has undefined behavior.

Platform specification is supported only on specific generators:

* For :ref:`Visual Studio Generators` with VS 2005 and above this
  specifies the target architecture.

* For :generator:`Green Hills MULTI` this specifies the target architecture.

See native build system documentation for allowed platform names.

.. _`Visual Studio Platform Selection`:

Visual Studio Platform Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :ref:`Visual Studio Generators` support platform specification
using one of these forms:

* ``platform``
* ``platform[,key=value]*``
* ``key=value[,key=value]*``

The ``platform`` specifies the target platform (VS target architecture),
such as ``x64``, ``ARM64``, or ``Win32``.  The selected platform
name is provided in the :variable:`CMAKE_VS_PLATFORM_NAME` variable.

The ``key=value`` pairs form a comma-separated list of options to
specify generator-specific details of the platform selection.
Supported pairs are:

``version=<version>``
  .. versionadded:: 3.27

  Specify the Windows SDK version to use.  This is supported by VS 2015 and
  above when targeting Windows or Windows Store.  CMake will set the
  :variable:`CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION` variable to the
  selected SDK version.

  The ``<version>`` may be one of:

  ``10.0``
    Specify that any 10.0 SDK version may be used, and let Visual Studio
    pick one.  This is supported by VS 2019 and above.

  ``10.0.<build>.<increment>``
    Specify the exact 4-component SDK version, e.g., ``10.0.19041.0``.
    The specified version of the SDK must be installed.  It may not exceed
    the value of :variable:`CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION_MAXIMUM`,
    if that variable is set.

  ``8.1``
    Specify the 8.1 SDK version.  This is always supported by VS 2015.
    On VS 2017 and above the 8.1 SDK must be installed.

  If the ``version`` field is not specified, CMake selects a version as
  described in the :variable:`CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION`
  variable documentation.
