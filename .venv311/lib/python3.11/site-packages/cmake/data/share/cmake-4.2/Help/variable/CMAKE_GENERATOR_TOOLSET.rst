CMAKE_GENERATOR_TOOLSET
-----------------------

Native build system toolset specification provided by user.

Some CMake generators support a toolset specification to tell the
native build system how to choose a compiler.  If the user specifies
a toolset (e.g. via the :option:`cmake -T` option or via
the :envvar:`CMAKE_GENERATOR_TOOLSET` environment variable) the value
will be available in this variable.

The value of this variable should never be modified by project code.
A toolchain file specified by the :variable:`CMAKE_TOOLCHAIN_FILE`
variable may initialize ``CMAKE_GENERATOR_TOOLSET``.  Once a given
build tree has been initialized with a particular value for this
variable, changing the value has undefined behavior.

Toolset specification is supported only on specific generators:

* :ref:`Visual Studio Generators` for VS 2010 and above
* The :generator:`Xcode` generator for Xcode 3.0 and above
* The :generator:`Green Hills MULTI` generator

See native build system documentation for allowed toolset names.

Visual Studio Toolset Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :ref:`Visual Studio Generators` support toolset specification
using one of these forms:

* ``toolset``
* ``toolset[,key=value]*``
* ``key=value[,key=value]*``

The ``toolset`` specifies the toolset name.  The selected toolset name
is provided in the :variable:`CMAKE_VS_PLATFORM_TOOLSET` variable.

The ``key=value`` pairs form a comma-separated list of options to
specify generator-specific details of the toolset selection.
Supported pairs are:

``cuda=<version>|<path>``
  Specify the CUDA toolkit version to use or the path to a
  standalone CUDA toolkit directory.  Supported by VS 2010
  and above. The version can only be used with the CUDA
  toolkit VS integration globally installed.
  See the :variable:`CMAKE_VS_PLATFORM_TOOLSET_CUDA` and
  :variable:`CMAKE_VS_PLATFORM_TOOLSET_CUDA_CUSTOM_DIR` variables.

``fortran=<compiler>``
  .. versionadded:: 3.29

  Specify the Fortran compiler to use, among those that have the required
  Visual Studio Integration feature installed.  The value may be one of:

  ``ifort``
    Intel classic Fortran compiler.

  ``ifx``
    Intel oneAPI Fortran compiler.

  See the :variable:`CMAKE_VS_PLATFORM_TOOLSET_FORTRAN` variable.

``host=<arch>``
  Specify the host tools architecture as ``x64`` or ``x86``.
  Supported by VS 2013 and above.
  See the :variable:`CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE`
  variable.

``version=<version>``
  Specify the toolset version to use.  Supported by VS 2017
  and above with the specified toolset installed.
  See the :variable:`CMAKE_VS_PLATFORM_TOOLSET_VERSION` variable.

``VCTargetsPath=<path>``
  Specify an alternative ``VCTargetsPath`` value for Visual Studio
  project files.  This allows use of VS platform extension configuration
  files (``.props`` and ``.targets``) that are not installed with VS.

Visual Studio Toolset Customization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**These are unstable interfaces with no compatibility guarantees**
because they hook into undocumented internal CMake implementation details.
Institutions may use these to internally maintain support for non-public
Visual Studio platforms and toolsets, but must accept responsibility to
make updates as changes are made to CMake.

Additional ``key=value`` pairs are available:

``customFlagTableDir=<path>``
  .. versionadded:: 3.21

  Specify the absolute path to a directory from which to load custom
  flag tables stored as JSON documents with file names of the form
  ``<platform>_<toolset>_<tool>.json`` or ``<platform>_<tool>.json``,
  where ``<platform>`` is the :variable:`CMAKE_VS_PLATFORM_NAME`,
  ``<toolset>`` is the :variable:`CMAKE_VS_PLATFORM_TOOLSET`,
  and ``<tool>`` is the tool for which the flag table is meant.
  **This naming pattern is an internal CMake implementation detail.**
  The ``<tool>`` names are undocumented.  The format of the ``.json``
  flag table files is undocumented.
