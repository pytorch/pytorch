CMAKE_GENERATOR_INSTANCE
------------------------

.. versionadded:: 3.11

Generator-specific instance specification provided by user.

Some CMake generators support selection of an instance of the native build
system when multiple instances are available.  If the user specifies an
instance (e.g. by setting this cache entry or via the
:envvar:`CMAKE_GENERATOR_INSTANCE` environment variable), or after a default
instance is chosen when a build tree is first configured, the value will be
available in this variable.

The value of this variable should never be modified by project code.
A toolchain file specified by the :variable:`CMAKE_TOOLCHAIN_FILE`
variable may initialize ``CMAKE_GENERATOR_INSTANCE`` as a cache entry.
Once a given build tree has been initialized with a particular value
for this variable, changing the value has undefined behavior.

Instance specification is supported only on specific generators.

Visual Studio Instance Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`Visual Studio Generators` support instance specification for
Visual Studio 2017 and above.  The ``CMAKE_GENERATOR_INSTANCE`` variable
may be set as a cache entry selecting an instance of Visual Studio
via one of the following forms:

* ``location``
* ``location[,key=value]*``
* ``key=value[,key=value]*``

The ``location`` specifies the absolute path to the top-level directory
of the VS installation.

The ``key=value`` pairs form a comma-separated list of options to
specify details of the instance selection.
Supported pairs are:

``version=<major>.<minor>.<date>.<build>``
  .. versionadded:: 3.23

  Specify the 4-component VS Build Version, a.k.a. Build Number.

  .. include:: include/CMAKE_VS_VERSION_BUILD_NUMBER_COMPONENTS.rst

.. versionadded:: 3.23

  A portable VS instance, which is not known to the Visual Studio Installer,
  may be specified by providing both ``location`` and ``version=``.

If the value of ``CMAKE_GENERATOR_INSTANCE`` is not specified explicitly
by the user or a toolchain file, CMake queries the Visual Studio Installer
to locate VS instances, chooses one, and sets the variable as a cache entry
to hold the value persistently.  If an environment variable of the form
``VS##0COMNTOOLS``, where ``##`` the Visual Studio major version number,
is set and points to the ``Common7/Tools`` directory within one of the
VS instances, that instance will be used.  Otherwise, if more than one
VS instance is installed we do not define which one is chosen by default.

The VS version build number of the selected VS instance is provided in
the :variable:`CMAKE_VS_VERSION_BUILD_NUMBER` variable.
