.. cmake-manual-description: CPack Command-Line Reference

cpack(1)
********

Synopsis
========

.. parsed-literal::

 cpack [<options>]

Description
===========

The :program:`cpack` executable is the CMake packaging program.  It generates
installers and source packages in a variety of formats.

For each installer or package format, :program:`cpack` has a specific backend,
called "generator". A generator is responsible for generating the required
inputs and invoking the specific package creation tools. These installer
or package generators are not to be confused with the makefile generators
of the :manual:`cmake <cmake(1)>` command.

All supported generators are specified in the :manual:`cpack-generators
<cpack-generators(7)>` manual.  The command ``cpack --help`` prints a
list of generators supported for the target platform.  Which of them are
to be used can be selected through the :variable:`CPACK_GENERATOR` variable
or through the command-line option :option:`-G <cpack -G>`.

The :program:`cpack` program is steered by a configuration file written in the
:manual:`CMake language <cmake-language(7)>`. Unless chosen differently
through the command-line option :option:`--config <cpack --config>`, the
file ``CPackConfig.cmake`` in the current directory is used.

In the standard CMake workflow, the file ``CPackConfig.cmake`` is generated
by the :manual:`cmake <cmake(1)>` executable, provided the :module:`CPack`
module is included by the project's ``CMakeLists.txt`` file.

Options
=======

.. program:: cpack

.. option:: -G <generators>

  ``<generators>`` is a :ref:`semicolon-separated list <CMake Language Lists>`
  of generator names.  :program:`cpack` will iterate through this list and produce
  package(s) in that generator's format according to the details provided in
  the ``CPackConfig.cmake`` configuration file.  If this option is not given,
  the :variable:`CPACK_GENERATOR` variable determines the default set of
  generators that will be used.

.. option:: -C <configurations>

  Specify the project configuration(s) to be packaged (e.g. ``Debug``,
  ``Release``, etc.), where ``<configurations>`` is a
  :ref:`semicolon-separated list <CMake Language Lists>`.
  When the CMake project uses a multi-configuration
  generator such as :generator:`Xcode` or
  :ref:`Visual Studio <Visual Studio Generators>`, this option is needed to tell
  :program:`cpack` which built executables to include in the package.
  The user is responsible for ensuring that the configuration(s) listed
  have already been built before invoking :program:`cpack`.

.. option:: -D <var>=<value>

  Set a CPack variable.  This will override any value set for ``<var>`` in the
  input file read by :program:`cpack`.

.. option:: --config <configFile>

  Specify the configuration file read by :program:`cpack` to provide the packaging
  details.  By default, ``CPackConfig.cmake`` in the current directory will
  be used.

.. option:: -V, --verbose

  Run :program:`cpack` with verbose output.  This can be used to show more details
  from the package generation tools and is suitable for project developers.

.. option:: --debug

  Run :program:`cpack` with debug output.  This option is intended mainly for the
  developers of :program:`cpack` itself and is not normally needed by project
  developers.

.. option:: --trace

  Put the underlying cmake scripts in trace mode.

.. option:: --trace-expand

  Put the underlying cmake scripts in expanded trace mode.

.. option:: -P <packageName>

  Override/define the value of the :variable:`CPACK_PACKAGE_NAME` variable used
  for packaging.  Any value set for this variable in the ``CPackConfig.cmake``
  file will then be ignored.

.. option:: -R <packageVersion>

  Override/define the value of the :variable:`CPACK_PACKAGE_VERSION`
  variable used for packaging.  It will override a value set in the
  ``CPackConfig.cmake`` file or one automatically computed from
  :variable:`CPACK_PACKAGE_VERSION_MAJOR`,
  :variable:`CPACK_PACKAGE_VERSION_MINOR` and
  :variable:`CPACK_PACKAGE_VERSION_PATCH`.

.. option:: -B <packageDirectory>

  Override/define :variable:`CPACK_PACKAGE_DIRECTORY`, which controls the
  directory where CPack will perform its packaging work.  The resultant
  package(s) will be created at this location by default and a
  ``_CPack_Packages`` subdirectory will also be created below this directory to
  use as a working area during package creation.

.. option:: --vendor <vendorName>

  Override/define :variable:`CPACK_PACKAGE_VENDOR`.

.. option:: --preset <presetName>

  Use a preset from :manual:`cmake-presets(7)`.

.. option:: --list-presets

  List presets from :manual:`cmake-presets(7)`.

.. include:: include/OPTIONS_HELP.rst

See Also
========

.. include:: include/LINKS.rst
