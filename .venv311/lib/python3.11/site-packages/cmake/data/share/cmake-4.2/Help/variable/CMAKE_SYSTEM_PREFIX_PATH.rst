CMAKE_SYSTEM_PREFIX_PATH
------------------------

:ref:`Semicolon-separated list <CMake Language Lists>` of directories specifying installation
*prefixes* to be searched by the :command:`find_package`,
:command:`find_program`, :command:`find_library`, :command:`find_file`, and
:command:`find_path` commands.  Each command will add appropriate
subdirectories (like ``bin``, ``lib``, or ``include``) as specified in its own
documentation.

By default this contains the system directories for the current system, the
:variable:`CMAKE_INSTALL_PREFIX`, and the :variable:`CMAKE_STAGING_PREFIX`.
The installation and staging prefixes may be excluded by setting
the :variable:`CMAKE_FIND_NO_INSTALL_PREFIX` variable before the
first :command:`project` invocation.

The system directories that are contained in ``CMAKE_SYSTEM_PREFIX_PATH`` are
locations that typically include installed software. An example being
``/usr/local`` for UNIX based platforms. In addition to standard platform
locations, CMake will also add values to ``CMAKE_SYSTEM_PREFIX_PATH`` based on
environment variables. The environment variables and search locations that
CMake uses may evolve over time, as platforms and their conventions also
evolve. The following provides an indicative list of environment variables
and locations that CMake searches, but they are subject to change:


CrayLinuxEnvironment:
  * ``ENV{SYSROOT_DIR}/``
  * ``ENV{SYSROOT_DIR}/usr``
  * ``ENV{SYSROOT_DIR}/usr/local``

Darwin:
  * ``ENV{SDKROOT}/usr`` When ``CMAKE_OSX_SYSROOT`` is not explicitly specified.

OpenBSD:
  * ``ENV{LOCALBASE}``

Unix:
  * ``ENV{CONDA_PREFIX}`` when using a conda compiler

MSYSTEM environment with MinGW toolchain:
  .. versionadded:: 3.28

  * ``ENV{MSYSTEM_PREFIX}/local``
  * ``ENV{MSYSTEM_PREFIX}``

Windows:
  * ``ENV{ProgramW6432}``
  * ``ENV{ProgramFiles}``
  * ``ENV{ProgramFiles(x86)}``
  * ``ENV{SystemDrive}/Program Files``
  * ``ENV{SystemDrive}/Program Files (x86)``


``CMAKE_SYSTEM_PREFIX_PATH`` is *not* intended to be modified by the project;
use :variable:`CMAKE_PREFIX_PATH` for this.

See also :variable:`CMAKE_SYSTEM_INCLUDE_PATH`,
:variable:`CMAKE_SYSTEM_LIBRARY_PATH`, :variable:`CMAKE_SYSTEM_PROGRAM_PATH`,
and :variable:`CMAKE_SYSTEM_IGNORE_PATH`.
