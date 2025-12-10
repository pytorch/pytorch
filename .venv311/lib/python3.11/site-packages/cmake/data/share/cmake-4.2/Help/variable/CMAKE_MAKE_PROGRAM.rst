CMAKE_MAKE_PROGRAM
------------------

Tool that can launch the native build system.
The value may be the full path to an executable or just the tool
name if it is expected to be in the ``PATH``.

The tool selected depends on the :variable:`CMAKE_GENERATOR` used
to configure the project:

* The :ref:`Makefile Generators` set this to ``make``, ``gmake``, or
  a generator-specific tool (e.g. ``nmake`` for :generator:`NMake Makefiles`).

  These generators store ``CMAKE_MAKE_PROGRAM`` in the CMake cache
  so that it may be edited by the user.

* The :generator:`Ninja` generator sets this to ``ninja``.

  This generator stores ``CMAKE_MAKE_PROGRAM`` in the CMake cache
  so that it may be edited by the user.

* The :generator:`Xcode` generator sets this to ``xcodebuild``.

  This generator prefers to lookup the build tool at build time
  rather than to store ``CMAKE_MAKE_PROGRAM`` in the CMake cache
  ahead of time.  This is because ``xcodebuild`` is easy to find.

  For compatibility with versions of CMake prior to 3.2, if
  a user or project explicitly adds ``CMAKE_MAKE_PROGRAM`` to
  the CMake cache then CMake will use the specified value.

* The :ref:`Visual Studio Generators` set this to the full path to
  ``MSBuild.exe`` or ``devenv.com``.
  (See also variables
  :variable:`CMAKE_VS_MSBUILD_COMMAND` and
  :variable:`CMAKE_VS_DEVENV_COMMAND`.

  These generators prefer to lookup the build tool at build time
  rather than to store ``CMAKE_MAKE_PROGRAM`` in the CMake cache
  ahead of time.  This is because the tools are version-specific
  and can be located using the Visual Studio Installer.  It is also
  necessary because the proper build tool may depend on the
  project content (e.g. the Intel Fortran plugin to Visual Studio
  requires ``devenv.com`` to build its ``.vfproj`` project files
  even though ``MSBuild.exe`` is normally preferred to support
  the :variable:`CMAKE_GENERATOR_TOOLSET`).

  For compatibility with versions of CMake prior to 3.0, if
  a user or project explicitly adds ``CMAKE_MAKE_PROGRAM`` to
  the CMake cache then CMake will use the specified value if
  possible.

* The :generator:`Green Hills MULTI` generator sets this to the full
  path to ``gbuild.exe(Windows)`` or ``gbuild(Linux)`` based upon
  the toolset being used.

  Once the generator has initialized a particular value for this
  variable, changing the value has undefined behavior.

The ``CMAKE_MAKE_PROGRAM`` variable is set for use by project code.
The value is also used by the :option:`cmake --build` and
:option:`ctest --build-and-test` tools to launch the native
build process.
