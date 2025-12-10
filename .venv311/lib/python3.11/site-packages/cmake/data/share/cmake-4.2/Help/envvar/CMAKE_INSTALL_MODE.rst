CMAKE_INSTALL_MODE
------------------

.. versionadded:: 3.22

.. include:: include/ENV_VAR.rst

The ``CMAKE_INSTALL_MODE`` environment variable allows users to operate
CMake in an alternate mode of :command:`file(INSTALL)` and :command:`install()`.

The default behavior for an installation is to copy a source file from a
source directory into a destination directory. This environment variable
however allows the user to override this behavior, causing CMake to create
symbolic links instead.

Usage Scenarios
^^^^^^^^^^^^^^^

Installing symbolic links rather than copying files can help in the following
ways:

* Conserving storage space because files do not have to be duplicated on disk.
* Changes to the source of the symbolic link are seen at the install
  destination without having to re-run the install step.
* Editing through the link at the install destination will modify the source
  of the link. This may be useful when dealing with CMake project hierarchies,
  i.e. using :module:`ExternalProject` and consistent source navigation and
  refactoring is desired across projects.

Allowed Values
^^^^^^^^^^^^^^

The following values are allowed for ``CMAKE_INSTALL_MODE``:

``COPY``, empty or unset
  Duplicate the file at its destination.  This is the default behavior.

``ABS_SYMLINK``
  Create an *absolute* symbolic link to the source file at the destination.
  Halt with an error if the link cannot be created.

``ABS_SYMLINK_OR_COPY``
  Like ``ABS_SYMLINK`` but fall back to silently copying if the symlink
  couldn't be created.

``REL_SYMLINK``
  Create a *relative* symbolic link to the source file at the destination.
  Halt with an error if the link cannot be created.

``REL_SYMLINK_OR_COPY``
  Like ``REL_SYMLINK`` but fall back to silently copying if the symlink
  couldn't be created.

``SYMLINK``
  Try as if through ``REL_SYMLINK`` and fall back to ``ABS_SYMLINK`` if the
  referenced file cannot be expressed using a relative path.
  Halt with an error if the link cannot be created.

``SYMLINK_OR_COPY``
  Like ``SYMLINK`` but fall back to silently copying if the symlink couldn't
  be created.

.. note::
  A symbolic link consists of a reference file path rather than contents of its
  own, hence there are two ways to express the relation, either by a *relative*
  or an *absolute* path.

When To Set The Environment Variable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the environment variable to take effect, it must be set during the correct
build phase(s).

* If the project calls :command:`file(INSTALL)` directly, the environment
  variable must be set during the configuration phase.
* In order to apply to :command:`install()`, the environment variable must be
  set during installation.  This could be during a build if using the
  ``install`` or ``package`` build targets, or separate from the build when
  invoking an install or running :manual:`cpack <cpack(1)>` from the command
  line.
* When using :module:`ExternalProject`, it might be required during the build
  phase, since the external project's own configure, build and install steps
  will execute during the main project's build phase.

Given the above, it is recommended to set the environment variable consistently
across all phases (configure, build and install).

Caveats
^^^^^^^

Use this environment variable with caution. The following highlights some
points to be considered:

* ``CMAKE_INSTALL_MODE`` only affects files, not directories.

* Symbolic links are not available on all platforms.

* The way this environment variable interacts with the install step of
  :module:`ExternalProject` is more complex. For further details, see that
  module's documentation.

* A symbolic link ties the destination to the source in a persistent way.
  Writing to either of the two affects both file system objects.
  This is in contrast to normal install behavior which only copies files as
  they were at the time the install was performed, with no enduring
  relationship between the source and destination of the install.

* Combining ``CMAKE_INSTALL_MODE`` with :prop_tgt:`IOS_INSTALL_COMBINED` is
  not supported.

* Changing ``CMAKE_INSTALL_MODE`` from what it was on a previous run can lead
  to unexpected results.  Moving from a non-symlinking mode to a symlinking
  mode will discard any previous file at the destination, but the reverse is
  not true.  Once a symlink exists at the destination, even if you switch to a
  non-symlink mode, the symlink will continue to exist at the destination and
  will not be replaced by an actual file.
