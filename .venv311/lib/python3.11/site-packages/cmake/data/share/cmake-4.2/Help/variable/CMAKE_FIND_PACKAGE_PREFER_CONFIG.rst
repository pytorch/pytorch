CMAKE_FIND_PACKAGE_PREFER_CONFIG
---------------------------------

.. versionadded:: 3.15

Tell :command:`find_package` to try "Config" mode before "Module" mode if no
mode was specified.

The command :command:`find_package` operates without an explicit mode when
the reduced signature is used without the ``MODULE`` option. In this case,
by default, CMake first tries Module mode by searching for a
``Find<pkg>.cmake`` module.  If it fails, CMake then searches for the package
using Config mode.

Set ``CMAKE_FIND_PACKAGE_PREFER_CONFIG`` to ``TRUE`` to tell
:command:`find_package` to first search using Config mode before falling back
to Module mode.

This variable may be useful when a developer has compiled a custom version of
a common library and wishes to link it to a dependent project.  If this
variable is set to ``TRUE``, it would prevent a dependent project's call
to :command:`find_package` from selecting the default library located by the
system's ``Find<pkg>.cmake`` module before finding the developer's custom
built library.

Once this variable is set, it is the responsibility of the exported
``<pkg>Config.cmake`` files to provide the same result variables as the
``Find<pkg>.cmake`` modules so that dependent projects can use them
interchangeably.
