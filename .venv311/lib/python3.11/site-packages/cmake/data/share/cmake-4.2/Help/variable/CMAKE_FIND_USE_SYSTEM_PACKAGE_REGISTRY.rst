CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY
--------------------------------------

.. versionadded:: 3.16

Controls searching the :ref:`System Package Registry` by the
:command:`find_package` command.

By default this variable is not set and the behavior will fall back
to that determined by the deprecated
:variable:`CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY` variable.
If that is also not set, then :command:`find_package()` will use the
:ref:`System Package Registry` unless the ``NO_CMAKE_SYSTEM_PACKAGE_REGISTRY``
option is provided.

This variable takes precedence over
:variable:`CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY` when both are set.

In some cases, for example to locate only user specific installations, it
is not desirable to use the :ref:`System Package Registry` when searching
for packages. If the ``CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY``
variable is ``FALSE``, all the :command:`find_package` commands will skip
the :ref:`System Package Registry` as if they were called with the
``NO_CMAKE_SYSTEM_PACKAGE_REGISTRY`` argument.

See also :ref:`Disabling the Package Registry`.

See also the :variable:`CMAKE_FIND_USE_CMAKE_PATH`,
:variable:`CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH`,
:variable:`CMAKE_FIND_USE_INSTALL_PREFIX`,
:variable:`CMAKE_FIND_USE_CMAKE_SYSTEM_PATH`,
:variable:`CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH`,
:variable:`CMAKE_FIND_USE_PACKAGE_REGISTRY`,
and :variable:`CMAKE_FIND_USE_PACKAGE_ROOT_PATH` variables.
