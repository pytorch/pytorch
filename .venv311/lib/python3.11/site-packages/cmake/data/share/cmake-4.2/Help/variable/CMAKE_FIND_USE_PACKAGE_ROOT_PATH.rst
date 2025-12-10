CMAKE_FIND_USE_PACKAGE_ROOT_PATH
--------------------------------

.. versionadded:: 3.16

Controls the default behavior of the following commands for whether or not to
search paths provided by :variable:`<PackageName>_ROOT` variables:

* :command:`find_program`
* :command:`find_library`
* :command:`find_file`
* :command:`find_path`
* :command:`find_package`

By default this variable is not set, which is equivalent to it having
a value of ``TRUE``.  Explicit options given to the above commands
take precedence over this variable.

See also the :variable:`CMAKE_FIND_USE_CMAKE_PATH`,
:variable:`CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH`,
:variable:`CMAKE_FIND_USE_INSTALL_PREFIX`,
:variable:`CMAKE_FIND_USE_CMAKE_SYSTEM_PATH`,
:variable:`CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH`,
:variable:`CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY`,
and :variable:`CMAKE_FIND_USE_PACKAGE_REGISTRY` variables.
