CMAKE_FIND_USE_INSTALL_PREFIX
-----------------------------------

.. versionadded:: 3.24

Controls the default behavior of the following commands for whether or not to
search the locations in the :variable:`CMAKE_INSTALL_PREFIX` and
:variable:`CMAKE_STAGING_PREFIX` variables.

* :command:`find_program`
* :command:`find_library`
* :command:`find_file`
* :command:`find_path`
* :command:`find_package`

This is useful in cross-compiling environments.

Due to backwards compatibility with :variable:`CMAKE_FIND_NO_INSTALL_PREFIX`,
the behavior of the find command change based on if this variable exists.

============================== ============================ ===========
 CMAKE_FIND_USE_INSTALL_PREFIX CMAKE_FIND_NO_INSTALL_PREFIX   Search
============================== ============================ ===========
 Not Defined                      On                          NO
 Not Defined                      Off || Not Defined          YES
 Off                              On                          NO
 Off                              Off || Not Defined          NO
 On                               On                          YES
 On                               Off || Not Defined          YES
============================== ============================ ===========

By default this variable is not defined. Explicit options given to the above
commands take precedence over this variable.

See also the :variable:`CMAKE_FIND_USE_CMAKE_PATH`,
:variable:`CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH`,
:variable:`CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH`,
:variable:`CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY`,
:variable:`CMAKE_FIND_USE_PACKAGE_REGISTRY`,
and :variable:`CMAKE_FIND_USE_PACKAGE_ROOT_PATH` variables.
