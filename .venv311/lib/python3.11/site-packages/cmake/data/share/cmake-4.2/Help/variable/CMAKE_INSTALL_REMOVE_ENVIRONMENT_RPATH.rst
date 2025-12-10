CMAKE_INSTALL_REMOVE_ENVIRONMENT_RPATH
--------------------------------------

.. versionadded:: 3.16

Sets the default for whether toolchain-defined rpaths should be removed during
installation.

``CMAKE_INSTALL_REMOVE_ENVIRONMENT_RPATH`` is a boolean that provides the
default value for the :prop_tgt:`INSTALL_REMOVE_ENVIRONMENT_RPATH` property
of all subsequently created targets.
