Swift_MODULE_DIRECTORY
----------------------

.. versionadded:: 3.15

Specify output directory for Swift modules provided by the target.

If the target contains Swift source files, this specifies the directory in which
the modules will be placed.  When this property is not set, the modules will be
placed in the build directory corresponding to the target's source directory.
If the variable :variable:`CMAKE_Swift_MODULE_DIRECTORY` is set when a target is
created its value is used to initialize this property.

.. versionadded:: 4.0

  The property value may use
  :manual:`generator expressions <cmake-generator-expressions(7)>`.
  Multi-configuration generators (:generator:`Ninja Multi-Config`) append a
  per-configuration subdirectory to the specified directory unless a generator
  expression is used.

.. warning::

  The :generator:`Xcode` generator ignores this property.
