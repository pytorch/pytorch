CMAKE_FIND_REQUIRED
-------------------

.. versionadded:: 4.1

If enabled, the following commands are treated as having the ``REQUIRED``
keyword unless provided with the ``OPTIONAL`` keyword:

* :command:`find_package`
* :command:`find_program`
* :command:`find_library`
* :command:`find_path`
* :command:`find_file`

When :command:`find_package` loads a ``Find<PackageName>.cmake``
or ``<PackageName>Config.cmake`` module, the ``CMAKE_FIND_REQUIRED``
variable is automatically unset within it to restore the default
behavior for nested find operations.  The module is free to set the
``CMAKE_FIND_REQUIRED`` variable itself to opt-in to the behavior.

Note that enabling this variable breaks some commonly used patterns.
Multiple calls to :command:`find_package` are sometimes used to obtain a
different search order to the default.

See also the :variable:`CMAKE_REQUIRE_FIND_PACKAGE_<PackageName>` for making
a :command:`find_package` call ``REQUIRED``, and for additional information on
how enabling these variables can break commonly used patterns.
