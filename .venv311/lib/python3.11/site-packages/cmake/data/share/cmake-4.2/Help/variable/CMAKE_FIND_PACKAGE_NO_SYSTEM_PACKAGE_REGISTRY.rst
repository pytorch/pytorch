CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY
---------------------------------------------

.. versionadded:: 3.1

.. deprecated:: 3.16

  Use the :variable:`CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY` variable instead.

By default this variable is not set. If neither
:variable:`CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY` nor
``CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY`` is set, then
:command:`find_package()` will use the :ref:`System Package Registry`
unless the ``NO_CMAKE_SYSTEM_PACKAGE_REGISTRY`` option is provided.

``CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY`` is ignored if
:variable:`CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY` is set.

In some cases, it is not desirable to use the
:ref:`System Package Registry` when searching for packages. If the
``CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY`` variable is
``TRUE``, all the :command:`find_package` commands will skip
the :ref:`System Package Registry` as if they were called with the
``NO_CMAKE_SYSTEM_PACKAGE_REGISTRY`` argument.

See also :ref:`Disabling the Package Registry`.
