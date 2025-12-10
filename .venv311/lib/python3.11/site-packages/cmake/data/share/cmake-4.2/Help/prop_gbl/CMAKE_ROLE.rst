CMAKE_ROLE
----------

.. versionadded:: 3.14

Tells what mode the current running script is in. Could be one of several
values:

``PROJECT``
  Running in project mode (processing a ``CMakeLists.txt`` file).

``SCRIPT``
  Running in :ref:`cmake -P <Script Processing Mode>` script mode.

``FIND_PACKAGE``
  Running in :ref:`cmake --find-package <Find-Package Tool Mode>` mode.

``CTEST``
  Running in CTest script mode.

``CPACK``
  Running in CPack.

See Also
^^^^^^^^

* The :variable:`CMAKE_SCRIPT_MODE_FILE` variable.
