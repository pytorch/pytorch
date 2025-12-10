build_name
----------

Disallowed since version 3.0.  See CMake Policy :policy:`CMP0036`.

Use ``${CMAKE_SYSTEM}`` and ``${CMAKE_CXX_COMPILER}`` instead.

.. code-block:: cmake

  build_name(variable)

Sets the specified variable to a string representing the platform and
compiler settings.  These values are now available through the
:variable:`CMAKE_SYSTEM` and
:variable:`CMAKE_CXX_COMPILER <CMAKE_<LANG>_COMPILER>` variables.
