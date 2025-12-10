VS_USE_DEBUG_LIBRARIES
----------------------

.. versionadded:: 3.30

.. |VS_USE_DEBUG_LIBRARIES| replace:: ``VS_USE_DEBUG_LIBRARIES``
.. |MSVC_RUNTIME_LIBRARY| replace:: :prop_tgt:`MSVC_RUNTIME_LIBRARY`

.. include:: include/VS_USE_DEBUG_LIBRARIES-PURPOSE.rst

Use :manual:`generator expressions <cmake-generator-expressions(7)>`
for per-configuration specification.  For example, the code:

.. code-block:: cmake

  add_executable(foo foo.c)
  set_property(TARGET foo PROPERTY
    VS_USE_DEBUG_LIBRARIES "$<CONFIG:Debug,Custom>")

indicates that target ``foo`` considers its "Debug" and "Custom"
configurations to be debug configurations, and its other configurations
to be non-debug configurations.

The property is initialized from the value of the
:variable:`CMAKE_VS_USE_DEBUG_LIBRARIES` variable, if it is set.
If the property is not set then CMake generates ``UseDebugLibraries`` using
heuristics to determine which configurations are debug configurations.
See policy :policy:`CMP0162`.
