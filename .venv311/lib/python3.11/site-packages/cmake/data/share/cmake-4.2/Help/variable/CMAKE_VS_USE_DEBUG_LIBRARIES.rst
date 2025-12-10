CMAKE_VS_USE_DEBUG_LIBRARIES
----------------------------

.. versionadded:: 3.30

.. |VS_USE_DEBUG_LIBRARIES| replace:: ``CMAKE_VS_USE_DEBUG_LIBRARIES``
.. |MSVC_RUNTIME_LIBRARY| replace:: :variable:`CMAKE_MSVC_RUNTIME_LIBRARY`

.. include:: ../prop_tgt/include/VS_USE_DEBUG_LIBRARIES-PURPOSE.rst

Use :manual:`generator expressions <cmake-generator-expressions(7)>`
for per-configuration specification.  For example, the code:

.. code-block:: cmake

  set(CMAKE_VS_USE_DEBUG_LIBRARIES "$<CONFIG:Debug,Custom>")

indicates that all following targets consider their "Debug" and "Custom"
configurations to be debug configurations, and their other configurations
to be non-debug configurations.

This variable is used to initialize the :prop_tgt:`VS_USE_DEBUG_LIBRARIES`
property on all targets as they are created.  It is also propagated by
calls to the :command:`try_compile` command into its test project.

If this variable is not set then the :prop_tgt:`VS_USE_DEBUG_LIBRARIES`
property will not be set automatically.  If that property is not set then
CMake generates ``UseDebugLibraries`` using heuristics to determine which
configurations are debug configurations.  See policy :policy:`CMP0162`.
