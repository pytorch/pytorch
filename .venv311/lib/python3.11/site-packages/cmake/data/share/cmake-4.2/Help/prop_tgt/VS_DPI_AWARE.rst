VS_DPI_AWARE
------------

.. versionadded:: 3.16

Set the Manifest Tool -> Input and Output -> DPI Awareness in the Visual Studio
target project properties.

Valid values are ``PerMonitor``, ``ON``, or ``OFF``.

For example:

.. code-block:: cmake

  add_executable(myproject myproject.cpp)
  set_property(TARGET myproject PROPERTY VS_DPI_AWARE "PerMonitor")
