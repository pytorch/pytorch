CMAKE_ECLIPSE_VERSION
---------------------

.. versionadded:: 3.6

This cache variable is used by the Eclipse project generator.  See
:manual:`cmake-generators(7)`.

When using the Eclipse project generator, CMake tries to find the Eclipse
executable and detect the version of it. Depending on the version it finds,
some features are enabled or disabled. If CMake doesn't find
Eclipse, it assumes the oldest supported version, Eclipse Callisto (3.2).
