CMAKE_ECLIPSE_GENERATE_LINKED_RESOURCES
---------------------------------------

.. versionadded:: 3.6

This cache variable is used by the Eclipse project generator.  See
:manual:`cmake-generators(7)`.

The Eclipse project generator generates so-called linked resources
e.g. to the subproject root dirs in the source tree or to the source files
of targets.
This can be disabled by setting this variable to FALSE.
