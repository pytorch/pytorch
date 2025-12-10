CMAKE_ECLIPSE_GENERATE_SOURCE_PROJECT
-------------------------------------

.. versionadded:: 3.6

This cache variable is used by the Eclipse project generator.  See
:manual:`cmake-generators(7)`.

If this variable is set to TRUE, the Eclipse project generator will generate
an Eclipse project in :variable:`CMAKE_SOURCE_DIR` . This project can then
be used in Eclipse e.g. for the version control functionality.
``CMAKE_ECLIPSE_GENERATE_SOURCE_PROJECT`` defaults to ``FALSE``; so
nothing is written into the source directory.
