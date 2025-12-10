CMAKE_KATE_MAKE_ARGUMENTS
-------------------------

.. versionadded:: 3.0

This cache variable is used by the Kate project generator.  See
:manual:`cmake-generators(7)`.

This variable holds arguments which are used when Kate invokes the make
tool. By default it is initialized to hold flags to enable parallel builds
(using -j typically).
