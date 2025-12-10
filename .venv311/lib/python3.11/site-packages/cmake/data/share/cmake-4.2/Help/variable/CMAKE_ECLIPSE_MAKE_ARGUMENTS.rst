CMAKE_ECLIPSE_MAKE_ARGUMENTS
----------------------------

.. versionadded:: 3.6

This cache variable is used by the Eclipse project generator.  See
:manual:`cmake-generators(7)`.

This variable holds arguments which are used when Eclipse invokes the make
tool. By default it is initialized to hold flags to enable parallel builds
(using -j typically).
