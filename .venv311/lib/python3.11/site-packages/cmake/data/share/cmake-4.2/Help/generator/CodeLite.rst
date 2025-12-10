CodeLite
----------

.. deprecated:: 3.27

  Support for :ref:`Extra Generators` is deprecated and will be removed from
  a future version of CMake.  IDEs may use the :manual:`cmake-file-api(7)`
  to view CMake-generated project build trees.

Generates CodeLite project files.

Project files for CodeLite will be created in the top directory and
in every subdirectory which features a CMakeLists.txt file containing
a :command:`project` call.
The appropriate make program can build the
project through the default ``all`` target.  An ``install`` target
is also provided.

.. versionadded:: 3.7
 The :variable:`CMAKE_CODELITE_USE_TARGETS` variable may be set to ``ON``
 to change the default behavior from projects to targets as the basis
 for project files.

This "extra" generator may be specified as:

``CodeLite - MinGW Makefiles``
 Generate with :generator:`MinGW Makefiles`.

``CodeLite - NMake Makefiles``
 Generate with :generator:`NMake Makefiles`.

``CodeLite - Ninja``
 Generate with :generator:`Ninja`.

``CodeLite - Unix Makefiles``
 Generate with :generator:`Unix Makefiles`.
