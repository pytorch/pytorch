CMAKE_XCODE_XCCONFIG
--------------------

.. versionadded:: 3.24

If set, the :generator:`Xcode` generator will register the specified
file as a global XCConfig file. For target-level XCConfig files see
the :prop_tgt:`XCODE_XCCONFIG` target property.

This feature is intended to ease migration from native Xcode projects
to CMake projects.

Contents of ``CMAKE_XCODE_XCCONFIG`` may use
:manual:`generator expressions <cmake-generator-expressions(7)>`.
