XCODE_XCCONFIG
--------------

.. versionadded:: 3.24

If set, the :generator:`Xcode` generator will register the specified
file as a target-level XCConfig file. For global XCConfig files see
the :variable:`CMAKE_XCODE_XCCONFIG` variable.

This feature is intended to ease migration from native Xcode projects
to CMake projects.

Contents of ``XCODE_XCCONFIG`` may use
:manual:`generator expressions <cmake-generator-expressions(7)>`.
