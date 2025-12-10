CMAKE_UNITY_BUILD_RELOCATABLE
-----------------------------

.. versionadded:: 4.0

This variable is used to initialize the :prop_tgt:`UNITY_BUILD_RELOCATABLE`
property of targets when they are created.  Setting it to true causes
sources generated for :variable:`CMAKE_UNITY_BUILD` to ``#include`` the
original source files using relative paths where possible.
