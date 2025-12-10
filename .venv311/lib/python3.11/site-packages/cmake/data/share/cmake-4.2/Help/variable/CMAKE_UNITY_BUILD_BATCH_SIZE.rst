CMAKE_UNITY_BUILD_BATCH_SIZE
----------------------------

.. versionadded:: 3.16

This variable is used to initialize the :prop_tgt:`UNITY_BUILD_BATCH_SIZE`
property of targets when they are created.  It specifies the default upper
limit on the number of source files that may be combined in any one unity
source file when unity builds are enabled for a target.
