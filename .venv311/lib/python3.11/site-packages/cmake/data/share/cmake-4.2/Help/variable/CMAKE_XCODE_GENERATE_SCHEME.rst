CMAKE_XCODE_GENERATE_SCHEME
---------------------------

.. versionadded:: 3.9

If enabled, the :generator:`Xcode` generator will generate schema files.  These
are useful to invoke analyze, archive, build-for-testing and test
actions from the command line.

This variable initializes the
:prop_tgt:`XCODE_GENERATE_SCHEME`
target property on all targets.
