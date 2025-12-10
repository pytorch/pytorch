CMAKE_XCODE_GENERATE_TOP_LEVEL_PROJECT_ONLY
-------------------------------------------

.. versionadded:: 3.11

If enabled, the :generator:`Xcode` generator will generate only a
single Xcode project file for the topmost :command:`project()` command
instead of generating one for every ``project()`` command.

This could be useful to speed up the CMake generation step for
large projects and to work-around a bug in the ``ZERO_CHECK`` logic.
