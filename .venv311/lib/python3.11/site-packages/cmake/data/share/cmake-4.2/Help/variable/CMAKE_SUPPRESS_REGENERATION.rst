CMAKE_SUPPRESS_REGENERATION
---------------------------

.. versionadded:: 3.12

If ``CMAKE_SUPPRESS_REGENERATION`` is ``OFF``, which is default, then CMake
adds a special target on which all other targets depend that checks the build
system and optionally re-runs CMake to regenerate the build system when
the target specification source changes.

If this variable evaluates to ``ON`` at the end of the top-level
``CMakeLists.txt`` file, CMake will not add the regeneration target to the
build system or perform any build system checks.
