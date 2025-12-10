CMake uses this environment variable value, in combination with its own
builtin default flags for the toolchain, to initialize and store the
|CMAKE_LANG_FLAGS| cache entry.
This occurs the first time a build tree is configured for language |LANG|.
For any configuration run (including the first), the environment variable
will be ignored if the |CMAKE_LANG_FLAGS| variable is already defined.
