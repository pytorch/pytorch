CMAKE_USER_MAKE_RULES_OVERRIDE
------------------------------

Specify a CMake file that overrides platform information.

CMake loads the specified file while enabling support for each
language from either the :command:`project` or :command:`enable_language`
commands.  It is loaded after CMake's builtin compiler and platform information
modules have been loaded but before the information is used.  The file
may set platform information variables to override CMake's defaults.
See :variable:`CMAKE_USER_MAKE_RULES_OVERRIDE_<LANG>` for the language-specific
version of this variable.

This feature is intended for use only in overriding information
variables that must be set before CMake builds its first test project
to check that the compiler for a language works.  It should not be
used to load a file in cases that a normal :command:`include` will work.  Use
it only as a last resort for behavior that cannot be achieved any
other way.  For example, one may set the
:variable:`CMAKE_C_FLAGS_INIT <CMAKE_<LANG>_FLAGS_INIT>` variable
to change the default value used to initialize the
:variable:`CMAKE_C_FLAGS <CMAKE_<LANG>_FLAGS>` variable
before it is cached.  The override file should NOT be used to set anything
that could be set after languages are enabled, such as variables like
:variable:`CMAKE_RUNTIME_OUTPUT_DIRECTORY` that affect the placement of
binaries.  Information set in the file will be used for :command:`try_compile`
and :command:`try_run` builds too.
