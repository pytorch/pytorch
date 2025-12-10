CMAKE_SYSTEM_PROCESSOR
----------------------

When not cross-compiling, this variable has the same value as the
:variable:`CMAKE_HOST_SYSTEM_PROCESSOR` variable.  In many cases,
this will correspond to the target architecture for the build, but
this is not guaranteed.  (E.g. on Windows, the host may be ``AMD64``
even when using a MSVC ``cl`` compiler with a 32-bit target.)

When cross-compiling, a :variable:`CMAKE_TOOLCHAIN_FILE` should set
the ``CMAKE_SYSTEM_PROCESSOR`` variable to match target architecture
that it specifies (via :variable:`CMAKE_<LANG>_COMPILER` and perhaps
:variable:`CMAKE_<LANG>_COMPILER_TARGET`).
