|CMAKE_IGNORE_VAR| is populated by CMake as part of its platform
and toolchain setup. Its purpose is to ignore locations containing
incompatible binaries meant for the host rather than the target platform.
The project or end user should not modify this variable, they should use
|CMAKE_IGNORE_NONSYSTEM_VAR| instead.
