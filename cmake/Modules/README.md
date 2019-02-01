This folder contains various custom cmake modules for finding libraries and packages. Details about some of them are listed below.

### [`FindOpenMP.cmake`](./FindOpenMP.cmake)

This is modified from [the file included in CMake 3.13 release](https://github.com/Kitware/CMake/blob/05a2ca7f87b9ae73f373e9967fde1ee5210e33af/Modules/FindOpenMP.cmake), with the following changes:

+ Update the `separate_arguments` commands to not use `NATIVE_COMMAND` which is not supported in CMake 3.5.

+ Make it respect the `QUIET` flag so that, when it is set, `try_compile` does not use the compiler verbose options, and compilation failures are not reported.
