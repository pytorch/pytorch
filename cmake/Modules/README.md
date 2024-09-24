This folder contains various custom cmake modules for finding libraries and packages. Details about some of them are listed below.

### [`FindOpenMP.cmake`](./FindOpenMP.cmake)

This is modified from [the file included in CMake 3.13 release](https://github.com/Kitware/CMake/blob/05a2ca7f87b9ae73f373e9967fde1ee5210e33af/Modules/FindOpenMP.cmake), with the following changes:

+ Replace `VERSION_GREATER_EQUAL` with `NOT ... VERSION_LESS` as `VERSION_GREATER_EQUAL` is not supported in CMake 3.5 (our min supported version).

+ Update the `separate_arguments` commands to not use `NATIVE_COMMAND` which is not supported in CMake 3.5 (our min supported version).

+ Make it respect the `QUIET` flag so that, when it is set, `try_compile` failures are not reported.

+ For `AppleClang` compilers, use `-Xpreprocessor` instead of `-Xclang` as the later is not documented.

+ For `AppleClang` compilers, an extra flag option is tried, which is `-Xpreprocessor -openmp -I${DIR_OF_omp_h}`, where `${DIR_OF_omp_h}` is a obtained using `find_path` on `omp.h` with `brew`'s default include directory as a hint. Without this, the compiler will complain about missing headers as they are not natively included in Apple's LLVM.

+ For non-GNU compilers, whenever we try a candidate OpenMP flag, first try it with directly linking MKL's `libomp` if it has one. Otherwise, we may end up linking two `libomp`s and end up with this nasty error:

  ```
  OMP: Error #15: Initializing libomp.dylib, but found libiomp5.dylib already
  initialized.

  OMP: Hint This means that multiple copies of the OpenMP runtime have been
  linked into the program. That is dangerous, since it can degrade performance
  or cause incorrect results. The best thing to do is to ensure that only a
  single OpenMP runtime is linked into the process, e.g. by avoiding static
  linking of the OpenMP runtime in any library. As an unsafe, unsupported,
  undocumented workaround you can set the environment variable
  KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but
  that may cause crashes or silently produce incorrect results. For more
  information, please see http://openmp.llvm.org/
  ```

  See NOTE [ Linking both MKL and OpenMP ] for details.
