This `./upstream` subfolder contains fixes for `FindCUDA` that are introduced in
later versions of cmake but cause generator expression errors in earlier CMake
versions. Specifically:

1. a problem where a generator expression for include directories was
passed to NVCC, where the generator expression itself was prefixed by `-I`.
As the NNPACK include directory generator expression expands to multiple
directories, the second and later ones were not prefixed by `-I`, causing
NVCC to return an error. First fixed in CMake 3.7 (see
[Kitware/CMake@7ded655f](https://github.com/Kitware/CMake/commit/7ded655f)).

2. Windows VS2017 fixes that allows one to define the ccbin path
differently between earlier versions of Visual Studio and VS2017. First
introduced after 3.10.1 master version (see
[Kitware/CMake@bc88329e](https://github.com/Kitware/CMake/commit/bc88329e)).

The downside of using these fixes is that `./upstream/CMakeInitializeConfigs.cmake`,
defining some new CMake variables (added in
[Kitware/CMake@48f7e2d3](https://github.com/Kitware/CMake/commit/48f7e2d3)),
must be included before `./upstream/FindCUDA.cmake` to support older CMake
versions. A wrapper `./FindCUDA.cmake` is created to do this automatically, and
to allow submodules to use these fixes because we can't patch their
`CMakeList.txt`.

If you need to update files under `./upstream` folder, we recommend you issue PRs
against [the CMake mainline branch](https://gitlab.kitware.com/cmake/cmake/tree/master/Modules/FindCUDA.cmake),
and then backport it here for earlier CMake compatibility.
