This folder contains fixes that are introduced in later versions of
cmake but cause generator expression errors in earlier cmake versions.
Specifically:

(1) a problem where a generator expression for include directories was
passed to NVCC, where the generator expression itself was prefixed by -I.
As the NNPACK include directory generator expression expands to multiple directories, the second and later ones were not prefixed by -I, causing
nvcc to return an error. First fixed in CMake 3.7 (see
Kitware/CMake@7ded655f).

(2) Windows VS2017 fixes that allows one to define the ccbin path
differently between earlier versions of Visual Studio and VS2017. First
introduced after 3.10.1 master version (see Kitware/CMake@bc88329e).

If you need to update files under this folder, we recommend you issue PRs
against the CMake mainline branch, and then backport it here for earlier
CMake compatibility.
