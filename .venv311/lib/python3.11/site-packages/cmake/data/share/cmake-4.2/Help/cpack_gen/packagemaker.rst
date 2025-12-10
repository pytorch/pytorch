CPack PackageMaker Generator
----------------------------

Removed.  This once generated PackageMaker installers, but the
generator has been removed since CMake 3.24.  Xcode no longer distributes
the PackageMaker tools.  Use the :cpack_gen:`CPack productbuild Generator`
instead.
