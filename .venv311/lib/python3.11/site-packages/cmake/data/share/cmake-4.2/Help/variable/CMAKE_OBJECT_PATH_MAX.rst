CMAKE_OBJECT_PATH_MAX
---------------------

Maximum object file full-path length allowed by native build tools.

CMake computes for every source file an object file name that is
unique to the source file and deterministic with respect to the full
path to the source file.  This allows multiple source files in a
target to share the same name if they lie in different directories
without rebuilding when one is added or removed.  However, it can
produce long full paths in a few cases, so CMake shortens the path
using a hashing scheme when the full path to an object file exceeds a
limit.  CMake has a built-in limit for each platform that is
sufficient for common tools, but some native tools may have a lower
limit.  This variable may be set to specify the limit explicitly.  The
value must be an integer no less than 128.
