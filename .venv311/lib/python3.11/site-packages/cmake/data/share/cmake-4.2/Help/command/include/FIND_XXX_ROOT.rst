The CMake variable :variable:`CMAKE_FIND_ROOT_PATH` specifies one or more
directories to be prepended to all other search directories.  This
effectively "re-roots" the entire search under given locations.
Paths which are descendants of the :variable:`CMAKE_STAGING_PREFIX` are excluded
from this re-rooting, because that variable is always a path on the host system.
By default the :variable:`CMAKE_FIND_ROOT_PATH` is empty.

The :variable:`CMAKE_SYSROOT` variable can also be used to specify exactly one
directory to use as a prefix.  Setting :variable:`CMAKE_SYSROOT` also has other
effects.  See the documentation for that variable for more.

These variables are especially useful when cross-compiling to
point to the root directory of the target environment and CMake will
search there too.  By default at first the directories listed in
:variable:`CMAKE_FIND_ROOT_PATH` are searched, then the :variable:`CMAKE_SYSROOT`
directory is searched, and then the non-rooted directories will be
searched.  The default behavior can be adjusted by setting
|CMAKE_FIND_ROOT_PATH_MODE_XXX|.  This behavior can be manually
overridden on a per-call basis using options:

``CMAKE_FIND_ROOT_PATH_BOTH``
  Search in the order described above.

``NO_CMAKE_FIND_ROOT_PATH``
  Do not use the :variable:`CMAKE_FIND_ROOT_PATH` variable.

``ONLY_CMAKE_FIND_ROOT_PATH``
  Search only the re-rooted directories and directories below
  :variable:`CMAKE_STAGING_PREFIX`.
