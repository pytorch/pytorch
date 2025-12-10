set(CMAKE_DL_LIBS "")
set(CMAKE_SHARED_LIBRARY_RPATH_ORIGIN_TOKEN "\$ORIGIN")
set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")

# Shared libraries with no builtin soname may not be linked safely by
# specifying the file path.
set(CMAKE_PLATFORM_USES_PATH_WHEN_NO_SONAME 1)

include(Platform/UnixPaths)
