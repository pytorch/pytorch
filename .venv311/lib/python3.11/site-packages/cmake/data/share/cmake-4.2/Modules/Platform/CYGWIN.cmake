set(CMAKE_SHARED_LIBRARY_PREFIX "cyg")
set(CMAKE_SHARED_LIBRARY_SUFFIX ".dll")
set(CMAKE_SHARED_MODULE_PREFIX "cyg")
set(CMAKE_SHARED_MODULE_SUFFIX ".dll")
set(CMAKE_IMPORT_LIBRARY_PREFIX "lib")
set(CMAKE_IMPORT_LIBRARY_SUFFIX ".dll.a")
set(CMAKE_EXECUTABLE_SUFFIX ".exe")          # .exe
# Modules have a different default prefix that shared libs.
set(CMAKE_MODULE_EXISTS 1)

set(CMAKE_FIND_LIBRARY_PREFIXES "lib")
set(CMAKE_FIND_LIBRARY_SUFFIXES ".dll.a" ".a")

# Shared libraries on cygwin can be named with their version number.
set(CMAKE_SHARED_LIBRARY_NAME_WITH_VERSION 1)

include(Platform/UnixPaths)

# Windows API on Cygwin
list(APPEND CMAKE_SYSTEM_INCLUDE_PATH
  /usr/include/w32api
  )

# Windows API on Cygwin
list(APPEND CMAKE_SYSTEM_LIBRARY_PATH
  /usr/lib/w32api
  )
