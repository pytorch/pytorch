#Catamount, which runs on the compute nodes of Cray machines, e.g. RedStorm, doesn't support shared libs
set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS FALSE)

set(CMAKE_SHARED_LIBRARY_C_FLAGS "")            # -pic
set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "")       # -shared
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")         # +s, flag for exe link to use shared lib
set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG "")       # -rpath
set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG_SEP "")   # : or empty

set(CMAKE_LINK_LIBRARY_SUFFIX "")
set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
set(CMAKE_STATIC_LIBRARY_SUFFIX ".a")
set(CMAKE_SHARED_LIBRARY_PREFIX "lib")          # lib
set(CMAKE_SHARED_LIBRARY_SUFFIX ".a")           # .a
set(CMAKE_EXECUTABLE_SUFFIX "")          # .exe
set(CMAKE_DL_LIBS "" )

set(CMAKE_FIND_LIBRARY_PREFIXES "lib")
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")

include(Platform/UnixPaths)

set(CMAKE_CXX_LINK_SHARED_LIBRARY)
set(CMAKE_CXX_LINK_MODULE_LIBRARY)
set(CMAKE_C_LINK_SHARED_LIBRARY)
set(CMAKE_C_LINK_MODULE_LIBRARY)
