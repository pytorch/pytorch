#the compute nodes on BlueGene/L don't support shared libs
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

if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_C_LINK_EXECUTABLE
    "<CMAKE_C_COMPILER> -Wl,-relax <FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> -Wl,-lgcc,-lc -lnss_files -lnss_dns -lresolv")
else()
  # when using IBM xlc we probably don't want to link to -lgcc
  set(CMAKE_C_LINK_EXECUTABLE
    "<CMAKE_C_COMPILER> -Wl,-relax <FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> -Wl,-lc -lnss_files -lnss_dns -lresolv")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_LINK_EXECUTABLE
    "<CMAKE_CXX_COMPILER> -Wl,-relax <FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> -Wl,-lstdc++,-lgcc,-lc -lnss_files -lnss_dns -lresolv")
else()
  # when using the IBM xlC we probably don't want to link to -lgcc
  set(CMAKE_CXX_LINK_EXECUTABLE
    "<CMAKE_CXX_COMPILER> -Wl,-relax <FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> -Wl,-lstdc++,-lc -lnss_files -lnss_dns -lresolv")
endif()
