set(CMAKE_DL_LIBS "")

if(CMAKE_SYSTEM MATCHES "OSF1-1.[012]")
endif()
if(CMAKE_SYSTEM MATCHES "OSF1-1")
  # OSF/1 1.3 from OSF using ELF, and derivatives, including AD2
  set(CMAKE_C_COMPILE_OPTIONS_PIC "-fpic")
  set(CMAKE_C_COMPILE_OPTIONS_PIE "-fpie")
  set(CMAKE_SHARED_LIBRARY_C_FLAGS "-fpic")     # -pic
  set(CMAKE_SHARED_LIBRARY_CXX_FLAGS "-fpic")   # -pic
endif()



if(CMAKE_SYSTEM MATCHES "OSF1-V")
  set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "-shared -Wl,-expect_unresolved,\\*")       # -shared
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG "-Wl,-rpath,")
  else()
    set(CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG "-rpath ")
  endif()
  if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG "-Wl,-rpath,")
  else()
    set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG "-rpath ")
  endif()
  set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG_SEP ":")
endif()

set(CMAKE_MAKE_INCLUDE_FROM_ROOT 1) # include $(CMAKE_BINARY_DIR)/...

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # include the gcc flags
else ()
  # use default OSF compiler flags
  set (CMAKE_C_FLAGS_INIT "")
  set (CMAKE_C_FLAGS_DEBUG_INIT "-g")
  set (CMAKE_C_FLAGS_MINSIZEREL_INIT "-O2 -DNDEBUG")
  set (CMAKE_C_FLAGS_RELEASE_INIT "-O2 -DNDEBUG")
  set (CMAKE_C_FLAGS_RELWITHDEBINFO_INIT "-O2")
  set (CMAKE_CXX_FLAGS_INIT "")
  set (CMAKE_CXX_FLAGS_DEBUG_INIT "-g")
  set (CMAKE_CXX_FLAGS_MINSIZEREL_INIT "-O2 -DNDEBUG")
  set (CMAKE_CXX_FLAGS_RELEASE_INIT "-O2 -DNDEBUG")
  set (CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "-O2")
endif()
include(Platform/UnixPaths)
