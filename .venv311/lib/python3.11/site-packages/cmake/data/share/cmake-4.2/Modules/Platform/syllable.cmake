# this is the platform file for the Syllable OS (http://www.syllable.org)
# Syllable is a free OS (GPL), which is mostly POSIX conform
# the linker accepts the rpath related arguments, but this is later on
# ignored by the runtime linker
# shared libs are found exclusively via the environment variable DLL_PATH,
# which may contain also dirs containing the special variable @bindir@
# by default @bindir@/lib is part of DLL_PATH
# in order to run the cmake tests successfully it is required that also
# @bindir@/. and @bindir@/../lib are in DLL_PATH


set(CMAKE_DL_LIBS "dl")
set(CMAKE_C_COMPILE_OPTIONS_PIC "-fPIC")
set(CMAKE_C_COMPILE_OPTIONS_PIE "-fPIE")
set(CMAKE_SHARED_LIBRARY_C_FLAGS "-fPIC")            # -pic
set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "-shared")       # -shared
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")         # +s, flag for exe link to use shared lib
set(CMAKE_SHARED_LIBRARY_SONAME_C_FLAG "-Wl,-soname,")
#set(CMAKE_EXE_EXPORTS_C_FLAG "-Wl,--export-dynamic")

# Initialize C link type selection flags.  These flags are used when
# building a shared library, shared module, or executable that links
# to other libraries to select whether to use the static or shared
# versions of the libraries.
foreach(type SHARED_LIBRARY SHARED_MODULE EXE)
  set(CMAKE_${type}_LINK_STATIC_C_FLAGS "-Wl,-Bstatic")
  set(CMAKE_${type}_LINK_DYNAMIC_C_FLAGS "-Wl,-Bdynamic")
endforeach()

include(Platform/UnixPaths)

# these are Syllable specific:
list(APPEND CMAKE_SYSTEM_PREFIX_PATH /usr/indexes)
