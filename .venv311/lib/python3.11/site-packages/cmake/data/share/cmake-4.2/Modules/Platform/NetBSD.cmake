set(CMAKE_DL_LIBS "")
set(CMAKE_C_COMPILE_OPTIONS_PIC "-fPIC")
set(CMAKE_C_COMPILE_OPTIONS_PIE "-fPIE")
# PIE link options are managed in Compiler/<compiler>.cmake file
set(CMAKE_SHARED_LIBRARY_C_FLAGS "-fPIC")            # -pic
set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "-shared")       # -shared
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")         # +s, flag for exe link to use shared lib
set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG "-Wl,-rpath,")       # -rpath
set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG_SEP ":")   # : or empty
set(CMAKE_SHARED_LIBRARY_RPATH_ORIGIN_TOKEN "\$ORIGIN")
set(CMAKE_SHARED_LIBRARY_RPATH_LINK_C_FLAG "-Wl,-rpath-link,")
set(CMAKE_SHARED_LIBRARY_SONAME_C_FLAG "-Wl,-soname,")
set(CMAKE_EXE_EXPORTS_C_FLAG "-Wl,--export-dynamic")


# Features for LINK_GROUP generator expression
## RESCAN: request the linker to rescan static libraries until there is
## no pending undefined symbols
set(CMAKE_LINK_GROUP_USING_RESCAN "LINKER:--start-group" "LINKER:--end-group")
set(CMAKE_LINK_GROUP_USING_RESCAN_SUPPORTED TRUE)


include(Platform/UnixPaths)
