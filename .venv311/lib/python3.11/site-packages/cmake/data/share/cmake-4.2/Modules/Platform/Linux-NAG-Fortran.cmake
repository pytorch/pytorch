set(CMAKE_Fortran_VERBOSE_FLAG "-Wl,-v") # Runs gcc under the hood.

# Need one "-Wl," level to send flag through to gcc.
# Use "-Xlinker" to get through gcc to real linker.
set(CMAKE_SHARED_LIBRARY_CREATE_Fortran_FLAGS "-Wl,-shared")
set(CMAKE_SHARED_LIBRARY_RUNTIME_Fortran_FLAG "-Wl,-Xlinker,-rpath,-Xlinker,")
set(CMAKE_SHARED_LIBRARY_RUNTIME_Fortran_FLAG_SEP ":")
set(CMAKE_SHARED_LIBRARY_RPATH_LINK_Fortran_FLAG "-Wl,-Xlinker,-rpath-link,-Xlinker,")
set(CMAKE_SHARED_LIBRARY_SONAME_Fortran_FLAG "-Wl,-Xlinker,-soname,-Xlinker,")
set(CMAKE_SHARED_LIBRARY_LINK_Fortran_FLAGS "-Wl,-rdynamic")
