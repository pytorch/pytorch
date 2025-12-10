# Sun C++ 5.9 does not support -Wl, but Sun C++ 5.11 does not work without it.
# Query the compiler flags to detect whether to use -Wl.
execute_process(COMMAND ${CMAKE_CXX_COMPILER} -flags OUTPUT_VARIABLE _cxx_flags ERROR_VARIABLE _cxx_error)
if("${_cxx_flags}" MATCHES "\n-W[^\n]*component")
  set(CMAKE_SHARED_LIBRARY_RPATH_LINK_CXX_FLAG "-Wl,-rpath-link,")
else()
  set(CMAKE_SHARED_LIBRARY_RPATH_LINK_CXX_FLAG "-rpath-link ")
endif()
set(CMAKE_EXE_EXPORTS_CXX_FLAG "--export-dynamic")
