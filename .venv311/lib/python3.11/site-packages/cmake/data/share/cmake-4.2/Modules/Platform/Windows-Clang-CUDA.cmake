include(Platform/Windows-Clang)
__windows_compiler_clang(CUDA)

# Tell Clang where to find the CUDA libraries.
set(__IMPLICIT_LINKS)
foreach(dir ${CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES})
  string(APPEND __IMPLICIT_LINKS " -L\"${dir}\"")
endforeach()
string(APPEND CMAKE_CUDA_LINK_EXECUTABLE "${__IMPLICIT_LINKS}")
string(APPEND CMAKE_CUDA_CREATE_SHARED_LIBRARY "${__IMPLICIT_LINKS}")
string(APPEND CMAKE_CUDA_CREATE_SHARED_MODULE "${__IMPLICIT_LINKS}")
unset(__IMPLICIT_LINKS)

# Device linking is just regular linking so these are the same.
set(CMAKE_CUDA_DEVICE_LINKER_WRAPPER_FLAG ${CMAKE_CUDA_LINKER_WRAPPER_FLAG})
set(CMAKE_CUDA_DEVICE_LINKER_WRAPPER_FLAG_SEP ${CMAKE_CUDA_LINKER_WRAPPER_FLAG_SEP})
