include(Platform/Windows-GNU)
__windows_compiler_gnu(Fortran)

# gfortran on 64-bit MinGW defines __SIZEOF_POINTER__
set(CMAKE_Fortran_SIZEOF_DATA_PTR_DEFAULT 4)
