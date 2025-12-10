include(Compiler/PGI-Fortran)
include(Compiler/NVHPC)
__compiler_nvhpc(Fortran)
if(CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 21.7)
  # Before NVHPC 21.7 nvfortran didn't support isystem
  unset(CMAKE_INCLUDE_SYSTEM_FLAG_Fortran)
endif()
