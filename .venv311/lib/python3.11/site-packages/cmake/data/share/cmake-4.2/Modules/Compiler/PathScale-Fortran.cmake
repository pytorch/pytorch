include(Compiler/PathScale)
__compiler_pathscale(Fortran)

set(CMAKE_Fortran_MODDIR_FLAG "-module ")
set(CMAKE_Fortran_FORMAT_FIXED_FLAG "-fixedform")
set(CMAKE_Fortran_FORMAT_FREE_FLAG "-freeform")

set(CMAKE_Fortran_COMPILE_OPTIONS_PREPROCESS_ON "-cpp")
set(CMAKE_Fortran_COMPILE_OPTIONS_PREPROCESS_OFF "-nocpp")
