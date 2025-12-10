include(Compiler/Fujitsu)
__compiler_fujitsu(Fortran)

set(CMAKE_Fortran_SUBMODULE_SEP ".")
set(CMAKE_Fortran_SUBMODULE_EXT ".smod")

set(CMAKE_Fortran_PREPROCESS_SOURCE
  "<CMAKE_Fortran_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -Cpp -P <SOURCE> -o <PREPROCESSED_SOURCE>")
set(CMAKE_Fortran_COMPILE_OPTIONS_PREPROCESS_ON "-Cpp")

set(CMAKE_Fortran_FORMAT_FIXED_FLAG "-Fixed")
set(CMAKE_Fortran_FORMAT_FREE_FLAG "-Free")

string(APPEND CMAKE_Fortran_FLAGS_DEBUG_INIT "")

set(CMAKE_Fortran_MODDIR_FLAG "-M ")
