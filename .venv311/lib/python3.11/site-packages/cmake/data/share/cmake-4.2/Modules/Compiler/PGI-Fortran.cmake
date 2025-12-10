include(Compiler/PGI)
__compiler_pgi(Fortran)

set(CMAKE_Fortran_SUBMODULE_SEP "-")
set(CMAKE_Fortran_SUBMODULE_EXT ".mod")

set(CMAKE_Fortran_PREPROCESS_SOURCE
  "<CMAKE_Fortran_COMPILER> -Mpreprocess <DEFINES> <INCLUDES> <FLAGS> -E <SOURCE> > <PREPROCESSED_SOURCE>")
set(CMAKE_Fortran_COMPILE_OPTIONS_PREPROCESS_ON "-Mpreprocess")

set(CMAKE_Fortran_FORMAT_FIXED_FLAG "-Mnofreeform")
set(CMAKE_Fortran_FORMAT_FREE_FLAG "-Mfreeform")

string(APPEND CMAKE_Fortran_FLAGS_DEBUG_INIT " -Mbounds")

set(CMAKE_Fortran_MODDIR_FLAG "-module ")
