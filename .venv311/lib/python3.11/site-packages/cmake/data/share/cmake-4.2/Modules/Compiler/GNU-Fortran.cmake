include(Compiler/GNU)
__compiler_gnu(Fortran)

set(CMAKE_Fortran_SUBMODULE_SEP "@")
set(CMAKE_Fortran_SUBMODULE_EXT ".smod")

set(CMAKE_Fortran_PREPROCESS_SOURCE
  "<CMAKE_Fortran_COMPILER> -cpp <DEFINES> <INCLUDES> <FLAGS> -E <SOURCE> -o <PREPROCESSED_SOURCE>")

set(CMAKE_Fortran_FORMAT_FIXED_FLAG "-ffixed-form")
set(CMAKE_Fortran_FORMAT_FREE_FLAG "-ffree-form")

if (NOT CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 4.4)
  set(CMAKE_Fortran_COMPILE_OPTIONS_PREPROCESS_ON "-cpp")
  set(CMAKE_Fortran_COMPILE_OPTIONS_PREPROCESS_OFF "-nocpp")
endif()

set(CMAKE_Fortran_POSTPROCESS_FLAG "-fpreprocessed")

# No -isystem for Fortran because it will not find .mod files.
unset(CMAKE_INCLUDE_SYSTEM_FLAG_Fortran)

# Fortran-specific feature flags.
set(CMAKE_Fortran_MODDIR_FLAG -J)
