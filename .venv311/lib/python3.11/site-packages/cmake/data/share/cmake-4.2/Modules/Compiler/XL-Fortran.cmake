include(Compiler/XL)
__compiler_xl(Fortran)

set(CMAKE_Fortran_SUBMODULE_SEP "_")
set(CMAKE_Fortran_SUBMODULE_EXT ".smod")

set(CMAKE_Fortran_FORMAT_FIXED_FLAG "-qfixed") # [=<right_margin>]
set(CMAKE_Fortran_FORMAT_FREE_FLAG "-qfree") # [=f90|ibm]

set(CMAKE_Fortran_MODDIR_FLAG "-qmoddir=")
set(CMAKE_Fortran_MODDIR_INCLUDE_FLAG "-I") # -qmoddir= does not affect search path

set(CMAKE_Fortran_DEFINE_FLAG "-WF,-D")

# -qthreaded     = Ensures that all optimizations will be thread-safe
# -qhalt=e       = Halt on error messages (rather than just severe errors)
string(APPEND CMAKE_Fortran_FLAGS_INIT " -qthreaded -qhalt=e")

# xlf: 1501-214 (W) command option E reserved for future use - ignored
set(CMAKE_Fortran_CREATE_PREPROCESSED_SOURCE)
set(CMAKE_Fortran_CREATE_ASSEMBLY_SOURCE)

set(CMAKE_Fortran_PREPROCESS_SOURCE
  "<CMAKE_Fortran_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -qpreprocess -qnoobject -qsuppress=1517-020 -tF -B \"${CMAKE_CURRENT_LIST_DIR}/XL-Fortran/\" -WF,--cpp,\"${CMAKE_Fortran_XL_CPP}\",--out,<PREPROCESSED_SOURCE> <SOURCE>"
  )

if (NOT CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 15.1.6)
  set(CMAKE_Fortran_COMPILE_OPTIONS_PREPROCESS_ON "-qpreprocess")
  set(CMAKE_Fortran_COMPILE_OPTIONS_PREPROCESS_OFF "-qnopreprocess")
endif()
