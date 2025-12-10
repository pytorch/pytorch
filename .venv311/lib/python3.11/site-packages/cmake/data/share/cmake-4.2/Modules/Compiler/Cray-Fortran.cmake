# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

include(Compiler/Cray)
__compiler_cray(Fortran)

set(CMAKE_Fortran_SUBMODULE_SEP "")
set(CMAKE_Fortran_SUBMODULE_EXT ".mod")
set(CMAKE_Fortran_MODOUT_FLAG -em)
set(CMAKE_Fortran_MODDIR_FLAG -J)
set(CMAKE_Fortran_MODDIR_DEFAULT .)
set(CMAKE_Fortran_FORMAT_FIXED_FLAG "-f fixed")
set(CMAKE_Fortran_FORMAT_FREE_FLAG "-f free")

if (NOT CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 8.5)
  set(CMAKE_Fortran_COMPILE_OPTIONS_PREPROCESS_ON "-eT")
  set(CMAKE_Fortran_COMPILE_OPTIONS_PREPROCESS_OFF "-dT")
else()
  set(CMAKE_Fortran_COMPILE_OPTIONS_PREPROCESS_ON "-eZ")
  set(CMAKE_Fortran_COMPILE_OPTIONS_PREPROCESS_OFF "-dZ")
endif()

if (NOT CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 11.0)
  set(CMAKE_Fortran_PREPROCESS_SOURCE "<CMAKE_Fortran_COMPILER> -o <PREPROCESSED_SOURCE> <DEFINES> <INCLUDES> <FLAGS> -eP <SOURCE>")
endif()
