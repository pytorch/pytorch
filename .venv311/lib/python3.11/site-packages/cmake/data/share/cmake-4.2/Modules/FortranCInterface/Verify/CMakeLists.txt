# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(VerifyFortranC C Fortran)

if(CMAKE_Fortran_COMPILER_ID STREQUAL "LFortran")
  if(CMAKE_Fortran_COMPILER_VERSION VERSION_GREATER_EQUAL "0.55")
    set(_LFORTRAN_SC_FLAG --separate-compilation)
  else()
    set(_LFORTRAN_SC_FLAG --generate-object-code)
  endif()
  add_compile_options("$<$<COMPILE_LANGUAGE:Fortran>:${_LFORTRAN_SC_FLAG}>")
endif()

option(VERIFY_CXX "Whether to verify C++ and Fortran" OFF)
if(VERIFY_CXX)
  enable_language(CXX)
  set(VerifyCXX VerifyCXX.cxx)
  add_definitions(-DVERIFY_CXX)
endif()

include(FortranCInterface)

FortranCInterface_HEADER(VerifyFortran.h SYMBOLS VerifyFortran)
include_directories(${VerifyFortranC_BINARY_DIR})

add_library(VerifyFortran STATIC VerifyFortran.f)
add_executable(VerifyFortranC main.c VerifyC.c ${VerifyCXX})
target_link_libraries(VerifyFortranC VerifyFortran)

if(NOT VERIFY_CXX)
  # The entry point (main) is defined in C; link with the C compiler.
  set_property(TARGET VerifyFortranC PROPERTY LINKER_LANGUAGE C)
endif()
