cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(IntelFortranImplicit Fortran)
add_custom_command(
  OUTPUT output.cmake
  COMMAND ${CMAKE_COMMAND} -P ${IntelFortranImplicit_SOURCE_DIR}/detect.cmake
  )
add_library(FortranLib hello.f output.cmake)
