# This file implements basic support for sdcc (http://sdcc.sourceforge.net/)
# a free C compiler for 8 and 16 bit microcontrollers.
# To use it either a toolchain file is required or cmake has to be run like this:
# cmake -DCMAKE_C_COMPILER=sdcc -DCMAKE_SYSTEM_NAME=Generic <dir...>
# Since sdcc doesn't support C++, C++ support should be disabled in the
# CMakeLists.txt using the project() command:
# project(my_project C)

set(CMAKE_STATIC_LIBRARY_PREFIX "")
set(CMAKE_STATIC_LIBRARY_SUFFIX ".lib")
set(CMAKE_SHARED_LIBRARY_PREFIX "")          # lib
set(CMAKE_SHARED_LIBRARY_SUFFIX ".lib")          # .so
set(CMAKE_IMPORT_LIBRARY_PREFIX )
set(CMAKE_IMPORT_LIBRARY_SUFFIX )
set(CMAKE_EXECUTABLE_SUFFIX ".ihx")          # intel hex file
set(CMAKE_LINK_LIBRARY_SUFFIX ".lib")
set(CMAKE_DL_LIBS "")

set(CMAKE_C_OUTPUT_EXTENSION ".rel")

# find sdar/sdcclib as CMAKE_AR
# since cmake may already have searched for "ar", sdar has to
# be searched with a different variable name (SDCCAR_EXECUTABLE)
# and must then be forced into the cache.
# sdcclib has been deprecated in SDCC 3.2.0 and removed in 3.8.6
# so we first look for sdar
get_filename_component(SDCC_LOCATION "${CMAKE_C_COMPILER}" PATH)
find_program(SDCCAR_EXECUTABLE sdar NAMES sdcclib PATHS "${SDCC_LOCATION}" NO_DEFAULT_PATH)
find_program(SDCCAR_EXECUTABLE sdar NAMES sdcclib)
# for compatibility, in case SDCCLIB_EXECUTABLE is set, we use it
if(DEFINED SDCCLIB_EXECUTABLE)
  set(CMAKE_AR "${SDCCLIB_EXECUTABLE}" CACHE FILEPATH "The sdcc librarian" FORCE)
else()
  set(CMAKE_AR "${SDCCAR_EXECUTABLE}" CACHE FILEPATH "The sdcc librarian" FORCE)
endif()

if("${SDCCAR_EXECUTABLE}" MATCHES "sdcclib")
  set(CMAKE_AR_OPTIONS "-a")
else()
  set(CMAKE_AR_OPTIONS "-rc")
endif()

set(CMAKE_C_LINKER_WRAPPER_FLAG "-Wl" ",")

# compile a C file into an object file
set(CMAKE_C_COMPILE_OBJECT  "<CMAKE_C_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")

# link object files to an executable
set(CMAKE_C_LINK_EXECUTABLE "<CMAKE_C_COMPILER> <FLAGS> <OBJECTS> -o <TARGET> <LINK_FLAGS> <LINK_LIBRARIES>")

# needs sdcc + sdar/sdcclib
set(CMAKE_C_CREATE_STATIC_LIBRARY
      "\"${CMAKE_COMMAND}\" -E remove <TARGET>"
      "<CMAKE_AR> ${CMAKE_AR_OPTIONS} <TARGET> <LINK_FLAGS> <OBJECTS> ")

# not supported by sdcc
set(CMAKE_C_CREATE_SHARED_LIBRARY "")
set(CMAKE_C_CREATE_MODULE_LIBRARY "")
