# This file is processed when the IAR C/C++ Compiler is used
#
# CPU <arch> supported in CMake: 8051, Arm, AVR, MSP430, RH850, RISC-V, RL78, RX, STM8 and V850
#
# The compiler user documentation is architecture-dependent
# and it can found with the product installation under <arch>/doc/{EW,BX}<arch>_DevelopmentGuide.ENU.pdf
#
#
include_guard()

macro(__compiler_iar_common lang)
  if ("x${lang}" MATCHES "^x(C|CXX)$")
    set(CMAKE_${lang}_COMPILE_OBJECT             "<CMAKE_${lang}_COMPILER> ${CMAKE_IAR_${lang}_FLAG} --silent <SOURCE> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT>")
    set(CMAKE_${lang}_CREATE_PREPROCESSED_SOURCE "<CMAKE_${lang}_COMPILER> ${CMAKE_IAR_${lang}_FLAG} --silent <SOURCE> <DEFINES> <INCLUDES> <FLAGS> --preprocess=cnl <PREPROCESSED_SOURCE>")
    set(CMAKE_${lang}_CREATE_ASSEMBLY_SOURCE     "<CMAKE_${lang}_COMPILER> ${CMAKE_IAR_${lang}_FLAG} --silent <SOURCE> <DEFINES> <INCLUDES> <FLAGS> -lAH <ASSEMBLY_SOURCE> -o <OBJECT>.dummy")

    set(CMAKE_DEPFILE_FLAGS_${lang} "--dependencies=ns <DEP_FILE>")

    string(APPEND CMAKE_${lang}_FLAGS_INIT " ")
    string(APPEND CMAKE_${lang}_FLAGS_DEBUG_INIT " -r")
    string(APPEND CMAKE_${lang}_FLAGS_RELEASE_INIT " -Oh -DNDEBUG")
    string(APPEND CMAKE_${lang}_FLAGS_MINSIZEREL_INIT " -Ohz -DNDEBUG")
    string(APPEND CMAKE_${lang}_FLAGS_RELWITHDEBINFO_INIT " -Oh -r -DNDEBUG")

    set(CMAKE_${lang}_LINK_MODE LINKER)
  endif()

  set(CMAKE_${lang}_OUTPUT_EXTENSION_REPLACE 1)
  set(CMAKE_${lang}_RESPONSE_FILE_FLAG "-f ")
  set(CMAKE_${lang}_RESPONSE_FILE_LINK_FLAG "-f ")

  set(CMAKE_${lang}_ARCHIVE_FINISH "")
endmacro()

macro(__compiler_iar_ilink lang)
  set(CMAKE_EXECUTABLE_SUFFIX ".elf")
  set(CMAKE_${lang}_OUTPUT_EXTENSION ".o")

  __compiler_iar_common(${lang})

  string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT " --silent")
  set(CMAKE_${lang}_LINK_EXECUTABLE "<CMAKE_LINKER> <OBJECTS> <LINK_FLAGS> <LINK_LIBRARIES> -o <TARGET>")

  set(CMAKE_${lang}_CREATE_STATIC_LIBRARY "<CMAKE_AR> <TARGET> --create <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_${lang}_ARCHIVE_CREATE "<CMAKE_AR> <TARGET> --create <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_${lang}_ARCHIVE_APPEND "<CMAKE_AR> <TARGET> --replace <LINK_FLAGS> <OBJECTS>")
endmacro()

macro(__compiler_iar_xlink lang)
  set(CMAKE_EXECUTABLE_SUFFIX ".bin")

  __compiler_iar_common(${lang})

  string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT " -S")
  set(CMAKE_${lang}_LINK_EXECUTABLE "<CMAKE_LINKER> <OBJECTS> <LINK_FLAGS> <LINK_LIBRARIES> -o <TARGET>")

  set(CMAKE_${lang}_CREATE_STATIC_LIBRARY "<CMAKE_AR> <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_${lang}_ARCHIVE_CREATE "<CMAKE_AR> <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_${lang}_ARCHIVE_APPEND "")

  set(CMAKE_LIBRARY_PATH_FLAG "-I")
endmacro()

macro(__assembler_iar_deps flag min_version)
  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL ${min_version})
    set(CMAKE_DEPFILE_FLAGS_ASM "${flag} <DEP_FILE>")
  endif()
endmacro()
