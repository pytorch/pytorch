# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# determine the compiler to use for ASM programs

include(${CMAKE_ROOT}/Modules/CMakeDetermineCompiler.cmake)

cmake_policy(GET CMP0194 _CMAKE_ASM_CMP0194)

if(NOT CMAKE_ASM${ASM_DIALECT}_COMPILER)
  # prefer the environment variable ASM
  if(NOT $ENV{ASM${ASM_DIALECT}} STREQUAL "")
    get_filename_component(CMAKE_ASM${ASM_DIALECT}_COMPILER_INIT $ENV{ASM${ASM_DIALECT}} PROGRAM PROGRAM_ARGS CMAKE_ASM${ASM_DIALECT}_FLAGS_ENV_INIT)
    if(CMAKE_ASM${ASM_DIALECT}_FLAGS_ENV_INIT)
      set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ARG1 "${CMAKE_ASM${ASM_DIALECT}_FLAGS_ENV_INIT}" CACHE STRING "Arguments to ASM${ASM_DIALECT} compiler")
    endif()
    if(NOT EXISTS ${CMAKE_ASM${ASM_DIALECT}_COMPILER_INIT})
      message(FATAL_ERROR "Could not find compiler set in environment variable ASM${ASM_DIALECT}:\n$ENV{ASM${ASM_DIALECT}}.")
    endif()
  endif()

  # finally list compilers to try
  if("ASM${ASM_DIALECT}" STREQUAL "ASM") # the generic assembler support
    if(NOT CMAKE_ASM_COMPILER_INIT)
      if(_CMAKE_ASM_CMP0194 STREQUAL "NEW")
        set(_CMAKE_ASM_REGEX_MSVC "^(MSVC)$")
        set(_CMAKE_ASM_REGEX_CL "(^|/)[Cc][Ll](\\.|$)")
        set(_CMAKE_ASM_MAYBE_CL "")
      else()
        set(_CMAKE_ASM_REGEX_MSVC "CMP0194_OLD_MSVC_NOT_EXCLUDED")
        set(_CMAKE_ASM_REGEX_CL "CMP0194_OLD_MSVC_NOT_EXCLUDED")
        set(_CMAKE_ASM_MAYBE_CL "cl")
      endif()
      if(CMAKE_C_COMPILER_LOADED AND NOT CMAKE_C_COMPILER_ID MATCHES "${_CMAKE_ASM_REGEX_MSVC}")
        set(CMAKE_ASM_COMPILER_LIST ${CMAKE_C_COMPILER})
      elseif(NOT CMAKE_C_COMPILER_LOADED AND CMAKE_C_COMPILER AND NOT CMAKE_C_COMPILER MATCHES "${_CMAKE_ASM_REGEX_CL}")
        set(CMAKE_ASM_COMPILER_LIST ${CMAKE_C_COMPILER})
      elseif(CMAKE_CXX_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_ID MATCHES "${_CMAKE_ASM_REGEX_MSVC}")
        set(CMAKE_ASM_COMPILER_LIST ${CMAKE_CXX_COMPILER})
      elseif(NOT CMAKE_CXX_COMPILER_LOADED AND CMAKE_CXX_COMPILER AND NOT CMAKE_CXX_COMPILER MATCHES "${_CMAKE_ASM_REGEX_CL}")
        set(CMAKE_ASM_COMPILER_LIST ${CMAKE_CXX_COMPILER})
      else()
        # List all default C and CXX compilers
        set(CMAKE_ASM_COMPILER_LIST
             ${_CMAKE_TOOLCHAIN_PREFIX}cc  ${_CMAKE_TOOLCHAIN_PREFIX}gcc xlc
             ${_CMAKE_ASM_MAYBE_CL}
          CC ${_CMAKE_TOOLCHAIN_PREFIX}c++ ${_CMAKE_TOOLCHAIN_PREFIX}g++ xlC)
        unset(_CMAKE_ASM_MAYBE_CL)
        unset(_CMAKE_ASM_REGEX_CL)
        unset(_CMAKE_ASM_REGEX_MSVC)
      endif()
    endif()
  else() # some specific assembler "dialect"
    if(NOT CMAKE_ASM${ASM_DIALECT}_COMPILER_INIT  AND NOT CMAKE_ASM${ASM_DIALECT}_COMPILER_LIST)
      message(FATAL_ERROR "CMAKE_ASM${ASM_DIALECT}_COMPILER_INIT or CMAKE_ASM${ASM_DIALECT}_COMPILER_LIST must be preset !")
    endif()
  endif()

  # Find the compiler.
  _cmake_find_compiler(ASM${ASM_DIALECT})

else()
  _cmake_find_compiler_path(ASM${ASM_DIALECT})
endif()
mark_as_advanced(CMAKE_ASM${ASM_DIALECT}_COMPILER)

if (NOT _CMAKE_TOOLCHAIN_LOCATION)
  get_filename_component(_CMAKE_TOOLCHAIN_LOCATION "${CMAKE_ASM${ASM_DIALECT}_COMPILER}" PATH)
endif ()


if(NOT CMAKE_ASM${ASM_DIALECT}_COMPILER_ID)

  # Table of per-vendor compiler id flags with expected output.
  list(APPEND CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDORS GNU )
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_FLAGS_GNU "--version")
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_REGEX_GNU "(GNU assembler)|(GCC)|(Free Software Foundation)")

  list(APPEND CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDORS AppleClang )
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_FLAGS_AppleClang "--version")
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_REGEX_AppleClang "(Apple (clang|LLVM) version)")

  list(APPEND CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDORS Clang )
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_FLAGS_Clang "--version")
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_REGEX_Clang "(clang version)")

  list(APPEND CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDORS ARMClang )
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_FLAGS_ARMClang "--version")
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_REGEX_ARMClang "armclang")

  list(APPEND CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDORS OrangeC )
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_FLAGS_OrangeC "--version")
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_REGEX_OrangeC "occ \\(OrangeC\\) Version")

  list(APPEND CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDORS HP )
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_FLAGS_HP "-V")
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_REGEX_HP "HP C")

  list(APPEND CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDORS Intel )
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_FLAGS_Intel "--version")
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_REGEX_Intel "(ICC)")

  list(APPEND CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDORS IntelLLVM )
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_FLAGS_IntelLLVM "--version")
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_REGEX_IntelLLVM "(Intel[^\n]+oneAPI)")

  list(APPEND CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDORS SunPro )
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_FLAGS_SunPro "-V")
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_REGEX_SunPro "Sun C")

  list(APPEND CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDORS XL )
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_FLAGS_XL "-qversion")
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_REGEX_XL "XL C")

  list(APPEND CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDORS MSVC )
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_FLAGS_MSVC "-?")
  if(_CMAKE_ASM_CMP0194 STREQUAL "NEW")
    set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_REGEX_MSVC "Microsoft.*Macro Assembler")
  else()
    set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_REGEX_MSVC "Microsoft")
  endif()

  list(APPEND CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDORS TI )
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_FLAGS_TI "-h")
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_REGEX_TI "Texas Instruments")

  list(APPEND CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDORS TIClang )
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_FLAGS_TIClang "--version")
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_REGEX_TIClang "(TI (.*) Clang Compiler)")

  list(APPEND CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDORS IAR)
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_FLAGS_IAR )
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_REGEX_IAR "IAR Assembler")

  list(APPEND CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDORS Diab)
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_FLAGS_Diab "-V" )
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_REGEX_Diab "Wind River Systems")

  list(APPEND CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDORS ARMCC)
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_FLAGS_ARMCC )
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_REGEX_ARMCC "(ARM Compiler)|(ARM Assembler)|(Arm Compiler)")

  list(APPEND CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDORS NASM)
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_FLAGS_NASM "-v")
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_REGEX_NASM "(NASM version)")

  list(APPEND CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDORS YASM)
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_FLAGS_YASM "--version")
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_REGEX_YASM "(yasm)")

  list(APPEND CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDORS ADSP)
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_FLAGS_ADSP "-version")
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_REGEX_ADSP "Analog Devices")

  list(APPEND CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDORS QCC)
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_FLAGS_QCC "-V")
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_REGEX_QCC "gcc_nto")

  list(APPEND CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDORS Tasking)
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_FLAGS_Tasking "--version")
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_REGEX_Tasking "TASKING")

  list(APPEND CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDORS Renesas)
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_FLAGS_Renesas "-v")
  set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_REGEX_Renesas "(RX Family C/C\\+\\+ Compiler)|(RL78 Family Compiler)|(RH850 Family Compiler)")

  include(CMakeDetermineCompilerId)
  set(userflags)
  CMAKE_DETERMINE_COMPILER_ID_VENDOR(ASM${ASM_DIALECT} "${userflags}")
  set(_variant "")
  if("x${CMAKE_ASM${ASM_DIALECT}_COMPILER_ID}" STREQUAL "xIAR")
    # primary necessary to detect architecture, so the right archiver and linker can be picked
    # eg. "IAR Assembler V8.10.1.12857/W32 for ARM" or "IAR Assembler V4.11.1.4666 for Renesas RX"
    # Earlier versions did not provide `--version`, so grep the full output to extract Assembler ID string
    string(REGEX MATCH "IAR Assembler[^\r\n]*" _compileid "${CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_OUTPUT}")
    if("${_compileid}" MATCHES "V([0-9]+\\.[0-9]+\\.[0-9]+)")
      set(CMAKE_ASM${ASM_DIALECT}_COMPILER_VERSION ${CMAKE_MATCH_1})
    endif()
    if("${_compileid}" MATCHES "for.*(MSP430|8051|ARM|AVR|RH850|RISC-?V|RL78|RX|STM8|V850)")
      set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ARCHITECTURE_ID ${CMAKE_MATCH_1})
    endif()
  elseif("x${CMAKE_ASM${ASM_DIALECT}_COMPILER_ID}" STREQUAL "xClang")
    # Test whether an MSVC-like command-line option works.
    execute_process(COMMAND ${CMAKE_ASM${ASM_DIALECT}_COMPILER} -?
      OUTPUT_VARIABLE _clang_output
      ERROR_VARIABLE _clang_output
      RESULT_VARIABLE _clang_result)
      if(_clang_result EQUAL 0)
        set(CMAKE_ASM${ASM_DIALECT}_COMPILER_FRONTEND_VARIANT "MSVC")
        set(CMAKE_ASM${ASM_DIALECT}_SIMULATE_ID MSVC)
      else()
        set(CMAKE_ASM${ASM_DIALECT}_COMPILER_FRONTEND_VARIANT "GNU")
      endif()
      set(_variant " with ${CMAKE_ASM${ASM_DIALECT}_COMPILER_FRONTEND_VARIANT}-like command-line")
  elseif("x${CMAKE_ASM${ASM_DIALECT}_COMPILER_ID}" STREQUAL "xRenesas")
    string(REGEX MATCH "[A-Za-z0-9]+ Family (C/C\\+\\+ )*Compiler [V|E][0-9|.]+" _compiler_id "${CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_OUTPUT}")
    string(REGEX MATCH "R[A-Za-z0-9]+" CMAKE_ASM${ASM_DIALECT}_COMPILER_ARCHITECTURE_ID "${_compiler_id}")
    string(REGEX MATCH "V[0-9|.]+" CMAKE_ASM${ASM_DIALECT}_COMPILER_VERSION "${_compiler_id}")
  endif()

  _cmake_find_compiler_sysroot(ASM${ASM_DIALECT})

  unset(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_OUTPUT)
  unset(_all_compileid_matches)
  unset(_compileid)
  unset(_clang_result)
  unset(_clang_output)
endif()

if(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID)
  if(CMAKE_ASM${ASM_DIALECT}_COMPILER_VERSION)
    set(_version " ${CMAKE_ASM${ASM_DIALECT}_COMPILER_VERSION}")
  else()
    set(_version "")
  endif()
  if(CMAKE_ASM${ASM_DIALECT}_COMPILER_ARCHITECTURE_ID AND "x${CMAKE_ASM${ASM_DIALECT}_COMPILER_ID}" STREQUAL "xIAR")
    set(_archid " ${CMAKE_ASM${ASM_DIALECT}_COMPILER_ARCHITECTURE_ID}")
  elseif(CMAKE_ASM${ASM_DIALECT}_COMPILER_ARCHITECTURE_ID AND "x${CMAKE_ASM${ASM_DIALECT}_COMPILER_ID}" STREQUAL "xRenesas")
    set(_archid " ${CMAKE_ASM${ASM_DIALECT}_COMPILER_ARCHITECTURE_ID} Family Assembler")
  else()
    set(_archid "")
  endif()
  message(STATUS "The ASM${ASM_DIALECT} compiler identification is ${CMAKE_ASM${ASM_DIALECT}_COMPILER_ID}${_archid}${_version}${_variant}")
  unset(_archid)
  unset(_version)
  unset(_variant)
else()
  message(STATUS "The ASM${ASM_DIALECT} compiler identification is unknown")
endif()

if("ASM${ASM_DIALECT}" STREQUAL "ASM" AND CMAKE_ASM_COMPILER_ID STREQUAL "MSVC" AND _CMAKE_ASM_CMP0194 STREQUAL "")
  cmake_policy(GET_WARNING CMP0194 _CMAKE_ASM_CMP0194_WARNING)
  message(AUTHOR_WARNING "${_CMAKE_ASM_CMP0194_WARNING}")
endif()

# If we have a gas/as cross compiler, they have usually some prefix, like
# e.g. powerpc-linux-gas, arm-elf-gas or i586-mingw32msvc-gas , optionally
# with a 3-component version number at the end
# The other tools of the toolchain usually have the same prefix
# NAME_WE cannot be used since then this test will fail for names like
# "arm-unknown-nto-qnx6.3.0-gas.exe", where BASENAME would be
# "arm-unknown-nto-qnx6" instead of the correct "arm-unknown-nto-qnx6.3.0-"
if (NOT _CMAKE_TOOLCHAIN_PREFIX)
  get_filename_component(COMPILER_BASENAME "${CMAKE_ASM${ASM_DIALECT}_COMPILER}" NAME)
  if (COMPILER_BASENAME MATCHES "^(.+-)g?as(-[0-9]+\\.[0-9]+\\.[0-9]+)?(\\.exe)?$")
    set(_CMAKE_TOOLCHAIN_PREFIX ${CMAKE_MATCH_1})
  endif ()
endif ()

# Now try the C compiler regexp:
if (NOT _CMAKE_TOOLCHAIN_PREFIX)
  if (COMPILER_BASENAME MATCHES "^(.+-)g?cc(-[0-9]+\\.[0-9]+\\.[0-9]+)?(\\.exe)?$")
    set(_CMAKE_TOOLCHAIN_PREFIX ${CMAKE_MATCH_1})
  endif ()
endif ()

# Finally try the CXX compiler regexp:
if (NOT _CMAKE_TOOLCHAIN_PREFIX)
  if (COMPILER_BASENAME MATCHES "^(.+-)[gc]\\+\\+(-[0-9]+\\.[0-9]+\\.[0-9]+)?(\\.exe)?$")
    set(_CMAKE_TOOLCHAIN_PREFIX ${CMAKE_MATCH_1})
  endif ()
endif ()


set(_CMAKE_PROCESSING_LANGUAGE "ASM")
include(CMakeFindBinUtils)
include(Compiler/${CMAKE_ASM${ASM_DIALECT}_COMPILER_ID}-FindBinUtils OPTIONAL)
unset(_CMAKE_PROCESSING_LANGUAGE)

set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ENV_VAR "ASM${ASM_DIALECT}")

if(CMAKE_ASM${ASM_DIALECT}_COMPILER)
  message(STATUS "Found assembler: ${CMAKE_ASM${ASM_DIALECT}_COMPILER}")
else()
  message(STATUS "Didn't find assembler")
endif()

if(CMAKE_ASM${ASM_DIALECT}_COMPILER_SYSROOT)
  string(CONCAT _SET_CMAKE_ASM_COMPILER_SYSROOT
    "set(CMAKE_ASM${ASM_DIALECT}_COMPILER_SYSROOT \"${CMAKE_ASM${ASM_DIALECT}_COMPILER_SYSROOT}\")\n"
    "set(CMAKE_COMPILER_SYSROOT \"${CMAKE_ASM${ASM_DIALECT}_COMPILER_SYSROOT}\")")
else()
  set(_SET_CMAKE_ASM_COMPILER_SYSROOT "")
endif()

if(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_MATCH)
  set(_SET_CMAKE_ASM_COMPILER_ID_VENDOR_MATCH
    "set(CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_MATCH [==[${CMAKE_ASM${ASM_DIALECT}_COMPILER_ID_VENDOR_MATCH}]==])")
else()
  set(_SET_CMAKE_ASM_COMPILER_ID_VENDOR_MATCH "")
endif()

# configure variables set in this file for fast reload later on
block()
  foreach(_var IN ITEMS
      # Keep in sync with Internal/CMakeTestASMLinker.
      COMPILER
      COMPILER_ID
      COMPILER_ARG1
      COMPILER_ENV_VAR
      COMPILER_AR
      COMPILER_RANLIB
      COMPILER_VERSION
      COMPILER_ARCHITECTURE_ID
      )
    set(_CMAKE_ASM_${_var} "${CMAKE_ASM${ASM_DIALECT}_${_var}}")
  endforeach()
  configure_file(${CMAKE_ROOT}/Modules/CMakeASMCompiler.cmake.in
    ${CMAKE_PLATFORM_INFO_DIR}/CMakeASM${ASM_DIALECT}Compiler.cmake @ONLY)
endblock()
