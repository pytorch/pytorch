# Please ensure your changes or patch meets minimum requirements.
#   The minimum requirements are 2.8.6. It roughly equates to
#   Ubuntu 14.05 LTS or Solaris 11.3. Please do not check in something
#   for 3.5.0 or higher because it will break LTS operating systems
#   and a number of developer boards used for testing. To test your
#   changes, please set up a Ubuntu 14.05 LTS system.

# Should we be setting things like this? We are not a C project
# so nothing should be done with the C compiler. But there is
# no reliable way to tell CMake we are C++.
# Cannot set this... Breaks Linux PowerPC with Clang:
# SET(CMAKE_C_COMPILER ${CMAKE_CXX_COMPILER})
# # error "The CMAKE_C_COMPILER is set to a C++ compiler"

if(NOT DEFINED cryptocpp_DISPLAY_CMAKE_SUPPORT_WARNING)
  set(cryptocpp_DISPLAY_CMAKE_SUPPORT_WARNING 1)
endif()
if(cryptocpp_DISPLAY_CMAKE_SUPPORT_WARNING)
  message( STATUS
"*************************************************************************\n"
"The Crypto++ library does not officially support CMake. CMake support is a\n"
"community effort, and the library works with the folks using CMake to help\n"
"improve it. If you find an issue then please fix it or report it at\n"
"https://github.com/noloader/cryptopp-cmake.\n"
"-- *************************************************************************"
)
endif()

# Print useful information
message( STATUS "CMake version ${CMAKE_VERSION}" )

cmake_minimum_required(VERSION 2.8.6)
if (${CMAKE_VERSION} VERSION_LESS "3.0.0")
  project(cryptopp)
  set(cryptopp_VERSION_MAJOR 8)
  set(cryptopp_VERSION_MINOR 3)
  set(cryptopp_VERSION_PATCH 0)
else ()
  cmake_policy(SET CMP0048 NEW)
  project(cryptopp VERSION 8.3.0)
  if (NOT ${CMAKE_VERSION} VERSION_LESS "3.1.0")
    cmake_policy(SET CMP0054 NEW)
  endif ()
endif ()

if(NOT CRYPTOPP_ROOT_DIR)
  set(CRYPTOPP_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# Need to set SRC_DIR manually after removing the Python library code.
set(SRC_DIR ${CRYPTOPP_ROOT_DIR})

# Make RelWithDebInfo the default (it does e.g. add '-O2 -g -DNDEBUG' for GNU)
#   If not in multi-configuration environments, no explicit build type or CXX
#   flags are set by the user and if we are the root CMakeLists.txt file.
if (NOT CMAKE_CONFIGURATION_TYPES AND
    NOT CMAKE_NO_BUILD_TYPE AND
    NOT CMAKE_BUILD_TYPE AND
    NOT CMAKE_CXX_FLAGS AND
    CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif ()

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

# We now carry around test programs. test_cxx.cxx is the default C++ one.
# Also see https://github.com/weidai11/cryptopp/issues/741.
set(TEST_PROG_DIR ${SRC_DIR}/TestPrograms)
set(TEST_CXX_FILE ${TEST_PROG_DIR}/test_cxx.cxx)

#============================================================================
# Settable options
#============================================================================

option(BUILD_STATIC "Build static library" ON)
option(BUILD_SHARED "Build shared library" ON)
option(BUILD_TESTING "Build library tests" ON)
option(BUILD_DOCUMENTATION "Use Doxygen to create the HTML based API documentation" OFF)
option(USE_INTERMEDIATE_OBJECTS_TARGET "Use a common intermediate objects target for the static and shared library targets" ON)

# These are IA-32 options. TODO: Add ARM A-32, Aarch64 and Power8 options.
option(DISABLE_ASM "Disable ASM" OFF)
option(DISABLE_SSSE3 "Disable SSSE3" OFF)
option(DISABLE_SSE4 "Disable SSE4" OFF)
option(DISABLE_AESNI "Disable AES-NI" OFF)
option(DISABLE_SHA "Disable SHA" OFF)
option(DISABLE_AVX "Disable AVX" OFF)
option(DISABLE_AVX2 "Disable AVX2" OFF)
option(CRYPTOPP_NATIVE_ARCH "Enable native architecture" OFF)
set(CRYPTOPP_DATA_DIR "" CACHE PATH "Crypto++ test data directory")

#============================================================================
# Compiler options
#============================================================================

set(CRYPTOPP_COMPILE_DEFINITIONS)
set(CRYPTOPP_COMPILE_OPTIONS)

# Stop hiding the damn output...
# set(CMAKE_VERBOSE_MAKEFILE on)

# Always 1 ahead in Master. Also see http://groups.google.com/forum/#!topic/cryptopp-users/SFhqLDTQPG4
set(LIB_VER ${cryptopp_VERSION_MAJOR}${cryptopp_VERSION_MINOR}${cryptopp_VERSION_PATCH})

# Don't use RPATH's. The resulting binary could fail a security audit.
set(CMAKE_MACOSX_RPATH 0)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  list(APPEND CRYPTOPP_COMPILE_OPTIONS -wd68 -wd186 -wd279 -wd327 -wd161 -wd3180)
endif ()

# Also see http://github.com/weidai11/cryptopp/issues/395
if (DISABLE_ASM)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_ASM)
endif ()
if (DISABLE_SSSE3)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_SSSE3)
endif ()
if (DISABLE_SSE4)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_SSSE4)
endif ()
if (DISABLE_AESNI)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_AESNI)
endif ()
if (DISABLE_SHA)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_SHA)
endif ()
if (DISABLE_ALTIVEC)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_ALTIVEC)
endif ()
if (DISABLE_POWER7)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_POWER7)
endif ()
if (DISABLE_POWER8)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_POWER8)
endif ()
if (DISABLE_POWER9)
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS CRYPTOPP_DISABLE_POWER9)
endif ()
if (NOT CRYPTOPP_DATA_DIR STREQUAL "")
  list(APPEND CRYPTOPP_COMPILE_DEFINITIONS "CRYPTOPP_DATA_DIR=${CRYPTOPP_DATA_DIR}")
endif ()

###############################################################################

# Try to find a Posix compatible grep and sed. Solaris, Digital Unix,
#   Tru64, HP-UX and a few others need tweaking

if (EXISTS /usr/xpg4/bin/grep)
  set(GREP_CMD /usr/xpg4/bin/grep)
elseif (EXISTS /usr/gnu/bin/grep)
  set(GREP_CMD /usr/gnu/bin/grep)
elseif (EXISTS /usr/linux/bin/grep)
  set(GREP_CMD /usr/linux/bin/grep)
else ()
  set(GREP_CMD grep)
endif ()

if (EXISTS /usr/xpg4/bin/sed)
  set(SED_CMD /usr/xpg4/bin/sed)
elseif (EXISTS /usr/gnu/bin/sed)
  set(SED_CMD /usr/gnu/bin/sed)
elseif (EXISTS /usr/linux/bin/sed)
  set(SED_CMD /usr/linux/bin/sed)
else ()
  set(SED_CMD sed)
endif ()

###############################################################################

function(CheckCompileOption opt var)

  if (MSVC)

    # TODO: improve this...
    CHECK_CXX_COMPILER_FLAG(${opt} ${var})

  elseif (CMAKE_CXX_COMPILER_ID MATCHES "SunPro")

    message(STATUS "Performing Test ${var}")
    execute_process(
      COMMAND sh -c "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${opt} -c ${TEST_CXX_FILE} 2>&1"
      COMMAND ${GREP_CMD} -i -c -E "illegal value ignored"
      RESULT_VARIABLE COMMAND_RESULT
      OUTPUT_VARIABLE COMMAND_OUTPUT
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    # No dereference below. Thanks for the warning, CMake (not!).
    if (COMMAND_RESULT AND NOT COMMAND_OUTPUT)
      set(${var} 1 PARENT_SCOPE)
      message(STATUS "Performing Test ${var} - Success")
    else ()
      set(${var} 0 PARENT_SCOPE)
      message(STATUS "Performing Test ${var} - Failed")
    endif ()

  # Must use CMAKE_CXX_COMPILER here due to XLC 13.1 and LLVM front-end.
  elseif (CMAKE_CXX_COMPILER MATCHES "xlC")

    message(STATUS "Performing Test ${var}")
    execute_process(
      COMMAND sh -c "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${opt} -c ${TEST_CXX_FILE} 2>&1"
      COMMAND ${GREP_CMD} -i -c -E "Unrecognized value"
      RESULT_VARIABLE COMMAND_RESULT
      OUTPUT_VARIABLE COMMAND_OUTPUT
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    # No dereference below. Thanks for the warning, CMake (not!).
    if (COMMAND_RESULT AND NOT COMMAND_OUTPUT)
      set(${var} 1 PARENT_SCOPE)
      message(STATUS "Performing Test ${var} - Success")
    else ()
      set(${var} 0 PARENT_SCOPE)
      message(STATUS "Performing Test ${var} - Failed")
    endif ()

  else ()

    CHECK_CXX_COMPILER_FLAG(${opt} ${var})

  endif ()

endfunction(CheckCompileOption)

function(CheckCompileLinkOption opt var prog)

  if (MSVC)

    # TODO: improve this...
    CHECK_CXX_COMPILER_FLAG(${opt} ${var})
    
  elseif (APPLE)

    message(STATUS "Performing Test ${var}")
    try_compile(COMMAND_SUCCESS ${CMAKE_BINARY_DIR} ${prog} COMPILE_DEFINITIONS ${opt})
    if (COMMAND_SUCCESS)
      set(${var} 1 PARENT_SCOPE)
      message(STATUS "Performing Test ${var} - Success")
    else ()
      set(${var} 0 PARENT_SCOPE)
      message(STATUS "Performing Test ${var} - Failed")
    endif ()

  else ()

    message(STATUS "Performing Test ${var}")
    execute_process(
      COMMAND sh -c "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${opt} ${prog} 2>&1"
      RESULT_VARIABLE COMMAND_RESULT
      OUTPUT_VARIABLE COMMAND_OUTPUT
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    # message(STATUS "RESULT_VARIABLE ${RESULT_VARIABLE}")
    # message(STATUS "COMMAND_RESULT ${COMMAND_RESULT}")
    # message(STATUS "OUTPUT_VARIABLE ${OUTPUT_VARIABLE}")
    # message(STATUS "COMMAND_OUTPUT ${COMMAND_OUTPUT}")

    # This test is strict. We require two things. First, the invocation
    # of the compile command must return 0. Second, there must be no
    # messages on the console. We are interested in diagnostics like
    # warnings to decide when to reject an option. But we will probably
    # capture chatty compiler that want to say, "Hooray, success". For
    # chatty compilers we will need to find a quiet option and use it
    # for the test. Microsoft compilers come to mind.
    if ("${COMMAND_RESULT}" EQUAL 0 AND "${COMMAND_OUTPUT}" STREQUAL "")
      set(${var} 1 PARENT_SCOPE)
      message(STATUS "Performing Test ${var} - Success")
    else ()
      set(${var} 0 PARENT_SCOPE)
      message(STATUS "Performing Test ${var} - Failed")
    endif ()

  endif ()

endfunction(CheckCompileLinkOption)

function(AddCompileOption opt)

    if ("${COMMAND_OUTPUT}" NOT STREQUAL "")
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "${opt}")
    endif ()

endfunction(AddCompileOption)

###############################################################################

function(DumpMachine output pattern)

  if (MSVC)

    # CMake does not provide a generic shell/terminal mechanism
    #  and Microsoft environments don't know what 'sh' is.
    set(${output} 0 PARENT_SCOPE)

  else ()

    execute_process(
      COMMAND sh -c "${CMAKE_CXX_COMPILER} -dumpmachine 2>&1"
      COMMAND ${GREP_CMD} -i -c -E "${pattern}"
      OUTPUT_VARIABLE ${output}
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(${output} "${${output}}" PARENT_SCOPE)

  endif()

endfunction(DumpMachine)

# Thansk to Anonimal for MinGW; see http://github.com/weidai11/cryptopp/issues/466
DumpMachine(CRYPTOPP_AMD64 "amd64|x86_64")
DumpMachine(CRYPTOPP_I386 "i.86")
DumpMachine(CRYPTOPP_MINGW32 "\\<mingw32\\>")
DumpMachine(CRYPTOPP_MINGW64 "w64-mingw32|mingw64")
DumpMachine(CRYPTOPP_X32 "x32")
DumpMachine(CRYPTOPP_AARCH32 "Aarch32")
DumpMachine(CRYPTOPP_AARCH64 "Aarch64")
DumpMachine(CRYPTOPP_ARMHF "armhf|arm7l|eabihf")
DumpMachine(CRYPTOPP_ARM "\\<arm\\>")

# Detecting PowerPC is only good with GCC. IBM XLC compiler is
# a little different and I don't know how to ask to the triplet
# XLC is targeting. Below we punt by setting CRYPTOPP_POWERPC64
# if we detect the compiler is XLC.
DumpMachine(CRYPTOPP_POWERPC "ppc|powerpc")
DumpMachine(CRYPTOPP_POWERPC64 "ppc64")

###############################################################################

# Test SunCC for a string like 'CC: Sun C++ 5.13 SunOS_i386'
if (NOT CRYPTOPP_SOLARIS)
  execute_process(COMMAND sh -c "${CMAKE_CXX_COMPILER} -V 2>&1"
    COMMAND ${GREP_CMD} -i -c "SunOS"
    OUTPUT_VARIABLE CRYPTOPP_SOLARIS
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif ()

# Test GCC for a string like 'i386-pc-solaris2.11'
if (NOT CRYPTOPP_SOLARIS)
  execute_process(COMMAND sh -c "${CMAKE_CXX_COMPILER} -dumpmachine 2>&1"
    COMMAND ${GREP_CMD} -i -c "Solaris"
    OUTPUT_VARIABLE CRYPTOPP_SOLARIS
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif ()

# Fixup PowerPC. If both 32-bit and 64-bit use 64-bit.
if (CRYPTOPP_POWERPC AND CRYPTOPP_POWERPC64)
  unset(CRYPTOPP_POWERPC)
endif ()

# Fixup for xlC compiler. -dumpmachine fails so we miss PowerPC
# TODO: something better than proxying the platform via compiler
# Must use CMAKE_CXX_COMPILER here due to XLC 13.1 and LLVM front-end.
if (CMAKE_CXX_COMPILER MATCHES "xlC")
  message ("-- Fixing platform due to IBM xlC")
  set(CRYPTOPP_POWERPC64 1)
endif ()

# DumpMachine SunCC style
if (CMAKE_CXX_COMPILER_ID STREQUAL "SunPro")

  # SunCC is 32-bit, but it builds both 32 and 64 bit. Use
  execute_process(COMMAND sh -c "${CMAKE_CXX_COMPILER} -V 2>&1"
    COMMAND ${GREP_CMD} -i -c "Sparc"
    OUTPUT_VARIABLE CRYPTOPP_SPARC
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(COMMAND sh -c "${CMAKE_CXX_COMPILER} -V 2>&1"
    COMMAND ${GREP_CMD} -i -c -E "i386|i86"
    OUTPUT_VARIABLE CRYPTOPP_I386
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(COMMAND isainfo -k
    COMMAND ${GREP_CMD} -i -c "i386"
    OUTPUT_VARIABLE KERNEL_I386
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(COMMAND isainfo -k
    COMMAND ${GREP_CMD} -i -c "amd64"
    OUTPUT_VARIABLE KERNEL_AMD64
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(COMMAND isainfo -k
    COMMAND ${GREP_CMD} -i -c "Sparc"
    OUTPUT_VARIABLE KERNEL_SPARC
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(COMMAND isainfo -k
    COMMAND ${GREP_CMD} -i -c -E "UltraSarc|Sparc64|SparcV9"
    OUTPUT_VARIABLE KERNEL_SPARC64
    OUTPUT_STRIP_TRAILING_WHITESPACE)

endif ()

###############################################################################

# TODO: what about ICC and LLVM on Windows?
if (MSVC)
  if (CMAKE_SYSTEM_VERSION MATCHES "10\\.0.*")
    list(APPEND CRYPTOPP_COMPILE_DEFINITIONS "_WIN32_WINNT=0x0A00")
  endif ()
  list(APPEND CRYPTOPP_COMPILE_OPTIONS "/FI\"winapifamily.h\"")
endif ()

# Enable PIC for all target machines except 32-bit i386 due to register pressures.
if (NOT CRYPTOPP_I386)
  SET(CMAKE_POSITION_INDEPENDENT_CODE 1)
endif ()

# IBM XLC compiler options for AIX and Linux.
# Must use CMAKE_CXX_COMPILER here due to XLC 13.1 and LLVM front-end.
if (CMAKE_CXX_COMPILER MATCHES "xlC")

  #CheckCompileLinkOption("-qxlcompatmacros" CRYPTOPP_XLC_COMPAT "${TEST_CXX_FILE}")
  #if (CRYPTOPP_XLC_COMPAT)
  #  list(APPEND CRYPTOPP_COMPILE_OPTIONS "-qxlcompatmacros")
  #endif ()

  CheckCompileLinkOption("-qrtti" CRYPTOPP_PPC_RTTI "${TEST_CXX_FILE}")
  if (CRYPTOPP_PPC_RTTI)
    list(APPEND CRYPTOPP_COMPILE_OPTIONS "-qrtti")
  endif ()

  CheckCompileLinkOption("-qmaxmem=-1" CRYPTOPP_PPC_MAXMEM "${TEST_CXX_FILE}")
  if (CRYPTOPP_PPC_MAXMEM)
    list(APPEND CRYPTOPP_COMPILE_OPTIONS "-qmaxmem=-1")
  endif ()

  CheckCompileLinkOption("-qthreaded" CRYPTOPP_PPC_THREADED "${TEST_CXX_FILE}")
  if (CRYPTOPP_PPC_THREADED)
    list(APPEND CRYPTOPP_COMPILE_OPTIONS "-qthreaded")
  endif ()
endif ()

# Solaris specific
if (CRYPTOPP_SOLARIS)

  # SunCC needs -template=no%extdef
  if (CMAKE_CXX_COMPILER_ID STREQUAL "SunPro")
    list(APPEND CRYPTOPP_COMPILE_OPTIONS "-template=no%extdef")
  endif ()

  # SunCC needs -xregs=no%appl on Sparc (not x86) for libraries (not test program)
  # TODO: wire this up properly
  if (CMAKE_CXX_COMPILER_ID STREQUAL "SunPro" AND (CRYPTOPP_SPARC OR CRYPTOPP_SPARC64))
    list(APPEND CRYPTOPP_COMPILE_OPTIONS "-xregs=no%appl")
  endif ()

  # GCC needs to enable use of '/' for division in the assembler
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND CRYPTOPP_COMPILE_OPTIONS "-Wa,--divide")
  endif ()

endif ()

#============================================================================
# Sources & headers
#============================================================================

# Library headers
file(GLOB cryptopp_HEADERS ${SRC_DIR}/*.h)

# Remove headers used to build test suite
list(REMOVE_ITEM cryptopp_HEADERS
    ${SRC_DIR}/bench.h
    ${SRC_DIR}/validate.h
    )

# Test sources. You can use the GNUmakefile to generate the list: `make sources`.
set(cryptopp_SOURCES_TEST
    ${SRC_DIR}/test.cpp
    ${SRC_DIR}/bench1.cpp
    ${SRC_DIR}/bench2.cpp
    ${SRC_DIR}/bench3.cpp
    ${SRC_DIR}/validat0.cpp
    ${SRC_DIR}/validat1.cpp
    ${SRC_DIR}/validat2.cpp
    ${SRC_DIR}/validat3.cpp
    ${SRC_DIR}/validat4.cpp
    ${SRC_DIR}/validat5.cpp
    ${SRC_DIR}/validat6.cpp
    ${SRC_DIR}/validat7.cpp
    ${SRC_DIR}/validat8.cpp
    ${SRC_DIR}/validat9.cpp
    ${SRC_DIR}/validat10.cpp
    ${SRC_DIR}/regtest1.cpp
    ${SRC_DIR}/regtest2.cpp
    ${SRC_DIR}/regtest3.cpp
    ${SRC_DIR}/regtest4.cpp
    ${SRC_DIR}/datatest.cpp
    ${SRC_DIR}/fipsalgt.cpp
    ${SRC_DIR}/fipstest.cpp
    ${SRC_DIR}/dlltest.cpp
    #${SRC_DIR}/adhoc.cpp
    )

# Library sources. You can use the GNUmakefile to generate the list: `make sources`.
# Makefile sorted them at http://github.com/weidai11/cryptopp/pull/426.
file(GLOB cryptopp_SOURCES ${SRC_DIR}/*.cpp)
list(SORT cryptopp_SOURCES)
list(REMOVE_ITEM cryptopp_SOURCES
    ${SRC_DIR}/cryptlib.cpp
    ${SRC_DIR}/cpu.cpp
    ${SRC_DIR}/integer.cpp
    ${SRC_DIR}/pch.cpp
    ${SRC_DIR}/simple.cpp
    ${SRC_DIR}/adhoc.cpp
    ${cryptopp_SOURCES_TEST}
    )
set(cryptopp_SOURCES
    ${SRC_DIR}/cryptlib.cpp
    ${SRC_DIR}/cpu.cpp
    ${SRC_DIR}/integer.cpp
    ${cryptopp_SOURCES}
    )

set(cryptopp_SOURCES_ASM)

if (MSVC AND NOT DISABLE_ASM)
  if (${CMAKE_GENERATOR} MATCHES ".*ARM")
    message(STATUS "Disabling ASM because ARM is specified as target platform.")
  else ()
    enable_language(ASM_MASM)
    list(APPEND cryptopp_SOURCES_ASM
      ${SRC_DIR}/rdrand.asm
      ${SRC_DIR}/rdseed.asm
      )
    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
      list(APPEND cryptopp_SOURCES_ASM
        ${SRC_DIR}/x64dll.asm
        ${SRC_DIR}/x64masm.asm
        )
      set_source_files_properties(${cryptopp_SOURCES_ASM} PROPERTIES COMPILE_DEFINITIONS "_M_X64")
    else ()
      set_source_files_properties(${cryptopp_SOURCES_ASM} PROPERTIES COMPILE_DEFINITIONS "_M_X86" COMPILE_FLAGS "/safeseh")
    endif ()
    set_source_files_properties(${cryptopp_SOURCES_ASM} PROPERTIES LANGUAGE ASM_MASM)
  endif ()
endif ()

#============================================================================
# Architecture flags
#============================================================================

# TODO: Android, AIX, IBM xlC, iOS and a few other profiles are missing.

# New as of Pull Request 461, http://github.com/weidai11/cryptopp/pull/461.
# Must use CMAKE_CXX_COMPILER here due to XLC 13.1 and LLVM front-end.
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Intel" OR CMAKE_CXX_COMPILER MATCHES "xlC")

  if (CRYPTOPP_AMD64 OR CRYPTOPP_I386 OR CRYPTOPP_X32)

    CheckCompileLinkOption("-msse2" CRYPTOPP_IA32_SSE2
                           "${TEST_PROG_DIR}/test_x86_sse2.cxx")
    CheckCompileLinkOption("-mssse3" CRYPTOPP_IA32_SSSE3
                           "${TEST_PROG_DIR}/test_x86_ssse3.cxx")
    CheckCompileLinkOption("-msse4.1" CRYPTOPP_IA32_SSE41
                           "${TEST_PROG_DIR}/test_x86_sse41.cxx")
    CheckCompileLinkOption("-msse4.2" CRYPTOPP_IA32_SSE42
                           "${TEST_PROG_DIR}/test_x86_sse42.cxx")
    CheckCompileLinkOption("-mssse3 -mpclmul" CRYPTOPP_IA32_CLMUL
                           "${TEST_PROG_DIR}/test_x86_clmul.cxx")
    CheckCompileLinkOption("-msse4.1 -maes" CRYPTOPP_IA32_AES
                           "${TEST_PROG_DIR}/test_x86_aes.cxx")
    CheckCompileLinkOption("-mavx" CRYPTOPP_IA32_AVX
                           "${TEST_PROG_DIR}/test_x86_avx.cxx")
    CheckCompileLinkOption("-mavx2" CRYPTOPP_IA32_AVX2
                           "${TEST_PROG_DIR}/test_x86_avx2.cxx")
    CheckCompileLinkOption("-msse4.2 -msha" CRYPTOPP_IA32_SHA
                           "${TEST_PROG_DIR}/test_x86_sha.cxx")
    CheckCompileLinkOption("" CRYPTOPP_MIXED_ASM
                           "${TEST_PROG_DIR}/test_asm_mixed.cxx")

    # https://github.com/weidai11/cryptopp/issues/756
    if (NOT CRYPTOPP_MIXED_ASM)
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_MIXED_ASM")
    endif ()

    if (NOT CRYPTOPP_IA32_SSE2 AND NOT DISABLE_ASM)
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_ASM")
    elseif (CRYPTOPP_IA32_SSE2 AND NOT DISABLE_ASM)
      set_source_files_properties(${SRC_DIR}/sse_simd.cpp PROPERTIES COMPILE_FLAGS "-msse2")
      set_source_files_properties(${SRC_DIR}/chacha_simd.cpp PROPERTIES COMPILE_FLAGS "-msse2")
      set_source_files_properties(${SRC_DIR}/donna_sse.cpp PROPERTIES COMPILE_FLAGS "-msse2")
    endif ()
    if (NOT CRYPTOPP_IA32_SSSE3 AND NOT DISABLE_SSSE3)
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_SSSE3")
    elseif (CRYPTOPP_IA32_SSSE3 AND NOT DISABLE_SSSE3)
      set_source_files_properties(${SRC_DIR}/aria_simd.cpp PROPERTIES COMPILE_FLAGS "-mssse3")
      set_source_files_properties(${SRC_DIR}/cham_simd.cpp PROPERTIES COMPILE_FLAGS "-mssse3")
      set_source_files_properties(${SRC_DIR}/keccak_simd.cpp PROPERTIES COMPILE_FLAGS "-mssse3")
      set_source_files_properties(${SRC_DIR}/lea_simd.cpp PROPERTIES COMPILE_FLAGS "-mssse3")
      set_source_files_properties(${SRC_DIR}/simeck_simd.cpp PROPERTIES COMPILE_FLAGS "-mssse3")
      set_source_files_properties(${SRC_DIR}/simon128_simd.cpp PROPERTIES COMPILE_FLAGS "-mssse3")
      set_source_files_properties(${SRC_DIR}/speck128_simd.cpp PROPERTIES COMPILE_FLAGS "-mssse3")
      if (NOT CRYPTOPP_IA32_SSE41 AND NOT DISABLE_SSE4)
        list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_SSE4")
      elseif (CRYPTOPP_IA32_SSE41 AND NOT DISABLE_SSE4)
        set_source_files_properties(${SRC_DIR}/blake2s_simd.cpp PROPERTIES COMPILE_FLAGS "-msse4.1")
        set_source_files_properties(${SRC_DIR}/blake2b_simd.cpp PROPERTIES COMPILE_FLAGS "-msse4.1")
        set_source_files_properties(${SRC_DIR}/simon64_simd.cpp PROPERTIES COMPILE_FLAGS "-msse4.1")
        set_source_files_properties(${SRC_DIR}/speck64_simd.cpp PROPERTIES COMPILE_FLAGS "-msse4.1")
      endif ()
      if (NOT CRYPTOPP_IA32_SSE42 AND NOT DISABLE_SSE4)
        list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_SSE4")
      elseif (CRYPTOPP_IA32_SSE42 AND NOT DISABLE_SSE4)
        set_source_files_properties(${SRC_DIR}/crc_simd.cpp PROPERTIES COMPILE_FLAGS "-msse4.2")
        if (NOT CRYPTOPP_IA32_CLMUL AND NOT DISABLE_AES)
          list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_CLMUL")
        elseif (CRYPTOPP_IA32_CLMUL AND NOT DISABLE_AES)
          set_source_files_properties(${SRC_DIR}/gcm_simd.cpp PROPERTIES COMPILE_FLAGS "-mssse3 -mpclmul")
          set_source_files_properties(${SRC_DIR}/gf2n_simd.cpp PROPERTIES COMPILE_FLAGS "-mpclmul")
        endif ()
        if (NOT CRYPTOPP_IA32_AES AND NOT DISABLE_AES)
          list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_AESNI")
        elseif (CRYPTOPP_IA32_AES AND NOT DISABLE_AES)
          set_source_files_properties(${SRC_DIR}/rijndael_simd.cpp PROPERTIES COMPILE_FLAGS "-msse4.1 -maes")
          set_source_files_properties(${SRC_DIR}/sm4_simd.cpp PROPERTIES COMPILE_FLAGS "-mssse3 -maes")
        endif ()
        #if (NOT CRYPTOPP_IA32_AVX AND NOT DISABLE_AVX)
        # list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_AVX")
        #elseif (CRYPTOPP_IA32_AVX AND NOT DISABLE_AVX)
        #  set_source_files_properties(${SRC_DIR}/XXX_avx.cpp PROPERTIES COMPILE_FLAGS "-mavx")
        #endif ()
        if (NOT CRYPTOPP_IA32_AVX2 AND NOT DISABLE_AVX2)
          list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_AVX2")
        elseif (CRYPTOPP_IA32_AVX2 AND NOT DISABLE_AVX2)
          set_source_files_properties(${SRC_DIR}/chacha_avx.cpp PROPERTIES COMPILE_FLAGS "-mavx2")
        endif ()
        if (NOT CRYPTOPP_IA32_SHA AND NOT DISABLE_SHA)
          list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_SHANI")
        elseif (CRYPTOPP_IA32_SHA AND NOT DISABLE_SHA)
          set_source_files_properties(${SRC_DIR}/sha_simd.cpp PROPERTIES COMPILE_FLAGS "-msse4.2 -msha")
          set_source_files_properties(${SRC_DIR}/shacal2_simd.cpp PROPERTIES COMPILE_FLAGS "-msse4.2 -msha")
        endif ()
      endif ()
    endif ()

  elseif (CRYPTOPP_AARCH32 OR CRYPTOPP_AARCH64)

    CheckCompileOption("-march=armv8-a" CRYPTOPP_ARMV8A_ASIMD)
    CheckCompileOption("-march=armv8-a+crc" CRYPTOPP_ARMV8A_CRC)
    CheckCompileOption("-march=armv8-a+crypto" CRYPTOPP_ARMV8A_CRYPTO)
    CheckCompileOption("-march=armv8-a" CRYPTOPP_ARMV8A_NATIVE)

    if (CRYPTOPP_ARMV8A_ASIMD)
      set_source_files_properties(${SRC_DIR}/aria_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a")
      set_source_files_properties(${SRC_DIR}/blake2s_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a")
      set_source_files_properties(${SRC_DIR}/blake2b_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a")
      set_source_files_properties(${SRC_DIR}/chacha_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a")
      set_source_files_properties(${SRC_DIR}/cham_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a")
      set_source_files_properties(${SRC_DIR}/lea_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a")
      set_source_files_properties(${SRC_DIR}/neon_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a")
      set_source_files_properties(${SRC_DIR}/simeck_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a")
      set_source_files_properties(${SRC_DIR}/simon64_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a")
      set_source_files_properties(${SRC_DIR}/simon128_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a")
      set_source_files_properties(${SRC_DIR}/speck64_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a")
      set_source_files_properties(${SRC_DIR}/speck128_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a")
    endif ()
    if (CRYPTOPP_ARMV8A_CRC)
      set_source_files_properties(${SRC_DIR}/crc_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a+crc")
    endif ()
    if (CRYPTOPP_ARMV8A_CRYPTO)
      set_source_files_properties(${SRC_DIR}/gcm_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a+crypto")
      set_source_files_properties(${SRC_DIR}/gf2n_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a+crypto")
      set_source_files_properties(${SRC_DIR}/rijndael_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a+crypto")
      set_source_files_properties(${SRC_DIR}/sha_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a+crypto")
      set_source_files_properties(${SRC_DIR}/shacal2_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a+crypto")
    endif ()

  elseif (CRYPTOPP_ARM OR CRYPTOPP_ARMHF)

    CheckCompileLinkOption("-march=armv7-a -mfpu=neon" CRYPTOPP_ARMV7A_NEON
                           "${TEST_PROG_DIR}/test_arm_neon.cxx")
    CheckCompileLinkOption("-march=armv7-a -mthumb -mfpu=neon" CRYPTOPP_ARMV7A_THUMB_NEON
                           "${TEST_PROG_DIR}/test_arm_neon.cxx")

    if (CRYPTOPP_ARMV7A_NEON)

      # Add Cryptogams ASM files for ARM on Linux. Linux is required due to GNU Assembler.
      # AES requires -mthumb under Clang. Do not add -mthumb for SHA for any files.
      if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        list(APPEND cryptopp_SOURCES ${SRC_DIR}/aes_armv4.S)
        list(APPEND cryptopp_SOURCES ${SRC_DIR}/sha1_armv4.S)
        list(APPEND cryptopp_SOURCES ${SRC_DIR}/sha256_armv4.S)
        list(APPEND cryptopp_SOURCES ${SRC_DIR}/sha512_armv4.S)

        set_source_files_properties(${SRC_DIR}/aes_armv4.S PROPERTIES LANGUAGE CXX)
        set_source_files_properties(${SRC_DIR}/sha1_armv4.S PROPERTIES LANGUAGE CXX)
        set_source_files_properties(${SRC_DIR}/sha256_armv4.S PROPERTIES LANGUAGE CXX)
        set_source_files_properties(${SRC_DIR}/sha512_armv4.S PROPERTIES LANGUAGE CXX)

        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
          set_source_files_properties(${SRC_DIR}/aes_armv4.S PROPERTIES COMPILE_FLAGS "-march=armv7-a -mthumb -mfpu=neon -Wa,--noexecstack")
        else ()
          set_source_files_properties(${SRC_DIR}/aes_armv4.S PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon -Wa,--noexecstack")
        endif ()
        
        set_source_files_properties(${SRC_DIR}/sha1_armv4.S PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon -Wa,--noexecstack")
        set_source_files_properties(${SRC_DIR}/sha256_armv4.S PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon -Wa,--noexecstack")
        set_source_files_properties(${SRC_DIR}/sha512_armv4.S PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon -Wa,--noexecstack")
      endif ()

      set_source_files_properties(${SRC_DIR}/aria_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/blake2s_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/blake2b_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/chacha_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/cham_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/crc_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/lea_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/gcm_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/rijndael_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/neon_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/sha_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/simeck_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/simon64_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/simon128_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/speck64_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/speck128_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
      set_source_files_properties(${SRC_DIR}/sm4_simd.cpp PROPERTIES COMPILE_FLAGS "-march=armv7-a -mfpu=neon")
    else ()
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_ARM_NEON_AVAILABLE=0")
    endif ()

  elseif (CRYPTOPP_POWERPC OR CRYPTOPP_POWERPC64)

    if (CMAKE_CXX_COMPILER MATCHES "xlC")
      set(CRYPTOPP_ALTIVEC_FLAGS "-qaltivec")
      set(CRYPTOPP_POWER4_FLAGS "-qarch=pwr4 -qaltivec")
      set(CRYPTOPP_POWER5_FLAGS "-qarch=pwr5 -qaltivec")
      set(CRYPTOPP_POWER6_FLAGS "-qarch=pwr6 -qaltivec")
      set(CRYPTOPP_POWER7_VSX_FLAG  "-qarch=pwr7 -qvsx -qaltivec")
      set(CRYPTOPP_POWER7_PWR_FLAGS "-qarch=pwr7 -qaltivec")
      set(CRYPTOPP_POWER7_ALT_FLAGS "-qaltivec")
      set(CRYPTOPP_POWER8_FLAGS "-qarch=pwr8 -qaltivec")
      set(CRYPTOPP_POWER9_FLAGS "-qarch=pwr9 -qaltivec")
    else ()
      set(CRYPTOPP_ALTIVEC_FLAGS "-maltivec")
      set(CRYPTOPP_POWER7_VSX_FLAGS "-mcpu=power7 -mvsx -maltivec")
      set(CRYPTOPP_POWER7_PWR_FLAGS "-mcpu=power7 -maltivec")
      set(CRYPTOPP_POWER7_ALT_FLAGS "-maltivec")
      set(CRYPTOPP_POWER8_FLAGS "-mcpu=power8 -maltivec")
      set(CRYPTOPP_POWER9_FLAGS "-mcpu=power9 -maltivec")
    endif ()

    CheckCompileLinkOption("${CRYPTOPP_ALTIVEC_FLAGS}" PPC_ALTIVEC_FLAG
                           "${TEST_PROG_DIR}/test_ppc_altivec.cxx")

    # Hack for XLC. Find the lowest PWR architecture.
    if (CMAKE_CXX_COMPILER MATCHES "xlC")
      if (NOT PPC_ALTIVEC_FLAG)
        CheckCompileLinkOption("${CRYPTOPP_POWER4_FLAGS}" PPC_POWER4_FLAG
                               "${TEST_PROG_DIR}/test_ppc_altivec.cxx")
        if (PPC_POWER4_FLAG)
          set(PPC_ALTIVEC_FLAG 1)
          set(CRYPTOPP_ALTIVEC_FLAGS "${CRYPTOPP_POWER4_FLAGS}")
        endif ()
      endif ()
      if (NOT PPC_ALTIVEC_FLAG)
        CheckCompileLinkOption("${CRYPTOPP_POWER5_FLAGS}" PPC_POWER5_FLAG
                               "${TEST_PROG_DIR}/test_ppc_altivec.cxx")
        if (PPC_POWER5_FLAG)
          set(PPC_ALTIVEC_FLAG 1)
          set(CRYPTOPP_ALTIVEC_FLAGS "${CRYPTOPP_POWER5_FLAGS}")
        endif ()
      endif ()
      if (NOT PPC_ALTIVEC_FLAG)
        CheckCompileLinkOption("${CRYPTOPP_POWER6_FLAGS}" PPC_POWER6_FLAG
                               "${TEST_PROG_DIR}/test_ppc_altivec.cxx")
        if (PPC_POWER6_FLAG)
          set(PPC_ALTIVEC_FLAG 1)
          set(CRYPTOPP_ALTIVEC_FLAGS "${CRYPTOPP_POWER6_FLAGS}")
        endif ()
      endif ()
    endif ()

    # Hack for XLC and GCC. Find the right combination for PWR7 and the VSX unit.
    CheckCompileLinkOption("${CRYPTOPP_POWER7_VSX_FLAGS}" PPC_POWER7_FLAG
                           "${TEST_PROG_DIR}/test_ppc_power7.cxx")
    if (PPC_POWER7_FLAG)
      set (CRYPTOPP_POWER7_FLAGS "${CRYPTOPP_POWER7_VSX_FLAGS}")
    else ()
      CheckCompileLinkOption("${CRYPTOPP_POWER7_PWR_FLAGS}" PPC_POWER7_FLAG
                             "${TEST_PROG_DIR}/test_ppc_power7.cxx")
      if (PPC_POWER7_FLAG)
        set (CRYPTOPP_POWER7_FLAGS "${CRYPTOPP_POWER7_PWR_FLAGS}")
      else ()
        CheckCompileLinkOption("${CRYPTOPP_POWER7_ALT_FLAGS}" PPC_POWER7_FLAG
                               "${TEST_PROG_DIR}/test_ppc_power7.cxx")
        if (PPC_POWER7_FLAG)
          set (CRYPTOPP_POWER7_FLAGS "${CRYPTOPP_POWER7_ALT_FLAGS}")
        endif ()
      endif ()
    endif ()

    CheckCompileLinkOption("${CRYPTOPP_POWER8_FLAGS}" PPC_POWER8_FLAG
                           "${TEST_PROG_DIR}/test_ppc_power8.cxx")

    CheckCompileLinkOption("${CRYPTOPP_POWER9_FLAGS}" PPC_POWER9_FLAG
                           "${TEST_PROG_DIR}/test_ppc_power9.cxx")

    if (PPC_POWER9_FLAG AND NOT DISABLE_POWER9)
      set_source_files_properties(${SRC_DIR}/ppc_power9.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER9_FLAGS})
    endif ()

    if (PPC_POWER8_FLAG AND NOT DISABLE_POWER8)
      set_source_files_properties(${SRC_DIR}/ppc_power8.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      #set_source_files_properties(${SRC_DIR}/aria_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/blake2b_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/blake2s_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/chacha_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/cham_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      #set_source_files_properties(${SRC_DIR}/crc_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/gcm_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/gf2n_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/lea_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/rijndael_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/sha_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/shacal2_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/simeck_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/simon64_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/simon128_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/speck64_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
      set_source_files_properties(${SRC_DIR}/speck128_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER8_FLAGS})
    endif ()

    if (PPC_POWER7_FLAG AND NOT DISABLE_POWER7)
      set_source_files_properties(${SRC_DIR}/ppc_power7.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER7_FLAGS})
    endif ()

    if (PPC_ALTIVEC_FLAG AND NOT DISABLE_ALTIVEC)
      set_source_files_properties(${SRC_DIR}/ppc_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_ALTIVEC_FLAGS})
    endif ()

    # Drop to Power7 if Power8 unavailable
    if (NOT PPC_POWER8_FLAG)
      if (PPC_POWER7_FLAG)
        set_source_files_properties(${SRC_DIR}/gcm_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_POWER7_FLAGS})
      endif ()
    endif ()

    # Drop to Altivec if Power7 unavailable
    if (NOT PPC_POWER7_FLAG)
      if (PPC_ALTIVEC_FLAG)
        set_source_files_properties(${SRC_DIR}/blake2s_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_ALTIVEC_FLAGS})
        set_source_files_properties(${SRC_DIR}/chacha_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_ALTIVEC_FLAGS})
        set_source_files_properties(${SRC_DIR}/simon64_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_ALTIVEC_FLAGS})
        set_source_files_properties(${SRC_DIR}/speck64_simd.cpp PROPERTIES COMPILE_FLAGS ${CRYPTOPP_ALTIVEC_FLAGS})
      endif ()
    endif ()

    if (NOT PPC_ALTIVEC_FLAG)
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_ALTIVEC")
    elseif (NOT PPC_POWER7_FLAG)
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_POWER7")
    elseif (NOT PPC_POWER8_FLAG)
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_POWER8")
    elseif (NOT PPC_POWER9_FLAG)
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "-DCRYPTOPP_DISABLE_POWER9")
    endif ()

  endif ()
endif ()

# New as of Pull Request 461, http://github.com/weidai11/cryptopp/pull/461.
if (CMAKE_CXX_COMPILER_ID STREQUAL "SunPro")

  if (CRYPTOPP_AMD64 OR CRYPTOPP_I386 OR CRYPTOPP_X32)

    CheckCompileLinkOption("-xarch=sse2" CRYPTOPP_IA32_SSE2
                           "${TEST_PROG_DIR}/test_x86_sse2.cxx")
    CheckCompileLinkOption("-xarch=ssse3" CRYPTOPP_IA32_SSSE3
                           "${TEST_PROG_DIR}/test_x86_ssse3.cxx")
    CheckCompileLinkOption("-xarch=sse4_1" CRYPTOPP_IA32_SSE41
                           "${TEST_PROG_DIR}/test_x86_sse41.cxx")
    CheckCompileLinkOption("-xarch=sse4_2" CRYPTOPP_IA32_SSE42
                           "${TEST_PROG_DIR}/test_x86_sse42.cxx")
    CheckCompileLinkOption("-xarch=aes" CRYPTOPP_IA32_CLMUL
                           "${TEST_PROG_DIR}/test_x86_clmul.cxx")
    CheckCompileLinkOption("-xarch=aes" CRYPTOPP_IA32_AES
                           "${TEST_PROG_DIR}/test_x86_aes.cxx")
    CheckCompileLinkOption("-xarch=avx" CRYPTOPP_IA32_AVX
                           "${TEST_PROG_DIR}/test_x86_avx.cxx")
    CheckCompileLinkOption("-xarch=avx2" CRYPTOPP_IA32_AVX2
                           "${TEST_PROG_DIR}/test_x86_avx2.cxx")
    CheckCompileLinkOption("-xarch=sha" CRYPTOPP_IA32_SHA
                           "${TEST_PROG_DIR}/test_x86_sha.cxx")

    # Each -xarch=XXX options must be added to LDFLAGS if the option is used during a compile.
    set(XARCH_LDFLAGS "")

    if (CRYPTOPP_IA32_SSE2 AND NOT DISABLE_ASM)
      set_source_files_properties(${SRC_DIR}/sse_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=sse2")
      set_source_files_properties(${SRC_DIR}/chacha_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=sse2")
      set(XARCH_LDFLAGS "-xarch=sse2")
    endif ()
    if (CRYPTOPP_IA32_SSSE3 AND NOT DISABLE_SSSE3)
      set_source_files_properties(${SRC_DIR}/aria_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=ssse3")
      set_source_files_properties(${SRC_DIR}/cham_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=ssse3")
      set_source_files_properties(${SRC_DIR}/lea_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=ssse3")
      set_source_files_properties(${SRC_DIR}/simeck_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=ssse3")
      set_source_files_properties(${SRC_DIR}/simon128_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=ssse3")
      set_source_files_properties(${SRC_DIR}/speck128_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=ssse3")
      set(XARCH_LDFLAGS "${XARCH_LDFLAGS} -xarch=ssse3")
      if (CRYPTOPP_IA32_SSE41 AND NOT DISABLE_SSE4)
        set_source_files_properties(${SRC_DIR}/blake2s_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=sse4_1")
        set_source_files_properties(${SRC_DIR}/blake2b_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=sse4_1")
        set_source_files_properties(${SRC_DIR}/simon64_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=sse4_1")
        set_source_files_properties(${SRC_DIR}/speck64_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=sse4_1")
        set(XARCH_LDFLAGS "${XARCH_LDFLAGS} -xarch=sse4_1")
      endif ()
      if (CRYPTOPP_IA32_SSE42 AND NOT DISABLE_SSE4)
        set_source_files_properties(${SRC_DIR}/crc_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=sse4_2")
        set(XARCH_LDFLAGS "${XARCH_LDFLAGS} -xarch=sse4_2")
        if (CRYPTOPP_IA32_CLMUL AND NOT DISABLE_CLMUL)
          set_source_files_properties(${SRC_DIR}/gcm_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=aes")
          set_source_files_properties(${SRC_DIR}/gf2n_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=aes")
        endif ()
        if (CRYPTOPP_IA32_AES AND NOT DISABLE_AES)
          set_source_files_properties(${SRC_DIR}/rijndael_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=aes")
          set_source_files_properties(${SRC_DIR}/sm4_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=aes")
          set(XARCH_LDFLAGS "${XARCH_LDFLAGS} -xarch=aes")
        endif ()
        #if (CRYPTOPP_IA32_AVX AND NOT DISABLE_AVX)
        #  set_source_files_properties(${SRC_DIR}/XXX_avx.cpp PROPERTIES COMPILE_FLAGS "-xarch=avx2")
        #  set(XARCH_LDFLAGS "${XARCH_LDFLAGS} -xarch=avx")
        #endif ()
        if (CRYPTOPP_IA32_AVX2 AND NOT DISABLE_AVX2)
          set_source_files_properties(${SRC_DIR}/chacha_avx.cpp PROPERTIES COMPILE_FLAGS "-xarch=avx2")
          set(XARCH_LDFLAGS "${XARCH_LDFLAGS} -xarch=avx2")
        endif ()
        if (CRYPTOPP_IA32_SHA AND NOT DISABLE_SHA)
          set_source_files_properties(${SRC_DIR}/sha_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=sha")
          set_source_files_properties(${SRC_DIR}/shacal2_simd.cpp PROPERTIES COMPILE_FLAGS "-xarch=sha")
          set(XARCH_LDFLAGS "${XARCH_LDFLAGS} -xarch=sha")
        endif ()
      endif ()
    endif ()

    # https://stackoverflow.com/a/6088646/608639
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${XARCH_LDFLAGS} -M${SRC_DIR}/cryptopp.mapfile")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${XARCH_LDFLAGS} -M${SRC_DIR}/cryptopp.mapfile")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${XARCH_LDFLAGS} -M${SRC_DIR}/cryptopp.mapfile")

  # elseif (CRYPTOPP_SPARC OR CRYPTOPP_SPARC64)

  endif ()
endif ()

# Attempt to determine a suitable native option
if (CRYPTOPP_NATIVE_ARCH)

    CheckCompileOption("-march=native" NATIVE_ARCH)
    if (NATIVE_ARCH)
      list(APPEND CRYPTOPP_COMPILE_OPTIONS "-march=native")
    else ()
      CheckCompileOption("-native" NATIVE_ARCH)
      if (NATIVE_ARCH)
        list(APPEND CRYPTOPP_COMPILE_OPTIONS "-native")
      endif ()
    endif ()

    if (NOT NATIVE_ARCH)
      message(WARNING "CRYPTOPP_NATIVE_ARCH enabled, but failed to detect native architecture option")
    endif ()

endif()

#============================================================================
# Compile targets
#============================================================================

# Work around the archaic versions of cmake that do not support
# target_compile_xxxx commands
# !!! DO NOT try to use the old way for newer version - it does not work !!!
function(cryptopp_target_compile_properties target)
  if (NOT ${CMAKE_VERSION} VERSION_LESS "2.8.11")
    target_compile_definitions(${target} PUBLIC ${CRYPTOPP_COMPILE_DEFINITIONS})
  else()
    string (REPLACE ";" " " PROP_STR "${CRYPTOPP_COMPILE_DEFINITIONS}")
    set_target_properties(${target} PROPERTIES COMPILE_DEFINITIONS "${CRYPTOPP_COMPILE_DEFINITIONS}")
  endif()
  if (NOT ${CMAKE_VERSION} VERSION_LESS "2.8.12")
    target_compile_options(${target} PUBLIC ${CRYPTOPP_COMPILE_OPTIONS})
  else()
    string (REPLACE ";" " " PROP_STR "${CRYPTOPP_COMPILE_OPTIONS}")
    set_target_properties(${target} PROPERTIES COMPILE_FLAGS "${PROP_STR}")
  endif()
endfunction()

set(cryptopp_LIBRARY_SOURCES ${cryptopp_SOURCES_ASM})
if (USE_INTERMEDIATE_OBJECTS_TARGET AND NOT ${CMAKE_VERSION} VERSION_LESS "2.8.8")
  add_library(cryptopp-object OBJECT ${cryptopp_SOURCES})
  cryptopp_target_compile_properties(cryptopp-object)

  list(APPEND cryptopp_LIBRARY_SOURCES
    $<TARGET_OBJECTS:cryptopp-object>
    )
else ()
  list(APPEND cryptopp_LIBRARY_SOURCES
    ${cryptopp_SOURCES}
    )
endif ()

if (BUILD_STATIC)
  add_library(cryptopp-static STATIC ${cryptopp_LIBRARY_SOURCES})
  cryptopp_target_compile_properties(cryptopp-static)
  if (NOT ${CMAKE_VERSION} VERSION_LESS "2.8.11")
    target_include_directories(cryptopp-static PUBLIC $<BUILD_INTERFACE:${SRC_DIR}> $<INSTALL_INTERFACE:include>)
  else ()
    set_target_properties(cryptopp-static PROPERTIES INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${SRC_DIR}> $<INSTALL_INTERFACE:include>")
  endif ()
endif ()

if (BUILD_SHARED)
  add_library(cryptopp-shared SHARED ${cryptopp_LIBRARY_SOURCES})
  cryptopp_target_compile_properties(cryptopp-shared)
  if (NOT ${CMAKE_VERSION} VERSION_LESS "2.8.11")
    target_include_directories(cryptopp-shared PUBLIC $<BUILD_INTERFACE:${SRC_DIR}> $<INSTALL_INTERFACE:include>)
  else ()
    set_target_properties(cryptopp-shared PROPERTIES INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${SRC_DIR}> $<INSTALL_INTERFACE:include>")
  endif ()
endif ()

# Set filenames for targets to be "cryptopp"
if (NOT MSVC)
  set(COMPAT_VERSION ${cryptopp_VERSION_MAJOR}.${cryptopp_VERSION_MINOR})

  if (BUILD_STATIC)
    set_target_properties(cryptopp-static
        PROPERTIES
        OUTPUT_NAME cryptopp)
  endif ()
  if (BUILD_SHARED)
    set_target_properties(cryptopp-shared
        PROPERTIES
        SOVERSION ${COMPAT_VERSION}
        OUTPUT_NAME cryptopp)
  endif ()
endif ()

# Add alternate ways to invoke the build for the shared library that are
# similar to how the crypto++ 'make' tool works.
# see https://github.com/noloader/cryptopp-cmake/issues/32
if (BUILD_STATIC)
  add_custom_target(static DEPENDS cryptopp-static)
endif ()
if (BUILD_SHARED)
  add_custom_target(shared DEPENDS cryptopp-shared)
  add_custom_target(dynamic DEPENDS cryptopp-shared)
endif ()

#============================================================================
# Third-party libraries
#============================================================================

if (WIN32)
  if (BUILD_STATIC)
    target_link_libraries(cryptopp-static ws2_32)
  endif ()
  if (BUILD_SHARED)
    target_link_libraries(cryptopp-shared ws2_32)
  endif ()
endif ()

# This may need to be expanded to "Solaris"
if (CRYPTOPP_SOLARIS)
  if (BUILD_STATIC)
    target_link_libraries(cryptopp-static nsl socket)
  endif ()
  if (BUILD_SHARED)
    target_link_libraries(cryptopp-shared nsl socket)
  endif ()
endif ()

find_package(Threads)
if (BUILD_STATIC)
  target_link_libraries(cryptopp-static ${CMAKE_THREAD_LIBS_INIT})
endif ()
if (BUILD_SHARED)
  target_link_libraries(cryptopp-shared ${CMAKE_THREAD_LIBS_INIT})
endif ()

#============================================================================
# Tests
#============================================================================

enable_testing()
if (BUILD_TESTING)
  add_executable(cryptest ${cryptopp_SOURCES_TEST})
  target_link_libraries(cryptest cryptopp-static)

  # Setting "cryptest" binary name to "cryptest.exe"
  if (NOT (WIN32 OR CYGWIN))
    set_target_properties(cryptest PROPERTIES OUTPUT_NAME cryptest.exe)
  endif ()
  if (NOT TARGET cryptest.exe)
    add_custom_target(cryptest.exe)
    add_dependencies(cryptest.exe cryptest)
  endif ()

  file(COPY ${SRC_DIR}/TestData DESTINATION ${PROJECT_BINARY_DIR})
  file(COPY ${SRC_DIR}/TestVectors DESTINATION ${PROJECT_BINARY_DIR})

  add_test(NAME build_cryptest COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target cryptest)
  add_test(NAME cryptest COMMAND $<TARGET_FILE:cryptest> v)
  set_tests_properties(cryptest PROPERTIES DEPENDS build_cryptest)
endif ()

#============================================================================
# Doxygen documentation
#============================================================================

if (BUILD_DOCUMENTATION)
  find_package(Doxygen REQUIRED)

  set(in_source_DOCS_DIR "${SRC_DIR}/html-docs")
  set(out_source_DOCS_DIR "${PROJECT_BINARY_DIR}/html-docs")

  add_custom_target(docs ALL
      COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile -d CRYPTOPP_DOXYGEN_PROCESSING
      WORKING_DIRECTORY ${SRC_DIR}
      SOURCES ${SRC_DIR}/Doxyfile
      )

  if (NOT ${in_source_DOCS_DIR} STREQUAL ${out_source_DOCS_DIR})
    add_custom_command(
        TARGET docs POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${in_source_DOCS_DIR}" "${out_source_DOCS_DIR}"
        COMMAND ${CMAKE_COMMAND} -E remove_directory "${in_source_DOCS_DIR}"
    )
  endif ()
endif ()

#============================================================================
# Install
#============================================================================

set(export_name "cryptopp-targets")

# Runtime package
if (BUILD_SHARED)
  export(TARGETS cryptopp-shared FILE ${export_name}.cmake )
  install(
      TARGETS cryptopp-shared
      EXPORT ${export_name}
      DESTINATION lib
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
  )
endif ()

# Development package
if (BUILD_STATIC)
  export(TARGETS cryptopp-static FILE ${export_name}.cmake )
  install(TARGETS cryptopp-static EXPORT ${export_name} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif ()
install(FILES ${cryptopp_HEADERS} DESTINATION include/cryptopp)

# CMake Package
if (NOT CMAKE_VERSION VERSION_LESS 2.8.8)
  include(CMakePackageConfigHelpers)
  write_basic_package_version_file("${PROJECT_BINARY_DIR}/cryptopp-config-version.cmake" VERSION ${cryptopp_VERSION_MAJOR}.${cryptopp_VERSION_MINOR}.${cryptopp_VERSION_PATCH} COMPATIBILITY SameMajorVersion)
  install(FILES cryptopp-config.cmake ${PROJECT_BINARY_DIR}/cryptopp-config-version.cmake DESTINATION "lib/cmake/cryptopp")
  install(EXPORT ${export_name} DESTINATION "lib/cmake/cryptopp")
endif ()

# Tests
if (BUILD_TESTING)
  install(TARGETS cryptest DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(DIRECTORY ${SRC_DIR}/TestData DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cryptopp)
  install(DIRECTORY ${SRC_DIR}/TestVectors DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cryptopp)
endif ()

# Documentation
if (BUILD_DOCUMENTATION)
  install(DIRECTORY "${out_source_DOCS_DIR}" DESTINATION ${CMAKE_INSTALL_DOCDIR})
endif ()

# Print a configuration summary. We want CXX and CXXFLAGS, but they are not includd in ALL.
if (CRYPTOPP_I386)
  message(STATUS "Platform: i386/i686")
elseif (CRYPTOPP_AMD64)
  message(STATUS "Platform: x86_64")
elseif (CRYPTOPP_X32)
  message(STATUS "Platform: x86_64-x32")
elseif (CRYPTOPP_ARMHF)
  message(STATUS "Platform: armhf")
elseif (CRYPTOPP_ARM)
  message(STATUS "Platform: arm")
elseif (CRYPTOPP_AARCH32)
  message(STATUS "Platform: Aarch32")
elseif (CRYPTOPP_AARCH64)
  message(STATUS "Platform: Aarch64")
elseif (CRYPTOPP_SPARC)
  message(STATUS "Platform: Sparc")
elseif (CRYPTOPP_SPARC64)
  message(STATUS "Platform: Sparc64")
elseif (CRYPTOPP_POWERPC)
  message(STATUS "Platform: PowerPC")
elseif (CRYPTOPP_POWERPC64)
  message(STATUS "Platform: PowerPC-64")
elseif (CRYPTOPP_MINGW32)
  message(STATUS "Platform: MinGW-32")
elseif (CRYPTOPP_MINGW32)
  message(STATUS "Platform: MinGW-64")
endif ()
if (CRYPTOPP_ARMV7A_NEON)
  message(STATUS "NEON: TRUE")
endif ()
if (CRYPTOPP_NATIVE_ARCH)
  message(STATUS "Native arch: TRUE")
else ()
  message(STATUS "Native arch: FALSE")
endif ()
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler options: ${CMAKE_CXX_FLAGS} ${CRYPTOPP_COMPILE_OPTIONS}")
message(STATUS "Compiler definitions: ${CRYPTOPP_COMPILE_DEFINITIONS}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")