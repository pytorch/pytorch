# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# determine the compiler to use for Fortran programs
# NOTE, a generator may set CMAKE_Fortran_COMPILER before
# loading this file to force a compiler.
# use environment variable FC first if defined by user, next use
# the cmake variable CMAKE_GENERATOR_FC which can be defined by a generator
# as a default compiler

include(${CMAKE_ROOT}/Modules/CMakeDetermineCompiler.cmake)
include(Platform/${CMAKE_SYSTEM_NAME}-Determine-Fortran OPTIONAL)
include(Platform/${CMAKE_SYSTEM_NAME}-Fortran OPTIONAL)
if(NOT CMAKE_Fortran_COMPILER_NAMES)
  set(CMAKE_Fortran_COMPILER_NAMES f95)
endif()

if(${CMAKE_GENERATOR} MATCHES "Visual Studio")
elseif("${CMAKE_GENERATOR}" MATCHES "Xcode")
  set(CMAKE_Fortran_COMPILER_XCODE_TYPE sourcecode.fortran.f90)
  _cmake_find_compiler_path(Fortran)
else()
  if(NOT CMAKE_Fortran_COMPILER)
    # prefer the environment variable FC
    if(NOT $ENV{FC} STREQUAL "")
      get_filename_component(CMAKE_Fortran_COMPILER_INIT $ENV{FC} PROGRAM PROGRAM_ARGS CMAKE_Fortran_FLAGS_ENV_INIT)
      if(CMAKE_Fortran_FLAGS_ENV_INIT)
        set(CMAKE_Fortran_COMPILER_ARG1 "${CMAKE_Fortran_FLAGS_ENV_INIT}" CACHE STRING "Arguments to Fortran compiler")
      endif()
      if(EXISTS ${CMAKE_Fortran_COMPILER_INIT})
      else()
        message(FATAL_ERROR "Could not find compiler set in environment variable FC:\n$ENV{FC}.")
      endif()
    endif()

    # next try prefer the compiler specified by the generator
    if(CMAKE_GENERATOR_FC)
      if(NOT CMAKE_Fortran_COMPILER_INIT)
        set(CMAKE_Fortran_COMPILER_INIT ${CMAKE_GENERATOR_FC})
      endif()
    endif()

    # finally list compilers to try
    if(NOT CMAKE_Fortran_COMPILER_INIT)
      # Known compilers:
      #  ftn: Cray fortran compiler wrapper
      #  gfortran: putative GNU Fortran 95+ compiler (in progress)
      #  frt: Fujitsu Fortran compiler
      #  pathf90/pathf95/pathf2003: PathScale Fortran compiler
      #  pgfortran: Portland Group Fortran compilers
      #  nvfortran: NVHPC Fotran compiler
      #  flang: Flang Fortran compiler
      #  xlf: IBM (AIX) Fortran compiler
      #  lf95: Lahey-Fujitsu F95 compiler
      #  fl32: Microsoft Fortran 77 "PowerStation" compiler
      #  af77: Apogee F77 compiler for Intergraph hardware running CLIX
      #  epcf90: "Edinburgh Portable Compiler" F90
      #  fort: Compaq (now HP) Fortran 90/95 compiler for Tru64 and Linux/Alpha
      #  ifx: Intel Fortran LLVM-based compiler
      #  ifort: Intel Classic Fortran compiler
      #  nagfor: NAG Fortran compiler
      #  lfortran: LFortran Fortran Compiler
      #
      #  GNU is last to be searched,
      #  so if you paid for a compiler it is picked by default.
      set(CMAKE_Fortran_COMPILER_LIST
        ftn
        ifx ifort nvfortran pgfortran lf95 xlf fort
        flang lfortran frt nagfor
        gfortran
        )

      # Vendor-specific compiler names.
      set(_Fortran_COMPILER_NAMES_LCC       lfortran gfortran)
      set(_Fortran_COMPILER_NAMES_GNU       gfortran)
      set(_Fortran_COMPILER_NAMES_Intel     ifort ifc efc ifx)
      set(_Fortran_COMPILER_NAMES_Absoft    af95 af90 af77)
      set(_Fortran_COMPILER_NAMES_PGI       pgf95 pgfortran pgf90 pgf77)
      set(_Fortran_COMPILER_NAMES_Flang     flang)
      set(_Fortran_COMPILER_NAMES_LLVMFlang flang)
      set(_Fortran_COMPILER_NAMES_PathScale pathf2003 pathf95 pathf90)
      set(_Fortran_COMPILER_NAMES_XL        xlf)
      set(_Fortran_COMPILER_NAMES_VisualAge xlf95 xlf90 xlf)
      set(_Fortran_COMPILER_NAMES_NAG       nagfor)
      set(_Fortran_COMPILER_NAMES_NVHPC     nvfortran)
    endif()

    _cmake_find_compiler(Fortran)

  else()
    _cmake_find_compiler_path(Fortran)
  endif()
  mark_as_advanced(CMAKE_Fortran_COMPILER)

  # Each entry in this list is a set of extra flags to try
  # adding to the compile line to see if it helps produce
  # a valid identification executable.
  set(CMAKE_Fortran_COMPILER_ID_TEST_FLAGS_FIRST
    # Get verbose output to help distinguish compilers.
    "-v"

    # Try compiling to an object file only, with verbose output.
    "-v -c"
    )
  set(CMAKE_Fortran_COMPILER_ID_TEST_FLAGS
    # Try compiling to an object file only.
    "-c"

    # Intel on windows does not preprocess by default.
    "-fpp"

    # LFortran does not preprocess by default.
    "--cpp-infer"
    )
endif()

if(CMAKE_Fortran_COMPILER_TARGET)
  set(CMAKE_Fortran_COMPILER_ID_TEST_FLAGS_FIRST "-v -c --target=${CMAKE_Fortran_COMPILER_TARGET}")
endif()

# Build a small source file to identify the compiler.
if(NOT CMAKE_Fortran_COMPILER_ID_RUN)
  set(CMAKE_Fortran_COMPILER_ID_RUN 1)

  # Table of per-vendor compiler output regular expressions.
  list(APPEND CMAKE_Fortran_COMPILER_ID_MATCH_VENDORS CCur)
  set(CMAKE_Fortran_COMPILER_ID_MATCH_VENDOR_REGEX_CCur "Concurrent Fortran [0-9]+ Compiler")

  # Table of per-vendor compiler id flags with expected output.
  list(APPEND CMAKE_Fortran_COMPILER_ID_VENDORS Compaq)
  set(CMAKE_Fortran_COMPILER_ID_VENDOR_FLAGS_Compaq "-what")
  set(CMAKE_Fortran_COMPILER_ID_VENDOR_REGEX_Compaq "Compaq Visual Fortran")
  list(APPEND CMAKE_Fortran_COMPILER_ID_VENDORS NAG) # Numerical Algorithms Group
  set(CMAKE_Fortran_COMPILER_ID_VENDOR_FLAGS_NAG "-V")
  set(CMAKE_Fortran_COMPILER_ID_VENDOR_REGEX_NAG "NAG Fortran Compiler")

  # Match the link line from xcodebuild output of the form
  #  Ld ...
  #      ...
  #      /path/to/cc ...CompilerIdFortran/...
  # to extract the compiler front-end for the language.
  set(CMAKE_Fortran_COMPILER_ID_TOOL_MATCH_REGEX "\nLd[^\n]*(\n[ \t]+[^\n]*)*\n[ \t]+([^ \t\r\n]+)[^\r\n]*-o[^\r\n]*CompilerIdFortran/(\\./)?(CompilerIdFortran.xctest/)?CompilerIdFortran[ \t\n\\\"]")
  set(CMAKE_Fortran_COMPILER_ID_TOOL_MATCH_INDEX 2)

  set(_version_info "")
  foreach(m IN ITEMS MAJOR MINOR PATCH TWEAK)
    set(_COMP "_${m}")
    string(APPEND _version_info "
#if defined(COMPILER_VERSION${_COMP})")
    foreach(d RANGE 1 8)
      string(APPEND _version_info "
# undef DEC
# undef HEX
# define DEC(n) DEC_${d}(n)
# define HEX(n) HEX_${d}(n)
# if COMPILER_VERSION${_COMP} == 0
        PRINT *, 'INFO:compiler_version${_COMP}_digit_${d}[0]'
# elif COMPILER_VERSION${_COMP} == 1
        PRINT *, 'INFO:compiler_version${_COMP}_digit_${d}[1]'
# elif COMPILER_VERSION${_COMP} == 2
        PRINT *, 'INFO:compiler_version${_COMP}_digit_${d}[2]'
# elif COMPILER_VERSION${_COMP} == 3
        PRINT *, 'INFO:compiler_version${_COMP}_digit_${d}[3]'
# elif COMPILER_VERSION${_COMP} == 4
        PRINT *, 'INFO:compiler_version${_COMP}_digit_${d}[4]'
# elif COMPILER_VERSION${_COMP} == 5
        PRINT *, 'INFO:compiler_version${_COMP}_digit_${d}[5]'
# elif COMPILER_VERSION${_COMP} == 6
        PRINT *, 'INFO:compiler_version${_COMP}_digit_${d}[6]'
# elif COMPILER_VERSION${_COMP} == 7
        PRINT *, 'INFO:compiler_version${_COMP}_digit_${d}[7]'
# elif COMPILER_VERSION${_COMP} == 8
        PRINT *, 'INFO:compiler_version${_COMP}_digit_${d}[8]'
# elif COMPILER_VERSION${_COMP} == 9
        PRINT *, 'INFO:compiler_version${_COMP}_digit_${d}[9]'
# endif
")
    endforeach()
    string(APPEND _version_info "
#endif")
  endforeach()
  set(CMAKE_Fortran_COMPILER_ID_VERSION_INFO "${_version_info}")
  unset(_version_info)
  unset(_COMP)

  # Try to identify the compiler.
  set(CMAKE_Fortran_COMPILER_ID)
  include(${CMAKE_ROOT}/Modules/CMakeDetermineCompilerId.cmake)
  CMAKE_DETERMINE_COMPILER_ID(Fortran FFLAGS CMakeFortranCompilerId.F)

  _cmake_find_compiler_sysroot(Fortran)

  # Fall back to old is-GNU test.
  if(NOT CMAKE_Fortran_COMPILER_ID)
    execute_process(COMMAND ${CMAKE_Fortran_COMPILER} ${CMAKE_Fortran_COMPILER_ID_FLAGS_LIST} -E "${CMAKE_ROOT}/Modules/CMakeTestGNU.c"
      OUTPUT_VARIABLE CMAKE_COMPILER_OUTPUT RESULT_VARIABLE CMAKE_COMPILER_RETURN)
    if(NOT CMAKE_COMPILER_RETURN)
      if(CMAKE_COMPILER_OUTPUT MATCHES "THIS_IS_GNU")
        set(CMAKE_Fortran_COMPILER_ID "GNU")
        message(CONFIGURE_LOG
          "Determining if the Fortran compiler is GNU succeeded with "
          "the following output:\n${CMAKE_COMPILER_OUTPUT}\n\n")
      else()
        message(CONFIGURE_LOG
          "Determining if the Fortran compiler is GNU failed with "
          "the following output:\n${CMAKE_COMPILER_OUTPUT}\n\n")
      endif()
      if(NOT CMAKE_Fortran_PLATFORM_ID)
        if(CMAKE_COMPILER_OUTPUT MATCHES "THIS_IS_MINGW")
          set(CMAKE_Fortran_PLATFORM_ID "MinGW")
        endif()
        if(CMAKE_COMPILER_OUTPUT MATCHES "THIS_IS_CYGWIN")
          set(CMAKE_Fortran_PLATFORM_ID "Cygwin")
        endif()
      endif()
    endif()
  endif()

  # Fall back for GNU MINGW, which is not always detected correctly
  # (__MINGW32__ is defined for the C language, but perhaps not for Fortran!)
  if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU" AND NOT CMAKE_Fortran_PLATFORM_ID)
    execute_process(COMMAND ${CMAKE_Fortran_COMPILER} ${CMAKE_Fortran_COMPILER_ID_FLAGS_LIST} -E "${CMAKE_ROOT}/Modules/CMakeTestGNU.c"
      OUTPUT_VARIABLE CMAKE_COMPILER_OUTPUT RESULT_VARIABLE CMAKE_COMPILER_RETURN)
    if(NOT CMAKE_COMPILER_RETURN)
      if(CMAKE_COMPILER_OUTPUT MATCHES "THIS_IS_MINGW")
        set(CMAKE_Fortran_PLATFORM_ID "MinGW")
      endif()
      if(CMAKE_COMPILER_OUTPUT MATCHES "THIS_IS_CYGWIN")
        set(CMAKE_Fortran_PLATFORM_ID "Cygwin")
      endif()
    endif()
  endif()

  # Set old compiler and platform id variables.
  if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    set(CMAKE_COMPILER_IS_GNUG77 1)
  endif()
endif()

if("${CMAKE_Fortran_COMPILER_ID};${CMAKE_Fortran_SIMULATE_ID}" STREQUAL "LLVMFlang;MSVC")
  # With LLVMFlang targeting the MSVC ABI we link using lld-link.
  # Detect the implicit link information from the compiler driver
  # so we can explicitly pass it to the linker.
  include(${CMAKE_ROOT}/Modules/CMakeParseImplicitLinkInfo.cmake)
  set(_LLVMFlang_COMMAND "${CMAKE_Fortran_COMPILER}" "-###" ${CMAKE_CURRENT_LIST_DIR}/CMakeFortranCompilerABI.F)
  if(CMAKE_Fortran_COMPILER_TARGET)
    list(APPEND _LLVMFlang_COMMAND --target=${CMAKE_Fortran_COMPILER_TARGET})
  endif()
  execute_process(COMMAND ${_LLVMFlang_COMMAND}
    OUTPUT_VARIABLE _LLVMFlang_OUTPUT
    ERROR_VARIABLE _LLVMFlang_OUTPUT
    RESULT_VARIABLE _LLVMFlang_RESULT)
  string(JOIN "\" \"" _LLVMFlang_COMMAND ${_LLVMFlang_COMMAND})
  message(CONFIGURE_LOG
    "Running the Fortran compiler: \"${_LLVMFlang_COMMAND}\"\n"
    "${_LLVMFlang_OUTPUT}"
    )
  if(_LLVMFlang_RESULT EQUAL 0)
    cmake_parse_implicit_link_info("${_LLVMFlang_OUTPUT}"
                                   CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES
                                   CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES
                                   CMAKE_Fortran_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES
                                   log
                                   "${CMAKE_Fortran_IMPLICIT_OBJECT_REGEX}"
                                   LANGUAGE Fortran)
    message(CONFIGURE_LOG
      "Parsed Fortran implicit link information:\n"
      "${log}\n"
      )
    set(_CMAKE_Fortran_IMPLICIT_LINK_INFORMATION_DETERMINED_EARLY 1)
    if("x${CMAKE_Fortran_COMPILER_ARCHITECTURE_ID}" STREQUAL "xARM64" AND CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 18.0)
      # LLVMFlang < 18.0 does not add `-defaultlib:` fields to object
      # files to specify link dependencies on its runtime libraries.
      # For now, we add them ourselves.
      list(APPEND CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES "clang_rt.builtins-aarch64.lib")
    endif()
  endif()
  unset(_LLVMFlang_COMMAND)
  unset(_LLVMFlang_OUTPUT)
  unset(_LLVMFlang_RESULT)
endif()

if (NOT _CMAKE_TOOLCHAIN_LOCATION)
  get_filename_component(_CMAKE_TOOLCHAIN_LOCATION "${CMAKE_Fortran_COMPILER}" PATH)
endif ()

# if we have a fortran cross compiler, they have usually some prefix, like
# e.g. powerpc-linux-gfortran, arm-elf-gfortran or i586-mingw32msvc-gfortran , optionally
# with a 3-component version number at the end (e.g. arm-eabi-gcc-4.5.2).
# The other tools of the toolchain usually have the same prefix
# NAME_WE cannot be used since then this test will fail for names like
# "arm-unknown-nto-qnx6.3.0-gcc.exe", where BASENAME would be
# "arm-unknown-nto-qnx6" instead of the correct "arm-unknown-nto-qnx6.3.0-"
if (NOT _CMAKE_TOOLCHAIN_PREFIX)

  if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    get_filename_component(COMPILER_BASENAME "${CMAKE_Fortran_COMPILER}" NAME)
    if (COMPILER_BASENAME MATCHES "^(.+-)g?fortran(-[0-9]+\\.[0-9]+\\.[0-9]+)?(\\.exe)?$")
      set(_CMAKE_TOOLCHAIN_PREFIX ${CMAKE_MATCH_1})
    endif ()

    # if "llvm-" is part of the prefix, remove it, since llvm doesn't have its own binutils
    # but uses the regular ar, objcopy, etc. (instead of llvm-objcopy etc.)
    if ("${_CMAKE_TOOLCHAIN_PREFIX}" MATCHES "(.+-)?llvm-$")
      set(_CMAKE_TOOLCHAIN_PREFIX ${CMAKE_MATCH_1})
    endif ()
  endif()

endif ()

set(_CMAKE_PROCESSING_LANGUAGE "Fortran")
include(CMakeFindBinUtils)
include(Compiler/${CMAKE_Fortran_COMPILER_ID}-FindBinUtils OPTIONAL)
unset(_CMAKE_PROCESSING_LANGUAGE)

if(CMAKE_Fortran_XL_CPP)
  set(_SET_CMAKE_Fortran_XL_CPP
    "set(CMAKE_Fortran_XL_CPP \"${CMAKE_Fortran_XL_CPP}\")")
endif()

if(CMAKE_Fortran_COMPILER_SYSROOT)
  string(CONCAT _SET_CMAKE_Fortran_COMPILER_SYSROOT
    "set(CMAKE_Fortran_COMPILER_SYSROOT \"${CMAKE_Fortran_COMPILER_SYSROOT}\")\n"
    "set(CMAKE_COMPILER_SYSROOT \"${CMAKE_Fortran_COMPILER_SYSROOT}\")")
else()
  set(_SET_CMAKE_Fortran_COMPILER_SYSROOT "")
endif()

if(MSVC_Fortran_ARCHITECTURE_ID)
  set(SET_MSVC_Fortran_ARCHITECTURE_ID
    "set(MSVC_Fortran_ARCHITECTURE_ID ${MSVC_Fortran_ARCHITECTURE_ID})")
endif()
if(CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
  set(CMAKE_Fortran_VENDOR_SOURCE_FILE_EXTENSIONS ";cuf;CUF")
endif()
# configure variables set in this file for fast reload later on
configure_file(${CMAKE_ROOT}/Modules/CMakeFortranCompiler.cmake.in
  ${CMAKE_PLATFORM_INFO_DIR}/CMakeFortranCompiler.cmake
  @ONLY
  )
set(CMAKE_Fortran_COMPILER_ENV_VAR "FC")
