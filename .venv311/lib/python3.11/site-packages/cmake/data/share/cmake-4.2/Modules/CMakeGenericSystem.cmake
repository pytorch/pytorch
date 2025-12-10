# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

include(CMakeInitializeConfigs)

set(CMAKE_SHARED_LIBRARY_C_FLAGS "")            # -pic
set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "-shared")       # -shared
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")         # +s, flag for exe link to use shared lib
set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG "")       # -rpath
set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG_SEP "")   # : or empty
set(CMAKE_INCLUDE_FLAG_C "-I")       # -I
set(CMAKE_LIBRARY_PATH_FLAG "-L")
set(CMAKE_LIBRARY_PATH_TERMINATOR "")  # for the Digital Mars D compiler the link paths have to be terminated with a "/"
set(CMAKE_LINK_LIBRARY_FLAG "-l")

set(CMAKE_LINK_LIBRARY_SUFFIX "")
set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
set(CMAKE_STATIC_LIBRARY_SUFFIX ".a")
set(CMAKE_SHARED_LIBRARY_PREFIX "lib")          # lib
set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")          # .so
set(CMAKE_EXECUTABLE_SUFFIX "")          # .exe
set(CMAKE_DL_LIBS "dl")

set(CMAKE_FIND_LIBRARY_PREFIXES "lib")
set(CMAKE_FIND_LIBRARY_SUFFIXES ".so" ".a")

# Define feature "DEFAULT" as supported. This special feature generates the
# default option to link a library
# This feature is intended to be used in LINK_LIBRARY_OVERRIDE and
# LINK_LIBRARY_OVERRIDE_<LIBRARY> target properties
set(CMAKE_LINK_LIBRARY_USING_DEFAULT_SUPPORTED TRUE)

if(NOT DEFINED CMAKE_AUTOGEN_ORIGIN_DEPENDS)
  set(CMAKE_AUTOGEN_ORIGIN_DEPENDS ON)
endif()
if(NOT DEFINED CMAKE_AUTOMOC_COMPILER_PREDEFINES)
  set(CMAKE_AUTOMOC_COMPILER_PREDEFINES ON)
endif()
if(NOT DEFINED CMAKE_AUTOMOC_PATH_PREFIX)
  set(CMAKE_AUTOMOC_PATH_PREFIX OFF)
endif()
if(NOT DEFINED CMAKE_AUTOMOC_MACRO_NAMES)
  set(CMAKE_AUTOMOC_MACRO_NAMES "Q_OBJECT" "Q_GADGET" "Q_NAMESPACE" "Q_NAMESPACE_EXPORT")
endif()

# basically all general purpose OSs support shared libs
set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

set (CMAKE_SKIP_RPATH "NO" CACHE BOOL
     "If set, runtime paths are not added when using shared libraries.")
set (CMAKE_SKIP_INSTALL_RPATH "NO" CACHE BOOL
     "If set, runtime paths are not added when installing shared libraries, but are added when building.")

set(CMAKE_VERBOSE_MAKEFILE FALSE CACHE BOOL "If this value is on, makefiles will be generated without the .SILENT directive, and all commands will be echoed to the console during the make.  This is useful for debugging only. With Visual Studio IDE projects all commands are done without /nologo.")

if(DEFINED ENV{CMAKE_COLOR_DIAGNOSTICS} AND NOT DEFINED CACHE{CMAKE_COLOR_DIAGNOSTICS})
  set(CMAKE_COLOR_DIAGNOSTICS $ENV{CMAKE_COLOR_DIAGNOSTICS} CACHE BOOL "Enable colored diagnostics throughout.")
endif()

if(CMAKE_GENERATOR MATCHES "Make")
  if(NOT DEFINED CMAKE_COLOR_DIAGNOSTICS)
    set(CMAKE_COLOR_MAKEFILE ON CACHE BOOL "Enable/Disable color output during build.")
  endif()
  mark_as_advanced(CMAKE_COLOR_MAKEFILE)

  if(DEFINED CMAKE_RULE_MESSAGES)
    set_property(GLOBAL PROPERTY RULE_MESSAGES ${CMAKE_RULE_MESSAGES})
  endif()
  if(DEFINED CMAKE_TARGET_MESSAGES)
    set_property(GLOBAL PROPERTY TARGET_MESSAGES ${CMAKE_TARGET_MESSAGES})
  endif()
endif()

if(NOT DEFINED CMAKE_EXPORT_COMPILE_COMMANDS AND CMAKE_GENERATOR MATCHES "Ninja|Unix Makefiles")
  set(CMAKE_EXPORT_COMPILE_COMMANDS "$ENV{CMAKE_EXPORT_COMPILE_COMMANDS}"
    CACHE BOOL "Enable/Disable output of compile commands during generation."
    )
  mark_as_advanced(CMAKE_EXPORT_COMPILE_COMMANDS)
endif()

if(NOT DEFINED CMAKE_EXPORT_BUILD_DATABASE AND CMAKE_GENERATOR MATCHES "Ninja")
  set(CMAKE_EXPORT_BUILD_DATABASE "$ENV{CMAKE_EXPORT_BUILD_DATABASE}"
    CACHE BOOL "Enable/Disable output of build database during the build."
    )
  mark_as_advanced(CMAKE_EXPORT_BUILD_DATABASE)
endif()

# GetDefaultWindowsPrefixBase
#
# Compute the base directory for CMAKE_INSTALL_PREFIX based on:
#  - is this 32-bit or 64-bit Windows
#  - is this 32-bit or 64-bit CMake running
#  - what architecture targets will be built
#
function(GetDefaultWindowsPrefixBase var)

  # Try to guess what architecture targets will end up being built as,
  # even if CMAKE_SIZEOF_VOID_P is not computed yet... We need to know
  # the architecture of the targets being built to choose the right
  # default value for CMAKE_INSTALL_PREFIX.
  #
  if("${CMAKE_GENERATOR}" MATCHES "(Win64|IA64)")
    set(arch_hint "x64")
  elseif("${CMAKE_GENERATOR_PLATFORM}" MATCHES "x64")
    set(arch_hint "x64")
  elseif("${CMAKE_GENERATOR_PLATFORM}" MATCHES "ARM64")
    set(arch_hint "ARM64")
  elseif("${CMAKE_GENERATOR}" MATCHES "ARM")
    set(arch_hint "ARM")
  elseif("${CMAKE_GENERATOR_PLATFORM}" MATCHES "ARM")
    set(arch_hint "ARM")
  elseif("${CMAKE_SIZEOF_VOID_P}" STREQUAL "8")
    set(arch_hint "x64")
  elseif("$ENV{LIB}" MATCHES "(amd64|ia64)")
    set(arch_hint "x64")
  endif()

  if(NOT arch_hint)
    set(arch_hint "x86")
  endif()

  # default env in a 64-bit app on Win64:
  # ProgramFiles=C:\Program Files
  # ProgramFiles(x86)=C:\Program Files (x86)
  # ProgramW6432=C:\Program Files
  #
  # default env in a 32-bit app on Win64:
  # ProgramFiles=C:\Program Files (x86)
  # ProgramFiles(x86)=C:\Program Files (x86)
  # ProgramW6432=C:\Program Files
  #
  # default env in a 32-bit app on Win32:
  # ProgramFiles=C:\Program Files
  # ProgramFiles(x86) NOT DEFINED
  # ProgramW6432 NOT DEFINED

  # By default, use the ProgramFiles env var as the base value of
  # CMAKE_INSTALL_PREFIX:
  #
  set(_PREFIX_ENV_VAR "ProgramFiles")

  if ("$ENV{ProgramW6432}" STREQUAL "")
    # running on 32-bit Windows
    # must be a 32-bit CMake, too...
    #message("guess: this is a 32-bit CMake running on 32-bit Windows")
  else()
    # running on 64-bit Windows
    if ("$ENV{ProgramW6432}" STREQUAL "$ENV{ProgramFiles}")
      # 64-bit CMake
      #message("guess: this is a 64-bit CMake running on 64-bit Windows")
      if(NOT "${arch_hint}" STREQUAL "x64")
      # building 32-bit targets
        set(_PREFIX_ENV_VAR "ProgramFiles(x86)")
      endif()
    else()
      # 32-bit CMake
      #message("guess: this is a 32-bit CMake running on 64-bit Windows")
      if("${arch_hint}" STREQUAL "x64")
      # building 64-bit targets
        set(_PREFIX_ENV_VAR "ProgramW6432")
      endif()
    endif()
  endif()

  #if("${arch_hint}" STREQUAL "x64")
  #  message("guess: you are building a 64-bit app")
  #else()
  #  message("guess: you are building a 32-bit app")
  #endif()

  if(NOT "$ENV{${_PREFIX_ENV_VAR}}" STREQUAL "")
    file(TO_CMAKE_PATH "$ENV{${_PREFIX_ENV_VAR}}" _base)
  elseif(NOT "$ENV{SystemDrive}" STREQUAL "")
    set(_base "$ENV{SystemDrive}/Program Files")
  else()
    set(_base "C:/Program Files")
  endif()

  set(${var} "${_base}" PARENT_SCOPE)
endfunction()


# Set a variable to indicate whether the value of CMAKE_INSTALL_PREFIX
# was initialized by the block below.  This is useful for user
# projects to change the default prefix while still allowing the
# command line to override it.
if(NOT DEFINED CMAKE_INSTALL_PREFIX AND
   NOT DEFINED ENV{CMAKE_INSTALL_PREFIX})
  set(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT 1)
endif()

if(DEFINED ENV{CMAKE_INSTALL_PREFIX})
  set(CMAKE_INSTALL_PREFIX "$ENV{CMAKE_INSTALL_PREFIX}"
    CACHE PATH "Install path prefix, prepended onto install directories.")
else()
  # If CMAKE_INSTALL_PREFIX env variable is not set,
  # choose a default install prefix for this platform.
  if(CMAKE_HOST_UNIX)
    set(CMAKE_INSTALL_PREFIX "/usr/local"
      CACHE PATH "Install path prefix, prepended onto install directories.")
  else()
    GetDefaultWindowsPrefixBase(CMAKE_GENERIC_PROGRAM_FILES)
    set(CMAKE_INSTALL_PREFIX
      "${CMAKE_GENERIC_PROGRAM_FILES}/${PROJECT_NAME}"
      CACHE PATH "Install path prefix, prepended onto install directories.")
    set(CMAKE_GENERIC_PROGRAM_FILES)
  endif()
endif()

# Set a variable which will be used as component name in install() commands
# where no COMPONENT has been given:
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "Unspecified")

mark_as_advanced(
  CMAKE_SKIP_RPATH
  CMAKE_SKIP_INSTALL_RPATH
  CMAKE_VERBOSE_MAKEFILE
)
