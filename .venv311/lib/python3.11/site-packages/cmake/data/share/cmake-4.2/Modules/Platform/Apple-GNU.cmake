# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
include_guard()

macro(__apple_compiler_gnu lang)
  set(CMAKE_${lang}_VERBOSE_FLAG "-v -Wl,-v") # also tell linker to print verbose output
  # GNU does not have -shared on OS X
  set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS "-dynamiclib -Wl,-headerpad_max_install_names")
  set(CMAKE_SHARED_MODULE_CREATE_${lang}_FLAGS "-bundle -Wl,-headerpad_max_install_names")

  if(NOT CMAKE_${lang}_COMPILER_VERSION VERSION_LESS 4.3)
    set(CMAKE_${lang}_SYSTEM_FRAMEWORK_SEARCH_FLAG "-iframework ")
  endif()

  set(CMAKE_${lang}_LINK_LIBRARY_USING_FRAMEWORK "-framework <LIBRARY>")
  set(CMAKE_${lang}_LINK_LIBRARY_USING_FRAMEWORK_SUPPORTED TRUE)
  set(CMAKE_LINK_LIBRARY_FRAMEWORK_ATTRIBUTES LIBRARY_TYPE=STATIC,SHARED DEDUPLICATION=DEFAULT OVERRIDE=DEFAULT)

  set(CMAKE_${lang}_USING_LINKER_SYSTEM "")
  set(CMAKE_${lang}_USING_LINKER_APPLE_CLASSIC "LINKER:-ld_classic")
endmacro()

macro(cmake_gnu_set_sysroot_flag lang)
  if(NOT DEFINED CMAKE_${lang}_SYSROOT_FLAG)
    set(_doc "${lang} compiler has -isysroot")
    message(CHECK_START "Checking whether ${_doc}")
    execute_process(
      COMMAND ${CMAKE_${lang}_COMPILER} "-v" "--help"
      OUTPUT_VARIABLE _gcc_help
      ERROR_VARIABLE _gcc_help
      )
    if("${_gcc_help}" MATCHES "isysroot")
      message(CHECK_PASS "yes")
      set(CMAKE_${lang}_SYSROOT_FLAG "-isysroot")
    else()
      message(CHECK_FAIL "no")
      set(CMAKE_${lang}_SYSROOT_FLAG "")
    endif()
    set(CMAKE_${lang}_SYSROOT_FLAG_CODE "set(CMAKE_${lang}_SYSROOT_FLAG \"${CMAKE_${lang}_SYSROOT_FLAG}\")")
  endif()
endmacro()

macro(cmake_gnu_set_osx_deployment_target_flag lang)
  if(NOT DEFINED CMAKE_${lang}_OSX_DEPLOYMENT_TARGET_FLAG)
    set(_doc "${lang} compiler supports OSX deployment target flag")
    message(CHECK_START "Checking whether ${_doc}")
    execute_process(
      COMMAND ${CMAKE_${lang}_COMPILER} "-v" "--help"
      OUTPUT_VARIABLE _gcc_help
      ERROR_VARIABLE _gcc_help
      )
    if("${_gcc_help}" MATCHES "macosx-version-min")
      message(CHECK_PASS "yes")
      set(CMAKE_${lang}_OSX_DEPLOYMENT_TARGET_FLAG "-mmacosx-version-min=")
    else()
      message(CHECK_FAIL "no")
      set(CMAKE_${lang}_OSX_DEPLOYMENT_TARGET_FLAG "")
    endif()
    set(CMAKE_${lang}_OSX_DEPLOYMENT_TARGET_FLAG_CODE "set(CMAKE_${lang}_OSX_DEPLOYMENT_TARGET_FLAG \"${CMAKE_${lang}_OSX_DEPLOYMENT_TARGET_FLAG}\")")
  endif()
endmacro()
