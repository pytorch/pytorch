# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

include(${CMAKE_ROOT}/Modules/CMakeDetermineCompiler.cmake)

# Local system-specific compiler preferences for this language.
include(Platform/${CMAKE_SYSTEM_NAME}-Determine-Swift OPTIONAL)
include(Platform/${CMAKE_SYSTEM_NAME}-Swift OPTIONAL)
if(NOT CMAKE_Swift_COMPILER_NAMES)
  set(CMAKE_Swift_COMPILER_NAMES swiftc)
endif()

if("${CMAKE_GENERATOR}" STREQUAL "Xcode")
  if(XCODE_VERSION VERSION_LESS 6.1)
    message(FATAL_ERROR "Swift language not supported by Xcode ${XCODE_VERSION}")
  endif()
  set(CMAKE_Swift_COMPILER_XCODE_TYPE sourcecode.swift)
  execute_process(COMMAND xcrun --find swiftc
    OUTPUT_VARIABLE _xcrun_out OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE _xcrun_err RESULT_VARIABLE _xcrun_result)
  if(_xcrun_result EQUAL 0 AND EXISTS "${_xcrun_out}")
    set(CMAKE_Swift_COMPILER "${_xcrun_out}")
  else()
    _cmake_find_compiler_path(Swift)
  endif()
elseif("${CMAKE_GENERATOR}" MATCHES "^Ninja")
  if(CMAKE_Swift_COMPILER)
    _cmake_find_compiler_path(Swift)
  else()
    set(CMAKE_Swift_COMPILER_INIT NOTFOUND)

    if(NOT $ENV{SWIFTC} STREQUAL "")
      get_filename_component(CMAKE_Swift_COMPILER_INIT $ENV{SWIFTC} PROGRAM
        PROGRAM_ARGS CMAKE_Swift_FLAGS_ENV_INIT)
      if(CMAKE_Swift_FLAGS_ENV_INIT)
        set(CMAKE_Swift_COMPILER_ARG1 "${CMAKE_Swift_FLAGS_ENV_INIT}" CACHE
          STRING "Arguments to the Swift compiler")
      endif()
      if(NOT EXISTS ${CMAKE_Swift_COMPILER_INIT})
        message(FATAL_ERROR "Could not find compiler set in environment variable SWIFTC\n$ENV{SWIFTC}.\n${CMAKE_Swift_COMPILER_INIT}")
      endif()
    endif()

    if(NOT CMAKE_Swift_COMPILER_INIT)
      set(CMAKE_Swift_COMPILER_LIST swiftc ${_CMAKE_TOOLCHAIN_PREFIX}swiftc)
    endif()

    _cmake_find_compiler(Swift)
  endif()
  mark_as_advanced(CMAKE_Swift_COMPILER)
else()
  message(FATAL_ERROR "Swift language not supported by \"${CMAKE_GENERATOR}\" generator")
endif()

# Build a small source file to identify the compiler.
if(NOT CMAKE_Swift_COMPILER_ID_RUN)
  set(CMAKE_Swift_COMPILER_ID_RUN 1)

  if("${CMAKE_GENERATOR}" STREQUAL "Xcode")
    list(APPEND CMAKE_Swift_COMPILER_ID_MATCH_VENDORS Apple)
    set(CMAKE_Swift_COMPILER_ID_MATCH_VENDOR_REGEX_Apple "com.apple.xcode.tools.swift.compiler")
  endif()

  # Try to identify the compiler.
  set(CMAKE_Swift_COMPILER_ID)
  include(${CMAKE_ROOT}/Modules/CMakeDetermineCompilerId.cmake)
  CMAKE_DETERMINE_COMPILER_ID(Swift "" CompilerId/main.swift)
endif()

# Check if we are using the old compiler driver.
if(CMAKE_GENERATOR STREQUAL "Xcode")
  # For Xcode, we can decide driver kind simply by Swift version.
  if(CMAKE_Swift_COMPILER_VERSION VERSION_GREATER_EQUAL 5.5)
    set(CMAKE_Swift_COMPILER_USE_OLD_DRIVER FALSE)
  else()
    set(CMAKE_Swift_COMPILER_USE_OLD_DRIVER TRUE)
  endif()
elseif(NOT DEFINED CMAKE_Swift_COMPILER_USE_OLD_DRIVER)
  # Dry-run a WMO build to identify the compiler driver.

  # Create a clean directory in which to run the test.
  set(CMAKE_Swift_COMPILER_DRIVER_TEST_DIR ${CMAKE_PLATFORM_INFO_DIR}/SwiftCompilerDriver)
  file(REMOVE_RECURSE "${CMAKE_Swift_COMPILER_DRIVER_TEST_DIR}")
  file(MAKE_DIRECTORY "${CMAKE_Swift_COMPILER_DRIVER_TEST_DIR}")

  # Create a Swift file and an arbitrary linker resource.
  file(WRITE ${CMAKE_Swift_COMPILER_DRIVER_TEST_DIR}/main.swift "print(\"Hello\")\n")
  file(WRITE ${CMAKE_Swift_COMPILER_DRIVER_TEST_DIR}/lib.in "\n")

  # Honor user-specified compiler flags.
  if(DEFINED CMAKE_Swift_FLAGS)
    separate_arguments(_CMAKE_Swift_COMPILER_FLAGS_LIST NATIVE_COMMAND "${CMAKE_Swift_FLAGS}")
  else()
    separate_arguments(_CMAKE_Swift_COMPILER_FLAGS_LIST NATIVE_COMMAND "${CMAKE_Swift_FLAGS_INIT}")
  endif()
  set(_CMAKE_Swift_COMPILER_CHECK_COMMAND "${CMAKE_Swift_COMPILER}" ${_CMAKE_Swift_COMPILER_FLAGS_LIST} -wmo main.swift lib.in "-###")
  unset(_CMAKE_Swift_COMPILER_FLAGS_LIST)

  # Execute in dry-run mode so no compilation will be actually performed.
  execute_process(COMMAND ${_CMAKE_Swift_COMPILER_CHECK_COMMAND}
    WORKING_DIRECTORY "${CMAKE_Swift_COMPILER_DRIVER_TEST_DIR}"
    OUTPUT_VARIABLE _CMAKE_Swift_COMPILER_CHECK_OUTPUT)

  # Check the first frontend execution.  It is on the first line of output.
  # The old driver treats all inputs as Swift sources while the new driver
  # can identify "lib.in" as a linker resource.
  if("${_CMAKE_Swift_COMPILER_CHECK_OUTPUT}" MATCHES "^[^\n]* lib\\.in")
    set(CMAKE_Swift_COMPILER_USE_OLD_DRIVER TRUE)
  else()
    set(CMAKE_Swift_COMPILER_USE_OLD_DRIVER FALSE)
  endif()

  # Record the check results in the configure log.
  list(TRANSFORM _CMAKE_Swift_COMPILER_CHECK_COMMAND PREPEND "\"")
  list(TRANSFORM _CMAKE_Swift_COMPILER_CHECK_COMMAND APPEND "\"")
  list(JOIN _CMAKE_Swift_COMPILER_CHECK_COMMAND " " _CMAKE_Swift_COMPILER_CHECK_COMMAND)
  string(REPLACE "\n" "\n  " _CMAKE_Swift_COMPILER_CHECK_OUTPUT "  ${_CMAKE_Swift_COMPILER_CHECK_OUTPUT}")
  message(CONFIGURE_LOG
    "Detected CMAKE_Swift_COMPILER_USE_OLD_DRIVER=\"${CMAKE_Swift_COMPILER_USE_OLD_DRIVER}\" from:\n"
    "  ${_CMAKE_Swift_COMPILER_CHECK_COMMAND}\n"
    "with output:\n"
    "${_CMAKE_Swift_COMPILER_CHECK_OUTPUT}"
    )

  unset(_CMAKE_Swift_COMPILER_CHECK_COMMAND)
  unset(_CMAKE_Swift_COMPILER_CHECK_OUTPUT)
endif()

if(CMAKE_Swift_COMPILER_VERSION VERSION_GREATER_EQUAL 5.2)
  set(target_info_command "${CMAKE_Swift_COMPILER}" -print-target-info)
  if(CMAKE_Swift_COMPILER_TARGET)
    list(APPEND target_info_command -target ${CMAKE_Swift_COMPILER_TARGET})
  endif()
  execute_process(
    COMMAND ${target_info_command}
    OUTPUT_VARIABLE swift_target_info)
  message(CONFIGURE_LOG "Swift target info:\n" "${swift_target_info}")
  string(JSON module_triple GET "${swift_target_info}" "target" "moduleTriple")
  set(CMAKE_Swift_MODULE_TRIPLE ${module_triple})
endif()

if (NOT _CMAKE_TOOLCHAIN_LOCATION)
  get_filename_component(_CMAKE_TOOLCHAIN_LOCATION "${CMAKE_Swift_COMPILER}" PATH)
endif ()

set(_CMAKE_PROCESSING_LANGUAGE "Swift")
include(CMakeFindBinUtils)
unset(_CMAKE_PROCESSING_LANGUAGE)

# configure variables set in this file for fast reload later on
configure_file(${CMAKE_ROOT}/Modules/CMakeSwiftCompiler.cmake.in
               ${CMAKE_PLATFORM_INFO_DIR}/CMakeSwiftCompiler.cmake @ONLY)

set(CMAKE_Swift_COMPILER_ENV_VAR "SWIFTC")
