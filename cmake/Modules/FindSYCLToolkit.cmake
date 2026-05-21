# This will define the following variables:
# SYCL_FOUND               : True if the system has the SYCL library.
# SYCL_INCLUDE_DIR         : Include directories needed to use SYCL.
# SYCL_LIBRARY_DIR         : The path to the SYCL library.
# SYCL_LIBRARY             : SYCL library fullname.
# SYCL_COMPILER_VERSION    : SYCL compiler version.

include(FindPackageHandleStandardArgs)

# DPCLANG is an open source DPC++ compiler shipped with Linux distros
function(parse_dpclang_version major minor patch)
  # Execute the SYCL compiler with the --version flag to match the version string.
  execute_process(COMMAND ${SYCL_COMPILER} --version OUTPUT_VARIABLE SYCL_VERSION_STRING)
  string(REGEX REPLACE "DPC\\+\\+ compiler ([0-9]+\\.[0-9]+\\.[0-9]+) (.*)" "\\1"
               SYCL_VERSION_STRING_MATCH ${SYCL_VERSION_STRING})
  string(REPLACE "." ";" SYCL_VERSION_LIST ${SYCL_VERSION_STRING_MATCH})
  # Split the version number list into major, minor, and patch components.
  list(GET SYCL_VERSION_LIST 0 VERSION_MAJOR)
  list(GET SYCL_VERSION_LIST 1 VERSION_MINOR)
  list(GET SYCL_VERSION_LIST 2 VERSION_PATCH)
  set(${major} "${VERSION_MAJOR}" PARENT_SCOPE)
  set(${minor} "${VERSION_MINOR}" PARENT_SCOPE)
  set(${patch} "${VERSION_PATCH}" PARENT_SCOPE)
endfunction()

# ICX is a closed source DPC++ compiler shipped with oneAPI Toolkits
function(parse_icx_compiler_version version_number)
  # Execute the SYCL compiler with the --version flag to match the version string.
  execute_process(COMMAND ${SYCL_COMPILER} --version OUTPUT_VARIABLE SYCL_VERSION_STRING)
  string(REGEX REPLACE "Intel\\(R\\) (.*) Compiler ([0-9]+\\.[0-9]+\\.[0-9]+) (.*)" "\\2"
               SYCL_VERSION_STRING_MATCH ${SYCL_VERSION_STRING})
  string(REPLACE "." ";" SYCL_VERSION_LIST ${SYCL_VERSION_STRING_MATCH})
  # Split the version number list into major, minor, and patch components.
  list(GET SYCL_VERSION_LIST 0 VERSION_MAJOR)
  list(GET SYCL_VERSION_LIST 1 VERSION_MINOR)
  list(GET SYCL_VERSION_LIST 2 VERSION_PATCH)
  # Calculate the version number in the format XXXXYYZZ, using the formula (major * 10000 + minor * 100 + patch).
  math(EXPR VERSION_NUMBER_MATCH "${VERSION_MAJOR} * 10000 + ${VERSION_MINOR} * 100 + ${VERSION_PATCH}")
  set(${version_number} "${VERSION_NUMBER_MATCH}" PARENT_SCOPE)
endfunction()

function(map_dpclang_version_to_icx major minor patch out_version_number)
  # Map dpclang compiler version to oneAPI icx version
  math(EXPR icx_major "2019 + ${major}")
  # Calculate the version number in the format XXXXYYZZ, using the formula (major * 10000 + minor * 100 + patch).
  math(EXPR VERSION_NUMBER_MATCH "${icx_major} * 10000 + ${minor} * 100 + ${patch}")
  set(${out_version_number} "${VERSION_NUMBER_MATCH}" PARENT_SCOPE)
endfunction()

if("${XPU_SYCL_COMPILER}" MATCHES "dpclang")
  find_program(SYCL_COMPILER dpclang++)
  if(SYCL_COMPILER)
    parse_dpclang_version(
      SYCL_COMPILER_VERSION_MAJOR
      SYCL_COMPILER_VERSION_MINOR
      SYCL_COMPILER_VERSION_PATCH)
    map_dpclang_version_to_icx(
      ${SYCL_COMPILER_VERSION_MAJOR}
      ${SYCL_COMPILER_VERSION_MINOR}
      ${SYCL_COMPILER_VERSION_PATCH}
      SYCL_COMPILER_VERSION)
  endif()
  if(NOT SYCL_COMPILER_VERSION)
    set(SYCL_FOUND False)
    set(SYCL_REASON_FAILURE "Cannot parse sycl compiler version to get SYCL_COMPILER_VERSION!")
    set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
    return()
  endif()

  find_package(PkgConfig)
  if(PkgConfig_FOUND)
    pkg_check_modules(LIBSYCL_DPCPP IMPORTED_TARGET sycl-dpcpp-${SYCL_COMPILER_VERSION_MAJOR})
    if(LIBSYCL_DPCPP_FOUND)
      set(SYCL_INCLUDE_DIR ${LIBSYCL_DPCPP_INCLUDE_DIRS})
      set(SYCL_LIBRARY_DIR ${LIBSYCL_DPCPP_LIBRARY_DIRS})
      set(SYCL_LIBRARY ${LIBSYCL_DPCPP_LIBRARIES})
    endif()
  endif()
elseif("${XPU_SYCL_COMPILER}" MATCHES "icx")
  set(SYCL_ROOT "")
  if(DEFINED ENV{SYCL_ROOT})
    set(SYCL_ROOT $ENV{SYCL_ROOT})
  elseif(DEFINED ENV{CMPLR_ROOT})
    set(SYCL_ROOT $ENV{CMPLR_ROOT})
  else()
    # Use the default path to ensure proper linking with torch::xpurt when the user is working with libtorch.
    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
      set(SYCL_ROOT "/opt/intel/oneapi/compiler/latest")
    elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
      set(SYCL_ROOT "C:/Program Files (x86)/Intel/oneAPI/compiler/latest")
    endif()
    if(NOT EXISTS ${SYCL_ROOT})
      set(SYCL_ROOT "")
    endif()
  endif()

  string(COMPARE EQUAL "${SYCL_ROOT}" "" nosyclfound)
  if(nosyclfound)
    set(SYCL_FOUND False)
    set(SYCL_REASON_FAILURE "SYCL library not set!!")
    set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
    return()
  endif()

  # Find SYCL compiler executable.
  find_program(
    SYCL_COMPILER
    NAMES icx
    PATHS "${SYCL_ROOT}"
    PATH_SUFFIXES bin bin64
    NO_DEFAULT_PATH
    )

  if(SYCL_COMPILER)
    parse_icx_compiler_version(SYCL_COMPILER_VERSION)
  endif()

  if(NOT SYCL_COMPILER_VERSION)
    set(SYCL_FOUND False)
    set(SYCL_REASON_FAILURE "Cannot parse sycl compiler version to get SYCL_COMPILER_VERSION!")
    set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
    return()
  endif()

  # Find include path from binary.
  find_file(
    SYCL_INCLUDE_DIR
    NAMES include
    HINTS ${SYCL_ROOT}
    NO_DEFAULT_PATH
    )

  # Find include/sycl path from include path.
  find_file(
    SYCL_INCLUDE_SYCL_DIR
    NAMES sycl
    HINTS ${SYCL_ROOT}/include/
    NO_DEFAULT_PATH
    )

  # Due to the unrecognized compilation option `-fsycl` in other compiler.
  list(APPEND SYCL_INCLUDE_DIR ${SYCL_INCLUDE_SYCL_DIR})

 # Find library directory from binary.
  find_file(
    SYCL_LIBRARY_DIR
    NAMES lib lib64
    HINTS ${SYCL_ROOT}
    NO_DEFAULT_PATH
    )

  # Find SYCL library fullname.
  find_library(
    SYCL_LIBRARY
    NAMES "sycl"
    HINTS ${SYCL_LIBRARY_DIR}
    NO_DEFAULT_PATH
  )
endif()

if(NOT SYCL_LIBRARY)
  set(SYCL_FOUND False)
  set(SYCL_REASON_FAILURE "SYCL library is incomplete!!")
  set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
  return()
endif()

find_package_handle_standard_args(
  SYCL
  FOUND_VAR SYCL_FOUND
  REQUIRED_VARS SYCL_INCLUDE_DIR SYCL_LIBRARY_DIR SYCL_LIBRARY
  REASON_FAILURE_MESSAGE "${SYCL_REASON_FAILURE}"
  VERSION_VAR SYCL_COMPILER_VERSION
  )
