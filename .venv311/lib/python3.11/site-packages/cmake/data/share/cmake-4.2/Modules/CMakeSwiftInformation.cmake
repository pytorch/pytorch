# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

if(UNIX)
  set(CMAKE_Swift_OUTPUT_EXTENSION .o)
else()
  set(CMAKE_Swift_OUTPUT_EXTENSION .obj)
endif()

# Load compiler-specific information.
if(CMAKE_Swift_COMPILER_ID)
  include(Compiler/${CMAKE_Swift_COMPILER_ID}-Swift OPTIONAL)

  if(CMAKE_SYSTEM_PROCESSOR)
    include(Platform/${CMAKE_EFFECTIVE_SYSTEM_NAME}-${CMAKE_Swift_COMPILER_ID}-Swift-${CMAKE_SYSTEM_PROCESSOR} OPTIONAL)
  endif()
  include(Platform/${CMAKE_EFFECTIVE_SYSTEM_NAME}-${CMAKE_Swift_COMPILER_ID}-Swift OPTIONAL)
endif()

set(CMAKE_INCLUDE_FLAG_Swift "-I ")

# FIXME: Move compiler- and platform-specific flags to the above-included modules.
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR CMAKE_SYSTEM_NAME STREQUAL "iOS"
    OR CMAKE_SYSTEM_NAME STREQUAL "tvOS" OR CMAKE_SYSTEM_NAME STREQUAL "visionOS"
    OR CMAKE_SYSTEM_NAME STREQUAL "watchOS")
  set(CMAKE_SHARED_LIBRARY_SONAME_Swift_FLAG "-Xlinker -install_name -Xlinker ")
elseif(NOT CMAKE_SYSTEM_NAME STREQUAL Windows)
  set(CMAKE_SHARED_LIBRARY_SONAME_Swift_FLAG "-Xlinker -soname -Xlinker ")
endif()
if(NOT CMAKE_SYSTEM_NAME STREQUAL Windows)
  set(CMAKE_EXECUTABLE_RUNTIME_Swift_FLAG "-Xlinker -rpath -Xlinker ")
  set(CMAKE_SHARED_LIBRARY_RUNTIME_Swift_FLAG "-Xlinker -rpath -Xlinker ")
  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR CMAKE_SYSTEM_NAME STREQUAL "iOS"
      OR CMAKE_SYSTEM_NAME STREQUAL "tvOS" OR CMAKE_SYSTEM_NAME STREQUAL "visionOS"
      OR CMAKE_SYSTEM_NAME STREQUAL "watchOS")
    set(CMAKE_EXECUTABLE_RUNTIME_Swift_FLAG_SEP "")
    set(CMAKE_SHARED_LIBRARY_RUNTIME_Swift_FLAG_SEP "")
  else()
    set(CMAKE_EXECUTABLE_RUNTIME_Swift_FLAG_SEP ":")
    set(CMAKE_SHARED_LIBRARY_RUNTIME_Swift_FLAG_SEP ":")
  endif()
endif()

set(CMAKE_Swift_COMPILE_OPTIONS_TARGET "-target ")
set(CMAKE_Swift_COMPILE_OPTIONS_EXTERNAL_TOOLCHAIN "-tools-directory ")
# NOTE(compnerd) the `-sdk` support is not yet ready in the compiler; when that
# is fully working, we should be able to enable this.
# set(CMAKE_Swift_COMPILE_OPTIONS_SYSROOT "-sdk ")
# NOTE(compnerd) do not setup `-frontend` as we use the compiler as the driver
# during the link phase and use that to drive the compilation
set(CMAKE_Swift_COMPILER_ARG1 "")
set(CMAKE_Swift_DEFINE_FLAG -D)
set(CMAKE_Swift_FRAMEWORK_SEARCH_FLAG "-F ")
set(CMAKE_Swift_LIBRARY_PATH_FLAG "-L ")
set(CMAKE_Swift_LIBRARY_PATH_TERMINATOR "")
set(CMAKE_Swift_LINK_LIBRARY_FLAG "-l")
set(CMAKE_Swift_LINKER_WRAPPER_FLAG "-Xlinker" " ")
set(CMAKE_Swift_RESPONSE_FILE_LINK_FLAG @)
set(CMAKE_Swift_RESPONSE_FILE_FLAG @)
set(CMAKE_Swift_COMPILE_OPTIONS_COLOR_DIAGNOSTICS -color-diagnostics)
set(CMAKE_Swift_COMPILE_OPTIONS_COLOR_DIAGNOSTICS_OFF -no-color-diagnostics)

set(CMAKE_Swift_LINKER_PREFERENCE 50)
set(CMAKE_Swift_LINKER_PREFERENCE_PROPAGATES 1)

# NOTE(compnerd) use the short form for convenience and ease of search.  They
# are treated equivalent to their long form names as well as custom Swift
# specific names.
set(CMAKE_Swift_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreaded -libc MT)
set(CMAKE_Swift_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDLL -libc MD)
set(CMAKE_Swift_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebug -libc MTd)
set(CMAKE_Swift_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebugDLL -libc MDd)

set(CMAKE_Swift_FLAGS_DEBUG_INIT "-Onone -g")
set(CMAKE_Swift_FLAGS_RELEASE_INIT "-O")
set(CMAKE_Swift_FLAGS_RELWITHDEBINFO_INIT "-O -g")
set(CMAKE_Swift_FLAGS_MINSIZEREL_INIT "-Osize")

if(CMAKE_GENERATOR STREQUAL "Xcode")
  string(APPEND CMAKE_Swift_FLAGS_DEBUG_INIT " ${CMAKE_Swift_FLAGS_DEBUG_LINKER_FLAGS}")
  string(APPEND CMAKE_Swift_FLAGS_RELWITHDEBINFO_INIT " ${CMAKE_Swift_FLAGS_RELWITHDEBINFO_LINKER_FLAGS}")
endif()

# Warns if unset and uses old policy.
# Old policy flag-smashes the wmo and incremental flags onto the compiler flags.
# New policy respects the Swift_COMPILATION_MODE target property to add
# incremental and wholemodule optimization flags as appropriate.
cmake_policy(GET CMP0157 __SWIFT_COMP_MODE_CMP0157)
if(__SWIFT_COMP_MODE_CMP0157 STREQUAL "NEW")
  set(CMAKE_Swift_COMPILATION_MODE_DEFAULT "incremental")
else()
  # Xcode has a separate Xcode project option (SWIFT_COMPILATION_MODE) used to set
  # whether compiling with whole-module optimizations or incrementally. Setting
  # these options here will have no effect when compiling with the built-in driver,
  # and will explode violently, leaving build products in the source directory, when
  # using the old swift driver. Don't append `-incremental` or `-wmo` to the
  # flags in the Xcode generator.
  if(NOT CMAKE_GENERATOR STREQUAL "Xcode")
    # Enable Whole Module Optimization by default unless the old
    # C++ driver is being used, which behaves differently under WMO.
    if(NOT CMAKE_Swift_COMPILER_USE_OLD_DRIVER)
      string(APPEND CMAKE_Swift_FLAGS_RELEASE_INIT " -wmo")
      string(APPEND CMAKE_Swift_FLAGS_RELWITHDEBINFO_INIT " -wmo")
      string(APPEND CMAKE_Swift_FLAGS_MINSIZEREL_INIT " -wmo")
    endif()
    string(APPEND CMAKE_Swift_FLAGS_DEBUG_INIT " -incremental")
  endif()
endif()
unset(__SWIFT_COMP_MODE_CMP0157)

cmake_initialize_per_config_variable(CMAKE_Swift_FLAGS "Swift Compiler Flags")

if(NOT CMAKE_Swift_NUM_THREADS MATCHES "^[0-9]+$")
  cmake_host_system_information(RESULT CMAKE_Swift_NUM_THREADS QUERY NUMBER_OF_LOGICAL_CORES)
endif()

# Swift split-compilation requires CMP0157 NEW policy
if(CMAKE_Swift_COMPILATION_MODE_DEFAULT)
  set(CMAKE_Swift_PARALLEL_FLAGS "-j ${CMAKE_Swift_NUM_THREADS} -num-threads ${CMAKE_Swift_NUM_THREADS}")
  if(NOT CMAKE_Swift_COMPILE_OBJECT)
    # Omit the object output. The output is controlled by the output-file-map
    # for normal builds. For wholemodule builds, CMake appends the appropriate
    # flags.
    set(CMAKE_Swift_COMPILE_OBJECT "<CMAKE_Swift_COMPILER> ${CMAKE_Swift_PARALLEL_FLAGS} -c <DEFINES> <FLAGS> <INCLUDES> <SOURCE>")
  endif()

  if(NOT CMAKE_Swift_CREATE_SHARED_LIBRARY)
    set(CMAKE_Swift_CREATE_SHARED_LIBRARY "<CMAKE_Swift_COMPILER> ${CMAKE_Swift_PARALLEL_FLAGS} -emit-library <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> ${CMAKE_Swift_IMPLIB_LINKER_FLAGS} <SONAME_FLAG> <TARGET_INSTALLNAME_DIR><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
  endif()

  if(NOT CMAKE_Swift_CREATE_SHARED_MODULE)
    set(CMAKE_Swift_CREATE_SHARED_MODULE "${CMAKE_Swift_CREATE_SHARED_LIBRARY}")
  endif()

  if(NOT CMAKE_Swift_LINK_EXECUTABLE)
    set(CMAKE_Swift_LINK_EXECUTABLE "<CMAKE_Swift_COMPILER> ${CMAKE_Swift_PARALLEL_FLAGS} -emit-executable -o <TARGET> <FLAGS> <OBJECTS> <LINK_FLAGS> <LINK_LIBRARIES>")
  endif()

  if(NOT CMAKE_Swift_CREATE_STATIC_LIBRARY)
    set(CMAKE_Swift_CREATE_STATIC_LIBRARY "<CMAKE_Swift_COMPILER> -emit-library -static -o <TARGET> <OBJECTS> <LINK_FLAGS>")
    set(CMAKE_Swift_ARCHIVE_CREATE "<CMAKE_AR> crs <TARGET> <OBJECTS>")
    set(CMAKE_Swift_ARCHIVE_FINISH "")
  endif()
  unset(CMAKE_Swift_PARALLEL_FLAGS)
else()
  # NOTE(compnerd) we do not have an object compile rule since we build the objects as part of the link step
  if(NOT CMAKE_Swift_COMPILE_OBJECT)
    set(CMAKE_Swift_COMPILE_OBJECT ":")
  endif()

  if(NOT CMAKE_Swift_CREATE_SHARED_LIBRARY)
    set(CMAKE_Swift_CREATE_SHARED_LIBRARY "<CMAKE_Swift_COMPILER> -j ${CMAKE_Swift_NUM_THREADS} -num-threads ${CMAKE_Swift_NUM_THREADS} -emit-library -o <TARGET> -module-name <SWIFT_MODULE_NAME> -module-link-name <SWIFT_LIBRARY_NAME> -emit-module -emit-module-path <SWIFT_MODULE> -emit-dependencies <DEFINES> <FLAGS> <INCLUDES> <SWIFT_SOURCES> <LINK_FLAGS> <SONAME_FLAG> <TARGET_INSTALLNAME_DIR><TARGET_SONAME> ${CMAKE_Swift_IMPLIB_LINKER_FLAGS} <LINK_LIBRARIES>")
  endif()

  if(NOT CMAKE_Swift_CREATE_SHARED_MODULE)
    set(CMAKE_Swift_CREATE_SHARED_MODULE "${CMAKE_Swift_CREATE_SHARED_LIBRARY}")
  endif()

  if(NOT CMAKE_Swift_LINK_EXECUTABLE)
    set(CMAKE_Swift_LINK_EXECUTABLE "<CMAKE_Swift_COMPILER> -j ${CMAKE_Swift_NUM_THREADS} -num-threads ${CMAKE_Swift_NUM_THREADS} -emit-executable -o <TARGET> -emit-dependencies <DEFINES> <FLAGS> <INCLUDES> <SWIFT_SOURCES> <LINK_FLAGS> <LINK_LIBRARIES>")
  endif()

  if(NOT CMAKE_Swift_LINK_EXECUTABLE_WITH_EXPORTS)
    set(CMAKE_Swift_LINK_EXECUTABLE_WITH_EXPORTS "${CMAKE_Swift_LINK_EXECUTABLE} -emit-module -emit-module-path <SWIFT_MODULE> ${CMAKE_Swift_IMPLIB_LINKER_FLAGS}")
  endif()

  if(NOT CMAKE_Swift_CREATE_STATIC_LIBRARY)
    set(CMAKE_Swift_CREATE_STATIC_LIBRARY "<CMAKE_Swift_COMPILER> -j ${CMAKE_Swift_NUM_THREADS} -num-threads ${CMAKE_Swift_NUM_THREADS} -emit-library -static -o <TARGET> -module-name <SWIFT_MODULE_NAME> -module-link-name <SWIFT_LIBRARY_NAME> -emit-module -emit-module-path <SWIFT_MODULE> -emit-dependencies <DEFINES> <FLAGS> <INCLUDES> <SWIFT_SOURCES> <LINK_FLAGS> <LINK_LIBRARIES>")

    set(CMAKE_Swift_ARCHIVE_CREATE "<CMAKE_AR> crs <TARGET> <OBJECTS>")
    set(CMAKE_Swift_ARCHIVE_FINISH "")
  endif()
endif()

set(CMAKE_Swift_USE_LINKER_INFORMATION TRUE)

set(CMAKE_Swift_INFORMATION_LOADED 1)
