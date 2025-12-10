set(APPLE 1)
set(UNIX 1)

# Ask xcode-select where to find /Developer or fall back to ancient location.
execute_process(COMMAND xcode-select -print-path
  OUTPUT_VARIABLE _stdout
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_VARIABLE _stderr
  RESULT_VARIABLE _failed)
if(NOT _failed AND IS_DIRECTORY ${_stdout})
  set(OSX_DEVELOPER_ROOT ${_stdout})
elseif(IS_DIRECTORY "/Developer")
  set(OSX_DEVELOPER_ROOT "/Developer")
else()
  set(OSX_DEVELOPER_ROOT "")
endif()

# Save CMAKE_OSX_ARCHITECTURES from the environment.
set(CMAKE_OSX_ARCHITECTURES "$ENV{CMAKE_OSX_ARCHITECTURES}" CACHE STRING
  "Build architectures for OSX")

if(NOT CMAKE_CROSSCOMPILING AND
   CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND
   CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(arm64|x86_64)$")
  execute_process(COMMAND sysctl -q hw.optional.arm64
    OUTPUT_VARIABLE _sysctl_stdout
    ERROR_VARIABLE _sysctl_stderr
    RESULT_VARIABLE _sysctl_result
    )
  # When building on an Apple Silicon host, we need to explicitly specify
  # the architecture to the toolchain since it will otherwise guess the
  # architecture based on that of the build system tool.
  # Set an *internal variable* to tell the generators to do this.
  if(_sysctl_result EQUAL 0 AND _sysctl_stdout MATCHES "hw.optional.arm64: 1")
    set(_CMAKE_APPLE_ARCHS_DEFAULT "${CMAKE_HOST_SYSTEM_PROCESSOR}")
  endif()
  unset(_sysctl_result)
  unset(_sysctl_stderr)
  unset(_sysctl_stdout)
endif()

# macOS, iOS, tvOS, visionOS, and watchOS should lookup compilers from
# Platform/Apple-${CMAKE_CXX_COMPILER_ID}-<LANG>
set(CMAKE_EFFECTIVE_SYSTEM_NAME "Apple")

#----------------------------------------------------------------------------
# CMAKE_OSX_SYSROOT

if(CMAKE_OSX_SYSROOT)
  # Use the existing value without further computation to choose a default.
  set(_CMAKE_OSX_SYSROOT_DEFAULT "${CMAKE_OSX_SYSROOT}")
elseif(NOT "x$ENV{SDKROOT}" STREQUAL "x" AND
        (NOT "x$ENV{SDKROOT}" MATCHES "/" OR IS_DIRECTORY "$ENV{SDKROOT}"))
  # Use the value of SDKROOT from the environment.
  set(_CMAKE_OSX_SYSROOT_DEFAULT "$ENV{SDKROOT}")
elseif(CMAKE_SYSTEM_NAME STREQUAL iOS)
  set(_CMAKE_OSX_SYSROOT_DEFAULT "iphoneos")
elseif(CMAKE_SYSTEM_NAME STREQUAL tvOS)
  set(_CMAKE_OSX_SYSROOT_DEFAULT "appletvos")
elseif(CMAKE_SYSTEM_NAME STREQUAL visionOS)
  set(_CMAKE_OSX_SYSROOT_DEFAULT "xros")
elseif(CMAKE_SYSTEM_NAME STREQUAL watchOS)
  set(_CMAKE_OSX_SYSROOT_DEFAULT "watchos")
else()
  set(_CMAKE_OSX_SYSROOT_DEFAULT "")
endif()

# Set cache variable - end user may change this during ccmake or cmake-gui configure.
# Choose the type based on the current value.
set(_CMAKE_OSX_SYSROOT_TYPE STRING)
foreach(_v CMAKE_OSX_SYSROOT _CMAKE_OSX_SYSROOT_DEFAULT)
  if("x${${_v}}" MATCHES "/")
    set(_CMAKE_OSX_SYSROOT_TYPE PATH)
    break()
  endif()
endforeach()
set(CMAKE_OSX_SYSROOT "${_CMAKE_OSX_SYSROOT_DEFAULT}" CACHE ${_CMAKE_OSX_SYSROOT_TYPE}
  "The product will be built against the headers and libraries located inside the indicated SDK.")

# Resolves the SDK name into a path
function(_apple_resolve_sdk_path sdk_name ret)
  execute_process(
    COMMAND xcrun -sdk ${sdk_name} --show-sdk-path
    OUTPUT_VARIABLE _stdout
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE _stderr
    RESULT_VARIABLE _failed
  )
  set(${ret} "${_stdout}" PARENT_SCOPE)
endfunction()

function(_apple_resolve_supported_archs_for_sdk_from_system_lib sdk_path ret ret_failed)
  # Detect the supported SDK architectures by inspecting the main libSystem library.
  set(common_lib_prefix "${sdk_path}/usr/lib/libSystem")
  set(system_lib_dylib_path "${common_lib_prefix}.dylib")
  set(system_lib_tbd_path "${common_lib_prefix}.tbd")

  # Newer SDKs ship text based dylib stub files which contain the architectures supported by the
  # library in text form.
  if(EXISTS "${system_lib_tbd_path}")
    cmake_policy(PUSH)
    cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>
    file(STRINGS "${system_lib_tbd_path}" tbd_lines REGEX "^(archs|targets): +\\[.+\\]")
    cmake_policy(POP)
    if(NOT tbd_lines)
      set(${ret_failed} TRUE PARENT_SCOPE)
      return()
    endif()

    # The tbd architectures line looks like the following:
    #   archs:           [ armv7, armv7s, arm64, arm64e ]
    # or for version 4 TBD files:
    #   targets:         [ armv7-ios, armv7s-ios, arm64-ios, arm64e-ios ]
    list(GET tbd_lines 0 first_arch_line)
    string(REGEX REPLACE
           "(archs|targets): +\\[ (.+) \\]" "\\2" arches_comma_separated "${first_arch_line}")
    string(STRIP "${arches_comma_separated}" arches_comma_separated)
    string(REPLACE "," ";" arch_list "${arches_comma_separated}")
    string(REPLACE " " "" arch_list "${arch_list}")

    # Remove -platform suffix from target (version 4 only)
    string(REGEX REPLACE "-[a-z-]+" "" arch_list "${arch_list}")

    if(NOT arch_list)
      set(${ret_failed} TRUE PARENT_SCOPE)
      return()
    endif()
    set(${ret} "${arch_list}" PARENT_SCOPE)
  elseif(EXISTS "${system_lib_dylib_path}")
    # Old SDKs (Xcode < 7) ship dylib files, use lipo to inspect the supported architectures.
    # Can't use -archs because the option is not available in older Xcode versions.
    execute_process(
      COMMAND lipo -info ${system_lib_dylib_path}
      OUTPUT_VARIABLE lipo_output
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_VARIABLE _stderr
      RESULT_VARIABLE _failed
    )
    if(_failed OR NOT lipo_output OR NOT lipo_output MATCHES "(Non-fat file:|Architectures in the fat file:)")
      set(${ret_failed} TRUE PARENT_SCOPE)
      return()
    endif()

    # The lipo output looks like the following:
    # Non-fat file: <path> is architecture: i386
    # Architectures in the fat file: <path> are: i386 x86_64
    string(REGEX REPLACE
           "^(.+)is architecture:(.+)" "\\2" arches_space_separated "${lipo_output}")
    string(REGEX REPLACE
            "^(.+)are:(.+)" "\\2" arches_space_separated "${arches_space_separated}")

    # Need to clean up the arches, with Xcode 4.6.3 the output of lipo -info contains some
    # additional info, e.g.
    # Architectures in the fat file: <path> are: armv7 (cputype (12) cpusubtype (11))
    string(REGEX REPLACE
            "\\(.+\\)" "" arches_space_separated "${arches_space_separated}")

    # The output is space separated.
    string(STRIP "${arches_space_separated}" arches_space_separated)
    string(REPLACE " " ";" arch_list "${arches_space_separated}")

    if(NOT arch_list)
      set(${ret_failed} TRUE PARENT_SCOPE)
      return()
    endif()
    set(${ret} "${arch_list}" PARENT_SCOPE)
  else()
    # This shouldn't happen, but keep it for safety.
    message(WARNING "No way to find architectures for given sdk_path '${sdk_path}'")
    set(${ret_failed} TRUE PARENT_SCOPE)
  endif()
endfunction()

# Handle multi-arch sysroots. Do this before CMAKE_OSX_SYSROOT is
# transformed into a path, so that we know the sysroot name.
function(_apple_resolve_multi_arch_sysroots)
  if(DEFINED CMAKE_APPLE_ARCH_SYSROOTS)
    return() # Already cached
  endif()

  list(LENGTH CMAKE_OSX_ARCHITECTURES _num_archs)
  if(NOT (_num_archs GREATER 1))
    return() # Only apply to multi-arch
  endif()

  if(NOT CMAKE_OSX_SYSROOT OR CMAKE_OSX_SYSROOT MATCHES "(^|/)[Mm][Aa][Cc][Oo][Ss][Xx]")
    # macOS doesn't have a simulator sdk / sysroot, so there is no need to handle per-sdk arches.
    return()
  endif()

  if(IS_DIRECTORY "${CMAKE_OSX_SYSROOT}")
    if(NOT CMAKE_OSX_SYSROOT STREQUAL _CMAKE_OSX_SYSROOT_DEFAULT)
      message(WARNING "Can not resolve multi-arch sysroots with CMAKE_OSX_SYSROOT set to path (${CMAKE_OSX_SYSROOT})")
    endif()
    return()
  endif()

  string(REPLACE "os" "simulator" _simulator_sdk "${CMAKE_OSX_SYSROOT}")
  set(_sdks "${CMAKE_OSX_SYSROOT};${_simulator_sdk}")
  foreach(sdk ${_sdks})
    _apple_resolve_sdk_path(${sdk} _sdk_path)
    if(NOT IS_DIRECTORY "${_sdk_path}")
      message(WARNING "Failed to resolve SDK path for '${sdk}'")
      continue()
    endif()

    _apple_resolve_supported_archs_for_sdk_from_system_lib(${_sdk_path} _sdk_archs _failed)

    if(_failed)
      # Failure to extract supported architectures for an SDK means that the installed SDK is old
      # and does not provide such information (SDKs that come with Xcode >= 10.x started providing
      # the information). In such a case, return early, and handle multi-arch builds the old way
      # (no per-sdk arches).
      return()
    endif()

    set(_sdk_archs_${sdk} ${_sdk_archs})
    set(_sdk_path_${sdk} ${_sdk_path})
  endforeach()

  foreach(arch ${CMAKE_OSX_ARCHITECTURES})
    set(_arch_sysroot "")
    foreach(sdk ${_sdks})
      list(FIND _sdk_archs_${sdk} ${arch} arch_index)
      if(NOT arch_index EQUAL -1)
        set(_arch_sysroot ${_sdk_path_${sdk}})
        break()
      endif()
    endforeach()
    if(_arch_sysroot)
      list(APPEND _arch_sysroots ${_arch_sysroot})
    else()
      message(WARNING "No SDK found for architecture '${arch}'")
      list(APPEND _arch_sysroots "${arch}-SDK-NOTFOUND")
    endif()
  endforeach()

  set(CMAKE_APPLE_ARCH_SYSROOTS "${_arch_sysroots}" CACHE INTERNAL
    "Architecture dependent sysroots, one per CMAKE_OSX_ARCHITECTURES")
endfunction()

_apple_resolve_multi_arch_sysroots()

if(CMAKE_OSX_SYSROOT MATCHES "/")
  # This is a path to a SDK.  Make sure it exists.
  if(NOT IS_DIRECTORY "${CMAKE_OSX_SYSROOT}")
    message(WARNING "Ignoring CMAKE_OSX_SYSROOT value:\n ${CMAKE_OSX_SYSROOT}\n"
      "because the directory does not exist.")
    set(CMAKE_OSX_SYSROOT "")
  endif()
  set(_CMAKE_OSX_SYSROOT_PATH "${CMAKE_OSX_SYSROOT}")
elseif(CMAKE_OSX_SYSROOT)
  # This is the name of a SDK.  Transform it to a path.
  _apple_resolve_sdk_path("${CMAKE_OSX_SYSROOT}" _CMAKE_OSX_SYSROOT_PATH)
  # Use the path for non-Xcode generators.
  if(IS_DIRECTORY "${_CMAKE_OSX_SYSROOT_PATH}" AND NOT CMAKE_GENERATOR MATCHES "Xcode")
    set(CMAKE_OSX_SYSROOT "${_CMAKE_OSX_SYSROOT_PATH}")
  endif()
endif()
if(NOT CMAKE_OSX_SYSROOT)
  # Without any explicit SDK we rely on the toolchain default,
  # which we assume to be what wrappers like /usr/bin/cc use.
  if(CMAKE_GENERATOR STREQUAL "Xcode")
    set(_sdk_macosx --sdk macosx)
  else()
    set(_sdk_macosx)
  endif()
  execute_process(
    COMMAND xcrun ${_sdk_macosx} --show-sdk-path
    OUTPUT_VARIABLE _CMAKE_OSX_SYSROOT_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE _stderr
    RESULT_VARIABLE _result
  )
  unset(_sdk_macosx)

  list(APPEND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES_EXCLUDE
    # Without -isysroot, some compiler drivers implicitly pass -L/usr/local/lib
    # to the linker.  Since the macOS dynamic loader does not search it by
    # default, it is not a fully-implemented implicit link directory.
    /usr/local/lib
  )
endif()

#----------------------------------------------------------------------------
# CMAKE_OSX_DEPLOYMENT_TARGET

if(NOT CMAKE_CROSSCOMPILING)
  execute_process(COMMAND sw_vers -productVersion
    OUTPUT_VARIABLE _CMAKE_HOST_OSX_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
  set(_CMAKE_OSX_DEPLOYMENT_TARGET_DEFAULT "$ENV{MACOSX_DEPLOYMENT_TARGET}")

  # Xcode chooses a default macOS deployment target based on the macOS SDK
  # version, which may be too new for binaries to run on the host.
  if(NOT _CMAKE_OSX_DEPLOYMENT_TARGET_DEFAULT
      AND CMAKE_GENERATOR STREQUAL "Xcode" AND NOT CMAKE_CROSSCOMPILING
      AND _CMAKE_HOST_OSX_VERSION MATCHES "^([0-9]+\\.[0-9]+)")
    set(_macos_version "${CMAKE_MATCH_1}")
    if(CMAKE_OSX_SYSROOT)
      set(_sdk_macosx --sdk ${CMAKE_OSX_SYSROOT})
    else()
      set(_sdk_macosx)
    endif()
    execute_process(
      COMMAND xcrun ${_sdk_macosx} --show-sdk-version
      OUTPUT_VARIABLE _sdk_version OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_VARIABLE _sdk_version_error
      RESULT_VARIABLE _sdk_version_result
    )
    if(_sdk_version_result EQUAL 0 AND _sdk_version
        AND "${_macos_version}" VERSION_LESS "${_sdk_version}")
      set(_CMAKE_OSX_DEPLOYMENT_TARGET_DEFAULT "${_macos_version}")
    endif()
    unset(_sdk_macosx)
    unset(_sdk_version_result)
    unset(_sdk_version_error)
    unset(_sdk_version)
    unset(_macos_version)
  endif()

  set(CMAKE_OSX_DEPLOYMENT_TARGET "${_CMAKE_OSX_DEPLOYMENT_TARGET_DEFAULT}" CACHE STRING
    "Minimum OS X version to target for deployment (at runtime); newer APIs weak linked. Set to empty string for default value.")
  unset(_CMAKE_OSX_DEPLOYMENT_TARGET_DEFAULT)
endif()
