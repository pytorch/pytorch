# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# In Android NDK r19 and above there is a single clang toolchain.
if(CMAKE_ANDROID_NDK_TOOLCHAIN_UNIFIED)
  if(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION AND NOT CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION STREQUAL "clang")
    message(FATAL_ERROR
      "Android: The CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION value '${CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION}' "
      "is not supported by this NDK.  It must be 'clang' or not set at all."
      )
  endif()
  message(STATUS "Android: Selected unified Clang toolchain")
  set(_ANDROID_TOOL_NDK_TOOLCHAIN_VERSION "clang")
  set(_ANDROID_TOOL_C_COMPILER "${CMAKE_ANDROID_NDK_TOOLCHAIN_UNIFIED}/bin/clang${_ANDROID_HOST_EXT}")
  set(_ANDROID_TOOL_C_TOOLCHAIN_MACHINE "${CMAKE_ANDROID_ARCH_TRIPLE}")
  set(_ANDROID_TOOL_C_TOOLCHAIN_VERSION "")
  set(_ANDROID_TOOL_C_COMPILER_EXTERNAL_TOOLCHAIN "")
  set(_ANDROID_TOOL_C_TOOLCHAIN_PREFIX "${CMAKE_ANDROID_NDK_TOOLCHAIN_UNIFIED}/bin/${CMAKE_ANDROID_ARCH_TRIPLE}-")
  set(_ANDROID_TOOL_C_TOOLCHAIN_SUFFIX "${_ANDROID_HOST_EXT}")
  set(_ANDROID_TOOL_CXX_COMPILER "${CMAKE_ANDROID_NDK_TOOLCHAIN_UNIFIED}/bin/clang++${_ANDROID_HOST_EXT}")
  set(_ANDROID_TOOL_CXX_TOOLCHAIN_MACHINE "${CMAKE_ANDROID_ARCH_TRIPLE}")
  set(_ANDROID_TOOL_CXX_TOOLCHAIN_VERSION "")
  set(_ANDROID_TOOL_CXX_COMPILER_EXTERNAL_TOOLCHAIN "")
  set(_ANDROID_TOOL_CXX_TOOLCHAIN_PREFIX "${CMAKE_ANDROID_NDK_TOOLCHAIN_UNIFIED}/bin/${CMAKE_ANDROID_ARCH_TRIPLE}-")
  set(_ANDROID_TOOL_CXX_TOOLCHAIN_SUFFIX "${_ANDROID_HOST_EXT}")
  set(_CMAKE_TOOLCHAIN_PREFIX "${CMAKE_ANDROID_ARCH_TRIPLE}-")
  return()
endif()

# In Android NDK releases there is build system toolchain selection logic in
# these files:
#
# * <ndk>/build/core/init.mk
# * <ndk>/build/core/setup-toolchain.mk
# * <ndk>/[build/core/]toolchains/<toolchain>/{config.mk,setup.mk}
#
# We parse information out of the ``config.mk`` and ``setup.mk`` files below.
#
# There is also a "toolchains" directory with the prebuilt toolchains themselves:
#
# * <triple-or-arch>-<gcc-version>/prebuilt/<host>/bin/<triple>-gcc(.exe)?
#   The gcc compiler to be invoked.
#
# * llvm*/prebuilt/<host>/bin/clang
#   The clang compiler to be invoked with flags:
#     -target <triple>
#     -gcc-toolchain <ndk>/toolchains/<triple-or-arch>-<gcc-version>

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

# Glob available toolchains in the NDK, restricted by any version request.
if(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION STREQUAL "clang")
  set(_ANDROID_TOOL_PATTERNS "*-clang" "*-clang[0-9].[0-9]")
elseif(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION)
  if(NOT CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION MATCHES "^(clang)?[0-9]\\.[0-9]$")
    message(FATAL_ERROR
      "Android: The CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION value '${CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION}' "
      "is not one of the allowed forms:\n"
      "  <major>.<minor>       = GCC of specified version\n"
      "  clang<major>.<minor>  = Clang of specified version\n"
      "  clang                 = Clang of most recent available version\n"
      )
  endif()
  set(_ANDROID_TOOL_PATTERNS "*-${CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION}")
else()
  # If we can find any gcc toolchains then use one by default.
  # Otherwise we look for clang toolchains (e.g. NDK r18+).
  file(GLOB _ANDROID_CONFIG_MKS_FOR_GCC
    "${CMAKE_ANDROID_NDK}/build/core/toolchains/*-[0-9].[0-9]/config.mk"
    "${CMAKE_ANDROID_NDK}/toolchains/*-[0-9].[0-9]/config.mk"
    )
  if(_ANDROID_CONFIG_MKS_FOR_GCC)
    set(_ANDROID_TOOL_PATTERNS "*-[0-9].[0-9]")
  else()
    set(_ANDROID_TOOL_PATTERNS "*-clang")
  endif()
  unset(_ANDROID_CONFIG_MKS_FOR_GCC)
endif()
set(_ANDROID_CONFIG_MK_PATTERNS)
foreach(base "build/core/toolchains" "toolchains")
  foreach(pattern IN LISTS _ANDROID_TOOL_PATTERNS)
    list(APPEND _ANDROID_CONFIG_MK_PATTERNS
      "${CMAKE_ANDROID_NDK}/${base}/${pattern}/config.mk"
      )
  endforeach()
endforeach()
unset(_ANDROID_TOOL_PATTERNS)
file(GLOB _ANDROID_CONFIG_MKS ${_ANDROID_CONFIG_MK_PATTERNS})
unset(_ANDROID_CONFIG_MK_PATTERNS)

# Find the newest toolchain version matching the ABI.
set(_ANDROID_TOOL_NAME "")
set(_ANDROID_TOOL_VERS 0)
set(_ANDROID_TOOL_VERS_NDK "")
set(_ANDROID_TOOL_SETUP_MK "")
foreach(config_mk IN LISTS _ANDROID_CONFIG_MKS)
  # Check that the toolchain matches the ABI.
  file(STRINGS "${config_mk}" _ANDROID_TOOL_ABIS REGEX "^TOOLCHAIN_ABIS :=.* ${CMAKE_ANDROID_ARCH_ABI}( |$)")
  if(NOT _ANDROID_TOOL_ABIS)
    continue()
  endif()
  unset(_ANDROID_TOOL_ABIS)

  # Check the version.
  if("${config_mk}" MATCHES [[/([^/]+-((clang)?([0-9]\.[0-9]|)))/config.mk$]])
    set(_ANDROID_CUR_NAME "${CMAKE_MATCH_1}")
    set(_ANDROID_CUR_VERS "${CMAKE_MATCH_4}")
    set(_ANDROID_CUR_VERS_NDK "${CMAKE_MATCH_2}")
    if(_ANDROID_TOOL_VERS STREQUAL "")
      # already the latest possible
    elseif(_ANDROID_CUR_VERS STREQUAL "" OR _ANDROID_CUR_VERS VERSION_GREATER _ANDROID_TOOL_VERS)
      set(_ANDROID_TOOL_NAME "${_ANDROID_CUR_NAME}")
      set(_ANDROID_TOOL_VERS "${_ANDROID_CUR_VERS}")
      set(_ANDROID_TOOL_VERS_NDK "${_ANDROID_CUR_VERS_NDK}")
      string(REPLACE "/config.mk" "/setup.mk" _ANDROID_TOOL_SETUP_MK "${config_mk}")
    endif()
    unset(_ANDROID_CUR_TOOL)
    unset(_ANDROID_CUR_VERS)
    unset(_ANDROID_CUR_VERS_NDK)
  endif()
endforeach()

# Verify that we have a suitable toolchain.
if(NOT _ANDROID_TOOL_NAME)
  if(_ANDROID_CONFIG_MKS)
    string(REPLACE ";" "\n  " _ANDROID_TOOLS_MSG "after considering:;${_ANDROID_CONFIG_MKS}")
  else()
    set(_ANDROID_TOOLS_MSG "")
  endif()
  if(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION)
    string(CONCAT _ANDROID_TOOLS_MSG
      "of the version specified by CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION:\n"
      "  ${CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION}\n"
      "${_ANDROID_TOOLS_MSG}")
  endif()
  message(FATAL_ERROR
    "Android: No toolchain for ABI '${CMAKE_ANDROID_ARCH_ABI}' found in the NDK:\n"
    "  ${CMAKE_ANDROID_NDK}\n"
    "${_ANDROID_TOOLS_MSG}"
    )
endif()
unset(_ANDROID_CONFIG_MKS)

# For clang toolchains we still need to find a gcc toolchain.
if(_ANDROID_TOOL_NAME MATCHES "-clang")
  set(_ANDROID_TOOL_CLANG_NAME "${_ANDROID_TOOL_NAME}")
  set(_ANDROID_TOOL_CLANG_VERS "${_ANDROID_TOOL_VERS}")
  set(_ANDROID_TOOL_NAME "")
  set(_ANDROID_TOOL_VERS "")
else()
  set(_ANDROID_TOOL_CLANG_NAME "")
  set(_ANDROID_TOOL_CLANG_VERS "")
endif()

# Parse the toolchain setup.mk file to extract information we need.
# Their content is not standardized across toolchains or NDK versions,
# so we match known cases.  Note that the parsing is stateful across
# lines because we need to substitute for some Make variable references.
if(CMAKE_ANDROID_NDK_TOOLCHAIN_DEBUG)
  message(STATUS "loading: ${_ANDROID_TOOL_SETUP_MK}")
endif()
file(STRINGS "${_ANDROID_TOOL_SETUP_MK}" _ANDROID_TOOL_SETUP REGEX "^(LLVM|TOOLCHAIN)_[A-Z_]+ +:= +.*$")
unset(_ANDROID_TOOL_SETUP_MK)
set(_ANDROID_TOOL_PREFIX "")
set(_ANDROID_TOOL_NAME_ONLY "")
set(_ANDROID_TOOL_LLVM_NAME "llvm")
set(_ANDROID_TOOL_LLVM_VERS "")
foreach(line IN LISTS _ANDROID_TOOL_SETUP)
  if(CMAKE_ANDROID_NDK_TOOLCHAIN_DEBUG)
    message(STATUS "setup.mk: ${line}")
  endif()

  if(line MATCHES [[^TOOLCHAIN_PREFIX +:= +.*/bin/([^$/ ]*) *$]])
    # We just matched the toolchain prefix with no Make variable references.
    set(_ANDROID_TOOL_PREFIX "${CMAKE_MATCH_1}")
  elseif(_ANDROID_TOOL_CLANG_NAME)
    # For clang toolchains we need to find more information.
    if(line MATCHES [[^TOOLCHAIN_VERSION +:= +([0-9.]+) *$]])
      # We just matched the gcc toolchain version number.  Save it for later.
      set(_ANDROID_TOOL_VERS "${CMAKE_MATCH_1}")
    elseif(line MATCHES [[^TOOLCHAIN_NAME +:= +(.*\$\(TOOLCHAIN_VERSION\)) *$]])
      # We just matched the gcc toolchain name with a version number placeholder, so substitute it.
      # The gcc toolchain version number will have already been extracted from a TOOLCHAIN_VERSION line.
      string(REPLACE "$(TOOLCHAIN_VERSION)" "${_ANDROID_TOOL_VERS}" _ANDROID_TOOL_NAME "${CMAKE_MATCH_1}")
    elseif(line MATCHES [[^TOOLCHAIN_NAME +:= +([^$/ ]+) *$]])
      # We just matched the gcc toolchain name without version number.  Save it for later.
      set(_ANDROID_TOOL_NAME_ONLY "${CMAKE_MATCH_1}")
    elseif(line MATCHES [[^TOOLCHAIN_PREFIX +:= +.*/bin/(\$\(TOOLCHAIN_NAME\)-) *$]])
      # We just matched the toolchain prefix with a name placeholder, so substitute it.
      # The gcc toolchain name will have already been extracted without version number from a TOOLCHAIN_NAME line.
      string(REPLACE "$(TOOLCHAIN_NAME)" "${_ANDROID_TOOL_NAME_ONLY}" _ANDROID_TOOL_PREFIX "${CMAKE_MATCH_1}")
    elseif(line MATCHES [[^LLVM_VERSION +:= +([0-9.]+)$]])
      # We just matched the llvm prebuilt binary toolchain version number.  Save it for later.
      set(_ANDROID_TOOL_LLVM_VERS "${CMAKE_MATCH_1}")
    elseif(line MATCHES [[^LLVM_NAME +:= +(llvm-\$\(LLVM_VERSION\)) *$]])
      # We just matched the llvm prebuilt binary toolchain directory name with a version number placeholder,
      # so substitute it. The llvm prebuilt binary toolchain version number will have already been extracted
      # from a LLVM_VERSION line.
      string(REPLACE "$(LLVM_VERSION)" "${_ANDROID_TOOL_LLVM_VERS}" _ANDROID_TOOL_LLVM_NAME "${CMAKE_MATCH_1}")
    elseif(line MATCHES [[^LLVM_TOOLCHAIN_PREBUILT_ROOT +:= +\$\(call get-toolchain-root.*,([^$ ]+)\) *$]])
      # We just matched the llvm prebuilt binary toolchain directory name.
      set(_ANDROID_TOOL_LLVM_NAME "${CMAKE_MATCH_1}")
    elseif(line MATCHES [[^TOOLCHAIN_ROOT +:= +\$\(call get-toolchain-root.*,(\$\(TOOLCHAIN_NAME\)-[0-9.]+)\) *$]])
      # We just matched a placeholder for the name followed by a version number.
      # The gcc toolchain name will have already been extracted without version number from a TOOLCHAIN_NAME line.
      # Substitute for the placeholder to get the full gcc toolchain name.
      string(REPLACE "$(TOOLCHAIN_NAME)" "${_ANDROID_TOOL_NAME_ONLY}" _ANDROID_TOOL_NAME "${CMAKE_MATCH_1}")
    elseif(line MATCHES [[^TOOLCHAIN_ROOT +:= +\$\(call get-toolchain-root.*,([^$ ]+)\) *$]])
      # We just matched the full gcc toolchain name without placeholder.
      set(_ANDROID_TOOL_NAME "${CMAKE_MATCH_1}")
    endif()
  endif()
endforeach()
unset(_ANDROID_TOOL_NAME_ONLY)
unset(_ANDROID_TOOL_LLVM_VERS)
unset(_ANDROID_TOOL_SETUP)

# Fall back to parsing the version and prefix from the tool name.
if(NOT _ANDROID_TOOL_VERS AND "${_ANDROID_TOOL_NAME}" MATCHES "-([0-9.]+)$")
  set(_ANDROID_TOOL_VERS "${CMAKE_MATCH_1}")
endif()
if(NOT _ANDROID_TOOL_PREFIX AND "${_ANDROID_TOOL_NAME}" MATCHES "^(.*-)[0-9.]+$")
  set(_ANDROID_TOOL_PREFIX "${CMAKE_MATCH_1}")
endif()

# Help CMakeFindBinUtils locate things.
set(_CMAKE_TOOLCHAIN_PREFIX "${_ANDROID_TOOL_PREFIX}")

set(_ANDROID_TOOL_NDK_TOOLCHAIN_VERSION "${_ANDROID_TOOL_VERS_NDK}")

# _ANDROID_TOOL_PREFIX should now match `gcc -dumpmachine`.
string(REGEX REPLACE "-$" "" _ANDROID_TOOL_C_TOOLCHAIN_MACHINE "${_ANDROID_TOOL_PREFIX}")

set(_ANDROID_TOOL_C_TOOLCHAIN_VERSION "${_ANDROID_TOOL_VERS}")
set(_ANDROID_TOOL_C_TOOLCHAIN_PREFIX "${CMAKE_ANDROID_NDK}/toolchains/${_ANDROID_TOOL_NAME}/prebuilt/${CMAKE_ANDROID_NDK_TOOLCHAIN_HOST_TAG}/bin/${_ANDROID_TOOL_PREFIX}")
set(_ANDROID_TOOL_C_TOOLCHAIN_SUFFIX "${_ANDROID_HOST_EXT}")

set(_ANDROID_TOOL_CXX_TOOLCHAIN_MACHINE "${_ANDROID_TOOL_C_TOOLCHAIN_MACHINE}")
set(_ANDROID_TOOL_CXX_TOOLCHAIN_VERSION "${_ANDROID_TOOL_C_TOOLCHAIN_VERSION}")
set(_ANDROID_TOOL_CXX_TOOLCHAIN_PREFIX "${_ANDROID_TOOL_C_TOOLCHAIN_PREFIX}")
set(_ANDROID_TOOL_CXX_TOOLCHAIN_SUFFIX "${_ANDROID_TOOL_C_TOOLCHAIN_SUFFIX}")

if(_ANDROID_TOOL_CLANG_NAME)
  message(STATUS "Android: Selected Clang toolchain '${_ANDROID_TOOL_CLANG_NAME}' with GCC toolchain '${_ANDROID_TOOL_NAME}'")
  set(_ANDROID_TOOL_C_COMPILER "${CMAKE_ANDROID_NDK}/toolchains/${_ANDROID_TOOL_LLVM_NAME}/prebuilt/${CMAKE_ANDROID_NDK_TOOLCHAIN_HOST_TAG}/bin/clang${_ANDROID_HOST_EXT}")
  set(_ANDROID_TOOL_C_COMPILER_EXTERNAL_TOOLCHAIN ${CMAKE_ANDROID_NDK}/toolchains/${_ANDROID_TOOL_NAME}/prebuilt/${CMAKE_ANDROID_NDK_TOOLCHAIN_HOST_TAG})
  set(_ANDROID_TOOL_CXX_COMPILER "${CMAKE_ANDROID_NDK}/toolchains/${_ANDROID_TOOL_LLVM_NAME}/prebuilt/${CMAKE_ANDROID_NDK_TOOLCHAIN_HOST_TAG}/bin/clang++${_ANDROID_HOST_EXT}")
  set(_ANDROID_TOOL_CXX_COMPILER_EXTERNAL_TOOLCHAIN "${_ANDROID_TOOL_C_COMPILER_EXTERNAL_TOOLCHAIN}")
else()
  message(STATUS "Android: Selected GCC toolchain '${_ANDROID_TOOL_NAME}'")
  set(_ANDROID_TOOL_C_COMPILER "${_ANDROID_TOOL_C_TOOLCHAIN_PREFIX}gcc${_ANDROID_TOOL_C_TOOLCHAIN_SUFFIX}")
  set(_ANDROID_TOOL_C_COMPILER_EXTERNAL_TOOLCHAIN "")
  set(_ANDROID_TOOL_CXX_COMPILER "${_ANDROID_TOOL_CXX_TOOLCHAIN_PREFIX}g++${_ANDROID_TOOL_CXX_TOOLCHAIN_SUFFIX}")
  set(_ANDROID_TOOL_CXX_COMPILER_EXTERNAL_TOOLCHAIN "")
endif()

if(CMAKE_ANDROID_NDK_TOOLCHAIN_DEBUG)
  message(STATUS "_ANDROID_TOOL_NAME=${_ANDROID_TOOL_NAME}")
  message(STATUS "_ANDROID_TOOL_VERS=${_ANDROID_TOOL_VERS}")
  message(STATUS "_ANDROID_TOOL_VERS_NDK=${_ANDROID_TOOL_VERS_NDK}")
  message(STATUS "_ANDROID_TOOL_PREFIX=${_ANDROID_TOOL_PREFIX}")
  message(STATUS "_ANDROID_TOOL_CLANG_NAME=${_ANDROID_TOOL_CLANG_NAME}")
  message(STATUS "_ANDROID_TOOL_CLANG_VERS=${_ANDROID_TOOL_CLANG_VERS}")
  message(STATUS "_ANDROID_TOOL_LLVM_NAME=${_ANDROID_TOOL_LLVM_NAME}")
endif()

unset(_ANDROID_TOOL_NAME)
unset(_ANDROID_TOOL_VERS)
unset(_ANDROID_TOOL_VERS_NDK)
unset(_ANDROID_TOOL_PREFIX)
unset(_ANDROID_TOOL_CLANG_NAME)
unset(_ANDROID_TOOL_CLANG_VERS)
unset(_ANDROID_TOOL_LLVM_NAME)

cmake_policy(POP)
