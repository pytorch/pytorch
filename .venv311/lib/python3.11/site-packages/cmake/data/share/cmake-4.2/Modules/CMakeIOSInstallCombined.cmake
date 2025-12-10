# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# Function to print messages of this module
function(_ios_install_combined_message)
  message(STATUS "[iOS combined] " ${ARGN})
endfunction()

# Get build settings for the current target/config/SDK by running
# `xcodebuild -sdk ... -showBuildSettings` and parsing it's output
function(_ios_install_combined_get_build_setting sdk variable resultvar)
  if("${sdk}" STREQUAL "")
    message(FATAL_ERROR "`sdk` is empty")
  endif()

  if("${variable}" STREQUAL "")
    message(FATAL_ERROR "`variable` is empty")
  endif()

  if("${resultvar}" STREQUAL "")
    message(FATAL_ERROR "`resultvar` is empty")
  endif()

  set(
      cmd
      xcodebuild -showBuildSettings
      -sdk "${sdk}"
      -target "${CURRENT_TARGET}"
      -config "${CURRENT_CONFIG}"
  )

  execute_process(
      COMMAND ${cmd}
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
      RESULT_VARIABLE result
      OUTPUT_VARIABLE output
  )

  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Command failed (${result}): ${cmd}")
  endif()

  if(NOT output MATCHES " ${variable} = ([^\n]*)")
    if("${variable}" STREQUAL "VALID_ARCHS")
      # VALID_ARCHS may be unset by user for given SDK
      # (e.g. for build without simulator).
      set("${resultvar}" "" PARENT_SCOPE)
      return()
    else()
      message(FATAL_ERROR "${variable} not found.")
    endif()
  endif()

  set("${resultvar}" "${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction()

# Get architectures of given SDK (iphonesimulator/iphoneos)
function(_ios_install_combined_get_valid_archs sdk resultvar)
  if("${resultvar}" STREQUAL "")
    message(FATAL_ERROR "`resultvar` is empty")
  endif()

  _ios_install_combined_get_build_setting("${sdk}" "VALID_ARCHS" valid_archs)

  separate_arguments(valid_archs)
  list(REMOVE_ITEM valid_archs "") # remove empty elements
  list(REMOVE_DUPLICATES valid_archs)

  string(REPLACE ";" " " printable "${valid_archs}")
  _ios_install_combined_message("Architectures (${sdk}): ${printable}")

  set("${resultvar}" "${valid_archs}" PARENT_SCOPE)
endfunction()

# Make both arch lists a disjoint set by preferring the current SDK
# (starting with Xcode 12 arm64 is available as device and simulator arch on iOS)
function(_ios_install_combined_prune_common_archs corr_sdk corr_archs_var this_archs_var)
  list(REMOVE_ITEM ${corr_archs_var} ${${this_archs_var}})

  string(REPLACE ";" " " printable "${${corr_archs_var}}")
  _ios_install_combined_message("Architectures (${corr_sdk}) after pruning: ${printable}")

  set("${corr_archs_var}" "${${corr_archs_var}}" PARENT_SCOPE)
endfunction()

# Final target can contain more architectures that specified by SDK. This
# function will run 'lipo -info' and parse output. Result will be returned
# as a CMake list.
function(_ios_install_combined_get_real_archs filename resultvar)
  set(cmd "${_lipo_path}" -info "${filename}")
  execute_process(
      COMMAND ${cmd}
      RESULT_VARIABLE result
      OUTPUT_VARIABLE output
      ERROR_VARIABLE output
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE
  )
  if(NOT result EQUAL 0)
    message(
        FATAL_ERROR "Command failed (${result}): ${cmd}\n\nOutput:\n${output}"
    )
  endif()

  if(NOT output MATCHES "(Architectures in the fat file: [^\n]+ are|Non-fat file: [^\n]+ is architecture): ([^\n]*)")
    message(FATAL_ERROR "Could not detect architecture from: ${output}")
  endif()

  separate_arguments(CMAKE_MATCH_2)
  set(${resultvar} ${CMAKE_MATCH_2} PARENT_SCOPE)
endfunction()

# Run build command for the given SDK
function(_ios_install_combined_build sdk)
  if("${sdk}" STREQUAL "")
    message(FATAL_ERROR "`sdk` is empty")
  endif()

  _ios_install_combined_message("Build `${CURRENT_TARGET}` for `${sdk}`")

  execute_process(
      COMMAND
      "${CMAKE_COMMAND}"
      --build
      .
      --target "${CURRENT_TARGET}"
      --config ${CURRENT_CONFIG}
      --
      -sdk "${sdk}"
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
      RESULT_VARIABLE result
  )

  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Build failed")
  endif()
endfunction()

# Remove given architecture from file. This step needed only in rare cases
# when target was built in "unusual" way. Emit warning message.
function(_ios_install_combined_remove_arch lib arch)
  _ios_install_combined_message(
    "Warning! Unexpected architecture `${arch}` detected and will be removed "
    "from file `${lib}`")
  set(cmd "${_lipo_path}" -remove ${arch} -output ${lib} ${lib})
  execute_process(
      COMMAND ${cmd}
      RESULT_VARIABLE result
      OUTPUT_VARIABLE output
      ERROR_VARIABLE output
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE
  )
  if(NOT result EQUAL 0)
    message(
        FATAL_ERROR "Command failed (${result}): ${cmd}\n\nOutput:\n${output}"
    )
  endif()
endfunction()

# Check that 'lib' contains only 'archs' architectures (remove others).
function(_ios_install_combined_keep_archs lib archs)
  _ios_install_combined_get_real_archs("${lib}" real_archs)
  set(archs_to_remove ${real_archs})
  list(REMOVE_ITEM archs_to_remove ${archs})
  foreach(x ${archs_to_remove})
    _ios_install_combined_remove_arch("${lib}" "${x}")
  endforeach()
endfunction()

function(_ios_install_combined_detect_associated_sdk corr_sdk_var)
  if("${PLATFORM_NAME}" STREQUAL "")
    message(FATAL_ERROR "PLATFORM_NAME should not be empty")
  endif()

  set(all_platforms "$ENV{SUPPORTED_PLATFORMS}")
  if("${SUPPORTED_PLATFORMS}" STREQUAL "")
    _ios_install_combined_get_build_setting(
      ${PLATFORM_NAME} SUPPORTED_PLATFORMS all_platforms)
    if("${all_platforms}" STREQUAL "")
      message(FATAL_ERROR
        "SUPPORTED_PLATFORMS not set as an environment variable nor "
        "able to be determined from project")
    endif()
  endif()

  separate_arguments(all_platforms)
  if(NOT PLATFORM_NAME IN_LIST all_platforms)
    message(FATAL_ERROR "`${PLATFORM_NAME}` not found in `${all_platforms}`")
  endif()

  list(REMOVE_ITEM all_platforms "" "${PLATFORM_NAME}")
  list(LENGTH all_platforms all_platforms_length)
  if(NOT all_platforms_length EQUAL 1)
    message(FATAL_ERROR "Expected one element: ${all_platforms}")
  endif()

  set(${corr_sdk_var} "${all_platforms}" PARENT_SCOPE)
endfunction()

# Create combined binary for the given target.
#
# Preconditions:
#  * Target already installed at ${destination}
#    for the ${PLATFORM_NAME} platform
#
# This function will:
#  * Run build for the lacking platform, i.e. opposite to the ${PLATFORM_NAME}
#  * Fuse both libraries by running lipo
function(ios_install_combined target destination)
  if("${target}" STREQUAL "")
    message(FATAL_ERROR "`target` is empty")
  endif()

  if("${destination}" STREQUAL "")
    message(FATAL_ERROR "`destination` is empty")
  endif()

  if(NOT IS_ABSOLUTE "${destination}")
    message(FATAL_ERROR "`destination` is not absolute: ${destination}")
  endif()

  if(IS_DIRECTORY "${destination}" OR IS_SYMLINK "${destination}")
    message(FATAL_ERROR "`destination` is no regular file: ${destination}")
  endif()

  if("${CMAKE_BINARY_DIR}" STREQUAL "")
    message(FATAL_ERROR "`CMAKE_BINARY_DIR` is empty")
  endif()

  if(NOT IS_DIRECTORY "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "Is not a directory: ${CMAKE_BINARY_DIR}")
  endif()

  if("${CMAKE_INSTALL_CONFIG_NAME}" STREQUAL "")
    message(FATAL_ERROR "CMAKE_INSTALL_CONFIG_NAME is empty")
  endif()

  set(cmd xcrun -f lipo)

  # Do not merge OUTPUT_VARIABLE and ERROR_VARIABLE since latter may contain
  # some diagnostic information even for the successful run.
  execute_process(
      COMMAND ${cmd}
      RESULT_VARIABLE result
      OUTPUT_VARIABLE output
      ERROR_VARIABLE error_output
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE
  )
  if(NOT result EQUAL 0)
    message(
        FATAL_ERROR "Command failed (${result}): ${cmd}\n\nOutput:\n${output}\nOutput(error):\n${error_output}"
    )
  endif()
  set(_lipo_path ${output})
  list(LENGTH _lipo_path len)
  if(NOT len EQUAL 1)
    message(FATAL_ERROR "Unexpected xcrun output: ${_lipo_path}")
  endif()
  if(NOT EXISTS "${_lipo_path}")
    message(FATAL_ERROR "File not found: ${_lipo_path}")
  endif()

  set(CURRENT_CONFIG "${CMAKE_INSTALL_CONFIG_NAME}")
  set(CURRENT_TARGET "${target}")

  _ios_install_combined_message("Target: ${CURRENT_TARGET}")
  _ios_install_combined_message("Config: ${CURRENT_CONFIG}")
  _ios_install_combined_message("Destination: ${destination}")

  # Get SDKs
  _ios_install_combined_detect_associated_sdk(corr_sdk)

  # Get architectures of the target
  _ios_install_combined_get_valid_archs("${PLATFORM_NAME}" this_valid_archs)
  _ios_install_combined_get_valid_archs("${corr_sdk}" corr_valid_archs)
  _ios_install_combined_prune_common_archs("${corr_sdk}" corr_valid_archs this_valid_archs)

  # Return if there are no valid architectures for the SDK.
  # (note that library already installed)
  if("${corr_valid_archs}" STREQUAL "")
    _ios_install_combined_message(
        "No architectures detected for `${corr_sdk}` (skip)"
    )
    return()
  endif()

  # Trigger build of corresponding target
  _ios_install_combined_build("${corr_sdk}")

  # Get location of the library in build directory
  _ios_install_combined_get_build_setting(
    "${corr_sdk}" "CONFIGURATION_BUILD_DIR" corr_build_dir)
  _ios_install_combined_get_build_setting(
    "${corr_sdk}" "EXECUTABLE_PATH" corr_executable_path)
  set(corr "${corr_build_dir}/${corr_executable_path}")

  _ios_install_combined_keep_archs("${corr}" "${corr_valid_archs}")
  _ios_install_combined_keep_archs("${destination}" "${this_valid_archs}")

  _ios_install_combined_message("Current: ${destination}")
  _ios_install_combined_message("Corresponding: ${corr}")

  set(cmd "${_lipo_path}" -create ${corr} ${destination} -output ${destination})

  execute_process(
      COMMAND ${cmd}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      RESULT_VARIABLE result
  )

  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Command failed: ${cmd}")
  endif()

  _ios_install_combined_message("Install done: ${destination}")
endfunction()
