# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# Setup variables used for Green Hills MULTI generator
# -- Allow users to override these values.

if(CMAKE_GENERATOR MATCHES "Green Hills MULTI")

  # Set the project primaryTarget value
  # If not set then primaryTarget will be determined by the generator
  if((NOT DEFINED GHS_PRIMARY_TARGET) OR (DEFINED CACHE{GHS_PRIMARY_TARGET}))
    set(GHS_PRIMARY_TARGET "IGNORE" CACHE STRING "GHS MULTI primaryTarget")
    mark_as_advanced(GHS_PRIMARY_TARGET)
  endif()

  # Setup MULTI toolset selection variables
  if((NOT DEFINED GHS_TOOLSET_ROOT) OR (DEFINED CACHE{GHS_TOOLSET_ROOT}))
    if(CMAKE_HOST_UNIX)
      set(_ts_root "/usr/ghs")
    else()
      set(_ts_root "C:/ghs")
    endif()
    set(GHS_TOOLSET_ROOT "${_ts_root}" CACHE PATH "GHS platform toolset root directory")
    mark_as_advanced(GHS_TOOLSET_ROOT)
    unset(_ts_root)
  endif()

  # Setup MULTI project variables
  if((NOT DEFINED GHS_CUSTOMIZATION) OR (DEFINED CACHE{GHS_CUSTOMIZATION}))
    set(GHS_CUSTOMIZATION "" CACHE FILEPATH "optional GHS customization")
    mark_as_advanced(GHS_CUSTOMIZATION)
  endif()

  if((NOT DEFINED GHS_GPJ_MACROS) OR (DEFINED CACHE{GHS_GPJ_MACROS}))
    set(GHS_GPJ_MACROS "" CACHE STRING "optional GHS macros generated in the .gpjs for legacy reasons")
    mark_as_advanced(GHS_GPJ_MACROS)
  endif()

endif()

# If project primaryTarget not set then set target platform name.
# -- May be used by the generator when determining the primaryTarget.
if(NOT GHS_PRIMARY_TARGET)
  if((NOT DEFINED GHS_TARGET_PLATFORM) OR (DEFINED CACHE{GHS_TARGET_PLATFORM}))
    set(GHS_TARGET_PLATFORM "integrity" CACHE STRING "GHS MULTI target platform")
    mark_as_advanced(GHS_TARGET_PLATFORM)
  endif()
endif()

# Settings for OS selection
if((NOT DEFINED GHS_OS_ROOT) OR (DEFINED CACHE{GHS_OS_ROOT}))
  if(CMAKE_HOST_UNIX)
    set(_os_root "/usr/ghs")
  else()
    set(_os_root "C:/ghs")
  endif()
  set(GHS_OS_ROOT "${_os_root}" CACHE PATH "GHS platform OS search root directory")
  unset(_os_root)
  mark_as_advanced(GHS_OS_ROOT)
endif()

# Search for GHS_OS_DIR if not set by user and is known to be required
if(GHS_PRIMARY_TARGET MATCHES "integrity" OR GHS_TARGET_PLATFORM MATCHES "integrity")
  # Needed - Use a value that will make it apparent RTOS selection failed
  set(_ghs_os_dir "GHS_OS_DIR-NOT-SPECIFIED")
else()
  # Not needed for this target
  set(_ghs_os_dir "IGNORE")
endif()

if(_ghs_os_dir AND NOT DEFINED GHS_OS_DIR)
  if(EXISTS ${GHS_OS_ROOT})

    # Get all directories in root directory
    file(GLOB GHS_CANDIDATE_OS_DIRS
      LIST_DIRECTORIES true RELATIVE ${GHS_OS_ROOT} ${GHS_OS_ROOT}/*)
    file(GLOB GHS_CANDIDATE_OS_FILES
      LIST_DIRECTORIES false RELATIVE ${GHS_OS_ROOT} ${GHS_OS_ROOT}/*)
    if(GHS_CANDIDATE_OS_FILES)
      list(REMOVE_ITEM GHS_CANDIDATE_OS_DIRS ${GHS_CANDIDATE_OS_FILES})
    endif ()

    # Filter based on platform name
    if(GHS_PRIMARY_TARGET MATCHES "integrity" OR GHS_TARGET_PLATFORM MATCHES "integrity")
      list(FILTER GHS_CANDIDATE_OS_DIRS INCLUDE REGEX "int[0-9][0-9][0-9][0-9a-z]")
    endif()

    # Select latest? of matching candidates
    if(GHS_CANDIDATE_OS_DIRS)
      list(SORT GHS_CANDIDATE_OS_DIRS)
      list(GET GHS_CANDIDATE_OS_DIRS -1 _ghs_os_dir)
      string(CONCAT _ghs_os_dir ${GHS_OS_ROOT} "/" ${_ghs_os_dir})
    endif()
  endif()
endif()

#Used for targets requiring RTOS
if((NOT DEFINED GHS_OS_DIR) OR (DEFINED CACHE{GHS_OS_DIR}))
  set(GHS_OS_DIR "${_ghs_os_dir}" CACHE PATH "GHS platform OS directory")
  mark_as_advanced(GHS_OS_DIR)
endif()
unset(_ghs_os_dir)

if((NOT DEFINED GHS_OS_DIR_OPTION) OR (DEFINED CACHE{GHS_OS_DIR_OPTION}))
  set(GHS_OS_DIR_OPTION "-os_dir " CACHE STRING "GHS compiler OS option")
  mark_as_advanced(GHS_OS_DIR_OPTION)
endif()

# Select GHS_BSP_NAME if not set by user and is known to be required
if(GHS_PRIMARY_TARGET MATCHES "integrity" OR GHS_TARGET_PLATFORM MATCHES "integrity")
  set(_ghs_bsp_name "GHS_BSP_NAME-NOT-SPECIFIED")
else()
  set(_ghs_bsp_name "IGNORE")
endif()

if(_ghs_bsp_name AND NOT DEFINED GHS_BSP_NAME)
  # First try taking architecture from `-A` option
  if(CMAKE_GENERATOR_PLATFORM)
    set(_ghs_bsp_name "sim${CMAKE_GENERATOR_PLATFORM}")
  else()
    set(_ghs_bsp_name "simarm")
  endif()
endif()

if((NOT DEFINED GHS_BSP_NAME) OR (DEFINED CACHE{GHS_BSP_NAME}))
  set(GHS_BSP_NAME "${_ghs_bsp_name}" CACHE STRING "BSP name")
  mark_as_advanced(GHS_BSP_NAME)
endif()
unset(_ghs_bsp_name)
