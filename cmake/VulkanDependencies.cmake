if(NOT USE_VULKAN)
  return()
endif()

if(ANDROID)
  if(NOT ANDROID_NDK)
    message(FATAL_ERROR "USE_VULKAN requires ANDROID_NDK set.")
  endif()

  # Vulkan from ANDROID_NDK
  set(VULKAN_INCLUDE_DIR "${ANDROID_NDK}/sources/third_party/vulkan/src/include")
  message(STATUS "VULKAN_INCLUDE_DIR:${VULKAN_INCLUDE_DIR}")

  set(VULKAN_ANDROID_NDK_WRAPPER_DIR "${ANDROID_NDK}/sources/third_party/vulkan/src/common")
  message(STATUS "Vulkan_ANDROID_NDK_WRAPPER_DIR:${VULKAN_ANDROID_NDK_WRAPPER_DIR}")
  set(VULKAN_WRAPPER_DIR "${VULKAN_ANDROID_NDK_WRAPPER_DIR}")

  add_library(
    VulkanWrapper
    STATIC
    ${VULKAN_WRAPPER_DIR}/vulkan_wrapper.h
    ${VULKAN_WRAPPER_DIR}/vulkan_wrapper.cpp)

  target_include_directories(VulkanWrapper PUBLIC .)
  target_include_directories(VulkanWrapper PUBLIC "${VULKAN_INCLUDE_DIR}")
  target_link_libraries(VulkanWrapper ${CMAKE_DL_LIBS})

  string(APPEND Vulkan_DEFINES " -DUSE_VULKAN_WRAPPER")
  list(APPEND Vulkan_INCLUDES ${VULKAN_WRAPPER_DIR})
  list(APPEND Vulkan_LIBS VulkanWrapper)

  # Shaderc
  if(USE_VULKAN_SHADERC_RUNTIME)
    # Shaderc from ANDROID_NDK
    set(Shaderc_ANDROID_NDK_INCLUDE_DIR "${ANDROID_NDK}/sources/third_party/shaderc/include")
    message(STATUS "Shaderc_ANDROID_NDK_INCLUDE_DIR:${Shaderc_ANDROID_NDK_INCLUDE_DIR}")

    find_path(
      GOOGLE_SHADERC_INCLUDE_DIRS
      NAMES shaderc/shaderc.hpp
      PATHS "${Shaderc_ANDROID_NDK_INCLUDE_DIR}")

    set(Shaderc_ANDROID_NDK_LIB_DIR "${ANDROID_NDK}/sources/third_party/shaderc/libs/${ANDROID_STL}/${ANDROID_ABI}")
    message(STATUS "Shaderc_ANDROID_NDK_LIB_DIR:${Shaderc_ANDROID_NDK_LIB_DIR}")

    find_library(
      GOOGLE_SHADERC_LIBRARIES
      NAMES shaderc
      PATHS "${Shaderc_ANDROID_NDK_LIB_DIR}")

    # Shaderc in NDK is not prebuilt
    if(NOT GOOGLE_SHADERC_LIBRARIES)
      set(NDK_SHADERC_DIR "${ANDROID_NDK}/sources/third_party/shaderc")
      set(NDK_BUILD_CMD "${ANDROID_NDK}/ndk-build")

      execute_process(
        COMMAND ${NDK_BUILD_CMD}
        NDK_PROJECT_PATH=${NDK_SHADERC_DIR}
        APP_BUILD_SCRIPT=${NDK_SHADERC_DIR}/Android.mk
        APP_PLATFORM=${ANDROID_PLATFORM}
        APP_STL=${ANDROID_STL}
        APP_ABI=${ANDROID_ABI}
        libshaderc_combined -j8
        WORKING_DIRECTORY "${NDK_SHADERC_DIR}"
        RESULT_VARIABLE error_code)

      if(error_code)
        message(FATAL_ERROR "Failed to build ANDROID_NDK Shaderc error_code:${error_code}")
      else()
        unset(GOOGLE_SHADERC_LIBRARIES CACHE)
        find_library(
          GOOGLE_SHADERC_LIBRARIES
          NAMES shaderc
          HINTS "${Shaderc_ANDROID_NDK_LIB_DIR}")
      endif()
    endif(NOT GOOGLE_SHADERC_LIBRARIES)

    if(GOOGLE_SHADERC_INCLUDE_DIRS AND GOOGLE_SHADERC_LIBRARIES)
      message(STATUS "Shaderc FOUND include:${GOOGLE_SHADERC_INCLUDE_DIRS}")
      message(STATUS "Shaderc FOUND libs:${GOOGLE_SHADERC_LIBRARIES}")
    endif()

    list(APPEND Vulkan_INCLUDES ${GOOGLE_SHADERC_INCLUDE_DIRS})
    list(APPEND Vulkan_LIBS ${GOOGLE_SHADERC_LIBRARIES})
  endif(USE_VULKAN_SHADERC_RUNTIME)
else()
  find_package(Vulkan)

  if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "USE_VULKAN requires either Vulkan installed on system path or environment var VULKAN_SDK set.")
  endif()

  list(APPEND Vulkan_INCLUDES ${Vulkan_INCLUDE_DIRS})
  list(APPEND Vulkan_LIBS ${Vulkan_LIBRARIES})

  set(GOOGLE_SHADERC_INCLUDE_SEARCH_PATH ${Vulkan_INCLUDE_DIR})
  set(GOOGLE_SHADERC_LIBRARY_SEARCH_PATH ${Vulkan_LIBRARY})

  if(USE_VULKAN_SHADERC_RUNTIME)
    find_path(
        GOOGLE_SHADERC_INCLUDE_DIRS
        NAMES shaderc/shaderc.hpp
        PATHS ${GOOGLE_SHADERC_INCLUDE_SEARCH_PATH})

    find_library(
        GOOGLE_SHADERC_LIBRARIES
        NAMES shaderc_combined
        PATHS ${GOOGLE_SHADERC_LIBRARY_SEARCH_PATH})

    find_package_handle_standard_args(
        Shaderc
        DEFAULT_MSG
        GOOGLE_SHADERC_INCLUDE_DIRS
        GOOGLE_SHADERC_LIBRARIES)

    if(NOT Shaderc_FOUND)
      message(FATAL_ERROR "USE_VULKAN: Shaderc not found in VULKAN_SDK")
    else()
      message(STATUS "shaderc FOUND include:${GOOGLE_SHADERC_INCLUDE_DIRS}")
      message(STATUS "shaderc FOUND libs:${GOOGLE_SHADERC_LIBRARIES}")
    endif()

    list(APPEND Vulkan_INCLUDES ${GOOGLE_SHADERC_INCLUDE_DIRS})
    list(APPEND Vulkan_LIBS ${GOOGLE_SHADERC_LIBRARIES})
  endif(USE_VULKAN_SHADERC_RUNTIME)
endif()
