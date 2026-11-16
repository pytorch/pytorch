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

else()
  find_package(Vulkan)

  if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "USE_VULKAN requires either Vulkan installed on system path or environment var VULKAN_SDK set.")
  endif()

  list(APPEND Vulkan_INCLUDES ${Vulkan_INCLUDE_DIRS})
  list(APPEND Vulkan_LIBS ${Vulkan_LIBRARIES})

  set(GOOGLE_SHADERC_INCLUDE_SEARCH_PATH ${Vulkan_INCLUDE_DIR})
  set(GOOGLE_SHADERC_LIBRARY_SEARCH_PATH ${Vulkan_LIBRARY})
endif()
