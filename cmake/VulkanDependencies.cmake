if(NOT USE_VULKAN)
  return()
endif()

find_package(Vulkan)

if(NOT Vulkan_FOUND)
  message(FATAL_ERROR "USE_VULKAN requires either Vulkan installed on system path or environment var VULKAN_SDK set.")
endif()

list(APPEND Vulkan_INCLUDES ${Vulkan_INCLUDE_DIRS})
list(APPEND Vulkan_LIBS ${Vulkan_LIBRARIES})

set(GOOGLE_SHADERC_INCLUDE_SEARCH_PATH ${Vulkan_INCLUDE_DIR})
set(GOOGLE_SHADERC_LIBRARY_SEARCH_PATH ${Vulkan_LIBRARY})
