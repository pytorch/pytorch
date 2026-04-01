#
# Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

option(VMA_STATIC_VULKAN_FUNCTIONS "Link statically with Vulkan API" ON)
option(VMA_DYNAMIC_VULKAN_FUNCTIONS "Fetch pointers to Vulkan functions internally (no static linking)" OFF)
set(VMA_VOLK_HEADER_PATH "" CACHE STRING "Path to volk.h file from the volk library (optional)")
option(VMA_DEBUG_ALWAYS_DEDICATED_MEMORY "Every allocation will have its own memory block" OFF)
option(VMA_DEBUG_INITIALIZE_ALLOCATIONS "Automatically fill new allocations and destroyed allocations with some bit pattern" OFF)
option(VMA_DEBUG_GLOBAL_MUTEX "Enable single mutex protecting all entry calls to the library" OFF)
option(VMA_DEBUG_DONT_EXCEED_MAX_MEMORY_ALLOCATION_COUNT "Never exceed VkPhysicalDeviceLimits::maxMemoryAllocationCount and return error" OFF)

message(STATUS "VMA_STATIC_VULKAN_FUNCTIONS = ${VMA_STATIC_VULKAN_FUNCTIONS}")
message(STATUS "VMA_DYNAMIC_VULKAN_FUNCTIONS = ${VMA_DYNAMIC_VULKAN_FUNCTIONS}")
message(STATUS "VMA_DEBUG_ALWAYS_DEDICATED_MEMORY = ${VMA_DEBUG_ALWAYS_DEDICATED_MEMORY}")
message(STATUS "VMA_DEBUG_INITIALIZE_ALLOCATIONS = ${VMA_DEBUG_INITIALIZE_ALLOCATIONS}")
message(STATUS "VMA_DEBUG_GLOBAL_MUTEX = ${VMA_DEBUG_GLOBAL_MUTEX}")
message(STATUS "VMA_DEBUG_DONT_EXCEED_MAX_MEMORY_ALLOCATION_COUNT = ${VMA_DEBUG_DONT_EXCEED_MAX_MEMORY_ALLOCATION_COUNT}")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan REQUIRED)

add_executable(VmaSample)
target_sources(VmaSample PRIVATE
    Common.cpp
    Common.h
    SparseBindingTest.cpp
    SparseBindingTest.h
    Tests.cpp
    Tests.h
    VmaUsage.cpp
    VmaUsage.h
    VolkUsage.cpp
    VulkanSample.cpp
    ../include/vk_mem_alloc.h
)

# Only link to Vulkan library if volk loader is not used, but always add Vulkan headers directory.
if("${VMA_VOLK_HEADER_PATH}" STREQUAL "")
    target_link_libraries(VmaSample PUBLIC Vulkan::Vulkan)
else()
    target_link_libraries(VmaSample PUBLIC Vulkan::Headers)
endif()

target_link_libraries(VmaSample PRIVATE GPUOpen::VulkanMemoryAllocator)

target_compile_definitions(VmaSample PUBLIC
    VMA_STATIC_VULKAN_FUNCTIONS=$<BOOL:${VMA_STATIC_VULKAN_FUNCTIONS}>
    VMA_DYNAMIC_VULKAN_FUNCTIONS=$<BOOL:${VMA_DYNAMIC_VULKAN_FUNCTIONS}>
    VMA_DEBUG_ALWAYS_DEDICATED_MEMORY=$<BOOL:${VMA_DEBUG_ALWAYS_DEDICATED_MEMORY}>
    VMA_DEBUG_INITIALIZE_ALLOCATIONS=$<BOOL:${VMA_DEBUG_INITIALIZE_ALLOCATIONS}>
    VMA_DEBUG_GLOBAL_MUTEX=$<BOOL:${VMA_DEBUG_GLOBAL_MUTEX}>
    VMA_DEBUG_DONT_EXCEED_MAX_MEMORY_ALLOCATION_COUNT=$<BOOL:${VMA_DEBUG_DONT_EXCEED_MAX_MEMORY_ALLOCATION_COUNT}>
)

# Provides MSVC users nicer debugging support
target_sources(VmaSample PRIVATE vk_mem_alloc.natvis)

add_subdirectory(Shaders)
add_dependencies(VmaSample VmaSampleShaders)

if(NOT "${VMA_VOLK_HEADER_PATH}" STREQUAL "")
    if(EXISTS "${VMA_VOLK_HEADER_PATH}")
        message(STATUS "File volk.h found and used from path: ${VMA_VOLK_HEADER_PATH}")
        target_compile_definitions(VmaSample PRIVATE VMA_VOLK_HEADER_PATH="${VMA_VOLK_HEADER_PATH}")
    else()
        message(FATAL_ERROR "File volk.h not found in path: ${VMA_VOLK_HEADER_PATH}")
    endif()
endif()

# Use Unicode instead of multibyte set
add_compile_definitions(UNICODE _UNICODE)

# Add C++ warnings and security checks
add_compile_options(/permissive- /sdl /W3)

# Set VmaSample as startup project
set_property(DIRECTORY "${PROJECT_SOURCE_DIR}" PROPERTY VS_STARTUP_PROJECT "VmaSample")

set_target_properties(VmaSample PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
