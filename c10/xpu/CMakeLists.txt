# Build file for the C10 XPU.
#
# C10 XPU is a minimal library, but it does depend on SYCL.

include(../../cmake/public/xpu.cmake)

if(USE_KINETO AND PYTORCH_FOUND_XPU_PTI)
    set(C10_XPU_PROFILER_SRCS
        profiler/XPUActivityProfiler.cpp
        profiler/XPUActivityApi.cpp
        profiler/XPUActivity.cpp
    )
    set(C10_XPU_PROFILER_HEADERS
        profiler/XPUActivityProfiler.h
        profiler/XPUActivityApi.h
        profiler/XPUActivity.h
        profiler/XPUActivityBuffer.h
    )
endif()

set(C10_XPU_SRCS
    XPUCachingAllocator.cpp
    XPUFunctions.cpp
    XPUStream.cpp
    impl/XPUGuardImpl.cpp
    ${C10_XPU_PROFILER_SRCS}
)
set(C10_XPU_HEADERS
    XPUCachingAllocator.h
    XPUDeviceProp.h
    XPUException.h
    XPUFunctions.h
    XPUMacros.h
    XPUStream.h
    impl/XPUGuardImpl.h
    ${C10_XPU_PROFILER_HEADERS}
)

add_library(c10_xpu ${C10_XPU_SRCS} ${C10_XPU_HEADERS})
target_compile_options(c10_xpu PRIVATE "-DC10_XPU_BUILD_MAIN_LIB")
# Enable hidden visibility if compiler supports it.
if(${COMPILER_SUPPORTS_HIDDEN_VISIBILITY})
  target_compile_options(c10_xpu PRIVATE "-fvisibility=hidden")
endif()

# ---[ Dependency of c10_xpu
target_link_libraries(c10_xpu PUBLIC c10 torch::xpurt)
target_include_directories(
    c10_xpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>
    )
if(USE_KINETO AND PYTORCH_FOUND_XPU_PTI)
    target_link_libraries(c10_xpu PUBLIC Pti::pti_view kineto fmt::fmt-header-only)
    target_include_directories(
        c10_xpu PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/kineto/libkineto/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/fmt/include>
        )
endif()

add_subdirectory(test)

# ---[ Installation
install(TARGETS c10_xpu EXPORT Caffe2Targets DESTINATION lib)
foreach(file ${C10_XPU_HEADERS})
  get_filename_component(dir ${file} DIRECTORY)
  install(FILES ${file} DESTINATION include/c10/xpu/${dir})
endforeach()
