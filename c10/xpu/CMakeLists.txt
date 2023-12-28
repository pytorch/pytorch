# Build file for the C10 XPU.
#
# C10 XPU is a minimal library, but it does depend on SYCL.

include(../../cmake/public/xpu.cmake)

# ---[ Configure macro file.
set(C10_XPU_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}) # used in cmake_macros.h.in
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/impl/xpu_cmake_macros.h.in
    ${CMAKE_BINARY_DIR}/c10/xpu/impl/xpu_cmake_macros.h)

set(C10_XPU_SRCS
    XPUFunctions.cpp
    XPUStream.cpp
)
set(C10_XPU_HEADERS
    XPUDeviceProp.h
    XPUException.h
    XPUFunctions.h
    XPUMacros.h
    XPUStream.h
)

add_library(c10_xpu ${C10_XPU_SRCS} ${C10_XPU_HEADERS})
# Enable hidden visibility if compiler supports it.
if(${COMPILER_SUPPORTS_HIDDEN_VISIBILITY})
  target_compile_options(c10_xpu PRIVATE "-fvisibility=hidden")
endif()

set_target_properties(c10_xpu PROPERTIES CXX_STANDARD 17)

# ---[ Dependency of c10_xpu
target_link_libraries(c10_xpu PUBLIC c10 torch::xpurt)
target_include_directories(
    c10_xpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>
    )

add_subdirectory(test)

# ---[ Installation
install(TARGETS c10_xpu EXPORT Caffe2Targets DESTINATION lib)
foreach(file ${C10_XPU_HEADERS})
  get_filename_component(dir ${file} DIRECTORY)
  install(FILES ${file} DESTINATION include/c10/xpu/${dir})
endforeach()
install(FILES ${CMAKE_BINARY_DIR}/c10/xpu/impl/xpu_cmake_macros.h
  DESTINATION include/c10/xpu/impl)

if(MSVC AND C10_XPU_BUILD_SHARED_LIBS)
  install(FILES $<TARGET_PDB_FILE:c10_xpu> DESTINATION lib OPTIONAL)
endif()
