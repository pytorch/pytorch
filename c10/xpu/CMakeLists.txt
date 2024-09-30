# Build file for the C10 XPU.
#
# C10 XPU is a minimal library, but it does depend on SYCL.

include(../../cmake/public/xpu.cmake)

if(NOT BUILD_LIBTORCHLESS)
  find_library(C10_XPU_LIB c10_xpu PATHS $ENV{LIBTORCH_LIB_PATH} NO_DEFAULT_PATH)
endif()

# ---[ Configure macro file.
set(C10_XPU_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}) # used in xpu_cmake_macros.h.in
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/impl/xpu_cmake_macros.h.in
    ${CMAKE_BINARY_DIR}/c10/xpu/impl/xpu_cmake_macros.h)

set(C10_XPU_SRCS
    XPUCachingAllocator.cpp
    XPUFunctions.cpp
    XPUStream.cpp
    impl/XPUGuardImpl.cpp
)
set(C10_XPU_HEADERS
    XPUCachingAllocator.h
    XPUDeviceProp.h
    XPUException.h
    XPUFunctions.h
    XPUMacros.h
    XPUStream.h
    impl/XPUGuardImpl.h
)
if(NOT BUILD_LIBTORCHLESS)
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
  install(TARGETS c10_xpu EXPORT Caffe2Targets DESTINATION lib)
  set(C10_XPU_LIB c10_xpu)
  add_subdirectory(test)
endif()

# ---[ Installation

foreach(file ${C10_XPU_HEADERS})
  get_filename_component(dir ${file} DIRECTORY)
  install(FILES ${file} DESTINATION include/c10/xpu/${dir})
endforeach()
install(FILES ${CMAKE_BINARY_DIR}/c10/xpu/impl/xpu_cmake_macros.h
  DESTINATION include/c10/xpu/impl)

if(MSVC AND C10_XPU_BUILD_SHARED_LIBS)
  install(FILES $<TARGET_PDB_FILE:c10_xpu> DESTINATION lib OPTIONAL)
endif()
