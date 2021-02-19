cmake_minimum_required(VERSION 3.4.1)
option(BUILD_LITE_INTERPRETER "Master flag to build pytorch_jni_lite" OFF)
message(STATUS "BUILD_LITE_INTERPRETER (pytorch_jni_lite): ${BUILD_LITE_INTERPRETER}")

if(BUILD_LITE_INTERPRETER)
  project(pytorch_jni_lite CXX)
  set(PYTORCH_JNI_TARGET pytorch_jni_lite)
else()
  project(pytorch_jni CXX)
  set(PYTORCH_JNI_TARGET pytorch_jni)
endif()

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)
message(STATUS "ANDROID_STL:${ANDROID_STL}")

set(TRACE_ENABLED OFF)
if(DEFINED ENV{TRACE_ENABLED})
  if($ENV{TRACE_ENABLED} STREQUAL "1")
    message(STATUS "TRACE_ENABLED ON")
    set(TRACE_ENABLED ON)
  endif()
endif()
if(NOT TRACE_ENABLED)
  message(STATUS "TRACE_ENABLED OFF")
endif()

set(USE_VULKAN OFF)

set(pytorch_android_DIR ${CMAKE_CURRENT_LIST_DIR}/src/main/cpp)

if(ANDROID_ABI)
  set(USE_VULKAN ON)
  set(libtorch_include_DIR ${pytorch_android_DIR}/libtorch_include/${ANDROID_ABI})
  set(BUILD_SUBDIR ${ANDROID_ABI})
elseif(BUILD_LIBTORCH_WITH_JNI)
  # Don't need LIBTORCH_HOME if we're building from within PyTorch.
else()
  # Building against a pre-built libtorch.
  if(NOT LIBTORCH_HOME)
    message(FATAL_ERROR
      "pytorch_android requires LIBTORCH_HOME to be defined for non-Android builds.")
  endif()
  set(libtorch_include_DIR ${LIBTORCH_HOME}/include)
  link_directories(${LIBTORCH_HOME}/lib)
  set(BUILD_SUBDIR host)
endif()

message(STATUS "libtorch dir:${libtorch_DIR}")

configure_file(
    ${pytorch_android_DIR}/cmake_macros.h.in
    ${pytorch_android_DIR}/cmake_macros.h)


if(BUILD_LITE_INTERPRETER)
  file(GLOB pytorch_android_SOURCES
    ${pytorch_android_DIR}/pytorch_jni_lite.cpp
    ${pytorch_android_DIR}/pytorch_jni_common.cpp
    ${pytorch_android_DIR}/pytorch_jni_common.h
  )
else()
  file(GLOB pytorch_android_SOURCES
    ${pytorch_android_DIR}/pytorch_jni_jit.cpp
    ${pytorch_android_DIR}/pytorch_jni_common.cpp
    ${pytorch_android_DIR}/pytorch_jni_common.h
  )
endif()
add_library(${PYTORCH_JNI_TARGET} SHARED ${pytorch_android_SOURCES})

if(APPLE)
  # Need to add rpath so dlopen can find dependencies.
  add_custom_command(TARGET pytorch_jni
      POST_BUILD COMMAND
      ${CMAKE_INSTALL_NAME_TOOL} -add_rpath "@loader_path"
        $<TARGET_FILE:pytorch_jni>)
endif()

target_compile_options(${PYTORCH_JNI_TARGET} PRIVATE
  -fexceptions
)
target_include_directories(${PYTORCH_JNI_TARGET} BEFORE
PUBLIC $<BUILD_INTERFACE:${libtorch_include_DIR}>)

set(fbjni_DIR ${CMAKE_CURRENT_LIST_DIR}/../libs/fbjni/)
set(fbjni_BUILD_DIR ${CMAKE_BINARY_DIR}/fbjni/${BUILD_SUBDIR})

add_subdirectory(${fbjni_DIR} ${fbjni_BUILD_DIR})

# ---[ Vulkan deps
if(USE_VULKAN)
  set(Vulkan_LIBS)
  set(Vulkan_INCLUDES)
  include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/VulkanDependencies.cmake)
endif()

if(ANDROID_ABI)
  function(import_static_lib name)
    add_library(${name} STATIC IMPORTED)
    set_property(
        TARGET ${name}
        PROPERTY IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/${name}.a)
  endfunction(import_static_lib)

  import_static_lib(libtorch)
  import_static_lib(libtorch_cpu)
  import_static_lib(libc10)
  import_static_lib(libnnpack)
  import_static_lib(libXNNPACK)
  import_static_lib(libpytorch_qnnpack)
  import_static_lib(libpthreadpool)
  import_static_lib(libeigen_blas)
  import_static_lib(libcpuinfo)
  import_static_lib(libclog)

  # Link most things statically on Android.
  set(pytorch_jni_LIBS
      fbjni
      -Wl,--gc-sections
      -Wl,--whole-archive
      libtorch
      libtorch_cpu
      -Wl,--no-whole-archive
      libc10
      libnnpack
      libXNNPACK
      libpytorch_qnnpack
      libpthreadpool
      libeigen_blas
      libcpuinfo
      libclog
  )
else()
  # Prefer dynamic linking on the host
  set(pytorch_jni_LIBS
      fbjni
      torch
      torch_cpu
      c10
      cpuinfo
      clog
  )

  if(USE_NNPACK)
    list(APPEND pytorch_jni_LIBS nnpack)
  endif()

  if(USE_XNNPACK)
    list(APPEND pytorch_jni_LIBS XNNPACK)
  endif()

  if(USE_SYSTEM_PTHREADPOOL)
    list(APPEND pytorch_jni_LIBS pthreadpool)
  endif()

  if(USE_PYTORCH_QNNPACK)
    list(APPEND pytorch_jni_LIBS pytorch_qnnpack)
  endif()

endif()

if(USE_VULKAN)
  list(APPEND pytorch_jni_LIBS ${Vulkan_LIBS})
endif()

target_link_libraries(${PYTORCH_JNI_TARGET} ${pytorch_jni_LIBS})

install(TARGETS ${PYTORCH_JNI_TARGET}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}) #For windows

if(MSVC)
  install(FILES $<TARGET_PDB_FILE:pytorch_jni> DESTINATION ${CMAKE_INSTALL_LIBDIR} OPTIONAL)
  install(TARGETS ${PYTORCH_JNI_TARGET} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
