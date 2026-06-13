# Vendored JACCL (MLX's RDMA-over-Thunderbolt collective library) built as a
# static lib. JACCL is macOS-only and requires the macOS 26.2 SDK. On any other
# platform / older SDK we emit a stub INTERFACE target so downstream
# `target_link_libraries(... jaccl)` still works and JACCL_AVAILABLE is 0.
#
# Source of truth: third_party/jaccl/SYNC.md

set(JACCL_AVAILABLE 0 PARENT_SCOPE)

if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  message(STATUS "JACCL: not Darwin, skipping.")
  add_library(jaccl INTERFACE)
  return()
endif()

if(NOT DEFINED MACOS_SDK_VERSION)
  execute_process(
    COMMAND xcrun --sdk macosx --show-sdk-version
    OUTPUT_VARIABLE MACOS_SDK_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(NOT MACOS_SDK_VERSION OR MACOS_SDK_VERSION VERSION_LESS "26.2")
  message(STATUS "JACCL: macOS SDK < 26.2 (got '${MACOS_SDK_VERSION}'), skipping.")
  add_library(jaccl INTERFACE)
  return()
endif()

message(STATUS "JACCL: building with macOS SDK ${MACOS_SDK_VERSION}")

add_library(
  jaccl STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/jaccl/tcp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/jaccl/rdma.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/jaccl/mesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/jaccl/ring.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/jaccl/jaccl.cpp)

# Public: consumers include as <jaccl/xxx.h>.
target_include_directories(
  jaccl PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

# Private: jaccl.cpp does `#include <json.hpp>` — reuse PyTorch's vendored
# nlohmann/json rather than a separate FetchContent download.
target_include_directories(
  jaccl PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../nlohmann/single_include/nlohmann)

target_compile_features(jaccl PUBLIC cxx_std_20)
set_target_properties(jaccl PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(JACCL_AVAILABLE 1 PARENT_SCOPE)
