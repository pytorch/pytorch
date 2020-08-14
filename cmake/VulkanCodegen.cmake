# Shaders processing
if(NOT USE_VULKAN)
  return()
endif()

set(VULKAN_GEN_OUTPUT_PATH "${CMAKE_BINARY_DIR}/vulkan/ATen/native/vulkan")
set(VULKAN_GEN_ARG_ENV "")

message(STATUS "XXX0 USE_VULKAN:${USE_VULKAN}")
message(STATUS "XXX0 USE_VULKAN_RELAXED_PRECISION:${USE_VULKAN_RELAXED_PRECISION}")

if(USE_VULKAN_RELAXED_PRECISION)
  message(STATUS "XXX IF RELAXED")
  string(APPEND VULKAN_GEN_ARG_ENV "precision=mediump")
endif()

message(STATUS "XXX ENV=${VULKAN_GEN_ARG_ENV}")

if(USE_VULKAN_SHADERC_RUNTIME)
  execute_process(
    COMMAND
    "${PYTHON_EXECUTABLE}"
    ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/gen_vulkan_glsl.py
    --glsl-path ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/vulkan/glsl
    --output-path ${VULKAN_GEN_OUTPUT_PATH}
    --tmp-dir-path=${CMAKE_BINARY_DIR}/vulkan/glsl
    --env ${VULKAN_GEN_ARG_ENV}
    RESULT_VARIABLE error_code)

  if(error_code)
    message(FATAL_ERROR "Failed to gen glsl.h and glsl.cpp with shaders sources for Vulkan backend")
  endif()

  set(vulkan_generated_cpp ${VULKAN_GEN_OUTPUT_PATH}/glsl.cpp)
  return()
endif()

if(NOT USE_VULKAN_SHADERC_RUNTIME)
  # Precompiling shaders
  if(ANDROID)
    if(NOT ANDROID_NDK)
      message(FATAL_ERROR "ANDROID_NDK not set")
    endif()

    set(GLSLC_PATH "${ANDROID_NDK}/shader-tools/${ANDROID_NDK_HOST_SYSTEM_NAME}/glslc")
  else()
    if(NOT DEFINED ENV{VULKAN_SDK})
      message(FATAL_ERROR "USE_VULKAN requires environment var VULKAN_SDK set")
    endif()

    find_program(
      GLSLC_PATH glslc
      PATHS
      ENV VULKAN_SDK
      PATHS "$ENV{VULKAN_SDK}/${CMAKE_HOST_SYSTEM_PROCESSOR}/bin")

    if(NOT GLSLC_PATH)
      message(FATAL_ERROR "USE_VULKAN glslc not found")
    endif(GLSLC_PATH)
  endif()

  execute_process(
    COMMAND
    "${PYTHON_EXECUTABLE}"
    ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/gen_vulkan_spv.py
    --glsl-path ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/vulkan/glsl
    --output-path ${VULKAN_GEN_OUTPUT_PATH}
    --glslc-path=${GLSLC_PATH}
    --tmp-dir-path=${CMAKE_BINARY_DIR}/vulkan/spv
    --env ${VULKAN_GEN_ARG_ENV}
    RESULT_VARIABLE error_code)

    if(error_code)
      message(FATAL_ERROR "Failed to gen spv.h and spv.cpp with precompiled shaders for Vulkan backend")
    endif()

  set(vulkan_generated_cpp ${VULKAN_GEN_OUTPUT_PATH}/spv.cpp)
endif()
