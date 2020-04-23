# Shaders processing

execute_process(
  COMMAND
  "${PYTHON_EXECUTABLE}" 
  ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/vulkan/gen_glsl.py
  --glsl-path ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/vulkan/glsl
  --output-path ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/vulkan
  RESULT_VARIABLE error_code)

if(error_code)
  message(FATAL_ERROR "Failed to gen glsl.h and glsl.cpp with shaders sources for Vulkan backend")
endif()

if(NOT USE_VULKAN_SHADERC_RUNTIME)

if(NOT ANDROID_NDK)
  message(FATAL_ERROR "Failed to find glslc to compile glsl")
endif()

set(GLSLC_PATH "${ANDROID_NDK}/shader-tools/${ANDROID_NDK_HOST_SYSTEM_NAME}/glslc")

execute_process(
  COMMAND
  "${PYTHON_EXECUTABLE}" 
  ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/vulkan/gen_spv.py
  --glsl-path ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/vulkan/glsl
  --output-path ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/vulkan
  --glslc-path=${GLSLC_PATH}
  --tmp-spv-path=${CMAKE_BINARY_DIR}
  RESULT_VARIABLE error_code)

  if(error_code)
    message(FATAL_ERROR "Failed to gen spv.h and spv.cpp with precompiled shaders for Vulkan backend")
  endif()

endif()
