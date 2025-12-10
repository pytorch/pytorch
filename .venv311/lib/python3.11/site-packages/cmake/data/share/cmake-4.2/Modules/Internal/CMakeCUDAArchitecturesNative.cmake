# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

function(cmake_cuda_architectures_native lang)
  # Run the test binary to detect the native architectures.
  execute_process(COMMAND "${CMAKE_PLATFORM_INFO_DIR}/CMakeDetermineCompilerABI_${lang}.bin"
    RESULT_VARIABLE archs_result
    OUTPUT_VARIABLE archs_output
    ERROR_VARIABLE  archs_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  if(archs_result EQUAL 0)
    if("$ENV{CMAKE_CUDA_ARCHITECTURES_NATIVE_CLAMP}")
      # Undocumented hook used by CMake's CI.
      # Clamp native architecture to version range supported by this CUDA.
      list(GET CMAKE_${lang}_ARCHITECTURES_ALL 0  arch_min)
      list(GET CMAKE_${lang}_ARCHITECTURES_ALL -1 arch_max)
      set(CMAKE_CUDA_ARCHITECTURES_NATIVE "")
      foreach(arch IN LISTS archs_output)
        if(arch LESS arch_min)
          set(arch "${arch_min}")
        endif()
        if(arch GREATER arch_max)
          set(arch "${arch_max}")
        endif()
        list(APPEND CMAKE_CUDA_ARCHITECTURES_NATIVE ${arch})
      endforeach()
      unset(arch)
      unset(arch_min)
      unset(arch_max)
    else()
      set(CMAKE_CUDA_ARCHITECTURES_NATIVE "${archs_output}")
    endif()
    list(REMOVE_DUPLICATES CMAKE_CUDA_ARCHITECTURES_NATIVE)
    list(TRANSFORM CMAKE_CUDA_ARCHITECTURES_NATIVE APPEND "-real")
  else()
    if(NOT archs_result MATCHES "[0-9]+")
      set(archs_status " (${archs_result})")
    else()
      set(archs_status "")
    endif()
    string(REPLACE "\n" "\n  " archs_output "  ${archs_output}")
    message(CONFIGURE_LOG
      "Detecting the CUDA native architecture(s) failed with "
      "the following output${archs_status}:\n${archs_output}\n\n")
  endif()

  set(CMAKE_${lang}_ARCHITECTURES_NATIVE "${CMAKE_CUDA_ARCHITECTURES_NATIVE}" PARENT_SCOPE)
endfunction()
