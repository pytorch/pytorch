# Device-side rocFFT JIT callbacks.
#
# aten/src/ATen/native/hip/rocfft/Callbacks.hip holds load/store callbacks that
# rocFFT links into its own kernels, so stft/istft can fuse framing and
# windowing into the transform. rocFFT takes them as a SPIR-V module, which is
# not something the normal HIP compile produces, so the file is built on its own
# here and embedded as a byte array.
#
# Two things the ROCm install may not have are needed: the JIT callback entry
# points in rocfft.h, and a clang that can target amdgcnspirv. Whether the
# *runtime* can then JIT the module into a kernel is a separate question that
# only plan creation can answer; rocfft_callbacks_available() handles that.

include(CheckCXXSourceCompiles)

if(NOT DEFINED USE_ROCFFT_CALLBACKS)
  find_program(AMD_LLVM_SPIRV amd-llvm-spirv
               HINTS ${ROCM_PATH}/lib/llvm/bin ${ROCM_PATH}/bin)

  set(CMAKE_REQUIRED_INCLUDES ${ROCM_INCLUDE_DIRS})
  # decltype only, so this stays a compile check and does not have to link rocFFT.
  check_cxx_source_compiles("
    #include <rocfft/rocfft.h>
    using load_cb_t = decltype(rocfft_plan_description_set_load_callback);
    using store_cb_t = decltype(rocfft_plan_description_set_store_callback);
    int main() { return 0; }
  " ROCFFT_HAS_JIT_CALLBACK_API)
  unset(CMAKE_REQUIRED_INCLUDES)

  if(NOT DEFINED ROCFFT_CAN_COMPILE_SPIRV AND AMD_LLVM_SPIRV)
    set(_probe "${CMAKE_BINARY_DIR}/rocfft_spirv_probe")
    file(WRITE "${_probe}.hip"
         "extern \"C\" __device__ float probe(float* p, unsigned long i) { return p[i]; }\n")
    # Separate execute_process calls: several COMMANDs in one would be run as a
    # pipeline, not in sequence.
    execute_process(
      COMMAND ${CMAKE_HIP_COMPILER} -x hip --offload-device-only
              --offload-arch=amdgcnspirv -fgpu-rdc -emit-llvm
              --rocm-path=${ROCM_PATH} -c "${_probe}.hip" -o "${_probe}.bc"
      RESULT_VARIABLE _probe_rc OUTPUT_QUIET ERROR_QUIET)
    if(_probe_rc EQUAL 0)
      execute_process(
        COMMAND ${AMD_LLVM_SPIRV} "${_probe}.bc" -o "${_probe}.spv"
        RESULT_VARIABLE _probe_rc OUTPUT_QUIET ERROR_QUIET)
    endif()
    if(_probe_rc EQUAL 0)
      message(STATUS "HIP compiler can target amdgcnspirv")
    else()
      message(STATUS "HIP compiler cannot target amdgcnspirv, rocFFT callbacks disabled")
    endif()
    set(ROCFFT_CAN_COMPILE_SPIRV ${_probe_rc} CACHE INTERNAL "amdgcnspirv probe exit status")
  endif()

  if(ROCFFT_HAS_JIT_CALLBACK_API AND AMD_LLVM_SPIRV AND ROCFFT_CAN_COMPILE_SPIRV EQUAL 0)
    set(_default ON)
  else()
    set(_default OFF)
  endif()
  set(USE_ROCFFT_CALLBACKS ${_default} CACHE BOOL "Fuse windowing into rocFFT kernels via JIT callbacks")
  message(STATUS "USE_ROCFFT_CALLBACKS: ${USE_ROCFFT_CALLBACKS}")
endif()

# Emits the rules turning Callbacks.hip into an embedded SPIR-V byte array, and
# the rocfft_callbacks target that torch_hip must depend on: the generated header
# lives under aten/src/ATen while the target compiling it is declared elsewhere,
# so the output rule is not visible from there.
function(rocfft_callbacks_add_rules)
  set(_dir "${CMAKE_SOURCE_DIR}/aten/src/ATen/native/hip/rocfft")
  set(_out "${CMAKE_CURRENT_BINARY_DIR}/native/hip")
  set(_bc "${_out}/rocfft_callbacks.bc")
  set(_spv "${_out}/rocfft_callbacks.spv")
  set(_hdr "${_out}/rocfft_callbacks_spirv.h")

  add_custom_command(
    OUTPUT ${_bc}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_out}
    COMMAND ${CMAKE_HIP_COMPILER} -x hip --offload-device-only
            --offload-arch=amdgcnspirv -fgpu-rdc -emit-llvm -O3
            --rocm-path=${ROCM_PATH} -I${CMAKE_SOURCE_DIR}/aten/src
            -c ${_dir}/Callbacks.hip -o ${_bc}
    DEPENDS ${_dir}/Callbacks.hip ${_dir}/CallbackData.h
    COMMENT "Compiling rocFFT callbacks for amdgcnspirv"
    VERBATIM)

  # The overlap-add callback does a float atomicAdd, which the translator refuses
  # to emit unless the extension is asked for by name.
  add_custom_command(
    OUTPUT ${_spv}
    COMMAND ${AMD_LLVM_SPIRV} --spirv-ext=+SPV_EXT_shader_atomic_float_add ${_bc} -o ${_spv}
    DEPENDS ${_bc}
    COMMENT "Translating rocFFT callbacks to SPIR-V"
    VERBATIM)

  add_custom_command(
    OUTPUT ${_hdr}
    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/write_rocfft_callbacks.py ${_spv} ${_hdr}
    DEPENDS ${_spv} ${CMAKE_SOURCE_DIR}/scripts/write_rocfft_callbacks.py
    COMMENT "Embedding rocFFT callback SPIR-V"
    VERBATIM)

  add_custom_target(rocfft_callbacks DEPENDS ${_hdr})
endfunction()
