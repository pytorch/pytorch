#include <c10/cuda/CUDAMiscFunctions.h>
#include <c10/util/env.h>
#include <string>

namespace c10::cuda {

// Explain common CUDA errors
// NOLINTNEXTLINE(bugprone-exception-escape,-warnings-as-errors)
std::string get_cuda_error_help(cudaError_t error) noexcept {
  std::string help_text;
  switch (error) {
    case cudaErrorInvalidDevice:
      help_text.append(
          "\nGPU device may be out of range, do you have enough GPUs?");
      break;
    default:
      help_text.append("\nSearch for `")
          .append(cudaGetErrorName(error))
#if defined(USE_ROCM)
          .append(
              "' in https://rocm.docs.amd.com/projects/HIP/en/latest/index.html for more information.");
#else
          .append(
              "' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.");
#endif
      break;
  }
  return help_text;
}

namespace {

inline auto get_cuda_blocking_enabled() {
#ifndef USE_ROCM
  static auto device_blocking_flag = c10::utils::check_env("CUDA_LAUNCH_BLOCKING");
  return device_blocking_flag.value_or(false);
#else
  static auto device_blocking_flag = c10::utils::get_env("AMD_SERIALIZE_KERNEL");
  static auto effective_flag = device_blocking_flag.value_or("0");
  if (effective_flag == "0") {
    return false;
  }
  if (effective_flag == "3") {
    return true;
  }
  if (effective_flag == "1" || effective_flag == "2") {
    TORCH_WARN_ONCE("AMD_SERIALIZE_KERNEL=3 waits for completion before AND after kernel enqueue"
                    ". 1/2 Only waits before or after enqueue.");
    return true;
  }
  TORCH_WARN_ONCE("Unsupported AMD_SERIALIZE_KERNEL value ", device_blocking_flag);
  return false;
#endif
}

}

// NOLINTNEXTLINE(bugprone-exception-escape,-warnings-as-errors)
const char* get_cuda_check_suffix() noexcept {
  static auto blocking_enabled = get_cuda_blocking_enabled();
  if (blocking_enabled) {
    return "";
  } else {
    return "\nCUDA kernel errors might be asynchronously reported at some"
           " other API call, so the stacktrace below might be incorrect."
           "\nFor debugging consider passing CUDA_LAUNCH_BLOCKING=1";
  }
}
// NOLINTNEXTLINE(bugprone-exception-escape,-warnings-as-errors)
const char* get_cuda_async_error_suffix(cudaError_t error) noexcept {
  switch (error) {
    case cudaErrorLaunchFailure:
    case cudaErrorIllegalAddress:
    case cudaErrorAssert:
#ifndef USE_ROCM
    case cudaErrorIllegalInstruction:
    case cudaErrorMisalignedAddress:
#endif
    {
      static auto blocking_enabled = get_cuda_blocking_enabled();
      if (!blocking_enabled) {
        return "\nCUDA kernel errors might be asynchronously reported at some"
               " other API call, so the stacktrace below might be incorrect."
               "\nFor debugging consider passing CUDA_LAUNCH_BLOCKING=1";
      }
      return "";
    }
    default:
      return "\nFor more detailed error information, run with"
             " CUDA_LOG_FILE=stderr";
  }
}

std::mutex* getFreeMutex() {
  static std::mutex cuda_free_mutex;
  return &cuda_free_mutex;
}

} // namespace c10::cuda
