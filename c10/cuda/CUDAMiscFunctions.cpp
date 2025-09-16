#include <c10/cuda/CUDAMiscFunctions.h>
#include <c10/util/env.h>
#include <cuda_runtime.h>
#include <cstring>
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
          .append(
              "' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.");
      break;
  }
  return help_text;
}

// NOLINTNEXTLINE(bugprone-exception-escape,-warnings-as-errors)
const char* get_cuda_check_suffix() noexcept {
  static auto device_blocking_flag =
      c10::utils::check_env("CUDA_LAUNCH_BLOCKING");
  static bool blocking_enabled =
      (device_blocking_flag.has_value() && device_blocking_flag.value());
  if (blocking_enabled) {
    return "";
  } else {
    return "\nCUDA kernel errors might be asynchronously reported at some"
           " other API call, so the stacktrace below might be incorrect."
           "\nFor debugging consider passing CUDA_LAUNCH_BLOCKING=1";
  }
}
std::mutex* getFreeMutex() {
  static std::mutex cuda_free_mutex;
  return &cuda_free_mutex;
}

} // namespace c10::cuda
