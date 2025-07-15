#include <c10/cuda/CUDAMiscFunctions.h>
#include <c10/util/env.h>
#include <cstring>
#include <iostream>
#include <string>

namespace c10::cuda {

// NOLINTNEXTLINE(bugprone-exception-escape,-warnings-as-errors)
std::string get_cuda_check_suffix(const char* error_string) noexcept {
  std::string suffix;

  // Explain common CUDA errors
  if (strstr(error_string, "invalid device ordinal")) {
    suffix.append("\nGPU device may be out of range, do you have enough GPUs?");
  }

  static auto device_blocking_flag =
      c10::utils::check_env("CUDA_LAUNCH_BLOCKING");
  static bool blocking_enabled =
      (device_blocking_flag.has_value() && device_blocking_flag.value());
  if (!blocking_enabled) {
    suffix.append(
        "\nCUDA kernel errors might be asynchronously reported at some"
        " other API call, so the stacktrace below might be incorrect."
        "\nFor debugging consider passing CUDA_LAUNCH_BLOCKING=1");
  }
  return suffix;
}
std::mutex* getFreeMutex() {
  static std::mutex cuda_free_mutex;
  return &cuda_free_mutex;
}

} // namespace c10::cuda
