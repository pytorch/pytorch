#include <c10/cuda/CUDAMiscFunctions.h>
#include <stdlib.h>

namespace c10 {
namespace cuda {

std::string get_cuda_check_suffix(cudaError_t err) noexcept {
  static char* device_blocking_flag = getenv("CUDA_LAUNCH_BLOCKING");
  static bool blocking_enabled =
      (device_blocking_flag && atoi(device_blocking_flag));
  std::ostringstream oss;
  if (!blocking_enabled) {
    oss << "\nCUDA kernel errors might be asynchronously reported at some"
           " other API call, so the stacktrace below might be incorrect."
           "\nFor debugging consider passing CUDA_LAUNCH_BLOCKING=1.";
  }
  // Found by looking through manual and finding all errors with
  // "To continue using CUDA, the process must be terminated and
  // relaunched." in their description
  if (
    err == cudaErrorIllegalAddress ||
    err == cudaErrorLaunchTimeout ||
    err == cudaErrorAssert ||
    err == cudaErrorHardwareStackError ||
    err == cudaErrorIllegalInstruction ||
    err == cudaErrorMisalignedAddress ||
    err == cudaErrorInvalidAddressSpace ||
    err == cudaErrorInvalidPc ||
    err == cudaErrorLaunchFailure ||
    err == cudaErrorExternalDevice ||
    true
  ) {
    oss << "\nWARNING: This error is not recoverable.  "
           "To continue using CUDA, the process/kernel must be terminated and relaunched.";
  }
  return oss.str();
}
} // namespace cuda
} // namespace c10
