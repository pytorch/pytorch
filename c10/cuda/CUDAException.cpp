#include <c10/cuda/CUDAException.h>

#include <c10/cuda/CUDADeviceAssertionHost.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>

#include <string>

namespace c10::cuda {

void c10_cuda_check_implementation(
    const int32_t err,
    const char* /*filename*/,
    const char* /*function_name*/,
    const int /*line_number*/,
    const bool include_device_assertions) {
  const auto cuda_error = static_cast<cudaError_t>(err);
  const auto cuda_kernel_failure = include_device_assertions
      ? c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref().has_failed()
      : false;

  if (C10_LIKELY(cuda_error == cudaSuccess && !cuda_kernel_failure)) {
    return;
  }

  [[maybe_unused]] auto error_unused = cudaGetLastError();

  std::string check_message;
#ifndef STRIP_ERROR_MESSAGES
  check_message.append("CUDA error: ");
  const char* error_string = cudaGetErrorString(cuda_error);
  check_message.append(error_string);
  check_message.append(c10::cuda::get_cuda_error_help(cuda_error));
  check_message.append(c10::cuda::get_cuda_check_suffix());
  check_message.append("\n");
  if (include_device_assertions) {
    check_message.append(c10_retrieve_device_side_assertion_info());
  } else {
    check_message.append(
        "Device-side assertions were explicitly omitted for this error check; the error probably arose while initializing the DSA handlers.");
  }
#endif
  throw c10::AcceleratorError(
      {__func__, __FILE__, int32_t(__LINE__)}, err, check_message);
}

} // namespace c10::cuda
