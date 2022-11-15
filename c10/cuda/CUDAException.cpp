#include <c10/cuda/CUDAException.h>

#include <c10/util/Exception.h>
#include <cuda_runtime.h>

#include <string>

namespace c10 {
namespace cuda {

void c10_cuda_check_implementation(
    const char* filename,
    const char* function_name,
    const int line_number,
    const bool include_device_assertions) {
  // We retrieve the error here in order to keep CUDA data types out of
  // CUDAException.h thereby simplifying including it in other files
  const cudaError_t err = cudaGetLastError();

  if (C10_LIKELY(err == cudaSuccess)) {
    return;
  }

  std::string check_message;
#ifndef STRIP_ERROR_MESSAGES
  check_message.append("CUDA error: ");
  check_message.append(cudaGetErrorString(err));
  check_message.append(c10::cuda::get_cuda_check_suffix());
#endif

  TORCH_CHECK(false, check_message);
}

} // namespace cuda
} // namespace c10
