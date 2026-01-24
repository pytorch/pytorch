#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/utils/cpp_stacktraces.h>
#include <cstring>

namespace {
// Helper to call the appropriate check implementation for CUDA vs ROCm.
// This is done in a separate function to avoid preprocessor directives inside
// macro (AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE) arguments, which
// is undefined behavior and fails on MSVC.
inline void call_c10_accelerator_check_implementation(
    int32_t err,
    const char* filename,
    const char* function_name,
    uint32_t line_number,
    bool include_device_assertions) {
#ifdef USE_ROCM
  c10::hip::c10_hip_check_implementation(
      err, filename, function_name, line_number, include_device_assertions);
#else
  c10::cuda::c10_cuda_check_implementation(
      err, filename, function_name, line_number, include_device_assertions);
#endif
}
} // namespace

AOTITorchError torch_get_current_cuda_blas_handle(void** ret_handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *(cublasHandle_t*)(ret_handle) = at::cuda::getCurrentCUDABlasHandle();
  });
}

AOTITorchError torch_set_current_cuda_stream(
    void* stream,
    int32_t device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::cuda::setCurrentCUDAStream(at::cuda::getStreamFromExternal(
        static_cast<cudaStream_t>(stream), device_index));
  });
}

AOTITorchError torch_get_cuda_stream_from_pool(
    const bool isHighPriority,
    int32_t device_index,
    void** ret_stream) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *(cudaStream_t*)(ret_stream) =
        at::cuda::getStreamFromPool(isHighPriority, device_index);
  });
}

AOTITorchError torch_cuda_stream_synchronize(
    void* stream,
    int32_t device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::cuda::getStreamFromExternal(
        static_cast<cudaStream_t>(stream), device_index)
        .synchronize();
  });
}

AOTITorchError torch_c10_cuda_check_msg(
    int32_t err,
    const char* filename,
    const char* function_name,
    uint32_t line_number,
    bool include_device_assertions,
    char** error_msg) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *error_msg = nullptr;

    try {
      call_c10_accelerator_check_implementation(
          err, filename, function_name, line_number, include_device_assertions);
    } catch (const c10::AcceleratorError& e) {
      // Match the behavior of Python exception translation:
      // use what() if C++ stacktraces are enabled, otherwise
      // what_without_backtrace()
      const char* what_str = torch::get_cpp_stacktraces_enabled()
          ? e.what()
          : e.what_without_backtrace();
      size_t msg_len = std::strlen(what_str);
      *error_msg = new char[msg_len + 1];
      std::memcpy(*error_msg, what_str, msg_len + 1);
    }
  });
}

void torch_c10_cuda_free_error_msg(char* error_msg) {
  delete[] error_msg;
}
