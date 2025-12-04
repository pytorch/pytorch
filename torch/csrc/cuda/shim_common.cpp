#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/stable/c/shim.h>

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
