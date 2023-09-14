
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#include <c10/cuda/CUDAStream.h>

AOTITorchError aoti_torch_get_current_cuda_stream(
    void** ret,
    int32_t device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    cudaStream_t cuda_stream = c10::cuda::getCurrentCUDAStream(device_index);
    *ret = reinterpret_cast<void*>(cuda_stream);
  });
}

AOTITorchError aoti_torch_set_current_cuda_stream(
    void* stream,
    int32_t device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    c10::cuda::setCurrentCUDAStream(
        at::cuda::getStreamFromExternal(cuda_stream, device_index));
  });
}
