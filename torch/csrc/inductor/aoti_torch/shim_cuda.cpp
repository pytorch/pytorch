
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

AOTITorchError aoti_torch_create_cuda_guard(
    int32_t device_index,
    CUDAGuardHandle* ret_guard // returns new reference
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE("aoti_torch_create_cuda_guard", {
    at::cuda::CUDAGuard* guard = new at::cuda::CUDAGuard(device_index);
    *ret_guard = reinterpret_cast<CUDAGuardHandle>(guard);
  });
}

AOTITorchError aoti_torch_delete_cuda_guard(CUDAGuardHandle guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE("aoti_torch_delete_cuda_guard", {
    delete reinterpret_cast<at::cuda::CUDAGuard*>(guard);
  });
}

AOTITorchError aoti_torch_cuda_guard_set_index(
    CUDAGuardHandle guard,
    int32_t device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      "aoti_torch_cuda_guard_set_index", {
        reinterpret_cast<at::cuda::CUDAGuard*>(guard)->set_index(device_index);
      });
}

AOTITorchError aoti_torch_create_cuda_stream_guard(
    void* stream,
    int32_t device_index,
    CUDAStreamGuardHandle* ret_guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      "aoti_torch_create_cuda_stream_guard", {
        at::cuda::CUDAStreamGuard* guard =
            new at::cuda::CUDAStreamGuard(at::cuda::getStreamFromExternal(
                static_cast<cudaStream_t>(stream), device_index));
        *ret_guard = reinterpret_cast<CUDAStreamGuardHandle>(guard);
      });
}

AOTITorchError aoti_torch_delete_cuda_stream_guard(
    CUDAStreamGuardHandle guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      "aoti_torch_delete_cuda_stream_guard",
      { delete reinterpret_cast<at::cuda::CUDAStreamGuard*>(guard); });
}

AOTITorchError aoti_torch_get_current_cuda_stream(
    int32_t device_index,
    void** ret_stream) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      "aoti_torch_get_current_cuda_stream", {
        *(cudaStream_t*)(ret_stream) =
            at::cuda::getCurrentCUDAStream(device_index);
      });
}
