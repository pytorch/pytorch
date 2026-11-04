
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#include <ATen/cuda/tunable/Tunable.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

AOTITorchError aoti_torch_create_cuda_guard(
    int32_t device_index,
    CUDAGuardHandle* ret_guard // returns new reference
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::cuda::CUDAGuard* guard = new at::cuda::CUDAGuard(device_index);
    *ret_guard = reinterpret_cast<CUDAGuardHandle>(guard);
  });
}

AOTITorchError aoti_torch_delete_cuda_guard(CUDAGuardHandle guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { delete reinterpret_cast<at::cuda::CUDAGuard*>(guard); });
}

AOTITorchError aoti_torch_cuda_guard_set_index(
    CUDAGuardHandle guard,
    int32_t device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    reinterpret_cast<at::cuda::CUDAGuard*>(guard)->set_index(device_index);
  });
}

AOTITorchError aoti_torch_create_cuda_stream_guard(
    void* stream,
    int32_t device_index,
    CUDAStreamGuardHandle* ret_guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::cuda::CUDAStreamGuard* guard =
        new at::cuda::CUDAStreamGuard(at::cuda::getStreamFromExternal(
            static_cast<cudaStream_t>(stream), device_index));
    *ret_guard = reinterpret_cast<CUDAStreamGuardHandle>(guard);
  });
}

AOTITorchError aoti_torch_delete_cuda_stream_guard(
    CUDAStreamGuardHandle guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { delete reinterpret_cast<at::cuda::CUDAStreamGuard*>(guard); });
}

AOTITorchError aoti_torch_get_current_cuda_stream(
    int32_t device_index,
    void** ret_stream) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *(cudaStream_t*)(ret_stream) = at::cuda::getCurrentCUDAStream(device_index);
  });
}

AOTITorchError aoti_torch_create_cuda_tunableop_dynamic_dims_guard(
    uint8_t dynamic_dims_mask,
    CUDATunableOpDynamicDimsGuardHandle* ret_guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    TORCH_CHECK(
        dynamic_dims_mask <= 0xF,
        "TunableOp dynamic-dimension mask must fit in four bits, but got ",
        static_cast<int>(dynamic_dims_mask));
    *ret_guard = nullptr;
    auto* tunable_context = at::cuda::tunable::getTuningContext();
    if (dynamic_dims_mask != 0 && tunable_context->IsTuningEnabled() &&
        tunable_context->IsTunableOpEnabled()) {
      auto* guard = new at::cuda::tunable::TunableDynamicDimsGuard(
          at::cuda::tunable::DynamicDimsMask{dynamic_dims_mask});
      *ret_guard = reinterpret_cast<CUDATunableOpDynamicDimsGuardHandle>(guard);
    }
  });
}

AOTITorchError aoti_torch_delete_cuda_tunableop_dynamic_dims_guard(
    CUDATunableOpDynamicDimsGuardHandle guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    delete reinterpret_cast<at::cuda::tunable::TunableDynamicDimsGuard*>(guard);
  });
}

AOTITorchError aoti_torch_cuda_caching_allocator_raw_alloc(
    uint64_t nbytes,
    void** ret_ptr) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    if (nbytes == 0) {
      *ret_ptr = nullptr;
      return AOTI_TORCH_SUCCESS;
    }

    *ret_ptr = c10::cuda::CUDACachingAllocator::raw_alloc(nbytes);

    if (*ret_ptr == nullptr) {
      TORCH_CHECK(
          false,
          "Failed to allocate ",
          nbytes,
          " bytes from CUDA caching allocator");
    }
  });
}

AOTITorchError aoti_torch_cuda_caching_allocator_raw_delete(void* ptr) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    if (ptr != nullptr) {
      c10::cuda::CUDACachingAllocator::raw_delete(ptr);
    }
  });
}
