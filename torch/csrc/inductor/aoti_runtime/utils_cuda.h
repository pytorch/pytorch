#pragma once

#ifdef USE_CUDA
// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
#include <torch/csrc/inductor/aoti_torch/utils.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace torch::aot_inductor {

inline void delete_cuda_stream_guard(void* ptr) {
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_delete_cuda_stream_guard(
      reinterpret_cast<CUDAStreamGuardHandle>(ptr)));
}

class AOTICudaStreamGuard {
 public:
  AOTICudaStreamGuard(cudaStream_t stream, int32_t device_index)
      : guard_(nullptr, delete_cuda_stream_guard) {
    CUDAStreamGuardHandle ptr;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_create_cuda_stream_guard(stream, device_index, &ptr));
    guard_.reset(ptr);
  }

 private:
  std::unique_ptr<CUDAStreamGuardOpaque, DeleterFnPtr> guard_;
};

} // namespace torch::aot_inductor
#endif // USE_CUDA
