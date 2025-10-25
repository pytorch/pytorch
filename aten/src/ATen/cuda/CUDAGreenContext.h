#pragma once
#include <ATen/cuda/CUDAEvent.h>

#if defined(CUDA_VERSION) && !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#include <cuda.h>
#include <memory>
#include <stdexcept>
#include <vector>
#define CUDA_HAS_GREEN_CONTEXT 1
#else
#define CUDA_HAS_GREEN_CONTEXT 0
#endif

namespace at::cuda {

class TORCH_CUDA_CPP_API GreenContext {
 public:
  GreenContext(uint32_t device_id, uint32_t num_sms);

  static std::unique_ptr<GreenContext> create(uint32_t num_sms, std::optional<uint32_t> device_id);

  // Delete copy constructor and assignment
  GreenContext(const GreenContext&) = delete;
  GreenContext& operator=(const GreenContext&) = delete;

  // Implement move operations
  GreenContext(GreenContext&& other) noexcept;
  GreenContext& operator=(GreenContext&& other) noexcept;
  ~GreenContext() noexcept;

  // Get the underlying CUDA context
  CUcontext getContext() const;

  // Get the underlying green context
#if CUDA_HAS_GREEN_CONTEXT
  CUgreenCtx getGreenContext() const;
#endif

  // Make this context current
  void setContext();

  void popContext();

 private:
#if CUDA_HAS_GREEN_CONTEXT
  int32_t device_id_ = -1;
  CUgreenCtx green_ctx_ = nullptr;
  CUcontext context_ = nullptr;
  cudaStream_t parent_stream_ = nullptr;
#endif
};
} // namespace at::cuda
