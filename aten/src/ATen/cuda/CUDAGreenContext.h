#pragma once
#include <ATen/cuda/CUDAEvent.h>
#if defined(CUDA_VERSION) && !defined(USE_ROCM)
#include <c10/cuda/driver_api.h>
#include <cuda.h>
#include <memory>
#include <stdexcept>
#include <vector>
#endif

namespace at {
namespace cuda {

class TORCH_CUDA_CPP_API GreenContext {
 public:
  GreenContext(int device_id, unsigned int num_sms);

  static std::unique_ptr<GreenContext> create(unsigned int num_sms, std::optional<unsigned int> device_id);

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
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080 && !defined(USE_ROCM)
  CUgreenCtx getGreenContext() const;
#endif

  // Make this context current
  void setContext();

  void popContext();

 private:
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080 && !defined(USE_ROCM)
  int device_id_ = -1;
  CUgreenCtx green_ctx_ = nullptr;
  CUcontext context_ = nullptr;
  cudaStream_t parent_stream_ = nullptr;
#endif
};
} // namespace cuda
} // namespace at
