#pragma once
#include <ATen/cuda/CUDAEvent.h>
#include <cuda.h>

// Forward declare green context as opaque ptr
typedef struct CUgreenCtx_st* CUgreenCtx;

namespace at::cuda {

namespace {
  constexpr int kStreamPerGreenContextPool = 32;
}

// Workqueue sharing scope for green contexts.
// Values match the CUDA driver API's CUdevWorkqueueConfigScope enum.
enum class WorkqueueScope : int32_t {
  DeviceCtx = 0,
  Balanced = 1,
};

class TORCH_CUDA_CPP_API GreenContext {
 public:
  static std::unique_ptr<GreenContext> create(
    std::optional<uint32_t> device_id,
    std::optional<uint32_t> num_sms,
    std::optional<int32_t> workqueue_scope = std::nullopt,
    std::optional<uint32_t> workqueue_concurrency_limit = std::nullopt);

  static uint32_t max_workqueue_concurrency(
      std::optional<uint32_t> device_id = std::nullopt);

  ~GreenContext() noexcept;

  // Delete copy constructor and assignment
  GreenContext(const GreenContext&) = delete;
  GreenContext& operator=(const GreenContext&) = delete;

  // Make this context current
  void setContext();

  void popContext();

  CUDAStream Stream();

 private:
  GreenContext(
    uint32_t device_id,
    std::optional<uint32_t> num_sms,
    std::optional<int32_t> workqueue_scope,
    std::optional<uint32_t> workqueue_concurrency_limit);

  // Implement move operations
  GreenContext(GreenContext&& other) noexcept;
  GreenContext& operator=(GreenContext&& other) noexcept;

  int32_t device_id_ = -1;
  CUgreenCtx green_ctx_ = nullptr;
  CUcontext context_ = nullptr;
  cudaStream_t parent_stream_ = nullptr;
  std::array<CUstream, kStreamPerGreenContextPool> green_ctx_streams_;
  std::atomic<int32_t> curr_stream_idx_ = -1;
};
} // namespace at::cuda
