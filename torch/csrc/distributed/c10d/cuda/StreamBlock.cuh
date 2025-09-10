#pragma once

#include <chrono>

#include <ATen/core/Tensor.h>

#include <torch/csrc/distributed/c10d/cuda/StreamBlock.hpp>

namespace c10d::cuda::detail {

class StreamBlock : public ::c10d::cuda::StreamBlock {
 public:
  StreamBlock(std::chrono::milliseconds timeout);

  void abort() override {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    comm_[0] = 1;
  }

  StreamBlockStatus status() override {
    return static_cast<StreamBlockStatus>(comm_[1].item<int32_t>());
  }

 private:
  // (abort, cycles)
  const at::Tensor comm_;
  const std::chrono::milliseconds timeout_;
};

} // namespace c10d::cuda::detail
