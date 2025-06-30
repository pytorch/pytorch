#pragma once

#include <chrono>

#include <ATen/core/Tensor.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/ops/empty.h>

#include <torch/csrc/distributed/c10d/cuda/Baton.h>

namespace c10d::cuda::detail {

class Baton : public ::c10d::cuda::Baton {
 public:
  Baton(std::chrono::milliseconds timeout);

  void abort() override {
    comm_[0] = 1;
  }

  BatonStatus status() override {
    return static_cast<BatonStatus>(comm_[1].item<int32_t>());
  }

 private:
  // (abort, cycles)
  const at::Tensor comm_;
  const std::chrono::milliseconds timeout_;
};

} // namespace c10d::cuda::detail
