#pragma once

#include <condition_variable>
#include <type_traits>

#include <ATen/ATen.h>
#include <c10d/ProcessGroup.hpp>
#include <torch/csrc/jit/python/pybind_utils.h>

namespace c10d {

// template <typename T>
class TORCH_API ProcessGroupFuture : public torch::jit::Future{
 public:
  explicit ProcessGroupFuture(std::shared_ptr<c10d::ProcessGroup::Work> work)
      : torch::jit::Future(c10::ListType::create(c10::TensorType::get())),
        work_(std::move(work)) {}

  // Override the wait method of Future and wait until the c10d ProcessGroup
  // work_ is completed. Once work_ has its result ready, copy the mark the
  // Future completed with that result.
  void wait() override {
    std::unique_lock<std::mutex> lock(mutex_);
    work_->wait();
    while (!work_->isCompleted()) {
      finished_cv_.wait(lock);
    }
    markCompleted(torch::jit::IValue(work_->result()));
  };

 private:
  std::shared_ptr<c10d::ProcessGroup::Work> work_;
};

} // namespace c10d
