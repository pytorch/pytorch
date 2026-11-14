/**
 * This file is adapted from PyTorch/XLA
 * https://github.com/pytorch/xla/blob/e0e5f937a0ba8d904f9608137dc8c51ba439df2d/third_party/xla_client/metrics.h
 */

#pragma once

#include <functional>
#include <memory>
#include <thread>

#include <c10/macros/Export.h>

namespace torch::lazy {

// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class TORCH_API Completion {
 public:
  class Data;

  explicit Completion(std::shared_ptr<Data> data);

  ~Completion();

  void Wait();

 private:
  std::shared_ptr<Data> data_;
};

// Schedules a closure which might wait for IO or other events/conditions.
TORCH_API void ScheduleIoClosure(std::function<void()> closure);
TORCH_API Completion
ScheduleIoClosureWithCompletion(std::function<void()> closure);

} // namespace torch::lazy
