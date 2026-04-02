/**
 * This file is adapted from PyTorch/XLA
 * https://github.com/pytorch/xla/blob/e0e5f937a0ba8d904f9608137dc8c51ba439df2d/third_party/xla_client/multi_wait.h
 */

#pragma once

#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>

#include <c10/macros/Export.h>

namespace torch::lazy {

// Support waiting for a number of tasks to complete.
class TORCH_API MultiWait {
 public:
  explicit MultiWait(size_t count) : count_(count) {}

  // Signal the completion of a single task.
  void Done();

  // Waits until at least count (passed as constructor value) completions
  // happened.
  void Wait();

  // Same as above, but waits up to wait_seconds.
  void Wait(double wait_seconds);

  // Resets the threshold counter for the MultiWait object. The completed count
  // is also reset to zero.
  void Reset(size_t count);

  // Creates a completer functor which signals the mult wait object once func
  // has completed. Handles exceptions by signaling the multi wait with the
  // proper status value. This API returns a function which captures a MultiWait
  // reference, so care must be taken such that the reference remains valid for
  // the whole lifetime of the returned function.
  std::function<void()> Completer(std::function<void()> func);

  // Similar as the above API, but with explicit capture of the MultiWait shared
  // pointer.
  static std::function<void()> Completer(
      std::shared_ptr<MultiWait> mwait,
      std::function<void()> func);

 private:
  void Complete(const std::function<void()>& func);

  std::mutex mutex_;
  std::condition_variable cv_;
  size_t count_ = 0;
  size_t completed_count_ = 0;
  std::exception_ptr exptr_;
};

} // namespace torch::lazy
