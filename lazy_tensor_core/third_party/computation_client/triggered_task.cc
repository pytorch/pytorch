#include "lazy_tensors/computation_client/triggered_task.h"

namespace lazy_tensors {
namespace util {

TriggeredTask::TriggeredTask(std::function<void()> function, size_t num_threads)
    : function_(std::move(function)), running_(num_threads) {
  // We set running_ to num_threads because until the threads reach the
  // condition wait point (the cv_.wait() call) in the Runner() function, they
  // are effectively running.
  for (size_t i = 0; i < num_threads; ++i) {
    threads_.emplace_back(new std::thread([this]() { Runner(); }));
  }
}

void TriggeredTask::Stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_ = true;
  }
  run_cv_.notify_all();
  cv_.notify_all();
  for (auto& thread : threads_) {
    thread->join();
  }
}

size_t TriggeredTask::Activate() {
  bool notify = false;
  size_t run_id;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    notify = !activated_;
    activated_ = true;
    run_id = run_id_ + running_;
  }
  if (notify) {
    cv_.notify_one();
  }
  return run_id;
}

size_t TriggeredTask::WaitForRun(size_t run_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  ++run_waiters_;
  run_cv_.wait(lock, [this, run_id] { return run_id_ > run_id || stopped_; });
  --run_waiters_;
  return run_id_;
}

void TriggeredTask::Runner() {
  while (true) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      ++run_id_;
      if (run_waiters_ > 0) {
        run_cv_.notify_all();
      }
      --running_;
      cv_.wait(lock, [this] { return activated_ || stopped_; });
      if (stopped_) {
        break;
      }
      ++running_;
      activated_ = false;
    }
    function_();
  }
}

}  // namespace util
}  // namespace lazy_tensors
