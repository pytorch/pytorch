/*
 * A simple multi-producer, multi-consumer queue we rolled on our own.
 *
 * This is a wrapper around std::deque that provides
 * lock-free readIfNotEmpty and writeIfNotFull.
 */

#pragma once

#include <deque>
#include <mutex>
#include <type_traits>

namespace torch::nativert {

// TODO (zhxchen17) Add wrapper for concurrentqueue.
template <typename T>
class MPMCQueue {
  static_assert(!std::is_reference_v<T>);

 public:
  explicit MPMCQueue(size_t capacity) : capacity_(capacity) {}

  bool readIfNotEmpty(T& out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (storage_.empty()) {
      return false;
    }
    out = std::move(storage_.front());
    storage_.pop_front();
    return true;
  }

  bool writeIfNotFull(T&& in) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (storage_.size() == capacity_) {
      return false;
    }
    storage_.push_back(std::move(in));
    return true;
  }

 private:
  std::mutex mutex_;
  std::deque<T> storage_;
  const size_t capacity_;
};
} // namespace torch::nativert
