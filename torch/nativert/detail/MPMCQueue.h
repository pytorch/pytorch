/*
 * A simple thread-safe multi-producer, multi-consumer queue.
 *
 * This is a wrapper around std::deque that provides non-blocking
 * queue operations like readIfNotEmpty and writeIfNotFull using
 * std mutexes and the underlying queue can only be accessed
 * with synchronized sections.
 *
 * For now the goal is to provide a simple implementation that
 * works in all cases and produces no surprises to users.
 */

#pragma once

#include <deque>
#include <mutex>
#include <type_traits>

namespace torch::nativert::detail {

// TODO(zhxchen17) Add wrapper for concurrentqueue.
template <typename T>
class MPMCQueue {
  static_assert(!std::is_reference_v<T>);

 public:
  explicit MPMCQueue(size_t capacity) : capacity_(capacity) {}

  /**
   * Read from the queue if it is not empty.
   * @param out The value to read into.
   * @return true if the read succeeded, false if the queue is empty.
   */
  bool readIfNotEmpty(T& out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (storage_.empty()) {
      return false;
    }
    out = std::move(storage_.front());
    storage_.pop_front();
    return true;
  }

  /**
   * Write to the queue if it is not full.
   * @param in The value to write. For now we only support moveable types.
   * @return true if the write succeeded, false if the queue is full.
   */
  bool writeIfNotFull(T in) {
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
  size_t capacity_;
};
} // namespace torch::nativert::detail
