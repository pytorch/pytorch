#pragma once

#include <torch/types.h>

#include <c10/util/Exception.h>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>

namespace torch {
namespace data {
namespace detail {

/// A basic locked, blocking MPMC queue.
///
/// Every `push` and `pop` is guarded by a mutex. A condition variable is used
/// to communicate insertion of new elements, such that waiting threads will be
/// woken up if they are currently waiting inside a call to `pop()`.
///
/// Note that this data structure is written specifically for use with the
/// `DataLoader`. Its behavior is tailored to this use case and may not be
/// applicable to more general uses.
template <typename T>
class Queue {
 public:
  /// Pushes a new value to the back of the `Queue` and notifies one thread on
  /// the waiting side about this event.
  void push(T value) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push(std::move(value));
    }
    cv_.notify_one();
  }

  /// Blocks until at least one element is ready to be popped from the front of
  /// the queue. An optional `timeout` in seconds can be used to limit the time
  /// spent waiting for an element. If the wait times out, an exception is
  /// raised.
  T pop(optional<std::chrono::milliseconds> timeout = nullopt) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (timeout) {
      if (!cv_.wait_for(
              lock, *timeout, [this] { return !this->queue_.empty(); })) {
        // clang-format off
        AT_ERROR(
            "Timeout in DataLoader queue while waiting for next batch"
            " (timeout was ", timeout->count(), " ms)");
        // clang-format on
      }
    } else {
      cv_.wait(lock, [this] { return !this->queue_.empty(); });
    }
    AT_ASSERT(!queue_.empty());
    T value = queue_.front();
    queue_.pop();
    lock.unlock();
    return value;
  }

  /// Empties the queue and returns the number of elements that were present at
  /// the start of the function. No threads are notified about this event as it
  /// is assumed to be used to drain the queue during shutdown of a
  /// `DataLoader`.
  size_t clear() {
    std::lock_guard<std::mutex> lock(this->mutex_);
    const auto size = queue_.size();
    while (!queue_.empty()) {
      queue_.pop();
    }
    return size;
  }

 private:
  std::queue<T> queue_;
  std::mutex mutex_;
  std::condition_variable cv_;
};
} // namespace detail
} // namespace data
} // namespace torch
