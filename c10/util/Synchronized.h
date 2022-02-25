#pragma once

#include <mutex>

namespace c10 {

/**
 * A very simple Synchronization class for error-free use of data
 * in a multi-threaded context. See folly/docs/Synchronized.md for
 * the inspiration of this class.
 *
 * This class implements a small subset of the generic functionality
 * implemented by folly:Synchronized<T>.
 */
template <typename T>
class Synchronized final {
  mutable std::mutex mutex_;
  T data_;

 public:
  Synchronized() = default;
  Synchronized(T const& data) : data_(data) {}
  Synchronized(T&& data) : data_(data) {}

  Synchronized(Synchronized const&) = delete;
  Synchronized(Synchronized&&) = delete;
  Synchronized operator=(Synchronized const&) = delete;
  Synchronized operator=(Synchronized&&) = delete;

  template <typename CB>
  void withLock(CB cb) {
    std::lock_guard<std::mutex> guard(this->mutex_);
    cb(this->data_);
  }

  template <typename CB>
  void withLock(CB cb) const {
    std::lock_guard<std::mutex> guard(this->mutex_);
    cb(this->data_);
  }
};
} // end namespace c10
