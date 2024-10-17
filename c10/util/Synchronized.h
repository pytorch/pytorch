#pragma once

#include <mutex>

namespace c10 {

/**
 * A very simple Synchronization class for error-free use of data
 * in a multi-threaded context. See folly/docs/Synchronized.md for
 * the inspiration of this class.
 *
 * Full URL:
 * https://github.com/facebook/folly/blob/main/folly/docs/Synchronized.md
 *
 * This class implements a small subset of the generic functionality
 * implemented by folly:Synchronized<T>. Specifically, only withLock<T>
 * is implemented here since it's the smallest possible API that is
 * able to cover a large surface area of functionality offered by
 * folly::Synchronized<T>.
 */
template <typename T>
class Synchronized final {
  mutable std::mutex mutex_;
  T data_;

 public:
  Synchronized() = default;
  Synchronized(T const& data) : data_(data) {}
  Synchronized(T&& data) : data_(std::move(data)) {}

  // Don't permit copy construction, move, assignment, or
  // move assignment, since the underlying std::mutex
  //  isn't necessarily copyable/moveable.
  Synchronized(Synchronized const&) = delete;
  Synchronized(Synchronized&&) = delete;
  Synchronized operator=(Synchronized const&) = delete;
  Synchronized operator=(Synchronized&&) = delete;

  /**
   * To use, call withLock<T> with a callback that accepts T either
   * by copy or by reference. Use the protected variable in the
   * provided callback safely.
   */
  template <typename CB>
  auto withLock(CB&& cb) {
    std::lock_guard<std::mutex> guard(this->mutex_);
    return std::forward<CB>(cb)(this->data_);
  }

  /**
   * To use, call withLock<T> with a callback that accepts T either
   * by copy or by const reference. Use the protected variable in
   * the provided callback safely.
   */
  template <typename CB>
  auto withLock(CB&& cb) const {
    std::lock_guard<std::mutex> guard(this->mutex_);
    return std::forward<CB>(cb)(this->data_);
  }
};
} // end namespace c10
