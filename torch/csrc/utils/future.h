#pragma once

#include <ATen/core/ivalue.h>

namespace torch {

namespace utils {

// FutureError inherits from std::exception, it can return const char* or
// std::string error message
class TORCH_API FutureError final : public std::exception {
 public:
  FutureError(std::string errorMsg) : errorMsg_(std::move(errorMsg)) {}

  FutureError() = default;

  const char* what() const noexcept override {
    return errorMsg_.c_str();
  }

 private:
  std::string errorMsg_;
};

// This class holds a value of type T that will be ready in the future.
// Most implementation is copied from FutureMessage and
// c10::ivalue::Future
template <typename T>
class TORCH_API Future final {
 public:
  using Callback =
      std::function<void(const T&, const c10::optional<FutureError>&)>;

  Future() = default;

  Future(T value) : completed_(true), value_(std::move(value)) {}

  const T& wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    finished_cv_.wait(lock, [this] { return completed_.load(); });

    if (error_) {
      throw *error_;
    }
    return value_;
  }

  const T& waitNoThrow() {
    std::unique_lock<std::mutex> lock(mutex_);
    finished_cv_.wait(lock, [this] { return completed_.load(); });
    return value_;
  }

  T&& moveValue() && {
    std::unique_lock<std::mutex> lock(mutex_);
    return std::move(value_);
  }

  void markCompleted(T value) {
    std::unique_lock<std::mutex> lock(mutex_);
    TORCH_CHECK(!completed_);
    // Set value first as completed_ is accessed without lock
    value_ = std::move(value);
    completed_ = true;

    // Move callbacks to a vector on the stack so we can access it without
    // holding a lock
    std::vector<Callback> cbs;
    cbs.swap(callbacks_);
    lock.unlock();
    finished_cv_.notify_all();
    // There is no need to protect callbacks_ with the lock.
    // Once completed_ is set to true, no one can add new callback to the
    // list. pass value_, error_ for callback to easily check state.
    for (auto& callback : cbs) {
      callback(value_, error_);
    }
  }

  // Sets error only if the future hasn't been marked completed already.
  // Useful in avoiding races where multiple threads try to setError
  // on a future.
  void setErrorIfNeeded(std::string errorMsg) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (completed_) {
      return;
    } else {
      setErrorInternal(std::move(errorMsg), lock);
    }
  }

  void setError(std::string errorMsg) {
    std::unique_lock<std::mutex> lock(mutex_);
    setErrorInternal(std::move(errorMsg), lock);
  }

  bool completed() const {
    return completed_;
  }

  bool hasError() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return error_ ? true : false;
  }

  c10::optional<FutureError> error() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return error_;
  }

  // If completed() the callback will be invoked in-place.
  void addCallback(const Callback& callback) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (completed_) {
      lock.unlock();
      callback(value_, error_);
      return;
    }
    callbacks_.push_back(callback);
  }

 private:
  void setErrorInternal(
      std::string errorMsg,
      std::unique_lock<std::mutex>& lock) {
    TORCH_CHECK(!completed_);
    error_ = FutureError(std::move(errorMsg));
    completed_ = true;

    // Move callbacks to a vector on the stack so we can access it without
    // holding a lock
    std::vector<Callback> cbs(std::move(callbacks_));
    lock.unlock();
    finished_cv_.notify_all();
    // There is no need to protect callbacks_ with the lock.
    // Once completed_ is set to true, no one can add new callback to the
    // list. pass value_, error_ for callback to easily check state.
    for (auto& callback : cbs) {
      callback(value_, error_);
    }
  }

  mutable std::mutex mutex_;
  std::atomic_bool completed_{false}; // is this future complete
  std::condition_variable finished_cv_;
  std::vector<Callback> callbacks_;
  T value_;
  c10::optional<FutureError> error_;
};

} // namespace utils
} // namespace torch
