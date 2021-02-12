#pragma once

#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include <c10/util/Logging.h>
#include <c10/util/Optional.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

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
class TORCH_PYTHON_API Future final {
 public:
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

  // These constValue/moveValue accessors should only be used if
  // we know that the future is completed() with no error.
  const T& constValue() const {
    std::unique_lock<std::mutex> lock(mutex_);
    AT_ASSERT(completed_);
    return value_;
  }

  T&& moveValue() && {
    std::unique_lock<std::mutex> lock(mutex_);
    AT_ASSERT(completed_);
    return std::move(value_);
  }

  // Marks the future complete only if it hasn't been marked completed already.
  void markCompletedIfNeeded(T value) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (completed_) {
      LOG(INFO) << "markCompletedIfNeeded skipped since future is already complete.";
      return;
    } else {
      markCompletedInternal(std::move(value), lock);
    }
  }

  void markCompleted(T value) {
    std::unique_lock<std::mutex> lock(mutex_);
    markCompletedInternal(std::move(value), lock);
  }

  // Sets error only if the future hasn't been marked completed already.
  // Useful in avoiding races where multiple threads try to setError
  // on a future.
  void setErrorIfNeeded(FutureError error) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (completed_) {
      // This should be rare and shouldn't cause log spew. Its important to
      // log errors and thats why we have this log here.
      LOG (INFO) << "Skipping setting following error on the Future since " <<
        "it is already marked completed (this is not neccessarily an error): "
        << error.what();
      return;
    } else {
      setErrorInternal(std::move(error), lock);
    }
  }

  void setErrorIfNeeded(std::string errorMsg) {
    setErrorIfNeeded(FutureError(std::move(errorMsg)));
  }

  void setError(FutureError error) {
    std::unique_lock<std::mutex> lock(mutex_);
    setErrorInternal(std::move(error), lock);
  }

  void setError(std::string errorMsg) {
    setError(FutureError(std::move(errorMsg)));
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
  void addCallback(std::function<void(void)> cb) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (completed_) {
      lock.unlock();
      cb();
      return;
    }
    callbacks_.emplace_back(std::move(cb));
  }

  void addCallback(std::function<void(const Future<T>& future)> cb) {
    addCallback([this, cb = std::move(cb)]() { cb(*this); });
  }

 private:
  void setErrorInternal(
      FutureError error,
      std::unique_lock<std::mutex>& lock) {
    TORCH_CHECK(!completed_);
    error_ = std::move(error);
    completed_ = true;

    // Move callbacks to a vector on the stack so we can access it without
    // holding a lock
    std::vector<std::function<void(void)>> cbs(std::move(callbacks_));
    lock.unlock();
    finished_cv_.notify_all();
    // There is no need to protect callbacks_ with the lock.
    // Once completed_ is set to true, no one can add new callback to the
    // list. pass value_, error_ for callback to easily check state.
    for (auto& callback : cbs) {
      callback();
    }
  }

  void markCompletedInternal(T value,
      std::unique_lock<std::mutex>& lock) {
    TORCH_CHECK(!completed_);
    value_ = std::move(value);
    completed_ = true;

    // Move callbacks to a vector on the stack so we can access it without
    // holding a lock
    std::vector<std::function<void(void)>> cbs;
    cbs.swap(callbacks_);
    lock.unlock();
    finished_cv_.notify_all();
    // There is no need to protect callbacks_ with the lock.
    for (auto& callback : cbs) {
      callback();
    }
  }

  mutable std::mutex mutex_;
  std::atomic_bool completed_{false}; // is this future complete
  std::condition_variable finished_cv_;
  std::vector<std::function<void(void)>> callbacks_;
  T value_;
  c10::optional<FutureError> error_;
};

} // namespace utils
} // namespace torch
