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

  void markCompleted(T value) {
    std::unique_lock<std::mutex> lock(mutex_);
    TORCH_CHECK(!completed_);
    // Set value first as completed_ is accessed without lock
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

  // Sets error only if the future hasn't been marked completed already.
  // Useful in avoiding races where multiple threads try to setError
  // on a future.
  void setErrorIfNeeded(std::string errorMsg) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (completed_) {
      // This should be rare and shouldn't cause log spew. Its important to
      // log errors and thats why we have this log here.
      LOG (INFO) << "Skipping setting following error on the Future since " <<
        "it is already marked completed (this is not neccessarily an error): "
        << errorMsg;
      return;
    } else {
      setErrorInternal(FutureError(std::move(errorMsg)), lock);
    }
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
    }
    callbacks_.emplace_back(std::move(cb));
  }

  // Remove this once we've migrated underlying use-cases.
  void addCallback(const std::function<
                   void(const T&, const c10::optional<torch::utils::FutureError>&)>& cb) {
    addCallback([cb,this]() { cb(value_, error_); });
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

  mutable std::mutex mutex_;
  std::atomic_bool completed_{false}; // is this future complete
  std::condition_variable finished_cv_;
  std::vector<std::function<void(void)>> callbacks_;
  T value_;
  c10::optional<FutureError> error_;
};

} // namespace utils
} // namespace torch
