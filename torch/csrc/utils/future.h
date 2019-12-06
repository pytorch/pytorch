#pragma once

#include <ATen/core/ivalue.h>

namespace torch {

namespace utils {

// FutuerError inherits from std::exception, it can return const char* or
// std::string error message
class TORCH_API FutureError final : public std::exception {
public:
  FutureError(std::string&& errorMsg) : errorMsg_(std::move(errorMsg)) {}

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
  using Callback = std::function<void(const T&, const FutureError*)>;

  const T& wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    finished_cv_.wait(lock, [this] { return completed_.load(); });

    if (error_) {
      throw *error_;
    }
    return value_;
  }

  void markCompleted(T value) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      TORCH_CHECK(!completed());
      completed_ = true;
      value_ = std::move(value);

      std::vector<Callback> cbs;
      cbs.swap(callbacks_);
      lock.unlock();
      // There is no need to protect callbacks_ with the lock.
      // Once completed_ is set to true, no one can add new callback to the
      // list. pass value_, error_ for callback to easily check state.
      for (auto& callback : cbs) {
        callback(value_, error_.get());
      }
    }
    finished_cv_.notify_all();
  }

  void setError(std::string&& errorMsg) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      TORCH_CHECK(!completed());
      completed_ = true;
      error_ = c10::guts::make_unique<FutureError>(std::move(errorMsg));

      std::vector<Callback> cbs;
      cbs.swap(callbacks_);
      lock.unlock();
      // There is no need to protect callbacks_ with the lock.
      // Once completed_ is set to true, no one can add new callback to the
      // list. pass value_, error_ for callback to easily check state.
      for (auto& callback : cbs) {
        callback(value_, error_.get());
      }
    }
    finished_cv_.notify_all();
  }

  bool completed() const {
    return completed_;
  }

  // If completed() the callback will be invoked in-place.
  void addCallback(const Callback& callback) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (completed()) {
      lock.unlock();
      callback(value_, error_.get());
      return;
    }
    callbacks_.push_back(callback);
  }

 private:
  mutable std::mutex mutex_;
  std::atomic_bool completed_{false}; // is this future complete
  std::condition_variable finished_cv_;
  std::vector<Callback> callbacks_;
  T value_;
  std::unique_ptr<FutureError> error_;
};

}
} // namespace torch::utils
