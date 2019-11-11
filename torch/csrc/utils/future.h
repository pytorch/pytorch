#pragma once

#include <ATen/core/ivalue.h>

namespace torch {

namespace utils {

// FutuerError inherits from std::exception, it can return const char* or
// std::string error message
struct TORCH_API FutureError final : public std::exception {
  FutureError(std::string&& error_msg_) : error_msg(std::move(error_msg_)) {}

  FutureError() = default;

  const char* what() const noexcept override {
    return error_msg.c_str();
  }

  std::string errMsg() const {
    return error_msg;
  }

  std::string error_msg;
};

// This class holds a value of type T that will be ready in the future.
// Most implementation is copied from FutureMessage and
// c10::ivalue::Future
template <typename T>
class TORCH_API Future final {
 public:
  using Callback = std::function<void(const T&, bool, const FutureError&)>;

  const T& wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    finished_cv_.wait(lock, [this] { return completed_.load(); });

    if (has_error_) {
      throw error_;
    }
    return value_;
  }

  void markCompleted(T value) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      TORCH_CHECK(!completed());
      completed_ = true;
      value_ = std::move(value);

      fireCallbacks();
    }
    finished_cv_.notify_all();
  }

  void markCompleted() {
    markCompleted(T());
  }

  void markCompleted(FutureError&& error) {
    std::unique_lock<std::mutex> lock(mutex_);
    AT_ASSERT(!completed());
    completed_ = true;
    has_error_ = true;
    error_ = std::move(error);

    fireCallbacks();
    finished_cv_.notify_all();
  }

  bool completed() const {
    return completed_;
  }

  T value() {
    std::unique_lock<std::mutex> lock(mutex_);
    AT_ASSERT(completed());
    if (has_error_) {
      throw error_;
    }
    return value_;
  }

  // If completed() the callback will be invoked in-place.
  void addCallback(const Callback& callback) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (completed()) {
      lock.unlock();
      callback(value_, has_error_, error_);
      return;
    }
    callbacks_.push_back(callback);
  }

 private:
  void fireCallbacks() {
    TORCH_CHECK(completed(), "Firing callbacks on incomplete Future.");
    // There is no need to protect callbacks_ with the lock.
    // Once completed_ is set to true, no one can add new callback to the list.
    // pass value_, has_error_, error_ for callback to easily check state.
    for (auto& callback : callbacks_) {
      callback(value_, has_error_, error_);
    }
    callbacks_.clear();
  }

  mutable std::mutex mutex_;
  std::atomic_bool completed_{false}; // is this future complete
  std::condition_variable finished_cv_;
  std::vector<Callback> callbacks_;
  T value_;
  bool has_error_ = false;
  FutureError error_;
};

}
} // namespace torch::utils
