// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/core/ivalue.h> // @manual=//caffe2:ATen-core
#include <c10/util/intrusive_ptr.h>
#include <chrono>
#include <functional>
#include <future>
#include <vector>

namespace torch::comms {

/**
 * TorchWork - Base class representing asynchronous work.
 *
 * Thread Safety:
 * TorchWork is NOT thread-safe. All methods (status(), isCompleted(), wait())
 * must be called from a single thread. Concurrent calls from multiple threads
 * are not supported.
 *
 * Work objects should not be destroyed while wait() is in progress.
 */
class TorchWork : public c10::intrusive_ptr_target {
 public:
  // Status of a work object
  enum class WorkStatus {
    NOT_STARTED, // Work has not started yet
    INPROGRESS, // Work is still in progress,
    COMPLETED, // Work has completed successfully
    TIMEDOUT, // Work has timed out
    ERROR // Work has encountered an error
  };

  TorchWork() = default;
  ~TorchWork() override = default;

  WorkStatus status() const {
    return status_.load(std::memory_order_relaxed);
  }
  bool isCompleted() const {
    return status() == WorkStatus::COMPLETED;
  }

  // Pure virtual functions that derived classes must implement
  virtual void wait() = 0;

  // Returns the timeout for this work object.
  // Derived classes with timeout support should override this.
  // Returns max() by default for work types that don't support timeout.
  virtual std::chrono::milliseconds getTimeout() const {
    return std::chrono::milliseconds::max();
  }

  // Fault Tolerance API

  /**
   * Block the CPU thread until the work is completed.
   * Unlike wait(), which blocks only the current CUDA stream, this method
   * blocks the CPU thread itself until the operation completes.
   *
   * @throws std::runtime_error if not implemented by the backend.
   */
  virtual void waitBlocking() {
    throw std::runtime_error(
        "[TorchWork]: waitBlocking not implemented for this work type");
  }

  // -- Work lifecycle hooks --
  //
  // These hooks allow external observers to track work object state
  // transitions without coupling to specific backend implementations.
  //
  // - Start hook:  fired when setStatus(INPROGRESS) is called
  // - End hook:    fired when setStatus(COMPLETED/ERROR/TIMEDOUT) is called
  // - Wait pre hook:  fired at the start of wait(), before the sync
  // - Wait post hook: fired at the end of wait(), after the sync
  //
  // Multiple hooks can be registered; they fire in registration order.
  // Hooks are NOT thread-safe -- register before concurrent status changes.

  using WorkHook = std::function<void()>;

  void registerWorkStartHook(WorkHook hook) {
    start_hooks_.push_back(std::move(hook));
  }

  void registerWorkEndHook(WorkHook hook) {
    // If work already reached a terminal state, fire the hook immediately
    // rather than enqueuing it (it would never fire otherwise).
    auto s = status();
    if (s == WorkStatus::COMPLETED || s == WorkStatus::ERROR ||
        s == WorkStatus::TIMEDOUT) {
      hook();
    } else {
      end_hooks_.push_back(std::move(hook));
    }
  }

  void registerWorkWaitPreHook(WorkHook hook) {
    wait_pre_hooks_.push_back(std::move(hook));
  }

  void registerWorkWaitPostHook(WorkHook hook) {
    wait_post_hooks_.push_back(std::move(hook));
  }

  // Disable copy and move semantics
  TorchWork(const TorchWork&) = delete;
  TorchWork& operator=(const TorchWork&) = delete;
  TorchWork(TorchWork&&) = delete;
  TorchWork& operator=(TorchWork&&) = delete;

 protected:
  void setStatus(WorkStatus status) {
    status_ = status;

    if (status == WorkStatus::INPROGRESS) {
      runStartHooks();
    } else if (
        status == WorkStatus::COMPLETED || status == WorkStatus::ERROR ||
        status == WorkStatus::TIMEDOUT) {
      runEndHooks();
    }
  }

  // Backend wait() implementations should call these around the actual wait.
  void runWaitPreHooks() {
    for (auto& hook : wait_pre_hooks_) {
      hook();
    }
  }

  void runWaitPostHooks() {
    for (auto& hook : wait_post_hooks_) {
      hook();
    }
  }

  friend class TorchComm;
  friend class WorkWrapper;

  virtual void markCompleted(
      c10::intrusive_ptr<c10::ivalue::Future> future_,
      std::vector<at::Tensor> outputTensors_);

  template <typename T, typename NullType>
  friend class c10::intrusive_ptr;

 private:
  void runStartHooks() {
    for (auto& hook : start_hooks_) {
      hook();
    }
  }

  void runEndHooks() {
    // Guard: end hooks fire at most once, even if setStatus is called
    // with multiple terminal states (e.g., ERROR then TIMEDOUT).
    if (end_hooks_fired_) {
      return;
    }
    end_hooks_fired_ = true;
    for (auto& hook : end_hooks_) {
      hook();
    }
  }

  // break weak-ref cycle: hooks registered via postHook() may capture a
  // weak_intrusive_ptr back to this object. after the strong refcount
  // reaches 0, release_resources() clears the hooks, destroying the weak
  // pointers and allowing the weak refcount to reach 0 so the object is
  // deleted.
  void release_resources() override {
    start_hooks_.clear();
    end_hooks_.clear();
    wait_pre_hooks_.clear();
    wait_post_hooks_.clear();
  }

  std::atomic<WorkStatus> status_{WorkStatus::NOT_STARTED};
  bool end_hooks_fired_{false};

  std::vector<WorkHook> start_hooks_;
  std::vector<WorkHook> end_hooks_;
  std::vector<WorkHook> wait_pre_hooks_;
  std::vector<WorkHook> wait_post_hooks_;
};

class TorchWorkCompleted : public TorchWork {
 public:
  TorchWorkCompleted();
  ~TorchWorkCompleted() override = default;

  // Override virtual functions from TorchWork
  void wait() override;

  void waitBlocking() override;
};

class TorchWorkThread : public TorchWork {
 public:
  explicit TorchWorkThread(std::function<void()> fn);
  ~TorchWorkThread() override = default;

  // Override virtual functions from TorchWork
  void wait() override;

 private:
  std::future<void> future_;
};

} // namespace torch::comms
