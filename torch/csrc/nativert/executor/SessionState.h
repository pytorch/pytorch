#pragma once

#include <atomic>

#include <c10/macros/Macros.h>

#include "torch/csrc/nativert/executor/ExecutionFrame.h"
#include "torch/csrc/nativert/graph/Graph.h"

namespace torch::nativert {

template <typename T, typename __atomic_base = std::atomic<T>>
struct __copyable_atomic : public __atomic_base {
 public:
  __copyable_atomic() = default;
  __copyable_atomic(const T& t) noexcept(__atomic_base::is_always_lock_free)
      : __atomic_base(t) {}
  __copyable_atomic(const __copyable_atomic& other) noexcept(
      __atomic_base::is_always_lock_free)
      : __atomic_base(other.load()) {}
  __copyable_atomic& operator=(const __copyable_atomic& other) noexcept(
      __atomic_base::is_always_lock_free) {
    this->store(other.load());
    return this;
  }
};

class SessionState {
 public:
  explicit SessionState(
      ExecutionFrame& frame,
      c10::FastMap<const Node*, __copyable_atomic<std::uint_fast32_t>>
          producers = {})
      : producers_(std::move(producers)), frame_(frame) {}

  C10_ALWAYS_INLINE void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&]() {
      return workOutstanding_.load(std::memory_order_seq_cst) == 0;
    });
  }

  C10_ALWAYS_INLINE void addWork(uint32_t ct = 1) {
    workOutstanding_.fetch_add(ct, std::memory_order_seq_cst);
  }

  C10_ALWAYS_INLINE void removeWork() {
    if (workOutstanding_.fetch_sub(1, std::memory_order_seq_cst) == 1) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.notify_one();
    }
  }

  C10_ALWAYS_INLINE ExecutionFrame& frame() {
    return frame_;
  }

  C10_ALWAYS_INLINE /* producersRemaining == 0 */ bool decrementProducers(
      const Node* node) {
    return producers_.at(node).fetch_sub(1, std::memory_order_seq_cst) == 1;
  }

  C10_ALWAYS_INLINE void setProducers(const Node* node, uint32_t v = 1) {
    producers_[node] += v;
  }

 private:
  std::atomic_uint_fast32_t workOutstanding_;
  c10::FastMap<const Node*, __copyable_atomic<std::uint_fast32_t>> producers_;

  std::condition_variable cv_;
  std::mutex mutex_;

  ExecutionFrame& frame_;
};

} // namespace torch::nativert
