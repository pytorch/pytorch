#pragma once

#include <atomic>

#include <c10/macros/Macros.h>

#include <torch/nativert/executor/ExecutionFrame.h>
#include <torch/nativert/graph/Graph.h>

namespace torch::nativert {

static_assert(
    alignof(std::uint_fast32_t) >=
    std::atomic_ref<std::uint_fast32_t>::required_alignment);

class SessionState {
 public:
  explicit SessionState(
      ExecutionFrame& frame,
      c10::FastMap<const Node*, std::uint_fast32_t> producers = {})
      : producers_(std::move(producers)), frame_(frame) {}

  C10_ALWAYS_INLINE void wait() {
    auto outstanding = workOutstanding_.load(std::memory_order_seq_cst);
    while (outstanding != 0) {
      workOutstanding_.wait(outstanding, std::memory_order_seq_cst);
      outstanding = workOutstanding_.load(std::memory_order_seq_cst);
    }
  }

  C10_ALWAYS_INLINE void addWork(uint32_t ct = 1) {
    workOutstanding_.fetch_add(ct, std::memory_order_seq_cst);
  }

  C10_ALWAYS_INLINE void removeWork() {
    if (workOutstanding_.fetch_sub(1, std::memory_order_seq_cst) == 1) {
      workOutstanding_.notify_one();
    }
  }

  C10_ALWAYS_INLINE ExecutionFrame& frame() {
    return frame_;
  }

  C10_ALWAYS_INLINE /* producersRemaining == 0 */ bool decrementProducers(
      const Node* node) {
    auto producerCount =
        std::atomic_ref<std::uint_fast32_t>(producers_.at(node));
    return producerCount.fetch_sub(1, std::memory_order_seq_cst) == 1;
  }

 private:
  std::atomic_uint_fast32_t workOutstanding_;
  c10::FastMap<const Node*, std::uint_fast32_t> producers_;

  ExecutionFrame& frame_;
};

} // namespace torch::nativert
