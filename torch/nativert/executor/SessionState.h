#pragma once

#include <atomic>

#include <c10/macros/Macros.h>

#include <torch/nativert/executor/ExecutionFrame.h>
#include <torch/nativert/graph/Graph.h>

namespace torch::nativert {

class SessionState {
 public:
  explicit SessionState(
      ExecutionFrame& frame,
      const c10::FastMap<const Node*, uint32_t>& producers)
      : producers_(producers.begin(), producers.end()), frame_(frame) {}

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

  // NOTE: removeWork may call notify_one() after wait() has already returned,
  // so the owner must keep the SessionState alive until every thread that can
  // call removeWork() has returned from it.
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
    return producers_.at(node).atomicRef().fetch_sub(
               1, std::memory_order_seq_cst) == 1;
  }

 private:
  // FBCODE uses folly::F14FastMap which requires values to be moveable.
  // This prevents us from using std::atomic<uint32_t> in c10::FastMap.
  struct AtomicRefableInt {
    using AtomicRef = std::atomic_ref<uint32_t>;
    alignas(AtomicRef::required_alignment) AtomicRef::value_type value;
    AtomicRef atomicRef() {
      return AtomicRef(value);
    }
  };

  std::atomic_uint32_t workOutstanding_{0};
  c10::FastMap<const Node*, AtomicRefableInt> producers_;

  ExecutionFrame& frame_;
};

} // namespace torch::nativert
