#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>

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
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return workOutstanding_.load() == 0; });
  }

  C10_ALWAYS_INLINE void addWork(uint32_t ct = 1) {
    workOutstanding_.fetch_add(ct);
  }

  C10_ALWAYS_INLINE void removeWork() {
    if (workOutstanding_.fetch_sub(1) == 1) {
      std::lock_guard<std::mutex> lock(mutex_);
      cv_.notify_one();
    }
  }

  C10_ALWAYS_INLINE ExecutionFrame& frame() {
    return frame_;
  }

  C10_ALWAYS_INLINE /* producersRemaining == 0 */ bool decrementProducers(
      const Node* node) {
    return producers_.at(node).fetch_sub(1) == 1;
  }

 private:
  // FBCODE uses folly::F14FastMap which requires values to be moveable.
  // This prevents us from using std::atomic_uint32_t in c10::FastMap.
#ifdef FBCODE_CAFFE2
  struct AtomicInt {
    uint32_t value;
    uint32_t fetch_sub(uint32_t x) {
      return __atomic_fetch_sub(&value, x, __ATOMIC_SEQ_CST);
    }
  };
#else
  using AtomicInt = std::atomic_uint32_t;
#endif

  std::atomic_uint32_t workOutstanding_{0};
  std::condition_variable cv_;
  std::mutex mutex_;
  c10::FastMap<const Node*, AtomicInt> producers_;

  ExecutionFrame& frame_;
};

} // namespace torch::nativert
