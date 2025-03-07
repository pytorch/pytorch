#pragma once

#include <cstddef>

// To use moodycamel semaphore, we need to include the header file
// for concurrentqueue first. Hiding implementation detail here.
#ifdef BLOCK_SIZE
#pragma push_macro("BLOCK_SIZE")
#undef BLOCK_SIZE
#include "torch/csrc/runtime/common/concurrentqueue.h"
#pragma pop_macro("BLOCK_SIZE")
#else
#include "torch/csrc/runtime/common/concurrentqueue.h"
#endif

#include "torch/csrc/runtime/common/lightweightsemaphore.h"

namespace torch::runtime {

// A textbook semaphore implementation. Nothing fancy.
// In the future, we can consider using C++20's semaphore.
class Semaphore {
  moodycamel::LightweightSemaphore impl_;

 public:
  void release() {
    impl_.signal();
  }

  void release(size_t n) {
    impl_.signal(n);
  }

  void acquire() {
    impl_.wait();
  }
};
} // namespace torch::runtime
