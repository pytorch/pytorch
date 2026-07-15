#pragma once

#include <version>

/*
  a simple semaphore interface.
*/

// note: __cpp_lib_semaphore will not be defined in some apple platforms
// even if >= C++20.
#if __has_include(<semaphore>) && defined(__cpp_lib_semaphore) && __cpp_lib_semaphore >= 201907L
#define C10_SEMAPHORE_USE_STL
#endif

#ifdef C10_SEMAPHORE_USE_STL
#include <semaphore>
#else
// To use moodycamel semaphore, we need to include the header file
// for concurrentqueue first. Hiding implementation detail here.
#ifdef BLOCK_SIZE
#pragma push_macro("BLOCK_SIZE")
#undef BLOCK_SIZE
#include <moodycamel/concurrentqueue.h> // @manual
#pragma pop_macro("BLOCK_SIZE")
#else
#include <moodycamel/concurrentqueue.h> // @manual
#endif

#include <moodycamel/lightweightsemaphore.h> // @manual
#endif

namespace c10 {

class Semaphore {
 public:
  Semaphore(int32_t initial_count = 0) : impl_(initial_count) {}

  void release(int32_t n = 1) {
#ifdef C10_SEMAPHORE_USE_STL
    impl_.release(n);
#else
    impl_.signal(n);
#endif
  }

  void acquire() {
#ifdef C10_SEMAPHORE_USE_STL
    impl_.acquire();
#else
    impl_.wait();
#endif
  }

  bool tryAcquire() {
#ifdef C10_SEMAPHORE_USE_STL
    return impl_.try_acquire();
#else
    return impl_.tryWait();
#endif
  }

 private:
#ifdef C10_SEMAPHORE_USE_STL
  std::counting_semaphore<> impl_;
#else
  moodycamel::LightweightSemaphore impl_;
#endif
};
} // namespace c10

#undef C10_SEMAPHORE_USE_STL
