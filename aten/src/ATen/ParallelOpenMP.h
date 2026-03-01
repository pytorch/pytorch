#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>

#ifdef _OPENMP
#define INTRA_OP_PARALLEL

#include <omp.h>
#endif

#ifdef _OPENMP
namespace at::internal {
template <typename F>
inline void invoke_parallel(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    const F& f) {
  if (end <= begin)
    return;

  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
  std::exception_ptr eptr;

  // choose number of tasks based on grain size and number of threads
  // can't use num_threads clause due to bugs in GOMP's thread pool (See
  // #32008)
  int64_t num_threads = omp_get_max_threads();
  if (grain_size > 0) {
    num_threads = std::min(num_threads, divup((end - begin), grain_size));
  }
  num_threads = std::max<int64_t>(1, num_threads);

  std::atomic<int64_t> tid_counter{0};
  std::atomic<int64_t> next{begin};
  int64_t chunk_size = std::max<int64_t>(
      grain_size > 0 ? grain_size : 1, divup((end - begin), num_threads));

#pragma omp parallel
  {
    int64_t tid = tid_counter.fetch_add(1, std::memory_order_relaxed);
    if (tid < num_threads) {
      internal::ThreadIdGuard tid_guard(tid);
      for (;;) {
        int64_t b = next.fetch_add(chunk_size, std::memory_order_relaxed);
        if (b >= end)
          break;
        try {
          f(b, std::min(end, b + chunk_size));
        } catch (...) {
          if (!err_flag.test_and_set()) {
            eptr = std::current_exception();
          }
          break;
        }
      }
    }
  }
  if (eptr) {
    std::rethrow_exception(eptr);
  }
}
} // namespace at::internal
#endif // _OPENMP
