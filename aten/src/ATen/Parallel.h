#pragma once
#include <ATen/ATen.h>
#include <atomic>
#include <cstddef>
#include <exception>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace at {
namespace internal {
// This parameter is heuristically chosen to determine the minimum number of
// work that warrants paralellism. For example, when summing an array, it is
// deemed inefficient to parallelise over arrays shorter than 32768. Further,
// no parallel algorithm (such as parallel_reduce) should split work into
// smaller than GRAIN_SIZE chunks.
constexpr int64_t GRAIN_SIZE = 32768;
} // namespace internal

inline int64_t divup(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

inline int get_max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

inline int get_thread_num() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

inline bool in_parallel_region() {
#ifdef _OPENMP
  return omp_in_parallel();
#else
  return false;
#endif
}

template <class F>
inline void parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const F& f) {
#ifdef _OPENMP
  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
  std::exception_ptr eptr;
#pragma omp parallel if (!omp_in_parallel() && ((end - begin) >= grain_size))
  {
    int64_t num_threads = omp_get_num_threads();
    int64_t tid = omp_get_thread_num();
    int64_t chunk_size = divup((end - begin), num_threads);
    int64_t begin_tid = begin + tid * chunk_size;
    if (begin_tid < end) {
      try {
        f(begin_tid, std::min(end, chunk_size + begin_tid));
      } catch (...) {
        if (!err_flag.test_and_set()) {
          eptr = std::current_exception();
        }
      }
    }
  }
  if (eptr) {
    std::rethrow_exception(eptr);
  }
#else
  if (begin < end) {
    f(begin, end);
  }
#endif
}

template <class scalar_t, class F, class SF>
inline scalar_t parallel_reduce(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const scalar_t ident,
    const F f,
    const SF sf) {
  if (get_num_threads() == 1) {
    return f(begin, end, ident);
  } else {
    const int64_t num_results = divup((end - begin), grain_size);
    std::vector<scalar_t> results(num_results);
    scalar_t* results_data = results.data();
#pragma omp parallel for if ((end - begin) >= grain_size)
    for (int64_t id = 0; id < num_results; id++) {
      int64_t i = begin + id * grain_size;
      results_data[id] = f(i, i + std::min(end - i, grain_size), ident);
    }
    return std::accumulate(
        results_data, results_data + results.size(), ident, sf);
  }
}

} // namespace at
