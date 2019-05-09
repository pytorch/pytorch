#pragma once
#include <ATen/ATen.h>

#include <cstddef>
#include <exception>

#ifdef _OPENMP
#define INTRA_OP_PARALLEL

#include <omp.h>
#endif

namespace at {

template <class F>
void parallel_for(
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
scalar_t parallel_reduce(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const scalar_t ident,
    const F f,
    const SF sf) {
  if (in_parallel_region() || get_num_threads() == 1) {
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
