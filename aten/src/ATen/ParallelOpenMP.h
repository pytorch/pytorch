#pragma once

#include <cstddef>
#include <exception>

#include <c10/util/SmallVector.h>

#ifdef _OPENMP
#define INTRA_OP_PARALLEL

#include <omp.h>
#endif

namespace at {

#ifdef _OPENMP
namespace internal {
template <typename F>
inline void invoke_parallel(int64_t begin, int64_t end, int64_t grain_size, const F& f) {
  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
  std::exception_ptr eptr;

#pragma omp parallel
  {
    // choose number of tasks based on grain size and number of threads
    // can't use num_threads clause due to bugs in GOMP's thread pool (See #32008)
    int64_t num_threads = omp_get_num_threads();
    if (grain_size > 0) {
      num_threads = std::min(num_threads, divup((end - begin), grain_size));
    }

    int64_t tid = omp_get_thread_num();
    int64_t chunk_size = divup((end - begin), num_threads);
    int64_t begin_tid = begin + tid * chunk_size;
    if (begin_tid < end) {
      try {
        internal::ThreadIdGuard tid_guard(tid);
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
}
} // namespace internal
#endif // _OPENMP


template <class F>
inline void parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const F& f) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grain_size >= 0);
  if (begin >= end) {
    return;
  }

#ifdef _OPENMP
  at::internal::lazy_init_num_threads();
  const auto numiter = end - begin;
  const bool use_parallel = (
    numiter > grain_size && numiter > 1 &&
    omp_get_max_threads() > 1 && !omp_in_parallel());
  if (!use_parallel) {
    internal::ThreadIdGuard tid_guard(0);
    f(begin, end);
    return;
  }

  internal::invoke_parallel(begin, end, grain_size, f);
#else
  internal::ThreadIdGuard tid_guard(0);
  f(begin, end);
#endif
}

template <class scalar_t, class F, class SF>
inline scalar_t parallel_reduce(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const scalar_t ident,
    const F& f,
    const SF& sf) {
  TORCH_CHECK(grain_size >= 0);
  if (begin >= end) {
    return ident;
  }

#ifdef _OPENMP
  at::internal::lazy_init_num_threads();
  const bool use_parallel = (
      (end - begin) <= grain_size ||
      in_parallel_region() ||
      get_num_threads() == 1);
  if (!use_parallel) {
    internal::ThreadIdGuard tid_guard(0);
    return f(begin, end, ident);
  }

  c10::SmallVector<scalar_t, 64> results(at::get_num_threads(), ident);
  internal::invoke_parallel(begin, end, grain_size,
    [&](const int64_t my_begin, const int64_t my_end) {
      const auto tid = at::get_thread_num();
      results[tid] = f(my_begin, my_end, ident);
    }
  );

  scalar_t result = ident;
  for (auto partial_result : results) {
    result = sf(result, partial_result);
  }
  return result;
#else
  internal::ThreadIdGuard tid_guard(0);
  return f(begin, end, ident);
#endif
}

} // namespace at
