#pragma once
#include <ATen/ATen.h>

#include <cstddef>
#include <exception>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif
#include "tbb/tbb.h"

#define INTRA_OP_PARALLEL

namespace at {

template <class F>
inline void parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const F& f) {
  TORCH_CHECK(grain_size >= 0);
  if (begin >= end) {
    return;
  }
  if ((end - begin) < grain_size || get_num_threads() == 1) {
    f(begin, end);
    return;
  }

  // Choose number of tasks based on grain size and number of threads.
  int64_t chunk_size = divup((end - begin), get_num_threads());
  // Make sure each task is at least grain_size size.
  chunk_size = std::max(grain_size, chunk_size);

  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
  std::exception_ptr eptr;
  tbb::parallel_for(tbb::blocked_range<int64_t>(begin, end, chunk_size),
    [&eptr, &err_flag, f](const tbb::blocked_range<int64_t>& r) {
      try {
        f(r.begin(), r.end());
      } catch (...) {
        if (!err_flag.test_and_set()) {
          eptr = std::current_exception();
        }
      }
    });
  if (eptr) {
    std::rethrow_exception(eptr);
  }
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
  if ((end - begin) < grain_size || get_num_threads() == 1) {
    return f(begin, end, ident);
  }

  // Choose number of tasks based on grain size and number of threads.
  int64_t chunk_size = divup((end - begin), get_num_threads());
  // Make sure each task is at least grain_size size.
  chunk_size = std::max(grain_size, chunk_size);

  scalar_t result;
  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
  std::exception_ptr eptr;
  result = tbb::parallel_reduce(
    tbb::blocked_range<int64_t>(begin, end, chunk_size), ident,
    [&eptr, &err_flag, f]
        (const tbb::blocked_range<int64_t>& r, scalar_t ident) {
      try {
        return f(r.begin(), r.end(), ident);
      } catch (...) {
        if (!err_flag.test_and_set()) {
          eptr = std::current_exception();
        }
        return ident;
      }
    },
    sf
  );
  if (eptr) {
    std::rethrow_exception(eptr);
  }
  return result;
}

template<typename F0, typename F1>
void intraop_invoke(const F0& f0, const F1& f1) {
  tbb::parallel_invoke(f0, f1);
}

} // namespace at
