#pragma once
#include <ATen/ATen.h>

#include <cstddef>
#include <exception>

#include "tbb/tbb.h"

#define INTRA_OP_PARALLEL

namespace at {

template <class F>
void parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const F& f) {
  if (begin >= end) {
    return;
  }
  if (grain_size < 0) {
    throw std::runtime_error("Invalid begin, end or grain_size in parallel_for");
  }

  if ((end - begin) < grain_size || get_num_threads() == 1) {
    f(begin, end);
  } else {
    thread_local static tbb::affinity_partitioner ap;

    std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
    std::exception_ptr eptr;
    tbb::parallel_for(tbb::blocked_range<int64_t>(begin, end, grain_size),
      [f, &err_flag, &eptr](const tbb::blocked_range<int64_t>& r) {
        try {
          f(r.begin(), r.end());
        } catch (...) {
          if (!err_flag.test_and_set()) {
            eptr = std::current_exception();
          }
        }
      },
      ap
    );
    if (eptr) {
      std::rethrow_exception(eptr);
    }
  }
}

template <class scalar_t, class F, class SF>
scalar_t parallel_reduce(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const scalar_t ident,
    const F f,
    const SF sf) {
  if (begin >= end) {
    return ident;
  }
  if (grain_size < 0) {
    throw std::runtime_error("Invalid begin, end or grain_size in parallel_reduce");
  }

  if ((end - begin) < grain_size || get_num_threads() == 1) {
    return f(begin, end, ident);
  } else {
    std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
    std::exception_ptr eptr;
    scalar_t result = tbb::parallel_reduce(
      tbb::blocked_range<int64_t>(begin, end, grain_size), ident,
      [f, &err_flag, &eptr](const tbb::blocked_range<int64_t>& r, scalar_t ident) {
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
}

} // namespace at
