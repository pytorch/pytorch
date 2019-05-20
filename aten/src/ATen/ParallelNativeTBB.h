#pragma once
#include <ATen/ATen.h>

#include <cstddef>
#include <exception>

#include "tbb/tbb.h"

#define INTRA_OP_PARALLEL

namespace at {

namespace internal {
tbb::task_arena& _get_arena();
}

template <class F>
inline void parallel_for(
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
    tbb::task_group tg;
    internal::_get_arena().execute([&tg, begin, end, grain_size, f](){
      tg.run([begin, end, grain_size, f]() {
        tbb::parallel_for(tbb::blocked_range<int64_t>(begin, end, grain_size),
          [&](const tbb::blocked_range<int64_t>& r) {
            f(r.begin(), r.end());
          });
      });
    });
    tg.wait();
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
  if (begin >= end) {
    return ident;
  }
  if (grain_size < 0) {
    throw std::runtime_error("Invalid begin, end or grain_size in parallel_reduce");
  }

  if ((end - begin) < grain_size || get_num_threads() == 1) {
    return f(begin, end, ident);
  } else {
    scalar_t result;
    tbb::task_group tg;
    internal::_get_arena().execute(
        [&tg, &result, begin, end, grain_size, ident, f, sf]() {
      tg.run([&result, begin, end, grain_size, ident, f, sf]() {
        result = tbb::parallel_reduce(
          tbb::blocked_range<int64_t>(begin, end, grain_size), ident,
          [&](const tbb::blocked_range<int64_t>& r, scalar_t ident) {
            return f(r.begin(), r.end(), ident);
          },
          sf
        );
      });
    });
    tg.wait();
    return result;
  }
}

} // namespace at
