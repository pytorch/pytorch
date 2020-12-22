#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>

#define INTRA_OP_PARALLEL

namespace at {
namespace internal {

inline std::tuple<size_t, size_t> calc_num_tasks_and_chunk_size(
    int64_t begin, int64_t end, int64_t grain_size) {
  if ((end - begin) < grain_size) {
    return std::make_tuple(1, std::max((int64_t)0, end - begin));
  }
  // Choose number of tasks based on grain size and number of threads.
  size_t chunk_size = divup((end - begin), get_num_threads());
  // Make sure each task is at least grain_size size.
  chunk_size = std::max((size_t)grain_size, chunk_size);
  size_t num_tasks = divup((end - begin), chunk_size);
  return std::make_tuple(num_tasks, chunk_size);
}

TORCH_API void _parallel_run(
  const int64_t begin,
  const int64_t end,
  const int64_t grain_size,
  const std::function<void(int64_t, int64_t, size_t)>& f);

} // namespace internal

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
  if ((end - begin) < grain_size || in_parallel_region()) {
    f(begin, end);
    return;
  }
  internal::_parallel_run(
      begin,
      end,
      grain_size,
      [f](int64_t start, int64_t end, size_t /* unused */) {
        f(start, end);
      }
  );
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
  if ((end - begin) < grain_size || in_parallel_region()) {
    return f(begin, end, ident);
  }
  size_t num_tasks, chunk_size;
  std::tie(num_tasks, chunk_size) =
      internal::calc_num_tasks_and_chunk_size(begin, end, grain_size);
  std::vector<scalar_t> results(num_tasks);
  scalar_t* results_data = results.data();
  internal::_parallel_run(
      begin,
      end,
      grain_size,
      [f, ident, results_data](int64_t start, int64_t end, size_t task_id) {
        results_data[task_id] = f(start, end, ident);
      }
  );
  scalar_t result = ident;
  for (auto partial_result : results) {
    result = sf(result, partial_result);
  }
  return result;
}

} // namespace at
