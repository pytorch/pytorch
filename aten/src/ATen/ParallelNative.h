#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>

#define INTRA_OP_PARALLEL

namespace at {
namespace internal {

// Run lambda function `fn` over `task_id` in [0, `range`) with threadpool.
// `fn` will be called with params: (thread_pool_task_id, task_id).
CAFFE2_API void _run_with_pool(const std::function<void(int, size_t)>& fn, size_t range);

// RAII guard helps to support in_parallel_region() and get_thread_num() API.
struct CAFFE2_API ParallelRegionGuard {
  ParallelRegionGuard(int64_t task_id);
  ~ParallelRegionGuard();
};

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
  size_t num_tasks, chunk_size;
  std::tie(num_tasks, chunk_size) =
      internal::calc_num_tasks_and_chunk_size(begin, end, grain_size);
  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
  std::exception_ptr eptr;
  std::vector<std::shared_ptr<c10::ivalue::Future>> futures(num_tasks);
  for (size_t i = 0; i < num_tasks; ++i) {
    futures[i] = std::make_shared<c10::ivalue::Future>(NoneType::get());
  }
  auto task = [f, &eptr, &err_flag, &futures, begin, end, chunk_size]
      (int idx, size_t task_id) {
    int64_t local_start = begin + task_id * chunk_size;
    if (local_start < end) {
      int64_t local_end = std::min(end, (int64_t)(chunk_size + local_start));
      try {
        internal::ParallelRegionGuard guard(task_id);
        f(local_start, local_end);
      } catch (...) {
        if (!err_flag.test_and_set()) {
          eptr = std::current_exception();
        }
      }
    }
    futures[task_id]->markCompleted();
  };
  internal::_run_with_pool(task, num_tasks);

  // Wait for all tasks to finish.
  for (size_t task_id = 0; task_id < num_tasks; ++task_id) {
    futures[task_id]->wait();
  }
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
  if ((end - begin) < grain_size || in_parallel_region()) {
    return f(begin, end, ident);
  }
  size_t num_tasks, chunk_size;
  std::tie(num_tasks, chunk_size) =
      internal::calc_num_tasks_and_chunk_size(begin, end, grain_size);
  std::vector<scalar_t> results(num_tasks);
  scalar_t* results_data = results.data();
  parallel_for(
    begin,
    end,
    grain_size,
    [f, results_data, ident](int64_t local_start, int64_t local_end) {
      results_data[get_thread_num()] = f(local_start, local_end, ident);
    }
  );
  scalar_t result = ident;
  for (auto partial_result : results) {
    result = sf(result, partial_result);
  }
  return result;
}

} // namespace at
