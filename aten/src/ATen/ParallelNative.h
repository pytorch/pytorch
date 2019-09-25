#pragma once
#include <ATen/ATen.h>

#include <c10/core/thread_pool.h>

#include <algorithm>
#include <cstddef>
#include <exception>

#define INTRA_OP_PARALLEL

namespace at {
namespace internal {
// internal function to get access to intra-op thread pool from
// template parallel primitives (parallel_for, parallel_reduce)
CAFFE2_API TaskThreadPoolBase& _get_intraop_pool();

// internal utility function to mark master thread as in parallel
// region when executing parallel primitives
CAFFE2_API void _set_in_parallel_region(bool);

// Simulate OMP's omp_get_thread_num() by force-setting thread local
// task id as thread number when executing parallel primitives
CAFFE2_API void _set_thread_num(size_t thread_num);
CAFFE2_API void _unset_thread_num();
}

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

  if (((end - begin) >= grain_size) && !in_parallel_region()) {
    // choose number of tasks based on grain size and number of threads
    size_t chunk_size = divup((end - begin), get_num_threads());
    // make sure each task is at least grain_size size
    chunk_size = std::max((size_t)grain_size, chunk_size);
    size_t num_tasks = divup((end - begin), chunk_size);

    std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
    std::exception_ptr eptr;
    auto task = [f, &eptr, &err_flag]
        (int64_t task_id, int64_t local_start, int64_t local_end) {
      internal::_set_thread_num(task_id);
      internal::_set_in_parallel_region(true);
      try {
        f(local_start, local_end);
      } catch (...) {
        if (!err_flag.test_and_set()) {
          eptr = std::current_exception();
        }
      }
      internal::_set_in_parallel_region(false);
      internal::_unset_thread_num();
    };

    std::vector<std::shared_ptr<c10::ivalue::Future>> futures(num_tasks);
    for (size_t i = 0; i < num_tasks; ++i) {
      futures[i] = std::make_shared<c10::ivalue::Future>(NoneType::get());
    }
    for (size_t task_id = 1; task_id < num_tasks; ++task_id) {
      int64_t local_start = begin + task_id * chunk_size;
      if (local_start < end) {
        int64_t local_end = std::min(end, (int64_t)(chunk_size + local_start));
        internal::_get_intraop_pool().run(
          // copy task_id, local_start, local_end
          [&task, &futures, task_id, local_start, local_end]() {
            task(task_id, local_start, local_end);
            futures[task_id]->markCompleted();
          }
        );
      } else {
        futures[task_id]->markCompleted();
      }
    }

    int64_t first_task_end = std::min(end, (int64_t)(chunk_size + begin));
    task(0, begin, first_task_end);
    // wait for all tasks to finish
    for (size_t task_id = 1; task_id < num_tasks; ++task_id) {
      futures[task_id]->wait();
    }
    if (eptr) {
      std::rethrow_exception(eptr);
    }
  } else {
    f(begin, end);
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

  if (((end - begin) >= grain_size) && !in_parallel_region()) {
    size_t chunk_size = divup((end - begin), get_num_threads());
    chunk_size = std::max((size_t)grain_size, chunk_size);
    size_t num_tasks = divup((end - begin), chunk_size);
    std::vector<scalar_t> results(num_tasks);
    scalar_t* results_data = results.data();

    std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
    std::exception_ptr eptr;
    auto task = [f, ident, results_data, &eptr, &err_flag]
        (int64_t task_id, int64_t local_start, int64_t local_end) {
      internal::_set_thread_num(task_id);
      internal::_set_in_parallel_region(true);
      try {
        results_data[task_id] = f(local_start, local_end, ident);
      } catch (...) {
        if (!err_flag.test_and_set()) {
          eptr = std::current_exception();
        }
      }
      internal::_set_in_parallel_region(false);
      internal::_unset_thread_num();
    };

    std::vector<std::shared_ptr<c10::ivalue::Future>> futures(num_tasks);
    for (size_t i = 0; i < num_tasks; ++i) {
      futures[i] = std::make_shared<c10::ivalue::Future>(NoneType::get());
    }
    for (size_t task_id = 1; task_id < num_tasks; ++task_id) {
      int64_t local_start = begin + task_id * chunk_size;
      if (local_start < end) {
        int64_t local_end = std::min(end, (int64_t)(chunk_size + local_start));
        internal::_get_intraop_pool().run(
          // copy task_id, local_start, local_end
          [&task, &futures, task_id, local_start, local_end]() {
            task(task_id, local_start, local_end);
            futures[task_id]->markCompleted();
          }
        );
      } else {
        futures[task_id]->markCompleted();
      }
    }

    int64_t first_task_end = std::min(end, (int64_t)(chunk_size + begin));
    task(0, begin, first_task_end);
    for (size_t task_id = 1; task_id < num_tasks; ++task_id) {
      futures[task_id]->wait();
    }
    if (eptr) {
      std::rethrow_exception(eptr);
    }

    scalar_t result = ident;
    for (auto partial_result : results) {
      result = sf(result, partial_result);
    }
    return result;
  } else {
    return f(begin, end, ident);
  }
}

} // namespace at
