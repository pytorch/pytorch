#pragma once

#include <atomic>
#include <cstddef>
#include <exception>

#include <c10/util/Exception.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif
#include <tbb/tbb.h>

#define INTRA_OP_PARALLEL

namespace at::internal {

template <typename F>
inline void invoke_parallel(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const F& f) {
  // Choose number of tasks based on grain size and number of threads.
  int64_t chunk_size = divup((end - begin), get_num_threads());
  // Make sure each task is at least grain_size size.
  chunk_size = std::max(grain_size, chunk_size);

  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
  std::exception_ptr eptr;
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(begin, end, chunk_size),
      [&eptr, &err_flag, f](const tbb::blocked_range<int64_t>& r) {
        try {
          internal::ThreadIdGuard tid_guard(
              tbb::this_task_arena::current_thread_index());
          f(r.begin(), r.end());
        } catch (...) {
          if (!err_flag.test_and_set()) {
            eptr = std::current_exception();
          }
        }
      },
      tbb::static_partitioner{});
  if (eptr) {
    std::rethrow_exception(eptr);
  }
}

} // namespace at::internal
