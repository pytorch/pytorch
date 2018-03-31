#pragma once
#include <ATen/ATen.h>
#include <tbb/tbb.h>
#include <cstddef>

namespace at {
namespace internal {
// This needs to be called before the first use of any algorithm such as
// parallel or it will have no effect and the default task scheduler is
// created which uses all available cores.
// See
// https://www.threadingbuildingblocks.org/docs/help/reference/task_scheduler/task_scheduler_init_cls.html
// This does not initializes the number of workers in the market (the overall
// of workers available to a process). It is merely a request to the market
// for a certain number of workers. If there are multiple threads making
// a request at the size of the maximum number of threads, they will
// be allocated a number proportional to the other requests.
void init_tbb_num_threads();
// This parameter is heuristically chosen to determine the minimum number of
// work that warrants paralellism. For example, when summing an array, it is
// deemed inefficient to parallelise over arrays shorter than 32768. Further,
// no parallel algorithm (such as parallel_reduce) should split work into
// smaller than GRAIN_SIZE chunks.
constexpr int64_t TBB_GRAIN_SIZE = 32768;
} // namespace internal

template <class T, template <class> class OP>
T parallel_reduce(
    T (*f)(const T*, size_t, size_t, T),
    const T* data,
    size_t start,
    size_t end,
    T init_) {
  internal::init_tbb_num_threads();

  T result_;
  static tbb::affinity_partitioner ap;
  if (end - start < internal::TBB_GRAIN_SIZE) {
    result_ = f(data, start, end, init_);
  } else {
    result_ = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(start, end, internal::TBB_GRAIN_SIZE),
        init_,
        [&data, &f](const tbb::blocked_range<size_t> r, T init) -> T {
          return f(data, r.begin(), r.end(), init);
        },
        OP<T>(),
        ap);
  }
  return result_;
}

template <class T>
void parallel_reduce_2d(
    void (*f)(const T*, T*, size_t, size_t),
    size_t num_rows,
    size_t num_cols,
    size_t numel,
    const T* arr_,
    T* outarr_) {
  internal::init_tbb_num_threads();

  static tbb::affinity_partitioner ap;

  size_t max_i_ =
      (numel && num_rows && num_cols) ? numel / (num_rows * num_cols) : 0;
  if (numel < internal::TBB_GRAIN_SIZE) {
    for (size_t i_ = 0; i_ < max_i_; i_++) {
      int64_t i = i_ * num_rows * num_cols;
      int64_t i_r = i_ * num_cols;
      const T* arr = arr_ + i;
      T* outarr = outarr_ + i_r;
      f(arr, outarr, num_rows, num_cols);
    }
  } else {
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, max_i_, 1),
        [&arr_, &outarr_, num_rows, num_cols, &f](
            const tbb::blocked_range<size_t> r) {
          for (size_t i_ = r.begin(); i_ < r.end(); i_++) {
            int64_t i = i_ * num_rows * num_cols;
            int64_t i_r = i_ * num_cols;
            const T* arr = arr_ + i;
            T* outarr = outarr_ + i_r;
            f(arr, outarr, num_rows, num_cols);
          }
        },
        ap);
  }
}

} // namespace at
