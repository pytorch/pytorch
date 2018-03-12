#pragma once
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/partitioner.h>
#include <tbb/tbb.h>

namespace at {
namespace native {

constexpr size_t _THRESHOLD = 32768;

template <class T, template <class> class PRED>
T parallel_reduce(T (*f)(const T *, size_t, size_t, T), const T *data,
                  size_t start, size_t end, T init_) {
  T result_;
  static tbb::affinity_partitioner ap;
  if ((size_t)(end - start) < _THRESHOLD) {
    result_ = f(data, start, end, init_);
  } else {
    result_ = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(start, end, _THRESHOLD), init_,
        [&data, &f](const tbb::blocked_range<size_t> r, T init) -> T {
          return f(data, r.begin(), r.end(), init);
        },
        PRED<T>(), ap);
  }
  return result_;
}

template <class T>
void parallel_for_2d(void (*f)(const T *, T *, size_t, size_t), size_t num_rows,
                     size_t num_cols, size_t numel, const T *arr_, T *outarr_) {

  static tbb::affinity_partitioner ap;

  if (numel < _THRESHOLD) {
    for (size_t i_ = 0; i_ < numel / (num_rows * num_cols); i_++) {
      int64_t i = i_ * num_rows * num_cols;
      int64_t i_r = i_ * num_cols;
      const T *arr = arr_ + i;
      T *outarr = outarr_ + i_r;
      f(arr, outarr, num_rows, num_cols);
    }
  } else {
    size_t max_i_ = numel / (num_rows * num_cols);
    tbb::parallel_for(tbb::blocked_range2d<size_t, size_t>(
                          0, num_cols, _THRESHOLD, 0, max_i_, 1),
                      [&arr_, &outarr_, num_rows, num_cols,
                       &f](const tbb::blocked_range2d<size_t, size_t> r) {
                        for (size_t i_ = r.cols().begin(); i_ < r.cols().end();
                             i_++) {
                          int64_t i = i_ * num_rows * num_cols;
                          int64_t i_r = i_ * num_cols;
                          const T *arr = arr_ + i;
                          T *outarr = outarr_ + i_r;
                          f(arr, outarr + r.rows().begin(), num_rows,
                            r.rows().end() - r.rows().begin());
                        }
                      },
                      ap);
  }
}
}
}
