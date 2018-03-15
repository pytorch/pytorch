#pragma once
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/partitioner.h>
#include <tbb/tbb.h>

namespace at {
namespace native {

// This parameter is heuristically chosen to determine the minimum number of
// work that warrants paralellism. For example, when summing an array, it is
// deemed inefficient to parallelise over arrays shorter than 32768. Further,
// no parallel algorithm (such as parallel_reduce) should split work into
// smaller than GRAIN_SIZE chunks.
constexpr size_t GRAIN_SIZE = 32768;

template <class T, template <class> class OP>
T parallel_reduce(T (*f)(const T *, size_t, size_t, T), const T *data,
                  size_t start, size_t end, T init_) {
  T result_;
  static tbb::affinity_partitioner ap;
  if ((size_t)(end - start) < GRAIN_SIZE) {
    result_ = f(data, start, end, init_);
  } else {
    result_ = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(start, end, GRAIN_SIZE), init_,
        [&data, &f](const tbb::blocked_range<size_t> r, T init) -> T {
          return f(data, r.begin(), r.end(), init);
        },
        OP<T>(), ap);
  }
  return result_;
}

template <class T>
void parallel_for_2d(void (*f)(const T *, T *, size_t, size_t), size_t num_rows,
                     size_t num_cols, size_t numel, const T *arr_, T *outarr_) {

  static tbb::affinity_partitioner ap;

  size_t max_i_ =
      (numel && num_rows && num_cols) ? numel / (num_rows * num_cols) : 0;
  if (numel < GRAIN_SIZE) {
    for (size_t i_ = 0; i_ < max_i_; i_++) {
      int64_t i = i_ * num_rows * num_cols;
      int64_t i_r = i_ * num_cols;
      const T *arr = arr_ + i;
      T *outarr = outarr_ + i_r;
      f(arr, outarr, num_rows, num_cols);
    }
  } else {
    tbb::parallel_for(tbb::blocked_range2d<size_t, size_t>(
                          0, num_cols, GRAIN_SIZE, 0, max_i_, 1),
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
