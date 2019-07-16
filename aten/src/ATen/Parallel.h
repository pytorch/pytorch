#pragma once
#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>

namespace at {
namespace internal {
// This parameter is heuristically chosen to determine the minimum number of
// work that warrants parallelism. For example, when summing an array, it is
// deemed inefficient to parallelise over arrays shorter than 32768. Further,
// no parallel algorithm (such as parallel_reduce) should split work into
// smaller than GRAIN_SIZE chunks.
constexpr int64_t GRAIN_SIZE = 32768;
} // namespace internal

inline int64_t divup(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

// Called during new thread initialization
CAFFE2_API void init_num_threads();

// Sets the number of threads to be used in parallel region
CAFFE2_API void set_num_threads(int);

// Returns the number of threads used in parallel region
CAFFE2_API int get_num_threads();

// Returns the current thread number (starting from 0)
// in the current parallel region, or 0 in the sequential region
CAFFE2_API int get_thread_num();

// Checks whether the code runs in parallel region
CAFFE2_API bool in_parallel_region();

/*
parallel_for

begin: index at which to start applying user function

end: index at which to stop applying user function

grain_size: number of elements per chunk. impacts the degree of parallelization

f: user function applied in parallel to the chunks, signature:
  void f(int64_t begin, int64_t end)
*/
template <class F>
inline void parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const F& f);

/*
parallel_reduce

begin: index at which to start applying reduction

end: index at which to stop applying reduction

grain_size: number of elements per chunk. impacts number of elements in
intermediate results tensor and degree of parallelization.

ident: identity for binary combination function sf. sf(ident, x) needs to return
x.

f: function for reduction over a chunk. f needs to be of signature scalar_t
f(int64_t partial_begin, int64_t partial_end, scalar_t identifiy)

sf: function to combine two partial results. sf needs to be of signature
scalar_t sf(scalar_t x, scalar_t y)

For example, you might have a tensor of 10000 entires and want to sum together
all the elements. Parallel_reduce with a grain_size of 2500 will then allocate
an intermediate result tensor with 4 elements. Then it will execute the function
"f" you provide and pass the beginning and end index of these chunks, so
0-2499, 2500-4999, etc. and the combination identity. It will then write out
the result from each of these chunks into the intermediate result tensor. After
that it'll reduce the partial results from each chunk into a single number using
the combination function sf and the identity ident. For a total summation this
would be "+" and 0 respectively. This is similar to tbb's approach [1], where
you need to provide a function to accumulate a subrange, a function to combine
two partial results and an identity.

[1] https://software.intel.com/en-us/node/506154
*/
template <class scalar_t, class F, class SF>
inline scalar_t parallel_reduce(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const scalar_t ident,
    const F& f,
    const SF& sf);

// Returns a detailed string describing parallelization settings
CAFFE2_API std::string get_parallel_info();

// Sets number of threads used for inter-op parallelism
CAFFE2_API void set_num_interop_threads(int);

// Returns the number of threads used for inter-op parallelism
CAFFE2_API int get_num_interop_threads();

// Launches inter-op parallel task
CAFFE2_API void launch(std::function<void()> func);

// Launches intra-op parallel task
CAFFE2_API void intraop_launch(std::function<void()> func);

// Launches intra-op parallel task, returns a future
CAFFE2_API std::shared_ptr<c10::ivalue::Future> intraop_launch_future(
    std::function<void()> func);

// Returns number of intra-op threads used by default
CAFFE2_API int intraop_default_num_threads();

} // namespace at

#if AT_PARALLEL_OPENMP
#include <ATen/ParallelOpenMP.h>
#elif AT_PARALLEL_NATIVE
#include <ATen/ParallelNative.h>
#elif AT_PARALLEL_NATIVE_TBB
#include <ATen/ParallelNativeTBB.h>
#endif
