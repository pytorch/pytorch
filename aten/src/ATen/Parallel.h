#pragma once
#include <ATen/ATen.h>
#include <cstddef>
#include <tbb/tbb.h>


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
AT_API void init_tbb_num_threads();
// This parameter is heuristically chosen to determine the minimum number of
// work that warrants paralellism. For example, when summing an array, it is
// deemed inefficient to parallelise over arrays shorter than 32768. Further,
// no parallel algorithm (such as parallel_reduce) should split work into
// smaller than GRAIN_SIZE chunks.
constexpr int64_t TBB_GRAIN_SIZE = 32768;
} // namespace internal

template <class F>
void parallel_for(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    F f) {
  internal::init_tbb_num_threads();

#ifdef __PPC64__
  using default_partitioner_type = tbb::simple_partitioner;
#else
  using default_partitioner_type = tbb::affinity_partitioner;
#endif

  thread_local static default_partitioner_type ap;

  if ((end - begin) < grain_size) {
    f(begin, end);
  } else {
    tbb::parallel_for(
        tbb::blocked_range<int64_t>(begin, end, grain_size),
        [f](const tbb::blocked_range<int64_t>& r) { f(r.begin(), r.end()); },
        ap);
  }
}

} // namespace at
