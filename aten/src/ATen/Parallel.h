#pragma once
#include <ATen/ATen.h>
#include <cstddef>

#if defined(CPU_THREADPOOL_TBB)
#include <tbb/tbb.h>
#elif defined(CPU_THREADPOOL_OPENMP)
#include <omp.h>
#endif


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

inline int64_t divup(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

template <class F>
inline void parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const F f) {
#if defined(CPU_THREADPOOL_TBB)
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
#elif defined(CPU_THREADPOOL_OPENMP)
#pragma omp parallel for if ((end - begin) >= grain_size)
  for (int64_t i = begin; i < end; i += grain_size) {
    f(i, i + std::min(end - i, grain_size));
  }
#else
#error("Specified CPU threadpool not supported")
#endif
}

template <class scalar_t, class F, class SF>
inline scalar_t parallel_reduce(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const scalar_t ident,
    const F f,
    const SF sf) {
  internal::init_tbb_num_threads();

#if defined(CPU_THREADPOOL_TBB)
#ifdef __PPC64__
  using default_partitioner_type = tbb::simple_partitioner;
#else
  using default_partitioner_type = tbb::affinity_partitioner;
#endif

  thread_local static default_partitioner_type ap;

  if ((end - begin) < grain_size) {
    return f(begin, end, ident);
  }
  return tbb::parallel_reduce(
      tbb::blocked_range<int64_t>(begin, end, grain_size),
      scalar_t(ident),
      [f](const tbb::blocked_range<int64_t>& r, scalar_t init) {
        return f(r.begin(), r.end(), init);
      },
      sf,
      ap);
#elif defined(CPU_THREADPOOL_OPENMP)
  int64_t num_results = divup((end - begin), grain_size);
  std::vector<scalar_t> results(num_results);
  scalar_t* results_data = results.data();
#pragma omp parallel for if ((end - begin) >= grain_size)
  for (int64_t id = 0; id < num_results; id++) {
    int64_t i = begin + id * grain_size;
    results_data[id] = f(i, i + std::min(end - i, grain_size), ident);
  }
  return std::accumulate(
      results_data, results_data + results.size(), ident, sf);
#else
#error("Specified CPU threadpool not supported")
#endif
}

} // namespace at
