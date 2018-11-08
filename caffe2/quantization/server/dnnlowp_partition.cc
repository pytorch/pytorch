#include "dnnlowp_partition.h"

#include "caffe2/core/logging.h"

namespace caffe2
{

static size_t GetWorkPerThread_(size_t work, int nthreads, int work_align) {
  return ((work + work_align - 1) / work_align + nthreads - 1) /
         nthreads * work_align;
}

std::pair<size_t, size_t> Get1DPartition(
    size_t work,
    int nthreads,
    int tid,
    int work_align /*=1*/) {
  size_t work_per_thread = GetWorkPerThread_(work, nthreads, work_align);
  size_t work_begin = std::min(tid * work_per_thread, work);
  size_t work_end = std::min(work_begin + work_per_thread, work);
  return {work_begin, work_end};
}

void Get1DPartitionOf2D(
    int m,
    int n,
    int nthreads,
    int tid,
    int *m_begin,
    int *m_end,
    int *n_begin,
    int *n_end,
    int n_align /*=1*/) {
  if (m >= nthreads) {
    // When m >= nthreads, just parallelize over m.
    std::tie(*m_begin, *m_end) = Get1DPartition(m, nthreads, tid);
    *n_begin = 0;
    *n_end = n;
  } else {
    // Otherwise, each row is parallelized by multiple threads.
    // nthreads_per_row is floor(nthreads / m). If we use ceil, some rows won't
    // be handled by any thread.
    int nthreads_per_row = nthreads / m;
    *m_begin = std::max(std::min(tid / nthreads_per_row, m - 1), 0);
    *m_end = std::min(*m_begin + 1, m);

    int tid_of_m_begin = std::min(*m_begin * nthreads_per_row, nthreads);
    int tid_of_m_end = std::min(
        (*m_end == m) ? nthreads : (tid_of_m_begin + nthreads_per_row),
        nthreads);
    int nthreads_within_row = tid_of_m_end - tid_of_m_begin;
    int tid_within_row = tid - tid_of_m_begin;
    CAFFE_ENFORCE_GE(tid_within_row, 0);
    CAFFE_ENFORCE_LT(tid_within_row, nthreads_within_row);

    std::tie(*n_begin, *n_end) = Get1DPartition(
        n, nthreads_within_row, tid_within_row, n_align);
  }
}

} // namespace caffe2
