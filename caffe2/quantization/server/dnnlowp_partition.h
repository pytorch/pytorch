#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>

namespace caffe2 {

std::pair<size_t, size_t> Get1DPartition(
    size_t work,
    int nthreads,
    int tid,
    int work_align = 1);

/**
 * 1D-partition m x n 2D work.
 * First try partitioning m if m >= nthreads.
 * Otherwise, each row is partitioned by multiple threads.
 * In this case, each thread only works on a single row.
 * Optionally, we can force the number of columns assigned per thread is a
 * multiple of n_align.
 */
void Get1DPartitionOf2D(
    int m,
    int n,
    int nthreads,
    int thread_id,
    int *m_begin,
    int *m_end,
    int *n_begin,
    int *n_end,
    int n_align = 1);

} // namespace caffe2
