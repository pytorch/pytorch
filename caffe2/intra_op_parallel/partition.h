#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>

namespace caffe2 {

namespace intra_op_parallel {

std::pair<size_t, size_t> Get1DPartition(size_t work, int nthreads, int tid);

/**
 * Compute a 2D decomposition of matrix multiplication where output matrix is
 * m-by-n. The caller thread will work on [m_begin:m_end, n_begin:n_end]
 * submatrix of the output. This function tries to find a decomposition that
 * (m_end - m_begin) ~= aspect_ratio * (n_end - n_begin) .
 * We control 2D decomposition by this aspect ratio because the performance is
 * mostly a function of the aspect ratio that we can auto-tune.
 * Alternatively, we can fine tune (m_end - m_begin) and (n_end - n_begin), but
 * doing so would require quadratic auto-tune complexity, so auto-tuning the
 * aspect ratio can be a good compromise.
 *
 * @param m_align when true, m_end - m_begin a multiple of 16 to be cache-line
 *                aligned. n_end - n_begin is always aligned
 */
void Get2DPartition(
    int* m_begin,
    int* m_end,
    int* n_begin,
    int* n_end,
    int m,
    int n,
    double aspect_ratio,
    bool m_align,
    int nthreads,
    int thread_id);

} // namespace intra_op_parallel

} // namespace caffe2
