#include "partition.h"

namespace caffe2 {

namespace intra_op_parallel {

static size_t GetWorkPerThread(size_t work, int nthreads) {
  return (work + nthreads - 1) / nthreads;
}

std::pair<size_t, size_t> Get1DPartition(size_t work, int nthreads, int tid) {
  size_t work_per_thread = GetWorkPerThread(work, nthreads);
  size_t work_begin = std::min(tid * work_per_thread, work);
  size_t work_end = std::min(work_begin + work_per_thread, work);
  return {work_begin, work_end};
}

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
    int thread_id) {
  // mb: number of thread blocks within a socket along m
  // nb: number of thread blocks along n
  // mb * nb = nthreads
  // bm: number of rows assigned per thread block (bm = ceil(m/mb))
  // bn: number of cols assigned per thread block (bn = ceil(n/nb))
  // find mb and nb such that bm / bn is as close as possible to aspect_ratio

  int mb = 1;
  int nb = nthreads / mb;

  int bm;
  if (m_align) {
    bm = ((m + 15) / 16 + mb - 1) / mb * 16;
  } else {
    bm = (m + mb - 1) / mb;
  }
  int bn = ((n + 15) / 16 + nb - 1) / nb * 16;

  double best_delta = std::abs((double)bm / bn - aspect_ratio);

  for (int mb_candidate = 2; mb_candidate <= nthreads; ++mb_candidate) {
    if (nthreads % mb_candidate != 0) {
      continue;
    }
    int nb_candidate = nthreads / mb_candidate;

    if (m_align) {
      if (aspect_ratio < 1 && bm == 16 &&
          (m + mb_candidate - 1) / mb_candidate < 16) {
        continue;
      }
      if ((m + mb_candidate - 1) / mb_candidate <= 8) {
        continue;
      }
    }
    if ((n + nb_candidate - 1) / nb_candidate <= 8) {
      continue;
    }

    int bm_candidate;
    if (m_align) {
      bm_candidate = ((m + 15) / 16 + mb_candidate - 1) / mb_candidate * 16;
    } else {
      bm_candidate = (m + mb_candidate - 1) / mb_candidate;
    }
    int bn_candidate = ((n + 15) / 16 + nb_candidate - 1) / nb_candidate * 16;
    double delta = std::abs((double)bm_candidate / bn_candidate - aspect_ratio);

    if (delta < best_delta) {
      best_delta = delta;

      bm = bm_candidate;
      bn = bn_candidate;
      nb = nb_candidate;
    } else {
      break;
    }
  }

  int m_tid = thread_id / nb;
  int n_tid = thread_id % nb;

  *m_begin = std::min(m_tid * bm, m);
  *m_end = std::min(*m_begin + bm, m);

  *n_begin = std::min(n_tid * bn, n);
  *n_end = std::min(*n_begin + bn, n);
}

} // namespace intra_op_parallel

} // namespace caffe2
