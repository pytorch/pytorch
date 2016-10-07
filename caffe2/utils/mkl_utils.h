#ifndef CAFFE2_UTILS_MKL_UTILS_H_
#define CAFFE2_UTILS_MKL_UTILS_H_
#ifdef CAFFE2_USE_MKL

#include <mkl.h>

#include "caffe2/core/logging.h"

#if INTEL_MKL_VERSION >= 20170000
#define CAFFE2_HAS_MKL_SGEMM_PACK
#define CAFFE2_HAS_MKL_DNN

namespace caffe2 {
namespace mkl {

struct MKLPackedMatrix {
  char identifier_;
  char trans_;
  int m_;
  int n_;
  int k_;
  float alpha_;
  int ld_;
  float* data_ = nullptr;

  MKLPackedMatrix(
      const char identifier,
      const char trans,
      const int m,
      const int n,
      const int k,
      const float alpha,
      const float* src,
      const int ld)
      : identifier_(identifier),
        trans_(trans),
        m_(m),
        n_(n),
        k_(k),
        alpha_(alpha),
        ld_(ld) {
    data_ = sgemm_alloc(&identifier, &m, &n, &k);
    CAFFE_ENFORCE(data_, "MKL runtime error: cannot allocate sgemm memory.");
    sgemm_pack(&identifier, &trans, &m, &n, &k, &alpha, src, &ld, data_);
  }

  ~MKLPackedMatrix() {
    if (data_) {
      sgemm_free(data_);
    }
  }
};

} // namespace mkl
} // namespace caffe2

#endif // INTEL_MKL_VERSION >= 20170000
#endif // CAFFE2_USE_MKL
#endif // CAFFE2_UTILS_MKL_UTILS_H_
