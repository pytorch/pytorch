#ifndef CAFFE2_UTILS_MKL_SGEMM_PACK_H_
#define CAFFE2_UTILS_MKL_SGEMM_PACK_H_

#include "caffe2/core/logging.h"

namespace caffe2 {
namespace mkl {
struct MKLPackedMatrix {
  CBLAS_IDENTIFIER identifier_;
  CBLAS_TRANSPOSE trans_;
  int m_;
  int n_;
  int k_;
  float alpha_;
  int ld_;
  float* data_ = nullptr;

  MKLPackedMatrix(
      const CBLAS_IDENTIFIER identifier,
      const CBLAS_TRANSPOSE trans,
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
    data_ = cblas_sgemm_alloc(identifier, m, n, k);
    CAFFE_ENFORCE(data_, "MKL runtime error: cannot allocate sgemm memory.");
    cblas_sgemm_pack(
        CblasRowMajor, identifier, trans, m, n, k, alpha, src, ld, data_);
  }

  ~MKLPackedMatrix() {
    if (data_) {
      cblas_sgemm_free(data_);
    }
  }
};

} // namespace mkl
} // namespace caffe2

#endif // CAFFE2_UTILS_MKL_SGEMM_PACK_H_
