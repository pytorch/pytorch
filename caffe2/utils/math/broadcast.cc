#include "caffe2/utils/math/broadcast.h"

#include "caffe2/core/context.h"
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {
namespace math {

#define CAFFE2_SPECIALIZED_AFFINE_CHANNEL(T)                        \
  template <>                                                       \
  C10_EXPORT void AffineChannel<T, CPUContext, StorageOrder::NCHW>( \
      const int N,                                                  \
      const int C,                                                  \
      const int HxW,                                                \
      const T* X,                                                   \
      const T* scale,                                               \
      const T* bias,                                                \
      T* Y,                                                         \
      CPUContext* /* context */) {                                  \
    ConstEigenVectorArrayMap<T> scale_arr(scale, C);                \
    ConstEigenVectorArrayMap<T> bias_arr(bias, C);                  \
    const int stride = C * HxW;                                     \
    const T* X_ptr = X;                                             \
    T* Y_ptr = Y;                                                   \
    for (int i = 0; i < N; ++i) {                                   \
      EigenArrayMap<T>(Y_ptr, HxW, C) =                             \
          (ConstEigenArrayMap<T>(X_ptr, HxW, C).rowwise() *         \
           scale_arr.transpose())                                   \
              .rowwise() +                                          \
          bias_arr.transpose();                                     \
      X_ptr += stride;                                              \
      Y_ptr += stride;                                              \
    }                                                               \
  }                                                                 \
  template <>                                                       \
  C10_EXPORT void AffineChannel<T, CPUContext, StorageOrder::NHWC>( \
      const int N,                                                  \
      const int C,                                                  \
      const int HxW,                                                \
      const T* X,                                                   \
      const T* scale,                                               \
      const T* bias,                                                \
      T* Y,                                                         \
      CPUContext* /* context */) {                                  \
    EigenArrayMap<T>(Y, C, N * HxW) =                               \
        (ConstEigenArrayMap<T>(X, C, N * HxW).colwise() *           \
         ConstEigenVectorArrayMap<T>(scale, C))                     \
            .colwise() +                                            \
        ConstEigenVectorArrayMap<T>(bias, C);                       \
  }
CAFFE2_SPECIALIZED_AFFINE_CHANNEL(float)
#undef CAFFE2_SPECIALIZED_AFFINE_CHANNEL

} // namespace math
} // namespace caffe2
