#include "caffe2/operators/softmax_utils.h"

#include "caffe2/core/context.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
namespace softmax_utils {

#define CAFFE2_SPECIALIZED_SOFTMAX_CPU(T)                        \
  template <>                                                    \
  void SoftmaxCPU<T>(                                            \
      const int N,                                               \
      const int D,                                               \
      const bool logarithmic,                                    \
      const T* X,                                                \
      T* Y,                                                      \
      T* scratch,                                                \
      CPUContext* context) {                                     \
    ConstEigenArrayMap<T> X_arr(X, D, N);                        \
    EigenArrayMap<T> Y_arr(Y, D, N);                             \
    EigenVectorArrayMap<T> scratch_arr(scratch, N);              \
    scratch_arr = X_arr.colwise().maxCoeff().transpose();        \
    Y_arr = X_arr.rowwise() - scratch_arr.transpose();           \
    math::Exp<T, CPUContext>(N * D, Y, Y, context);              \
    if (logarithmic) {                                           \
      scratch_arr += Y_arr.colwise().sum().log().transpose();    \
      Y_arr = X_arr.rowwise() - scratch_arr.transpose();         \
    } else {                                                     \
      scratch_arr = Y_arr.colwise().sum().inverse().transpose(); \
      Y_arr = Y_arr.rowwise() * scratch_arr.transpose();         \
    }                                                            \
  }
CAFFE2_SPECIALIZED_SOFTMAX_CPU(float)
#undef CAFFE2_SPECIALIZED_SOFTMAX_CPU

} // namespace softmax_utils
} // namespace caffe2
