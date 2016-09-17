#include <cmath>

#include "caffe2/operators/elementwise_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

struct SoftsignCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* device_context) {
    ConstEigenVectorArrayMap<T> x_arr(x, n);
    EigenVectorMap<T>(y, n) = (1 + x_arr.abs()).inverse() * x_arr;
  }
};

namespace {
REGISTER_CPU_OPERATOR(
    Softsign,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, SoftsignCPUFunctor>);

OPERATOR_SCHEMA(Softsign)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Calculates the softsign (x/1+|x|) of the given input tensor element-wise. This
operation can be done in an in-place fashion too, by providing the same input
and output blobs.
)DOC")
    .Input(0, "input", "1-D input tensor")
    .Output(
        0,
        "output",
        "The softsign (x/1+|x|) values of the input tensor "
        "computed element-wise");

} // namespace
} // namespace caffe2
