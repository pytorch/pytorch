#include "caffe2/mkl/operators/operator_fallback_mkl.h"

#include "caffe2/mkl/utils/mkl_operator.h"
#include "caffe2/operators/channel_shuffle_op.h"
#include "caffe2/operators/cross_entropy_op.h"
#include "caffe2/operators/dropout_op.h"
#include "caffe2/operators/elementwise_linear_op.h"
#include "caffe2/operators/elementwise_op.h"
#include "caffe2/operators/filler_op.h"
#include "caffe2/operators/load_save_op.h"
#include "caffe2/operators/loss_op.h"
#include "caffe2/operators/reshape_op.h"
#include "caffe2/operators/softmax_op.h"
#include "caffe2/operators/utility_ops.h"

namespace caffe2 {
namespace {
struct SigmoidCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* /*device_context*/) {
    ConstEigenVectorArrayMap<T> xM(x, n);
    EigenVectorArrayMap<T>(y, n) = 1. / (1. + (-xM).exp());
  }
};
} // namespace
} // namespace caffe2

// can add more non-MKL operators if needed
namespace caffe2 {
REGISTER_MKL_OPERATOR(
    Softmax,
    mkl::MKLFallbackOp<SoftmaxOp<float, CPUContext>>);
REGISTER_MKL_OPERATOR(
    Reshape,
    mkl::MKLFallbackOp<ReshapeOp<float, CPUContext>, SkipIndices<1>>);
REGISTER_MKL_OPERATOR(
    LabelCrossEntropy,
    mkl::MKLFallbackOp<LabelCrossEntropyOp<float, CPUContext>>);
REGISTER_MKL_OPERATOR(
    AveragedLoss,
    mkl::MKLFallbackOp<AveragedLoss<float, CPUContext>>);

// filter operators
REGISTER_MKL_OPERATOR(
    XavierFill,
    mkl::MKLFallbackOp<XavierFillOp<float, CPUContext>>);
REGISTER_MKL_OPERATOR(
    ConstantFill,
    mkl::MKLFallbackOp<ConstantFillOp<CPUContext>>);
REGISTER_MKL_OPERATOR(
    GaussianFill,
    mkl::MKLFallbackOp<GaussianFillOp<float, CPUContext>>);
REGISTER_MKL_OPERATOR(
    MSRAFill,
    mkl::MKLFallbackOp<MSRAFillOp<float, CPUContext>>);
REGISTER_MKL_OPERATOR(Load, mkl::MKLFallbackOp<LoadOp<CPUContext>>);
REGISTER_MKL_OPERATOR(Save, mkl::MKLFallbackOp<SaveOp<CPUContext>>);

REGISTER_MKL_OPERATOR(
    Sigmoid,
    mkl::MKLFallbackOp<
        UnaryElementwiseOp<TensorTypes<float>, CPUContext, SigmoidCPUFunctor>>);
REGISTER_MKL_OPERATOR(
    Dropout,
    mkl::MKLFallbackOp<DropoutOp<float, CPUContext>, SkipIndices<1>>);
REGISTER_MKL_OPERATOR(
    ElementwiseLinear,
    mkl::MKLFallbackOp<ElementwiseLinearOp<float, CPUContext>>);
REGISTER_MKL_OPERATOR(
    ChannelShuffle,
    mkl::MKLFallbackOp<ChannelShuffleOp<CPUContext>>);

} // namespace caffe2
