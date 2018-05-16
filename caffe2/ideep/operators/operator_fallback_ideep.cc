#include <caffe2/ideep/operators/operator_fallback_ideep.h>
#include <caffe2/ideep/utils/ideep_operator.h>

#include <caffe2/operators/channel_shuffle_op.h>
#include <caffe2/operators/conv_transpose_op.h>
#include <caffe2/operators/cross_entropy_op.h>
#include <caffe2/operators/dropout_op.h>
#include <caffe2/operators/filler_op.h>
#include <caffe2/operators/flatten_op.h>
#include <caffe2/operators/given_tensor_fill_op.h>
#include <caffe2/operators/loss_op.h>
#include <caffe2/operators/reshape_op.h>
#include <caffe2/operators/softmax_op.h>
#include <caffe2/operators/transpose_op.h>
#include <caffe2/operators/utility_ops.h>

// can add more non-IDEEP operators if needed
namespace caffe2 {

REGISTER_IDEEP_OPERATOR(Softmax, IDEEPFallbackOp<SoftmaxOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    ChannelShuffle,
    IDEEPFallbackOp<ChannelShuffleOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    LabelCrossEntropy,
    IDEEPFallbackOp<LabelCrossEntropyOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    AveragedLoss,
    IDEEPFallbackOp<AveragedLoss<float, CPUContext>, SkipIndices<0>>);
REGISTER_IDEEP_OPERATOR(
    ConvTranspose,
    IDEEPFallbackOp<ConvTransposeOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(Flatten, IDEEPFallbackOp<FlattenOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(ResizeLike, IDEEPFallbackOp<ResizeLikeOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(Transpose, IDEEPFallbackOp<TransposeOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    Reshape,
    IDEEPFallbackOp<ReshapeOp<float, CPUContext>, SkipIndices<1>>);

// filter operators
REGISTER_IDEEP_OPERATOR(
    XavierFill,
    IDEEPFallbackOp<XavierFillOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    ConstantFill,
    IDEEPFallbackOp<ConstantFillOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    GaussianFill,
    IDEEPFallbackOp<GaussianFillOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    MSRAFill,
    IDEEPFallbackOp<MSRAFillOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    GivenTensorFill,
    IDEEPFallbackOp<GivenTensorFillOp<float, CPUContext>>);

} // namespace caffe2
