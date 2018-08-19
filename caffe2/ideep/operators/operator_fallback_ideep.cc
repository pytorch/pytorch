#include <caffe2/ideep/operators/operator_fallback_ideep.h>
#include <caffe2/ideep/utils/ideep_operator.h>

#include <caffe2/operators/accuracy_op.h>
#include <caffe2/operators/bbox_transform_op.h>
#include <caffe2/operators/box_with_nms_limit_op.h>
#include <caffe2/operators/channel_shuffle_op.h>
#include <caffe2/operators/collect_and_distribute_fpn_rpn_proposals_op.h>
#include <caffe2/operators/conv_transpose_op.h>
#include <caffe2/operators/cross_entropy_op.h>
#include <caffe2/operators/ctc_beam_search_decoder_op.h>
#include <caffe2/operators/ctc_greedy_decoder_op.h>
#include <caffe2/operators/distance_op.h>
#include <caffe2/operators/dropout_op.h>
#include <caffe2/operators/elementwise_add_op.h>
#include <caffe2/operators/elementwise_mul_op.h>
#include <caffe2/operators/elementwise_ops.h>
#include <caffe2/operators/filler_op.h>
#include <caffe2/operators/flatten_op.h>
#include <caffe2/operators/generate_proposals_op.h>
#include <caffe2/operators/given_tensor_fill_op.h>
#include <caffe2/operators/leaky_relu_op.h>
#include <caffe2/operators/load_save_op.h>
#include <caffe2/operators/loss_op.h>
#include <caffe2/operators/pad_op.h>
#include <caffe2/operators/prelu_op.h>
#include <caffe2/operators/reshape_op.h>
#include <caffe2/operators/roi_align_op.h>
#include <caffe2/operators/roi_align_rotated_op.h>
#include <caffe2/operators/scale_op.h>
#include <caffe2/operators/softmax_op.h>
#include <caffe2/operators/tanh_op.h>
#include <caffe2/operators/transpose_op.h>
#include <caffe2/operators/utility_ops.h>
#include <caffe2/sgd/adam_op.h>
#include <caffe2/sgd/iter_op.h>
#include <caffe2/sgd/learning_rate_op.h>

// can add more non-IDEEP operators if needed
namespace caffe2 {
namespace {

struct SigmoidCPUFunctor {
  template <typename T>
  bool operator()(const int n, const T* x, T* y, CPUContext* /* context */)
      const {
    ConstEigenVectorArrayMap<T> xM(x, n);
    EigenVectorArrayMap<T>(y, n) = 1. / (1. + (-xM).exp());
    return true;
  }
};

} // namespace

REGISTER_IDEEP_OPERATOR(Softmax, IDEEPFallbackOp<SoftmaxOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    ChannelShuffle,
    IDEEPFallbackOp<ChannelShuffleOp<float, CPUContext>>);
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
REGISTER_IDEEP_OPERATOR(Load, IDEEPFallbackOp<LoadOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(Save, IDEEPFallbackOp<SaveOp<CPUContext>>);

REGISTER_IDEEP_OPERATOR(
    Sigmoid,
    IDEEPFallbackOp<
        UnaryElementwiseOp<TensorTypes<float>, CPUContext, SigmoidCPUFunctor>>);
REGISTER_IDEEP_OPERATOR(
    RoIAlign,
    IDEEPFallbackOp<RoIAlignOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    RoIAlignRotated,
    IDEEPFallbackOp<RoIAlignRotatedOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    GenerateProposals,
    IDEEPFallbackOp<GenerateProposalsOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    GenerateProposalsCPP,
    IDEEPFallbackOp<GenerateProposalsOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    CollectAndDistributeFpnRpnProposals,
    IDEEPFallbackOp<CollectAndDistributeFpnRpnProposalsOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    BoxWithNMSLimit,
    IDEEPFallbackOp<BoxWithNMSLimitOp<CPUContext>, SkipIndices<0,1,2>>);
REGISTER_IDEEP_OPERATOR(
    BBoxTransform,
    IDEEPFallbackOp<BBoxTransformOp<float, CPUContext>>);

REGISTER_IDEEP_OPERATOR(
    PadImage,
    IDEEPFallbackOp<PadImageOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    PRelu,
    IDEEPFallbackOp<PReluOp<float, CPUContext>>);
  
// ctc decoder operators
REGISTER_IDEEP_OPERATOR(
    CTCGreedyDecoder,
    IDEEPFallbackOp<CTCGreedyDecoderOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    CTCBeamSearchDecoder,
    IDEEPFallbackOp<CTCBeamSearchDecoderOp<CPUContext>>);

REGISTER_IDEEP_OPERATOR(
    AveragedLossGradient,
    IDEEPFallbackOp<AveragedLossGradient<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    LabelCrossEntropyGradient,
    IDEEPFallbackOp<LabelCrossEntropyGradientOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    SoftmaxGradient,
    IDEEPFallbackOp<SoftmaxGradientOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    Iter,
    IDEEPFallbackOp<IterOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    LearningRate,
    IDEEPFallbackOp<LearningRateOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    WeightedSum,
    IDEEPFallbackOp<WeightedSumOp<CPUContext>>);

REGISTER_IDEEP_OPERATOR(
    LeakyRelu,
    IDEEPFallbackOp<LeakyReluOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    Mul,
    IDEEPFallbackOp<
        BinaryElementwiseOp<NumericTypes, CPUContext, MulFunctor<CPUContext>>>);
REGISTER_IDEEP_OPERATOR(
    Tanh,
    IDEEPFallbackOp<UnaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        TanhFunctor<CPUContext>>>);
REGISTER_IDEEP_OPERATOR(
    L1Distance,
    IDEEPFallbackOp<L1DistanceOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(Scale, IDEEPFallbackOp<ScaleOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    Accuracy,
    IDEEPFallbackOp<AccuracyOp<float, CPUContext>>);

REGISTER_IDEEP_OPERATOR(
    AddGradient,
    IDEEPFallbackOp<BinaryElementwiseGradientOp<
        NumericTypes,
        CPUContext,
        AddFunctor<CPUContext>>>);
REGISTER_IDEEP_OPERATOR(
    TanhGradient,
    IDEEPFallbackOp<BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        TanhGradientFunctor<CPUContext>>>);
REGISTER_IDEEP_OPERATOR(
    ConvTransposeGradient,
    IDEEPFallbackOp<ConvTransposeGradientOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    LeakyReluGradient,
    IDEEPFallbackOp<LeakyReluGradientOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    MulGradient,
    IDEEPFallbackOp<BinaryElementwiseGradientOp<
        NumericTypes,
        CPUContext,
        MulFunctor<CPUContext>>>);
REGISTER_IDEEP_OPERATOR(Adam, IDEEPFallbackOp<AdamOp<float, CPUContext>>);

} // namespace caffe2
