#include <caffe2/ideep/operators/operator_fallback_ideep.h>
#include <caffe2/ideep/utils/ideep_operator.h>

#include <caffe2/operators/accuracy_op.h>
#include <caffe2/operators/affine_channel_op.h>
#include <caffe2/operators/batch_matmul_op.h>
#include <caffe2/operators/bbox_transform_op.h>
#include <caffe2/operators/box_with_nms_limit_op.h>
#include <caffe2/operators/clip_op.h>
#include <caffe2/operators/collect_and_distribute_fpn_rpn_proposals_op.h>
#include <caffe2/operators/conv_transpose_op.h>
#include <caffe2/operators/cross_entropy_op.h>
#include <caffe2/operators/ctc_beam_search_decoder_op.h>
#include <caffe2/operators/ctc_greedy_decoder_op.h>
#include <caffe2/operators/distance_op.h>
#include <caffe2/operators/dropout_op.h>
#include <caffe2/operators/elementwise_add_op.h>
#include <caffe2/operators/elementwise_div_op.h>
#include <caffe2/operators/elementwise_mul_op.h>
#include <caffe2/operators/elementwise_ops.h>
#include <caffe2/operators/expand_op.h>
#include <caffe2/operators/filler_op.h>
#include <caffe2/operators/flatten_op.h>
#include <caffe2/operators/gather_op.h>
#include <caffe2/operators/generate_proposals_op.h>
#include <caffe2/operators/given_tensor_fill_op.h>
#include <caffe2/operators/load_save_op.h>
#include <caffe2/operators/loss_op.h>
#include <caffe2/operators/order_switch_ops.h>
#include <caffe2/operators/pad_op.h>
#include <caffe2/operators/prelu_op.h>
#include <caffe2/operators/reduce_ops.h>
#include <caffe2/operators/roi_align_op.h>
#include <caffe2/operators/roi_align_rotated_op.h>
#include <caffe2/operators/scale_op.h>
#include <caffe2/operators/softmax_op.h>
#include <caffe2/operators/softmax_with_loss_op.h>
#include <caffe2/operators/stop_gradient.h>
#include <caffe2/operators/tanh_op.h>
#include <caffe2/operators/tensor_protos_db_input.h>
#include <caffe2/operators/transpose_op.h>
#include <caffe2/queue/queue_ops.h>
#include <caffe2/sgd/iter_op.h>
#include <caffe2/sgd/learning_rate_op.h>

#ifdef CAFFE2_USE_GLOO
#include <caffe2/contrib/gloo/common_world_ops.h>
#include <caffe2/contrib/gloo/broadcast_ops.h>
#include <caffe2/contrib/gloo/allreduce_ops.h>
#include <caffe2/contrib/gloo/allgather_ops.h>
#include <caffe2/contrib/gloo/barrier_ops.h>
#include <caffe2/contrib/gloo/reduce_scatter_ops.h>
#endif

// can add more non-IDEEP operators if needed
namespace caffe2 {

REGISTER_IDEEP_OPERATOR(Softmax, IDEEPFallbackOp<SoftmaxOp<float, CPUContext>>);
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
    AffineChannel,
    IDEEPFallbackOp<AffineChannelOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    StopGradient,
    IDEEPFallbackOp<StopGradientOp<CPUContext>>);

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
    MulGradient,
    IDEEPFallbackOp<BinaryElementwiseGradientOp<
        NumericTypes,
        CPUContext,
        MulFunctor<CPUContext>>>);
REGISTER_IDEEP_OPERATOR(TensorProtosDBInput, IDEEPFallbackOp<TensorProtosDBInput<CPUContext>>);
REGISTER_IDEEP_OPERATOR(CloseBlobsQueue, IDEEPFallbackOp<CloseBlobsQueueOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    SoftmaxWithLoss,
    IDEEPFallbackOp<SoftmaxWithLossOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    SoftmaxWithLossGradient,
    IDEEPFallbackOp<SoftmaxWithLossGradientOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    NHWC2NCHW,
    IDEEPFallbackOp<NHWC2NCHWOp<float, CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    NCHW2NHWC,
    IDEEPFallbackOp<NCHW2NHWCOp<float, CPUContext>>);

REGISTER_IDEEP_OPERATOR(
    Expand,
    IDEEPFallbackOp<ExpandOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CPUContext>>);
REGISTER_IDEEP_OPERATOR(Gather, IDEEPFallbackOp<GatherOp<CPUContext>>);

REGISTER_IDEEP_OPERATOR(
    ReduceL2,
    IDEEPFallbackOp<
        ReduceOp<TensorTypes<float>, CPUContext, L2Reducer<CPUContext>>>);
REGISTER_IDEEP_OPERATOR(
    ReduceSum,
    IDEEPFallbackOp<ReduceOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CPUContext,
        SumReducer<CPUContext>>>);

REGISTER_IDEEP_OPERATOR(
    BatchMatMul,
    IDEEPFallbackOp<BatchMatMulOp<CPUContext>>);

REGISTER_IDEEP_OPERATOR(
    Div,
    IDEEPFallbackOp<
        BinaryElementwiseOp<NumericTypes, CPUContext, DivFunctor<CPUContext>>>);

REGISTER_IDEEP_OPERATOR(Clip, IDEEPFallbackOp<ClipOp<float, CPUContext>>);

#ifdef CAFFE2_USE_GLOO
namespace gloo {
// gloo operators
REGISTER_IDEEP_OPERATOR(
    CreateCommonWorld,
    IDEEPFallbackOp<CreateCommonWorld<CPUContext>, SkipIndices<0>>);
REGISTER_IDEEP_OPERATOR(
    CloneCommonWorld,
    IDEEPFallbackOp<CloneCommonWorld<CPUContext>, SkipIndices<0>>);
REGISTER_IDEEP_OPERATOR(
    DestroyCommonWorld,
    IDEEPFallbackOp<DestroyCommonWorld>);
REGISTER_IDEEP_OPERATOR(
    Broadcast,
    IDEEPFallbackOp<BroadcastOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    Allreduce,
    IDEEPFallbackOp<AllreduceOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    Allgather,
    IDEEPFallbackOp<AllgatherOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    Barrier,
    IDEEPFallbackOp<BarrierOp<CPUContext>>);
REGISTER_IDEEP_OPERATOR(
    ReduceScatter,
    IDEEPFallbackOp<ReduceScatterOp<CPUContext>>);

} // namespace gloo
#endif

} // namespace caffe2
