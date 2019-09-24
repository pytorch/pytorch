#include <ATen/ATen.h>
#include <ATen/SmallVector.h>
#include <ATen/Parallel.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qmkldnn_utils.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <caffe2/utils/threadpool/ThreadPoolMobile.h>
#include <cmath>

namespace at {
namespace native {
namespace {

SmallVector<int64_t, 4> convOutputShape(
    int N, // mini-batch
    int K, // output channels
    int H, // input height
    int W, // input width
    const std::vector<int64_t>& kernel,
    const torch::List<int64_t>& stride,
    const torch::List<int64_t>& padding,
    const torch::List<int64_t>& dilation) {
  SmallVector<int64_t, 4> out_shape;
  out_shape.push_back(N);

  int H_out = std::floor(
      (H + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1);
  int W_out = std::floor(
      (W + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1);
  out_shape.push_back(H_out);
  out_shape.push_back(W_out);
  // TODO: reorder it to NCHW order once the memory format regression is fixed
  out_shape.push_back(K);

  return out_shape;
}

/*
 * FBGEMM uses vpmaddubsw instruction to multiply activations (uint8_t) and
 * weights (int8_t).
 *
 * https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maddubs_epi16&expand=3284,3530
 *
 * vpmaddubsw operates on a vector of activations and a vector of
 * weights. If these vectors are
 *
 *    A (uint8_t) = a0, a1, a2, a3 ...
 *
 * and
 *
 *    B (int8_t)  = b0, b1, b2, b3 ...
 *
 * the result of this instruction is an int16_t vector with values
 *
 *    C (int16_t) = a0*b0 + a1*b1, a2*b2 + a3*b3 ...
 *
 * For large values of A and/or B the result (a0*b0 + a1*b1) might not fit into
 * an int16_t number. So the instruction saturates them to max (or min) possible
 * value of an int16_t number. Such behavior is expected for the
 * implementation below.
 *
 * For example, a0 = 255, a1 = 255, b0 = 127 and b1 = 127 the actual result
 * 64770 overflows for an int16_t number (-32768, 32767) so the returned result
 * is 32767.
 *
 */
template <bool ReluFused>
class QConv2dInt8 final : public c10::OperatorKernel {
 public:
  void conv_checks(
      int64_t act_dims,
      int64_t stride_dims,
      int64_t padding_dims,
      int64_t dilation_dims) {
    TORCH_CHECK(
        act_dims == 4,
        "quantized::conv2d(): Expected activation tensor to have 4 dimensions.");
    TORCH_CHECK(
        stride_dims == 2, "quantized::conv2d(): Supports 2D convolution only");
    TORCH_CHECK(
        padding_dims == 2, "quantized::conv2d(): Supports 2D convolution only");
    TORCH_CHECK(
        dilation_dims == 2,
        "quantized::conv2d(): Supports 2D convolution only");
  }
#ifdef USE_FBGEMM
  at::Tensor fbgemm_conv(
      Tensor act,
      Tensor packed_weight,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      double output_scale,
      int64_t output_zero_point) {
    // Quantized kernels are all written with NHWC (channels last) layout in
    // mind. Ideally, we'd be compatible with conv2d behavior and preserve the
    // inputs layout as is (doing necessary upconversions).
    //
    // However, to be more robust, for now we just force output layout to always
    // be NHWC (channels last), thus opportunistically improving perf.
    //
    // This might change when full memory format support lands
    // See https://github.com/pytorch/pytorch/issues/23403
    TORCH_CHECK(
        fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");
    conv_checks(
        act.ndimension(), stride.size(), padding.size(), dilation.size());

    int N = act.size(0);
    int C = act.size(1);
    int H = act.size(2);
    int W = act.size(3);

    // FBGEMM requires NHWC
    // TODO: change it to contiguous(MemoryFormat::ChannelsLast) once a perf
    // regression of it is fixed. Today it's equivalent because `act` sizes
    // are not used below
    Tensor act_contig = act.permute({0, 2, 3, 1}).contiguous();
    const uint8_t* act_ptr =
        reinterpret_cast<uint8_t*>(act_contig.data_ptr<c10::quint8>());

    auto& pack_ptr =
        cpp_custom_type_hack::cast<PackedConvWeight>(packed_weight);
    auto packB = pack_ptr.w.get();
    auto& col_offsets = pack_ptr.col_offsets;
    auto& kernel = pack_ptr.kernel;

    int K = packB->outputChannels();

    int pad_l = padding[0];
    int pad_t = padding[1];
    int stride_h = stride[0];
    int stride_w = stride[1];
    int kernel_h = kernel[0];
    int kernel_w = kernel[1];
    // clang-format off
    TORCH_CHECK(C == (packB->inputChannels()),
        "[QConv2D] Given groups=", groups, ", weight of size ",
        K, ", ",  kernel_h, ", ", kernel_w, ", ", packB->inputChannels(),
        ", expected input (NCHW) ", N, ", ", C, ", ", H, ", ", W,
        " to have ", (packB->inputChannels() * groups),
        " channels, but got ", C, " channels instead");
    // clang-format on
    fbgemm::conv_param_t<> conv_p(
        N, // Batch size
        C, // Number of input channels
        K, // Number of output channels
        {H, W},
        groups,
        {kernel_h, kernel_w},
        {stride_h, stride_w},
        {pad_l, pad_t, pad_l, pad_t},
        {static_cast<int>(dilation[0]), static_cast<int>(dilation[1])});

    float act_scale = act.q_scale();
    int32_t act_zero_point = act.q_zero_point();

    const float* bias_ptr = nullptr;
    at::Tensor bias;
    if (pack_ptr.bias.has_value()) {
      bias = pack_ptr.bias.value();
      TORCH_CHECK(
          bias.dtype() == at::kFloat,
          "[QConv2D] The 'bias' tensor must have 'torch.float' dtype");
      bias = bias.contiguous();
      TORCH_CHECK(bias.dim() == 1, "bias should be a vector (1D Tensor)");
      TORCH_CHECK(
          bias.size(0) == K,
          "bias should have K elements: " + std::to_string(K));
      bias_ptr = bias.data_ptr<float>();
    }

    std::vector<float> output_multiplier_float(1, 0.0);
    std::vector<float> act_times_w_scale(1, 1.0);
    TORCH_CHECK(
        pack_ptr.w_scale.size() == pack_ptr.w_zp.size(),
        "Weight scales and zero points vectors should have the same size.");

    if (pack_ptr.q_scheme == kPerTensorAffine) {
      act_times_w_scale[0] = (act_scale * pack_ptr.w_scale[0]);
      output_multiplier_float[0] =
          act_times_w_scale[0] / static_cast<float>(output_scale);
    } else if (pack_ptr.q_scheme == kPerChannelAffine) {
      output_multiplier_float.resize(K, 0.0);
      act_times_w_scale.resize(K, 1.0);
      for (int i = 0; i < K; ++i) {
        act_times_w_scale[i] = (act_scale * pack_ptr.w_scale[i]);
        output_multiplier_float[i] =
            act_times_w_scale[i] / static_cast<float>(output_scale);
      }
    } else {
      TORCH_CHECK(false, "[QConv2D] Unknown quantization scheme");
    }

    // TODO: change the following to NCHW sizes once perf is fixed
    SmallVector<int64_t, 4> outShape{
        N, conv_p.OUT_DIM[0], conv_p.OUT_DIM[1], K};
    TORCH_CHECK(
        std::all_of(
            outShape.begin(), outShape.end(), [](int64_t i) { return i > 0; }),
        "[QConv2D] each dimension of output tensor should be greater than 0")

    // Force output format to be NHWC
    // TODO: consider preserving input format
    // TODO: add MemoryFormat::ChannelsLast here once perf is fixed
    Tensor output = _empty_affine_quantized(
        outShape, device(kCPU).dtype(kQUInt8), output_scale, output_zero_point);
    auto buffer = at::empty(output.sizes(), output.options().dtype(at::kInt));

    int num_tasks = at::get_num_threads();
    at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
      fbgemm::DoNothing<> NoOpObj{};
      for (int task_id = begin; task_id < end; ++task_id) {
        if (pack_ptr.q_scheme == kPerTensorAffine) {
          fbgemm::ReQuantizeOutput<
              ReluFused,
              fbgemm::QuantizationGranularity::TENSOR,
              float>
              outputProcObj(
                  NoOpObj,
                  output_multiplier_float.data(),
                  output_zero_point,
                  act_zero_point,
                  pack_ptr.w_zp.data(),
                  nullptr, /* row offset buffer */
                  col_offsets.data(),
                  bias_ptr,
                  K,
                  groups,
                  act_times_w_scale.data());
          fbgemm::fbgemmConv(
              conv_p,
              act_ptr,
              *packB,
              reinterpret_cast<uint8_t*>(output.data_ptr<c10::quint8>()),
              buffer.data_ptr<int32_t>(),
              outputProcObj,
              task_id /* thread_id*/,
              num_tasks /* num_threads */);

        } else if (pack_ptr.q_scheme == kPerChannelAffine) {
          fbgemm::ReQuantizeOutput<
              ReluFused,
              fbgemm::QuantizationGranularity::OUT_CHANNEL,
              float>
              outputProcObj(
                  NoOpObj,
                  output_multiplier_float.data(),
                  output_zero_point,
                  act_zero_point,
                  pack_ptr.w_zp.data(),
                  nullptr, /* row offset buffer */
                  col_offsets.data(),
                  bias_ptr,
                  K,
                  groups,
                  act_times_w_scale.data());

          fbgemm::fbgemmConv(
              conv_p,
              act_ptr,
              *packB,
              reinterpret_cast<uint8_t*>(output.data_ptr<c10::quint8>()),
              buffer.data_ptr<int32_t>(),
              outputProcObj,
              task_id /* thread_id*/,
              num_tasks /* num_threads */);
        }
      }
    });

    // TODO: remove permute once MemoryLayout is added above
    return output.permute({0, 3, 1, 2});
  }

#if AT_MKLDNN_ENABLED()
Tensor mkldnn_qconv2d(
    Tensor& act,
    Tensor& packed_weight,
    torch::List<int64_t>& stride,
    torch::List<int64_t>& padding,
    torch::List<int64_t>& dilation,
    int64_t groups,
    double output_scale,
    int64_t output_zero_point) {
  TORCH_CHECK(
      act.scalar_type() == ScalarType::QUInt8,
      "Only QUInt8 ScalarType activations are support")
  TORCH_CHECK(
      output_zero_point == 0,
      "Only 0 point are support for MKL-DNN");
  conv_checks(
      act.ndimension(), stride.size(), padding.size(), dilation.size());

  auto& pack_ptr =
      cpp_custom_type_hack::cast<PackedConvWeightQmkldnn>(packed_weight);
  TORCH_CHECK(
      pack_ptr.w_scale.size() == pack_ptr.w_zp.size(),
      "Weight scales and zero points vectors should have the same size.");
  ideep::tensor weight_ = *pack_ptr.w.get();
  TORCH_CHECK(
      weight_.ndims() == 4 || weight_.ndims() == 5,
      "Packed weights are supposed to have 4 or 5 dimensions.");
  
  int N = act.size(0);
  int C = act.size(1);
  int H = act.size(2);
  int W = act.size(3);

  int K;
  std::vector<int64_t> kernel;
  auto weight_shape = weight_.get_dims();

  if (weight_.ndims() == act.ndimension() + 1) {
    TORCH_CHECK(
        groups > 1,
        "Only weights of group 2d convolution could have been prepacked to 5d");
    K = weight_shape[0] * weight_shape[1];
    kernel = {weight_shape[3], weight_shape[4]};
  } else {
    K = weight_shape[0];
    kernel = {weight_shape[2], weight_shape[3]};
  }

  auto outShape =
      convOutputShape(N, K, H, W, kernel, stride, padding, dilation);

  TORCH_CHECK(
      std::all_of(
          outShape.begin(), outShape.end(), [](int64_t i) { return i > 0; }),
      "[QConv2D] each dimension of output tensor should be greater than 0")
  ideep::scale_t output_scale_ = ConvertScales({output_scale});

  ideep::tensor act_, output_;
  float act_scale = act.q_scale();
  Tensor act_contig = act.contiguous();
  auto act_dtype = get_mkldnn_dtype(act.scalar_type());
  auto act_ptr = reinterpret_cast<uint8_t*>(act_contig.data_ptr());

  ideep::tensor::dims inputShape {N, C, H, W};
  act_.init({inputShape, act_dtype}, act_ptr);
  act_.set_scale(ConvertScales({act_scale}));

  // TODO fuse sum / sum+relu
  auto attr_ = ReluFused ?
      ideep::descriptor_group::attr_t::fuse_relu() :
      ideep::descriptor_group::attr_t();

  Tensor output = _empty_affine_quantized(
      outShape, device(kCPU).dtype(kQUInt8), output_scale, output_zero_point);
  uint8_t* output_ptr = reinterpret_cast<uint8_t*>(output.data_ptr());

  // nhwc format's logical dimensions come in the order: (n, c, h, w) in mkldnn
  ideep::tensor::dims outShape_in {N, K, outShape[1], outShape[2]};
  output_.init({outShape_in,
                ideep::tensor::data_type::u8, ideep::format::nhwc}, output_ptr);
  output_.set_scale(output_scale_);

  if (pack_ptr.bias.has_value()) {
    at::Tensor bias = pack_ptr.bias.value();
    TORCH_CHECK(
        bias.dtype() == at::kFloat,
        "[QConv2D] The 'bias' tensor must have 'torch.float' dtype");
    bias = bias.contiguous();
    TORCH_CHECK(bias.dim() == 1, "bias should be a vector (1D Tensor)");
    TORCH_CHECK(
        bias.size(0) == K,
        "bias should have K elements: " + std::to_string(K));
    auto bias_ptr = reinterpret_cast<float*>(bias.data_ptr<float>());

    ideep::tensor bias_;
    bias_.init({{K}, ideep::tensor::data_type::f32}, bias_ptr);

    ideep::convolution_forward::compute<AllocForMKLDNN>(
        act_, weight_, bias_, outShape_in, output_,
        {stride.begin(), stride.end()}, {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()}, {padding.begin(), padding.end()},
        groups, ideep::scale_t(), ideep::scale_t(), output_scale_, attr_,
        ideep::algorithm::convolution_direct, // TODO winograd
        ideep::prop_kind::forward_inference,
        ideep::padding_kind::zero,
        ideep::lowp_kind::LOWP_U8S8); // TODO s8s8
  } else {
    ideep::convolution_forward::compute<AllocForMKLDNN>(
        act_, weight_, outShape_in, output_,
        {stride.begin(), stride.end()}, {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()}, {padding.begin(), padding.end()},
        groups, ideep::scale_t(), ideep::scale_t(), output_scale_, attr_,
        ideep::algorithm::convolution_direct,
        ideep::prop_kind::forward_inference,
        ideep::padding_kind::zero,
        ideep::lowp_kind::LOWP_U8S8);
  }
  // TODO: remove permute once MemoryLayout is added above
  return output.permute({0, 3, 1, 2});
}
#endif // AT_MKLDNN_ENABLED()
#endif
#ifdef USE_PYTORCH_QNNPACK
  at::Tensor qnnpack_conv(
      Tensor act,
      Tensor packed_weight,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      double output_scale,
      int64_t output_zero_point) {
    conv_checks(
        act.ndimension(), stride.size(), padding.size(), dilation.size());

    PackedConvWeightsQnnp& pack_ptr =
        cpp_custom_type_hack::cast<PackedConvWeightsQnnp>(packed_weight);
    auto packB = pack_ptr.w.get();
    auto kernel = pack_ptr.kernel;
    auto kernel_zp = pack_ptr.w_zp;
    auto kernel_scale = pack_ptr.w_scale;

    const uint32_t kernel_h = kernel[0];
    const uint32_t kernel_w = kernel[1];
    // TODO Can be replaced with packB->getOutputChannels() when update pre-pack
    // to actually do the packing.
    const auto out_ch = pack_ptr.bias.size(0);
    // inputs are in semantic NCHW format
    int N = act.size(0);
    int in_ch = act.size(1);
    int H = act.size(2);
    int W = act.size(3);
    int K = out_ch; // output channels
    // TODO: change it to contiguous(MemoryFormat::ChannelsLast) once a perf
    // regression of it is fixed. Today it's equivalent because `act` sizes
    // are not used below
    Tensor input_contig = act.permute({0, 2, 3, 1}).contiguous();

    uint32_t stride_h = stride[0];
    uint32_t stride_w = stride[1];
    uint32_t pad_t = padding[0];
    uint32_t pad_l = padding[1];
    uint32_t dilation_h = dilation[0];
    uint32_t dilation_w = dilation[1];

    auto output_min = ReluFused
        ? activationLimits(output_scale, output_zero_point, Activation::RELU)
              .first
        : std::numeric_limits<uint8_t>::min();
    auto output_max = ReluFused
        ? activationLimits(output_scale, output_zero_point, Activation::RELU)
              .second
        : std::numeric_limits<uint8_t>::max();
    qnnpack::conv_param_t conv_p(
        {kernel_w, kernel_h},
        {stride_w, stride_h},
        {dilation_w, dilation_h},
        {pad_t, pad_l, pad_t, pad_l},
        groups,
        in_ch,
        out_ch,
        kernel_zp,
        kernel_scale,
        output_min,
        output_max);

    // TODO: change convOutputShape to return NCHW sizes once perf is fixed
    // Force output format to be NHWC
    // TODO: consider preserving input format
    // TODO: add MemoryFormat::ChannelsLast here once perf is fixed
    auto input_scale = input_contig.q_scale();

    // Re-quantizing the bias based on input scale and weight scale.
    if (!pack_ptr.input_scale.has_value() ||
        pack_ptr.input_scale.value() != input_scale) {
      // Get the original weight and adjust it to uint8 from int8
      auto weight_contig =
          pack_ptr.orig_weight.contiguous(MemoryFormat::ChannelsLast);
      auto bias_fp32 = pack_ptr.bias;
      int8_t* w_data = (int8_t*)weight_contig.data_ptr<c10::qint8>();
      Tensor qnnp_weight = at::_empty_affine_quantized(
          weight_contig.sizes(),
          at::device(kCPU).dtype(kQUInt8),
          kernel_scale,
          kernel_zp,
          MemoryFormat::ChannelsLast);
      auto* qnnp_w_data = qnnp_weight.data_ptr<c10::quint8>();
      auto wt_numel = weight_contig.numel();
      for (int i = 0; i < wt_numel; ++i) {
        qnnp_w_data[i] = static_cast<c10::quint8>(w_data[i] + 128);
      }
      // Original bias was float, so we requantize it here.
      auto bias = at::quantize_per_tensor(
          bias_fp32, kernel_scale * input_scale, 0, kQInt32);
      // Update the input scale to not pack again.
      pack_ptr.input_scale = input_scale;
      pack_ptr.w.reset();
      pack_ptr.w = guts::make_unique<qnnpack::PrePackConvWeights>(
          conv_p,
          (uint8_t*)qnnp_w_data,
          (int32_t*)bias.data_ptr<c10::qint32>());
      packB = pack_ptr.w.get();
    }
    TORCH_INTERNAL_ASSERT(packB != nullptr, "Packed Weights are NULL");
    auto outShape =
        convOutputShape(N, K, H, W, kernel, stride, padding, dilation);
    TORCH_CHECK(
        std::all_of(
            outShape.begin(), outShape.end(), [](int64_t i) { return i > 0; }),
        "quantized::conv2d (qnnpack): each dimension of output tensor should be greater "
        "than 0")
    TORCH_CHECK(
        (outShape[3] == out_ch),
        "quantized::conv2d (qnnpack): Number of filters must be equal to number of "
        "output channels")

    // Allocate output Tensor and a buffer for QNNPACK to use
    Tensor output = at::_empty_affine_quantized(
        outShape,
        at::device(kCPU).dtype(kQUInt8),
        output_scale,
        output_zero_point);

    const pytorch_qnnp_status runStatus = qnnpack::qnnpackConv(
        conv_p,
        packB->getPackedWeights(),
        N,
        H,
        W,
        input_contig.q_scale(),
        input_contig.q_zero_point(),
        (uint8_t*)input_contig.data_ptr<c10::quint8>(),
        output.q_scale(),
        output.q_zero_point(),
        (uint8_t*)output.data_ptr<c10::quint8>(),
        caffe2::mobile_pthreadpool());

    TORCH_INTERNAL_ASSERT(
        runStatus == pytorch_qnnp_status_success,
        "failed to run quantized::conv2d (qnnpack) operator");

    // TODO: remove permute once MemoryLayout is added above
    return output.permute({0, 3, 1, 2});
  }
#endif
  Tensor operator()(
      Tensor act,
      Tensor packed_weight,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      double output_scale,
      int64_t output_zero_point) {
    auto& ctx = at::globalContext();
#ifdef USE_FBGEMM
#if AT_MKLDNN_ENABLED()
    bool is_mkldnn_weight =
        cpp_custom_type_hack::is_type<PackedConvWeightQmkldnn>(
            packed_weight);

    if (is_mkldnn_weight && use_mkldnn(act, output_zero_point, ReluFused)) {
      return mkldnn_qconv2d(
          act,
          packed_weight,
          stride,
          padding,
          dilation,
          groups,
          output_scale,
          output_zero_point);
      // We will fallback to FBGEMM if input don't meet mkldnn requirement even
      // if weights is mkldnn tensor.
    } else if (at::globalContext().qEngine() == at::kQMKLDNN) {
      Tensor repacked_weight;
      if (is_mkldnn_weight) {
        auto weights_org = mkldnn_conv_unpack(packed_weight);
        repacked_weight = fbgemm_conv_prepack(
            std::get<0>(weights_org),
            std::get<1>(weights_org),
            stride,
            padding,
            dilation,
            groups);
      } else {
        repacked_weight = packed_weight;
      }
      return fbgemm_conv(
          act,
          repacked_weight,
          stride,
          padding,
          dilation,
          groups,
          output_scale,
          output_zero_point);
    }
#endif // AT_MKLDNN_ENABLED()
    if (ctx.qEngine() == at::QEngine::FBGEMM) {
      return fbgemm_conv(
          act,
          packed_weight,
          stride,
          padding,
          dilation,
          groups,
          output_scale,
          output_zero_point);
    }
#endif // USE_FBGEMM
#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      return qnnpack_conv(
          act,
          packed_weight,
          stride,
          padding,
          dilation,
          groups,
          output_scale,
          output_zero_point);
    }
#endif
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::conv ",
        toString(ctx.qEngine()));
  }
};

static auto registry =
    c10::RegisterOperators()
        .op("quantized::conv2d",
            c10::RegisterOperators::options().kernel<QConv2dInt8<false>>(
                TensorTypeId::QuantizedCPUTensorId))
        .op("quantized::conv2d_relu",
            c10::RegisterOperators::options().kernel<QConv2dInt8<true>>(
                TensorTypeId::QuantizedCPUTensorId));

} // namespace
} // namespace native
} // namespace at
