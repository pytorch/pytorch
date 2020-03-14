#include <algorithm>
#include <cmath>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/SmallVector.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <caffe2/utils/threadpool/ThreadPoolMobile.h>

namespace at {
namespace native {
namespace {

template <int kSpatialDim = 2>
bool ConvDimChecks(
    int64_t act_dims,
    int64_t stride_dims,
    int64_t padding_dims,
    int64_t dilation_dims) {
  TORCH_CHECK(
      act_dims == kSpatialDim + 2,
      "quantized::conv",
      kSpatialDim,
      "d(): Expected activation tensor to have ",
      kSpatialDim + 2,
      " dimensions.");
  TORCH_CHECK(
      stride_dims == kSpatialDim,
      "quantized::conv",
      kSpatialDim,
      "d(): Expected stride tensor to have ",
      kSpatialDim,
      " dimensions.");
  TORCH_CHECK(
      padding_dims == kSpatialDim,
      "quantized::conv",
      kSpatialDim,
      "d(): Expected padding tensor to have ",
      kSpatialDim,
      " dimensions.");
  TORCH_CHECK(
      dilation_dims == kSpatialDim,
      "quantized::conv",
      kSpatialDim,
      "d(): Expected dilation tensor to have ",
      kSpatialDim,
      " dimensions.");
  return true;
}

#ifdef USE_FBGEMM

template <int kSpatialDim = 2>
SmallVector<int64_t, kSpatialDim + 2> MakeConvOutputShape(
    int N,
    int M,
    const std::array<int, kSpatialDim>& output_image_shape);

template <>
SmallVector<int64_t, 4> MakeConvOutputShape<2>(
    int N,
    int M,
    const std::array<int, 2>& output_image_shape) {
  return {N, M, output_image_shape[0], output_image_shape[1]};
}

template <>
SmallVector<int64_t, 5> MakeConvOutputShape<3>(
    int N,
    int M,
    const std::array<int, 3>& output_image_shape) {
  return {N,
          M,
          output_image_shape[0],
          output_image_shape[1],
          output_image_shape[2]};
}

#endif // USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK

template <int kSpatialDim>
SmallVector<int64_t, kSpatialDim + 2> MakeConvOutputShape(
    int N, // mini-batch
    int M, // output channels
    const std::vector<int>& input_image_shape,
    const std::vector<int64_t>& kernel,
    const torch::List<int64_t>& stride,
    const torch::List<int64_t>& padding,
    const torch::List<int64_t>& dilation);

template <>
SmallVector<int64_t, 4> MakeConvOutputShape<2>(
    int N, // mini-batch
    int M, // output channels
    const std::vector<int>& input_image_shape,
    const std::vector<int64_t>& kernel,
    const torch::List<int64_t>& stride,
    const torch::List<int64_t>& padding,
    const torch::List<int64_t>& dilation) {
  const int H = input_image_shape[0];
  const int W = input_image_shape[1];
  const int64_t Y_H =
      (H + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1;
  const int64_t Y_W =
      (W + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1;
  return {N, M, Y_H, Y_W};
}

template <>
SmallVector<int64_t, 5> MakeConvOutputShape<3>(
    int N, // mini-batch
    int M, // output channels
    const std::vector<int>& input_image_shape,
    const std::vector<int64_t>& kernel,
    const torch::List<int64_t>& stride,
    const torch::List<int64_t>& padding,
    const torch::List<int64_t>& dilation) {
  const int D = input_image_shape[0];
  const int H = input_image_shape[1];
  const int W = input_image_shape[2];
  const int64_t Y_D =
      (D + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1;
  const int64_t Y_H =
      (H + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1;
  const int64_t Y_W =
      (W + 2 * padding[2] - dilation[2] * (kernel[2] - 1) - 1) / stride[2] + 1;
  return {N, M, Y_D, Y_H, Y_W};
}

#endif // USE_PYTORCH_QNNPACK

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
template <int kSpatialDim, bool kReluFused>
class QConvInt8 final : public c10::OperatorKernel {
 public:
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
    if (ctx.qEngine() == at::QEngine::FBGEMM) {
      return FbgemmConv(
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
      TORCH_CHECK(kSpatialDim == 2, "QNNPACK only suuports Conv2d now.");
      return QnnpackConv(
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

 private:
#ifdef USE_FBGEMM
  static const float* GetBiasData(
      const PackedConvWeight<kSpatialDim>& pack_data,
      Tensor* bias) {
    const float* bias_data = nullptr;
    if (pack_data.bias.has_value()) {
      *bias = pack_data.bias.value();
      TORCH_CHECK(
          bias->dtype() == at::kFloat,
          "[QConv3D] The 'bias' tensor must have 'torch.float' dtype");
      *bias = bias->contiguous();
      TORCH_CHECK(bias->dim() == 1, "bias should be a vector (1D Tensor)");
      const int M = pack_data.w->outputChannels();
      TORCH_CHECK(bias->size(0) == M, "bias should have ", M, " elements.");
      bias_data = bias->data_ptr<float>();
    }
    return bias_data;
  }

  static void GetQuantizationParams(
      const PackedConvWeight<kSpatialDim>& pack_data,
      float act_scale,
      float out_scale,
      std::vector<float>* output_multiplier_float,
      std::vector<float>* act_times_w_scale) {
    if (pack_data.q_scheme == kPerTensorAffine) {
      *act_times_w_scale = {(act_scale * pack_data.w_scale[0])};
      *output_multiplier_float = {act_times_w_scale->front() / out_scale};
    } else if (pack_data.q_scheme == kPerChannelAffine) {
      const int M = pack_data.w->outputChannels();
      output_multiplier_float->resize(M);
      act_times_w_scale->resize(M);
      for (int i = 0; i < M; ++i) {
        act_times_w_scale->at(i) = (act_scale * pack_data.w_scale[i]);
        output_multiplier_float->at(i) = act_times_w_scale->at(i) / out_scale;
      }
    } else {
      TORCH_CHECK(
          false, "[QConv", kSpatialDim, "D] Unknown quantization scheme");
    }
  }

  at::Tensor FbgemmConv(
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
    ConvDimChecks<kSpatialDim>(
        act.ndimension(), stride.size(), padding.size(), dilation.size());

    const int N = act.size(0);
    const int C = act.size(1);
    const int D = kSpatialDim == 2 ? 1 : act.size(2);
    const int H = act.size(kSpatialDim);
    const int W = act.size(kSpatialDim + 1);

    const Tensor act_nhwc = kSpatialDim == 2
        ? act.contiguous(MemoryFormat::ChannelsLast)
        : fbgemm_utils::ConvertToChannelsLast3dTensor(act);
    const uint8_t* act_data =
        reinterpret_cast<uint8_t*>(act_nhwc.data_ptr<c10::quint8>());
    PackedConvWeight<kSpatialDim>& pack_data =
        cpp_custom_type_hack::cast<PackedConvWeight<kSpatialDim>>(
            packed_weight);
    auto* pack_w = pack_data.w.get();
    const auto& col_offsets = pack_data.col_offsets;
    const auto& kernel = pack_data.kernel;

    const int M = pack_w->outputChannels();
    const int kernel_d = kSpatialDim == 2 ? 1 : kernel[0];
    const int kernel_h = kernel[kSpatialDim - 2];
    const int kernel_w = kernel[kSpatialDim - 1];
    const int pad_d = kSpatialDim == 2 ? 0 : padding[0];
    const int pad_h = padding[kSpatialDim - 2];
    const int pad_w = padding[kSpatialDim - 1];
    const int stride_d = kSpatialDim == 2 ? 1 : stride[0];
    const int stride_h = stride[kSpatialDim - 2];
    const int stride_w = stride[kSpatialDim - 1];
    const int dilation_d = kSpatialDim == 2 ? 1 : dilation[0];
    const int dilation_h = dilation[kSpatialDim - 2];
    const int dilation_w = dilation[kSpatialDim - 1];

    if (kSpatialDim == 2) {
      TORCH_CHECK(
          C == pack_w->inputChannels(),
          "[QConv2D] Given groups=",
          groups,
          ", weight of size ",
          M,
          ", ",
          kernel_h,
          ", ",
          kernel_w,
          ", ",
          pack_w->inputChannels(),
          ", expected input (NCHW) ",
          N,
          ", ",
          C,
          ", ",
          H,
          ", ",
          W,
          " to have ",
          pack_w->inputChannels(),
          " channels, but got ",
          C,
          " channels instead");
    } else {
      TORCH_CHECK(
          C == pack_w->inputChannels(),
          "[QConv3D] Given groups=",
          groups,
          ", weight of size ",
          M,
          ", ",
          kernel_d,
          ", ",
          kernel_h,
          ", ",
          kernel_w,
          ", ",
          pack_w->inputChannels(),
          ", expected input (NCDHW) ",
          N,
          ", ",
          C,
          ", ",
          D,
          ", ",
          H,
          ", ",
          W,
          " to have ",
          pack_w->inputChannels(),
          " channels, but got ",
          C,
          " channels instead");
    }

    fbgemm::conv_param_t<kSpatialDim> conv_p =
        fbgemm_utils::MakeFbgemmConvParam<kSpatialDim>(
            N, // Batch size
            C, // Number of input channels
            M, // Number of output channels
            kSpatialDim == 2 ? std::vector<int>{H, W}
                             : std::vector<int>{D, H, W},
            groups,
            kSpatialDim == 2 ? std::vector<int>{kernel_h, kernel_w}
                             : std::vector<int>{kernel_d, kernel_h, kernel_w},
            kSpatialDim == 2 ? std::vector<int>{stride_h, stride_w}
                             : std::vector<int>{stride_d, stride_h, stride_w},
            kSpatialDim == 2 ? std::vector<int>{pad_h, pad_w}
                             : std::vector<int>{pad_d, pad_h, pad_w},
            kSpatialDim == 2
                ? std::vector<int>{dilation_h, dilation_w}
                : std::vector<int>{dilation_d, dilation_h, dilation_w});

    const float act_scale = act.q_scale();
    const int32_t act_zero_point = act.q_zero_point();

    Tensor bias;
    const float* bias_data = GetBiasData(pack_data, &bias);

    TORCH_CHECK(
        pack_data.w_scale.size() == pack_data.w_zp.size(),
        "Weight scales and zero points vectors should have the same size.");
    std::vector<float> output_multiplier_float;
    std::vector<float> act_times_w_scale;
    GetQuantizationParams(
        pack_data,
        act_scale,
        output_scale,
        &output_multiplier_float,
        &act_times_w_scale);

    const SmallVector<int64_t, kSpatialDim + 2> output_shape =
        MakeConvOutputShape<kSpatialDim>(N, M, conv_p.OUT_DIM);
    TORCH_CHECK(
        std::all_of(
            output_shape.begin(),
            output_shape.end(),
            [](int64_t i) { return i > 0; }),
        "[QConv",
        kSpatialDim,
        "D] each dimension of output tensor should be greater than 0");

    Tensor output = kSpatialDim == 2
        ? _empty_affine_quantized(
              output_shape,
              device(kCPU).dtype(kQUInt8),
              output_scale,
              output_zero_point,
              MemoryFormat::ChannelsLast)
        : fbgemm_utils::MakeEmptyAffineQuantizedChannelsLast3dTensor(
              output_shape[0],
              output_shape[1],
              output_shape[2],
              output_shape[3],
              output_shape[4],
              device(kCPU).dtype(kQUInt8),
              output_scale,
              output_zero_point);
    Tensor buffer = at::empty(output.sizes(), output.options().dtype(at::kInt));
    const int num_tasks = at::get_num_threads();
    at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
      fbgemm::DoNothing<> kNoOpObj{};
      for (int task_id = begin; task_id < end; ++task_id) {
        if (pack_data.q_scheme == kPerTensorAffine) {
          fbgemm::ReQuantizeOutput<
              kReluFused,
              fbgemm::QuantizationGranularity::TENSOR,
              float>
              output_proc_obj(
                  kNoOpObj,
                  output_multiplier_float.data(),
                  output_zero_point,
                  act_zero_point,
                  pack_data.w_zp.data(),
                  nullptr, /* row offset buffer */
                  col_offsets.data(),
                  bias_data,
                  M,
                  groups,
                  act_times_w_scale.data());
          fbgemm::fbgemmConv<decltype(output_proc_obj), kSpatialDim, int32_t>(
              conv_p,
              act_data,
              *pack_w,
              reinterpret_cast<uint8_t*>(output.data_ptr<c10::quint8>()),
              buffer.data_ptr<int32_t>(),
              output_proc_obj,
              task_id /* thread_id*/,
              num_tasks /* num_threads */);
        } else if (pack_data.q_scheme == kPerChannelAffine) {
          fbgemm::ReQuantizeOutput<
              kReluFused,
              fbgemm::QuantizationGranularity::OUT_CHANNEL,
              float>
              output_proc_obj(
                  kNoOpObj,
                  output_multiplier_float.data(),
                  output_zero_point,
                  act_zero_point,
                  pack_data.w_zp.data(),
                  nullptr, /* row offset buffer */
                  col_offsets.data(),
                  bias_data,
                  M,
                  groups,
                  act_times_w_scale.data());

          fbgemm::fbgemmConv<decltype(output_proc_obj), kSpatialDim, int32_t>(
              conv_p,
              act_data,
              *pack_w,
              reinterpret_cast<uint8_t*>(output.data_ptr<c10::quint8>()),
              buffer.data_ptr<int32_t>(),
              output_proc_obj,
              task_id /* thread_id*/,
              num_tasks /* num_threads */);
        }
      }
    });

    return output;
  }
#endif

#ifdef USE_PYTORCH_QNNPACK
  at::Tensor QnnpackConv(
      Tensor act,
      Tensor packed_weight,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      double output_scale,
      int64_t output_zero_point) {
    ConvDimChecks<kSpatialDim>(
        act.ndimension(), stride.size(), padding.size(), dilation.size());

    PackedConvWeightsQnnp& pack_data =
        cpp_custom_type_hack::cast<PackedConvWeightsQnnp>(packed_weight);
    auto* pack_w = pack_data.w.get();
    const auto& kernel = pack_data.kernel;
    // Adjust weight zero point, similar to weight data.
    const auto kernel_zp = pack_data.w_zp + 128;
    const auto& kernel_scale = pack_data.w_scale;

    const uint32_t kernel_h = kernel[0];
    const uint32_t kernel_w = kernel[1];
    // TODO Can be replaced with packB->getOutputChannels() when update pre-pack
    // to actually do the packing.
    const auto out_ch = pack_data.bias.size(0);
    // inputs are in semantic NCHW format
    const int N = act.size(0);
    const int C = act.size(1);
    const int H = act.size(2);
    const int W = act.size(3);
    const int M = out_ch; // output channels

    const Tensor act_nhwc = act.contiguous(MemoryFormat::ChannelsLast);

    const uint32_t stride_h = stride[0];
    const uint32_t stride_w = stride[1];
    const uint32_t pad_h = padding[0];
    const uint32_t pad_w = padding[1];
    const uint32_t dilation_h = dilation[0];
    const uint32_t dilation_w = dilation[1];

    auto output_min = kReluFused
        ? activationLimits(output_scale, output_zero_point, Activation::RELU)
              .first
        : std::numeric_limits<uint8_t>::min();
    auto output_max = kReluFused
        ? activationLimits(output_scale, output_zero_point, Activation::RELU)
              .second
        : std::numeric_limits<uint8_t>::max();
    qnnpack::conv_param_t conv_p(
        {kernel_w, kernel_h},
        {stride_w, stride_h},
        {dilation_w, dilation_h},
        {pad_h, pad_w, pad_h, pad_w},
        groups,
        C,
        M,
        kernel_zp,
        kernel_scale,
        output_min,
        output_max);

    auto input_scale = act_nhwc.q_scale();

    // Re-quantizing the bias based on input scale and weight scale.
    if (!pack_data.input_scale.has_value() ||
        pack_data.input_scale.value() != input_scale) {
      // Get the original weight and adjust it to uint8 from int8
      auto weight_contig =
          pack_data.orig_weight.contiguous(MemoryFormat::ChannelsLast);
      auto bias_fp32 = pack_data.bias;
      int8_t* w_data =
          reinterpret_cast<int8_t*>(weight_contig.data_ptr<c10::qint8>());
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
      pack_data.input_scale = input_scale;
      pack_data.w.reset();
      pack_data.w = std::make_unique<qnnpack::PrePackConvWeights>(
          conv_p,
          reinterpret_cast<uint8_t*>(qnnp_w_data),
          reinterpret_cast<int32_t*>(bias.data_ptr<c10::qint32>()));
      pack_w = pack_data.w.get();
    }
    TORCH_INTERNAL_ASSERT(pack_w != nullptr, "Packed Weights are NULL");
    const auto output_shape = MakeConvOutputShape<kSpatialDim>(
        N, M, {H, W}, kernel, stride, padding, dilation);
    TORCH_CHECK(
        std::all_of(
            output_shape.begin(),
            output_shape.end(),
            [](int64_t i) { return i > 0; }),
        "quantized::conv2d (qnnpack): each dimension of output tensor should "
        "be greater than 0.")

    // Allocate output Tensor and a buffer for QNNPACK to use
    Tensor output = at::_empty_affine_quantized(
        output_shape,
        at::device(kCPU).dtype(kQUInt8),
        output_scale,
        output_zero_point,
        MemoryFormat::ChannelsLast);

    const pytorch_qnnp_status run_status = qnnpack::qnnpackConv(
        conv_p,
        pack_w->getPackedWeights(),
        N,
        H,
        W,
        act_nhwc.q_scale(),
        act_nhwc.q_zero_point(),
        reinterpret_cast<uint8_t*>(act_nhwc.data_ptr<c10::quint8>()),
        output.q_scale(),
        output.q_zero_point(),
        reinterpret_cast<uint8_t*>(output.data_ptr<c10::quint8>()),
        caffe2::mobile_pthreadpool());

    TORCH_INTERNAL_ASSERT(
        run_status == pytorch_qnnp_status_success,
        "failed to run quantized::conv2d (qnnpack) operator");

    return output;
  }
#endif
};

static auto registry =
    c10::RegisterOperators()
        .op("quantized::conv2d",
            c10::RegisterOperators::options().kernel<QConvInt8<2, false>>(
                DispatchKey::QuantizedCPUTensorId))
        .op("quantized::conv2d_relu",
            c10::RegisterOperators::options().kernel<QConvInt8<2, true>>(
                DispatchKey::QuantizedCPUTensorId))
        .op("quantized::conv3d",
            c10::RegisterOperators::options().kernel<QConvInt8<3, false>>(
                DispatchKey::QuantizedCPUTensorId))
        .op("quantized::conv3d_relu",
            c10::RegisterOperators::options().kernel<QConvInt8<3, true>>(
                DispatchKey::QuantizedCPUTensorId));

} // namespace
} // namespace native
} // namespace at
