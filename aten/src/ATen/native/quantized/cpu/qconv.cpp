#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <ATen/TensorOperators.h>
#include <ATen/SmallVector.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/XnnpackUtils.h>
#include <ATen/native/quantized/cpu/OnednnUtils.h>
#include <ATen/native/quantized/ConvUtils.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <ATen/native/quantized/cpu/qconv.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <torch/library.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_empty_affine_quantized_native.h>
#include <ATen/ops/_empty_per_channel_affine_quantized_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/quantize_per_channel_native.h>
#include <ATen/ops/quantize_per_tensor_native.h>
#include <ATen/ops/zeros.h>
#endif

#include <c10/util/irange.h>

namespace {
// To have a sanity check for maximum matrix size.
constexpr int64_t kReasonableMaxDim = 1000000;
} // namespace

template <int kSpatialDim = 2>
bool ConvDimChecks(
    int64_t act_dims,
    int64_t stride_dims,
    int64_t padding_dims,
    int64_t output_padding_dims,
    int64_t dilation_dims,
    std::string func_name,
    bool transpose = false) {
  TORCH_CHECK(
      act_dims == kSpatialDim + 2,
      func_name,
      kSpatialDim,
      "d(): Expected activation tensor to have ",
      kSpatialDim + 2,
      " dimensions, got ",
      act_dims);
  TORCH_CHECK(
      stride_dims == kSpatialDim,
      func_name,
      kSpatialDim,
      "d(): Expected stride tensor to have ",
      kSpatialDim,
      " dimensions, got ",
      stride_dims);
  TORCH_CHECK(
      padding_dims == kSpatialDim,
      func_name,
      kSpatialDim,
      "d(): Expected padding tensor to have ",
      kSpatialDim,
      " dimensions, got ",
      padding_dims);
  TORCH_CHECK(
      !transpose || (output_padding_dims == kSpatialDim),
      func_name,
      kSpatialDim,
      "d(): Expected output padding tensor to have ",
      kSpatialDim,
      " dimensions, got ",
      output_padding_dims);
  TORCH_CHECK(
      dilation_dims == kSpatialDim,
      func_name,
      kSpatialDim,
      "d(): Expected dilation tensor to have ",
      kSpatialDim,
      " dimensions, got ",
      dilation_dims);
  return true;
}

inline int64_t compute_deconv_shape(int64_t input,
                                    int64_t kernel,
                                    int64_t stride,
                                    int64_t input_padding,
                                    int64_t output_padding,
                                    int64_t dilation) {
  int64_t out = (input - 1) * stride - 2 * input_padding
                + dilation * (kernel - 1) + output_padding + 1;
  return out;
}

template <int64_t kSpatialDim>
at::SmallVector<int64_t, kSpatialDim + 2> MakeDeConvOutputShape(
    int64_t N, int64_t M,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& kernel,
    const torch::List<int64_t>& stride,
    const torch::List<int64_t>& input_padding,
    const torch::List<int64_t>& output_padding,
    const torch::List<int64_t>& dilation) {
  at::SmallVector<int64_t, kSpatialDim + 2> output_shape;
  output_shape.resize(kSpatialDim + 2);
  output_shape[0] = N;  // Batch size
  output_shape[1] = M;  // Output channels
  for (const auto idx : c10::irange(kSpatialDim)) {
    output_shape[idx + 2] = compute_deconv_shape(input_shape[idx],
                                                 kernel[idx],
                                                 stride[idx],
                                                 input_padding[idx],
                                                 output_padding[idx],
                                                 dilation[idx]);
    TORCH_CHECK(output_shape[idx + 2] > 0,
                "Output dimension is zero for ", idx, " axis;"
                " kernel: ", kernel[idx],
                ", stride: ", stride[idx],
                ", input padding: ", input_padding[idx],
                ", output padding: ", output_padding[idx],
                ", dilation: ", dilation[idx])
    TORCH_CHECK(output_shape[idx + 2] < kReasonableMaxDim,
                "Output dimension is beyond reasonable maximum for ", idx,
                " axis;"
                " kernel: ", kernel[idx],
                ", stride: ", stride[idx],
                ", input padding: ", input_padding[idx],
                ", output padding: ", output_padding[idx],
                ", dilation: ", dilation[idx]);
  }
  return output_shape;
}

#ifdef USE_FBGEMM

template <int kSpatialDim = 2>
at::SmallVector<int64_t, kSpatialDim + 2> MakeConvOutputShape(
    int N,
    int M,
    const std::array<int, kSpatialDim>& output_image_shape);

template <>
at::SmallVector<int64_t, 4> MakeConvOutputShape<2>(
    int N,
    int M,
    const std::array<int, 2>& output_image_shape) {
  return {N, M, output_image_shape[0], output_image_shape[1]};
}

template <>
at::SmallVector<int64_t, 5> MakeConvOutputShape<3>(
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

template <size_t kSpatialDim>
std::array<int64_t, kSpatialDim> MakeInputShape(
    int64_t D,
    int64_t H,
    int64_t W);

template <>
std::array<int64_t, 2> MakeInputShape(int64_t /*D*/, int64_t H, int64_t W) {
  return {H, W};
}
template <>
std::array<int64_t, 3> MakeInputShape(int64_t D, int64_t H, int64_t W) {
  return {D, H, W};
}

#endif // USE_PYTORCH_QNNPACK

#ifdef USE_FBGEMM
template <int kSpatialDim>
const float* PackedConvWeight<kSpatialDim>::GetBiasData(at::Tensor* bias_ptr) {
  const float* bias_data = nullptr;
  if (bias.has_value()) {
    *bias_ptr = bias.value();
    TORCH_CHECK(
        bias_ptr->dtype() == at::kFloat,
        "[QConv3D] The 'bias' tensor must have 'torch.float' dtype");
    *bias_ptr = bias_ptr->contiguous();
    TORCH_CHECK(bias_ptr->dim() == 1, "bias should be a vector (1D Tensor)");
    const int M = w->outputChannels();
    TORCH_CHECK(bias_ptr->size(0) == M, "bias should have ", M, " elements.");
    bias_data = bias_ptr->data_ptr<float>();
  }
  return bias_data;
}

template <int kSpatialDim>
void PackedConvWeight<kSpatialDim>::GetQuantizationParams(
    float act_scale,
    float out_scale,
    std::vector<float>* output_multiplier_float,
    std::vector<float>* act_times_w_scale) {
  if (q_scheme == c10::kPerTensorAffine) {
    *act_times_w_scale = {(act_scale * w_scale[0])};
    *output_multiplier_float = {act_times_w_scale->front() / out_scale};
  } else if (q_scheme == c10::kPerChannelAffine) {
    const int M = w->outputChannels();
    output_multiplier_float->resize(M);
    act_times_w_scale->resize(M);
    for (const auto i : c10::irange(M)) {
      act_times_w_scale->at(i) = (act_scale * w_scale[i]);
      output_multiplier_float->at(i) = act_times_w_scale->at(i) / out_scale;
    }
  } else {
    TORCH_CHECK(false, "[QConv", kSpatialDim, "D] Unknown quantization scheme");
  }
}

template <int kSpatialDim>
at::Tensor PackedConvWeight<kSpatialDim>::apply(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<false>(input, output_scale, output_zero_point);
}

template <int kSpatialDim>
at::Tensor PackedConvWeight<kSpatialDim>::apply_relu(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<true>(input, output_scale, output_zero_point);
}

template <int kSpatialDim>
template <bool kReluFused>
at::Tensor PackedConvWeight<kSpatialDim>::apply_impl(
    const at::Tensor& act,
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
  const std::string func_name = transpose() ? "quantized::conv_transpose"
                                            : "quantized::conv";
  TORCH_CHECK(
      fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");
  TORCH_CHECK(act.scalar_type() == c10::kQUInt8,
                func_name,
                "(FBGEMM): Expected activation data type ",
                toString(c10::kQUInt8),
                " but got ",
                toString(act.scalar_type()));

  ConvDimChecks<kSpatialDim>(
      act.ndimension(), stride().size(), padding().size(),
      output_padding().size(), dilation().size(), func_name, transpose());

  const int N = act.size(0);
  const int C = act.size(1);
  const int D = kSpatialDim == 2 ? 1 : act.size(2);
  const int H = act.size(kSpatialDim);
  const int W = act.size(kSpatialDim + 1);

  const at::Tensor act_ndhwc = kSpatialDim == 2
      ? act.contiguous(c10::MemoryFormat::ChannelsLast)
      : at::native::fbgemm_utils::ConvertToChannelsLast3dTensor(act);
  const uint8_t* act_data =
      reinterpret_cast<uint8_t*>(act_ndhwc.data_ptr<c10::quint8>());
  auto* pack_w = w.get();

  const int M = pack_w->outputChannels();
  const int kernel_d = kSpatialDim == 2 ? 1 : kernel[0];
  const int kernel_h = kernel[kSpatialDim - 2];
  const int kernel_w = kernel[kSpatialDim - 1];
  const int pad_d = kSpatialDim == 2 ? 0 : padding_[0];
  const int pad_h = padding_[kSpatialDim - 2];
  const int pad_w = padding_[kSpatialDim - 1];
  const int stride_d = kSpatialDim == 2 ? 1 : stride_[0];
  const int stride_h = stride_[kSpatialDim - 2];
  const int stride_w = stride_[kSpatialDim - 1];
  const int dilation_d = kSpatialDim == 2 ? 1 : dilation_[0];
  const int dilation_h = dilation_[kSpatialDim - 2];
  const int dilation_w = dilation_[kSpatialDim - 1];
  const int output_padding_d = kSpatialDim == 2 ? 0 : output_padding_[0];
  const int output_padding_h = output_padding_[kSpatialDim - 2];
  const int output_padding_w = output_padding_[kSpatialDim - 1];

  if (kSpatialDim == 2) {
    TORCH_CHECK(
        C == pack_w->inputChannels(),
        "[QConv2D] Given groups=",
        groups_,
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
        groups_,
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
      at::native::fbgemm_utils::MakeFbgemmConvParam<kSpatialDim>(
          N, // Batch size
          C, // Number of input channels
          M, // Number of output channels
          kSpatialDim == 2 ? std::vector<int>{H, W} : std::vector<int>{D, H, W},
          groups_,
          kSpatialDim == 2 ? std::vector<int>{kernel_h, kernel_w}
                           : std::vector<int>{kernel_d, kernel_h, kernel_w},
          kSpatialDim == 2 ? std::vector<int>{stride_h, stride_w}
                           : std::vector<int>{stride_d, stride_h, stride_w},
          kSpatialDim == 2 ? std::vector<int>{pad_h, pad_w}
                           : std::vector<int>{pad_d, pad_h, pad_w},
          kSpatialDim == 2
              ? std::vector<int>{dilation_h, dilation_w}
              : std::vector<int>{dilation_d, dilation_h, dilation_w},
          kSpatialDim == 2
              ? std::vector<int>{output_padding_h, output_padding_w}
              : std::vector<int>{output_padding_d,
                                 output_padding_h,
                                 output_padding_w},
          transpose());

  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  const float act_scale = act.q_scale();
  const int32_t act_zero_point = act.q_zero_point();

  at::Tensor bias;
  const float* bias_data = GetBiasData(&bias);

  TORCH_CHECK(
      w_scale.size() == w_zp.size(),
      "Weight scales and zero points vectors should have the same size.");
  std::vector<float> output_multiplier_float;
  std::vector<float> act_times_w_scale;
  GetQuantizationParams(
      act_scale, output_scale, &output_multiplier_float, &act_times_w_scale);

  at::SmallVector<int64_t, kSpatialDim + 2> output_shape;
  if (transpose()) {
    output_shape = MakeDeConvOutputShape<kSpatialDim>(
        N,
        M,
        kSpatialDim == 2 ? std::vector<int64_t>{H, W} : std::vector<int64_t>{D, H, W},
        kernel,
        stride(),
        padding(),
        output_padding(),
        dilation());

    // if use direct convolution implementation, compute the col_offsets
    // of the weight matrix at model initialization stage.
    // We need to know the shape of output matrix
    // to compute col_offsets for direct convolution.
    // Hence it cannot be called from inside weight packing function
    // like other quantized conv implementation
    if (pack_w->getPackedWForDirectconv().get() &&
        pack_w->getPackedWForDirectconv().get()->is_first_call()) {
          pack_w->getPackedWForDirectconv().get()->col_offsets_with_zero_pt_s8acc32_DirectConvT(
              conv_p,
              w_zp.data(),
              col_offsets,
              M);
    }
  } else {
    output_shape = MakeConvOutputShape<kSpatialDim>(N, M, conv_p.OUT_DIM);
  }
  if (N > 0) {
    TORCH_CHECK(
        std::all_of(
            output_shape.begin(),
            output_shape.end(),
            [](int64_t i) { return i > 0; }),
        "[QConv",
        kSpatialDim,
        "D] each dimension of output tensor should be greater than 0");
  }
  at::Tensor output = kSpatialDim == 2
      ? at::_empty_affine_quantized(
            output_shape,
            device(c10::kCPU)
                .dtype(c10::kQUInt8)
                .memory_format(c10::MemoryFormat::ChannelsLast),
            output_scale,
            output_zero_point,
            std::nullopt)
      : at::native::fbgemm_utils::MakeEmptyAffineQuantizedChannelsLast3dTensor(
            output_shape[0],
            output_shape[1],
            output_shape[2],
            output_shape[3],
            output_shape[4],
            device(c10::kCPU).dtype(c10::kQUInt8),
            output_scale,
            output_zero_point);
  at::Tensor buffer =
      at::empty(output.sizes(), output.options().dtype(c10::kInt));
  const int num_tasks = at::get_num_threads();
  at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
    fbgemm::DoNothing<> kNoOpObj{};
    for (const auto task_id : c10::irange(begin, end)) {
      if (q_scheme == c10::kPerTensorAffine) {
        fbgemm::ReQuantizeOutput<
            kReluFused,
            fbgemm::QuantizationGranularity::TENSOR,
            float>
            output_proc_obj(
                kNoOpObj,
                output_multiplier_float.data(),
                output_zero_point,
                act_zero_point,
                w_zp.data(),
                nullptr, /* row offset buffer */
                col_offsets.data(),
                bias_data,
                M,
                groups_,
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
      } else if (q_scheme == c10::kPerChannelAffine) {
        fbgemm::ReQuantizeOutput<
            kReluFused,
            fbgemm::QuantizationGranularity::OUT_CHANNEL,
            float>
            output_proc_obj(
                kNoOpObj,
                output_multiplier_float.data(),
                output_zero_point,
                act_zero_point,
                w_zp.data(),
                nullptr, /* row offset buffer */
                col_offsets.data(),
                bias_data,
                M,
                groups_,
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

template at::Tensor PackedConvWeight<2>::apply(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeight<2>::apply_relu(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeight<3>::apply(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeight<3>::apply_relu(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeight<2>::apply_impl<false>(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeight<3>::apply_impl<false>(
  const at::Tensor& act,
  double output_scale,
  int64_t output_zero_point);

#endif // USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK

#ifdef USE_XNNPACK
template <int kSpatialDim>
template <typename scalar_t, bool kReluFused>
at::Tensor PackedConvWeightsQnnp<kSpatialDim>::apply_impl_xnnp(
    const at::Tensor& act, double output_scale, int64_t output_zero_point) {
  using underlying_t = typename scalar_t::underlying;

  std::lock_guard<std::mutex> lock(qnnp_mutex_);

  const std::string func_name = transpose()
      ? "quantized::conv_transpose (xnnpack)"
      : "quantized::conv (xnnpack)";
  TORCH_CHECK(
      kSpatialDim == 2,
      func_name, ": xnnpack does not currently support 3d convolution.");

  /*
   * NB:
   * [de]conv_prepack prepares weights (values, scale, and zero_points) ahead of
   * time during prepack() call assuming the activation will be uint8_t. But it
   * may not always be the case. A solution may involve making prepack routine
   * aware of the input qdtype. But currently all the pieces are not ready to
   * pass that model level info to the prepack function. So, for now, here in
   * this function we have to massage weights if we learn the input qdtype is
   * not uint8_t. This involves copying and converting uint8_t to int8_t
   * whenever necessary. To add to that, since XNNPACK, as of writing this,
   * doesn't support per_channel weights for quint8_t, we add following assert
   * makes sure we don't run into that case. Also take shortcuts when processing
   * weights, which means we have to revisit and fix some weight massging logic
   * when we enable the missing feature in XNNPACK.
   *
   * Table below summarizes how the weights are handled,
   *
   * .-------------------------------------------------------------------------.
   * | input_qdtype |              uint8_t            |            int8_t      |
   * | per_channel  |       yes       |       no      |      yes     |    no   |
   * |-------------------------------------------------------------------------|
   * | zero_points  | at::zeros()*    | orig_zp + 128 | at:zeros()** | orig_zp |
   * | scale        |            dtype = float, no changes needed              |
   * | values       |        always processed before passing to XNNPACK        |
   * .-------------------------------------------------------------------------.
   *
   * Notes: * - zero_points for uint8_t + per_channel: no support in xnnpack, need
   * to fix when support is added. ** - zero_points for int8_t: symmetric
   * quantization means XNNPACK will ignore kernel zero point(s).
   */

  if constexpr (std::is_same_v<underlying_t, c10::quint8>) {
    TORCH_CHECK(!per_channel(),
      func_name, ": xnnpack does not currently have per_channel support with activation dtype of c10::quint8."
    );
  }

  // More checks
  ConvDimChecks<kSpatialDim>(
      act.ndimension(),
      stride().size(),
      padding().size(),
      output_padding().size(),
      dilation().size(),
      func_name,
      transpose());

  const int64_t N = act.size(0);
  const int64_t H = act.size(2);
  const int64_t W = act.size(3);
  const int64_t D = 1;
  const int64_t M = bias.size(0);

  const auto act_nhwc = act.contiguous(c10::MemoryFormat::ChannelsLast);
  const auto act_input_scale = act_nhwc.q_scale();

  auto status = xnn_status_invalid_state;

  // Create an operator iff necessary
  if (!xnnp_convolution_op ||
      (!input_scale.has_value() || input_scale.value() != act_input_scale)) {
    xnn_operator_t xnnp_op = nullptr;

    // Update the input scale so we may cache the op
    input_scale = act_input_scale;

    // create an empty tensor for packing the weights
    const at::Tensor weight_contig =
        orig_weight.contiguous(c10::MemoryFormat::ChannelsLast);
    const float* w_scales_data = w_scales.const_data_ptr<float>();
    underlying_t w_zp = 0;
    at::Tensor weight_tensor;

    if (!per_channel()) {
      w_zp = static_cast<underlying_t>(
          weight_contig.q_zero_point() +
          (std::is_same_v<underlying_t, uint8_t> ? 128 : 0));

      weight_tensor = at::native::empty_affine_quantized(
          weight_contig.sizes(),
          c10::CppTypeToScalarType<scalar_t>::value,
          std::nullopt /* layout */,
          c10::kCPU,
          std::nullopt /* pin_memory */,
          w_scales_data[0],
          w_zp,
          c10::MemoryFormat::ChannelsLast);
    } else { /* per_channel */
      weight_tensor = at::native::empty_per_channel_affine_quantized(
          weight_contig.sizes(),
          w_scales,
          at::zeros(w_scales.sizes(), at::kInt), /* see comment above about w_zp */
          weight_contig.q_per_channel_axis(),
          c10::CppTypeToScalarType<scalar_t>::value,
          std::nullopt /* layout */,
          c10::kCPU,
          std::nullopt /* pin_memory */,
          c10::MemoryFormat::ChannelsLast);
    }

    // copy from the original weight and take care of dtype change if necessary
    at::native::xnnp_utils::q8_copy_int8_weight_and_add_offset<scalar_t>(
        weight_contig, weight_tensor);
    const at::Tensor xnnp_weight =
        at::native::xnnp_utils::convert_conv_weights_to_channel_last_tensor<
            kSpatialDim>(weight_tensor, groups(), transpose());

    auto output_min = kReluFused
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        ? activationLimits<underlying_t>(output_scale, output_zero_point, Activation::RELU).first
        : std::numeric_limits<underlying_t>::min();
    auto output_max = kReluFused
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        ? activationLimits<underlying_t>(output_scale, output_zero_point, Activation::RELU).second
        : std::numeric_limits<underlying_t>::max();


    // Original bias was float, so we requantize it here.
    at::Tensor qbias = quant_utils::QuantizeBias(per_channel(), bias, weight_contig, act_input_scale);

    status = at::native::xnnp_utils::xnnp_create_convolution2d_nhwc(
        padding()[0],
        padding()[1],
        padding()[0],
        padding()[1],
        kernel_[0],
        kernel_[1],
        stride()[0],
        stride()[1],
        dilation()[0],
        dilation()[1],
        groups(),
        !transpose() ? orig_weight.size(1) : orig_weight.size(0) / groups(),
        !transpose() ? orig_weight.size(0) / groups() : orig_weight.size(1),
        !transpose() ? orig_weight.size(1) * groups() : orig_weight.size(0),
        !transpose() ? orig_weight.size(0) : orig_weight.size(1) * groups(),
        act_nhwc.q_zero_point(),
        act_input_scale,
        w_zp, /* will be ignored for Q[SC]8, see comment
                above about w_zp*/
        w_scales_data,
        reinterpret_cast<const underlying_t*>(
            xnnp_weight.template data_ptr<scalar_t>()),
        reinterpret_cast<int32_t*>(qbias.template data_ptr<c10::qint32>()),
        output_zero_point,
        output_scale,
        output_min,
        output_max,
        0,
        &xnnp_op,
        per_channel(),
        transpose());

    xnnp_convolution_op = xnnpack_operator(xnnp_op);
    TORCH_CHECK(
        status == xnn_status_success,
        func_name,
        ": xnn create operator failed(",
        status,
        ")");
  }

  at::SmallVector<int64_t, kSpatialDim + 2> output_shape;
  const auto input_shape = MakeInputShape<kSpatialDim>(D, H, W);
  if (transpose()) {
    output_shape = MakeDeConvOutputShape<kSpatialDim>(
        N, M, {H, W}, kernel_, stride(), padding(), output_padding(), dilation());
  } else {
    output_shape = at::native::quantized::MakeConvOutputShape<kSpatialDim>(
        N, M, input_shape, kernel_, stride(), padding(), dilation());
  }

  if (act_nhwc.numel() > 0) {
    TORCH_CHECK(
        std::all_of(
            output_shape.begin(),
            output_shape.end(),
            [](int64_t i) { return i > 0; }),
        func_name, ": ", kSpatialDim, "d (xnnpack): each dimension of output tensor should be greater than 0.")
  }

  // Allocate output Tensor and a buffer for XNNPACK to use
  at::Tensor output = at::native::empty_affine_quantized(
      output_shape,
      c10::CppTypeToScalarType<scalar_t>::value,
      std::nullopt /* layout */,
      c10::kCPU,
      std::nullopt /* pin_memory */,
      output_scale,
      output_zero_point,
      c10::MemoryFormat::ChannelsLast);

  // Reshape the operator
  status = at::native::xnnp_utils::xnnp_reshape_convolution2d_nhwc(
      xnnp_convolution_op.get(),
      N,
      H,
      W,
      caffe2::pthreadpool_(),
      per_channel(),
      transpose(),
      output_padding()[0],
      output_padding()[1]);

  TORCH_CHECK(
      status == xnn_status_success,
      func_name,
      ": xnn setup operator failed(",
      status,
      ")");

  // Setup the operator
  status = at::native::xnnp_utils::xnnp_setup_convolution2d_nhwc(
      xnnp_convolution_op.get(),
      reinterpret_cast<const underlying_t*>(act_nhwc.template data_ptr<scalar_t>()),
      reinterpret_cast<underlying_t*>(output.template data_ptr<scalar_t>()),
      per_channel(),
      transpose());

  TORCH_CHECK(
      status == xnn_status_success,
      func_name,
      ": xnn setup operator failed(",
      status,
      ")");

  // Run the operator
  status = xnn_run_operator(
      xnnp_convolution_op.get(), /* xnn_operator_t op */
      caffe2::pthreadpool_()); /* pthreadpool_t threadpool */

  TORCH_CHECK(
      status == xnn_status_success,
      func_name,
      ": xnn run operator failed(",
      status,
      ")");

  return output;
}

#endif // USE_XNNPACK

template <int kSpatialDim>
template <bool kReluFused>
at::Tensor PackedConvWeightsQnnp<kSpatialDim>::apply_impl(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point) {
  // QNNPack is not thread safe
  std::lock_guard<std::mutex> lock(qnnp_mutex_);
  const std::string func_name = transpose() ? "quantized::conv_transpose"
                                            : "quantized::conv";
  TORCH_CHECK(!(kReluFused && transpose()),
              kSpatialDim == 2,
              func_name, kSpatialDim,
              "d (qnnpack): ConvTranspose cannot be fused with ReLU.");
  TORCH_CHECK(act.scalar_type() == c10::kQUInt8,
              func_name,
              "(qnnpack): Expected activation data type ",
              toString(c10::kQUInt8),
              " but got ",
              toString(act.scalar_type()));
  ConvDimChecks<kSpatialDim>(
      act.ndimension(), stride().size(), padding().size(),
      output_padding().size(), dilation().size(), func_name, transpose());

  auto* pack_w = w.get();

  // TODO Can be replaced with packB->getOutputChannels() when update pre-pack
  // to actually do the packing.
  const int out_ch_idx = transpose() ? 1 : 0;
  const auto out_ch = bias.size(0);
  // inputs are in semantic NCHW format
  const int N = act.size(0);
  const int C = act.size(1);
  const int D = kSpatialDim == 3 ? act.size(2) : 1;
  const int H = act.size(kSpatialDim);
  const int W = act.size(kSpatialDim + 1);
  const int M = out_ch; // output channels

  const auto channels_last = kSpatialDim == 2
      ? c10::MemoryFormat::ChannelsLast
      : c10::MemoryFormat::ChannelsLast3d;
  const at::Tensor act_ndhwc = act.contiguous(channels_last);

  auto output_min = kReluFused
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      ? activationLimits<uint8_t>(output_scale, output_zero_point, Activation::RELU)
            .first
      : std::numeric_limits<uint8_t>::min();
  auto output_max = kReluFused
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      ? activationLimits<uint8_t>(output_scale, output_zero_point, Activation::RELU)
            .second
      : std::numeric_limits<uint8_t>::max();

  double act_input_scale = act_ndhwc.q_scale();

  // Re-quantizing the bias based on input scale and weight scale.
  if (!input_scale.has_value() || input_scale.value() != act_input_scale) {
    TORCH_CHECK(M == (transpose() ? groups() : 1) * orig_weight.size(out_ch_idx),
        "Output channel size of weight and bias must match.");
    TORCH_CHECK(C == (transpose() ? 1 : groups()) * orig_weight.size(1 - out_ch_idx),
        "Input channel size of weight and bias must match.");

    // Get the original weight and adjust it to uint8 from int8
    auto weight_contig = orig_weight.contiguous(channels_last);
    auto bias_fp32 = bias;
    int8_t* w_data =
        reinterpret_cast<int8_t*>(weight_contig.template data_ptr<c10::qint8>());

    float* weight_scales_data = w_scales.data_ptr<float>();
    // We calculate requant scale here as the vector holding the requant scale
    // is owned by this module. The pointer is then passed to qnnpack backend.
    generate_requantization_scales(
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        w_scales, act_input_scale, output_scale, requantization_scales);

    // TODO Kimish, we are allocating affine_quantized regardless of per channel or not.
    // This allocation is actually used only for packing weight and thus will be freed.
    // Still we should be consistent. Fix this.
    at::Tensor qnnp_weight = at::_empty_affine_quantized(
        weight_contig.sizes(),
        at::device(c10::kCPU).dtype(c10::kQUInt8).memory_format(channels_last),
        weight_scales_data[0],
        w_zero_points[0],
        std::nullopt);
    auto* qnnp_w_data = qnnp_weight.template data_ptr<c10::quint8>();
    auto wt_numel = weight_contig.numel();
    for (const auto i : c10::irange(wt_numel)) {
      qnnp_w_data[i] = static_cast<c10::quint8>(w_data[i] + 128);
    }
    // Original bias was float, so we requantize it here.
    at::Tensor qbias = quant_utils::QuantizeBias(convolution_op->per_channel, bias_fp32, weight_contig, act_input_scale);

    // Update the input scale to not pack again.
    input_scale = act_input_scale;
    w.reset();
    w = std::make_unique<qnnpack::PrePackConvWeights>(
        convolution_op.get(),
        w_zero_points.data(),
        reinterpret_cast<uint8_t*>(qnnp_w_data),
        reinterpret_cast<int32_t*>(qbias.template data_ptr<c10::qint32>()));
    pack_w = w.get();
    if (at::globalContext().releaseWeightsWhenPrepacking()) {
        // On mobile, we release the original weight by resetting the intrusive_ptr.
        // Calling unpack after this will throw an assertion.
        orig_weight.reset();
    }

    // Set padding buffer to zero point. This can only be done if we want
    // to do it only once.
    if (zero_buffer_size) {
      memset(
          convolution_op->zero_buffer,
          act_ndhwc.q_zero_point(),
          zero_buffer_size);
    }
  }

  TORCH_INTERNAL_ASSERT(pack_w != nullptr, "Packed Weights are NULL");
  at::SmallVector<int64_t, kSpatialDim + 2> output_shape;
  const auto input_shape = MakeInputShape<kSpatialDim>(D, H, W);
  if (transpose()) {
    output_shape = MakeDeConvOutputShape<kSpatialDim>(
        N,
        M,
        kSpatialDim == 2 ? std::vector<int64_t>{H, W} : std::vector<int64_t>{D, H, W},
        kernel_,
        stride(),
        padding(),
        output_padding(),
        dilation());
  } else {
    output_shape = at::native::quantized::MakeConvOutputShape<kSpatialDim>(
        N, M, input_shape, kernel_, stride(), padding(), dilation());
  }

  if (act_ndhwc.numel() > 0) {
    TORCH_CHECK(
        std::all_of(
            output_shape.begin(),
            output_shape.end(),
            [](int64_t i) { return i > 0; }),
        func_name,
        kSpatialDim,
        "d (qnnpack): each dimension of output tensor should "
        "be greater than 0.")
  }

  // Allocate output Tensor and a buffer for QNNPACK to use
  at::Tensor output = at::native::empty_affine_quantized(
      output_shape,
      c10::kQUInt8,
      std::nullopt /* layout */,
      c10::kCPU,
      std::nullopt /* pin_memory */,
      output_scale,
      output_zero_point,
      channels_last);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  pytorch_qnnp_status run_status;
  if (transpose()) {
    run_status = qnnpack::qnnpackDeConv(
        convolution_op.get(),
        pack_w->getPackedWeights(),
        N,
        H,
        W,
        act_ndhwc.q_zero_point(),
        reinterpret_cast<uint8_t*>(act_ndhwc.template data_ptr<c10::quint8>()),
        w_zero_points.data(),
        requantization_scales.data(),
        output.q_zero_point(),
        output_min,
        output_max,
        reinterpret_cast<uint8_t*>(output.template data_ptr<c10::quint8>()),
        caffe2::pthreadpool_());
  } else {
    run_status = qnnpack::qnnpackConv(
        convolution_op.get(),
        pack_w->getPackedWeights(),
        N,
        D,
        H,
        W,
        act_ndhwc.q_zero_point(),
        reinterpret_cast<uint8_t*>(act_ndhwc.template data_ptr<c10::quint8>()),
        w_zero_points.data(),
        requantization_scales.data(),
        output.q_zero_point(),
        output_min,
        output_max,
        reinterpret_cast<uint8_t*>(output.template data_ptr<c10::quint8>()),
        caffe2::pthreadpool_());
  }

  TORCH_INTERNAL_ASSERT(
      run_status == pytorch_qnnp_status_success,
      "failed to run quantized::conv2d (qnnpack) operator");

  return output;
}

#ifdef USE_XNNPACK
static bool can_use_xnnp(
    c10::ScalarType dtype,
    int kSpatialDim,
    bool per_channel,
    bool transpose) {
  if (!at::native::xnnpack::available()) {
    return false;
  }
  bool supported_dtypes = dtype == c10::kQInt8;
  bool invalid_config =
      (kSpatialDim != 2 /* No support for 3d convolution */
        || (dtype == c10::kQInt8 && transpose &&
            per_channel)); /* int8_t deconv does not support per-channel */
  if (supported_dtypes && invalid_config) {
    /* don't want this to fall through to QNNPACK */
    const std::string func_name =
        transpose ? "quantized::conv_transpose" : "quantized::conv";
    TORCH_CHECK(
        false,
        func_name,
        " (xnnpack): Unsupported conv config for dtype KQInt8");
  }
  return supported_dtypes && !invalid_config;
}
#endif  // USE_XNNPACK

template <int kSpatialDim>
at::Tensor PackedConvWeightsQnnp<kSpatialDim>::apply(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
#ifdef USE_XNNPACK
  if (can_use_xnnp(input.scalar_type(), kSpatialDim, per_channel(), transpose())) {
    return apply_impl_xnnp<c10::qint8, false>(
        input, output_scale, output_zero_point);
  } /* fall through for unsupported types, configs, or shapes */
#endif // USE_XNNPACK
  return apply_impl<false>(input, output_scale, output_zero_point);
}

template <int kSpatialDim>
at::Tensor PackedConvWeightsQnnp<kSpatialDim>::apply_relu(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
#ifdef USE_XNNPACK
  if (can_use_xnnp(input.scalar_type(), kSpatialDim, per_channel(), transpose())) {
    return apply_impl_xnnp<c10::qint8, true>(
        input, output_scale, output_zero_point);
  } /* fall through for unsupported types, configs, or shapes */
#endif // USE_XNNPACK
  return apply_impl<true>(input, output_scale, output_zero_point);
}

template at::Tensor PackedConvWeightsQnnp<2>::apply(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeightsQnnp<2>::apply_relu(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeightsQnnp<3>::apply(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeightsQnnp<3>::apply_relu(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeightsQnnp<2>::apply_impl<false>(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeightsQnnp<3>::apply_impl<false>(
  const at::Tensor& act,
  double output_scale,
  int64_t output_zero_point);

#endif // USE_PYTORCH_QNNPACK

#if AT_MKLDNN_ENABLED()
template <int kSpatialDim>
at::Tensor PackedConvWeightsOnednn<kSpatialDim>::apply(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<false>(input, std::nullopt, output_scale, output_zero_point);
}

template <int kSpatialDim>
at::Tensor PackedConvWeightsOnednn<kSpatialDim>::apply_relu(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<true>(input, std::nullopt, output_scale, output_zero_point);
}

template <int kSpatialDim>
at::Tensor PackedConvWeightsOnednn<kSpatialDim>::apply_add(
    const at::Tensor& input,
    const at::Tensor& accum,
    double output_scale,
    int64_t output_zero_point) {
  TORCH_CHECK(kSpatialDim == 2, " Currently, only conv2d with add is supported.");
  return apply_impl<false>(input, accum, output_scale, output_zero_point);
}

template <int kSpatialDim>
at::Tensor PackedConvWeightsOnednn<kSpatialDim>::apply_add_relu(
    const at::Tensor& input,
    const at::Tensor& accum,
    double output_scale,
    int64_t output_zero_point) {
  TORCH_CHECK(kSpatialDim == 2, " Currently, only conv2d add relu is supported.");
  return apply_impl<true>(input, accum, output_scale, output_zero_point);
}

template <int kSpatialDim>
template <bool kReluFused>
at::Tensor PackedConvWeightsOnednn<kSpatialDim>::apply_impl(
    const at::Tensor& act,
    const std::optional<at::Tensor>& accum,
    double output_scale,
    int64_t output_zero_point) {
  std::string func_name = "quantized::conv";
  if (transpose()) {
    func_name += "_transpose";
  }
  func_name += std::to_string(kSpatialDim) + "d";

  // has_accum: extra input besides the conv to do conv add fusion.
  bool has_accum = accum.has_value() ? true : false;
  if (has_accum) {
    auto& ctx = at::globalContext();
    func_name += "_add";
    TORCH_CHECK(
      !transpose(),
      "Didn't support transposed conv for conv with add ",
      c10::toString(ctx.qEngine()));
  }

  if (kReluFused) {
    func_name += "_relu";
  }
  ConvDimChecks<kSpatialDim>(
      act.ndimension(), stride().size(), padding().size(),
      output_padding().size(), dilation().size(), func_name, transpose());
  TORCH_CHECK(act.scalar_type() == c10::ScalarType::QUInt8,
      func_name, " (ONEDNN): data type of input should be QUint8.");

  // src
  auto act_contig = act.contiguous(kSpatialDim == 2 ? c10::MemoryFormat::ChannelsLast : c10::MemoryFormat::ChannelsLast3d);
  auto src_dims = act_contig.sizes().vec();
  auto src_data_type = dnnl::memory::data_type::u8;
  auto src_desc = ideep::tensor::desc(src_dims, src_data_type,
      kSpatialDim == 2 ? ideep::format_tag::nhwc : ideep::format_tag::ndhwc);
  ideep::tensor src(src_desc, act_contig.data_ptr());
  // weights & bias
  ideep::tensor& weights = *(weight_.get());
  bool with_bias = bias_.has_value();
  const auto& kernel_size = weights.get_dims();
  // dst
  const std::vector<int64_t>& input_size = src.get_dims();
  std::vector<int64_t> output_sizes;
  if (transpose()) {
    // Prepacked weight format: [o, i, ...]
    const int N = act.size(0); // batch size
    const int C = act.size(1); // input channels
    const int M = weights.get_dim(0); // output channels
    const int D = kSpatialDim == 2 ? 1 : act.size(2); // input depth
    const int H = act.size(kSpatialDim); // input height
    const int W = act.size(kSpatialDim + 1); // input width
    const int KH = weights.get_dim(kSpatialDim); // kernel height
    const int KW = weights.get_dim(kSpatialDim + 1); // kernel width
    const int KD = kSpatialDim == 2 ? 1 : weights.get_dim(2); // kernel depth
    TORCH_CHECK(C == groups() * weights.get_dim(1), // weight: [o, i, ...]
                func_name, " (ONEDNN): input channel number should be ",
                groups() * weights.get_dim(1), ", but got ", C);
    auto output_shape = MakeDeConvOutputShape<kSpatialDim>(
        N,
        M,
        kSpatialDim == 2 ? std::vector<int64_t>{H, W} : std::vector<int64_t>{D, H, W},
        kSpatialDim == 2 ? std::vector<int64_t>{KH, KW} : std::vector<int64_t>{KD, KH, KW},
        stride(),
        padding(),
        output_padding(),
        dilation());
    output_sizes = c10::IntArrayRef(output_shape).vec();
  } else {
    output_sizes = at::native::conv_output_size(input_size, kernel_size, padding().vec(), stride().vec(), dilation().vec());
  }
  ideep::dims dst_dims = ideep::dims({output_sizes.cbegin(), output_sizes.cend()});
  at::Tensor output = at::_empty_affine_quantized(
      dst_dims,
      device(c10::kCPU)
          .dtype(c10::kQUInt8)
          .memory_format(kSpatialDim == 2 ?
              c10::MemoryFormat::ChannelsLast :
              c10::MemoryFormat::ChannelsLast3d),
      output_scale,
      output_zero_point,
      std::nullopt);
  if (output.numel() == 0) {
    return output;
  }
  ideep::tensor dst;
  at::Tensor accum_contig;
  if (has_accum) {
    auto dst_desc = ideep::tensor::desc(dst_dims, src_data_type,
        kSpatialDim == 2 ? ideep::format_tag::nhwc : ideep::format_tag::ndhwc);
    accum_contig = accum.value().contiguous(kSpatialDim == 2 ? c10::MemoryFormat::ChannelsLast : c10::MemoryFormat::ChannelsLast3d);
    TORCH_CHECK(accum_contig.dtype() == output.dtype(), "The output tensor should have same dtype as the accum tensor.");
    // When fused with sum, the dst tensor will share the data ptr as the accum tensor.
    dst.init(dst_desc, accum_contig.data_ptr());
  } else {
    dst = ideep::tensor({dst_dims, ideep::tensor::data_type::u8, {output.strides().cbegin(), output.strides().cend()}},
                      output.data_ptr());
  }

  // Parameters
  const ideep::dims& strides = stride().vec();
  const ideep::dims& dilates = dilation().vec();
  const ideep::dims& padding_l = padding().vec();
  const ideep::dims& padding_r = padding().vec();
  double input_scale = act.q_scale();
  int64_t input_zp = act.q_zero_point();
  // Scales of ONEDNN and PyTorch are reciprocal
  const ideep::scale_t& src_scales = ideep::scale_t(1, 1.0/input_scale);
  const ideep::scale_t& weights_scales = weights.get_scale();
  double inv_output_scale = 1.0/output_scale;
  const ideep::zero_point_t src_zero_points = ideep::zero_point_t(1, input_zp);
  const ideep::zero_point_t dst_zero_points = ideep::zero_point_t(1, output_zero_point);

  ideep::attr_t op_attr;
  float sum_scale = has_accum ? accum.value().q_scale() : 1.0;
  int32_t sum_zero_point = has_accum ? accum.value().q_zero_point() : 0;
  if (has_accum) {
    // Just tells we have these post op, the actual value such as scale and zero point will be setted later.
    op_attr = kReluFused ? ideep::attr_t::residual_with_sum_zero_point() : ideep::attr_t::fuse_sum();
    const ideep::scale_t accum_scale = ideep::scale_t(1, 1.0/sum_scale);
    const ideep::zero_point_t accum_zero_points = ideep::zero_point_t(1, sum_zero_point);
    // Set the dst scale and zero point with the value of accum.
    // The true scale and zero point is stored in ideep::scale_t(scale_size, inv_output_scale) and dst_zero_points.
    dst.set_scale(accum_scale);
    dst.set_zero_point(accum_zero_points);
  } else if (kReluFused) {
    op_attr = ideep::attr_t::fuse_relu();
  }

  // Bias might be modified outside (e.g. by quantization bias correction).
  // If so, update the prepacked bias as well.
  if (with_bias && bias_.value().get_data_handle() != orig_bias_.value().data_ptr()) {
    bias_.value().init(bias_.value().get_desc(), orig_bias_.value().data_ptr());
  }
  const auto& b = with_bias ? bias_.value() : ideep::tensor();
  int num_threads = at::get_num_threads();
  if (transpose()) {
    // Primitive cache is initialized when called for the first time
    // and won't be updated afterwards.
    PrimitiveCacheKey cache_key = std::make_tuple(
        input_scale, input_zp, src_dims, output_scale, output_zero_point, num_threads, sum_scale, sum_zero_point);
    c10::call_once(*cache_initialized_flag, [&](){
        DeconvParams params;
        ideep::convolution_transpose_forward::prepare(
            params, src, weights, b, dst_dims, dst,
            strides, padding_l, padding_r, dilates, groups(),
            src_scales, weights_scales, ideep::scale_t(1, inv_output_scale),
            src_zero_points, dst_zero_points, op_attr,
            dnnl::algorithm::deconvolution_direct,
            dnnl::prop_kind::forward_inference,
            ideep::u8s8, ideep::engine::cpu_engine());
        get_deconv_cache() = DeconvPrimitiveCache(cache_key, params);
        auto expected_weight_desc = ideep::tensor::desc(params.pd.weights_desc(), groups());
        weights = weights.reorder_if_differ_in(expected_weight_desc);
    });
    if (get_deconv_cache().hit(cache_key)) {
      DeconvParams& params = get_deconv_cache().get_params();
      ideep::convolution_transpose_forward::compute<false, false>(
          params, src, weights, b, dst);
    } else {
      ideep::convolution_transpose_forward::compute(
          src, weights, b, dst_dims, dst,
          strides, padding_l, padding_r, dilates,
          groups(), src_scales, weights_scales,
          ideep::scale_t(1, inv_output_scale),
          src_zero_points, dst_zero_points, op_attr,
          dnnl::algorithm::deconvolution_direct,
          dnnl::prop_kind::forward_inference,
          ideep::u8s8, ideep::engine::cpu_engine());
    }
  } else {  // not transposed
    PrimitiveCacheKey cache_key = std::make_tuple(
        input_scale, input_zp, src_dims, output_scale, output_zero_point, num_threads, sum_scale, sum_zero_point);
    c10::call_once(*cache_initialized_flag, [&](){
        ConvParams params;
        ideep::convolution_forward::prepare(
            params, src, weights, b, dst_dims, dst,
            strides, dilates, padding_l, padding_r, groups(),
            src_scales, weights_scales, ideep::scale_t(1, inv_output_scale),
            src_zero_points, dst_zero_points,
            op_attr, dnnl::algorithm::convolution_direct,
            dnnl::prop_kind::forward_inference,
            ideep::u8s8, ideep::engine::cpu_engine());
        get_conv_cache() = ConvPrimitiveCache(cache_key, params);
        auto expected_weight_desc = ideep::tensor::desc(params.pd.weights_desc(), groups());
        weights = weights.reorder_if_differ_in(expected_weight_desc);
    });
    // If hit, use cached data. If miss, fall back to normal path.
    if (get_conv_cache().hit(cache_key)) {
      auto& params = get_conv_cache().get_params();
      ideep::convolution_forward::compute<false, false>(params, src, weights, b, dst);
    } else {
      ideep::convolution_forward::compute(
          src, weights, b, dst_dims, dst,
          strides, dilates, padding_l, padding_r, groups(),
          src_scales, weights_scales, ideep::scale_t(1, inv_output_scale),
          src_zero_points, dst_zero_points, op_attr,
          dnnl::algorithm::convolution_direct,
          dnnl::prop_kind::forward_inference,
          ideep::u8s8, ideep::engine::cpu_engine());
    }
  }
  if (has_accum) {
    // When fused with sum, the accum tensor share the data ptr as dst tensor as the output.
    // Reset output's scale and zero point into accum_contig.
    set_quantizer_(accum_contig, at::make_per_tensor_affine_quantizer(
        output_scale, output_zero_point, accum_contig.scalar_type()));
    return accum_contig;
  } else {
    return output;
  }
}

template at::Tensor PackedConvWeightsOnednn<2>::apply(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeightsOnednn<2>::apply_relu(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeightsOnednn<3>::apply(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeightsOnednn<3>::apply_relu(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

static at::Tensor _quantized_convolution_onednn(
    at::Tensor act, // contains quantized values but not QTensor
    double act_scale,
    int64_t act_zero_point,
    at::Tensor weight, // MKLDNN tensor with quantized values
    at::Tensor weight_scales,
    at::Tensor weight_zero_points,
    std::optional<at::Tensor> bias, // Bias is not packed into MKLDNN tensor
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    bool transposed,
    int64_t groups,
    double output_scale,
    int64_t output_zero_point,
    std::optional<at::Tensor> accum, // accum to fused with conv add
    double accum_scale,
    int64_t accum_zero_point,
    std::optional<c10::ScalarType> output_dtype,
    std::optional<std::string_view> binary_attr,
    std::optional<at::Scalar> binary_alpha,
    std::optional<std::string_view> unary_attr,
    torch::List<std::optional<at::Scalar>> unary_scalars,
    std::optional<std::string_view> unary_algorithm) {
  /*********************************/
  /*          Checks               */
  /*********************************/
  // Due the constant folding inside Inductor freeze,
  // https://github.com/pytorch/pytorch/blob/b99d605a3070de35677cc43f0196c2f2e807b822/torch/ao/quantization/fx/_decomposed.py#L62-L63
  // inv_scale = 1.0 / scale will be folded.
  // So, we can only get inv_scale from quant node which is used as
  // output_scale of this op.
  bool fp32_output = output_dtype.has_value() && (output_dtype.value() == c10::kFloat);
  bool bfloat16_output = output_dtype.has_value() && (output_dtype.value() == c10::kBFloat16);
  if (fp32_output || bfloat16_output) {
    // When fp32 or bf16 output, oneDNN expects op_attr doesn't set_scales and set_zero_points.
    // So, we will use default output_scale as 1.0 and output_zero_point as 0, since
    // when output_scale is 1.0, we will skip invoking of op_attr.set_scales in ideep;
    // when output_zero_point is 0, we will skip invoking of op_attr.set_zero_points in ideep.
    TORCH_CHECK(output_scale == 1.0,  " (ONEDNN): fp32 or bf16 output, output_scale must be 1.0.");
    TORCH_CHECK(output_zero_point == 0,  " (ONEDNN): fp32 or bf16 output, output_zero_point must be 0");
  }

  int kSpatialDim = act.dim() - 2;
  bool is_1d = (1 == kSpatialDim);

  bool has_binary_post_op = binary_attr.has_value() && binary_attr.value() != "none";
  bool has_unary_post_op = unary_attr.has_value() && unary_attr.value() != "none";
  // has_accum_postop_sum: extra input besides the conv to do conv post op sum fusion.
  bool has_accum_postop_sum = has_binary_post_op && binary_attr.value() == "sum";
  bool has_binary_postop_add = has_binary_post_op && binary_attr.value() == "add";

  if (has_accum_postop_sum || has_binary_postop_add) {
    TORCH_CHECK(accum.has_value(), "For post op sum or post op binary_add, accum tensor should not be empty.");
    TORCH_CHECK(
      accum.value().is_contiguous(
        kSpatialDim == 2
        ? c10::MemoryFormat::ChannelsLast
        : c10::MemoryFormat::ChannelsLast3d
      ),
      "For post op sum or post op binary_add, accum tensor must be contiguous."
    );
    if (fp32_output || bfloat16_output) {
      TORCH_CHECK(accum_scale == 1.0,  " (ONEDNN): fp32 or bf16 output, accum_scale must be 1.0.");
      TORCH_CHECK(accum_zero_point == 0,  " (ONEDNN): fp32 or bf16 output, accum_zero_point must be 0");
      TORCH_CHECK((accum.value().scalar_type() == c10::kFloat) || (accum.value().scalar_type() == c10::kBFloat16), "The accum tensor should be KFloat or KBFloat.");
    }
  }

  std::string func_name = "quantized::packed_weights_conv";
  func_name += std::to_string(kSpatialDim) + "d";
  if (has_binary_post_op) {
    func_name += binary_attr.value().data();
  }
  if (has_unary_post_op) {
    func_name += unary_attr.value().data();
  }

  if (kSpatialDim == 1) {
    kSpatialDim += 1;
  }
  TORCH_CHECK(
    weight.is_mkldnn(),
    func_name, ": Weight should be prepacked as an MKLDNN tensor"
  );
  if (transposed) {
    TORCH_CHECK(
      false,
      func_name, ": to support transposed convolution."
    );
  }
  if (is_1d) {
    // N, C, L -> N, C, 1, L
    act = act.unsqueeze(quant_utils::kConv1dSqueezeDim + 2);
    stride = quant_utils::MakeArgForConv1d(stride, 1);
    padding = quant_utils::MakeArgForConv1d(padding, 0);
    dilation = quant_utils::MakeArgForConv1d(dilation, 1);
  }
  TORCH_CHECK(
    act.scalar_type() == c10::ScalarType::Byte,
    func_name, ": Input tensor should have uint8 (unsigned char) data type");
  TORCH_CHECK(
    weight.scalar_type() == c10::ScalarType::Char,
    func_name, ": Weight tensor should have int8 (char) data type");
  TORCH_CHECK(
    weight.ndimension() == kSpatialDim + 2,
    func_name, ": Weights are expected to have ", kSpatialDim + 2, " dimensions");
  TORCH_CHECK(
    stride.size() == (decltype(stride.size()))kSpatialDim,
    func_name, ": stride should contain ", kSpatialDim, " elements for ",
    kSpatialDim, "D convolution.");
  TORCH_CHECK(
    padding.size() == (decltype(padding.size()))kSpatialDim,
    func_name, ": Specify front/top/left padding only. "
    "end/bottom/right padding assumed to be equal to front/top/left");
  TORCH_CHECK(
    dilation.size() == (decltype(dilation.size()))kSpatialDim,
    func_name, ": dilation should contain ", kSpatialDim, " elements for ",
    kSpatialDim, "D convolution.");

  // Parameters
#if IDEEP_PREREQ(3, 1, 0, 1)
  // 1. If the weight scale generated by observer should with dtype float32
  // https://github.com/pytorch/pytorch/blob/d2c24eca8a60c56b31ca967a44d5cc4522802aa6/torch/ao/quantization/observer.py#L323
  // 2. If the weight scale got from the quantized tensor, like did in the UT. It's with dtype of double.
  // https://github.com/pytorch/pytorch/blob/d2fa3f608b5e4f582a8aaf752f10efe4ca72a7d0/aten/src/ATen/quantized/Quantizer.cpp#L69
  TORCH_CHECK(
    weight_scales.scalar_type() == c10::ScalarType::Double || weight_scales.scalar_type() == c10::ScalarType::Float,
    "weight_scales should be with data type Double or float");
  if (weight_scales.scalar_type() == c10::ScalarType::Double) {
    // For case 2, we will convert it from double to float, since ideep::scale_t is alias of std::vector<float>
    weight_scales = weight_scales.to(c10::ScalarType::Float);
  }
  TORCH_CHECK(
    weight_scales.ndimension() == 0 ||
    (weight_scales.strides().size() == 1 || weight_scales.stride(0) == 1),
    "weight_scales should be scalar tensor or contiguous 1D tensor.");
  ideep::scale_t weights_scales(weight_scales.data_ptr<float>(), weight_scales.data_ptr<float>()+weight_scales.numel());
#elif IDEEP_PREREQ(3, 1, 0, 0)
  // TODO (leslie): optimize the performance here:
  // 1. Remove the reciprocal of weight scale, we have done the reciprocal of weight scale back in Ideep:
  // https://github.com/intel/ideep/blob/3c90e365526e19c110371d23831678a7e9d4353d/include/ideep/operators/conv.hpp#L163-L168
  // 2. Remove 2 memory copies of weight_scales:
  //   2.1 Input of weights_scales is PyTorch Dense tensor, we convert it to vector<float>
  //   2.2 OneDNN stream submit convert weights_scales from vector to ideep::tensor
  //   https://github.com/intel/ideep/blob/3c90e365526e19c110371d23831678a7e9d4353d/include/ideep/operators/conv.hpp#L1855-L1860
  // We should be able to directly convert weights_scales from PyTorch Dense Tensor to IDeep Tensor which can share same data ptr.
  ideep::scale_t weights_scales(weight_scales.numel());
  if (weight_scales.ndimension() == 0) {
    // Weight is quant per tensor, then weight_scales will be a scalar Tensor
    weights_scales[0] = 1.0 / weight_scales.item().toDouble(); // Scales of ONEDNN and PyTorch are reciprocal
  } else {
    // Weight is quant per channel
    for (int i = 0; i < weight_scales.numel(); ++i) {
      weights_scales[i] = 1.0 / weight_scales[i].item().toDouble();
    }
  }
#else
  TORCH_CHECK(false, "Unexpected IDeep version to do qconv calculation.");
#endif

  const ideep::zero_point_t src_zero_points = ideep::zero_point_t(1, act_zero_point);
  const ideep::zero_point_t dst_zero_points = ideep::zero_point_t(1, output_zero_point);

  // Weight
  auto packed_weight = at::native::itensor_from_mkldnn(weight);

  // Bias
  ideep::tensor onednn_bias;
  const int output_channels = weight.size(0);
  bool with_bias = bias.has_value();

  at::Tensor bias_val_float;
  if (with_bias) {
    // For int8-mixed-bf16, we will also use float32 bias
    bias_val_float = bias.value().to(at::kFloat);
    TORCH_CHECK(bias_val_float.dim() == 1, "bias should be a vector (1D Tensor)");
    TORCH_CHECK(
        bias_val_float.size(0) == output_channels,
        "bias should have K elements: " + std::to_string(output_channels));
    auto bias_desc = ideep::tensor::desc(bias_val_float.sizes().vec(), dnnl::memory::data_type::f32);
    onednn_bias.init(bias_desc, bias_val_float.data_ptr());
  }

  const auto& expected_bias = with_bias ? onednn_bias : ideep::tensor();

  /*********************************/
  /*        Computation            */
  /*********************************/
  // src
  auto act_contig = act.contiguous(kSpatialDim == 2 ?
                                   c10::MemoryFormat::ChannelsLast :
                                   c10::MemoryFormat::ChannelsLast3d);
  auto src_dims = act_contig.sizes().vec();
  auto src_data_type = dnnl::memory::data_type::u8;
  auto src_desc = ideep::tensor::desc(src_dims, src_data_type,
      kSpatialDim == 2 ? ideep::format_tag::nhwc : ideep::format_tag::ndhwc);
  ideep::tensor src;
  src.init(src_desc, act_contig.data_ptr());
  // dst
  const std::vector<int64_t>& input_size = src.get_dims();
  const auto& kernel_size = packed_weight.get_dims();
  std::vector<int64_t> output_sizes;
  output_sizes = at::native::conv_output_size(input_size, kernel_size, padding.vec(), stride.vec(), dilation.vec());
  ideep::dims dst_dims = ideep::dims({output_sizes.cbegin(), output_sizes.cend()});
  // Output is not a quantized tensor but data type is uint8
  at::Tensor output = has_accum_postop_sum ?
    accum.value() :
    at::empty(
      dst_dims,
      device(c10::kCPU)
          .dtype(fp32_output ? c10::kFloat : (bfloat16_output ? c10::kBFloat16 : c10::kByte))
          .memory_format(kSpatialDim == 2 ?
              c10::MemoryFormat::ChannelsLast :
              c10::MemoryFormat::ChannelsLast3d)
    );
  if (output.numel() == 0) {
    return output;
  }
  ideep::tensor dst = at::native::itensor_view_from_dense(output);
  static ideep::tensor empty_tensor;
  static ideep::tensor::desc empty_tensor_desc;
  ideep::tensor src1 = has_binary_postop_add ? at::native::itensor_view_from_dense(accum.value()) : empty_tensor;
  auto accum_desc = has_binary_postop_add ? src1.get_desc() : empty_tensor_desc;
  ideep::attr_t op_attr = onednn_utils::create_attr_by_post_op(
    binary_attr.has_value() ? binary_attr.value() : "none",
    binary_alpha.has_value() ? binary_alpha.value().to<double>() : 1.0,
    accum_scale,
    accum_zero_point,
    accum_desc,
    unary_attr.has_value() ? unary_attr.value() : "none",
    unary_scalars,
    unary_algorithm.has_value() ? unary_algorithm.value() : ""
  );

#if IDEEP_PREREQ(3, 1, 0, 0)
  // Use oneDNN's APIs instead of prepare/compute from ideep to reduce integration overhead.
  // The functions from ideep are heavy because they have complex data structures for unified API
  // oneDNN version >= 3.1.0 is required.
  using ideep::tensor;
  auto weight_grouped = packed_weight.make_grouped_weights(groups, /* is_deconv */false);
  auto weights_desc = tensor::desc(weight_grouped.get_dims(), ideep::data_type::s8, ideep::format_tag::any);
  if (groups > 1) {
    weights_desc = weights_desc.to_grouped(groups);
  }
  auto dst_desc = dst.get_desc();
  auto bias_desc = with_bias ?
      tensor::desc(expected_bias.get_dims(), ideep::data_type::f32, ideep::format_tag::any) :
      tensor::desc();
  if (act_scale != 1.0f) {
    op_attr.set_scales_mask(DNNL_ARG_SRC, 0);
  }
  if (act_zero_point != 0) {
    op_attr.set_zero_points_mask(DNNL_ARG_SRC, 0);
  }
  int oc_per_group = weight_grouped.get_dim(0) / groups;
  int wei_scale_mask = ideep::utils::conv_weight_scale_mask(weight_scales.numel(), oc_per_group, groups, false);
  op_attr.set_scales_mask(DNNL_ARG_WEIGHTS, wei_scale_mask);
  if (output_scale != 1.0f) {
    op_attr.set_scales_mask(DNNL_ARG_DST, 0);
  }
  if (output_zero_point != 0) {
    op_attr.set_zero_points_mask(DNNL_ARG_DST, 0);
  }
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  auto engine = ideep::engine::cpu_engine();
  auto dilates_dnnl = ideep::utils::get_compatible_dilates(dilation.vec());
  auto primitive_desc = with_bias ?
      dnnl::convolution_forward::primitive_desc(
        engine, dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
        src_desc, weights_desc, bias_desc, dst_desc,
        stride.vec(), dilates_dnnl, padding.vec(), padding.vec(), op_attr
      ) :
      dnnl::convolution_forward::primitive_desc(
        engine, dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
        src_desc, weights_desc, dst_desc,
        stride.vec(), dilates_dnnl, padding.vec(), padding.vec(), op_attr
      );
  auto primitive = dnnl::convolution_forward(primitive_desc);

  // Reorder weight if needed
  auto expected_weight = weight_grouped.reorder_if_differ_in(primitive_desc.weights_desc());

  // Prepare args and execute primitive
  tensor scratchpad(primitive_desc.scratchpad_desc());
  ideep::exec_args args;
  args.insert({DNNL_ARG_SRC, src});
  args.insert({DNNL_ARG_WEIGHTS, expected_weight});
  args.insert({DNNL_ARG_DST, dst});
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
  if (with_bias) {
    args.insert({DNNL_ARG_BIAS, expected_bias});
  }
  tensor src_scales_t = tensor(ideep::scale_t(1, act_scale));
  tensor wei_scales_t = tensor(weights_scales);
  tensor dst_scales_t = tensor(ideep::scale_t(1, output_scale));
  tensor src_zp_t = tensor(ideep::zero_point_t(1, act_zero_point));
  tensor dst_zp_t = tensor(ideep::zero_point_t(1, output_zero_point));
  if (act_scale != 1.0f) {
    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scales_t});
  }
  if (output_scale != 1.0f) {
    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_scales_t});
  }
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scales_t});
  if (act_zero_point != 0) {
    args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zp_t});
  }
  if (output_zero_point != 0) {
    args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zp_t});
  }
  if (has_binary_postop_add) {
    args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, src1});
  }
  primitive.execute(ideep::stream::default_stream(), args);
#else
  // Scales of ONEDNN and PyTorch are reciprocal
  const ideep::scale_t& src_scales = ideep::scale_t(1, 1.0 / act_scale);

  // set accum scale/zero point to dst
  if (has_accum_postop_sum) {
    const ideep::scale_t accum_ideep_scale = ideep::scale_t(1, 1.0/accum_scale);
    const ideep::zero_point_t accum_ideep_zero_points = ideep::zero_point_t(1, accum_zero_point);
    // Set the dst scale and zero point with the value of accum.
    // The true scale and zero point is stored in ideep::scale_t(scale_size, output_scale) and dst_zero_points.
    dst.set_scale(accum_ideep_scale);
    dst.set_zero_point(accum_ideep_zero_points);
  }

  // Weight Reorder
  ConvParams params;
  ideep::convolution_forward::prepare(
      params, src, packed_weight, expected_bias, dst_dims, dst,
      stride.vec(), dilation.vec(), padding.vec(), padding.vec(), groups,
      src_scales, weights_scales, ideep::scale_t(1, 1.0f / output_scale),
      src_zero_points, dst_zero_points,
      op_attr, dnnl::algorithm::convolution_direct,
      dnnl::prop_kind::forward_inference,
      ideep::u8s8, ideep::engine::cpu_engine());
  auto expected_weight_desc = ideep::tensor::desc(params.pd.weights_desc(), groups);
  ideep::tensor expected_weight = packed_weight.reorder_if_differ_in(expected_weight_desc);

  // Computation
  ideep::convolution_forward::compute<false, false>(params, src, expected_weight, expected_bias, dst);
#endif

  if (is_1d) {
    output.squeeze_(quant_utils::kConv1dSqueezeDim + 2);
    return output;
  }
  if (has_accum_postop_sum) {
    return accum.value();
  } else {
    return output;
  }
}

#endif // #if AT_MKLDNN_ENABLED()

namespace at::native {

  at::Tensor QConvoneDNN::run_pointwise(
      at::Tensor act, // contains quantized values but not QTensor
      double act_scale,
      int64_t act_zero_point,
      at::Tensor weight, // contains quantized values but not QTensor
      at::Tensor weight_scales,
      at::Tensor weight_zero_points,
      std::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      std::string_view attr,
      torch::List<std::optional<at::Scalar>> scalars,
      std::optional<std::string_view> algorithm) {
#if AT_MKLDNN_ENABLED()

    if (act.dim() == 3 || act.dim() == 5) {
      // Conv1D/3D post op check
      TORCH_CHECK(
        attr == "none",
        "quantized pointwise conv",
        act.dim()-2,
        "d doesn't support unary_post_op fusion. Got unary_post_op: ",
        attr,
        ".")
    } else {
      // Conv2D post op check
      TORCH_CHECK(
        attr == "none" || attr == "relu" || attr == "hardtanh" || attr == "hardswish" || attr == "swish",
        "none post_op or post_op relu/hardtanh/hardswish is supported for quantized pointwise conv2d. Got unary_post_op: ",
        attr,
        ".")
    }
    return _quantized_convolution_onednn(
        act, act_scale, act_zero_point,
        weight, weight_scales, weight_zero_points,
        bias, stride, padding, dilation, /*transposed*/false,
        groups, output_scale, output_zero_point,
        /*accum*/std::nullopt, /*accum_scale*/0.0, /*accum_zero_point*/0,
        /*output_dtype*/output_dtype, /*binary_attr*/std::nullopt, /*binary_alpha*/std::nullopt,
        /*unary_attr*/attr, /*unary_scalars*/scalars, /*unary_algorithm*/algorithm
    );
#else
    TORCH_CHECK(false, "Unimplemented as onednn is not available.")
#endif
  }

  at::Tensor QConvoneDNN::run_pointwise_tensor(
      at::Tensor act, // contains quantized values but not QTensor
      at::Tensor act_scale,
      at::Tensor act_zero_point,
      at::Tensor weight, // contains quantized values but not QTensor
      at::Tensor weight_scales,
      at::Tensor weight_zero_points,
      std::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      std::string_view attr,
      torch::List<std::optional<at::Scalar>> scalars,
      std::optional<std::string_view> algorithm) {
#if AT_MKLDNN_ENABLED()
    TORCH_CHECK(act_scale.numel() == 1 && act_zero_point.numel() == 1,
        "onednn int8 linear: act scale/zp size should be 1");

    return run_pointwise(
        act, act_scale.item().toDouble(), act_zero_point.item().toLong(),
        weight, weight_scales, weight_zero_points,
        bias, stride, padding, dilation,
        groups, output_scale, output_zero_point,
        /*output_dtype*/output_dtype,
        /*unary_attr*/attr, /*unary_scalars*/scalars, /*unary_algorithm*/algorithm
    );
#else
    TORCH_CHECK(false, "Unimplemented as onednn is not available.")
#endif
  }


  at::Tensor QConvoneDNN::run_pointwise_binary(
      at::Tensor act, // contains quantized values but not QTensor
      double act_scale,
      int64_t act_zero_point,
      at::Tensor weight, // contains quantized values but not QTensor
      at::Tensor weight_scales,
      at::Tensor weight_zero_points,
      at::Tensor accum, // contains quantized values but not QTensor
      std::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      double accum_scale,
      int64_t accum_zero_point,
      std::string_view binary_attr,
      std::optional<at::Scalar> alpha,
      std::optional<std::string_view> unary_attr,
      torch::List<std::optional<at::Scalar>> unary_scalars,
      std::optional<std::string_view> unary_algorithm) {
#if AT_MKLDNN_ENABLED()
    // Conv2D post op check
    TORCH_CHECK(
      act.dim() == 4 && (binary_attr == "sum" || binary_attr == "add") && (
        !unary_attr.has_value() ||
        (unary_attr.has_value() &&
          (
            unary_attr.value() == "none" || unary_attr.value() == "relu"
          )
        )
      ),
      "post_op sum, binary_add, sum_relu and binary_add_relu are supported for quantized pointwise conv2d. Got binary_post_op: ",
      binary_attr,
      " unary_post_op: ",
      unary_attr.has_value() ? unary_attr.value() : "none",
      ".")
    return _quantized_convolution_onednn(
        act, act_scale, act_zero_point,
        weight, weight_scales, weight_zero_points,
        bias, stride, padding, dilation, /*transposed*/false,
        groups, output_scale, output_zero_point,
        accum, accum_scale, accum_zero_point,
        /*output_dtype*/output_dtype, binary_attr, alpha,
        unary_attr, unary_scalars, unary_algorithm
    );
#else
    TORCH_CHECK(false, "Unimplemented as onednn is not available.")
#endif
  }

  at::Tensor QConvoneDNN::run_pointwise_binary_tensor(
      at::Tensor act, // contains quantized values but not QTensor
      at::Tensor act_scale,
      at::Tensor act_zero_point,
      at::Tensor weight, // contains quantized values but not QTensor
      at::Tensor weight_scales,
      at::Tensor weight_zero_points,
      at::Tensor accum, // contains quantized values but not QTensor
      std::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      double accum_scale,
      int64_t accum_zero_point,
      std::string_view binary_attr,
      std::optional<at::Scalar> alpha,
      std::optional<std::string_view> unary_attr,
      torch::List<std::optional<at::Scalar>> unary_scalars,
      std::optional<std::string_view> unary_algorithm) {

    TORCH_CHECK(act_scale.numel() == 1 && act_zero_point.numel() == 1,
        "onednn int8 linear: act scale/zp size should be 1");
    return run_pointwise_binary(
      act, act_scale.item().toDouble(), act_zero_point.item().toLong(),
      weight, weight_scales, weight_zero_points, accum, bias,
      stride, padding, dilation, groups,
      output_scale, output_zero_point, output_dtype, accum_scale, accum_zero_point,
      binary_attr, alpha, unary_attr, unary_scalars, unary_algorithm
    );
}


namespace {

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
class QConvInt8 final {
 public:
  static Tensor run(
      Tensor act,
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight,
      double output_scale,
      int64_t output_zero_point) {
    if (kReluFused) {
      return packed_weight->apply_relu(act, output_scale, output_zero_point);
    } else {
      return packed_weight->apply(act, output_scale, output_zero_point);
    }
  }
};

template <int kSpatialDim, bool kReluFused>
class QConvAddInt8 final {
 public:
  static Tensor run(
      Tensor act,
      Tensor accum,
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight,
      double output_scale,
      int64_t output_zero_point) {
#if AT_MKLDNN_ENABLED() || !defined(STRIP_ERROR_MESSAGES)
    auto& ctx = at::globalContext();
#endif
#if AT_MKLDNN_ENABLED()
    if (ctx.qEngine() == at::QEngine::ONEDNN) {
      if (kReluFused) {
        return dynamic_cast<PackedConvWeightsOnednn<kSpatialDim>*>(packed_weight.get())->apply_add_relu(
          act, accum, output_scale, output_zero_point);
      } else {
        return dynamic_cast<PackedConvWeightsOnednn<kSpatialDim>*>(packed_weight.get())->apply_add(
          act, accum, output_scale, output_zero_point);
      }
    }
#endif
    TORCH_CHECK(
    false,
    "Didn't find engine for operation quantized::conv2d_add.",
    toString(ctx.qEngine()));
  }
};

template <bool kReluFused>
class QConv1dInt8 final {
 public:
  static Tensor run(
      Tensor act,
      const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
      double output_scale,
      int64_t output_zero_point) {
    at::Tensor output;
    // N, C, L -> N, C, 1, L
    act = act.unsqueeze(quant_utils::kConv1dSqueezeDim + 2);
    if (kReluFused) {
      output = packed_weight->apply_relu(act, output_scale, output_zero_point);
    } else {
      output = packed_weight->apply(act, output_scale, output_zero_point);
    }
    // N, C, 1, L -> N, C, L
    return output.squeeze_(quant_utils::kConv1dSqueezeDim + 2);
  }
};

// kernel for maintaining backward compatibility
template <int kSpatialDim, bool kReluFused>
class QConvInt8ForBC final {
 public:
  static Tensor run(
      Tensor act,
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight,
      torch::List<int64_t> /*stride*/,
      torch::List<int64_t> /*padding*/,
      torch::List<int64_t> /*dilation*/,
      int64_t /*groups*/,
      double output_scale,
      int64_t output_zero_point) {
    if (kReluFused) {
      TORCH_WARN_ONCE(
          "Arguments [stride, padding, dilation, groups] in ops.quantized.conv" +
              std::to_string(kSpatialDim),
          "d_relu, have been removed, please update your model to remove these arguments.");
      return packed_weight->apply_relu(act, output_scale, output_zero_point);
    } else {
      TORCH_WARN_ONCE(
          "Arguments [stride, padding, dilation, groups] in ops.quantized.conv",
          std::to_string(kSpatialDim),
          "d, have been removed, please update your model to remove these arguments.");
      return packed_weight->apply(act, output_scale, output_zero_point);
    }
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv1d"),          QConv1dInt8<false>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv1d_relu"),     QConv1dInt8<true>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d.new"),      QConvInt8<2, false>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_relu.new"), QConvInt8<2, true>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_add"),      QConvAddInt8<2, false>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_add_relu"), QConvAddInt8<2, true>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d.new"),      QConvInt8<3, false>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_relu.new"), QConvInt8<3, true>::run);
  // for backward compatibility
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d"), QConvInt8ForBC<2, false>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_relu"), QConvInt8ForBC<2, true>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d"), QConvInt8ForBC<3, false>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_relu"), QConvInt8ForBC<3, true>::run);

  // transpose
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose1d"),  QConv1dInt8<false>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d"),  QConvInt8<2, false>::run);
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::conv_transpose3d"),
      QConvInt8<3, false>::run);
}

TORCH_LIBRARY_IMPL(_quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("_quantized::conv2d"),      QConvInt8<2, false>::run);
  m.impl(TORCH_SELECTIVE_NAME("_quantized::conv2d_relu"), QConvInt8<2, true>::run);

  // transpose
  m.impl(TORCH_SELECTIVE_NAME("_quantized::conv_transpose1d"),  QConv1dInt8<false>::run);
  m.impl(TORCH_SELECTIVE_NAME("_quantized::conv_transpose2d"),  QConvInt8<2, false>::run);
}

TORCH_LIBRARY_IMPL(onednn, MkldnnCPU, m) {
  // Conv1D/2D/3D with unary postop
  m.impl(TORCH_SELECTIVE_NAME("onednn::qconv1d_pointwise"), at::native::QConvoneDNN::run_pointwise);
  m.impl(TORCH_SELECTIVE_NAME("onednn::qconv2d_pointwise"), at::native::QConvoneDNN::run_pointwise);
  m.impl(TORCH_SELECTIVE_NAME("onednn::qconv2d_pointwise.tensor"), at::native::QConvoneDNN::run_pointwise_tensor);
  m.impl(TORCH_SELECTIVE_NAME("onednn::qconv3d_pointwise"), at::native::QConvoneDNN::run_pointwise);

  // Conv2D with binary postop
  m.impl(TORCH_SELECTIVE_NAME("onednn::qconv2d_pointwise.binary"), at::native::QConvoneDNN::run_pointwise_binary);
  m.impl(TORCH_SELECTIVE_NAME("onednn::qconv2d_pointwise.binary_tensor"), at::native::QConvoneDNN::run_pointwise_binary_tensor);
}

} // namespace
} // namespace at::native
