#pragma once

#ifdef USE_PYTORCH_QNNPACK
#include <ATen/ATen.h>
#include <pytorch_qnnpack.h>
#include <qnnpack_func.h>

#include <ATen/native/quantized/cpu/conv_packed_params.h>
#include <ATen/native/quantized/cpu/packed_params.h>

#include <utility>

struct QnnpackOperatorDeleter {
  void operator()(pytorch_qnnp_operator_t op) {
    pytorch_qnnp_delete_operator(op);
  }
};

// PackedWeight struct for QNNPACK stores the original Weight and Bias as
// QNNPACK currently does not support an unpack function.
// For PyTorch Mobile, once the model is scripted and serialized we don't need
// to call unpack, so we can save some memory by checking for this case and free
// the original weights after packing.
// Input scale is set to null in pre-pack step. QNNPACK needs bias quantized
// with input scale which is available at runtime in pytorch. During runtime if
// input scale value changes then we requantize bias with the updated scale. For
// inference we expect the graph to be static so the input scale should not
// change across consecutive inference calls.
struct PackedLinearWeightsQnnp : public LinearPackedParamsBase {
  PackedLinearWeightsQnnp(
      std::unique_ptr<qnnpack::PackBMatrix> w,
      at::Tensor orig_weight,
      at::Tensor bias,
      c10::optional<double> input_scale,
      at::Tensor w_scales,
      std::vector<uint8_t>&& w_zps)
      : w(std::move(w)),
        orig_weight(std::move(orig_weight)),
        bias_(std::move(bias)),
        input_scale(std::move(input_scale)),
        w_scales(w_scales),
        w_zero_points(std::move(w_zps)) {}

  std::unique_ptr<qnnpack::PackBMatrix> w;
  at::Tensor orig_weight;
  at::Tensor bias_;
  c10::optional<double> input_scale;
  at::Tensor w_scales;
  std::vector<uint8_t> w_zero_points;
  std::vector<float> requantization_scales;

  at::Tensor apply(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;
  at::Tensor apply_relu(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;

  at::Tensor apply_dynamic(at::Tensor input) override;
  at::Tensor apply_dynamic_relu(at::Tensor input) override;

  std::tuple<at::Tensor, c10::optional<at::Tensor>> unpack() override;

  c10::optional<at::Tensor> bias() override {
    return bias_;
  }

  static c10::intrusive_ptr<LinearPackedParamsBase> prepack(
      at::Tensor weight,
      c10::optional<at::Tensor> bias);

 private:
  template <bool ReluFused>
  at::Tensor apply_impl(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point);

  template <bool ReluFused>
  at::Tensor apply_dynamic_impl(at::Tensor input);
};

template <int kSpatialDim = 2>
struct PackedConvWeightsQnnp : public ConvPackedParamsBase<kSpatialDim> {
  PackedConvWeightsQnnp(
      std::unique_ptr<qnnpack::PrePackConvWeights> w,
      at::Tensor orig_weight,
      at::Tensor bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      c10::optional<double> input_scale,
      std::vector<int64_t> kernel,
      at::Tensor w_scale,
      std::vector<uint8_t>&& w_zps)
      : w(std::move(w)),
        orig_weight(std::move(orig_weight)),
        bias(std::move(bias)),
        stride_(std::move(stride)),
        padding_(std::move(padding)),
        dilation_(std::move(dilation)),
        groups_(groups),
        input_scale(input_scale),
        kernel_(std::move(kernel)),
        w_scales(w_scale),
        w_zero_points(std::move(w_zps)),
        conv_p(
            {(uint32_t)kernel_[1], (uint32_t)kernel_[0]},
            {(uint32_t)stride_[1], (uint32_t)stride_[0]},
            {(uint32_t)dilation_[1], (uint32_t)dilation_[0]},
            {(uint32_t)padding_[0], (uint32_t)padding_[1],
             (uint32_t)padding_[0], (uint32_t)padding_[1]},
            /*adjustment=*/{0, 0},
            groups_,
            groups_ * this->orig_weight.size(1),
            this->orig_weight.size(0),
            /*transpose=*/false)
        {}

  std::unique_ptr<qnnpack::PrePackConvWeights> w;
  at::Tensor orig_weight;
  at::Tensor bias;
  torch::List<int64_t> stride_;
  torch::List<int64_t> padding_;
  torch::List<int64_t> dilation_;
  int64_t groups_;
  c10::optional<double> input_scale;
  std::vector<int64_t> kernel_;
  at::Tensor w_scales;
  std::vector<uint8_t> w_zero_points;
  std::vector<float> requantization_scales;
  qnnpack::conv_param_t conv_p;

  at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  std::tuple<at::Tensor, c10::optional<at::Tensor>> unpack() override;

  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> prepack(
      at::Tensor weight,
      c10::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups);

  torch::List<int64_t> stride() const override {
    return stride_;
  }

  torch::List<int64_t> padding() const override {
    return padding_;
  }

  torch::List<int64_t> dilation() const override {
    return dilation_;
  }

  int64_t groups() const override {
    return groups_;
  }

 private:
  template <bool ReluFused>
  at::Tensor apply_impl(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point);
};

enum class Activation : uint8_t { NONE = 0, RELU = 1 };

#if defined(__ANDROID__) && !defined(__NDK_MAJOR__)
template <class T>
inline float Round(const float x) {
  return ::nearbyintf(x);
}
inline double Round(const double x) {
  return ::nearbyint(x);
}
#else
template <class T>
inline T Round(const T x) {
  return std::nearbyint(x);
}
#endif

inline uint8_t QuantizeUint8(float scale, int32_t zero_point, float value) {
  const int32_t qmin = std::numeric_limits<uint8_t>::min();
  const int32_t qmax = std::numeric_limits<uint8_t>::max();
  auto r = zero_point + static_cast<int32_t>(Round(value / scale));
  r = std::max(r, qmin);
  r = std::min(r, qmax);
  return static_cast<uint8_t>(r);
}

inline std::pair<uint8_t, uint8_t> activationLimits(
    float scale,
    int32_t zero_point,
    Activation Ac) {
  switch (Ac) {
    case Activation::NONE:
      return {std::numeric_limits<uint8_t>::min(),
              std::numeric_limits<uint8_t>::max()};
    case Activation::RELU:
      return {QuantizeUint8(scale, zero_point, 0.0),
              std::numeric_limits<uint8_t>::max()};
    default:
#ifdef _MSC_VER
      __assume(0);
#else
      __builtin_unreachable();
#endif
  }
}

namespace at {
namespace native {
namespace qnnp_avgpool_helper {
Tensor qnnpack_avg_pool2d(
    Tensor input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);
} // qnnp_avgpool_helper
} // namespace native
} // namespace at
#endif

namespace {
std::vector<float> generate_requantization_scales(
    const at::Tensor& weight_scales,
    const float input_scale,
    const float output_scale) {
  // Since weight scale is allocated with padding
  // weight_scales.numel() gives us padded num elements.
  auto num_output_channels_padded = weight_scales.numel();
  float* weight_scales_data = weight_scales.data_ptr<float>();
  std::vector<float> requant_scales(num_output_channels_padded, 1.f);
  for (int i = 0; i < num_output_channels_padded; ++i) {
    requant_scales[i] = weight_scales_data[i] * input_scale / output_scale;
    TORCH_CHECK(
        (requant_scales[i] > 0.0f && std::isnormal(requant_scales[i])),
        "failed to create op with requantization scale: ",
        requant_scales[i],
        ": requantization scale must be finite and positive");
  }
  return requant_scales;
}

std::pair<std::vector<uint8_t>, at::Tensor> make_zero_points_and_scales_tensor(
    const at::Tensor& weight_contig
    ) {
  auto num_output_channels = weight_contig.size(0);
  // Add 8 to account for bufferring needed by QNNPACK.
  auto num_output_channels_padded = weight_contig.size(0) + 8;
  const auto qtype = weight_contig.qscheme();
  std::vector<uint8_t> weight_zp(1, 0);
  // Adjust weight zero point, similar to weight data.
  if (qtype == at::kPerTensorAffine) {
    weight_zp[0] = (uint8_t)(weight_contig.q_zero_point() + 128);
  } else if (qtype == at::kPerChannelAffine) {
    weight_zp.resize(num_output_channels_padded);
    for (int i = 0; i < num_output_channels; ++i) {
      weight_zp[i] =
          (uint8_t)(
              weight_contig.q_per_channel_zero_points()[i].item<int32_t>() +
              128);
    }
    std::fill(weight_zp.begin() + num_output_channels, weight_zp.end(), 0);
  } else {
    TORCH_INTERNAL_ASSERT("Unsupported quantization scheme.");
  }
  at:: Tensor weight_scales =
    at::empty({1}, at::device(at::kCPU).dtype(at::kFloat));
  float* weight_scales_data;
  if (qtype == at::kPerTensorAffine) {
    weight_scales_data = weight_scales.data_ptr<float>();
    weight_scales_data[0] = weight_contig.q_scale();
  } else if (qtype == at::kPerChannelAffine) {
    weight_scales.resize_({num_output_channels_padded});
    weight_scales_data = weight_scales.data_ptr<float>();
    for (int i = 0; i < num_output_channels; ++i) {
      weight_scales_data[i] =
        weight_contig.q_per_channel_scales()[i].item<float>();
    }
  } else {
    TORCH_INTERNAL_ASSERT("Unsupported quantization scheme.");
  }
  return {weight_zp, weight_scales};
}

} // namespace
