#pragma once

#ifdef USE_PYTORCH_QNNPACK
#include <pytorch_qnnpack.h>
#include <qnnpack_func.h>

#include <ATen/native/quantized/cpu/packed_params.h>

struct QnnpackOperatorDeleter {
  void operator()(pytorch_qnnp_operator_t op) {
    pytorch_qnnp_delete_operator(op);
  }
};

// PackedWeight struct for QNNPACK stores the original Weight and Bias as
// QNNPACK currently does not support an unpack function. Possible optimization -
// For PyTorch Mobile, once the model is scripted and serialized we don't need
// to call unpack, so we can save some memory by checking for this case.
// Input scale is set to null in pre-pack step. QNNPACK needs bias quantized with
// input scale which is available at runtime in pytorch. During runtime if input
// scale value changes then we requantize bias with the updated scale.
// For inference we expect the graph to be static so the input scale should
// not change across consecutive inference calls.
struct PackedLinearWeightsQnnp : public LinearPackedParamsBase {
  PackedLinearWeightsQnnp(
      std::unique_ptr<qnnpack::PackBMatrix> w,
      at::Tensor orig_weight,
      at::Tensor bias,
      c10::optional<double> input_scale,
      double w_scale,
      int64_t w_zp)
      : w(std::move(w)),
        orig_weight(std::move(orig_weight)),
        bias(std::move(bias)),
        input_scale(std::move(input_scale)),
        w_scale(w_scale),
        w_zp(w_zp) {}

  std::unique_ptr<qnnpack::PackBMatrix> w;
  at::Tensor orig_weight;
  at::Tensor bias;
  c10::optional<double> input_scale;
  double w_scale;
  int64_t w_zp;

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

  std::string backend() override {
    return "QNNPACK";
  }

  std::string bit_width() override {
    return "INT8";
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

struct PackedConvWeightsQnnp {
  std::unique_ptr<qnnpack::PrePackConvWeights> w;
  at::Tensor orig_weight;
  at::Tensor bias;
  c10::optional<double> input_scale;
  std::vector<int64_t> kernel;
  double w_scale;
  int64_t w_zp;
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
#endif
