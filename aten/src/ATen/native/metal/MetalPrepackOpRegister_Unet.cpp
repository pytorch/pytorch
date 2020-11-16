#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/metal/MetalPrepackOpContext.h>
#include <ATen/quantized/Quantizer.h>

#include <torch/custom_class.h>

#if (C10_IOS || TARGET_OS_MAC)
#import <ATen/native/metal/mpscnn/MPSCNNOps.h>
#endif

namespace at {
namespace native {
namespace metal {

class UnetConv2dOpContext final : public Conv2dOpContext {
 public:
  UnetConv2dOpContext() = delete;
  UnetConv2dOpContext(
      at::Tensor&& weight,
      c10::optional<at::Tensor>&& bias,
      const std::vector<int64_t>& stride,
      const std::vector<int64_t>& padding,
      const std::vector<int64_t>& dilation,
      int64_t groups,
      c10::optional<Scalar> output_min,
      c10::optional<Scalar> output_max)
      : Conv2dOpContext(
            std::move(weight),
            std::move(bias),
            stride,
            padding,
            dilation,
            groups,
            output_min,
            output_max) {}
};

constexpr double kMinScaleFactor = 100000000.f;

c10::intrusive_ptr<UnetConv2dOpContext> unpack_unet(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    int64_t groups,
    c10::optional<Scalar> output_min,
    c10::optional<Scalar> output_max) {
  weight = weight.contiguous();
  if (weight.is_quantized()) {
    double scale = weight.q_scale();
    int64_t zero_point = weight.q_zero_point();
    // SCale down the min value
    auto min = static_cast<float>(zero_point) / kMinScaleFactor;
    auto float_weight = at::empty(weight.sizes(), at::kFloat);
    auto quantized_data =
        reinterpret_cast<uint8_t*>(weight.template data_ptr<c10::quint8>());
    auto float_data = float_weight.data_ptr<float>();
    for (size_t i = 0; i < weight.numel(); ++i) {
      float_data[i] = (static_cast<float>(quantized_data[i]) * scale) + min;
    }
    weight = float_weight;
  }
  const auto ws = weight.sizes();
  auto packed_buffer = permuteWeights(weight.data_ptr<float>(), ws.vec());
  auto packedWeight = at::empty(ws);
  int64_t size_bytes = at::prod_intlist(ws) * sizeof(float);
  memcpy(packedWeight.data_ptr(), packed_buffer.data(), size_bytes);
  return c10::make_intrusive<UnetConv2dOpContext>(
      std::move(packedWeight),
      std::move(bias),
      stride,
      padding,
      dilation,
      groups,
      output_min,
      output_max);
}

TORCH_LIBRARY(metal_unet, m) {
  m.class_<UnetConv2dOpContext>("UnetConv2dOpContext")
      .def_pickle(
          [](const c10::intrusive_ptr<UnetConv2dOpContext>& op_context)
              -> SerializationTypeConv2dPrePack { // __getstate__
            return op_context->pack();
          },
          [](SerializationTypeConv2dPrePack state)
              -> c10::intrusive_ptr<UnetConv2dOpContext> { // __setstate__
            return unpack_unet(
                std::move(std::get<0>(state)),
                std::move(std::get<1>(state)),
                std::move(std::get<2>(state)),
                std::move(std::get<3>(state)),
                std::move(std::get<4>(state)),
                std::move(std::get<5>(state)),
                std::move(std::get<6>(state)),
                std::move(std::get<7>(state)));
          });
}

TORCH_LIBRARY(metal_prepack_unet, m) {
  m.def(
      "conv2d_prepack(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __torch__.torch.classes.metal_unet.UnetConv2dOpContext");
  m.def(
      "conv2d_run(Tensor X, "
      "__torch__.torch.classes.metal_unet.UnetConv2dOpContext W_prepack) -> Tensor Y");
}

c10::intrusive_ptr<UnetConv2dOpContext> conv2d_prepack_unet(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    int64_t groups,
    c10::optional<Scalar> output_min,
    c10::optional<Scalar> output_max) {
  TORCH_CHECK(weight.dim() == 4);
  Tensor serialized_weight = weight;
  if (weight.sizes()[1] >= 12 && weight.sizes()[0] >= 12) {
    std::cout << "conv2d_prepacking: " << weight.sizes() << std::endl;
    auto min = weight.min().item<float>();
    auto max = weight.max().item<float>();
    constexpr size_t kBins = 255;
    double scale = (max - min) / kBins;
    auto quantizer =
        at::make_per_tensor_affine_quantizer(scale, -min, at::kQUInt8);
    serialized_weight = quantizer->quantize(weight);
    auto quantized_data = reinterpret_cast<uint8_t*>(
        serialized_weight.template data_ptr<c10::quint8>());
    auto data = weight.data_ptr<float>();
    for (size_t i = 0; i < serialized_weight.numel(); ++i) {
      int64_t temp = std::nearbyint((data[i] - min) / scale);
      temp = std::max<int64_t>(temp, 0);
      temp = std::min<int64_t>(temp, 255);
      quantized_data[i] = static_cast<uint8_t>(temp);
    }
    // Multiplye by 1000000.f so as to store min value as zero point
    // Since min value is pretty small scale it up. And then down in setstate.
    quantizer = at::make_per_tensor_affine_quantizer(
        scale, static_cast<int64_t>(min * kMinScaleFactor), at::kQUInt8);
    serialized_weight.set_quantizer_(quantizer);
  }
  return c10::make_intrusive<UnetConv2dOpContext>(
      std::move(serialized_weight),
      std::move(bias),
      stride,
      padding,
      dilation,
      groups,
      output_min,
      output_max);
}

TORCH_LIBRARY_IMPL(metal_prepack_unet, CPU, m) {
  m.impl("conv2d_prepack", TORCH_FN(conv2d_prepack_unet));
}

Tensor conv2d_prepack_run_unet(
    const Tensor& input,
    const c10::intrusive_ptr<UnetConv2dOpContext>& op_context) {
#if (C10_IOS || TARGET_OS_MAC)
  return mpscnn::conv2d(input, *op_context);
#else
  TORCH_CHECK(false, "conv2d_prepack_run can only be invoked on iOS and MacOS");
  return input;
#endif
}

TORCH_LIBRARY_IMPL(metal_prepack_unet, Metal, m) {
  m.impl("conv2d_run", conv2d_prepack_run_unet);
}

} // namespace metal
} // namespace native
} // namespace at
