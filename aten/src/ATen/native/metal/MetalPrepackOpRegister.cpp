#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/metal/MetalPrepackOpContext.h>
#include <c10/util/accumulate.h>

namespace at {
namespace native {
namespace metal {

c10::intrusive_ptr<Conv2dOpContext> unpack(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    int64_t groups,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max) {
  auto packedWeight = weight.contiguous(MemoryFormat::ChannelsLast);
  return c10::make_intrusive<Conv2dOpContext>(
      std::move(packedWeight),
      std::move(bias),
      stride,
      padding,
      dilation,
      groups,
      output_min,
      output_max);
}

c10::intrusive_ptr<LinearOpContext> unpack(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max) {
  TORCH_CHECK(weight.dim() == 2);
  // Don't need to do `weight.t()`
  auto packedWeight = weight.view({weight.size(0), weight.size(1), 1, 1})
                          .contiguous(MemoryFormat::ChannelsLast);
  return c10::make_intrusive<LinearOpContext>(
      std::move(packedWeight), std::move(bias), output_min, output_max);
}

TORCH_LIBRARY(metal, m) {
  m.class_<Conv2dOpContext>("Conv2dOpContext")
      .def_pickle(
          [](const c10::intrusive_ptr<Conv2dOpContext>& op_context)
              -> SerializationTypeConv2dPrePack { // __getstate__
            return op_context->pack();
          },
          [](SerializationTypeConv2dPrePack state)
              -> c10::intrusive_ptr<Conv2dOpContext> { // __setstate__
            return unpack(
                std::move(std::get<0>(state)),
                std::move(std::get<1>(state)),
                std::move(std::get<2>(state)),
                std::move(std::get<3>(state)),
                std::move(std::get<4>(state)),
                std::move(std::get<5>(state)),
                std::move(std::get<6>(state)),
                std::move(std::get<7>(state)));
          });
  m.class_<LinearOpContext>("LinearOpContext")
      .def_pickle(
          [](const c10::intrusive_ptr<LinearOpContext>& op_context)
              -> SerializationTypeLinearPrePack { // __getstate__
            return op_context->pack();
          },
          [](SerializationTypeLinearPrePack state)
              -> c10::intrusive_ptr<LinearOpContext> { // __setstate__
            return unpack(
                std::move(std::get<0>(state)),
                std::move(std::get<1>(state)),
                std::get<2>(state),
                std::get<3>(state));
          });
  m.def("copy_to_host(Tensor X) -> Tensor Y");
}

TORCH_LIBRARY(metal_prepack, m) {
  m.def(
      TORCH_SELECTIVE_SCHEMA("metal_prepack::conv2d_prepack(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __torch__.torch.classes.metal.Conv2dOpContext"));
  m.def(
      TORCH_SELECTIVE_SCHEMA("metal_prepack::conv2d_run(Tensor X, "
      "__torch__.torch.classes.metal.Conv2dOpContext W_prepack) -> Tensor Y"));

  m.def(
      TORCH_SELECTIVE_SCHEMA("metal_prepack::linear_prepack(Tensor W, Tensor? B, Scalar? output_min=None, Scalar? output_max=None) -> __torch__.torch.classes.metal.LinearOpContext"));

  m.def(
      TORCH_SELECTIVE_SCHEMA("metal_prepack::linear_run(Tensor X, __torch__.torch.classes.metal.LinearOpContext W_prepack) -> Tensor Y"));
}

c10::intrusive_ptr<Conv2dOpContext> conv2d_prepack(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    int64_t groups,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max) {
  TORCH_CHECK(weight.dim() == 4);
  return c10::make_intrusive<Conv2dOpContext>(
      std::move(weight),
      std::move(bias),
      stride,
      padding,
      dilation,
      groups,
      output_min,
      output_max);
}

c10::intrusive_ptr<LinearOpContext> linear_prepack(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max) {
  return c10::make_intrusive<LinearOpContext>(
      std::move(weight), std::move(bias), output_min, output_max);
}

TORCH_LIBRARY_IMPL(metal_prepack, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("metal_prepack::conv2d_prepack"), TORCH_FN(conv2d_prepack));
  m.impl(TORCH_SELECTIVE_NAME("metal_prepack::linear_prepack"), TORCH_FN(linear_prepack));
}

} // namespace metal
} // namespace native
} // namespace at
