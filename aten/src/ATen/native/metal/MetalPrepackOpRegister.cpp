#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/metal/MetalPrepackOpContext.h>
#include <c10/util/accumulate.h>


#if (C10_IOS || TARGET_OS_MAC)
#import <ATen/native/metal/ops/MetalConvolution.h>
#endif

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
  const Tensor weightContig = weight.contiguous();
  const auto ws = weightContig.sizes();
  auto packed_buffer = permuteWeights(weightContig.data_ptr<float>(), ws.vec());
  auto packedWeight = at::empty(ws);
  int64_t size_bytes = c10::multiply_integers(ws) * sizeof(float);
  memcpy(packedWeight.data_ptr(), packed_buffer.data(), size_bytes);
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
  m.def("copy_to_host(Tensor X) -> Tensor Y");
}

TORCH_LIBRARY(metal_prepack, m) {
  m.def(
      "conv2d_prepack(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __torch__.torch.classes.metal.Conv2dOpContext");
  m.def(
      "conv2d_run(Tensor X, "
      "__torch__.torch.classes.metal.Conv2dOpContext W_prepack) -> Tensor Y");
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

Tensor conv2d_prepack_run(
    const Tensor& input,
    const c10::intrusive_ptr<Conv2dOpContext>& op_context) {
#if (C10_IOS || TARGET_OS_MAC)
  return prepack::conv2d(input, *op_context);
#else
  TORCH_CHECK(false, "conv2d_prepack_run can only be invoked on iOS and MacOS");
  return input;
#endif
}

TORCH_LIBRARY_IMPL(metal_prepack, CPU, m) {
  m.impl("conv2d_prepack", TORCH_FN(conv2d_prepack));
}

TORCH_LIBRARY_IMPL(metal_prepack, Metal, m) {
  m.impl("conv2d_run", conv2d_prepack_run);
}

} // namespace metal
} // namespace native
} // namespace at
