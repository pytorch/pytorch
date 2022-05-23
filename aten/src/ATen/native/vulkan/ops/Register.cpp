#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Convolution.h>
#include <ATen/native/vulkan/ops/Gru.h>
#include <ATen/native/vulkan/ops/TransposeConvolution2d.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <torch/custom_class.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

TORCH_LIBRARY(vulkan, m) {
  m.class_<Conv2dOpContext>("Conv2dOpContext")
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<Conv2dOpContext>& context) {
            return context->unpack();
          },
          // __setstate__
          [](Conv2dOpContext::State state) {
            return conv2d_clamp_prepack(
                std::move(std::get<0>(state)),
                std::move(std::get<1>(state)),
                std::move(std::get<2>(state)),
                std::move(std::get<3>(state)),
                std::move(std::get<4>(state)),
                std::get<5>(state),
                std::get<6>(state),
                std::get<7>(state));
          });
  m.class_<TransposeConv2dOpContext>("TransposeConv2dOpContext")
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<TransposeConv2dOpContext>& context) {
            return context->unpack();
          },
          // __setstate__
          [](TransposeConv2dOpContext::State state) {
            return conv2d_transpose_clamp_prepack(
                std::move(std::get<0>(state)),
                std::move(std::get<1>(state)),
                std::move(std::get<2>(state)),
                std::move(std::get<3>(state)),
                std::move(std::get<4>(state)),
                std::move(std::get<5>(state)),
                std::get<6>(state),
                std::get<7>(state),
                std::get<8>(state));
          });
  m.class_<LinearOpContext>("LinearOpContext")
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<LinearOpContext>& context) {
            return context->unpack();
          },
          // __setstate__
          [](LinearOpContext::State state) {
            return linear_prepack(
                std::move(std::get<0>(state)), std::move(std::get<1>(state)));
          });
  m.class_<GruOpContext>("GruOpContext")
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<GruOpContext>& context) {
            return context->unpack();
          },
          // __setstate__
          [](GruOpContext::State state) {
            return gru_prepack(
                std::move(std::get<0>(state)),
                std::get<1>(state),
                std::get<2>(state),
                std::get<3>(state),
                std::get<4>(state),
                std::get<5>(state),
                std::get<6>(state));
          });
}

TORCH_LIBRARY(vulkan_prepack, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::conv2d_clamp_prepack(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __torch__.torch.classes.vulkan.Conv2dOpContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::conv2d_clamp_run(Tensor X, "
      "__torch__.torch.classes.vulkan.Conv2dOpContext W_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::conv2d_transpose_clamp_prepack(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] output_padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __torch__.torch.classes.vulkan.TransposeConv2dOpContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::conv2d_transpose_clamp_run(Tensor X, "
      "__torch__.torch.classes.vulkan.TransposeConv2dOpContext W_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::linear_prepack(Tensor W, Tensor? B) "
      "-> __torch__.torch.classes.vulkan.LinearOpContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::linear_run(Tensor X, "
      "__torch__.torch.classes.vulkan.LinearOpContext BW_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::gru_prepack(Tensor[] params_cpu, "
      "bool has_biases, "
      "int num_layers, "
      "float dropout, "
      "bool train, "
      "bool bidirectional, "
      "bool batch_first) "
      "-> __torch__.torch.classes.vulkan.GruOpContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::gru_run(Tensor input_vk, "
      "Tensor hx_vk, "
      "__torch__.torch.classes.vulkan.GruOpContext G_prepack) -> (Tensor next_input, Tensor hidden_layer)"));
}

TORCH_LIBRARY_IMPL(vulkan_prepack, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("vulkan_prepack::conv2d_clamp_prepack"), TORCH_FN(conv2d_clamp_prepack));
  m.impl(TORCH_SELECTIVE_NAME("vulkan_prepack::conv2d_transpose_clamp_prepack"), TORCH_FN(conv2d_transpose_clamp_prepack));
  m.impl(TORCH_SELECTIVE_NAME("vulkan_prepack::linear_prepack"), TORCH_FN(linear_prepack));
  m.impl(TORCH_SELECTIVE_NAME("vulkan_prepack::gru_prepack"), TORCH_FN(gru_prepack));
}

TORCH_LIBRARY_IMPL(vulkan_prepack, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("vulkan_prepack::conv2d_clamp_run"), TORCH_FN(conv2d_clamp_run));
  m.impl(TORCH_SELECTIVE_NAME("vulkan_prepack::conv2d_transpose_clamp_run"), TORCH_FN(conv2d_transpose_clamp_run));
  m.impl(TORCH_SELECTIVE_NAME("vulkan_prepack::linear_run"), TORCH_FN(linear_run));
  m.impl(TORCH_SELECTIVE_NAME("vulkan_prepack::gru_run"), TORCH_FN(gru_run));
}

Tensor convolution(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const bool transposed,
    const IntArrayRef output_padding,
    const int64_t groups) {
  if (transposed) {
    return TransposeConv2dOpContext::create(
        weight,
        bias,
        stride,
        padding,
        output_padding,
        dilation,
        groups
    ).run(input);
  }
  return Conv2dOpContext::create(
      weight,
      bias,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups
  ).run(input);
}

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("convolution_overrideable", convolution);
}

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
