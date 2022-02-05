#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Convolution.h>
#include <ATen/native/vulkan/ops/TransposeConvolution2d.h>
#include <ATen/native/vulkan/ops/McLarenEncoderBlock.h>
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
                std::move(std::get<5>(state)),
                std::move(std::get<6>(state)),
                std::move(std::get<7>(state)));
          });
  m.class_<McLarenEncoderBlockOpContext>("McLarenEncoderBlockOpContext")
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<McLarenEncoderBlockOpContext>& context) {
            return context->unpack();
          },
          // __setstate__
          [](McLarenEncoderBlockOpContext::State state) {
            return mclaren_encoder_block_prepack(
                std::move(std::get<0>(state)),
                std::move(std::get<1>(state)),
                std::move(std::get<2>(state)),
                std::move(std::get<3>(state)),
                std::move(std::get<4>(state)),
                std::move(std::get<5>(state)),
                std::move(std::get<6>(state)),
                std::move(std::get<7>(state)),
                std::move(std::get<8>(state)),
                std::move(std::get<9>(state)),
                std::move(std::get<10>(state)),
                std::move(std::get<11>(state)),
                std::move(std::get<12>(state)),
                std::move(std::get<13>(state)),
                std::move(std::get<14>(state)));
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
                std::move(std::get<6>(state)),
                std::move(std::get<7>(state)),
                std::move(std::get<8>(state)));
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
}

TORCH_LIBRARY(mclaren_prepack, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mclaren_prepack::mclaren_encoder_block_prepack("
      "Tensor W_1, Tensor? B_1, int[2] stride_1, int[2] padding_1, int[2] output_padding_1, int[2] dilation_1, int groups_1, "
      "Tensor W_2, Tensor? B_2, int[2] stride_2, int[2] padding_2, int[2] output_padding_2, int[2] dilation_2, int groups_2, "
      "bool transposed) "
      "-> __torch__.torch.classes.vulkan.McLarenEncoderBlockOpContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mclaren_prepack::mclaren_encoder_block_run(Tensor X_1, Tensor X_2, "
      "__torch__.torch.classes.vulkan.McLarenEncoderBlockOpContext W_prepack) -> Tensor Y"));
}

TORCH_LIBRARY_IMPL(mclaren_prepack, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("mclaren_prepack::mclaren_encoder_block_prepack"), TORCH_FN(mclaren_encoder_block_prepack));
}

TORCH_LIBRARY_IMPL(mclaren_prepack, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("mclaren_prepack::mclaren_encoder_block_run"), TORCH_FN(mclaren_encoder_block_run));
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
}

TORCH_LIBRARY_IMPL(vulkan_prepack, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("vulkan_prepack::conv2d_clamp_prepack"), TORCH_FN(conv2d_clamp_prepack));
  m.impl(TORCH_SELECTIVE_NAME("vulkan_prepack::conv2d_transpose_clamp_prepack"), TORCH_FN(conv2d_transpose_clamp_prepack));
  m.impl(TORCH_SELECTIVE_NAME("vulkan_prepack::linear_prepack"), TORCH_FN(linear_prepack));
}

TORCH_LIBRARY_IMPL(vulkan_prepack, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("vulkan_prepack::conv2d_clamp_run"), TORCH_FN(conv2d_clamp_run));
  m.impl(TORCH_SELECTIVE_NAME("vulkan_prepack::conv2d_transpose_clamp_run"), TORCH_FN(conv2d_transpose_clamp_run));
  m.impl(TORCH_SELECTIVE_NAME("vulkan_prepack::linear_run"), TORCH_FN(linear_run));
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
