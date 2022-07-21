#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Convolution.h>
#include <ATen/native/vulkan/ops/Gru.h>
#include <ATen/native/vulkan/ops/Lstm.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/QuantizedConvolution.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <ATen/native/vulkan/ops/TransposeConvolution2d.h>
#include <ATen/native/vulkan/ops/VulkanOpContext.h>
#include <torch/custom_class.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

TORCH_LIBRARY(vulkan, m) {
  m.class_<VulkanOpContext>("VulkanOpContext")
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<VulkanOpContext>& context) {
            return context->get_state();
          },
          // __setstate__
          [](VulkanOpContext::State state) {
            return c10::make_intrusive<VulkanOpContext>(VulkanOpContext::create(
                std::get<0>(state), std::get<1>(state)));
          });
  // To maintain backwards compatibility.
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
  // To maintain backwards compatibility.
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
  // To maintain backwards compatibility.
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
  // To maintain backwards compatibility.
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
      "vulkan_prepack::create_conv2d_clamp_context(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __torch__.torch.classes.vulkan.VulkanOpContext"));
  m.def(TORCH_SELECTIVE_SCHEMA( // Backwards compatibility
      "vulkan_prepack::conv2d_clamp_prepack(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __torch__.torch.classes.vulkan.Conv2dOpContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_conv2d_clamp_context(Tensor X, "
      "__torch__.torch.classes.vulkan.VulkanOpContext W_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA( // Backwards compatibility
      "vulkan_prepack::conv2d_clamp_run(Tensor X, "
      "__torch__.torch.classes.vulkan.Conv2dOpContext W_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_conv2d_transpose_clamp_context(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] output_padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __torch__.torch.classes.vulkan.VulkanOpContext"));
  m.def(TORCH_SELECTIVE_SCHEMA( // Backwards compatibility
      "vulkan_prepack::conv2d_transpose_clamp_prepack(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] output_padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __torch__.torch.classes.vulkan.TransposeConv2dOpContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_conv2d_transpose_clamp_context(Tensor X, "
      "__torch__.torch.classes.vulkan.VulkanOpContext W_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA( // Backwards compatibility
      "vulkan_prepack::conv2d_transpose_clamp_run(Tensor X, "
      "__torch__.torch.classes.vulkan.TransposeConv2dOpContext W_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_linear_context(Tensor W, Tensor? B) "
      "-> __torch__.torch.classes.vulkan.VulkanOpContext"));
  m.def(TORCH_SELECTIVE_SCHEMA( // Backwards compatibility
      "vulkan_prepack::linear_prepack(Tensor W, Tensor? B) "
      "-> __torch__.torch.classes.vulkan.LinearOpContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_linear_context(Tensor X, "
      "__torch__.torch.classes.vulkan.VulkanOpContext BW_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA( // Backwards compatibility
      "vulkan_prepack::linear_run(Tensor X, "
      "__torch__.torch.classes.vulkan.LinearOpContext BW_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_gru_context(Tensor[] params_cpu, "
      "bool has_biases, "
      "int num_layers, "
      "float dropout, "
      "bool train, "
      "bool bidirectional, "
      "bool batch_first) "
      "-> __torch__.torch.classes.vulkan.VulkanOpContext"));
  m.def(TORCH_SELECTIVE_SCHEMA( // Backwards compatibility
      "vulkan_prepack::gru_prepack(Tensor[] params_cpu, "
      "bool has_biases, "
      "int num_layers, "
      "float dropout, "
      "bool train, "
      "bool bidirectional, "
      "bool batch_first) "
      "-> __torch__.torch.classes.vulkan.GruOpContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_gru_context(Tensor input_vk, "
      "Tensor hx_vk, "
      "__torch__.torch.classes.vulkan.VulkanOpContext G_prepack) -> (Tensor next_input, Tensor hidden_layer)"));
  m.def(TORCH_SELECTIVE_SCHEMA( // Backwards compatibility
      "vulkan_prepack::gru_run(Tensor input_vk, "
      "Tensor hx_vk, "
      "__torch__.torch.classes.vulkan.GruOpContext G_prepack) -> (Tensor next_input, Tensor hidden_layer)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_lstm_context(Tensor[] params_cpu, "
      "bool has_biases, "
      "int num_layers, "
      "float dropout, "
      "bool train, "
      "bool bidirectional, "
      "bool batch_first) "
      "-> __torch__.torch.classes.vulkan.VulkanOpContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_lstm_context(Tensor input_vk, "
      "Tensor hx_vk, "
      "Tensor cx_vk, "
      "__torch__.torch.classes.vulkan.VulkanOpContext L_prepack) -> (Tensor next_input, Tensor hidden_state, Tensor cell_state)"));
}

TORCH_LIBRARY_IMPL(vulkan_prepack, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_conv2d_clamp_context"),
      TORCH_FN(create_conv2d_clamp_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::conv2d_clamp_prepack"),
      TORCH_FN(conv2d_clamp_prepack)); // Backwards compatibility
  m.impl(
      TORCH_SELECTIVE_NAME(
          "vulkan_prepack::create_conv2d_transpose_clamp_context"),
      TORCH_FN(create_conv2d_transpose_clamp_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::conv2d_transpose_clamp_prepack"),
      TORCH_FN(conv2d_transpose_clamp_prepack)); // Backwards compatibility
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_linear_context"),
      TORCH_FN(create_linear_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::linear_prepack"),
      TORCH_FN(linear_prepack)); // Backwards compatibility
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_gru_context"),
      TORCH_FN(create_gru_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::gru_prepack"),
      TORCH_FN(gru_prepack)); // Backwards compatibility
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_lstm_context"),
      TORCH_FN(create_lstm_context));
}

TORCH_LIBRARY_IMPL(vulkan_prepack, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_conv2d_clamp_context"),
      TORCH_FN(run_conv2d_clamp_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::conv2d_clamp_run"),
      TORCH_FN(conv2d_clamp_run)); // Backwards compatibility
  m.impl(
      TORCH_SELECTIVE_NAME(
          "vulkan_prepack::run_conv2d_transpose_clamp_context"),
      TORCH_FN(run_conv2d_transpose_clamp_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::conv2d_transpose_clamp_run"),
      TORCH_FN(conv2d_transpose_clamp_run)); // Backwards compatibility
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_linear_context"),
      TORCH_FN(run_linear_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::linear_run"),
      TORCH_FN(linear_run)); // Backwards compatibility
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_gru_context"),
      TORCH_FN(run_gru_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::gru_run"),
      TORCH_FN(gru_run)); // Backwards compatibility
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_lstm_context"),
      TORCH_FN(run_lstm_context));
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
    VulkanOpContext vulkan_context = conv2d_transpose_context_create(
        weight, bias, stride, padding, output_padding, dilation, groups);
    return conv2d_transpose_context_run(
        input, vulkan_context.get_packed(), vulkan_context.get_unpacked());
  }
  VulkanOpContext vulkan_context = conv2d_context_create(
      weight,
      bias,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups);
  return conv2d_context_run(
      input, vulkan_context.get_packed(), vulkan_context.get_unpacked());
}

Tensor quantized_convolution(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const bool transposed,
    const IntArrayRef output_padding,
    const int64_t groups,
    const double out_scale,
    const int64_t out_zero_point) {
  if (transposed) {
    VulkanOpContext vulkan_context = conv2d_transpose_context_create(
        weight, bias, stride, padding, output_padding, dilation, groups);
    return conv2d_transpose_context_run(
        input, vulkan_context.get_packed(), vulkan_context.get_unpacked());
  }
  VulkanOpContext vulkan_context = conv2d_context_create_q(
      weight,
      bias,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      c10::nullopt,
      c10::nullopt);
  return conv2d_context_run_q(
      input,
      vulkan_context.get_packed(),
      vulkan_context.get_unpacked(),
      out_scale,
      out_zero_point);
}
} // namespace

static std::tuple<Tensor, bool> batchify(
    const Tensor& input,
    const int64_t num_spatial_dims,
    const std::string& func_name) {
  const auto dim_count_no_batch = num_spatial_dims + 1;
  const auto dim_count_batch = dim_count_no_batch + 1;
  const auto is_batched = (input.dim() == dim_count_batch);
  TORCH_CHECK(
      input.dim() == dim_count_no_batch || is_batched,
      "Expected ",
      dim_count_no_batch,
      "D (unbatched) or ",
      dim_count_batch,
      "D (batched) input to ",
      func_name,
      ", but got input of size: ",
      input.sizes());
  return std::make_tuple(is_batched ? input : input.unsqueeze(0), is_batched);
}

Tensor conv2d(
    const Tensor& input_,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    double out_scale,
    int64_t out_zero_point) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  Tensor input;
  bool is_batched;
  std::tie(input, is_batched) =
      batchify(input_, /*num_spatial_dims=*/2, "conv2d");
  Tensor output;
  output = quantized_convolution(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      false,
      {{0, 0}},
      groups,
      out_scale,
      out_zero_point);
  return is_batched ? output : output.squeeze(0);
}

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("convolution_overrideable", convolution);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
