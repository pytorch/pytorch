#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Batchnorm.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Convolution.h>
#include <ATen/native/vulkan/ops/Gru.h>
#include <ATen/native/vulkan/ops/Lstm.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <torch/custom_class.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

TORCH_LIBRARY(vulkan, m) {
  m.class_<BatchNormPackedContext>("BatchNormPackedContext")
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<BatchNormPackedContext>& context) {
            // context is packed
            return context->unpack();
          },
          // __setstate__
          [](c10::impl::GenericList state) {
            // state is unpacked
            return c10::make_intrusive<BatchNormPackedContext>(
                BatchNormPackedContext::pack(state));
          });
  m.class_<LinearPackedContext>("LinearPackedContext")
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<LinearPackedContext>& context) {
            // context is packed
            return context->unpack();
          },
          // __setstate__
          [](c10::impl::GenericList state) {
            // state is unpacked
            return c10::make_intrusive<LinearPackedContext>(
                LinearPackedContext::pack(state));
          });
  m.class_<GruPackedContext>("GruPackedContext")
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<GruPackedContext>& context) {
            // context is packed
            return context->unpack();
          },
          // __setstate__
          [](c10::impl::GenericList state) {
            // state is unpacked
            return c10::make_intrusive<GruPackedContext>(
                GruPackedContext::pack(state));
          });
  m.class_<LstmPackedContext>("LstmPackedContext")
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<LstmPackedContext>& context) {
            // context is packed
            return context->unpack();
          },
          // __setstate__
          [](c10::impl::GenericList state) {
            // state is unpacked
            return c10::make_intrusive<LstmPackedContext>(
                LstmPackedContext::pack(state));
          });
  m.class_<Conv2dPackedContext>("Conv2dPackedContext")
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<Conv2dPackedContext>& context) {
            // context is packed
            return context->unpack();
          },
          // __setstate__
          [](c10::impl::GenericList state) {
            // state is unpacked
            return c10::make_intrusive<Conv2dPackedContext>(
                Conv2dPackedContext::pack(state));
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
}

TORCH_LIBRARY(vulkan_prepack, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_conv2d_context(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __torch__.torch.classes.vulkan.Conv2dPackedContext"));
  m.def(TORCH_SELECTIVE_SCHEMA( // Backwards compatibility
      "vulkan_prepack::conv2d_clamp_prepack(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __torch__.torch.classes.vulkan.Conv2dOpContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_conv2d_context(Tensor X, "
      "__torch__.torch.classes.vulkan.Conv2dPackedContext W_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA( // Backwards compatibility
      "vulkan_prepack::conv2d_clamp_run(Tensor X, "
      "__torch__.torch.classes.vulkan.Conv2dOpContext W_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_tconv2d_context(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] output_padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __torch__.torch.classes.vulkan.Conv2dPackedContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_tconv2d_context(Tensor X, "
      "__torch__.torch.classes.vulkan.Conv2dPackedContext W_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_linear_context(Tensor W, Tensor? B) "
      "-> __torch__.torch.classes.vulkan.LinearPackedContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_linear_context(Tensor X, "
      "__torch__.torch.classes.vulkan.LinearPackedContext BW_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_gru_context(Tensor[] params_cpu, "
      "bool has_biases, "
      "int num_layers, "
      "float dropout, "
      "bool train, "
      "bool bidirectional, "
      "bool batch_first) "
      "-> __torch__.torch.classes.vulkan.GruPackedContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_gru_context(Tensor input_vk, "
      "Tensor hx_vk, "
      "__torch__.torch.classes.vulkan.GruPackedContext G_prepack) -> (Tensor next_input, Tensor hidden_layer)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_lstm_context(Tensor[] params_cpu, "
      "bool has_biases, "
      "int num_layers, "
      "float dropout, "
      "bool train, "
      "bool bidirectional, "
      "bool batch_first) "
      "-> __torch__.torch.classes.vulkan.LstmPackedContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_lstm_context(Tensor input_vk, "
      "Tensor hx_vk, "
      "Tensor cx_vk, "
      "__torch__.torch.classes.vulkan.LstmPackedContext L_prepack) -> (Tensor next_input, Tensor hidden_state, Tensor cell_state)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_batchnorm_context("
      "Tensor? weight_opt, "
      "Tensor? bias_opt, "
      "Tensor? running_mean_opt, "
      "Tensor? running_var_opt, "
      "bool training, "
      "float momentum, "
      "float eps, "
      "bool cudnn_enable) "
      "-> __torch__.torch.classes.vulkan.BatchNormPackedContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_batchnorm_context("
      "Tensor input_vk, "
      "__torch__.torch.classes.vulkan.BatchNormPackedContext context) "
      "-> Tensor out"));
}

TORCH_LIBRARY_IMPL(vulkan_prepack, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_conv2d_context"),
      TORCH_FN(create_conv2d_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::conv2d_clamp_prepack"),
      TORCH_FN(conv2d_clamp_prepack)); // Backwards compatibility
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_tconv2d_context"),
      TORCH_FN(create_tconv2d_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_linear_context"),
      TORCH_FN(create_linear_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_gru_context"),
      TORCH_FN(create_gru_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_lstm_context"),
      TORCH_FN(create_lstm_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_batchnorm_context"),
      TORCH_FN(create_batchnorm_context));
}

TORCH_LIBRARY_IMPL(vulkan_prepack, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_conv2d_context"),
      TORCH_FN(run_conv2d_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::conv2d_clamp_run"),
      TORCH_FN(conv2d_clamp_run)); // Backwards compatibility
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_tconv2d_context"),
      TORCH_FN(run_tconv2d_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_linear_context"),
      TORCH_FN(run_linear_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_gru_context"),
      TORCH_FN(run_gru_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_lstm_context"),
      TORCH_FN(run_lstm_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_batchnorm_context"),
      TORCH_FN(run_batchnorm_context));
}

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
