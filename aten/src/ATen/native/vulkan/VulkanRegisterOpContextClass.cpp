#include <ATen/core/op_registration/op_registration.h>
#include <torch/custom_class.h>

#include <ATen/native/vulkan/VulkanConvolution.h>
#include <ATen/native/vulkan/VulkanOpContext.h>

namespace at {
namespace native {
namespace vulkan {

#ifndef USE_VULKAN_API

using detail::convolution2d::createConv2dClampPrePackOpContext;

TORCH_LIBRARY(vulkan, m) {
  m.class_<Conv2dOpContext>("Conv2dOpContext")
      .def_pickle(
          [](const c10::intrusive_ptr<Conv2dOpContext>& op_context)
              -> SerializationTypeConv2dPrePack { // __getstate__
            return op_context->unpack();
          },
          [](SerializationTypeConv2dPrePack state)
              -> c10::intrusive_ptr<Conv2dOpContext> { // __setstate__
            return createConv2dClampPrePackOpContext(
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

TORCH_LIBRARY(vulkan_prepack, m) {
  m.def(
      "conv2d_clamp_prepack(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __torch__.torch.classes.vulkan.Conv2dOpContext");
  m.def(
      "conv2d_clamp_run(Tensor X, "
      "__torch__.torch.classes.vulkan.Conv2dOpContext W_prepack) -> Tensor Y");
}

TORCH_LIBRARY_IMPL(vulkan_prepack, CPU, m) {
  m.impl("conv2d_clamp_prepack", TORCH_FN(createConv2dClampPrePackOpContext));
}

TORCH_LIBRARY_IMPL(vulkan_prepack, Vulkan, m) {
  m.impl("conv2d_clamp_run", detail::convolution2d::conv2d_clamp_run);
}

#endif /* USE_VULKAN_API */

} // namespace vulkan
} // namespace native
} // namespace at
