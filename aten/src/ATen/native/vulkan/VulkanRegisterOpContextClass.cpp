#include <ATen/core/op_registration/op_registration.h>
#include <torch/custom_class.h>

#include <ATen/native/vulkan/VulkanConvolution.h>
#include <ATen/native/vulkan/VulkanOpContext.h>

namespace at {
namespace native {
namespace vulkan {

using details::convolution2d::createConv2dClampPrePackOpContext;

namespace {
torch::jit::class_<Conv2dOpContext> register_packed_conv2d_op_context_class() {
  static auto register_conv2d_op_context_class =
      torch::jit::class_<Conv2dOpContext>("vulkan", "Conv2dOpContext")
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
  return register_conv2d_op_context_class;
}

static auto conv2d_op_context_class = register_packed_conv2d_op_context_class();

// Op registeration
static auto registry =
    torch::RegisterOperators()
        .op("vulkan::conv2d_clamp_prepack(Tensor W, Tensor? B, int[2] stride, "
            "int[2] padding, int[2] dilation, int groups, "
            "Scalar? output_min=None, Scalar? output_max=None) "
            "-> __torch__.torch.classes.vulkan.Conv2dOpContext",
            torch::RegisterOperators::options()
                .aliasAnalysis(at::AliasAnalysisKind::PURE_FUNCTION)
                .kernel<
                    decltype(createConv2dClampPrePackOpContext),
                    createConv2dClampPrePackOpContext>(
                    DispatchKey::CPUTensorId))
        .op("vulkan::conv2d_clamp_run(Tensor X, "
            "__torch__.torch.classes.vulkan.Conv2dOpContext W_prepack) -> Tensor Y",
            torch::RegisterOperators::options()
                .aliasAnalysis(at::AliasAnalysisKind::PURE_FUNCTION)
                .kernel<details::convolution2d::Conv2dClampRun>(
                    DispatchKey::VulkanTensorId));
} // namespace

} // namespace vulkan
} // namespace native
} // namespace at
